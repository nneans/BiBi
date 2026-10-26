"""Resumable batch analysis of current normal reviews with the saved classifier.

Only (text_hash, model_version) pairs without a stored result are computed, so re-running
skips finished work, identical texts are computed once, and unchanged reviews are reused.
A failing batch falls back to per-text processing; failures are recorded and retried only on request.

    python -m bibi_reviews.service.batch --product-id 296889036 --limit 2000
    python -m bibi_reviews.service.batch --all            # whole DB, requires a validated active model
"""
import argparse
import json
import time
import uuid

from sqlalchemy import select,and_,update

from ..backbone import ReviewStore,now
from ..review_pipeline.common import ROOT,load_config,load_taxonomy,resolve
from .storage import AnalysisStore
from ..opinion_graph.store import OpinionGraphStore


def pending_texts(store,astore,version,upload_id=None,product_id=None,retry_failed=False,max_attempts=3,limit=None):
    r=store.review
    stmt=select(r.c.text_hash,r.c.review_text).where(r.c.display_status=='정상',r.c.text_hash.is_not(None))
    if upload_id:stmt=stmt.join(store.membership,store.membership.c.review_id==r.c.review_id).where(store.membership.c.upload_id==upload_id)
    if product_id:stmt=stmt.where(r.c.product_id==product_id)
    stmt=stmt.outerjoin(astore.status,and_(astore.status.c.text_hash==r.c.text_hash,astore.status.c.model_version==version)).where(astore.status.c.text_hash.is_(None))
    f=astore.failures
    stmt=stmt.outerjoin(f,and_(f.c.text_hash==r.c.text_hash,f.c.model_version==version))
    stmt=stmt.where(f.c.text_hash.is_(None) if not retry_failed else ((f.c.text_hash.is_(None))|(f.c.attempts<max_attempts)))
    stmt=stmt.group_by(r.c.text_hash,r.c.review_text).order_by(r.c.text_hash)
    if limit:stmt=stmt.limit(limit)
    with store.engine.connect() as c:return [tuple(x) for x in c.execute(stmt)]


def analyze(store,analyzer,upload_id=None,product_id=None,limit=None,batch_size=256,retry_failed=False,log=None,graph=None):
    astore=AnalysisStore(store);run_id=uuid.uuid4().hex;version=analyzer.version
    graph=graph or OpinionGraphStore(store)
    graph.register_model(version,getattr(analyzer,'manifest',{}).get('model_kind'),getattr(analyzer,'artifact_dir',None),
                         getattr(analyzer,'validated',False),getattr(analyzer,'test_only',False))
    details={'upload_id':upload_id,'product_id':product_id,'model_version':version,'computed_texts':0,'failed_texts':0,
             'review_required_texts':0,'batches':0,'production_validated':getattr(analyzer,'validated',False)}
    with store.engine.begin() as c:c.execute(store.runs.insert().values(run_id=run_id,model_version=version,status='running',
                                                                      details_json=json.dumps(details),created_at=now()))
    try:
        import torch
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    except Exception:torch=None
    import psutil,os
    proc=psutil.Process(os.getpid());rss_peak=proc.memory_info().rss;started=time.perf_counter()
    texts=pending_texts(store,astore,version,upload_id,product_id,retry_failed,limit=limit)
    details['pending_texts_at_start']=len(texts)
    try:
        for a in range(0,len(texts),batch_size):
            batch=texts[a:a+batch_size]
            try:
                results=analyzer.analyze_batch([t for _,t in batch]);items=list(zip([h for h,_ in batch],results))
            except Exception:
                items=[]
                for h,t in batch:  # isolate the bad text; others continue
                    try:items.append((h,analyzer.analyze_batch([t])[0]))
                    except Exception as exc:astore.record_failure(h,version,f'{type(exc).__name__}: {exc}');details['failed_texts']+=1
            with store.engine.begin() as c:  # labels and opinion instances commit together
                astore.save_batch(c,version,items);details['opinion_predictions']=details.get('opinion_predictions',0)+graph.save_predictions(c,version,items)
            details['computed_texts']+=len(items);details['batches']+=1
            details['review_required_texts']+=sum(r['status']=='review_required' for _,r in items)
            rss_peak=max(rss_peak,proc.memory_info().rss)
            if log:log(f"{details['computed_texts']}/{len(texts)} texts")
            with store.engine.begin() as c:c.execute(update(store.runs).where(store.runs.c.run_id==run_id).values(details_json=json.dumps(details)))
        elapsed=time.perf_counter()-started
        details.update(elapsed_seconds=round(elapsed,2),texts_per_second=round(details['computed_texts']/max(elapsed,1e-9),1),
                       peak_rss_mib=round(rss_peak/2**20,1),
                       peak_gpu_mib=round(torch.cuda.max_memory_allocated()/2**20,1) if torch and torch.cuda.is_available() else None)
        with store.engine.begin() as c:c.execute(update(store.runs).where(store.runs.c.run_id==run_id).values(status='completed',details_json=json.dumps(details)))
        return {'run_id':run_id,**details}
    except Exception as exc:
        details['error']=str(exc)[:500]
        with store.engine.begin() as c:c.execute(update(store.runs).where(store.runs.c.run_id==run_id).values(status='failed',details_json=json.dumps(details)))
        raise


def active_analyzer(config,registry_path=None,require_validated=True,device=None):
    from ..classifier import registry
    from ..classifier.inference import ClassifierAnalyzer
    reg=registry.read(registry_path or registry.DEFAULT_REGISTRY)
    if not reg.get('active_version'):raise LookupError('No active model in the registry; nothing is analyzed')
    if require_validated and not reg.get('validated'):raise PermissionError('Active model is not validated on human gold')
    a=ClassifierAnalyzer(resolve(reg['artifact_dir']),config,validated=reg.get('validated',False),device=device,taxonomy=load_taxonomy(config))
    if a.version!=reg['active_version']:raise ValueError('Registry version does not match artifact manifest')
    return a


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config',default=str(ROOT/'configs/review_pipeline/pipeline.json'));p.add_argument('--db')
    p.add_argument('--registry');p.add_argument('--product-id');p.add_argument('--upload-id');p.add_argument('--limit',type=int)
    p.add_argument('--all',action='store_true');p.add_argument('--batch-size',type=int,default=256);p.add_argument('--retry-failed',action='store_true')
    a=p.parse_args()
    if not (a.all or a.product_id or a.upload_id or a.limit):p.error('Choose --product-id/--upload-id/--limit, or --all explicitly')
    config=load_config(a.config)
    try:analyzer=active_analyzer(config,a.registry)
    except (LookupError,PermissionError) as exc:p.exit(2,f'NOT RUN: {exc}\n')
    store=ReviewStore(a.db or config['database'])
    print(json.dumps(analyze(store,analyzer,a.upload_id,a.product_id,a.limit,a.batch_size,a.retry_failed,log=print),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
