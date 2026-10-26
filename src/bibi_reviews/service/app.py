"""Dashboard service = existing authenticated upload API + offline analysis worker + /api/v2 statistics.

    BIBI_PASSWORD=... uvicorn --factory bibi_reviews.service.app:app_from_env

Upload flow: the existing ReviewStore import (validation, new/updated/unchanged, blind scrubbing)
runs unchanged; this worker then analyzes the upload's reviews with the registry's active model,
reusing results by (text_hash, model_version). It never calls an external API and never retrains.
Without an active model, uploads complete and analysis stays pending (reported, not faked).
"""
import json
import os
import threading
import time
from datetime import date
from pathlib import Path

from fastapi import HTTPException,Query
from pydantic import BaseModel,Field
from sqlalchemy import select

from ..api import create_app
from ..backbone import ReviewStore,now
from ..review_pipeline.common import ROOT,load_config,load_taxonomy,resolve
from .. import paths
from ..classifier import registry as registry_module
from .storage import AnalysisStore
from .analytics import Analytics
from . import batch
from ..opinion_graph.store import OpinionGraphStore
from ..opinion_graph.queries import OpinionAnalytics


class Opinion(BaseModel):
    target_scope:str=Field(pattern='^(current_product|other_product|general_experience|unclear)$')
    statement_type:str=Field(pattern='^(evaluation|fact|deferred|uninterpretable)$')
    aspect:str
    sentiment:str=Field(pattern='^(positive|negative|mixed|neutral|ambiguous|not_applicable)$')
    reason_code:str|None=None
    reason_text:str=''
    evidence_text:str=''


class Correction(BaseModel):
    annotator:str=Field(min_length=1)
    complete:bool=Field(description='true only if every opinion in the review was checked')
    opinions:list[Opinion]


class AnalysisWorker:
    def __init__(self,store,astore,config,registry_path,analyzer_factory=None,poll_seconds=1.0,graph=None):
        self.store=store;self.astore=astore;self.config=config;self.registry_path=registry_path
        self.graph=graph or OpinionGraphStore(store)
        self.factory=analyzer_factory;self.poll=poll_seconds;self._analyzer=None;self._stop=threading.Event()
        self.thread=threading.Thread(target=self.loop,name='bibi-analysis',daemon=True)

    def analyzer(self):
        reg=registry_module.read(self.registry_path)
        if not reg.get('active_version'):return None,reg
        if self._analyzer is None or self._analyzer.version!=reg['active_version']:
            self._analyzer=(self.factory(reg) if self.factory else
                            batch.active_analyzer(self.config,self.registry_path,require_validated=not reg.get('test_only')))
        return self._analyzer,reg

    def pending_uploads(self):
        u=self.store.upload;a=self.astore.upload_analysis
        with self.store.engine.connect() as c:
            return c.execute(select(u.c.job_id).outerjoin(a,a.c.upload_id==u.c.job_id).where(u.c.status=='completed',
                             (a.c.upload_id.is_(None))|(a.c.status.in_(['pending_no_active_model','running'])))
                             .order_by(u.c.created_at)).scalars().all()

    def set_status(self,upload_id,status,version=None,details=None,finished=False):
        with self.store.engine.begin() as c:
            self.store.put(c,self.astore.upload_analysis,[{'upload_id':upload_id,'model_version':version,'status':status,
                'details_json':json.dumps(details or {},ensure_ascii=False),'started_at':now(),'finished_at':now() if finished else None}],['upload_id'])

    def run_once(self):
        jobs=self.pending_uploads()
        if not jobs:return 0
        purged=self.astore.purge_orphans();purged['opinion_graph']=self.graph.purge_orphans()
        try:analyzer,reg=self.analyzer()
        except Exception as exc:
            for job in jobs:self.set_status(job,'model_load_failed',None,{'error':f'{type(exc).__name__}: {exc}'[:500]},True)
            return len(jobs)
        for job in jobs:
            if analyzer is None:
                self.set_status(job,'pending_no_active_model',None,{'message':'검증된 운영 모델이 없어 분석 대기','purged':purged},False);continue
            self.set_status(job,'running',analyzer.version)
            try:
                result=batch.analyze(self.store,analyzer,upload_id=job,graph=self.graph)
                self.set_status(job,'completed' if not result['failed_texts'] else 'completed_with_failures',analyzer.version,
                                {**result,'purged_orphans':purged},True)
            except Exception as exc:
                self.set_status(job,'failed',analyzer.version,{'error':f'{type(exc).__name__}: {exc}'[:500]},True)
        return len(jobs)

    def loop(self):
        while not self._stop.is_set():
            try:
                if not self.run_once():self._stop.wait(self.poll)
            except Exception:self._stop.wait(self.poll)

    def start(self):self.thread.start();return self
    def stop(self):self._stop.set();self.thread.join(timeout=10)


def create_service_app(store=None,password=None,upload_dir=paths.DB_FILES/'uploads',config=None,
                       registry_path=None,analyzer_factory=None,start_worker=True,poll_seconds=1.0):
    config=config or load_config()
    store=store or ReviewStore(os.environ.get('BIBI_DATABASE_URL',config['database']))
    registry_path=registry_path or os.environ.get('BIBI_MODEL_REGISTRY',str(registry_module.DEFAULT_REGISTRY))
    # Base app: authentication, upload, ingestion and status are reused unchanged; its own experimental
    # analysis stays off. Our worker performs analysis after ingestion completes.
    app=create_app(store=store,password=password,upload_dir=upload_dir,enable_analysis=False)
    astore=AnalysisStore(store);analytics=Analytics(store,astore)
    graph=OpinionGraphStore(store);opinions=OpinionAnalytics(store,graph)
    worker=AnalysisWorker(store,astore,config,registry_path,analyzer_factory,poll_seconds,graph)
    app.state.worker=worker;app.state.astore=astore;app.state.graph=graph
    if start_worker:worker.start()

    def model(version=None):
        reg=registry_module.read(registry_path)
        version=version or reg.get('active_version')
        if not version:raise HTTPException(409,'검증된 운영 모델이 활성화되지 않아 AI 분석 통계가 없습니다. CSV 기반 통계만 제공합니다.')
        manifest=None
        if reg.get('active_version')==version and reg.get('artifact_dir'):
            path=resolve(reg['artifact_dir'])/'manifest.json'
            manifest=json.loads(path.read_text(encoding='utf-8')) if path.exists() else None
        support=dict(zip(manifest['labels'],manifest['supported'])) if manifest else None
        meta={'model_version':version,'production_validated':bool(reg.get('validated')) and reg.get('active_version')==version,
              'test_only_model':bool(reg.get('test_only')) if reg.get('active_version')==version else None,
              'result_kind':'model_prediction','human_validated':False,
              'note':'모델 예측 결과입니다. 사람 검수 정답이나 확정 통계가 아닙니다.'}
        return version,support,meta

    def product_or_404(pid):
        p=analytics.product(pid)
        if not p:raise HTTPException(404,'상품을 찾을 수 없습니다.')
        return p

    @app.get('/api/v2/models')
    def models():
        reg=registry_module.read(registry_path)
        return {'registry':reg,'llm_calls_in_service':False,'retraining_on_upload':False}

    @app.get('/api/v2/products')
    def products(q:str|None=None):
        rows=store.products()
        return [r for r in rows if not q or q in r['product_name'] or q==r['product_id']]

    @app.get('/api/v2/products/{product_id}/overview')
    def overview(product_id:str,start:str|None=None,end:str|None=None):
        product_or_404(product_id);return analytics.overview(product_id,start,end)

    @app.get('/api/v2/products/{product_id}/aspects')
    def aspects(product_id:str,start:str|None=None,end:str|None=None,model_version:str|None=None):
        product_or_404(product_id);v,support,meta=model(model_version)
        if support is None:raise HTTPException(409,'해당 모델 버전의 출력 지원 정보를 찾을 수 없습니다.')
        return {'meta':meta,'product':analytics.product(product_id),'period':{'start':start,'end':end,'end_exclusive':True},
                **analytics.aspects(product_id,v,support,start,end)}

    @app.get('/api/v2/products/{product_id}/trends')
    def trends(product_id:str,granularity:str=Query('month',pattern='^(month|week)$'),aspect:str|None=None,
               start:str|None=None,end:str|None=None,model_version:str|None=None):
        product_or_404(product_id);v,_,meta=model(model_version)
        return {'meta':meta,**analytics.trend(product_id,v,granularity,aspect,start,end)}

    @app.get('/api/v2/products/{product_id}/emerging')
    def emerging(product_id:str,as_of:str|None=None,recent_days:int=Query(28,ge=7,le=180),baseline_days:int=Query(84,ge=14,le=365)):
        product_or_404(product_id);v,support,meta=model()
        return {'meta':meta,**analytics.emerging(product_id,v,support,as_of or date.today().isoformat(),recent_days,baseline_days)}

    @app.get('/api/v2/products/{product_id}/reviews')
    def product_reviews(product_id:str,aspect:str|None=None,sentiment:str|None=Query(None,pattern='^(positive|negative)$'),
                        rating:int|None=Query(None,ge=1,le=5),start:str|None=None,end:str|None=None,
                        status:str|None=Query(None,pattern='^(classified|unclassified|review_required)$'),
                        limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0),model_version:str|None=None):
        product_or_404(product_id);v,_,meta=model(model_version)
        return {'meta':meta,'reviews':analytics.reviews(product_id,v,aspect,sentiment,rating,start,end,status,limit,offset)}

    @app.get('/api/v2/reviews/{review_id}')
    def review(review_id:str,model_version:str|None=None):
        reg=registry_module.read(registry_path);v=model_version or reg.get('active_version')
        row=analytics.review(review_id,v)
        if not row:raise HTTPException(404,'리뷰를 찾을 수 없습니다.')
        return {'model_version':v,'review':row}

    @app.get('/api/v2/processing/status')
    def processing():
        reg=registry_module.read(registry_path)
        return {'active_model':reg.get('active_version'),'production_validated':bool(reg.get('validated')),
                **analytics.processing(reg.get('active_version'))}

    @app.get('/api/v2/review-queue')
    def review_queue(product_id:str|None=None,limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
        v,_,meta=model();return {'meta':meta,'items':analytics.queue(v,product_id,limit,offset)}

    @app.post('/api/v2/reviews/{review_id}/corrections',status_code=201)
    def correction(review_id:str,body:Correction):
        reg=registry_module.read(registry_path)
        try:cid=astore.add_correction(review_id,body.annotator,[o.model_dump() for o in body.opinions],body.complete,reg.get('active_version'))
        except LookupError:raise HTTPException(404,'리뷰를 찾을 수 없습니다.')
        except ValueError as exc:raise HTTPException(400,str(exc))
        return {'correction_id':cid,'stored_as':'human correction (separate from model predictions; used only by admin retraining)'}

    # --- Opinion graph (ontology v1): hierarchy, instances, evidence, recurring/common reasons -------------
    def tier_args(tier,model_version):
        """prediction tier = active (or explicit) model; gold/silver = annotation layers, never mixed."""
        if tier=='prediction':
            v,_,_=model(model_version);return v
        return None

    def opinion_call(fn):
        try:return fn()
        except ValueError as exc:raise HTTPException(400,str(exc))

    @app.get('/api/v2/ontology')
    def ontology_tree():
        return graph.ontology.tree()

    @app.get('/api/v2/products/{product_id}/opinion-tree')
    def opinion_tree(product_id:str,tier:str=Query('prediction',pattern='^(prediction|gold|silver)$'),source:str|None=None,model_version:str|None=None,start:str|None=None,end:str|None=None):
        product_or_404(product_id);v=tier_args(tier,model_version)
        return opinion_call(lambda:{'product':analytics.product(product_id),**opinions.hierarchy(product_id,tier,v,source,start,end)})

    @app.get('/api/v2/products/{product_id}/opinions')
    def opinion_list(product_id:str,tier:str=Query('prediction',pattern='^(prediction|gold|silver)$'),source:str|None=None,model_version:str|None=None,aspect:str|None=None,
                     sentiment:str|None=Query(None,pattern='^(positive|negative|mixed|neutral|uncertain|not_applicable)$'),
                     reason:str|None=None,target_scope:str|None=Query('current_product',pattern='^(current_product|other_product|general_experience|unclear)$'),
                     start:str|None=None,end:str|None=None,limit:int=Query(50,ge=1,le=200),offset:int=Query(0,ge=0)):
        product_or_404(product_id);v=tier_args(tier,model_version)
        return opinion_call(lambda:{'meta':opinions.meta(tier,v,source),'opinions':opinions.opinion_instances(product_id,tier,v,source,aspect,sentiment,
                                    reason,target_scope,start,end,limit,offset)})

    @app.get('/api/v2/products/{product_id}/reasons')
    def product_reasons(product_id:str,tier:str=Query('prediction',pattern='^(prediction|gold|silver)$'),source:str|None=None,model_version:str|None=None,
                        sentiment:str=Query('negative',pattern='^(positive|negative)$'),start:str|None=None,end:str|None=None,min_reviews:int=Query(1,ge=1)):
        product_or_404(product_id);v=tier_args(tier,model_version)
        return opinion_call(lambda:opinions.recurring_reasons(product_id,tier,v,source,sentiment,start,end,min_reviews))

    @app.get('/api/v2/products/{product_id}/opinion-cooccurrence')
    def opinion_cooccurrence(product_id:str,tier:str=Query('prediction',pattern='^(prediction|gold|silver)$'),source:str|None=None,model_version:str|None=None,
                             level:str=Query('reason',pattern='^(reason|aspect)$'),start:str|None=None,end:str|None=None,min_reviews:int=Query(2,ge=1)):
        product_or_404(product_id);v=tier_args(tier,model_version)
        return opinion_call(lambda:opinions.cooccurrence(product_id,tier,v,source,start,end,level,min_reviews))

    @app.get('/api/v2/reasons/{reason_code}')
    def reason_detail(reason_code:str,tier:str=Query('prediction',pattern='^(prediction|gold|silver)$'),source:str|None=None,model_version:str|None=None,product_id:str|None=None,
                      sentiment:str|None=Query(None,pattern='^(positive|negative)$'),limit:int=Query(20,ge=1,le=200)):
        if reason_code not in graph.ontology.reasons:raise HTTPException(404,'이유 코드를 찾을 수 없습니다.')
        v=tier_args(tier,model_version)
        return opinion_call(lambda:opinions.evidence_trace(reason_code,tier,v,source,product_id,sentiment,limit))

    @app.get('/api/v2/reason-candidates')
    def reason_candidates(product_id:str|None=None,status:str=Query('pending_review',pattern='^(pending_review|accepted|rejected)$'),
                          limit:int=Query(100,ge=1,le=500),offset:int=Query(0,ge=0)):
        return opinions.reason_candidates(product_id,status,None,limit,offset)

    @app.get('/api/v2/opinion-graph/status')
    def opinion_graph_status():
        reg=registry_module.read(registry_path)
        return opinions.status(reg.get('active_version'))

    return app


def app_from_env():
    return create_service_app(password=os.environ.get('BIBI_PASSWORD',''))
