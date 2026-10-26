"""Persistent cached inference for new/missing current review texts."""
import argparse
import json
import uuid
from sqlalchemy import select,and_,update
from .backbone import ReviewStore,now


def analyze_pending(store,analyzer,upload_id=None,product_id=None,limit=10000):
    run_id=uuid.uuid4().hex
    details={'upload_id':upload_id,'product_id':product_id,'computed_texts':0,'cached_texts':0,'review_rows':0}
    with store.engine.begin() as c:c.execute(store.runs.insert().values(run_id=run_id,
        model_version=analyzer.version,status='running',details_json=json.dumps(details),created_at=now()))
    try:
        stmt=select(store.review.c.review_id,store.review.c.review_text,store.review.c.text_hash).where(
            store.review.c.display_status=='정상')
        if upload_id:stmt=stmt.join(store.membership,store.membership.c.review_id==store.review.c.review_id).where(store.membership.c.upload_id==upload_id)
        if product_id:stmt=stmt.where(store.review.c.product_id==product_id)
        # Select only missing current input/version combinations.
        stmt=stmt.outerjoin(store.cache,and_(store.cache.c.text_hash==store.review.c.text_hash,
            store.cache.c.model_version==analyzer.version)).where(store.cache.c.text_hash.is_(None))
        stmt=stmt.order_by(store.review.c.review_id).limit(limit)
        with store.engine.connect() as c:rows=list(c.execute(stmt).mappings())
        for row in rows:
            hit=store.cached(row['text_hash'],analyzer.version)
            if hit is None:
                result=analyzer.analyze(row['review_text'])
                store.cache_result(row['text_hash'],analyzer.version,result)
                details['computed_texts']+=1
            else:details['cached_texts']+=1
            details['review_rows']+=1
        with store.engine.begin() as c:c.execute(update(store.runs).where(store.runs.c.run_id==run_id)
            .values(status='completed',details_json=json.dumps(details)))
        return {'run_id':run_id,**details}
    except Exception as exc:
        details['error']=str(exc)[:500]
        with store.engine.begin() as c:c.execute(update(store.runs).where(store.runs.c.run_id==run_id)
            .values(status='failed',details_json=json.dumps(details)))
        raise


def main():
    from .local_inference import LocalAnalyzer
    p=argparse.ArgumentParser()
    p.add_argument('--db',default=None)
    p.add_argument('--product-id');p.add_argument('--upload-id');p.add_argument('--limit',type=int,default=1000)
    args=p.parse_args()
    print(json.dumps(analyze_pending(ReviewStore(args.db),LocalAnalyzer(),args.upload_id,args.product_id,args.limit),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
