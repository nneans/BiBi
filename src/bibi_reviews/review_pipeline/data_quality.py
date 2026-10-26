"""STEP 1: read-only data quality profile of the review DB. No model and no LLM.

Cue counts are surface-expression candidates for sampling and exploration only;
they are not sentiment labels, and rating is never converted into sentiment.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .common import ROOT,save_json,load_config,readonly_engine,display_url
from .. import paths

CUES={
    'negative_candidate':r'별로|실망|최악|비리|비린|냄새|늦게|늦어|녹아|녹았|터져|터졌|짜요|너무 짜|질기|딱딱|가시가|비싸|양이 적|작아요|환불|반품|상했|상한|이상한|아쉬|불만|다시는|안 사|안사|쓰레기|버렸',
    'positive_candidate':r'맛있|맛나|신선|빠르|빨라|좋아요|좋네|만족|재구매|저렴|푸짐|넉넉|부드럽|바삭|최고|추천|싱싱',
    'contrast_marker':r'근데|그런데|하지만|그러나|지만|는데|은데|다만|그래도',
    'other_product_reference':r'다른 (?:상품|제품|곳|데)|타사|마트(?:에서|보다)|전에 (?:산|샀)',
}
ASPECT_CUES={
    'taste':r'맛|간이|짜|싱겁|비린|냄새|고소',
    'freshness':r'신선|싱싱|상했|상한|유통기한',
    'quality':r'상태|가시|뼈|이물|부서|살이|해동|녹',
    'delivery':r'배송|도착|택배',
    'packaging':r'포장|아이스팩|박스|보냉|터져|새서|샜',
    'price':r'가격|가성비|저렴|비싸|할인|배송비',
    'quantity':r'양이|양도|양은|크기|사이즈|중량|두께|개수|넉넉|푸짐',
}


def load_reviews(db):
    with readonly_engine(db).connect() as con:
        frame=pd.read_sql_query('''select r.review_id,r.product_id,p.product_name,r.product_name_original,r.review_type,
            r.rating,r.created_at,r.related_review_id,r.display_status,r.review_text,r.text_hash
            from reviews r join products p on p.product_id=r.product_id order by r.review_id''',con)  # fixed order: same on every engine
        jobs=pd.read_sql_query('select job_id,filename,status,created_at,finished_at,error from upload_jobs order by created_at,job_id',con)
        cache=pd.read_sql_query('select model_version,count(*) n from analysis_cache group by model_version order by model_version',con)
    return frame,jobs,cache


def profile(frame,jobs,cache):
    normal=frame[frame.display_status=='정상'].copy()
    normal['length']=normal.review_text.str.len()
    for key,pattern in {**CUES,**{'aspect:'+k:v for k,v in ASPECT_CUES.items()}}.items():
        normal[key]=normal.review_text.str.contains(pattern,regex=True)
    dup_counts=normal.text_hash.map(normal.text_hash.value_counts())
    normal['exact_duplicate_text']=dup_counts>1
    normal['month']=normal.created_at.str[:7]
    normal['week']=pd.to_datetime(normal.created_at.str[:10]).dt.to_period('W-SUN').astype(str)
    names=frame.groupby('product_id').product_name_original.nunique()
    q=normal.length.quantile([.05,.25,.5,.75,.95,.99]).round(1)
    summary={
        'rows':int(len(frame)),'normal_reviews':int(len(normal)),
        'blind_reviews':int((frame.display_status=='블라인드').sum()),
        'products_with_normal_reviews':int(normal.product_id.nunique()),
        'products_total':int(frame.product_id.nunique()),
        'duplicate_review_ids':int(frame.review_id.duplicated().sum()),
        'review_id_uniqueness':'enforced by DB primary key',
        'products_with_multiple_source_names':int((names>1).sum()),
        'representative_name':'products.product_name = most frequent normal-review name per product_id',
        'review_type_counts':frame.review_type.value_counts().to_dict(),
        'linked_related_reviews':int((frame.related_review_id.fillna('')!='').sum()),
        'period':{'start':normal.created_at.min(),'end':normal.created_at.max()},
        'rating_counts':{str(k):int(v) for k,v in normal.rating.value_counts().sort_index().items()},
        'mean_rating':round(float(normal.rating.mean()),4),
        'length_chars_quantiles':{str(k):float(v) for k,v in q.items()},
        'empty_or_symbol_only_texts':int((~normal.review_text.str.contains(r'[가-힣A-Za-z0-9]',regex=True)).sum()),
        'distinct_texts':int(normal.text_hash.nunique()),
        'reviews_with_exact_duplicate_text':int(normal.exact_duplicate_text.sum()),
        'cue_shares':{k:round(float(normal[k].mean()),4) for k in CUES},
        'five_star_with_negative_candidate_cue':int(((normal.rating==5)&normal.negative_candidate).sum()),
        'low_rating_without_negative_candidate_cue':int(((normal.rating<=2)&~normal.negative_candidate).sum()),
        'upload_jobs':jobs.to_dict('records'),
        'stale_running_upload_jobs':jobs[jobs.status.isin(['running','queued','analyzing'])].job_id.tolist(),
        'analysis_cache_by_model_version':cache.to_dict('records'),
        'analysis_cache_note':'existing cache rows are experimental local-v1 outputs; not production predictions',
        'cue_note':'surface-expression candidates only; not sentiment labels. Rating is not an aspect sentiment label.',
    }
    per_product=normal.groupby(['product_id','product_name']).agg(
        normal_reviews=('review_id','size'),mean_rating=('rating','mean'),
        low_rating_1_2=('rating',lambda s:int((s<=2).sum())),rating_3=('rating',lambda s:int((s==3).sum())),
        median_length=('length','median'),exact_duplicate_share=('exact_duplicate_text','mean'),
        negative_candidate_share=('negative_candidate','mean'),positive_candidate_share=('positive_candidate','mean'),
        contrast_share=('contrast_marker','mean'),first=('created_at','min'),last=('created_at','max'),
        **{f'cue_{k}':(f'aspect:{k}','mean') for k in ASPECT_CUES}).reset_index()
    blind=frame[frame.display_status=='블라인드'].groupby('product_id').size()
    per_product['blind_reviews']=per_product.product_id.map(blind).fillna(0).astype(int)
    per_product=per_product.sort_values('normal_reviews',ascending=False).round(4)
    monthly=normal.groupby(['month','rating']).size().unstack(fill_value=0)
    monthly['total']=monthly.sum(axis=1)
    duplicates=(normal.groupby('text_hash').agg(count=('review_id','size'),text=('review_text','first'),
        products=('product_id','nunique'),mean_rating=('rating','mean')).sort_values('count',ascending=False).head(50))
    length_bins=pd.cut(normal.length,[0,5,10,20,50,100,200,500,10**6],right=True).value_counts().sort_index()
    rating_cue=pd.crosstab(normal.rating,normal.negative_candidate,normalize='index').round(4)
    return summary,per_product,monthly,duplicates,length_bins,rating_cue,normal


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',default=str(ROOT/'configs/review_pipeline/pipeline.json'))
    p.add_argument('--output',default=str(paths.RAW/'data_quality'))
    args=p.parse_args()
    config=load_config(args.config)
    frame,jobs,cache=load_reviews(config)
    summary,per_product,monthly,dups,bins,rating_cue,_=profile(frame,jobs,cache)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary['database']=display_url(config);summary['read_only']=True
    save_json(out/'summary.json',summary)
    per_product.to_csv(out/'product_profile.csv',index=False,encoding='utf-8-sig')
    monthly.to_csv(out/'monthly_rating_counts.csv',encoding='utf-8-sig')
    dups.to_csv(out/'top_duplicate_texts.csv',encoding='utf-8-sig')
    bins.rename('reviews').to_csv(out/'length_bins.csv',encoding='utf-8-sig')
    rating_cue.to_csv(out/'rating_vs_negative_cue.csv',encoding='utf-8-sig')
    print(f"normal={summary['normal_reviews']} products={summary['products_with_normal_reviews']} output={out}")


if __name__=='__main__':main()
