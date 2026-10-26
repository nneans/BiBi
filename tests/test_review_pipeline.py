"""Contract tests for the review pipeline (synthetic data, isolated DBs; no network, no LLM)."""
import json
import socket
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pytest

from bibi_reviews.backbone import ReviewStore,normalize_text,hash_text
from bibi_reviews.review_pipeline.common import load_config,load_taxonomy,output_aspects,label_names,LLMDisabledError
from bibi_reviews.classifier.data import split
from bibi_reviews.classifier import registry
from bibi_reviews.classifier.inference import ClassifierAnalyzer,ModelArtifactError
from bibi_reviews.service.storage import AnalysisStore
from bibi_reviews.service import batch

CONFIG=load_config();TAX=load_taxonomy(CONFIG);NAMES=label_names(output_aspects(TAX))
TEMPLATES=[('맛있어요 정말 맛있게 먹었습니다',['taste:positive']),('맛이 별로예요 실망입니다',['taste:negative']),
           ('배송이 빨라서 좋아요',['delivery:positive']),('배송이 너무 늦게 왔어요',['delivery:negative']),
           ('포장이 터져서 왔어요',['packaging:negative']),('포장 꼼꼼해요',['packaging:positive']),
           ('가격이 저렴해서 만족',['price:positive']),('양이 너무 적어요',['quantity:negative']),
           ('맛있는데 배송이 늦었어요',['taste:positive','delivery:negative'])]


def synthetic_records(n=180):
    rng=np.random.default_rng(0);out=[]
    for i in range(n):
        text,labels=TEMPLATES[i%len(TEMPLATES)];text=f'{text} {"!"*(i%3)}{i}'
        y=[int(name in labels) for name in NAMES]
        out.append({'review_id':f'r{i}','product_id':f'p{i%6}','text':text,'text_hash':hash_text(normalize_text(text)),
                    'label_source':'llm_silver','targets':y,'mask':[1]*len(NAMES),'reserved_human_eval':False})
    return out


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    from bibi_reviews.classifier.train import train
    out=tmp_path_factory.mktemp('model')/'tfidf'
    manifest,_=train(CONFIG,synthetic_records(),'tfidf_lr','silver_smoke',out)
    assert manifest['status']=='smoke_test_not_for_production' and not manifest['gate_passed']
    return out


def csv_bytes(rows):
    head='상품번호,상품명,리뷰글번호,리뷰상세내용,구매자평점,리뷰등록일,최종수정일,전시상태\n'
    return (head+''.join(','.join(map(str,r))+'\n' for r in rows)).encode('utf-8-sig')


# ---------- labels, masking, leakage ----------

def test_train_test_family_leakage_is_blocked():
    recs=synthetic_records(60)
    for r in recs[:30]:r['label_source']='human_gold'
    recs[1]['text_hash']=recs[0]['text_hash'];recs[2]['related_review_id']='r0'   # duplicates / related review family
    parts=split(recs,CONFIG,'gold_plus_silver')
    where={i:k for k in ('train','validation','test') for i in parts[k]}
    assert len({where[i] for i in (0,1,2) if i in where})==1
    assert all(recs[i]['label_source']=='human_gold' for i in parts['validation']+parts['test'])


# ---------- model artifact and registry ----------

def test_corrupt_or_missing_model_fails_clearly(artifact,tmp_path):
    import shutil
    broken=tmp_path/'broken';shutil.copytree(artifact,broken)
    (broken/'model/tfidf_lr.pkl').write_bytes(b'corrupted')
    with pytest.raises(ModelArtifactError,match='checksum'):ClassifierAnalyzer(broken,CONFIG)
    with pytest.raises(ModelArtifactError,match='No model artifact'):ClassifierAnalyzer(tmp_path/'none',CONFIG)


def test_unvalidated_model_cannot_be_activated_in_production_registry(artifact,tmp_path):
    with pytest.raises(PermissionError):registry.activate(artifact,tmp_path/'reg.json')
    with pytest.raises(PermissionError):registry.activate(artifact,registry.DEFAULT_REGISTRY,allow_unvalidated_for_testing=True)


def test_evidence_offsets_match_unicode_text(artifact):
    a=ClassifierAnalyzer(artifact,CONFIG,taxonomy=TAX)
    text='😀 정말 맛있어요! 그런데 배송이 너무 늦게 왔어요 🚚'
    result=a.analyze(text)
    assert result['labels'] and not result['production_validated']
    for l in result['labels']:assert text[l['evidence_start']:l['evidence_end']]==l['evidence_sentence']


# ---------- service end-to-end (isolated DB, network blocked) ----------

@pytest.fixture
def no_network(monkeypatch):
    real=socket.socket.connect
    def guarded(self,address):
        host=address[0] if isinstance(address,tuple) else address
        if host not in ('127.0.0.1','::1','localhost'):raise AssertionError(f'network access attempted: {address}')
        return real(self,address)
    monkeypatch.setattr(socket.socket,'connect',guarded);monkeypatch.delenv('LLM_ENABLED',raising=False)
    monkeypatch.delenv('GROQ_API_KEY',raising=False)


def wait(client,auth,job):
    for _ in range(300):
        s=client.get('/api/uploads/'+job,auth=auth).json()
        if s['status'] in ('completed','failed'):return s
        time.sleep(.02)
    raise AssertionError('upload did not finish')


def test_end_to_end_csv_to_dashboard_offline(artifact,tmp_path,no_network,review_db_url):  # SQLite and PostgreSQL
    from fastapi.testclient import TestClient
    from bibi_reviews.service.app import create_service_app
    reg=tmp_path/'registry.json';registry.activate(artifact,reg,allow_unvalidated_for_testing=True)
    store=ReviewStore(review_db_url)
    app=create_service_app(store=store,password='test-passphrase-123',upload_dir=tmp_path/'up',config=CONFIG,registry_path=reg,start_worker=False)
    auth=('bibi','test-passphrase-123');worker=app.state.worker
    base=[('p1','새우',f'r{i}',t,5 if i%2 else 2,f'2026-0{1+i%3}-1{i%9} 10:00:00','','정상') for i,(t,_) in enumerate(TEMPLATES)]
    base.append(('p1','새우','dup','맛있어요 정말 맛있게 먹었습니다',5,'2026-03-01 10:00:00','','정상'))  # same text as r0
    with TestClient(app) as c:
        assert c.get('/api/v2/products/p1/aspects',auth=auth).status_code in (404,409)
        s=wait(c,auth,c.post('/api/uploads',auth=auth,files={'file':('a.csv',csv_bytes(base),'text/csv')}).json()['job_id'])
        assert s['new']==len(base);worker.run_once()
        status=c.get('/api/v2/processing/status',auth=auth).json()
        up=status['uploads'][0];details=json.loads(up['analysis_details'])
        assert up['analysis_status']=='completed' and details['computed_texts']==len(TEMPLATES)   # duplicate text computed once
        assert status['totals']['analyzed_reviews']==len(base) and status['totals']['pending_reviews']==0
        aspects=c.get('/api/v2/products/p1/aspects',auth=auth).json()
        assert aspects['meta']['production_validated'] is False and aspects['meta']['test_only_model'] is True
        assert aspects['coverage']['analyzed']==len(base)
        taste=next(a for a in aspects['aspects'] if a['aspect']=='taste')
        assert taste['positive_reviews']>=2 and taste['positive_share_of_analyzed']==taste['positive_reviews']/len(base)
        listed=c.get('/api/v2/products/p1/reviews',auth=auth,params={'aspect':'delivery','sentiment':'negative'}).json()['reviews']
        assert listed and all(any(p['aspect']=='delivery' and p['sentiment']=='negative' for p in r['predictions']) for r in listed)
        for r in listed:
            for p in r['predictions']:assert r['review_text'][p['evidence_start']:p['evidence_end']]==p['evidence_text']
        # (1) re-upload identical CSV: nothing new, nothing recomputed
        s=wait(c,auth,c.post('/api/uploads',auth=auth,files={'file':('a.csv',csv_bytes(base),'text/csv')}).json()['job_id'])
        assert s['new']==0 and s['updated']==0;worker.run_once()
        assert json.loads(c.get('/api/v2/processing/status',auth=auth).json()['uploads'][0]['analysis_details'])['computed_texts']==0
        # (4) rating-only change reuses inference; (3) text edit re-analyzes; (5) blind removes text and analysis
        changed=list(map(list,base))
        changed[1][4]=1;changed[1][6]='2026-04-01 10:00:00'
        changed[2][3]='배송이 빨라서 좋아요 근데 양이 너무 적어요';changed[2][6]='2026-04-01 10:00:00'
        changed[3][7]='블라인드';changed[3][6]='2026-04-01 10:00:00'
        s=wait(c,auth,c.post('/api/uploads',auth=auth,files={'file':('b.csv',csv_bytes(changed),'text/csv')}).json()['job_id'])
        assert s['updated']==3;worker.run_once()
        details=json.loads(c.get('/api/v2/processing/status',auth=auth).json()['uploads'][0]['analysis_details'])
        assert details['computed_texts']==1 and details['purged_orphans']['labels']>=1
        r2=c.get('/api/v2/reviews/r2',auth=auth).json()['review']
        assert r2['review_text'].endswith('양이 너무 적어요') and r2['analysis_status'] is not None
        assert c.get('/api/v2/reviews/r3',auth=auth).status_code==404
        old_hash=hash_text(normalize_text(TEMPLATES[3][0]))
        with store.engine.connect() as conn:
            from sqlalchemy import text
            assert conn.execute(text('select count(*) from analysis_labels where text_hash=:h'),{'h':old_hash}).scalar()==0
            assert conn.execute(text('select count(*) from analysis_cache where text_hash=:h'),{'h':old_hash}).scalar()==0
        # human correction is stored separately from predictions
        body={'annotator':'tester','complete':True,'opinions':[{'target_scope':'current_product','statement_type':'evaluation',
              'aspect':'quantity','sentiment':'negative','evidence_text':'양이 너무 적어요'}]}
        assert c.post('/api/v2/reviews/r2/corrections',auth=auth,json=body).status_code==201
        assert c.post('/api/v2/reviews/r2/corrections',auth=auth,json={**body,'annotator':''}).status_code==422
        trend=c.get('/api/v2/products/p1/trends',auth=auth).json()
        assert trend['single_model_version']==aspects['meta']['model_version'] and trend['series']
        assert c.get('/api/v2/products/p1/emerging',auth=auth,params={'as_of':'2026-04-30'}).status_code==200
        assert c.get('/api/v2/products/p1/overview',auth=auth).json()['source']=='uploaded CSV data (not AI)'
        # invalid CSV: rejected, nothing applied
        bad=c.post('/api/uploads',auth=auth,files={'file':('bad.csv','상품번호,상품명\np,x\n'.encode('utf-8-sig'),'text/csv')}).json()['job_id']
        assert wait(c,auth,bad)['status']=='failed'
        assert c.get('/api/v2/processing/status',auth=auth).json()['totals']['normal_reviews']==len(base)-1


def test_partial_failures_preserve_other_results_and_versions_stay_separate(artifact,tmp_path):
    store=ReviewStore('sqlite:///'+str(tmp_path/'v.db'))
    frame=pd.DataFrame([{'상품번호':'p','상품명':'n','리뷰글번호':f'r{i}','리뷰상세내용':t,'구매자평점':5,'리뷰등록일':'2026-01-01','전시상태':'정상'}
                        for i,(t,_) in enumerate(TEMPLATES)])
    store.import_frame(frame)
    real=ClassifierAnalyzer(artifact,CONFIG,taxonomy=TAX)
    class Flaky:
        version='flaky-v1';validated=False
        def analyze_batch(self,texts):
            if any('터져' in t for t in texts):raise RuntimeError('boom')
            return [r|{'model_version':self.version} for r in real.analyze_batch(texts)]
    result=batch.analyze(store,Flaky(),batch_size=4)
    assert result['failed_texts']==1 and result['computed_texts']==len(TEMPLATES)-1
    again=batch.analyze(store,Flaky())
    assert again['computed_texts']==0 and again['failed_texts']==0                       # failures not retried silently
    batch.analyze(store,real)
    astore=AnalysisStore(store)
    with store.engine.connect() as c:
        from sqlalchemy import text
        versions=dict(c.execute(text('select model_version,count(*) from analysis_status group by 1')).all())
    assert versions=={'flaky-v1':len(TEMPLATES)-1,real.version:len(TEMPLATES)}
