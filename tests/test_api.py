import time
import pytest
from fastapi.testclient import TestClient
from bibi_reviews.api import create_app
from bibi_reviews.backbone import ReviewStore


def test_auth_upload_status_and_queries(tmp_path):
    store=ReviewStore('sqlite:///'+str(tmp_path/'api.db'))
    app=create_app(store=store,password='test-passphrase-123',upload_dir=tmp_path/'uploads')
    with TestClient(app) as c:
        assert c.get('/api/summary').status_code==401
        assert c.get('/api/summary',auth=('bibi','wrong')).status_code==401
        auth=('bibi','test-passphrase-123')
        assert c.get('/api/summary',auth=auth).json()['review_count']==0
        raw='상품번호,상품명,리뷰글번호,리뷰상세내용,구매자평점,리뷰등록일,전시상태\np,새우,r,맛있다,5,2025-01-01,정상\n'.encode('utf-8-sig')
        response=c.post('/api/uploads',auth=auth,files={'file':('reviews.csv',raw,'text/csv')})
        assert response.status_code==202
        job=response.json()['job_id']
        for _ in range(100):
            status=c.get('/api/uploads/'+job,auth=auth).json()
            if status['status'] in ['completed','failed']:break
            time.sleep(.02)
        assert status['status']=='completed' and status['new']==1
        assert c.get('/api/products',auth=auth).json()[0]['product_id']=='p'
        assert c.get('/api/reviews',auth=auth,params={'product_id':'p'}).json()[0]['review_text']=='맛있다'
        assert c.get('/api/uploads/missing',auth=auth).status_code==404
        assert c.get('/openapi.json').status_code==404
        assert c.post('/api/uploads',auth=auth,files={'file':('x.exe',b'x')}).status_code==400


def test_no_default_or_short_password(tmp_path):
    with pytest.raises(ValueError):create_app(password='short',upload_dir=tmp_path)


def test_incomplete_upload_is_not_imported_on_restart(tmp_path):
    store=ReviewStore('sqlite:///'+str(tmp_path/'partial.db'))
    directory=tmp_path/'uploads';directory.mkdir()
    job=store.create_upload('partial.csv',status='receiving')
    (directory/f'{job}.csv.part').write_text('unfinished',encoding='utf-8')
    app=create_app(store=store,password='test-passphrase-123',upload_dir=directory)
    with TestClient(app):pass
    assert store.get_upload(job)['status']=='failed'
    assert store.summary()['review_count']==0


def test_unauthorized_body_is_never_received_and_oversize_rejected_early(tmp_path):
    import asyncio
    store=ReviewStore('sqlite:///'+str(tmp_path/'guard.db'))
    app=create_app(store=store,password='test-passphrase-123',upload_dir=tmp_path/'uploads',max_upload_bytes=100)
    received=[];sent=[]
    async def receive():
        received.append(True)
        return {'type':'http.request','body':b'x'*1000,'more_body':False}
    async def send(message):sent.append(message)
    scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','method':'POST',
        'scheme':'http','path':'/api/uploads','raw_path':b'/api/uploads','query_string':b'',
        'headers':[(b'content-type',b'multipart/form-data; boundary=x')],
        'server':('test',80),'client':('test',123)}
    asyncio.run(app(scope,receive,send))
    assert not received and sent[0]['status']==401
    import base64
    scope['headers'].extend([(b'authorization',b'Basic '+base64.b64encode(b'bibi:test-passphrase-123')),
                             (b'content-length',b'999999')])
    received.clear();sent.clear()
    asyncio.run(app(scope,receive,send))
    assert not received and sent[0]['status']==413


def test_automatic_analysis_runs_after_upload_and_replay_is_cached(tmp_path,monkeypatch):
    import bibi_reviews.local_inference
    class Fake:
        version='api-analysis-test'
        def analyze(self,text):return {'status':'classified','opinions':[]}
    monkeypatch.setattr(bibi_reviews.local_inference,'LocalAnalyzer',Fake)
    store=ReviewStore('sqlite:///'+str(tmp_path/'auto.db'))
    app=create_app(store=store,password='test-passphrase-123',upload_dir=tmp_path/'uploads',enable_analysis=True)
    raw='상품번호,상품명,리뷰글번호,리뷰상세내용,구매자평점,리뷰등록일,전시상태\np,새우,r,맛있다,5,2025-01-01,정상\n'.encode('utf-8-sig')
    with TestClient(app) as c:
        auth=('bibi','test-passphrase-123')
        for expected in [1,0]:
            job=c.post('/api/uploads',auth=auth,files={'file':('reviews.csv',raw)}).json()['job_id']
            for _ in range(100):
                status=c.get('/api/uploads/'+job,auth=auth).json()
                if status['status'] in ['completed','failed','completed_with_analysis_error']:break
                time.sleep(.01)
            assert status['status']=='completed'
            assert status['analysis']['computed_texts']==expected
        assert c.get('/api/reviews',auth=auth,params={'model_version':'api-analysis-test'}).json()[0]['analysis'] is not None
