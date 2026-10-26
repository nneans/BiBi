import pandas as pd
import pytest
from bibi_reviews.backbone import ReviewStore


def frame(rows):
    base={'상품번호':'p1','상품명':'생선','리뷰글번호':'r1','리뷰상세내용':'맛있어요',
          '구매자평점':'5','리뷰등록일':'2025.01.01. 10:00:00','최종수정일':'',
          '리뷰구분':'일반','관련리뷰글번호':'','전시상태':'정상'}
    return pd.DataFrame([base|r for r in rows])


@pytest.fixture
def store(tmp_path):
    return ReviewStore('sqlite:///'+str(tmp_path/'reviews.db'))


def test_add_repeat_edit_rating_cache_and_stale(store):
    assert store.import_frame(frame([{}]))['new']==1
    assert store.import_frame(frame([{}]))['unchanged']==1
    old=store.list_reviews()[0]
    store.cache_result(old['text_hash'],'test-v1',{'status':'classified','opinions':[]})
    assert store.import_frame(frame([{'구매자평점':'1'}]))['updated']==1
    assert store.list_reviews(model_version='test-v1')[0]['analysis'] is not None
    assert store.summary()['mean_rating']==1
    assert store.import_frame(frame([{'리뷰상세내용':'너무 짜요','최종수정일':'2025.02.01. 10:00:00'}]))['updated']==1
    assert store.list_reviews(model_version='test-v1')[0]['analysis'] is None
    assert store.import_frame(frame([{}]))['stale']==1
    assert store.summary()['review_count']==1


def test_blind_transitions_and_different_ids_same_text(store):
    store.import_frame(frame([{}, {'리뷰글번호':'r2'}]))
    assert store.summary()['review_count']==2
    store.import_frame(frame([{'전시상태':'블라인드','최종수정일':'2025.03.01. 10:00:00'}]))
    assert store.summary()['review_count']==1
    assert len(store.list_reviews())==1
    store.import_frame(frame([{'최종수정일':'2025.04.01. 10:00:00'}]))
    assert store.summary()['review_count']==2


def test_invalid_frame_rolls_back_and_duplicate_ids_rejected(store):
    with pytest.raises(ValueError):
        store.import_frame(frame([{}, {'리뷰글번호':'r2','구매자평점':'8'}]))
    assert store.summary()['review_count']==0
    with pytest.raises(ValueError): store.import_frame(frame([{},{}]))
    assert store.summary()['review_count']==0


def test_csv_jobs_and_canonical_names(store,tmp_path):
    p=tmp_path/'new.csv'
    frame([{}, {'리뷰글번호':'r2','상품명':'명태포'}, {'리뷰글번호':'r3','상품명':'명태포'}]).to_csv(p,index=False,encoding='utf-8-sig')
    result=store.import_csv(p)
    assert result['new']==3 and result['status']=='completed'
    assert store.products()[0]['product_name']=='명태포'
    assert store.get_upload(result['job_id'])['new']==3
    assert store.import_csv(p)['unchanged']==3


def test_failure_after_first_chunk_rolls_back(store,tmp_path):
    p=tmp_path/'bad.csv'
    frame([{}, {'리뷰글번호':'r2','리뷰등록일':'invalid'}]).to_csv(p,index=False,encoding='utf-8-sig')
    with pytest.raises(ValueError): store.import_csv(p,chunksize=1)
    assert store.summary()['review_count']==0


def test_unchanged_upload_can_retry_missing_analysis(store,tmp_path):
    from bibi_reviews.local_worker import analyze_pending
    class Fake:
        version='retry-test'
        def analyze(self,text):return {'status':'classified','opinions':[]}
    path=tmp_path/'retry.csv';frame([{}]).to_csv(path,index=False,encoding='utf-8-sig')
    store.import_csv(path)
    job=store.import_csv(path)
    assert job['unchanged']==1
    assert analyze_pending(store,Fake(),upload_id=job['job_id'])['computed_texts']==1


def test_blind_transition_scrubs_history_and_orphaned_cache(store):
    from sqlalchemy import select
    import json
    store.import_frame(frame([{'리뷰상세내용':'private old text'}]))
    h=store.list_reviews()[0]['text_hash']
    store.cache_result(h,'private-test',{'evidence':'private old text'})
    store.import_frame(frame([{'리뷰상세내용':'private old text','구매자평점':'4'}]))
    store.import_frame(frame([{'전시상태':'블라인드','최종수정일':'2026-01-01'}]))
    with store.engine.connect() as c:
        histories=c.execute(select(store.history.c.previous_json)).scalars().all()
    assert all(json.loads(x)['review_text'] is None for x in histories)
    assert store.cached(h,'private-test') is None


def test_analysis_phase_is_committed_with_import(store,tmp_path):
    p=tmp_path/'pending.csv';frame([{}]).to_csv(p,index=False,encoding='utf-8-sig')
    job=store.import_csv(p,completion_status='awaiting_analysis')
    assert store.get_upload(job['job_id'])['status']=='awaiting_analysis'
