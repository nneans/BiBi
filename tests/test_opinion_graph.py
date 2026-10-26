"""Ontology, opinion instances, opinion graph storage/queries and the offline CSV -> opinion path."""
import json

import pandas as pd
import pytest
from sqlalchemy import text

from bibi_reviews.backbone import ReviewStore,normalize_text,hash_text
from bibi_reviews.ontology import Ontology,load_ontology,DEFAULT_ONTOLOGY
from bibi_reviews.opinion_graph.store import OpinionGraphStore,OntologyVersionConflict
from bibi_reviews.opinion_graph.queries import OpinionAnalytics
from bibi_reviews.opinion_extraction.from_predictions import opinions_from_result
from bibi_reviews.classifier import registry

from test_review_pipeline import CONFIG,TAX,TEMPLATES,artifact,csv_bytes,wait,no_network  # noqa: F401  (shared fixtures)

O=load_ontology()
EXAMPLE='치즈가 고소하고 맛있는데, 생각보다 잘 안 늘어나고 튀김옷도 너무 두꺼워요.'


def frame(rows):
    return pd.DataFrame(rows,columns=['상품번호','상품명','리뷰글번호','리뷰상세내용','구매자평점','리뷰등록일','최종수정일','전시상태'])


def span(sub,text=EXAMPLE):
    a=text.index(sub);return a,a+len(sub)


@pytest.fixture
def db(tmp_path):
    store=ReviewStore('sqlite:///'+str(tmp_path/'g.db'))
    store.import_frame(frame([('296889036','모짜렐라 치즈스틱','R1',EXAMPLE,3,'2026-05-01 10:00:00','','정상'),
                              ('296889036','모짜렐라 치즈스틱','R2','배송이 빨라서 좋아요. 치즈가 잘 안 늘어나요',4,'2026-05-02 10:00:00','','정상'),
                              ('7826600725','손질 명태포','R3','고소하고 맛있어요 가시가 많아요',2,'2026-05-03 10:00:00','','정상')]))
    return store,tmp_path


# ---------- ontology ----------

def test_ontology_is_consistent_with_base_taxonomy_and_inference_cues():
    from bibi_reviews.classifier.inference import REASON_CUES
    assert O.check()==[] and O.base_version==TAX['version']
    base_aspects={a['code'] for a in TAX['aspects']};base_reasons={r['code'] for r in TAX['reason_candidates']}
    assert base_aspects<=set(O.aspects) and base_reasons<=set(O.reasons)
    for r in TAX['reason_candidates']:assert O.reasons[r['code']]['primary_aspect']==r['aspect']   # single source of definitions
    assert all(O.reason_cues()[k]==v for k,v in REASON_CUES.items())                         # classifier uses ontology cues
    assert {a['code'] for a in O.leaf_aspects() if a.get('model_output')}=={a['code'] for a in TAX['aspects'] if a.get('model_output')}
    for a in O.leaf_aspects():assert O.aspects[a['parent']]['kind']=='group'                    # two-level tree, no orphans
    # polarity-neutral reasons: one concept, two display names
    assert O.display('stretchability','negative')=='치즈 늘어남 부족' and O.display('stretchability','positive')!=O.display('stretchability','negative')


def test_reason_matching_respects_aspect():
    taste=O.match_reasons('치즈가 고소하고 맛있어요','taste')
    assert [r['reason_code'] for r in taste]==['cheese_flavor'] and taste[0]['role']=='primary'
    assert O.match_reasons('고소하고 맛있어요','taste')==[]                        # no product groups: the cue needs 치즈 itself
    cross=O.match_reasons('잘 안 늘어나요','quality')
    assert cross==[{'reason_code':'stretchability','role':'secondary'}]          # another aspect's reason never becomes primary


def test_opinion_validation_rules():
    good={'target_scope':'current_product','statement_type':'evaluation','aspect':'texture','sentiment':'negative',
          'evidence':[{'start':span('잘 안 늘어나고')[0],'end':span('잘 안 늘어나고')[1],'text':'잘 안 늘어나고'}],
          'reasons':[{'reason_code':'stretchability','role':'primary'}]}
    assert O.validate_opinion(good,EXAMPLE)==[]
    assert 'error:evidence_text_mismatch' in O.validate_opinion({**good,'evidence':[{'start':0,'end':3,'text':'없는말'}]},EXAMPLE)
    assert 'error:sentiment_statement_conflict' in O.validate_opinion({**good,'statement_type':'fact'},EXAMPLE)
    assert 'warn:reason_aspect_mismatch' in O.validate_opinion({**good,'aspect':'delivery'},EXAMPLE)
    assert 'error:unknown_reason_code' in O.validate_opinion({**good,'reasons':[{'reason_code':'made_up','role':'primary'}]},EXAMPLE)
    assert 'error:invalid_aspect' in O.validate_opinion({**good,'aspect':'grp_texture'},EXAMPLE)  # groups are for aggregation only
    t='다른 제품보다 맛있어요';other={**good,'aspect':'taste','sentiment':'positive','reasons':[],'evidence':[{'start':0,'end':len(t),'text':t}]}
    assert 'warn:possible_misattribution' in O.validate_opinion(other,t)


def test_frozen_ontology_version_cannot_change_content(db,tmp_path):
    store,_=db;OpinionGraphStore(store,O)
    with store.engine.begin() as c:c.execute(text('update ontology_versions set frozen=1'))
    changed=json.loads(DEFAULT_ONTOLOGY.read_text(encoding='utf-8'));changed['reason_semantics']['odor']['display']['negative']='비린내'
    path=tmp_path/'ontology.json';path.write_text(json.dumps(changed,ensure_ascii=False),encoding='utf-8')
    with pytest.raises(OntologyVersionConflict):OpinionGraphStore(store,Ontology(path))


# ---------- prediction layer (approach A) ----------

def fake_result(text_,labels,version='fake-v1',status='classified'):
    return {'model_version':version,'status':status,'uncertain_labels':[],
            'labels':[{'aspect':a,'sentiment':s,'score':.9,'threshold':.5,'evidence_sentence':text_[x:y],'evidence_start':x,'evidence_end':y}
                      for a,s,x,y in labels]}


def test_prediction_result_becomes_opinion_instances_with_candidate_reasons():
    t='배송이 빨라서 좋아요. 치즈가 잘 안 늘어나요';cut=t.index('치즈')
    ops=opinions_from_result('h',fake_result(t,[('delivery','positive',0,cut-1),('quality','negative',cut,len(t))]),O)
    assert [o['aspect'] for o in ops]==['delivery','quality'] and all(o['target_scope']=='current_product' for o in ops)
    assert ops[0]['reasons']==[{'reason_code':'delivery_speed','role':'primary','link_status':'candidate','method':'lexicon'}]
    assert {r['reason_code'] for r in ops[1]['reasons']}>={'stretchability'} and all(r['link_status']=='candidate' for r in ops[1]['reasons'])
    assert ops[0]['evidence'][0]['evidence_kind']=='sentence' and not ops[0]['evidence'][0]['verified']


def test_service_csv_upload_builds_prediction_opinions_offline(artifact,tmp_path,no_network,review_db_url):  # SQLite and PostgreSQL
    from fastapi.testclient import TestClient
    from bibi_reviews.service.app import create_service_app
    reg=tmp_path/'registry.json';registry.activate(artifact,reg,allow_unvalidated_for_testing=True)
    store=ReviewStore(review_db_url)
    app=create_service_app(store=store,password='test-passphrase-123',upload_dir=tmp_path/'up',config=CONFIG,registry_path=reg,start_worker=False)
    auth=('bibi','test-passphrase-123');worker=app.state.worker
    base=[('p1','새우',f'r{i}',t,5 if i%2 else 2,f'2026-0{1+i%3}-1{i%9} 10:00:00','','정상') for i,(t,_) in enumerate(TEMPLATES)]
    base.append(('p1','새우','dup',TEMPLATES[0][0],5,'2026-03-01 10:00:00','','정상'))
    with TestClient(app) as c:
        assert c.get('/api/v2/products/p1/opinion-tree',auth=auth).status_code==404                    # no product yet
        wait(c,auth,c.post('/api/uploads',auth=auth,files={'file':('a.csv',csv_bytes(base),'text/csv')}).json()['job_id']);worker.run_once()
        details=json.loads(c.get('/api/v2/processing/status',auth=auth).json()['uploads'][0]['analysis_details'])
        assert details['opinion_predictions']>0
        tree=c.get('/api/v2/products/p1/opinion-tree',auth=auth).json()
        assert tree['meta']['label_tier']=='prediction' and tree['meta']['human_validated'] is False
        taste=next(a for g in tree['groups'] for a in g['aspects'] if a['code']=='taste')
        assert taste['sentiments']['positive']['reviews']>=2                                            # 'dup' shares r0's opinions
        assert all(not r['cross_aspect'] for g in tree['groups'] for a in g['aspects'] for s in a['sentiments'].values() for r in s['reasons'])
        assert all(r['cross_aspect'] for g in tree['groups'] for a in g['aspects'] for s in a['sentiments'].values() for r in s['cross_aspect_reason_hints'])
        delivery=c.get('/api/v2/products/p1/opinions',auth=auth,params={'aspect':'delivery','sentiment':'negative'}).json()['opinions']
        assert delivery and all(o['review_text'][o['evidence'][0]['start_offset']:o['evidence'][0]['end_offset']]==o['evidence'][0]['evidence_text']
                                for o in delivery)
        assert c.get('/api/v2/products/p1/opinion-tree',auth=auth,params={'tier':'gold'}).json()['counts']['opinions']==0
        assert c.get('/api/v2/ontology',auth=auth).json()['ontology_version']==O.version
        assert c.get('/api/v2/categories/seafood_raw/reasons',auth=auth).status_code==404           # product groups removed
        assert c.get('/api/v2/reasons/delivery_speed',auth=auth).status_code==200
        # unchanged re-upload computes nothing; rating-only change reuses; text edit replaces opinions; blind removes them
        wait(c,auth,c.post('/api/uploads',auth=auth,files={'file':('a.csv',csv_bytes(base),'text/csv')}).json()['job_id']);worker.run_once()
        assert json.loads(c.get('/api/v2/processing/status',auth=auth).json()['uploads'][0]['analysis_details']).get('opinion_predictions',0)==0
        changed=list(map(list,base));changed[1][4]=1;changed[1][6]='2026-04-01 10:00:00'
        changed[2][3]='배송이 빨라서 좋아요 근데 양이 너무 적어요';changed[2][6]='2026-04-01 10:00:00'
        changed[3][7]='블라인드';changed[3][6]='2026-04-01 10:00:00'
        wait(c,auth,c.post('/api/uploads',auth=auth,files={'file':('b.csv',csv_bytes(changed),'text/csv')}).json()['job_id']);worker.run_once()
        details=json.loads(c.get('/api/v2/processing/status',auth=auth).json()['uploads'][0]['analysis_details'])
        assert details['computed_texts']==1 and details['opinion_predictions']>=1                         # only the edited text
        assert details['purged_orphans']['opinion_graph']['predictions']>=2                                # old text + blinded text
        edited=hash_text(normalize_text(changed[2][3]));old=hash_text(normalize_text(TEMPLATES[2][0]))
        with store.engine.connect() as conn:
            assert conn.execute(text('select count(*) from opinion_predictions where text_hash=:h'),{'h':edited}).scalar()>=1
            assert conn.execute(text('select count(*) from opinion_predictions where text_hash=:h'),{'h':old}).scalar()==0
        blind_hash=hash_text(normalize_text(TEMPLATES[3][0]))
        with store.engine.connect() as conn:
            assert conn.execute(text('select count(*) from opinion_predictions where text_hash=:h'),{'h':blind_hash}).scalar()==0
            assert conn.execute(text('select count(*) from model_versions')).scalar()==1
        status=c.get('/api/v2/opinion-graph/status',auth=auth).json();assert status['prediction_opinions']
