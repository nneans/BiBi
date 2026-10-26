"""Opinion-centred relational graph on the review DB engine (additive tables; SQLite or PostgreSQL).

Layers never mix:
- opinions:            human gold and AI silver opinions, bound to (review_id, text_hash)
- opinion_predictions: own-model opinions, bound to (text_hash, model_version) so identical texts are
                       computed once and every review with that text shares them (same rule as analysis_labels)
Both share opinion_evidence (Unicode [start,end) spans) and opinion_reasons (junction to ontology reasons).
Reasons outside the ontology go to reason_candidates (pending review) and are never auto-registered.
"""
import json
import uuid

from sqlalchemy import (MetaData,Table,Column,String,Text,Integer,Float,Boolean,Index,select,delete,update,exists,and_,
                        func,literal_column)

from ..backbone import now
from ..ontology import load_ontology,SPECIAL_ASPECTS

TIERS=('gold','silver','prediction')


class OntologyVersionConflict(RuntimeError):
    pass


class OpinionGraphStore:
    def __init__(self,store,ontology=None,register=True):
        self.store=store;self.engine=store.engine;self.ontology=ontology or load_ontology();m=self.meta=MetaData()
        S=lambda n=200:String(n)
        self.versions=Table('ontology_versions',m,Column('version',S(),primary_key=True),Column('sha256',S(64),primary_key=True),
            Column('base_taxonomy_version',S()),Column('frozen',Boolean),Column('status',Text),Column('registered_at',S(50)),Column('content_json',Text))
        self.aspects=Table('ontology_aspects',m,Column('version',S(),primary_key=True),Column('code',S(80),primary_key=True),
            Column('parent_code',S(80)),Column('name',Text),Column('kind',S(20)),Column('status',S(40)),Column('aggregation',S(20)),
            Column('model_output',Boolean))
        self.reasons=Table('ontology_reasons',m,Column('version',S(),primary_key=True),Column('code',S(80),primary_key=True),
            Column('name',Text),Column('primary_aspect',S(80)),Column('status',S(40)),Column('display_positive',Text),
            Column('display_negative',Text))
        self.reason_aspects=Table('ontology_reason_aspects',m,Column('version',S(),primary_key=True),Column('reason_code',S(80),primary_key=True),
            Column('aspect_code',S(80),primary_key=True),Column('role',S(20),nullable=False))
        self.reason_relations=Table('ontology_reason_relations',m,Column('version',S(),primary_key=True),
            Column('reason_code',S(80),primary_key=True),Column('related_code',S(80),primary_key=True),Column('relation',S(40)))
        self.models=Table('model_versions',m,Column('model_version',S(),primary_key=True),Column('model_kind',S(80)),
            Column('artifact_dir',Text),Column('validated',Boolean),Column('test_only',Boolean),Column('registered_at',S(50)),
            Column('details_json',Text))
        self.opinions=Table('opinions',m,Column('opinion_id',S(),primary_key=True),Column('review_id',S(100),nullable=False,index=True),
            Column('product_id',S(100),index=True),Column('text_hash',S(64),nullable=False,index=True),
            Column('ontology_version',S(),nullable=False),Column('ontology_sha256',S(64)),
            Column('label_tier',S(20),nullable=False),Column('source_name',S(),nullable=False),Column('source_ref',Text),
            Column('target_scope',S(30),nullable=False),Column('statement_type',S(30),nullable=False),
            Column('aspect',S(80),nullable=False),Column('sentiment',S(30),nullable=False),Column('reason_text',Text),
            Column('uncertain',Boolean),Column('status',S(30),nullable=False),Column('issues_json',Text),Column('note',Text),
            Column('ingest_run_id',S(40)),Column('created_at',S(50)),
            Index('ix_opinions_product_tier','product_id','label_tier','aspect','sentiment'))
        self.predictions=Table('opinion_predictions',m,Column('opinion_id',S(),primary_key=True),
            Column('text_hash',S(64),nullable=False),Column('model_version',S(),nullable=False),
            Column('ontology_version',S(),nullable=False),Column('aspect',S(80),nullable=False),Column('sentiment',S(30),nullable=False),
            Column('target_scope',S(30),nullable=False),Column('score',Float),Column('threshold',Float),Column('status',S(30),nullable=False),
            Column('created_at',S(50)),
            Index('ix_pred_version_hash','model_version','text_hash'),Index('ix_pred_version_aspect','model_version','aspect','sentiment'))
        self.evidence=Table('opinion_evidence',m,Column('evidence_id',S(),primary_key=True),Column('opinion_id',S(),nullable=False,index=True),
            Column('text_hash',S(64),nullable=False),Column('start_offset',Integer,nullable=False),Column('end_offset',Integer,nullable=False),
            Column('evidence_text',Text),Column('sentence_ids_json',Text),Column('cross_sentence',Boolean),Column('evidence_kind',S(20)),
            Column('verified',Boolean))
        self.links=Table('opinion_reasons',m,Column('opinion_id',S(),primary_key=True),Column('reason_code',S(80),primary_key=True),
            Column('ontology_version',S()),Column('role',S(20),nullable=False),Column('link_status',S(20),nullable=False),
            Column('method',S(30),nullable=False),Index('ix_links_reason','reason_code','link_status'))
        self.candidates=Table('reason_candidates',m,Column('candidate_id',S(40),primary_key=True),Column('opinion_id',S(),index=True),
            Column('review_id',S(100)),Column('product_id',S(100),index=True),Column('aspect',S(80)),Column('candidate_text',Text),
            Column('source_name',S()),Column('label_tier',S(20)),Column('status',S(30),nullable=False),Column('created_at',S(50)),
            Column('resolved_by',Text),Column('resolved_at',S(50)))
        self.runs=Table('opinion_ingest_runs',m,Column('run_id',S(40),primary_key=True),Column('kind',S(40)),Column('source_name',S()),
            Column('details_json',Text),Column('created_at',S(50)))
        m.create_all(self.engine)
        if register:self.register_ontology()

    # --- ontology & models ------------------------------------------------------------------
    def register_ontology(self):
        o=self.ontology
        with self.engine.begin() as c:
            rows=c.execute(select(self.versions.c.sha256,self.versions.c.frozen).where(self.versions.c.version==o.version)).all()
            if any(r[1] for r in rows) and o.sha256 not in {r[0] for r in rows}:
                raise OntologyVersionConflict(f'{o.version} is frozen in the DB with different content; bump ontology_version')
            if o.sha256 in {r[0] for r in rows}:return False
            c.execute(self.versions.insert().values(version=o.version,sha256=o.sha256,base_taxonomy_version=o.base_version,
                frozen=bool(o.raw.get('frozen')),status=o.raw.get('status'),registered_at=now(),content_json=json.dumps(o.tree(),ensure_ascii=False)))
            for t in (self.aspects,self.reasons,self.reason_aspects,self.reason_relations):c.execute(delete(t).where(t.c.version==o.version))
            c.execute(self.aspects.insert(),[{'version':o.version,'code':a['code'],'parent_code':a['parent'],'name':a['name'],'kind':a['kind'],
                'status':a.get('status'),'aggregation':a['aggregation'],'model_output':bool(a.get('model_output'))} for a in o.aspects.values()])
            c.execute(self.reasons.insert(),[{'version':o.version,'code':r['code'],'name':r['name'],'primary_aspect':r['primary_aspect'],
                'status':r['status'],'display_positive':r['display']['positive'],'display_negative':r['display']['negative']} for r in o.reasons.values()])
            c.execute(self.reason_aspects.insert(),[{'version':o.version,'reason_code':r['code'],'aspect_code':a,'role':role}
                for r in o.reasons.values() for a,role in [(r['primary_aspect'],'primary')]+[(x,'related') for x in r['related_aspects']]])
            rel={(r['code'],x) for r in o.reasons.values() for x in r['related_reasons']}
            c.execute(self.reason_relations.insert(),[{'version':o.version,'reason_code':a,'related_code':b,
                'relation':'related_to' if (b,a) in rel else 'related_to_oneway'} for a,b in sorted(rel)])
        return True

    def register_model(self,model_version,model_kind=None,artifact_dir=None,validated=False,test_only=False,details=None):
        with self.engine.begin() as c:
            self.store.put(c,self.models,[{'model_version':model_version,'model_kind':model_kind,'artifact_dir':str(artifact_dir) if artifact_dir else None,
                'validated':bool(validated),'test_only':bool(test_only),'registered_at':now(),
                'details_json':json.dumps(details or {},ensure_ascii=False)}],['model_version'])

    # --- writes -------------------------------------------------------------------------------
    def save_predictions(self,connection,model_version,items):
        """items: [(text_hash, analyzer result)] in the caller's transaction (same one as analysis_labels)."""
        from ..opinion_extraction.from_predictions import opinions_from_result
        hashes=[h for h,_ in items]
        if not hashes:return 0
        old=select(self.predictions.c.opinion_id).where(self.predictions.c.model_version==model_version,self.predictions.c.text_hash.in_(hashes))
        connection.execute(delete(self.evidence).where(self.evidence.c.opinion_id.in_(old)))
        connection.execute(delete(self.links).where(self.links.c.opinion_id.in_(old)))
        connection.execute(delete(self.predictions).where(self.predictions.c.model_version==model_version,self.predictions.c.text_hash.in_(hashes)))
        rows,ev,links=[],[],[]
        for h,result in items:
            for o in opinions_from_result(h,result,self.ontology):
                rows.append({k:o[k] for k in ('opinion_id','text_hash','model_version','aspect','sentiment','target_scope','score','threshold','status')}
                            |{'ontology_version':self.ontology.version,'created_at':now()})
                ev+=[{**e,'opinion_id':o['opinion_id'],'text_hash':h} for e in o['evidence']]
                links+=[{**r,'opinion_id':o['opinion_id'],'ontology_version':self.ontology.version} for r in o['reasons']]
        if rows:connection.execute(self.predictions.insert(),rows)
        if ev:connection.execute(self.evidence.insert(),ev)
        if links:connection.execute(self.links.insert(),links)
        return len(rows)

    def add_annotations(self,opinions,kind,source_name,details=None):
        """Insert normalized gold/silver opinions. Idempotent per opinion_id."""
        run_id=uuid.uuid4().hex;ids=[o['opinion_id'] for o in opinions]
        if any(o['label_tier'] not in ('gold','silver') for o in opinions):raise ValueError('annotations must be gold or silver')
        with self.engine.begin() as c:
            for a in range(0,len(ids),500):
                chunk=ids[a:a+500]
                for t in (self.evidence,self.links,self.candidates):c.execute(delete(t).where(t.c.opinion_id.in_(chunk)))
                c.execute(delete(self.opinions).where(self.opinions.c.opinion_id.in_(chunk)))
            base=('opinion_id','review_id','product_id','text_hash','label_tier','source_name','source_ref','target_scope','statement_type',
                  'aspect','sentiment','reason_text','uncertain','status','note')
            rows=[{k:o.get(k) for k in base}|{'issues_json':json.dumps(o.get('issues',[])),'ontology_version':self.ontology.version,
                  'ontology_sha256':self.ontology.sha256,'ingest_run_id':run_id,'created_at':now()} for o in opinions]
            if rows:c.execute(self.opinions.insert(),rows)
            ev=[{**e,'opinion_id':o['opinion_id'],'text_hash':o['text_hash']} for o in opinions for e in o['evidence']]
            if ev:c.execute(self.evidence.insert(),ev)
            links=[{**r,'opinion_id':o['opinion_id'],'ontology_version':self.ontology.version} for o in opinions for r in o['reasons']]
            if links:c.execute(self.links.insert(),links)
            cand=[{'candidate_id':uuid.uuid4().hex,'opinion_id':o['opinion_id'],'review_id':o['review_id'],'product_id':o['product_id'],
                   'aspect':x.get('aspect'),'candidate_text':x['text'],'source_name':source_name,'label_tier':o['label_tier'],
                   'status':'pending_review','created_at':now()} for o in opinions for x in o.get('candidates',[])]
            if cand:c.execute(self.candidates.insert(),cand)
            c.execute(self.runs.insert().values(run_id=run_id,kind=kind,source_name=source_name,created_at=now(),
                details_json=json.dumps({'opinions':len(rows),'evidence':len(ev),'reason_links':len(links),'reason_candidates':len(cand),
                                         **(details or {})},ensure_ascii=False)))
        return {'run_id':run_id,'opinions':len(rows),'evidence':len(ev),'reason_links':len(links),'reason_candidates':len(cand)}

    def purge_orphans(self):
        """Predictions without a normal review are deleted (blind/edited text). Annotation opinions are kept as
        assets but lose their text when the review is blinded, and are marked stale when the review text changed."""
        r=self.store.review;p=self.predictions;o=self.opinions;counts={}
        with self.engine.begin() as c:
            alive=exists().where(r.c.text_hash==p.c.text_hash,r.c.display_status=='정상')
            dead=select(p.c.opinion_id).where(~alive)
            counts['prediction_evidence']=c.execute(delete(self.evidence).where(self.evidence.c.opinion_id.in_(dead))).rowcount
            counts['prediction_reasons']=c.execute(delete(self.links).where(self.links.c.opinion_id.in_(dead))).rowcount
            counts['predictions']=c.execute(delete(p).where(~alive)).rowcount
            blinded=select(r.c.review_id).where(r.c.display_status!='정상')
            ids=select(o.c.opinion_id).where(o.c.review_id.in_(blinded))
            counts['annotation_evidence_scrubbed']=c.execute(update(self.evidence).where(self.evidence.c.opinion_id.in_(ids),
                self.evidence.c.evidence_text.is_not(None)).values(evidence_text=None)).rowcount
            counts['annotations_blinded']=c.execute(update(o).where(o.c.review_id.in_(blinded),o.c.status!='source_blinded')
                .values(status='source_blinded')).rowcount
            changed=select(r.c.review_id).where(r.c.review_id==o.c.review_id,r.c.display_status=='정상',r.c.text_hash!=o.c.text_hash)
            counts['annotations_stale']=c.execute(update(o).where(changed.exists(),o.c.status.notin_(['stale_text','source_blinded']))
                .values(status='stale_text')).rowcount
        return counts
