"""Additive analysis tables on the existing review DB engine (ReviewStore is reused unchanged).

Results are keyed by (text_hash, model_version): identical texts share one computation, a text
edit produces a new hash (old results stop joining), rating-only edits keep the hash, and
model versions never mix.
"""
import json
import uuid
from datetime import datetime,timezone

from sqlalchemy import MetaData,Table,Column,String,Text,Integer,Float,Boolean,Index,select,delete,update,func,and_,exists


def now():
    return datetime.now(timezone.utc).isoformat()


class AnalysisStore:
    def __init__(self,store):
        self.store=store;self.engine=store.engine;m=self.meta=MetaData()
        self.status=Table('analysis_status',m,Column('text_hash',String(64),primary_key=True),Column('model_version',String(200),primary_key=True),
            Column('status',String(30),nullable=False),Column('label_count',Integer),Column('created_at',String(50)),
            Index('ix_status_version',  'model_version','status'))
        self.labels=Table('analysis_labels',m,Column('text_hash',String(64),primary_key=True),Column('model_version',String(200),primary_key=True),
            Column('aspect',String(60),primary_key=True),Column('sentiment',String(20),primary_key=True),Column('score',Float),
            Column('threshold',Float),Column('evidence_text',Text),Column('evidence_start',Integer),Column('evidence_end',Integer),
            Column('reason_candidates_json',Text),Index('ix_labels_version_aspect','model_version','aspect','sentiment'))
        self.failures=Table('analysis_failures',m,Column('text_hash',String(64),primary_key=True),Column('model_version',String(200),primary_key=True),
            Column('attempts',Integer,nullable=False),Column('error',Text),Column('last_attempt_at',String(50)))
        self.queue=Table('review_queue',m,Column('text_hash',String(64),primary_key=True),Column('model_version',String(200),primary_key=True),
            Column('reason',Text),Column('status',String(20),nullable=False),Column('created_at',String(50)),Column('resolved_at',String(50)))
        self.corrections=Table('human_corrections',m,Column('correction_id',String(40),primary_key=True),Column('review_id',String(100),index=True),
            Column('text_hash',String(64)),Column('annotator',Text,nullable=False),Column('opinions_json',Text,nullable=False),
            Column('complete',Boolean,nullable=False),Column('model_version_seen',String(200)),Column('created_at',String(50)))
        self.upload_analysis=Table('upload_analysis',m,Column('upload_id',String(40),primary_key=True),Column('model_version',String(200)),
            Column('status',String(40),nullable=False),Column('details_json',Text),Column('started_at',String(50)),Column('finished_at',String(50)))
        m.create_all(self.engine)

    def save_batch(self,connection,model_version,items):
        """items: [(text_hash, result)] for one transaction: cache JSON, status, labels, queue."""
        s=self.store
        rows_cache=[{'text_hash':h,'model_version':model_version,'result_json':json.dumps(r,ensure_ascii=False),'created_at':now()} for h,r in items]
        s.put(connection,s.cache,rows_cache,['text_hash','model_version'])
        s.put(connection,self.status,[{'text_hash':h,'model_version':model_version,'status':r['status'],'label_count':len(r['labels']),
                                       'created_at':now()} for h,r in items],['text_hash','model_version'])
        hashes=[h for h,_ in items]
        connection.execute(delete(self.labels).where(self.labels.c.model_version==model_version,self.labels.c.text_hash.in_(hashes)))
        labels=[{'text_hash':h,'model_version':model_version,'aspect':l['aspect'],'sentiment':l['sentiment'],'score':l['score'],
                 'threshold':l['threshold'],'evidence_text':l['evidence_sentence'],'evidence_start':l['evidence_start'],
                 'evidence_end':l['evidence_end'],'reason_candidates_json':json.dumps(l['reason_candidates'])}
                for h,r in items for l in r['labels']]
        if labels:connection.execute(self.labels.insert(),labels)
        queue=[{'text_hash':h,'model_version':model_version,'status':'open','created_at':now(),
                'reason':json.dumps({'uncertain_labels':r['uncertain_labels'],'flags':r['flags']},ensure_ascii=False)}
               for h,r in items if r['status']=='review_required']
        s.put(connection,self.queue,queue,['text_hash','model_version'])
        connection.execute(delete(self.failures).where(self.failures.c.model_version==model_version,self.failures.c.text_hash.in_(hashes)))

    def record_failure(self,text_hash,model_version,error):
        with self.engine.begin() as c:
            old=c.execute(select(self.failures.c.attempts).where(self.failures.c.text_hash==text_hash,
                          self.failures.c.model_version==model_version)).scalar_one_or_none()
            self.store.put(c,self.failures,[{'text_hash':text_hash,'model_version':model_version,'attempts':(old or 0)+1,
                                             'error':str(error)[:500],'last_attempt_at':now()}],['text_hash','model_version'])

    def purge_orphans(self):
        """Remove analysis rows whose text no longer belongs to any normal review (e.g. blinded)."""
        r=self.store.review
        alive=exists().where(r.c.text_hash==self.status.c.text_hash,r.c.display_status=='정상')
        counts={}
        with self.engine.begin() as c:
            for name,table in [('labels',self.labels),('status',self.status),('queue',self.queue),('failures',self.failures),('cache',self.store.cache)]:
                alive=exists().where(r.c.text_hash==table.c.text_hash,r.c.display_status=='정상')
                counts[name]=c.execute(delete(table).where(~alive)).rowcount
        return counts

    def add_correction(self,review_id,annotator,opinions,complete,model_version=None):
        if not str(annotator).strip():raise ValueError('annotator is required for a human correction')
        with self.engine.begin() as c:
            row=c.execute(select(self.store.review.c.text_hash,self.store.review.c.display_status).where(self.store.review.c.review_id==review_id)).first()
            if row is None or row[1]!='정상':raise LookupError('review not found or not normal')
            cid=uuid.uuid4().hex
            c.execute(self.corrections.insert().values(correction_id=cid,review_id=review_id,text_hash=row[0],annotator=annotator.strip(),
                opinions_json=json.dumps(opinions,ensure_ascii=False),complete=bool(complete),model_version_seen=model_version,created_at=now()))
            c.execute(update(self.queue).where(self.queue.c.text_hash==row[0]).values(status='resolved',resolved_at=now()))
        return cid
