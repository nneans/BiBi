"""Transactional review storage and idempotent CSV ingestion."""
import argparse
import hashlib
import html
import json
import re
import unicodedata
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from .review_pipeline.common import plain_row
from sqlalchemy import (create_engine, event, MetaData, Table, Column, String, Text,
                        Integer, Boolean, Float, ForeignKey, select, func, update, delete)


def now():
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value):
    value=unicodedata.normalize('NFC',html.unescape(str(value)))
    value=re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]','',value)
    return re.sub(r'\s+',' ',value).strip()


def hash_text(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


class ReviewStore:
    def __init__(self,url=None):
        if url is None:  # the configured review DB (PostgreSQL by default; BIBI_DATABASE_URL overrides)
            from .review_pipeline.common import load_config
            url=load_config()['database']
        if url.startswith('sqlite:///'):
            Path(url.removeprefix('sqlite:///')).parent.mkdir(parents=True,exist_ok=True)
        self.engine=create_engine(url,connect_args={'timeout':60,'check_same_thread':False} if url.startswith('sqlite') else {})
        if url.startswith('sqlite'):
            @event.listens_for(self.engine,'connect')
            def pragmas(connection,_):
                connection.execute('PRAGMA journal_mode=WAL')
                connection.execute('PRAGMA foreign_keys=ON')
                connection.execute('PRAGMA busy_timeout=60000')
        self.meta=MetaData()
        self.product=Table('products',self.meta,
            Column('product_id',String(100),primary_key=True),Column('product_name',Text,nullable=False))
        self.review=Table('reviews',self.meta,
            Column('review_id',String(100),primary_key=True),
            Column('product_id',String(100),ForeignKey('products.product_id'),nullable=False,index=True),
            Column('product_name_original',Text,nullable=False),Column('review_type',String(30),nullable=False),
            Column('rating',Integer,nullable=False),Column('created_at',String(50),nullable=False,index=True),
            Column('source_updated_at',String(50)),Column('effective_at',String(50),nullable=False),
            Column('related_review_id',String(100)),Column('display_status',String(30),nullable=False,index=True),
            Column('review_text_raw',Text),Column('review_text',Text),Column('text_hash',String(64),index=True),
            Column('row_hash',String(64),nullable=False),Column('last_upload_id',String(40)),
            Column('stored_at',String(50),nullable=False))
        self.upload=Table('upload_jobs',self.meta,
            Column('job_id',String(40),primary_key=True),Column('filename',Text),Column('file_sha256',String(64)),
            Column('status',String(30),nullable=False),Column('created_at',String(50)),Column('finished_at',String(50)),
            Column('counts_json',Text),Column('error',Text))
        self.history=Table('review_history',self.meta,
            Column('history_id',Integer,primary_key=True,autoincrement=True),Column('review_id',String(100),index=True),
            Column('upload_id',String(40)),Column('previous_json',Text),Column('changed_at',String(50)))
        self.membership=Table('upload_reviews',self.meta,
            Column('upload_id',String(40),ForeignKey('upload_jobs.job_id'),primary_key=True),
            Column('review_id',String(100),primary_key=True,index=True))
        self.cache=Table('analysis_cache',self.meta,
            Column('text_hash',String(64),primary_key=True),Column('model_version',String(200),primary_key=True),
            Column('result_json',Text,nullable=False),Column('created_at',String(50)))
        self.runs=Table('analysis_runs',self.meta,
            Column('run_id',String(40),primary_key=True),Column('model_version',String(200)),
            Column('status',String(30)),Column('details_json',Text),Column('created_at',String(50)))
        self.meta.create_all(self.engine)

    def put(self,connection,table,rows,keys):
        if not rows:return
        if self.engine.dialect.name=='sqlite':
            from sqlalchemy.dialects.sqlite import insert
        elif self.engine.dialect.name=='postgresql':
            from sqlalchemy.dialects.postgresql import insert
        else: raise ValueError('Only SQLite and PostgreSQL supported')
        stmt=insert(table)
        changes={c.name:getattr(stmt.excluded,c.name) for c in table.columns if c.name not in keys}
        stmt=(stmt.on_conflict_do_update(index_elements=keys,set_=changes) if changes
              else stmt.on_conflict_do_nothing(index_elements=keys))
        connection.execute(stmt,rows)

    @staticmethod
    def prepare(frame):
        aliases={'상품번호':'product_id','상품명':'product_name','리뷰글번호':'review_id',
            '리뷰상세내용':'review_text_raw','구매자평점':'rating','리뷰등록일':'created_at',
            '최종수정일':'source_updated_at','리뷰구분':'review_type','관련리뷰글번호':'related_review_id',
            '전시상태':'display_status'}
        d=frame.rename(columns=aliases).copy()
        required=['product_id','product_name','review_id','review_text_raw','rating','created_at','display_status']
        if not set(required)<=set(d):raise ValueError('Missing required columns')
        if d[required].isna().any().any():raise ValueError('Missing required values')
        for key in ['product_id','product_name','review_id']:
            d[key]=d[key].astype(str).str.strip()
            if d[key].eq('').any():raise ValueError(f'Empty {key}')
        if d.review_id.duplicated().any():raise ValueError('Duplicate review IDs in CSV')
        ratings=pd.to_numeric(d.rating,errors='coerce')
        if not ratings.isin([1,2,3,4,5]).all():raise ValueError('Ratings must be integers 1..5')
        if not d.display_status.isin(['정상','블라인드']).all():raise ValueError('Unknown display status')
        d['rating']=ratings.astype(int)
        for key,default in [('source_updated_at',''),('related_review_id',''),('review_type','일반')]:
            if key not in d:d[key]=default
            d[key]=d[key].fillna('').astype(str)
        for key in ['created_at','source_updated_at']:
            values=pd.to_datetime(d[key].replace('',None),format='mixed',errors='coerce')
            invalid=values.isna() & (d[key].ne('') if key=='source_updated_at' else True)
            if invalid.any():raise ValueError(f'Invalid {key}')
            try:
                if values.dt.tz is None:values=values.dt.tz_localize('Asia/Seoul')
                else:values=values.dt.tz_convert('Asia/Seoul')
            except AttributeError:raise ValueError('Mixed/invalid timezone values')
            d[key]=values.map(lambda v:None if pd.isna(v) else v.isoformat())
        records=[]
        for row in d.to_dict('records'):
            raw=str(row['review_text_raw']) if row['display_status']=='정상' else None
            text=normalize_text(raw) if raw is not None else None
            r={k:row[k] for k in ['review_id','product_id','review_type','rating','created_at','source_updated_at','related_review_id','display_status']}
            r.update(product_name_original=row['product_name'],review_text_raw=raw,review_text=text,
                     text_hash=hash_text(text) if text is not None else None,
                     effective_at=row['source_updated_at'] or row['created_at'])
            r['row_hash']=hash_text(json.dumps(r,sort_keys=True,ensure_ascii=False))
            records.append(r)
        return records

    def create_upload(self,filename,file_sha256=None,status='queued'):
        job_id=uuid.uuid4().hex
        with self.engine.begin() as c:
            c.execute(self.upload.insert().values(job_id=job_id,filename=Path(filename).name,
                file_sha256=file_sha256,status=status,created_at=now()))
        return job_id

    def import_frame(self,frame):
        records=self.prepare(frame)
        job_id=self.create_upload('in-memory.csv')
        return self._import([records],job_id)

    def import_csv(self,path,chunksize=10000,job_id=None,completion_status='completed'):
        path=Path(path)
        if job_id is None:
            h=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
            job_id=self.create_upload(path.name,h.hexdigest())
        frames=pd.read_csv(path,encoding='utf-8-sig',dtype=str,keep_default_na=False,chunksize=chunksize)
        return self._import((self.prepare(f) for f in frames),job_id,completion_status)

    def _import(self,chunks,job_id,completion_status='completed'):
        counts={k:0 for k in ['rows','new','updated','unchanged','stale','same_timestamp_conflicts','blind_rows']}
        seen=set()
        with self.engine.begin() as c:c.execute(update(self.upload).where(self.upload.c.job_id==job_id).values(status='running'))
        try:
            with self.engine.begin() as c:
                for records in chunks:
                    ids=[r['review_id'] for r in records]
                    if seen.intersection(ids):raise ValueError('Duplicate review IDs across CSV chunks')
                    seen.update(ids)
                    self.put(c,self.membership,[{'upload_id':job_id,'review_id':rid} for rid in ids],['upload_id','review_id'])
                    counts['rows']+=len(records)
                    counts['blind_rows']+=sum(r['display_status']=='블라인드' for r in records)
                    for offset in range(0,len(records),500):
                        batch=records[offset:offset+500]
                        found={r['review_id']:dict(r) for r in c.execute(select(self.review).where(self.review.c.review_id.in_([r['review_id'] for r in batch]))).mappings()}
                        changed=[];history=[];redact=[]
                        for r in batch:
                            old=found.get(r['review_id'])
                            if old and old['row_hash']==r['row_hash']:counts['unchanged']+=1;continue
                            if old and r['effective_at']<old['effective_at']:counts['stale']+=1;continue
                            if old:
                                counts['updated']+=1
                                if r['effective_at']==old['effective_at']:counts['same_timestamp_conflicts']+=1
                                history.append({'review_id':r['review_id'],'upload_id':job_id,
                                                'previous_json':json.dumps(old,ensure_ascii=False),'changed_at':now()})
                                if r['display_status']=='블라인드':redact.append((r['review_id'],old['text_hash']))
                            else:counts['new']+=1
                            changed.append(r|{'stored_at':now(),'last_upload_id':job_id})
                        if changed:
                            pids={r['product_id']:r['product_name_original'] for r in changed}
                            # Name is rebuilt at the end from current normal rows.
                            self.put(c,self.product,[{'product_id':pid,'product_name':name} for pid,name in pids.items()],['product_id'])
                            if history:c.execute(self.history.insert(),history)
                            self.put(c,self.review,changed,['review_id'])
                            for rid,old_hash in redact:
                                versions=c.execute(select(self.history.c.history_id,self.history.c.previous_json)
                                    .where(self.history.c.review_id==rid)).all()
                                for history_id,payload in versions:
                                    old_record=json.loads(payload)
                                    old_record.update(review_text=None,review_text_raw=None,text_hash=None)
                                    c.execute(update(self.history).where(self.history.c.history_id==history_id)
                                        .values(previous_json=json.dumps(old_record,ensure_ascii=False)))
                                if old_hash:
                                    references=c.execute(select(func.count()).select_from(self.review).where(
                                        self.review.c.text_hash==old_hash,self.review.c.display_status=='정상')).scalar_one()
                                    if not references:c.execute(delete(self.cache).where(self.cache.c.text_hash==old_hash))
                # A deterministic most-frequent alias for each product; preserve every source alias in reviews.
                freq=c.execute(select(self.review.c.product_id,self.review.c.product_name_original,func.count().label('n'))
                    .where(self.review.c.display_status=='정상').group_by(self.review.c.product_id,self.review.c.product_name_original)
                    .order_by(self.review.c.product_id,func.count().desc(),self.review.c.product_name_original)).all()
                chosen={}
                for pid,name,_ in freq:chosen.setdefault(pid,name)
                self.put(c,self.product,[{'product_id':pid,'product_name':name} for pid,name in chosen.items()],['product_id'])
                c.execute(update(self.upload).where(self.upload.c.job_id==job_id).values(status=completion_status,
                    finished_at=now(),counts_json=json.dumps(counts)))
        except Exception as exc:
            with self.engine.begin() as c:
                c.execute(update(self.upload).where(self.upload.c.job_id==job_id).values(status='failed',finished_at=now(),error=str(exc)[:500]))
            raise
        return {'job_id':job_id,'status':completion_status,**counts}

    def get_upload(self,job_id):
        with self.engine.connect() as c:
            r=c.execute(select(self.upload).where(self.upload.c.job_id==job_id)).mappings().first()
        if not r:return None
        result=dict(r)
        counts=json.loads(result.pop('counts_json') or '{}')
        # Stored counters must never overwrite the authoritative job state.
        return counts|result

    def summary(self):
        with self.engine.connect() as c:
            r=c.execute(select(func.count().label('review_count'),func.count(func.distinct(self.review.c.product_id)).label('product_count'),
                func.avg(self.review.c.rating).label('mean_rating')).where(self.review.c.display_status=='정상')).mappings().one()
            ratings=dict(c.execute(select(self.review.c.rating,func.count()).where(self.review.c.display_status=='정상').group_by(self.review.c.rating)).all())
        return plain_row(r)|{'rating_counts':{str(k):v for k,v in ratings.items()}}

    def products(self):
        with self.engine.connect() as c:
            rows=c.execute(select(self.product.c.product_id,self.product.c.product_name,func.count().label('review_count'),
                func.avg(self.review.c.rating).label('mean_rating')).join(self.review,self.product.c.product_id==self.review.c.product_id)
                .where(self.review.c.display_status=='정상').group_by(self.product.c.product_id,self.product.c.product_name)
                .order_by(func.count().desc())).mappings().all()
        return [plain_row(r) for r in rows]

    def list_reviews(self,product_id=None,limit=100,offset=0,model_version=None):
        stmt=select(self.review).where(self.review.c.display_status=='정상')
        if product_id:stmt=stmt.where(self.review.c.product_id==product_id)
        stmt=stmt.order_by(self.review.c.created_at.desc(),self.review.c.review_id).limit(limit).offset(offset)
        with self.engine.connect() as c:
            rows=[dict(r) for r in c.execute(stmt).mappings()]
            if model_version:
                for r in rows:
                    v=c.execute(select(self.cache.c.result_json).where(self.cache.c.text_hash==r['text_hash'],
                        self.cache.c.model_version==model_version)).scalar_one_or_none()
                    r['analysis']=json.loads(v) if v is not None else None
        return rows

    def cached(self,text_hash,model_version):
        with self.engine.connect() as c:
            value=c.execute(select(self.cache.c.result_json).where(self.cache.c.text_hash==text_hash,
                self.cache.c.model_version==model_version)).scalar_one_or_none()
        return json.loads(value) if value else None

    def cache_result(self,text_hash,model_version,result):
        with self.engine.begin() as c:self.put(c,self.cache,[{'text_hash':text_hash,'model_version':model_version,
            'result_json':json.dumps(result,ensure_ascii=False),'created_at':now()}],['text_hash','model_version'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('command',choices=['import','summary'])
    parser.add_argument('--db',default=None,help='DB URL; default is configs/review_pipeline/pipeline.json')
    parser.add_argument('--csv',type=Path)
    args=parser.parse_args()
    store=ReviewStore(args.db)
    if args.command=='import':
        if not args.csv:parser.error('--csv required')
        result=store.import_csv(args.csv)
    else:result=store.summary()
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
