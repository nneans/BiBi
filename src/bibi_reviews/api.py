"""Authenticated backbone API; dashboard implementation is a separate client."""
import hashlib
import hmac
import json
import os
import secrets
import base64
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI,Depends,HTTPException,UploadFile,Query
from fastapi.security import HTTPBasic,HTTPBasicCredentials
from starlette.responses import JSONResponse

from . import paths
from sqlalchemy import select,update
from .backbone import ReviewStore,now


def create_app(store=None,password=None,username='bibi',upload_dir=paths.DB_FILES/'uploads',
               enable_analysis=False,max_upload_bytes=512*1024*1024):
    password=password or os.environ.get('BIBI_PASSWORD','')
    if len(password)<12:raise ValueError('Set BIBI_PASSWORD to a passphrase of at least 12 characters')
    store=store or ReviewStore(os.environ.get('BIBI_DATABASE_URL',f"sqlite:///{(paths.DB_FILES/'reviews.sqlite3').as_posix()}"))
    upload_dir=Path(upload_dir);upload_dir.mkdir(parents=True,exist_ok=True)
    salt=secrets.token_bytes(16)
    password_digest=hashlib.pbkdf2_hmac('sha256',password.encode(),salt,200000)
    executor=ThreadPoolExecutor(max_workers=1,thread_name_prefix='bibi-worker')
    analyzer=None
    security=HTTPBasic()

    def authenticate(credentials:HTTPBasicCredentials=Depends(security)):
        check=hashlib.pbkdf2_hmac('sha256',credentials.password.encode(),salt,200000)
        good_password=hmac.compare_digest(check,password_digest)
        good_user=hmac.compare_digest(credentials.username.encode(),username.encode())
        if not (good_password and good_user):
            raise HTTPException(401,'인증 정보를 확인해 주세요.',headers={'WWW-Authenticate':'Basic'})

    def process(job_id,path):
        nonlocal analyzer
        try:
            counts=store.import_csv(path,job_id=job_id,
                completion_status='awaiting_analysis' if enable_analysis else 'completed')
            if enable_analysis:
                with store.engine.begin() as c:c.execute(update(store.upload).where(store.upload.c.job_id==job_id).values(status='analyzing'))
                from .local_inference import LocalAnalyzer
                from .local_worker import analyze_pending
                if analyzer is None:analyzer=LocalAnalyzer()
                result=analyze_pending(store,analyzer,upload_id=job_id,limit=None)
                counts['analysis']=result
                with store.engine.begin() as c:c.execute(update(store.upload).where(store.upload.c.job_id==job_id)
                    .values(status='completed',finished_at=now(),counts_json=json.dumps(counts)))
        except Exception as exc:
            existing=store.get_upload(job_id)
            status='completed_with_analysis_error' if existing and existing['status'] in ['awaiting_analysis','analyzing'] else 'failed'
            with store.engine.begin() as c:c.execute(update(store.upload).where(store.upload.c.job_id==job_id)
                .values(status=status,finished_at=now(),error=str(exc)[:500]))

    @asynccontextmanager
    async def lifespan(app):
        # Queued/running uploads can be replayed safely after process interruption.
        with store.engine.connect() as c:
            pending=c.execute(select(store.upload.c.job_id).where(store.upload.c.status.in_(['queued','running','awaiting_analysis','analyzing']))).scalars().all()
        with store.engine.begin() as c:
            c.execute(update(store.upload).where(store.upload.c.status=='receiving').values(
                status='failed',error='File transfer interrupted; please upload again.',finished_at=now()))
        for job_id in pending:
            path=upload_dir/f'{job_id}.csv'
            if path.exists():executor.submit(process,job_id,path)
            else:
                with store.engine.begin() as c:c.execute(update(store.upload).where(store.upload.c.job_id==job_id)
                    .values(status='failed',error='Source upload file unavailable; please upload again.',finished_at=now()))
        yield
        executor.shutdown(wait=True)

    app=FastAPI(title='비비수산 리뷰 분석 백본',docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)
    app.state.store=store

    class Guard:
        def __init__(self,app):self.app=app
        async def __call__(self,scope,receive,send):
            if scope['type']!='http' or not scope['path'].startswith('/api/'):
                return await self.app(scope,receive,send)
            headers=dict(scope.get('headers',[]))
            try:
                kind,encoded=headers.get(b'authorization',b'').split(b' ',1)
                if kind.lower()!=b'basic':raise ValueError('basic required')
                user,secret=base64.b64decode(encoded,validate=True).decode('utf-8').split(':',1)
                authenticate(HTTPBasicCredentials(username=user,password=secret))
            except Exception:
                return await JSONResponse({'detail':'인증 정보를 확인해 주세요.'},401,
                    headers={'WWW-Authenticate':'Basic'})(scope,receive,send)
            cap=max_upload_bytes+65536
            try:length=int(headers.get(b'content-length',b'0'))
            except ValueError:return await JSONResponse({'detail':'Invalid content length'},400)(scope,receive,send)
            if length<0 or length>cap:
                return await JSONResponse({'detail':'파일 크기 제한을 초과했습니다.'},413)(scope,receive,send)
            total=0
            async def limited_receive():
                nonlocal total
                message=await receive()
                total+=len(message.get('body',b''))
                if total>cap:raise HTTPException(413,'파일 크기 제한을 초과했습니다.')
                return message
            await self.app(scope,limited_receive,send)
    app.add_middleware(Guard)

    @app.get('/health')
    def health():return {'status':'ok'}

    @app.get('/api/schema',dependencies=[Depends(authenticate)])
    def schema():return app.openapi()

    @app.get('/api/summary',dependencies=[Depends(authenticate)])
    def summary():return store.summary()|{'automatic_analysis_enabled':enable_analysis,'production_model_validated':False}

    @app.get('/api/products',dependencies=[Depends(authenticate)])
    def products():return store.products()

    @app.get('/api/reviews',dependencies=[Depends(authenticate)])
    def reviews(product_id:str|None=None,limit:int=Query(50,ge=1,le=100),offset:int=Query(0,ge=0),model_version:str|None=None):
        rows=store.list_reviews(product_id,limit,offset,model_version)
        keys=['review_id','product_id','rating','created_at','review_type','review_text','analysis']
        return [{k:r[k] for k in keys if k in r} for r in rows]

    @app.post('/api/uploads',status_code=202,dependencies=[Depends(authenticate)])
    async def upload(file:UploadFile):
        if not file.filename or Path(file.filename).suffix.lower()!='.csv':raise HTTPException(400,'CSV 파일을 선택해 주세요.')
        # File path is server generated; never trust a client-provided path.
        job_id=store.create_upload(Path(file.filename).name,status='receiving')
        path=upload_dir/f'{job_id}.csv'
        partial=upload_dir/f'{job_id}.csv.part'
        total=0;digest=hashlib.sha256()
        try:
            with partial.open('xb') as f:
                while block:=await file.read(1024*1024):
                    total+=len(block)
                    if total>max_upload_bytes:raise HTTPException(413,'파일 크기 제한을 초과했습니다.')
                    digest.update(block);f.write(block)
            partial.replace(path)
            with store.engine.begin() as c:c.execute(update(store.upload).where(store.upload.c.job_id==job_id).values(file_sha256=digest.hexdigest(),status='queued'))
        except Exception as exc:
            path.unlink(missing_ok=True)
            partial.unlink(missing_ok=True)
            with store.engine.begin() as c:c.execute(update(store.upload).where(store.upload.c.job_id==job_id).values(status='failed',error='upload not received',finished_at=now()))
            raise
        finally:await file.close()
        executor.submit(process,job_id,path)
        return {'job_id':job_id,'status':'queued','message':'파일 반영을 시작했습니다.'}

    @app.get('/api/uploads/{job_id}',dependencies=[Depends(authenticate)])
    def upload_status(job_id:str):
        result=store.get_upload(job_id)
        if result is None:raise HTTPException(404,'작업을 찾을 수 없습니다.')
        return result
    return app


def app_from_env():
    return create_app(username=os.environ.get('BIBI_USERNAME','bibi'),
                      enable_analysis=os.environ.get('BIBI_AUTO_ANALYZE','0')=='1')
