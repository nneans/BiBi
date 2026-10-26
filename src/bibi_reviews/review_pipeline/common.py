"""Config, taxonomy, digests and sentence units shared by labeling, training and serving."""
import hashlib
import json
import os
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
DEFAULT_CONFIG=ROOT/'configs/review_pipeline/pipeline.json'


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False,default=str)+'\n',encoding='utf-8')
    temp.replace(path)


def save_jsonl(path,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False,default=str)+'\n')


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8-sig').splitlines() if x.strip()]


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()


def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def resolve(path):
    path=Path(path)
    return path if path.is_absolute() else ROOT/path


def load_config(path=DEFAULT_CONFIG):
    config=load_json(path)
    config['_path']=str(path)
    # The review DB location can be overridden per machine (e.g. a company PostgreSQL server).
    if os.environ.get('BIBI_DATABASE_URL'):config['database']=os.environ['BIBI_DATABASE_URL']
    return config


def database_url(config_or_url):
    """SQLAlchemy URL of the review DB. A bare file path is treated as SQLite."""
    url=config_or_url['database'] if isinstance(config_or_url,dict) else str(config_or_url)
    return url if '://' in url else 'sqlite:///'+url


_READONLY={}


def readonly_engine(config_or_url):
    """Engine that cannot write: SQLite opens the file with mode=ro, PostgreSQL runs every transaction READ ONLY.
    Passwords never go in the URL; PostgreSQL reads them from the user's pgpass file."""
    url=database_url(config_or_url)
    if url not in _READONLY:
        from sqlalchemy import create_engine
        if url.startswith('sqlite'):
            import sqlite3
            path=resolve(url.split(':///',1)[1])
            _READONLY[url]=create_engine('sqlite://',creator=lambda:sqlite3.connect(f'file:{path}?mode=ro',uri=True,check_same_thread=False))
        else:
            _READONLY[url]=create_engine(url,execution_options={'postgresql_readonly':True},pool_pre_ping=True)
    return _READONLY[url]


def plain_row(mapping):
    """Row as a dict of JSON-ready values. PostgreSQL returns Decimal for avg()/numeric; SQLite returns float."""
    from decimal import Decimal
    return {k:(float(v) if isinstance(v,Decimal) else v) for k,v in dict(mapping).items()}


def read_rows(config_or_url,sql,**params):
    """Run one read-only query and return rows as dicts. Use :name parameters."""
    from sqlalchemy import text
    with readonly_engine(config_or_url).connect() as c:
        return [plain_row(r) for r in c.execute(text(sql),params).mappings()]


def in_clause(name,values):
    """(':name0,:name1,...', params) for an IN list that works on SQLite and PostgreSQL."""
    values=list(values)
    return ','.join(f':{name}{i}' for i in range(len(values))) or 'NULL',{f'{name}{i}':v for i,v in enumerate(values)}


def display_url(config_or_url):
    """URL safe to print or store in a report (no password)."""
    from sqlalchemy.engine import make_url
    return make_url(database_url(config_or_url)).render_as_string(hide_password=True)


def load_taxonomy(config_or_path):
    path=config_or_path['taxonomy'] if isinstance(config_or_path,dict) else config_or_path
    return load_json(resolve(path))


def output_aspects(taxonomy):
    """Model outputs are generated from the taxonomy, never hard-coded."""
    return [a['code'] for a in taxonomy['aspects'] if a.get('model_output')]


def label_names(aspects):
    return [f'{a}:{s}' for a in aspects for s in ('positive','negative')]


def llm_enabled():
    return os.environ.get('LLM_ENABLED','false').strip().lower() in {'1','true','yes'}


class LLMDisabledError(RuntimeError):
    pass


def require_llm_enabled():
    # Production default is false; labeling commands must opt in explicitly.
    if not llm_enabled():
        raise LLMDisabledError('LLM_ENABLED is not true; external LLM calls are blocked in this environment')


_KIWI=None


def kiwi():
    global _KIWI
    if _KIWI is None:
        from kiwipiepy import Kiwi
        _KIWI=Kiwi(num_workers=1)
    return _KIWI


def split_sentences(text):
    """Sentences with [start,end) Unicode code-point offsets into the given text."""
    if not text or not text.strip():return []
    spans=kiwi().split_into_sents(text)
    if not spans:spans=[type('S',(),{'start':0,'end':len(text)})()]
    result=[]
    for span in spans:
        raw=text[span.start:span.end]
        start=span.start+len(raw)-len(raw.lstrip());value=raw.strip()
        if value:result.append({'sentence_id':len(result),'text':value,'start':start,'end':start+len(value)})
    return result


def has_content(text):
    return bool(text and re.search(r'[가-힣A-Za-z0-9]',text))
