"""Shared fixtures. `review_db_url` runs a test on SQLite and, when the configured PostgreSQL server is reachable,
again on a scratch PostgreSQL database that is created for the test and dropped afterwards."""
import uuid

import pytest


def _postgres_admin():
    from sqlalchemy import create_engine,text
    from sqlalchemy.engine import make_url
    from bibi_reviews.review_pipeline.common import load_config
    url=make_url(load_config()['database'])
    if not url.drivername.startswith('postgresql'):return None,None
    try:
        admin=create_engine(url.set(database='postgres'),isolation_level='AUTOCOMMIT')
        with admin.connect() as c:c.execute(text('select 1'))
        return admin,url
    except Exception:
        return None,None


@pytest.fixture(params=['sqlite','postgresql'])
def review_db_url(request,tmp_path):
    if request.param=='sqlite':
        yield 'sqlite:///'+str(tmp_path/'svc.db');return
    admin,url=_postgres_admin()
    if admin is None:pytest.skip('PostgreSQL server not reachable')
    from sqlalchemy import text
    name='bibi_test_'+uuid.uuid4().hex[:12]
    with admin.connect() as c:c.execute(text(f"CREATE DATABASE {name} TEMPLATE template0 ENCODING 'UTF8'"))
    yield url.set(database=name).render_as_string(hide_password=False)
    with admin.connect() as c:c.execute(text(f'DROP DATABASE IF EXISTS {name} WITH (FORCE)'))
