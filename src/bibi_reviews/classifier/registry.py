"""Which model version serves production. Activation requires a passed human-gold gate.

    python -m bibi_reviews.classifier.registry show
    python -m bibi_reviews.classifier.registry activate --artifact 04_AI분석/models/classifiers/<dir>
    python -m bibi_reviews.classifier.registry deactivate
"""
import argparse
import json
import time
from pathlib import Path

from ..review_pipeline.common import ROOT,load_config,load_json,save_json,resolve

DEFAULT_REGISTRY=ROOT/'configs/review_pipeline/production_model.json'
EMPTY={'active_version':None,'artifact_dir':None,'validated':False,'activated_at':None,'history':[],
       'note':'No model is active until one passes the human-gold production gate.'}


def read(path=DEFAULT_REGISTRY):
    path=Path(path)
    return load_json(path) if path.exists() else dict(EMPTY)


def activate(artifact_dir,path=DEFAULT_REGISTRY,allow_unvalidated_for_testing=False):
    artifact_dir=Path(artifact_dir);manifest=load_json(artifact_dir/'manifest.json')
    if allow_unvalidated_for_testing and Path(path).resolve()==DEFAULT_REGISTRY.resolve():
        raise PermissionError('Unvalidated activation is only allowed in an isolated test registry')
    if not manifest.get('gate_passed') and not allow_unvalidated_for_testing:
        raise PermissionError(f"{manifest['model_version']} has not passed the human-gold production gate ({manifest['status']})")
    reg=read(path)
    if reg.get('active_version'):reg['history'].append({k:reg[k] for k in ('active_version','artifact_dir','validated','activated_at')})
    reg.update(active_version=manifest['model_version'],artifact_dir=str(artifact_dir),validated=bool(manifest.get('gate_passed')),
               activated_at=time.strftime('%Y-%m-%dT%H:%M:%S%z'),taxonomy_version=manifest['taxonomy_version'],
               test_only=bool(allow_unvalidated_for_testing))
    save_json(path,reg);return reg


def deactivate(path=DEFAULT_REGISTRY):
    reg=read(path)
    if reg.get('active_version'):reg['history'].append({k:reg.get(k) for k in ('active_version','artifact_dir','validated','activated_at')})
    reg.update(active_version=None,artifact_dir=None,validated=False,activated_at=None);save_json(path,reg);return reg


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=['show','activate','deactivate']);p.add_argument('--artifact');p.add_argument('--registry',default=str(DEFAULT_REGISTRY))
    a=p.parse_args()
    try:
        r=read(a.registry) if a.command=='show' else activate(resolve(a.artifact),a.registry) if a.command=='activate' else deactivate(a.registry)
    except PermissionError as exc:p.exit(2,f'REFUSED: {exc}\n')
    print(json.dumps(r,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
