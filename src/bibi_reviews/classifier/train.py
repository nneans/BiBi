"""Train and evaluate one candidate model on a fixed split; save a self-describing artifact.

    python -m bibi_reviews.classifier.train --model tfidf_lr --data-mode gold_only
    python -m bibi_reviews.classifier.train --model klue_roberta --data-mode gold_plus_silver

Thresholds come from the validation split only; the test split is scored once after the
model and thresholds are fixed. In silver_smoke mode every number is agreement with AI
labels (not accuracy) and the artifact can never pass the production gate.
"""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from ..review_pipeline.common import (ROOT,load_config,load_taxonomy,output_aspects,label_names,save_json,save_jsonl,read_jsonl,
                                      resolve,digest,file_sha256,load_json)
from . import models as M
from .data import split,arrays
from .evaluate import masked_metrics,select_thresholds,gate

PREPROCESSING_VERSION='backbone.normalize_text (NFC, html unescape, control chars removed, whitespace collapsed) v1'


def micro_scorer(Y,Mk,names,floor):
    def score(S):
        return masked_metrics(Y,Mk,S,[.5]*len(names),[True]*len(names),names,floor)['micro_f1']
    return score


def files_digest(directory):
    directory=Path(directory)
    return {str(p.relative_to(directory)).replace('\\','/'):file_sha256(p) for p in sorted(directory.rglob('*'))
            if p.is_file() and p.name!='manifest.json'}


def train(config,records,kind,mode,output,holdout_products=(),device=None,note=None):
    import torch
    taxonomy=load_taxonomy(config);aspects=output_aspects(taxonomy);names=label_names(aspects)
    if any(len(r['targets'])!=len(names) for r in records):raise ValueError('Records do not match taxonomy outputs')
    parts=split(records,config,mode,holdout_products)
    tr=arrays(records,parts['train']);va=arrays(records,parts['validation']);te=arrays(records,parts['test'])
    if not len(tr[0]):raise ValueError('Empty training split')
    floor=config['training']['minimum_label_support']
    model=M.create(kind,config,device)
    if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
    started=time.perf_counter()
    fit=model.fit(tr[0],tr[1],tr[2],{'texts':va[0],'score':micro_scorer(va[1],va[2],names,floor)} if len(va[0]) else None)
    train_seconds=time.perf_counter()-started
    supported=list(map(bool,getattr(model,'supported',[True]*len(names))))
    val_scores=model.scores(va[0]) if len(va[0]) else np.zeros((0,len(names)))
    thresholds,threshold_meta=(select_thresholds(va[1],va[2],val_scores,supported,config['training']['threshold_grid'])
                               if len(va[0]) else ([.5]*len(names),['default_unvalidated']*len(names)))
    # Test is scored only now, with model and thresholds fixed.
    started=time.perf_counter();test_scores=model.scores(te[0]);infer_seconds=time.perf_counter()-started
    test=masked_metrics(te[1],te[2],test_scores,thresholds,supported,names,floor)
    unseen_idx=[parts['test'].index(i) for i in parts['unseen_product_test']]
    unseen=masked_metrics(te[1][unseen_idx],te[2][unseen_idx],test_scores[unseen_idx],thresholds,supported,names,floor) if unseen_idx else None
    by_product={}
    for pid in sorted({records[i]['product_id'] for i in parts['test']}):
        k=[n for n,i in enumerate(parts['test']) if records[i]['product_id']==pid]
        by_product[pid]=masked_metrics(te[1][k],te[2][k],test_scores[k],thresholds,supported,names,floor)['micro_f1']
    smoke=mode=='silver_smoke'
    verdict={'passed':False,'reason':'silver_smoke: AI labels cannot validate a production model'} if smoke else \
        gate(test,unseen,len(by_product),config['production_gate'])
    output=Path(output)
    if output.exists():raise FileExistsError(f'{output} exists; artifacts are never overwritten')
    (output/'model').mkdir(parents=True);model.save(output/'model')
    peak=torch.cuda.max_memory_allocated()/2**20 if torch.cuda.is_available() else None
    errors=[]
    P=(test_scores>=np.array(thresholds))&np.array(supported)
    for n,i in enumerate(parts['test']):
        for j,name in enumerate(names):
            if te[2][n,j] and bool(te[1][n,j])!=bool(P[n,j]):
                errors.append({'review_id':records[i]['review_id'],'product_id':records[i]['product_id'],'label':name,
                               'error':'false_negative' if te[1][n,j] else 'false_positive','score':round(float(test_scores[n,j]),4),
                               'threshold':thresholds[j],'text':records[i]['text']})
    save_jsonl(output/'test_errors.jsonl',errors)
    evaluation={'evaluation_labels':parts['evaluation_labels'],'metric_meaning':'agreement with AI labels, NOT accuracy' if smoke else 'human gold',
                'test':test,'unseen_product_test':unseen,'test_micro_f1_by_product':by_product,'gate':verdict}
    save_json(output/'evaluation.json',evaluation)
    manifest={'model_version':None,'model_kind':kind,'created_at':time.strftime('%Y-%m-%dT%H:%M:%S%z'),
        'status':'smoke_test_not_for_production' if smoke else ('validated_candidate' if verdict['passed'] else 'candidate_failed_gate'),
        'taxonomy_version':taxonomy['version'],'taxonomy_sha256':digest(taxonomy),'aspects':aspects,'labels':names,
        'label_mapping':{n:{'aspect':n.split(':')[0],'sentiment':n.split(':')[1]} for n in names},
        'thresholds':[float(t) for t in thresholds],'threshold_source':threshold_meta,'supported':supported,
        'uncertain_margin':config['training']['uncertain_margin'],'preprocessing_version':PREPROCESSING_VERSION,
        'evidence':'review-level classifier; evidence = max-scoring sentence re-scored by the same model (not span extraction)',
        'training_config':config['training'],'data_mode':mode,'note':note,
        'dataset_fingerprint':digest([(r['review_id'],r['label_source'],r['targets'],r['mask']) for r in records]),
        'split':{k:[records[i]['review_id'] for i in v] if isinstance(v,list) else v for k,v in parts.items()},
        'label_sources_in_train':sorted({records[i]['label_source'] for i in parts['train']}),
        'fit':fit,'train_seconds':round(train_seconds,2),'test_inference_seconds':round(infer_seconds,3),
        'test_reviews_per_second':round(len(te[0])/max(infer_seconds,1e-9),1),'peak_gpu_memory_mib':peak,
        'device':str(getattr(model,'device','cpu')),'platform':platform.platform(),'gate_passed':bool(verdict['passed']),
        'scores_are_calibrated_probabilities':False,'human_gold_evaluated':not smoke}
    if kind.startswith('sbert'):
        # The frozen encoder is shared, not copied; pin its weights so a silent change is detected at load.
        from ..local_inference import weights_digest
        path=resolve(config['training']['sbert']['local_path'])
        manifest['external_dependencies']={'sbert_path':config['training']['sbert']['local_path'],'sbert_weights_sha256':weights_digest(path)}
    manifest['files_sha256']=files_digest(output)
    manifest['model_version']=f"review-clf-{kind}-{digest(manifest)[:12]}"
    save_json(output/'manifest.json',manifest)
    # Reload check: the saved artifact must reproduce the scores.
    from .inference import load_artifact
    reloaded,_=load_artifact(output,config,device)
    diff=float(np.abs(reloaded.scores(te[0][:64])-test_scores[:64]).max()) if len(te[0]) else 0.
    if diff>1e-4:raise ValueError(f'Reloaded model changed scores by {diff}')
    manifest['reload_max_abs_difference']=diff;save_json(output/'manifest.json',manifest)
    return manifest,evaluation


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--config',default=str(ROOT/'configs/review_pipeline/pipeline.json'))
    p.add_argument('--records');p.add_argument('--model',choices=M.MODEL_KINDS,required=True)
    p.add_argument('--data-mode',choices=['gold_only','gold_plus_silver','silver_smoke'],default='gold_only')
    p.add_argument('--output');p.add_argument('--note')
    args=p.parse_args()
    config=load_config(args.config);work=resolve(config['workspace'])
    records=read_jsonl(args.records or work/'training/training_records.jsonl')
    holdout=load_json(work/'sampling/sampling_manifest.json')['holdout_products']
    output=args.output or resolve(config['artifacts'])/f"{args.model}-{args.data_mode}-{time.strftime('%Y%m%d-%H%M%S')}"
    try:
        manifest,evaluation=train(config,records,args.model,args.data_mode,output,holdout,note=args.note)
    except (ValueError,FileNotFoundError) as exc:
        p.exit(2,f'NOT TRAINED: {exc}\n')
    print(json.dumps({'model_version':manifest['model_version'],'status':manifest['status'],'output':str(output),
                      'test_micro':evaluation['test']['micro_f1'],'metric_meaning':evaluation['metric_meaning']},ensure_ascii=False))


if __name__=='__main__':main()
