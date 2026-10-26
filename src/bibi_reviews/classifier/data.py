"""Leakage-controlled splits. Human gold alone may validate thresholds and measure quality."""
import numpy as np
from sklearn.model_selection import GroupShuffleSplit


def families(records):
    parent={}
    def root(x):
        parent.setdefault(x,x)
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    def union(a,b):
        a,b=root(a),root(b)
        if a!=b:parent[max(a,b)]=min(a,b)
    by_hash={}
    for r in records:
        rid=r['review_id'];root(rid)
        if r.get('related_review_id'):union(rid,r['related_review_id'])
        if r.get('text_hash'):
            if r['text_hash'] in by_hash:union(rid,by_hash[r['text_hash']])
            by_hash[r['text_hash']]=rid
    return {r['review_id']:root(r['review_id']) for r in records}


def split(records,config,mode,holdout_products=()):
    """Returns {'train','validation','test'} index lists and a description.

    gold_only / gold_plus_silver: validation and test are human gold only; reserved human-eval
    gold is always test. silver_smoke: pipeline smoke run on AI labels; metrics are agreement only.
    """
    fam=families(records);seed=config['seed'];cfg=config['training']['split']
    gold=[i for i,r in enumerate(records) if r['label_source']=='human_gold']
    silver=[i for i,r in enumerate(records) if r['label_source']!='human_gold']
    if mode=='silver_smoke':
        evaluable=silver;train_pool=[]
    else:
        evaluable=gold;train_pool=silver if mode=='gold_plus_silver' else []
    test=[i for i in evaluable if records[i].get('reserved_human_eval')]
    rest=[i for i in evaluable if i not in set(test)]
    groups=[fam[records[i]['review_id']] for i in rest]
    if len(set(groups))<3:raise ValueError(f'Need at least 3 independent review families with {"AI" if mode=="silver_smoke" else "human gold"} labels; found {len(set(groups))}')
    if not test:
        a,b=next(GroupShuffleSplit(1,test_size=cfg['test'],random_state=seed).split(rest,groups=groups))
        test=[rest[k] for k in b];rest=[rest[k] for k in a];groups=[groups[k] for k in a]
    a,b=next(GroupShuffleSplit(1,test_size=cfg['validation']/(1-cfg['test']),random_state=seed+1).split(rest,groups=groups))
    train=[rest[k] for k in a];val=[rest[k] for k in b]
    held={fam[records[i]['review_id']] for i in val+test}
    dropped=[i for i in train_pool if fam[records[i]['review_id']] in held]
    train+=[i for i in train_pool if fam[records[i]['review_id']] not in held]
    sets=[{fam[records[i]['review_id']] for i in s} for s in (train,val,test)]
    if sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2]:raise AssertionError('Family leakage across splits')
    unseen=[i for i in test if records[i]['product_id'] in set(holdout_products)]
    return {'train':train,'validation':val,'test':test,'unseen_product_test':unseen,
            'silver_dropped_for_leakage':len(dropped),'mode':mode,
            'split_unit':'review family (related_review_id + identical normalized text)',
            'evaluation_labels':'llm/ai labels (agreement only, not accuracy)' if mode=='silver_smoke' else 'human_gold'}


def arrays(records,idx):
    return ([records[i]['text'] for i in idx],np.array([records[i]['targets'] for i in idx],np.float32).reshape(len(idx),-1),
            np.array([records[i]['mask'] for i in idx],np.float32).reshape(len(idx),-1))
