"""Masked multi-label metrics, validation-only thresholds and the production gate."""
import numpy as np


def _prf(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0.;r=tp/(tp+fn) if tp+fn else 0.
    return p,r,(2*p*r/(p+r) if p+r else 0.)


def masked_metrics(Y,M,S,thresholds,supported,names,min_support=10):
    Y=np.asarray(Y);M=np.asarray(M);P=(np.asarray(S)>=np.asarray(thresholds))&np.asarray(supported,bool)
    rows=[];TP=FP=FN=0
    for j,name in enumerate(names):
        m=M[:,j]>0;tp=int(((P[:,j]==1)&(Y[:,j]==1)&m).sum());fp=int(((P[:,j]==1)&(Y[:,j]==0)&m).sum())
        fn=int(((P[:,j]==0)&(Y[:,j]==1)&m).sum());support=int((Y[:,j]*m).sum())
        p,r,f=_prf(tp,fp,fn);TP+=tp;FP+=fp;FN+=fn
        rows.append({'label':name,'support':support,'known_cells':int(m.sum()),'tp':tp,'fp':fp,'fn':fn,
                     'precision':p,'recall':r,'f1':f,'evaluated':support>=min_support,
                     'status':'evaluated' if support>=min_support else '평가 불충분'})
    _,_,micro=_prf(TP,FP,FN)
    evaluated=[r for r in rows if r['evaluated']]
    neg=[r for r in evaluated if r['label'].endswith(':negative')]
    # False complaint: predicted negative where the reviewed truth has no such negative.
    neg_idx=[j for j,n in enumerate(names) if n.endswith(':negative')]
    reviewed=(M[:,neg_idx]>0).all(1) if neg_idx else np.zeros(len(Y),bool)
    false_complaint=float(((P[:,neg_idx]&(Y[:,neg_idx]==0)).any(1)&reviewed).sum()/max(reviewed.sum(),1)) if neg_idx else None
    missed_negative=int(sum(r['fn'] for r in rows if r['label'].endswith(':negative')))
    return {'micro_f1':micro,'macro_f1_evaluated':float(np.mean([r['f1'] for r in evaluated])) if evaluated else None,
            'negative_recall_evaluated_min':min((r['recall'] for r in neg),default=None),
            'false_complaint_review_rate':false_complaint,'missed_negative_opinions':missed_negative,
            'reviews':int(len(Y)),'per_label':rows,'insufficient_labels':[r['label'] for r in rows if not r['evaluated']]}


def select_thresholds(Y,M,S,supported,grid,min_val_positive=5):
    """Per-label threshold where validation support allows; otherwise one shared micro-F1 threshold."""
    Y=np.asarray(Y);M=np.asarray(M);S=np.asarray(S);L=Y.shape[1]
    def micro(t):
        P=(S>=t)&np.asarray(supported,bool);m=M>0
        tp=((P==1)&(Y==1)&m).sum();fp=((P==1)&(Y==0)&m).sum();fn=((P==0)&(Y==1)&m).sum()
        return _prf(tp,fp,fn)[2]
    shared=max(grid,key=micro);out=[];meta=[]
    for j in range(L):
        m=M[:,j]>0;pos=int((Y[m,j]==1).sum())
        if pos>=min_val_positive and (Y[m,j]==0).sum()>=min_val_positive:
            def f(t):
                p=S[m,j]>=t;y=Y[m,j]==1
                return _prf(int((p&y).sum()),int((p&~y).sum()),int((~p&y).sum()))[2]
            t=max(grid,key=f);out.append(t);meta.append('per_label_validation')
        else:out.append(shared);meta.append('shared_validation_micro')
    return out,meta


def gate(test_metrics,unseen_metrics,test_products,gate_config):
    g=gate_config;checks={}
    checks['minimum_test_reviews']=test_metrics['reviews']>=g['minimum_test_reviews']
    checks['minimum_test_products']=test_products>=g['minimum_test_products']
    checks['review_micro_f1']=test_metrics['micro_f1']>=g['minimum_review_micro_f1']
    macro=test_metrics['macro_f1_evaluated'];checks['review_macro_f1_evaluated']=macro is not None and macro>=g['minimum_review_macro_f1_evaluated']
    neg=test_metrics['negative_recall_evaluated_min'];checks['negative_recall']=neg is not None and neg>=g['minimum_negative_recall_evaluated']
    fc=test_metrics['false_complaint_review_rate'];checks['false_complaint_rate']=fc is not None and fc<=g['maximum_false_complaint_rate']
    if unseen_metrics and unseen_metrics['reviews']:
        checks['unseen_product_drop']=test_metrics['micro_f1']-unseen_metrics['micro_f1']<=g['unseen_product_micro_f1_drop_max']
    else:checks['unseen_product_drop']=False
    return {'passed':all(checks.values()),'checks':checks,'gate':g}
