"""Five comparable multi-label models behind one interface: fit / scores / save / load.

All models predict review-level aspect:polarity scores and ignore masked cells in the loss.
Long inputs are split into token windows without truncation; a review takes the per-label
maximum over its windows (scores are not calibrated probabilities).
"""
import json
import pickle
import random
from pathlib import Path

import numpy as np

from ..review_pipeline.common import resolve


def seed_everything(seed):
    import torch
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def positive_weights(Y,M,cap):
    pos=(Y*M).sum(0);neg=((1-Y)*M).sum(0)
    return np.clip(neg/np.maximum(pos,1),1,cap),pos>0


class TfidfLogReg:
    """Character 2-4 grams + word 1-2 grams; one logistic regression per label on unmasked rows."""
    kind='tfidf_lr'

    def __init__(self,config):self.config=config

    def fit(self,texts,Y,M,val=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion
        self.vectorizer=FeatureUnion([('char',TfidfVectorizer(analyzer='char_wb',ngram_range=(2,4),min_df=1,sublinear_tf=True,max_features=200000)),
                                      ('word',TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=1,sublinear_tf=True,token_pattern=r'(?u)\b\w+\b'))])
        X=self.vectorizer.fit_transform(texts);self.models=[];self.supported=[]
        for j in range(Y.shape[1]):
            rows=M[:,j]>0;labels=Y[rows,j]
            if rows.sum()<2 or len(set(labels))<2:self.models.append(None);self.supported.append(bool(labels.sum()>0));continue
            clf=LogisticRegression(C=4.0,class_weight='balanced',max_iter=2000)
            clf.fit(X[rows],labels);self.models.append(clf);self.supported.append(True)
        return {'epochs':None,'history':[]}

    def scores(self,texts):
        X=self.vectorizer.transform(texts);out=np.zeros((len(texts),len(self.models)),np.float32)
        for j,clf in enumerate(self.models):
            if clf is not None:out[:,j]=clf.predict_proba(X)[:,1]
        return out

    def save(self,directory):
        with open(Path(directory)/'tfidf_lr.pkl','wb') as f:pickle.dump({'vectorizer':self.vectorizer,'models':self.models,'supported':self.supported},f)

    @classmethod
    def load(cls,directory,config,device=None):
        m=cls(config)
        with open(Path(directory)/'tfidf_lr.pkl','rb') as f:state=pickle.load(f)
        m.vectorizer,m.models,m.supported=state['vectorizer'],state['models'],state['supported'];return m


def token_windows(tokenizer,text,max_tokens,overlap):
    capacity=max_tokens-tokenizer.num_special_tokens_to_add(pair=False)
    offsets=tokenizer(text,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
    if len(offsets)<=capacity:return [text]
    out=[];step=capacity-overlap
    for a in range(0,len(offsets),step):
        b=min(a+capacity,len(offsets));out.append(text[offsets[a][0]:offsets[b-1][1]])
        if b==len(offsets):break
    return out


class _TorchBase:
    def _train_loop(self,forward,params,n,Y,M,lr,val_fn):
        import torch
        cfg=self.config['training'];device=self.device
        w,supported=positive_weights(Y,M,cfg['positive_weight_cap'])
        weight=torch.as_tensor(w,dtype=torch.float32,device=device)
        opt=torch.optim.AdamW(params,lr=lr,weight_decay=cfg['weight_decay'])
        rng=np.random.default_rng(self.config['seed']);history=[];best=None;best_score=-1;stale=0
        for epoch in range(1,cfg['epochs']+1):
            self.train_mode(True);order=rng.permutation(n);total=0.
            for a in range(0,n,cfg['batch_size']):
                idx=order[a:a+cfg['batch_size']];mask=torch.as_tensor(M[idx],dtype=torch.float32,device=device)
                if not mask.sum():continue
                logits=forward(idx)
                loss=torch.nn.functional.binary_cross_entropy_with_logits(logits.float(),torch.as_tensor(Y[idx],dtype=torch.float32,device=device),
                     pos_weight=weight,reduction='none')
                loss=(loss*mask).sum()/mask.sum()
                opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(params,1.);opt.step()
                total+=float(loss)*len(idx)
                if not np.isfinite(float(loss)):raise ValueError('Non-finite training loss')
            self.train_mode(False)
            score=val_fn() if val_fn else None
            history.append({'epoch':epoch,'train_loss':total/max(n,1),'validation_score':score})
            if val_fn is None:best=self.state();continue
            if score>best_score:best_score=score;best=self.state();stale=0;best_epoch=epoch
            else:
                stale+=1
                if stale>=cfg['patience']:break
        self.restore(best);self.supported=supported.tolist()
        return {'history':history,'best_validation_score':best_score if val_fn else None,
                'best_epoch':best_epoch if val_fn else len(history)}


class SbertHead(_TorchBase):
    """Frozen ko-sroberta embeddings (mean of window embeddings) + linear or MLP head."""

    def __init__(self,config,kind='sbert_linear',device=None):
        import torch
        self.config=config;self.kind=kind;self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu');self._encoder=None

    def encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder=SentenceTransformer(str(resolve(self.config['training']['sbert']['local_path'])),device=self.device,local_files_only=True)
        return self._encoder

    def embed(self,texts):
        enc=self.encoder();tok=enc.tokenizer;flat=[];spans=[]
        for t in texts:
            w=token_windows(tok,t,enc.max_seq_length,self.config['training']['token_overlap']);spans.append((len(flat),len(flat)+len(w)));flat+=w
        unique=list(dict.fromkeys(flat));lookup={t:i for i,t in enumerate(unique)}
        vec=enc.encode(unique,batch_size=128,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        out=np.stack([vec[[lookup[t] for t in flat[a:b]]].mean(0) for a,b in spans]).astype(np.float32)
        return out/np.maximum(np.linalg.norm(out,axis=1,keepdims=True),1e-12)

    def build(self,dim,labels):
        from torch import nn
        hidden=self.config['training']['mlp_hidden_size']
        net=(nn.Sequential(nn.Dropout(.1),nn.Linear(dim,labels)) if self.kind=='sbert_linear'
             else nn.Sequential(nn.Linear(dim,hidden),nn.ReLU(),nn.Dropout(.1),nn.Linear(hidden,labels)))
        self.head=net.to(self.device);self.dim=dim;self.labels=labels

    def train_mode(self,on):self.head.train(on)
    def state(self):return {k:v.detach().cpu().clone() for k,v in self.head.state_dict().items()}
    def restore(self,s):self.head.load_state_dict(s)

    def fit(self,texts,Y,M,val=None):
        import torch
        seed_everything(self.config['seed']);X=self.embed(texts);self.build(X.shape[1],Y.shape[1])
        Xt=torch.as_tensor(X,device=self.device)
        val_fn=None
        if val is not None:
            Xv=self.embed(val['texts'])
            val_fn=lambda:val['score'](self._scores_from(Xv))
        return self._train_loop(lambda idx:self.head(Xt[idx]),list(self.head.parameters()),len(texts),Y,M,
                                self.config['training']['head_learning_rate'],val_fn)

    def _scores_from(self,X):
        import torch
        with torch.inference_mode():return torch.sigmoid(self.head(torch.as_tensor(X,device=self.device))).float().cpu().numpy()

    def scores(self,texts):return self._scores_from(self.embed(texts))

    def save(self,directory):
        from safetensors.torch import save_file
        save_file({k:v.contiguous() for k,v in self.state().items()},str(Path(directory)/'head.safetensors'))
        (Path(directory)/'head.json').write_text(json.dumps({'dim':self.dim,'labels':self.labels,'kind':self.kind}),encoding='utf-8')

    @classmethod
    def load(cls,directory,config,device=None):
        from safetensors.torch import load_file
        meta=json.loads((Path(directory)/'head.json').read_text(encoding='utf-8'))
        m=cls(config,meta['kind'],device);m.build(meta['dim'],meta['labels'])
        m.head.load_state_dict(load_file(str(Path(directory)/'head.safetensors')));m.head.eval();return m


class EncoderClassifier(_TorchBase):
    """Fine-tuned Korean encoder ([CLS] + linear) with window max-pooling per review."""

    def __init__(self,config,kind='klue_roberta',device=None,source=None):
        import torch
        self.config=config;self.kind=kind;self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu');self.source=source

    def _source(self):
        if self.source:return self.source
        from huggingface_hub import try_to_load_from_cache
        spec=self.config['training']['encoders'][self.kind]
        found=try_to_load_from_cache(spec['checkpoint'],'config.json',revision=spec['revision'])
        if not isinstance(found,str):raise FileNotFoundError(f"{spec['checkpoint']}@{spec['revision']} is not in the local model cache")
        return str(Path(found).parent)

    def build(self,labels,source=None):
        from transformers import AutoModel,AutoTokenizer
        from ..train import AspectClassifier
        source=source or self._source()
        self.tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
        self.model=AspectClassifier(AutoModel.from_pretrained(source,local_files_only=True),labels).to(self.device);self.labels=labels

    def train_mode(self,on):self.model.train(on)
    def state(self):return {k:v.detach().cpu().clone() for k,v in self.model.state_dict().items()}
    def restore(self,s):self.model.load_state_dict(s)

    def _logits(self,texts):
        import torch
        cfg=self.config['training'];flat=[];sizes=[]
        for t in texts:
            w=token_windows(self.tokenizer,t,cfg['max_tokens'],cfg['token_overlap']);sizes.append(len(w));flat+=w
        tok=self.tokenizer(flat,padding=True,truncation=False,return_tensors='pt')
        if getattr(self.model.encoder.config,'type_vocab_size',2)<=1:tok.pop('token_type_ids',None)
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16,enabled=self.device.startswith('cuda')):
            logits=self.model({k:v.to(self.device) for k,v in tok.items()})
        return torch.stack([g.max(0).values for g in logits.float().split(sizes)])

    def fit(self,texts,Y,M,val=None):
        seed_everything(self.config['seed']);self.build(Y.shape[1])
        val_fn=(lambda:val['score'](self.scores(val['texts']))) if val is not None else None
        return self._train_loop(lambda idx:self._logits([texts[i] for i in idx]),list(self.model.parameters()),len(texts),Y,M,
                                self.config['training']['encoder_learning_rate'],val_fn)

    def scores(self,texts,batch=64):
        import torch
        self.model.eval();out=[]
        with torch.inference_mode():
            for a in range(0,len(texts),batch):out.append(torch.sigmoid(self._logits(texts[a:a+batch])).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0,self.labels),np.float32)

    def save(self,directory):
        self.model.save(Path(directory)/'encoder');self.tokenizer.save_pretrained(Path(directory)/'tokenizer')
        (Path(directory)/'encoder_meta.json').write_text(json.dumps({'labels':self.labels,'kind':self.kind}),encoding='utf-8')

    @classmethod
    def load(cls,directory,config,device=None):
        from transformers import AutoTokenizer
        from ..train import AspectClassifier
        import torch
        meta=json.loads((Path(directory)/'encoder_meta.json').read_text(encoding='utf-8'))
        m=cls(config,meta['kind'],device);m.labels=meta['labels']
        m.tokenizer=AutoTokenizer.from_pretrained(Path(directory)/'tokenizer',local_files_only=True)
        m.model=AspectClassifier.load(Path(directory)/'encoder',meta['labels'],torch.device(m.device));m.model.eval();return m


MODEL_KINDS=['tfidf_lr','sbert_linear','sbert_mlp','klue_roberta','kcelectra']


def create(kind,config,device=None):
    if kind=='tfidf_lr':return TfidfLogReg(config)
    if kind in {'sbert_linear','sbert_mlp'}:return SbertHead(config,kind,device)
    if kind in config['training']['encoders']:return EncoderClassifier(config,kind,device)
    raise ValueError('Unknown model kind: '+kind)


def load(kind,directory,config,device=None):
    if kind=='tfidf_lr':return TfidfLogReg.load(directory,config)
    if kind in {'sbert_linear','sbert_mlp'}:return SbertHead.load(directory,config,device)
    return EncoderClassifier.load(directory,config,device)
