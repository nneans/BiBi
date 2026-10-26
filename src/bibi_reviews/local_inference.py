"""Local SBERT candidate retrieval and NLI inference. No remote review calls."""
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def make_version(settings,catalogue):
    value=json.dumps({'settings':settings,'catalogue':catalogue},sort_keys=True,ensure_ascii=False)
    return 'local-v1-'+hashlib.sha256(value.encode()).hexdigest()[:24]


def weights_digest(directory):
    digest=hashlib.sha256()
    for path in sorted(Path(directory).rglob('*.safetensors')):
        digest.update(str(path.relative_to(directory)).encode())
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def supported_model_inputs(tokens,type_vocab_size):
    return {k:v for k,v in tokens.items() if k!='token_type_ids' or type_vocab_size>1}


def choose_states(rows,threshold):
    grouped={}
    for row in rows:
        if row['score']>=threshold:grouped.setdefault(row['topic_id'],[]).append(row)
    result=[]
    for candidates in grouped.values():
        candidates=sorted(candidates,key=lambda x:x['score'],reverse=True)
        # Ambiguous or prospective assessment is a separate state, never neutral.
        if candidates[0]['sentiment']=='deferred':result.append(candidates[0]);continue
        result.extend(r for r in candidates if r['sentiment']!='deferred')
    return result


class LocalAnalyzer:
    def __init__(self,embedding_path='04_AI분석/models/sbert_ko-sroberta-multitask',
                 nli_path='04_AI분석/models/nli/mdeberta-nli',catalogue_path='configs/local_analysis_catalogue.json',
                 mode='nli',threshold=None,top_k=4,batch_size=24,device=None):
        import torch
        from sentence_transformers import SentenceTransformer
        from kiwipiepy import Kiwi
        self.torch=torch
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.mode=mode
        self.threshold=threshold if threshold is not None else (.70 if mode=='nli' else .60)
        self.top_k=top_k;self.batch_size=batch_size
        self.catalogue=json.loads(Path(catalogue_path).read_text(encoding='utf-8'))
        self.topics=self.catalogue['topics']
        self.embedder=SentenceTransformer(embedding_path,device=self.device,local_files_only=True)
        self.kiwi=Kiwi(num_workers=2)
        self.hypotheses=[]
        for topic in self.topics:
            for sentiment in ['positive','negative','deferred']:
                self.hypotheses.append({'topic_id':topic['id'],'topic_name':topic['name'],'aspect':topic['aspect'],
                    'sentiment':sentiment,'hypothesis':topic[sentiment]})
        self.example_vectors=self.embedder.encode([r['hypothesis'] for r in self.hypotheses],
            normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)
        emb_manifest=json.loads((Path(embedding_path)/'config.json').read_text(encoding='utf-8'))
        settings={'mode':mode,'threshold':self.threshold,'top_k':top_k,'preprocess':'nfc-html-whitespace-v1',
            'implementation':'local-v2-pair-segments-checked',
            'segmentation':'kiwi-sentences-token-windows-v1','evidence':'sentence',
            'embedding_revision':'8fca7c9c98c26599be0e14b9916b11a756a26f19',
            'embedding_config':emb_manifest,
            'embedding_weights_sha256':weights_digest(embedding_path),
            'nli_weights_sha256':weights_digest(nli_path) if mode=='nli' else None,
            'nli_config_sha256':hashlib.sha256((Path(nli_path)/'config.json').read_bytes()).hexdigest() if mode=='nli' else None,
            'nli_model_path':str(nli_path) if mode=='nli' else None,
            'nli_revision':('2c32ad98958a2e1d470f550efa3cc12848e8d681' if 'klue-roberta-nli' in str(nli_path)
                            else '8adb042d524ecd5c26d3e3ba0e3fbcf7e2d0864c') if mode=='nli' else None,
            'nli_max_tokens':512,'context':'sentence-only; full review retained; temporal flags only'}
        self.settings=settings
        self.version=make_version(settings,self.catalogue)
        self.tokenizer=None
        if mode=='nli':
            from transformers import AutoTokenizer,AutoModelForSequenceClassification
            self.tokenizer=AutoTokenizer.from_pretrained(nli_path,local_files_only=True)
            self.model=AutoModelForSequenceClassification.from_pretrained(nli_path,local_files_only=True,
                use_safetensors=True).to(self.device).eval()
            self.entailment=[int(k) for k,v in self.model.config.id2label.items() if str(v).lower()=='entailment'][0]
            torch.set_num_threads(4)
        elif mode!='prototype':raise ValueError('mode must be nli or prototype')

    def sentence_windows(self,text):
        spans=self.kiwi.split_into_sents(text)
        if not spans:
            return []
        result=[]
        # Keep embeddings within their 128-token limit. NLI sees the same window.
        tokenizer=self.embedder.tokenizer
        capacity=self.embedder.max_seq_length-tokenizer.num_special_tokens_to_add(pair=False)
        for span in spans:
            value=text[span.start:span.end]
            offsets=tokenizer(value,add_special_tokens=False,return_offsets_mapping=True)['offset_mapping']
            if not offsets:continue
            if len(offsets)<=capacity:windows=[(0,len(value))]
            else:
                windows=[]
                for a in range(0,len(offsets),capacity-24):
                    b=min(a+capacity,len(offsets))
                    windows.append((offsets[a][0],offsets[b-1][1]))
                    if b==len(offsets):break
            for a,b in windows:
                chunk=value[a:b]
                if len(tokenizer(chunk)['input_ids'])>self.embedder.max_seq_length:
                    raise ValueError('Window token limit exceeded; no silent truncation')
                result.append({'text':chunk,'start':span.start+a,'end':span.start+b,'split_window':len(windows)>1})
        return result

    def analyze(self,text):
        torch=self.torch
        if not re.search(r'[가-힣A-Za-z0-9]',text):
            return {'status':'unclassified','opinions':[],'flags':['no_meaningful_text'],
                    'model_version':self.version,'evidence_granularity':'sentence'}
        windows=self.sentence_windows(text)
        if not windows:
            return {'status':'unclassified','opinions':[],'flags':['no_sentence'], 'model_version':self.version}
        vectors=self.embedder.encode([w['text'] for w in windows],normalize_embeddings=True,
            convert_to_numpy=True,show_progress_bar=False)
        sims=vectors @ self.example_vectors.T
        jobs=[]
        selected=[]
        for i,w in enumerate(windows):
            per_topic=sims[i].reshape(len(self.topics),3).max(axis=1)
            top=np.argsort(-per_topic)[:self.top_k]
            indices=[int(t*3+s) for t in top for s in range(3)]
            selected.append(indices)
            for j in indices:jobs.append((i,j))
        if self.mode=='nli':
            scores=[]
            with torch.inference_mode():
                for a in range(0,len(jobs),self.batch_size):
                    batch=jobs[a:a+self.batch_size]
                    tokens=self.tokenizer([windows[i]['text'] for i,j in batch],
                        [self.hypotheses[j]['hypothesis'] for i,j in batch],padding=True,truncation=False,return_tensors='pt')
                    if tokens.input_ids.shape[1]>512:raise ValueError('NLI pair exceeds token limit')
                    tokens={k:v.to(self.device) for k,v in supported_model_inputs(tokens,self.model.config.type_vocab_size).items()}
                    logits=self.model(**tokens).logits
                    scores.extend(logits.softmax(-1)[:,self.entailment].float().cpu().tolist())
        else:scores=[float(sims[i,j]) for i,j in jobs]
        by_window={i:[] for i in range(len(windows))}
        for (i,j),score in zip(jobs,scores):
            by_window[i].append(self.hypotheses[j]|{'score':float(score),
                'candidate_similarity':float(sims[i,j])})
        opinions=[];all_candidates=[]
        for i,w in enumerate(windows):
            candidates=by_window[i]
            all_candidates.extend([r|{'start':w['start'],'end':w['end']} for r in candidates])
            for r in choose_states(candidates,self.threshold):
                opinions.append(r|{'evidence':w['text'],'start':w['start'],'end':w['end'],
                    'assessment_status':'deferred' if r['sentiment']=='deferred' else 'evaluated'})
        flags=[]
        if any(w['split_window'] for w in windows):flags.append('long_sentence_windowed')
        if re.search(r'저번|지난번|예전|이전|전에는|이번엔|그러나|하지만',text):flags.append('temporal_or_contrast_context_requires_review')
        if re.search(r'가시|뼈|냄새|포장|배송|가격|맛|두께',text) and len(windows)==1 and len(opinions)>1:
            flags.append('possibly_compound_sentence')
        if any(sum(o['topic_id']==t and o['assessment_status']=='evaluated' for o in opinions)>1 for t in {o['topic_id'] for o in opinions}):
            flags.append('multiple_or_conflicting_topic_states')
        return {'status':'review_required' if flags and opinions else ('classified' if opinions else 'unclassified'),
            'model_version':self.version,'opinions':opinions,'flags':flags,'candidate_scores':all_candidates,
            'score_kind':'nli_entailment_softmax_uncalibrated' if self.mode=='nli' else 'cosine_similarity',
            'evidence_granularity':'sentence','threshold':self.threshold,'candidate_top_k_topics':self.top_k,
            'human_approved':False,'production_validated':False,'sentence_window_count':len(windows)}
