"""Offline inference with a saved artifact. No network, no LLM, no training.

The analyzer exposes `version` and `analyze_batch(texts)`; results keep the model version,
thresholds, the evidence sentence with Unicode offsets into the normalized review text, and
unverified reason candidates. Low-margin scores route the review to human review.
"""
import re
from pathlib import Path

import numpy as np

from ..review_pipeline.common import load_json,split_sentences,has_content,file_sha256
from ..ontology import load_ontology
from . import models as M


class ModelArtifactError(RuntimeError):
    pass


# Surface cues for reason candidates; a match is a candidate for review, never a verified reason.
# Single source: the opinion ontology (configs/ontology/opinion_ontology_v1.json, reason_semantics.*.cue).
REASON_CUES=load_ontology().reason_cues()


def verify_files(directory,manifest):
    expected=manifest.get('files_sha256') or {}
    if not expected:raise ModelArtifactError('Artifact manifest lists no files')
    for rel,sha in expected.items():
        path=Path(directory)/rel
        if not path.exists():raise ModelArtifactError(f'Missing model file: {rel}')
        if file_sha256(path)!=sha:raise ModelArtifactError(f'Model file checksum mismatch: {rel}')


def load_artifact(directory,config,device=None):
    directory=Path(directory)
    if not (directory/'manifest.json').exists():raise ModelArtifactError(f'No model artifact at {directory}')
    manifest=load_json(directory/'manifest.json')
    verify_files(directory,manifest)
    dep=manifest.get('external_dependencies') or {}
    if dep.get('sbert_weights_sha256'):
        from ..local_inference import weights_digest
        from ..review_pipeline.common import resolve
        path=resolve(dep['sbert_path'])
        if not path.exists() or weights_digest(path)!=dep['sbert_weights_sha256']:
            raise ModelArtifactError('Shared Sentence-BERT encoder missing or changed since training')
    try:
        model=M.load(manifest['model_kind'],directory/'model',{**config,'training':manifest['training_config']},device)
    except Exception as exc:
        raise ModelArtifactError(f'Model could not be loaded: {type(exc).__name__}: {exc}') from exc
    return model,manifest


class ClassifierAnalyzer:
    def __init__(self,artifact_dir,config,validated=False,device=None,taxonomy=None):
        self.model,self.manifest=load_artifact(artifact_dir,config,device)
        self.version=self.manifest['model_version'];self.validated=bool(validated and self.manifest.get('gate_passed'))
        self.labels=self.manifest['labels'];self.thresholds=np.array(self.manifest['thresholds'])
        self.supported=np.array(self.manifest['supported'],bool);self.margin=self.manifest['uncertain_margin']
        reasons=(taxonomy or {}).get('reason_candidates',[])
        self.reasons={a:[r for r in reasons if r['aspect']==a] for a in self.manifest['aspects']}

    def analyze(self,text):return self.analyze_batch([text])[0]

    def analyze_batch(self,texts):
        results=[None]*len(texts);work=[i for i,t in enumerate(texts) if has_content(t)]
        for i,t in enumerate(texts):
            if i not in set(work):results[i]=self._result('unclassified',[],[],['no_meaningful_text'],0)
        if not work:return results
        review_scores=self.model.scores([texts[i] for i in work])
        sentences={i:split_sentences(texts[i]) for i in work}
        flat=[(i,s) for i in work for s in sentences[i]] if any(len(sentences[i])>1 for i in work) else []
        multi=[(i,s) for i,s in flat if len(sentences[i])>1]
        sent_scores=self.model.scores([s['text'] for _,s in multi]) if multi else np.zeros((0,len(self.labels)))
        by_review={}
        for (i,s),sc in zip(multi,sent_scores):by_review.setdefault(i,[]).append((s,sc))
        for k,i in enumerate(work):
            scores=review_scores[k];labels=[];uncertain=[]
            for j,name in enumerate(self.labels):
                if not self.supported[j]:continue
                aspect,sentiment=name.split(':');score=float(scores[j]);t=float(self.thresholds[j])
                if abs(score-t)<self.margin:uncertain.append({'aspect':aspect,'sentiment':sentiment,'score':round(score,4),'threshold':t})
                if score<t:continue
                if i in by_review:s,_=max(by_review[i],key=lambda x:x[1][j])
                else:s=sentences[i][0] if sentences[i] else {'text':texts[i],'start':0,'end':len(texts[i])}
                labels.append({'aspect':aspect,'sentiment':sentiment,'score':round(score,4),'threshold':t,
                               'evidence_sentence':s['text'],'evidence_start':s['start'],'evidence_end':s['end'],
                               'reason_candidates':[r['code'] for r in self.reasons.get(aspect,[]) if r['code'] in REASON_CUES
                                                    and re.search(REASON_CUES[r['code']],s['text'])]})
            status='review_required' if uncertain else ('classified' if labels else 'unclassified')
            results[i]=self._result(status,labels,uncertain,[],len(sentences[i]))
        return results

    def _result(self,status,labels,uncertain,flags,n_sentences):
        return {'model_version':self.version,'status':status,'labels':labels,'uncertain_labels':uncertain,'flags':flags,
                'label_support':dict(zip(self.labels,map(bool,self.supported))),'sentence_count':n_sentences,
                'evidence_method':'max-scoring sentence (review model re-scored per sentence); not exact span extraction',
                'reason_status':'unverified_candidate','offset_basis':'normalized review_text; Unicode code points; end exclusive',
                'scores_are_calibrated_probabilities':False,'human_approved':False,'production_validated':self.validated,
                'no_prediction_means':'not detected by the model; not proof that the customer had no opinion'}
