"""Approach A in production: classifier labels -> Opinion instances. Offline; no LLM, no network.

One supported (aspect, sentiment) label of the saved classifier becomes one opinion on the current
product. Evidence is the classifier's max-scoring sentence (evidence_kind='sentence', not an exact span).
Reasons are lexical candidates from the ontology cues inside that sentence (link_status='candidate');
they are never confirmed and never registered as new ontology reasons.
"""
import hashlib


def prediction_opinion_id(text_hash,model_version,aspect,sentiment):
    return 'p:'+hashlib.sha256(f'{text_hash}|{model_version}|{aspect}|{sentiment}'.encode()).hexdigest()[:32]


def opinions_from_result(text_hash,result,ontology):
    out=[];uncertain={(u['aspect'],u['sentiment']) for u in result.get('uncertain_labels',[])}
    for label in result.get('labels',[]):
        aspect,sentiment=label['aspect'],label['sentiment']
        oid=prediction_opinion_id(text_hash,result['model_version'],aspect,sentiment)
        sentence=label.get('evidence_sentence') or ''
        reasons=[{'reason_code':r['reason_code'],'role':r['role'],'link_status':'candidate','method':'lexicon'}
                 for r in ontology.match_reasons(sentence,aspect)]
        out.append({'opinion_id':oid,'text_hash':text_hash,'model_version':result['model_version'],'aspect':aspect,'sentiment':sentiment,
                    'target_scope':'current_product','score':label.get('score'),'threshold':label.get('threshold'),
                    'status':'review_required' if result.get('status')=='review_required' or (aspect,sentiment) in uncertain else 'unverified',
                    'evidence':[{'evidence_id':oid+'#0','start_offset':label['evidence_start'],'end_offset':label['evidence_end'],
                                 'evidence_text':sentence,'sentence_ids_json':None,'cross_sentence':False,'evidence_kind':'sentence',
                                 'verified':False}],
                    'reasons':reasons})
    return out
