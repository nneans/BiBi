"""Opinion ontology: concepts, aspect hierarchy, reasons and integrity rules (JSON + relational).

Definitions of aspects and reasons come from the base taxonomy (configs/review_pipeline/taxonomy_v2.json);
configs/ontology/opinion_ontology_v1.json adds hierarchy, relations, polarity display names, cues and rules.
No inference engine: relations are plain data that the opinion graph stores as rows.
"""
import argparse
import copy
import hashlib
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

from ..review_pipeline.common import ROOT,load_json,resolve

DEFAULT_ONTOLOGY=ROOT/'configs/ontology/opinion_ontology_v1.json'
SPECIAL_ASPECTS=('none','unknown','out_of_scope')


class OntologyError(ValueError):
    pass


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Ontology:
    def __init__(self,path=DEFAULT_ONTOLOGY):
        self.path=Path(path);raw=load_json(self.path);self.raw=raw
        base_path=resolve(raw['base_taxonomy']['path']);base=load_json(base_path);self.base=base
        if base['version']!=raw['base_taxonomy']['version']:
            raise OntologyError(f"base taxonomy is {base['version']}, ontology expects {raw['base_taxonomy']['version']}")
        self.version=raw['ontology_version'];self.base_version=base['version']
        self.sha256=hashlib.sha256((_sha(self.path)+_sha(base_path)).encode()).hexdigest()
        self.enums=raw['enums'];self.rules=raw['integrity_rules'];self.counting=raw['counting_rules']
        parent={child:g['code'] for g in raw['aspect_groups'] for child in g['children']}
        self.aspects={}
        for g in raw['aspect_groups']:
            self.aspects[g['code']]={'code':g['code'],'name':g['name'],'kind':'group','parent':None,'status':'structure',
                                     'aggregation':g.get('aggregation','attribute'),'children':list(g['children']),'model_output':False,
                                     'confused_with':[],'note':g.get('note')}
        for a in base['aspects']+raw.get('aspect_extensions',[]):
            if a['code'] in self.aspects:raise OntologyError(f"duplicate aspect code {a['code']}")
            item=copy.deepcopy(a);item.update(kind='aspect',parent=parent.get(a['code']),
                                              aggregation=a.get('aggregation','attribute'),source='extension' if a in raw.get('aspect_extensions',[]) else 'base')
            self.aspects[a['code']]=item
        sem=raw['reason_semantics'];self.reasons={}
        for r in base['reason_candidates']+raw.get('reason_extensions',[]):
            if r['code'] in self.reasons:raise OntologyError(f"duplicate reason code {r['code']}")
            s=sem.get(r['code'],{})
            self.reasons[r['code']]={'code':r['code'],'name':r['name'],'primary_aspect':r['aspect'],
                'status':r.get('status','candidate' if base.get('frozen') is False else 'confirmed'),
                'source':'extension' if r in raw.get('reason_extensions',[]) else 'base',
                'display':s.get('display',{}),'related_aspects':s.get('related_aspects',[]),
                'related_reasons':s.get('related_reasons',[]),'cue':s.get('cue')}
        self._cue={k:re.compile(v['cue']) for k,v in self.reasons.items() if v['cue']}
        problems=self.check()
        if problems:raise OntologyError('; '.join(problems))

    # --- structure -------------------------------------------------------------------------
    def check(self):
        p=[];aspects=self.aspects
        for g in (a for a in aspects.values() if a['kind']=='group'):
            for c in g['children']:
                if c not in aspects or aspects[c]['kind']!='aspect':p.append(f"group {g['code']} child {c} is not an aspect")
        for a in (a for a in aspects.values() if a['kind']=='aspect'):
            if not a['parent']:p.append(f"aspect {a['code']} has no group")
            for c in a.get('confused_with',[]):
                if c not in aspects:p.append(f"aspect {a['code']} confused_with unknown {c}")
        for r in self.reasons.values():
            if aspects.get(r['primary_aspect'],{}).get('kind')!='aspect':p.append(f"reason {r['code']} primary aspect {r['primary_aspect']} invalid")
            for a in r['related_aspects']:
                if aspects.get(a,{}).get('kind')!='aspect':p.append(f"reason {r['code']} related aspect {a} invalid")
                if a==r['primary_aspect']:p.append(f"reason {r['code']} lists its primary aspect as related")
            for x in r['related_reasons']:
                if x not in self.reasons:p.append(f"reason {r['code']} related reason {x} unknown")
            if set(r['display'])!={'positive','negative'}:p.append(f"reason {r['code']} needs positive/negative display names")
            if not r['cue']:p.append(f"reason {r['code']} has no lexical cue")
        missing=set(self.raw['reason_semantics'])-set(self.reasons)
        if missing:p.append(f'reason_semantics for undefined reasons: {sorted(missing)}')
        return p

    def leaf_aspects(self):return [a for a in self.aspects.values() if a['kind']=='aspect']
    def group_of(self,aspect):return self.aspects.get(aspect,{}).get('parent')
    def is_attribute(self,aspect):
        a=self.aspects.get(aspect);return bool(a and a['kind']=='aspect' and a['aggregation']=='attribute')

    def reason_compatible(self,reason,aspect):
        r=self.reasons.get(reason)
        if not r:return False
        if aspect in (r['primary_aspect'],*r['related_aspects']):return True
        return aspect in self.aspects[r['primary_aspect']].get('confused_with',[])

    def eligible_reasons(self,aspect):
        """Reasons an opinion on `aspect` may link to: primary first, then related/confusable ones."""
        primary=[c for c,r in self.reasons.items() if r['primary_aspect']==aspect]
        return primary+[c for c in self.reasons if c not in primary and self.reason_compatible(c,aspect)]

    def match_reasons(self,text,aspect):
        """Lexical reason candidates in `text` for an opinion on `aspect`. Candidates only, never confirmed."""
        out=[]
        for code in self.eligible_reasons(aspect):
            r=self.reasons[code]
            if code in self._cue and self._cue[code].search(text or ''):
                out.append({'reason_code':code,'role':'primary' if r['primary_aspect']==aspect else 'secondary'})
        # A reason owned by another aspect stays secondary: it hints that the opinion's aspect may be off.
        if sum(o['role']=='primary' for o in out)>1:
            for o in [o for o in out if o['role']=='primary'][1:]:o['role']='secondary'
        return out

    def reason_cues(self):return {k:v['cue'] for k,v in self.reasons.items() if v['cue']}

    def display(self,reason,sentiment):
        r=self.reasons.get(reason)
        if not r:return None
        return r['display'].get(sentiment) or r['name']

    def normalize_sentiment(self,value):
        value=self.enums['sentiment_aliases'].get(value,value)
        return value if value in self.enums['sentiment'] else None

    # --- opinion integrity -------------------------------------------------------------------
    def validate_opinion(self,o,text):
        """Issues for one opinion against the review text. 'error:' issues make the opinion unusable."""
        issues=[];allowed=self.rules['sentiment_by_statement_type']
        if o.get('target_scope') not in self.enums['target_scope']:issues.append('error:invalid_target_scope')
        kind=o.get('statement_type')
        if kind not in self.enums['statement_type']:issues.append('error:invalid_statement_type')
        sentiment=self.normalize_sentiment(o.get('sentiment'))
        if sentiment is None:issues.append('error:invalid_sentiment')
        elif kind in allowed and sentiment not in allowed[kind]:issues.append('error:sentiment_statement_conflict')
        aspect=o.get('aspect')
        if aspect not in SPECIAL_ASPECTS and self.aspects.get(aspect,{}).get('kind')!='aspect':issues.append('error:invalid_aspect')
        if kind=='evaluation' and aspect=='none':issues.append('error:evaluation_without_aspect')
        spans=o.get('evidence') or []
        if not spans:issues.append('error:no_evidence')
        for s in spans:
            a,b=s.get('start'),s.get('end')
            if not (isinstance(a,int) and isinstance(b,int) and 0<=a<b<=len(text or '')):issues.append('error:evidence_offset_invalid');continue
            if text[a:b]!=s.get('text'):issues.append('error:evidence_text_mismatch')
        reasons=o.get('reasons') or []
        if sum(r.get('role')=='primary' for r in reasons)>self.rules['max_primary_reasons_per_opinion']:issues.append('error:multiple_primary_reasons')
        for r in reasons:
            if r['reason_code'] not in self.reasons:issues.append('error:unknown_reason_code')
            elif aspect not in SPECIAL_ASPECTS and not self.reason_compatible(r['reason_code'],aspect):issues.append('warn:reason_aspect_mismatch')
        rt=(o.get('reason_text') or '').strip()
        # Free-text reason is a summary unless quoted verbatim; the evidence span is what is verified.
        if rt and text and rt not in text:issues.append('info:reason_text_not_verbatim')
        if aspect=='out_of_scope':issues.append('warn:out_of_taxonomy')
        if o.get('target_scope')=='current_product' and text and spans:
            from ..review_pipeline.data_quality import CUES
            window=' '.join(s.get('text','') for s in spans)
            if re.search(CUES['other_product_reference'],window):issues.append('warn:possible_misattribution')
        return sorted(set(issues))

    # --- views -----------------------------------------------------------------------------
    def tree(self):
        def aspect_node(code):
            a=self.aspects[code]
            return {'code':code,'name':a['name'],'status':a.get('status'),'model_output':bool(a.get('model_output')),
                    'aggregation':a['aggregation'],
                    'reasons':[{'code':c,'name':self.reasons[c]['name'],'display':self.reasons[c]['display'],
                                'status':self.reasons[c]['status'],'related_aspects':self.reasons[c]['related_aspects']}
                               for c,r in self.reasons.items() if r['primary_aspect']==code]}
        return {'ontology_version':self.version,'base_taxonomy_version':self.base_version,'sha256':self.sha256,'frozen':bool(self.raw.get('frozen')),
                'groups':[{'code':g['code'],'name':g['name'],'aggregation':g['aggregation'],'aspects':[aspect_node(c) for c in g['children']]}
                          for g in self.aspects.values() if g['kind']=='group']}


@lru_cache(maxsize=4)
def load_ontology(path=None):
    return Ontology(Path(path) if path else DEFAULT_ONTOLOGY)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('command',choices=['check','tree']);p.add_argument('--path',default=str(DEFAULT_ONTOLOGY))
    a=p.parse_args()
    try:o=Ontology(a.path)
    except OntologyError as exc:print(f'INVALID: {exc}');sys.exit(1)
    if a.command=='check':
        print(json.dumps({'ontology_version':o.version,'base_taxonomy_version':o.base_version,'sha256':o.sha256,'valid':True,
                          'groups':sum(x['kind']=='group' for x in o.aspects.values()),'aspects':len(o.leaf_aspects()),
                          'reasons':len(o.reasons),'reason_related_aspect_links':sum(len(r['related_aspects']) for r in o.reasons.values()),
                          'reason_reason_links':sum(len(r['related_reasons']) for r in o.reasons.values())},ensure_ascii=False,indent=2))
    else:print(json.dumps(o.tree(),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
