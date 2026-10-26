"""Opinion-graph queries for the dashboard. One tier and (for predictions) one model version per call.

Counting follows ontology.counting_rules: reviews are distinct per node, opinions are counted separately,
mixed opinions count toward both polarities and are reported as mixed, group totals are distinct reviews
(not the sum of children), related-aspect links never add to totals, and reason links keep their status
(confirmed only for human gold; candidates are reported apart from confirmed counts).
"""
import json
from collections import defaultdict

from sqlalchemy import text

from ..review_pipeline.common import plain_row

POLARITIES=('positive','negative','neutral','uncertain')


def _period(start,end,column='r.created_at'):
    clauses=[];params={}
    if start:clauses.append(f'{column}>=:start');params['start']=start
    if end:clauses.append(f'{column}<:end');params['end']=end
    return (' and '+' and '.join(clauses)) if clauses else '',params


class OpinionAnalytics:
    def __init__(self,store,graph):
        self.store=store;self.graph=graph;self.engine=store.engine;self.ontology=graph.ontology

    def _rows(self,sql,**params):
        with self.engine.connect() as c:return [plain_row(r) for r in c.execute(text(sql),params).mappings()]

    def base(self,tier,model_version=None,source=None):
        """SQL for one opinion layer joined to its current normal reviews, plus params."""
        if tier=='prediction':
            if not model_version:raise ValueError('prediction tier needs a model_version')
            return ('''select o.opinion_id,r.review_id,r.product_id,r.created_at,r.rating,o.target_scope,'evaluation' as statement_type,
                       o.aspect,o.sentiment,o.status from opinion_predictions o
                       join reviews r on r.text_hash=o.text_hash and r.display_status='정상' where o.model_version=:mv''',{'mv':model_version})
        if tier not in ('gold','silver'):raise ValueError(f'unknown tier {tier}')
        extra=' and o.source_name=:src' if source else ''
        return (f'''select o.opinion_id,r.review_id,r.product_id,r.created_at,r.rating,o.target_scope,o.statement_type,o.aspect,o.sentiment,o.status
                   from opinions o join reviews r on r.review_id=o.review_id and r.text_hash=o.text_hash and r.display_status='정상'
                   where o.label_tier=:tier and o.status not in ('stale_text','source_blinded'){extra}''',{'tier':tier,'src':source})

    def _opinions(self,product_id,tier,model_version,source,start,end,extra='',**params):
        base,bp=self.base(tier,model_version,source);where,pp=_period(start,end,'b.created_at')
        cond=' and b.product_id=:pid' if product_id else ''
        return self._rows(f'select b.* from ({base}) b where 1=1{cond}{where}{extra}',pid=product_id,**bp,**pp,**params)

    def _links(self,opinion_ids):
        out=defaultdict(list)
        ids=list(opinion_ids)
        for a in range(0,len(ids),900):
            chunk=ids[a:a+900];marks=','.join(f':i{k}' for k in range(len(chunk)))
            for r in self._rows(f'select opinion_id,reason_code,role,link_status,method from opinion_reasons where opinion_id in ({marks})',
                                **{f'i{k}':v for k,v in enumerate(chunk)}):out[r['opinion_id']].append(r)
        return out

    def meta(self,tier,model_version=None,source=None):
        notes={'gold':'사람이 승인한 의견입니다.','silver':'AI가 만든 후보 의견입니다. 사람 검수 전이며 확정 통계가 아닙니다.',
               'prediction':'자체 모델 예측입니다. 이유 연결은 어휘 단서 후보이며 확정 이유가 아닙니다.'}
        return {'label_tier':tier,'model_version':model_version if tier=='prediction' else None,'source_name':source,
                'ontology_version':self.ontology.version,'ontology_sha256':self.ontology.sha256,'human_validated':tier=='gold','note':notes[tier]}

    # --- hierarchy ----------------------------------------------------------------------------
    def hierarchy(self,product_id,tier,model_version=None,source=None,start=None,end=None):
        rows=self._opinions(product_id,tier,model_version,source,start,end)
        links=self._links(o['opinion_id'] for o in rows)
        where,pp=_period(start,end)
        total=self._rows(f"select count(*) as n from reviews r where r.product_id=:p and r.display_status='정상'{where}",p=product_id,**pp)[0]['n']
        current=[o for o in rows if o['target_scope']=='current_product']
        scope=defaultdict(lambda:{'opinions':0,'reviews':set()})
        for o in rows:scope[o['target_scope']]['opinions']+=1;scope[o['target_scope']]['reviews'].add(o['review_id'])
        kinds=defaultdict(lambda:{'opinions':0,'reviews':set()})
        for o in current:kinds[o['statement_type']]['opinions']+=1;kinds[o['statement_type']]['reviews'].add(o['review_id'])
        evals=[o for o in current if o['statement_type']=='evaluation']
        node=lambda:{'opinions':0,'reviews':set()}
        def polarity(o):return ['positive','negative'] if o['sentiment']=='mixed' else ([o['sentiment']] if o['sentiment'] in POLARITIES else [])
        groups=[]
        for g in (a for a in self.ontology.aspects.values() if a['kind']=='group'):
            g_reviews=set();g_ops=0;aspects=[]
            for code in g['children']:
                ops=[o for o in evals if o['aspect']==code]
                if not ops:aspects.append({'code':code,'name':self.ontology.aspects[code]['name'],'reviews':0,'opinions':0,'sentiments':{}});continue
                sent={}
                for pol in POLARITIES:
                    p_ops=[o for o in ops if pol in polarity(o)]
                    if not p_ops:continue
                    reasons=defaultdict(lambda:{'opinions':0,'reviews':set(),'confirmed_reviews':set(),'candidate_reviews':set(),'roles':set()})
                    for o in p_ops:
                        for l in links.get(o['opinion_id'],[]):
                            n=reasons[l['reason_code']];n['opinions']+=1;n['reviews'].add(o['review_id']);n['roles'].add(l['role'])
                            (n['confirmed_reviews'] if l['link_status']=='confirmed' else n['candidate_reviews']).add(o['review_id'])
                    listed=sorted([{'reason_code':k,'name':self.ontology.display(k,pol) if pol in ('positive','negative') else self.ontology.reasons[k]['name'],
                                   'concept_name':self.ontology.reasons[k]['name'],'reviews':len(v['reviews']),'opinions':v['opinions'],
                                   'confirmed_reviews':len(v['confirmed_reviews']),'candidate_reviews':len(v['candidate_reviews']),
                                   'cross_aspect':self.ontology.reasons[k]['primary_aspect']!=code,
                                   'reason_primary_aspect':self.ontology.reasons[k]['primary_aspect']} for k,v in reasons.items()],
                                  key=lambda x:(-x['reviews'],x['reason_code']))
                    own={o['review_id'] for o in p_ops if any(self.ontology.reasons.get(l['reason_code'],{}).get('primary_aspect')==code
                                                               for l in links.get(o['opinion_id'],[]))}
                    sent[pol]={'reviews':len({o['review_id'] for o in p_ops}),'opinions':len(p_ops),
                               'mixed_opinions':sum(o['sentiment']=='mixed' for o in p_ops),
                               'review_required_opinions':sum(o['status']=='review_required' for o in p_ops),
                               # reasons owned by this aspect (ontology primary) form the tree; others are hints that the
                               # opinion's aspect may be wrong or that a related aspect is involved, and never add to totals
                               'reasons':[x for x in listed if not x['cross_aspect']],
                               'cross_aspect_reason_hints':[x for x in listed if x['cross_aspect']],
                               'without_specific_reason_reviews':len({o['review_id'] for o in p_ops})-len(own)}
                rv={o['review_id'] for o in ops};g_reviews|=rv;g_ops+=len(ops)
                aspects.append({'code':code,'name':self.ontology.aspects[code]['name'],'reviews':len(rv),'opinions':len(ops),
                                'model_output':bool(self.ontology.aspects[code].get('model_output')),'sentiments':sent})
            groups.append({'code':g['code'],'name':g['name'],'aggregation':g['aggregation'],'reviews':len(g_reviews),'opinions':g_ops,'aspects':aspects})
        other={k:sum(1 for o in evals if o['aspect']==k) for k in ('unknown','out_of_scope')}
        return {'meta':self.meta(tier,model_version,source),'product_id':product_id,'period':{'start':start,'end':end,'end_exclusive':True},
                'denominators':{'normal_reviews_in_period':total,'reviews_with_opinions_in_tier':len({o['review_id'] for o in rows}),
                                'current_product_evaluation_reviews':len({o['review_id'] for o in evals})},
                'counts':{'opinions':len(rows),'reviews':len({o['review_id'] for o in rows})},
                'target_scopes':{k:{'opinions':v['opinions'],'reviews':len(v['reviews'])} for k,v in scope.items()},
                'statement_types_current_product':{k:{'opinions':v['opinions'],'reviews':len(v['reviews'])} for k,v in kinds.items()},
                'unmapped_evaluation_opinions':other,'groups':groups,
                'counting':self.ontology.counting}

    # --- instances & evidence ----------------------------------------------------------------
    def opinion_instances(self,product_id,tier,model_version=None,source=None,aspect=None,sentiment=None,reason=None,
                          target_scope='current_product',start=None,end=None,limit=50,offset=0):
        extra='';params={}
        if aspect:extra+=' and b.aspect=:aspect';params['aspect']=aspect
        if sentiment:extra+=" and (b.sentiment=:sentiment or (b.sentiment='mixed' and :sentiment in ('positive','negative')))";params['sentiment']=sentiment
        if target_scope:extra+=' and b.target_scope=:scope';params['scope']=target_scope
        if reason:extra+=' and exists (select 1 from opinion_reasons x where x.opinion_id=b.opinion_id and x.reason_code=:reason)';params['reason']=reason
        rows=self._opinions(product_id,tier,model_version,source,start,end,extra+' order by b.created_at desc,b.opinion_id limit :lim offset :off',
                            lim=limit,off=offset,**params)
        return self._attach(rows,tier)

    def _attach(self,rows,tier):
        if not rows:return []
        links=self._links(o['opinion_id'] for o in rows)
        ids=[o['opinion_id'] for o in rows];marks=','.join(f':i{k}' for k in range(len(ids)))
        ev=defaultdict(list)
        for e in self._rows(f'''select opinion_id,start_offset,end_offset,evidence_text,evidence_kind,cross_sentence,sentence_ids_json
                                from opinion_evidence where opinion_id in ({marks}) order by start_offset''',**{f'i{k}':v for k,v in enumerate(ids)}):
            e['sentence_ids']=json.loads(e.pop('sentence_ids_json') or 'null');ev[e.pop('opinion_id')].append(e)
        rids=sorted({o['review_id'] for o in rows});rm=','.join(f':r{k}' for k in range(len(rids)))
        reviews={r['review_id']:r for r in self._rows(f'select review_id,review_text,rating,created_at,product_id from reviews where review_id in ({rm})',
                                                      **{f'r{k}':v for k,v in enumerate(rids)})}
        extra={}
        if tier!='prediction':
            extra={r['opinion_id']:r for r in self._rows(f'select opinion_id,reason_text,source_name,source_ref,issues_json,note,uncertain from opinions where opinion_id in ({marks})',
                                                         **{f'i{k}':v for k,v in enumerate(ids)})}
        out=[]
        for o in rows:
            x=extra.get(o['opinion_id'],{})
            out.append({**o,'review_text':reviews[o['review_id']]['review_text'],'evidence':ev.get(o['opinion_id'],[]),
                        'reasons':[{**l,'name':self.ontology.display(l['reason_code'],o['sentiment']) if l['reason_code'] in self.ontology.reasons else None}
                                   for l in links.get(o['opinion_id'],[])],
                        'reason_text':x.get('reason_text'),'source_name':x.get('source_name'),'source_ref':x.get('source_ref'),
                        'issues':json.loads(x['issues_json']) if x.get('issues_json') else [],'note':x.get('note')})
        return out

    def evidence_trace(self,reason_code,tier,model_version=None,source=None,product_id=None,sentiment=None,limit=50):
        return {'meta':self.meta(tier,model_version,source),'reason':self.reason_related(reason_code,tier,model_version,source,product_id),
                'opinions':self.opinion_instances(product_id,tier,model_version,source,reason=reason_code,sentiment=sentiment,target_scope=None,limit=limit)}

    # --- recurring / common / co-occurrence ---------------------------------------------------
    def recurring_reasons(self,product_id,tier,model_version=None,source=None,sentiment='negative',start=None,end=None,min_reviews=1):
        base,bp=self.base(tier,model_version,source);where,pp=_period(start,end,'b.created_at')
        rows=self._rows(f'''select l.reason_code,b.aspect,count(distinct b.review_id) as reviews,count(*) as opinions,
            sum(case when l.link_status='confirmed' then 1 else 0 end) as confirmed_links
            from ({base}) b join opinion_reasons l on l.opinion_id=b.opinion_id
            where b.product_id=:pid and b.target_scope='current_product' and (b.sentiment=:s or b.sentiment='mixed'){where}
            group by l.reason_code,b.aspect having count(distinct b.review_id)>=:m order by reviews desc,l.reason_code''',
            pid=product_id,s=sentiment,m=min_reviews,**bp,**pp)
        for r in rows:r['name']=self.ontology.display(r['reason_code'],sentiment);r['reason_primary_aspect']=self.ontology.reasons.get(r['reason_code'],{}).get('primary_aspect')
        return {'meta':self.meta(tier,model_version,source),'product_id':product_id,'sentiment':sentiment,'reasons':rows}

    def cooccurrence(self,product_id,tier,model_version=None,source=None,start=None,end=None,level='reason',min_reviews=1):
        rows=[o for o in self._opinions(product_id,tier,model_version,source,start,end)
              if o['target_scope']=='current_product' and o['statement_type']=='evaluation']
        links=self._links(o['opinion_id'] for o in rows) if level=='reason' else {}
        items=defaultdict(set)
        for o in rows:
            pols=['positive','negative'] if o['sentiment']=='mixed' else [o['sentiment']]
            for pol in pols:
                if level=='reason':
                    for l in links.get(o['opinion_id'],[]):items[o['review_id']].add(f"{o['aspect']}:{pol}:{l['reason_code']}")
                else:items[o['review_id']].add(f"{o['aspect']}:{pol}")
        pairs=defaultdict(set)
        for rid,keys in items.items():
            ks=sorted(keys)
            for i in range(len(ks)):
                for j in range(i+1,len(ks)):pairs[(ks[i],ks[j])].add(rid)
        out=[{'a':a,'b':b,'reviews':len(v),'example_review_ids':sorted(v)[:5]} for (a,b),v in pairs.items() if len(v)>=min_reviews]
        return {'meta':self.meta(tier,model_version,source),'product_id':product_id,'level':level,
                'reviews_with_two_or_more':sum(len(v)>1 for v in items.values()),'pairs':sorted(out,key=lambda x:(-x['reviews'],x['a'],x['b']))}

    def reason_related(self,reason_code,tier=None,model_version=None,source=None,product_id=None):
        r=self.ontology.reasons.get(reason_code)
        if not r:return None
        observed=[]
        if tier:
            base,bp=self.base(tier,model_version,source);cond=' and b.product_id=:pid' if product_id else ''
            observed=self._rows(f'''select b.aspect,b.sentiment,count(distinct b.review_id) as reviews,count(*) as opinions
                from ({base}) b join opinion_reasons l on l.opinion_id=b.opinion_id where l.reason_code=:rc{cond}
                group by b.aspect,b.sentiment order by reviews desc''',rc=reason_code,pid=product_id,**bp)
        return {'reason_code':reason_code,'name':r['name'],'display':r['display'],'status':r['status'],
                'primary_aspect':r['primary_aspect'],'related_aspects':r['related_aspects'],'related_reasons':r['related_reasons'],
                'observed_opinion_aspects':observed,
                'note':'related_aspects는 온톨로지 설계상 연관이며, observed는 실제 의견 인스턴스의 속성 분포입니다.'}

    def reason_candidates(self,product_id=None,status='pending_review',tier=None,limit=100,offset=0):
        cond='';params={'st':status,'lim':limit,'off':offset}
        if product_id:cond+=' and c.product_id=:pid';params['pid']=product_id
        if tier:cond+=' and c.label_tier=:tier';params['tier']=tier
        rows=self._rows(f'''select c.candidate_id,c.opinion_id,c.review_id,c.product_id,c.aspect,c.candidate_text,c.source_name,c.label_tier,c.status,
            c.created_at,e.evidence_text from reason_candidates c left join opinion_evidence e on e.opinion_id=c.opinion_id
            where c.status=:st{cond} order by c.created_at,c.candidate_id limit :lim offset :off''',**params)
        grouped=self._rows(f'''select c.candidate_text,count(*) as n,count(distinct c.review_id) as reviews from reason_candidates c
            where c.status=:st{cond} group by c.candidate_text order by n desc limit 50''',**params)
        return {'status':status,'items':rows,'by_text':grouped,
                'policy':'새 이유 후보는 자동으로 온톨로지에 등록하지 않습니다. 사람이 검토해 다음 온톨로지 버전에 반영합니다.'}

    def status(self,model_version=None):
        tiers=self._rows('''select label_tier,source_name,status,count(*) as opinions,count(distinct review_id) as reviews,
            count(distinct product_id) as products from opinions group by 1,2,3 order by 1,2,3''')
        preds=self._rows('''select model_version,status,count(*) as opinions,count(distinct text_hash) as texts from opinion_predictions
            group by 1,2 order by 1,2''')
        cands=self._rows('select status,count(*) as n from reason_candidates group by 1')
        versions=self._rows('select version,sha256,frozen,registered_at from ontology_versions order by registered_at')
        models=self._rows('select model_version,model_kind,validated,test_only,registered_at from model_versions order by registered_at')
        return {'ontology_versions':versions,'current_ontology':{'version':self.ontology.version,'sha256':self.ontology.sha256},
                'annotation_opinions':tiers,'prediction_opinions':preds,'reason_candidates':cands,'model_versions':models,
                'active_model_version':model_version}
