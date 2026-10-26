"""Product statistics for the dashboard. One model version per query; denominators are explicit.

Review-level counting: a review counts once per aspect/sentiment however often it repeats the
opinion. A review can carry several aspects, so aspect shares do not sum to 100%.
"""
from collections import defaultdict
from datetime import date,timedelta

from sqlalchemy import text

from ..review_pipeline.common import plain_row


def iso_week(day):
    """'YYYY-Www' ISO week of a 'YYYY-MM-DD' string."""
    y,w,_=date.fromisoformat(day[:10]).isocalendar()
    return f'{y}-W{w:02d}'


def roll_up_weeks(rows,count_keys,group_keys=()):
    """Sum daily rows (with a 'day' column) into ISO weeks, keeping group_keys separate."""
    out=defaultdict(lambda:defaultdict(int))
    for r in rows:
        key=(iso_week(r['day']),*(r[k] for k in group_keys))
        for k in count_keys:out[key][k]+=r[k]
    return [{'period':k[0],**dict(zip(group_keys,k[1:])),**v} for k,v in sorted(out.items())]


def _period(start,end,column='r.created_at'):
    clauses=[];params={}
    if start:clauses.append(f'{column}>=:start');params['start']=start
    if end:clauses.append(f'{column}<:end');params['end']=end
    return (' and '+' and '.join(clauses)) if clauses else '',params


class Analytics:
    def __init__(self,store,astore):
        self.store=store;self.astore=astore;self.engine=store.engine

    def _rows(self,sql,**params):
        with self.engine.connect() as c:return [plain_row(r) for r in c.execute(text(sql),params).mappings()]

    def product(self,product_id):
        rows=self._rows('select product_id,product_name from products where product_id=:p',p=product_id)
        return rows[0] if rows else None

    def overview(self,product_id,start=None,end=None):
        where,params=_period(start,end)
        base=f"from reviews r where r.product_id=:p and r.display_status='정상'{where}"
        totals=self._rows(f'select count(*) as review_count,avg(r.rating) as mean_rating '+base,p=product_id,**params)[0]
        ratings={str(r['rating']):r['n'] for r in self._rows(f'select r.rating,count(*) as n {base} group by r.rating order by r.rating',p=product_id,**params)}
        monthly=self._rows(f'select substr(r.created_at,1,7) as period,count(*) as reviews,avg(r.rating) as mean_rating {base} group by 1 order by 1',p=product_id,**params)
        weekly=roll_up_weeks(self._rows(f"select substr(r.created_at,1,10) as day,count(*) as reviews {base} group by 1 order by 1",p=product_id,**params),['reviews'])
        blind=self._rows("select count(*) as n from reviews where product_id=:p and display_status='블라인드'",p=product_id)[0]['n']
        return {**self.product(product_id),'period':{'start':start,'end':end,'timezone':'Asia/Seoul','end_exclusive':True},
                'review_count':totals['review_count'],'mean_rating':totals['mean_rating'],
                'rating_counts':{k:ratings.get(k,0) for k in '12345'},'blind_reviews_excluded':blind,
                'monthly':monthly,'weekly':weekly,'source':'uploaded CSV data (not AI)'}

    def coverage(self,product_id,version,start=None,end=None):
        where,params=_period(start,end)
        row=self._rows(f'''select count(*) as normal_reviews,
            sum(case when s.status is not null then 1 else 0 end) as analyzed,
            sum(case when s.status='unclassified' then 1 else 0 end) as unclassified,
            sum(case when s.status='review_required' then 1 else 0 end) as review_required,
            sum(case when s.status is null and f.text_hash is not null then 1 else 0 end) as failed
            from reviews r left join analysis_status s on s.text_hash=r.text_hash and s.model_version=:v
            left join analysis_failures f on f.text_hash=r.text_hash and f.model_version=:v
            where r.product_id=:p and r.display_status='정상'{where}''',p=product_id,v=version,**params)[0]
        row={k:int(v or 0) for k,v in row.items()}
        row['pending']=row['normal_reviews']-row['analyzed']-row['failed']
        return row

    def aspects(self,product_id,version,support,start=None,end=None):
        where,params=_period(start,end)
        cov=self.coverage(product_id,version,start,end)
        counts=self._rows(f'''select l.aspect,l.sentiment,count(*) as reviews from reviews r
            join analysis_labels l on l.text_hash=r.text_hash and l.model_version=:v
            where r.product_id=:p and r.display_status='정상'{where} group by 1,2''',p=product_id,v=version,**params)
        mixed=self._rows(f'''select a.aspect,count(*) as reviews from reviews r
            join analysis_labels a on a.text_hash=r.text_hash and a.model_version=:v and a.sentiment='positive'
            join analysis_labels b on b.text_hash=r.text_hash and b.model_version=:v and b.sentiment='negative' and b.aspect=a.aspect
            where r.product_id=:p and r.display_status='정상'{where} group by 1''',p=product_id,v=version,**params)
        idx={(c['aspect'],c['sentiment']):c['reviews'] for c in counts};mix={m['aspect']:m['reviews'] for m in mixed}
        n=cov['analyzed'];out=[]
        for aspect in dict.fromkeys(k.split(':')[0] for k in support):
            ps,ns=support[aspect+':positive'],support[aspect+':negative']
            pos=idx.get((aspect,'positive'),0) if ps else None;neg=idx.get((aspect,'negative'),0) if ns else None
            m=mix.get(aspect,0) if ps and ns else None;mentioned=pos+neg-m if ps and ns else None
            out.append({'aspect':aspect,'positive_reviews':pos,'negative_reviews':neg,'mixed_reviews':m,'mentioned_reviews':mentioned,
                        'positive_share_of_analyzed':pos/n if n and ps else None,'negative_share_of_analyzed':neg/n if n and ns else None,
                        'negative_share_of_mentioned':neg/mentioned if mentioned else None,
                        'model_supported':{'positive':ps,'negative':ns}})
        return {'coverage':cov,'aspects':out,
                'denominators':{'share_of_analyzed':'analyzed normal reviews of this product in the period (coverage.analyzed)',
                                'share_of_mentioned':'reviews with a positive or negative prediction for that aspect'},
                'counting':'distinct reviews; shares across aspects do not sum to 100%'}

    def trend(self,product_id,version,granularity='month',aspect=None,start=None,end=None):
        where,params=_period(start,end)
        # weeks are ISO weeks rolled up in Python from daily counts, so both database engines agree
        bucket='substr(r.created_at,1,7)' if granularity=='month' else 'substr(r.created_at,1,10)'
        analyzed=self._rows(f'''select {bucket} as period,count(*) as analyzed from reviews r join analysis_status s on s.text_hash=r.text_hash
            and s.model_version=:v where r.product_id=:p and r.display_status='정상'{where} group by 1 order by 1''',p=product_id,v=version,**params)
        extra=' and l.aspect=:a' if aspect else ''
        labels=self._rows(f'''select {bucket} as period,l.aspect,l.sentiment,count(*) as reviews from reviews r
            join analysis_labels l on l.text_hash=r.text_hash and l.model_version=:v
            where r.product_id=:p and r.display_status='정상'{where}{extra} group by 1,2,3 order by 1''',p=product_id,v=version,a=aspect,**params)
        if granularity!='month':
            analyzed=roll_up_weeks([{**r,'day':r['period']} for r in analyzed],['analyzed'])
            labels=roll_up_weeks([{**r,'day':r['period']} for r in labels],['reviews'],('aspect','sentiment'))
        series={}
        for row in analyzed:series[row['period']]={'period':row['period'],'analyzed':row['analyzed'],'labels':{}}
        for row in labels:
            s=series[row['period']]['labels'];s[f"{row['aspect']}:{row['sentiment']}"]={'reviews':row['reviews'],
                'share_of_analyzed':row['reviews']/series[row['period']]['analyzed']}
        return {'granularity':granularity,'series':list(series.values()),'single_model_version':version}

    def emerging(self,product_id,version,support,as_of,recent_days=28,baseline_days=84,min_reviews=5,ratio=2.0):
        """Negative aspects whose recent share at least doubled vs the baseline window. Candidates only."""
        day=date.fromisoformat(as_of[:10])
        def window(back_from,back_to):
            # [as_of - back_from, as_of - back_to) as ISO date strings; compared as text on both engines
            lo,hi=(day-timedelta(days=back_from)).isoformat(),(day-timedelta(days=back_to)).isoformat()
            n=self._rows('''select count(*) as n from reviews r join analysis_status s on s.text_hash=r.text_hash and s.model_version=:v
                where r.product_id=:p and r.display_status='정상' and substr(r.created_at,1,10)>=:lo
                and substr(r.created_at,1,10)<:hi''',p=product_id,v=version,lo=lo,hi=hi)[0]['n']
            neg={x['aspect']:x['n'] for x in self._rows('''select l.aspect,count(*) as n from reviews r
                join analysis_labels l on l.text_hash=r.text_hash and l.model_version=:v and l.sentiment='negative'
                where r.product_id=:p and r.display_status='정상' and substr(r.created_at,1,10)>=:lo
                and substr(r.created_at,1,10)<:hi group by 1''',p=product_id,v=version,lo=lo,hi=hi)}
            return n,neg
        rn,rneg=window(recent_days,0);bn,bneg=window(recent_days+baseline_days,recent_days)
        out=[]
        for key,ok in support.items():
            aspect,sentiment=key.split(':')
            if sentiment!='negative' or not ok:continue
            r=rneg.get(aspect,0);b=bneg.get(aspect,0);rs=r/rn if rn else None;bs=b/bn if bn else None
            flagged=bool(rs is not None and r>=min_reviews and (bs in (None,0) or rs>=ratio*bs))
            out.append({'aspect':aspect,'recent_negative_reviews':r,'recent_analyzed':rn,'recent_share':rs,
                        'baseline_negative_reviews':b,'baseline_analyzed':bn,'baseline_share':bs,'increase_candidate':flagged})
        return {'as_of':as_of,'recent_days':recent_days,'baseline_days':baseline_days,'rule':f'recent >= {min_reviews} reviews and share >= {ratio}x baseline',
                'status':'candidate signal; verify with the review list before acting','aspects':out}

    def reviews(self,product_id,version,aspect=None,sentiment=None,rating=None,start=None,end=None,status=None,limit=50,offset=0):
        where,params=_period(start,end);join='left join'
        cond=''
        if aspect or sentiment:
            join='join';cond+=(' and l.aspect=:aspect' if aspect else '')+(' and l.sentiment=:sentiment' if sentiment else '')
        if rating:cond+=' and r.rating=:rating'
        if status:cond+=' and s.status=:status'
        rows=self._rows(f'''select distinct r.review_id,r.product_id,r.rating,r.created_at,r.review_type,r.review_text,r.text_hash,s.status as analysis_status,
            q.status as queue_status from reviews r
            left join analysis_status s on s.text_hash=r.text_hash and s.model_version=:v
            left join review_queue q on q.text_hash=r.text_hash and q.model_version=:v
            {join} analysis_labels l on l.text_hash=r.text_hash and l.model_version=:v
            where r.product_id=:p and r.display_status='정상'{where}{cond}
            order by r.created_at desc,r.review_id limit :limit offset :offset''',p=product_id,v=version,aspect=aspect,
            sentiment=sentiment,rating=rating,status=status,limit=limit,offset=offset,**params)
        for row in rows:row['predictions']=self.labels_for(row.pop('text_hash'),version)
        return rows

    def labels_for(self,text_hash,version):
        import json
        rows=self._rows('''select aspect,sentiment,score,threshold,evidence_text,evidence_start,evidence_end,reason_candidates_json
            from analysis_labels where text_hash=:h and model_version=:v order by aspect,sentiment''',h=text_hash,v=version)
        for r in rows:
            r['reason_candidates']=json.loads(r.pop('reason_candidates_json') or '[]');r['reason_status']='unverified_candidate'
        return rows

    def review(self,review_id,version):
        rows=self._rows('''select r.review_id,r.product_id,p.product_name,r.rating,r.created_at,r.source_updated_at,r.review_type,
            r.related_review_id,r.review_text,r.text_hash,s.status as analysis_status from reviews r join products p using(product_id)
            left join analysis_status s on s.text_hash=r.text_hash and s.model_version=:v
            where r.review_id=:id and r.display_status='정상' ''',id=review_id,v=version)
        if not rows:return None
        row=rows[0];row['predictions']=self.labels_for(row.pop('text_hash'),version) if version else [];return row

    def processing(self,version,limit=20):
        uploads=self._rows('''select u.job_id,u.filename,u.status,u.created_at,u.finished_at,u.counts_json,u.error,
            a.status as analysis_status,a.model_version,a.details_json as analysis_details from upload_jobs u
            left join upload_analysis a on a.upload_id=u.job_id order by u.created_at desc limit :n''',n=limit)
        totals=self._rows('''select count(*) as normal_reviews,count(distinct r.text_hash) as distinct_texts,
            sum(case when s.status is not null then 1 else 0 end) as analyzed_reviews,
            sum(case when s.status='review_required' then 1 else 0 end) as review_required_reviews,
            sum(case when s.status='unclassified' then 1 else 0 end) as unclassified_reviews,
            sum(case when s.status is null and f.text_hash is not null then 1 else 0 end) as failed_reviews
            from reviews r left join analysis_status s on s.text_hash=r.text_hash and s.model_version=:v
            left join analysis_failures f on f.text_hash=r.text_hash and f.model_version=:v where r.display_status='정상' ''',v=version or '')[0]
        totals={k:int(v or 0) for k,v in totals.items()}
        totals['pending_reviews']=totals['normal_reviews']-totals['analyzed_reviews']-totals['failed_reviews']
        queue=self._rows("select count(*) as n from review_queue where status='open' and model_version=:v",v=version or '')[0]['n']
        return {'model_version':version,'totals':totals,'open_review_queue':queue,'uploads':uploads}

    def queue(self,version,product_id=None,limit=50,offset=0):
        cond=' and r.product_id=:p' if product_id else ''
        rows=self._rows(f'''select r.review_id,r.product_id,r.rating,r.created_at,r.review_text,q.reason,q.created_at as queued_at,r.text_hash
            from review_queue q join reviews r on r.text_hash=q.text_hash and r.display_status='정상'
            where q.status='open' and q.model_version=:v{cond} order by q.created_at,r.review_id limit :limit offset :offset''',
            v=version,p=product_id,limit=limit,offset=offset)
        for row in rows:row['predictions']=self.labels_for(row.pop('text_hash'),version)
        return rows
