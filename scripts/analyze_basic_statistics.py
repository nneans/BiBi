"""Descriptive review statistics. No model inference or source modification.

Run from repository root: python scripts/analyze_basic_statistics.py
"""
from pathlib import Path
import argparse
import hashlib
import json
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from PIL import Image

from bibi_reviews import paths


RATING_COLS = [f'rating_{i}_count' for i in range(1, 6)]
COLORS = ['#d75445', '#f09645', '#e3c85c', '#79b4d4', '#1687ca']
SEASONS = {12: '겨울', 1: '겨울', 2: '겨울', 3: '봄', 4: '봄', 5: '봄',
           6: '여름', 7: '여름', 8: '여름', 9: '가을', 10: '가을', 11: '가을'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def aggregate(data, keys):
    counts = pd.crosstab([data[k] for k in keys], data.rating).reindex(columns=range(1, 6), fill_value=0)
    counts.columns = RATING_COLS
    counts['review_count'] = counts.sum(axis=1)
    counts['mean_rating'] = counts[RATING_COLS].mul(range(1, 6)).sum(axis=1) / counts.review_count
    counts['low_1_2_count'] = counts.rating_1_count + counts.rating_2_count
    counts['low_1_3_count'] = counts.low_1_2_count + counts.rating_3_count
    counts['low_1_2_pct'] = counts.low_1_2_count / counts.review_count * 100
    counts['low_1_3_pct'] = counts.low_1_3_count / counts.review_count * 100
    return counts


def wilson(count, total):
    z = 1.959963984540054
    p = count / total
    center = (p + z*z/(2*total)) / (1+z*z/total)
    half = z*np.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1+z*z/total)
    return (center-half)*100, (center+half)*100


def export(df, out, name):
    df.to_csv(out / name, index=False, encoding='utf-8-sig')


def labels(frame, width=33):
    return [textwrap.fill(r.product_name, width) + '\n' + str(r.product_id)
            for r in frame.itertuples()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=paths.CANONICAL_REVIEWS)
    parser.add_argument('--reports', type=Path, default=paths.STATISTICS/'basic_statistics'/'2026-10-16')
    parser.add_argument('--figures', type=Path, default=paths.STATISTICS/'figures'/'basic-statistics')
    parser.add_argument('--overwrite', action='store_true', help='Replace generated report files and PNGs in the selected directories')
    args = parser.parse_args()
    for directory in (args.reports, args.figures):
        if directory.exists() and any(directory.iterdir()) and not args.overwrite:
            raise FileExistsError(f'Use an empty output directory: {directory}')
    source_hash = sha(args.input)
    d = pd.read_parquet(args.input, columns=['review_id', 'product_id', 'product_name',
                                            'rating', 'created_at', 'review_type', 'display_status'])
    assert d.review_id.is_unique and d.rating.isin(range(1, 6)).all()
    assert d.display_status.eq('정상').all()
    assert d.groupby('product_id').product_name.nunique().eq(1).all()
    dates = pd.to_datetime(d.created_at)
    if dates.dt.tz is not None:
        dates = dates.dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    assert dates.notna().all()
    d['date'] = dates.dt.normalize()
    d['year_month'] = dates.dt.to_period('M').astype(str)
    d['year'] = dates.dt.year
    d['month'] = dates.dt.month
    first, last = d.date.min(), d.date.max()
    calendar = pd.date_range(first, last)
    missing_dates = calendar.difference(pd.DatetimeIndex(d.date.unique()))
    periods = pd.period_range(first, last, freq='M').astype(str)
    exposure = []
    for ym in periods:
        month = pd.Period(ym, freq='M')
        covered = pd.date_range(max(first, month.start_time.normalize()), min(last, month.end_time.normalize()))
        exposure.append({'year_month': ym, 'calendar_days': month.days_in_month,
                         'covered_calendar_days': len(covered),
                         'complete_calendar_month': len(covered) == month.days_in_month,
                         'dates_with_any_review': int(d.loc[d.year_month.eq(ym), 'date'].nunique()),
                         'dates_without_any_review': int(covered.isin(missing_dates).sum())})
    exposure = pd.DataFrame(exposure).set_index('year_month')

    args.reports.mkdir(parents=True, exist_ok=True)
    args.figures.mkdir(parents=True, exist_ok=True)
    store_mean = float(d.rating.mean())
    store_low = float(d.rating.le(2).mean()*100)
    overall = d.rating.value_counts().reindex(range(1, 6), fill_value=0).rename_axis('rating').rename('review_count').reset_index()
    overall['review_share_pct'] = overall.review_count / len(d)*100
    export(overall, args.reports, 'overall_rating_distribution.csv')

    products = aggregate(d, ['product_id']).join(d.groupby('product_id').product_name.first()).reset_index()
    activity = d.groupby('product_id').date.agg(first_review_date='min', last_review_date='max')
    products = products.merge(activity.reset_index(), on='product_id', validate='one_to_one')
    products['mean_rating_gap_vs_store'] = products.mean_rating-store_mean
    products['low_rate_gap_pp'] = products.low_1_2_pct-store_low
    products['low_rate_ratio_vs_store'] = products.low_1_2_pct/store_low
    products['low_1_2_wilson95_lower_pct'], products['low_1_2_wilson95_upper_pct'] = wilson(products.low_1_2_count, products.review_count)
    products['at_least_100_reviews'] = products.review_count.ge(100)
    products['elevated_low_rating_candidate'] = products.at_least_100_reviews & products.low_1_2_wilson95_lower_pct.gt(store_low)
    products = products.sort_values('review_count', ascending=False).reset_index(drop=True)
    export(products, args.reports, 'product_rating_distribution.csv')
    readable = products[['product_id','product_name','review_count','mean_rating',*RATING_COLS,
                         'low_1_2_count','low_1_2_pct','low_1_3_count','low_1_3_pct',
                         'low_rate_gap_pp','elevated_low_rating_candidate']].rename(columns={
        'product_id':'상품ID','product_name':'대표상품명','review_count':'리뷰수','mean_rating':'평균별점',
        **{f'rating_{i}_count':f'{i}점_리뷰수' for i in range(1,6)},
        'low_1_2_count':'1~2점_리뷰수','low_1_2_pct':'1~2점_비율_pct',
        'low_1_3_count':'1~3점_리뷰수','low_1_3_pct':'1~3점_비율_pct',
        'low_rate_gap_pp':'전체대비_1~2점비율차이_pp',
        'elevated_low_rating_candidate':'저별점_우선확인후보'})
    export(readable, args.reports, '상품별_별점분포.csv')
    candidates = products[products.elevated_low_rating_candidate].sort_values('low_1_2_pct', ascending=False)
    export(candidates, args.reports, 'low_rating_candidates.csv')
    types = aggregate(d, ['product_id', 'review_type']).reset_index()
    export(types, args.reports, 'product_rating_by_review_type.csv')

    panel = aggregate(d, ['product_id', 'year_month']).reindex(
        pd.MultiIndex.from_product([products.product_id, periods], names=['product_id','year_month']))
    countcols = RATING_COLS + ['review_count','low_1_2_count','low_1_3_count']
    panel[countcols] = panel[countcols].fillna(0).astype(int)
    panel = panel.reset_index().merge(exposure.reset_index(), on='year_month', validate='many_to_one')
    panel = panel.merge(products[['product_id','product_name']], on='product_id', validate='many_to_one')
    panel['reviews_per_covered_calendar_day'] = panel.review_count/panel.covered_calendar_days
    panel['year'] = panel.year_month.str[:4].astype(int)
    panel['month'] = panel.year_month.str[5:].astype(int)
    monthly = aggregate(d, ['year_month']).join(exposure).reset_index()
    monthly['reviews_per_covered_calendar_day'] = monthly.review_count/monthly.covered_calendar_days
    export(panel, args.reports, 'product_year_month_statistics.csv')
    export(monthly, args.reports, 'overall_year_month_statistics.csv')

    p25 = panel[panel.year.eq(2025)].copy()
    p25['season'] = p25.month.map(SEASONS)
    totals25 = p25.groupby('product_id').review_count.sum()
    p25['review_share_within_2025_pct'] = p25.review_count/p25.product_id.map(totals25).replace(0,np.nan)*100
    export(p25, args.reports, 'product_monthly_2025.csv')
    seas = p25.groupby(['product_id','product_name','season'], sort=False)[countcols+['covered_calendar_days']].sum().reset_index()
    seas['reviews_per_covered_calendar_day'] = seas.review_count/seas.covered_calendar_days
    seas['review_share_within_2025_pct'] = seas.review_count/seas.product_id.map(totals25).replace(0,np.nan)*100
    seas['mean_rating'] = seas[RATING_COLS].mul(range(1,6)).sum(axis=1)/seas.review_count.replace(0,np.nan)
    export(seas, args.reports, 'product_season_2025.csv')

    # Complete matching months only. 2026-10 is partial and is excluded here.
    common_months = sorted(set(panel.loc[panel.year.eq(2025) & panel.complete_calendar_month,'month']) &
                           set(panel.loc[panel.year.eq(2026) & panel.complete_calendar_month,'month']))
    season_rows = []
    for prod in products.itertuples():
        a = p25[p25.product_id.eq(prod.product_id)].sort_values('month')
        peak = a.loc[a.review_count.idxmax()] if a.review_count.sum() else None
        ss = seas[seas.product_id.eq(prod.product_id)]
        peak_season = ss.loc[ss.reviews_per_covered_calendar_day.idxmax()] if ss.review_count.sum() else None
        c = panel[panel.product_id.eq(prod.product_id) & panel.month.isin(common_months)]
        c25 = c[c.year.eq(2025)].sort_values('month')
        c26 = c[c.year.eq(2026)].sort_values('month')
        v25 = c25.reviews_per_covered_calendar_day.to_numpy()
        v26 = c26.reviews_per_covered_calendar_day.to_numpy()
        eligible = (c25.review_count.sum()>=100 and c26.review_count.sum()>=100 and
                    c25.review_count.gt(0).sum()>=6 and c26.review_count.gt(0).sum()>=6)
        corr = float(np.corrcoef(v25,v26)[0,1]) if eligible and v25.std()>0 and v26.std()>0 else np.nan
        peak25 = int(c25.loc[c25.review_count.idxmax(),'month']) if c25.review_count.sum() else None
        peak26 = int(c26.loc[c26.review_count.idxmax(),'month']) if c26.review_count.sum() else None
        peak25ties = c25.loc[c25.review_count.eq(c25.review_count.max()),'month'].tolist() if peak25 is not None else []
        peak26ties = c26.loc[c26.review_count.eq(c26.review_count.max()),'month'].tolist() if peak26 is not None else []
        gap = abs(peak25-peak26) if peak25 is not None and peak26 is not None else None
        repeated = bool(eligible and np.isfinite(corr) and corr>=.5 and gap<=1 and len(peak25ties)==len(peak26ties)==1)
        season_rows.append({'product_id':prod.product_id,'product_name':prod.product_name,
            'review_count_2025':int(a.review_count.sum()),'months_with_reviews_2025':int(a.review_count.gt(0).sum()),
            'peak_month_2025':int(peak.month) if peak is not None else None,
            'peak_month_ties_2025':','.join(map(str,a.loc[a.review_count.eq(a.review_count.max()),'month'])) if peak is not None else '',
            'peak_month_review_count_2025':int(peak.review_count) if peak is not None else 0,
            'peak_month_share_2025_pct':float(peak.review_share_within_2025_pct) if peak is not None else np.nan,
            'peak_month_vs_mean_month_count_ratio_2025':float(peak.review_count/(a.review_count.sum()/12)) if peak is not None else np.nan,
            'peak_season_by_daily_rate_2025':peak_season.season if peak_season is not None else None,
            'common_months':','.join(map(str,common_months)),
            'review_count_2025_common_months':int(c25.review_count.sum()),
            'review_count_2026_common_months':int(c26.review_count.sum()),
            'peak_month_2025_common_months':peak25,'peak_month_2026_common_months':peak26,
            'enough_data_for_repeat_comparison':bool(eligible),
            'daily_rate_pattern_pearson_r':corr,'repeated_pattern_candidate':repeated})
    season_summary = pd.DataFrame(season_rows)
    export(season_summary, args.reports, 'product_seasonality_summary.csv')
    season_readable = season_summary[['product_id','product_name','review_count_2025','months_with_reviews_2025',
        'peak_month_2025','peak_month_review_count_2025','peak_month_share_2025_pct',
        'peak_season_by_daily_rate_2025','peak_month_2025_common_months','peak_month_2026_common_months',
        'enough_data_for_repeat_comparison','repeated_pattern_candidate']].rename(columns={
        'product_id':'상품ID','product_name':'대표상품명','review_count_2025':'2025년_리뷰수',
        'months_with_reviews_2025':'2025년_리뷰있는월수','peak_month_2025':'2025년_최다리뷰월',
        'peak_month_review_count_2025':'최다월_리뷰수','peak_month_share_2025_pct':'최다월_연간리뷰비중_pct',
        'peak_season_by_daily_rate_2025':'2025년_일평균리뷰_최다계절',
        'peak_month_2025_common_months':'2025년_1~9월중_최다월',
        'peak_month_2026_common_months':'2026년_1~9월중_최다월',
        'enough_data_for_repeat_comparison':'연도비교_표본조건충족',
        'repeated_pattern_candidate':'월별패턴_반복관찰후보'})
    export(season_readable, args.reports, '상품별_월별계절분포.csv')

    font_manager.fontManager.addfont('C:/Windows/Fonts/malgun.ttf')
    plt.rcParams.update({'font.family':'Malgun Gothic','font.size':11,'axes.unicode_minus':False})
    figure_paths = []
    def style(ax, title, grid='x'):
        ax.set_title(title, loc='left', weight='bold', fontsize=17, pad=20)
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#bbbbbb')
        ax.tick_params(length=0,pad=7)
        ax.set_axisbelow(True)
        ax.grid(axis=grid,color='#e9e9e9')
    def save(fig, name):
        fig.tight_layout()
        path=args.figures/name
        fig.savefig(path,dpi=160,bbox_inches='tight',pad_inches=.25,facecolor='white')
        plt.close(fig)
        with Image.open(path) as im: im.verify()
        figure_paths.append({'file':str(path),'sha256':sha(path)})
    numfmt=FuncFormatter(lambda x,pos:f'{x:,.0f}')

    fig,ax=plt.subplots(figsize=(11,5.2))
    ax.bar([f'{i}점' for i in range(1,6)],overall.review_count,color=COLORS,width=.56)
    for i,r in enumerate(overall.itertuples()):
        ax.text(i,r.review_count+10000,f'{r.review_count:,}건\n({r.review_share_pct:.2f}%)',ha='center',va='bottom')
    ax.set_ylim(0,670000);ax.set_ylabel('리뷰 수 (건)');ax.yaxis.set_major_formatter(numfmt)
    style(ax,f'1. 전체 별점 분포 · {len(d):,}건 · 평균 {store_mean:.3f}점','y')
    save(fig,'01-overall-rating-distribution.png')

    bins=[0,10,50,100,500,1000,5000,10000,np.inf]
    cats=['1~10','11~50','51~100','101~500','501~1,000','1,001~5,000','5,001~10,000','10,001 이상']
    hist=pd.cut(products.review_count,bins=bins,labels=cats).value_counts(sort=False)
    fig,ax=plt.subplots(figsize=(12,5))
    ax.barh(cats,hist,color='#1687ca');ax.invert_yaxis()
    for i,n in enumerate(hist): ax.text(n+1,i,f'{n}개 ({n/len(products)*100:.1f}%)',va='center')
    ax.set_xlim(0,hist.max()*1.26);ax.set_xlabel('상품 ID 수 (개)');ax.set_ylabel('상품별 리뷰 수 (건)')
    style(ax,'2. 상품별 리뷰 규모 분포 · 539개 상품')
    save(fig,'02-product-review-volume.png')

    top=products.head(25)
    fig,ax=plt.subplots(figsize=(15,19));left=np.zeros(len(top))
    for i,col in enumerate(RATING_COLS):
        ax.barh(np.arange(len(top)),top[col],left=left,color=COLORS[i],label=f'{i+1}점',height=.65)
        left+=top[col].to_numpy()
    ax.set_yticks(range(len(top)),labels(top));ax.invert_yaxis()
    for i,n in enumerate(top.review_count): ax.text(n+700,i,f'{n:,}',va='center',fontsize=10)
    ax.set_xlim(0,top.review_count.max()*1.15);ax.xaxis.set_major_formatter(numfmt)
    ax.set_xlabel('리뷰 수 (건)');ax.legend(ncol=5,loc='lower right')
    style(ax,'3. 리뷰가 많은 상품 25개 · 별점별 실제 건수')
    save(fig,'03-top-products-rating-counts.png')

    low=products[products.at_least_100_reviews].sort_values('low_1_2_pct',ascending=False).head(20)
    fig,ax=plt.subplots(figsize=(15,11))
    errs=np.vstack([low.low_1_2_pct-low.low_1_2_wilson95_lower_pct,
                    low.low_1_2_wilson95_upper_pct-low.low_1_2_pct])
    ax.errorbar(low.low_1_2_pct,np.arange(len(low)),xerr=errs,fmt='o',color='#d75445',capsize=3)
    ax.set_yticks(range(len(low)),labels(low));ax.invert_yaxis()
    ax.axvline(store_low,color='#1687ca',ls='--',label=f'전체 {store_low:.3f}%')
    xmax=low.low_1_2_wilson95_upper_pct.max()
    for i,r in enumerate(low.itertuples()):
        ax.text(xmax+1,i,f'{r.low_1_2_pct:.2f}% · {r.low_1_2_count:,}/{r.review_count:,}건',va='center',fontsize=10)
    ax.set_xlim(0,xmax+9);ax.set_xlabel('1~2점 비율 (%) · 오차막대: Wilson 95% 구간')
    ax.legend(loc='lower right');style(ax,'4. 1~2점 비율이 높은 상품 · 리뷰 100건 이상')
    save(fig,'04-low-rating-rate-products.png')

    low_n=products.sort_values('low_1_2_count',ascending=False).head(20)
    fig,ax=plt.subplots(figsize=(14,15))
    ax.barh(np.arange(len(low_n)),low_n.low_1_2_count,color='#d75445',height=.6)
    ax.set_yticks(range(len(low_n)),labels(low_n));ax.invert_yaxis()
    for i,r in enumerate(low_n.itertuples()):
        ax.text(r.low_1_2_count+2,i,f'{r.low_1_2_count:,}건 ({r.low_1_2_pct:.2f}%)',va='center',fontsize=10)
    ax.set_xlim(0,low_n.low_1_2_count.max()*1.38);ax.set_xlabel('1~2점 리뷰 수 (건)')
    style(ax,'5. 1~2점 리뷰 건수가 많은 상품 20개')
    save(fig,'05-low-rating-raw-count-products.png')

    fig,ax=plt.subplots(figsize=(14,5.5))
    xx=np.arange(len(monthly))
    ax.bar(xx,monthly.review_count,color=['#1687ca' if x else '#f09645' for x in monthly.complete_calendar_month])
    xt=[r.year_month+(' (1~13일)' if not r.complete_calendar_month else '') for r in monthly.itertuples()]
    ax.set_xticks(xx,xt,rotation=55,ha='right');ax.set_ylabel('리뷰 수 (건)')
    ax.yaxis.set_major_formatter(numfmt);style(ax,'6. 연월별 리뷰 등록 건수','y')
    save(fig,'06-year-month-review-counts.png')

    top25=p25.groupby('product_id').review_count.sum().sort_values(ascending=False).head(30).index
    heat=p25.pivot(index='product_id',columns='month',values='review_share_within_2025_pct').loc[top25]
    lookup=products.set_index('product_id')
    fig,ax=plt.subplots(figsize=(15,25))
    image=ax.imshow(heat.to_numpy(),aspect='auto',cmap='Blues',vmin=0,vmax=40)
    ax.set_xticks(range(12),[f'{i}월' for i in range(1,13)])
    rows=lookup.loc[top25].reset_index()
    ax.set_yticks(range(len(rows)),labels(rows,30))
    ax.tick_params(length=0,pad=7)
    for i in range(len(heat)):
        for j in range(12):
            v=heat.iloc[i,j]
            ax.text(j,i,f'{v:.0f}',ha='center',va='center',fontsize=9,color='white' if v>=23 else '#222222')
    ax.set_title('7. 2025년 상품별 월별 리뷰 비중 · 리뷰 수 상위 30개',loc='left',fontsize=17,weight='bold',pad=20)
    fig.colorbar(image,ax=ax,pad=.02,label='해당 상품의 2025년 리뷰 중 비중 (%)',extend='max')
    save(fig,'07-product-monthly-share-2025.png')

    s_top=top25[:20]
    season_grid=seas.pivot(index='product_id',columns='season',values='review_share_within_2025_pct').loc[s_top,['봄','여름','가을','겨울']]
    fig,ax=plt.subplots(figsize=(14,16));left=np.zeros(len(s_top))
    for season,color in zip(['봄','여름','가을','겨울'],['#78b878','#edaa55','#cd7b64','#6b9ec4']):
        vals=season_grid[season].to_numpy()
        ax.barh(np.arange(len(s_top)),vals,left=left,label=season,color=color,height=.7)
        for i,v in enumerate(vals):
            if v>=8: ax.text(left[i]+v/2,i,f'{v:.0f}%',ha='center',va='center',fontsize=9)
        left+=vals
    ax.set_yticks(range(len(s_top)),labels(lookup.loc[s_top].reset_index()));ax.invert_yaxis()
    ax.set_xlim(0,100);ax.set_xlabel('해당 상품의 2025년 리뷰 중 비중 (%)')
    ax.legend(ncol=4,loc='upper center',bbox_to_anchor=(.5,-.055))
    style(ax,'8. 2025년 계절별 리뷰 비중 · 리뷰 수 상위 20개')
    save(fig,'08-product-season-share-2025.png')

    eligible_patterns=season_summary[season_summary.enough_data_for_repeat_comparison &
        season_summary.review_count_2025.ge(300) & season_summary.months_with_reviews_2025.ge(6)]
    selected=eligible_patterns.sort_values(['repeated_pattern_candidate','peak_month_share_2025_pct'],ascending=False).head(6)
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for ax,r in zip(axes.flat,selected.itertuples()):
        for year,color in [(2025,'#1687ca'),(2026,'#ed8c3c')]:
            vals=panel[panel.product_id.eq(r.product_id) & panel.year.eq(year) & panel.complete_calendar_month].sort_values('month')
            ax.plot(vals.month,vals.reviews_per_covered_calendar_day,marker='o',ms=4,color=color,label=str(year))
        ax.set_xticks(range(1,13));ax.set_xlabel('월');ax.set_ylabel('일평균 리뷰 수 (건)')
        title=textwrap.fill(r.product_name,29)+'\n'+str(r.product_id)
        style(ax,title,'y');ax.title.set_fontsize(11);ax.legend(fontsize=9)
    for ax in list(axes.flat)[len(selected):]: ax.set_visible(False)
    fig.suptitle('9. 월별 리뷰 등록 패턴 · 연도별 비교 (완료된 월)',fontsize=18,weight='bold',y=1.01)
    save(fig,'09-monthly-pattern-year-comparison.png')

    # Independent row-count and weighted-rating reconciliation on all exports.
    assert overall.review_count.sum()==products.review_count.sum()==monthly.review_count.sum()==len(d)
    assert panel.review_count.sum()==types.review_count.sum()==len(d)
    assert products[RATING_COLS].sum().tolist()==overall.review_count.tolist()
    assert np.isclose(products[RATING_COLS].mul(range(1,6)).sum().sum()/len(d),store_mean)
    assert p25.review_count.sum()==seas.review_count.sum()==d.year.eq(2025).sum()
    assert panel.shape[0]==len(products)*len(periods)
    assert np.allclose(p25.groupby('product_id').review_share_within_2025_pct.sum().loc[totals25.gt(0)],100)
    assert panel.loc[panel.review_count.eq(0),'mean_rating'].isna().all()
    assert candidates.review_count.ge(100).all() and candidates.low_1_2_wilson95_lower_pct.gt(store_low).all()
    for filename,expected in [('product_rating_distribution.csv',len(products)),
                              ('product_year_month_statistics.csv',len(panel)),
                              ('product_seasonality_summary.csv',len(products))]:
        loaded=pd.read_csv(args.reports/filename,dtype={'product_id':str})
        assert len(loaded)==expected
        if 'review_count' in loaded.columns: assert loaded.review_count.sum()==len(d)
    assert sha(args.input)==source_hash
    summary={
        'input':str(args.input),'input_sha256':source_hash,'review_count':len(d),
        'product_count':len(products),'date_min':str(dates.min()),'date_max':str(dates.max()),
        'mean_rating':store_mean,'low_1_2_count':int(d.rating.le(2).sum()),'low_1_2_pct':store_low,
        'rating_counts':dict(zip(overall.rating.astype(str),overall.review_count.astype(int))),
        'missing_calendar_dates':[x.strftime('%Y-%m-%d') for x in missing_dates],
        'low_rating_comparison_min_reviews':100,'low_rating_candidate_count':len(candidates),
        'candidates_are_descriptive_not_multiple_testing_corrected':True,
        'seasonality_primary_year':2025,'cross_year_comparison_months':common_months,
        'year_comparison_eligible_products':int(season_summary.enough_data_for_repeat_comparison.sum()),
        'repeated_pattern_candidate_count':int(season_summary.repeated_pattern_candidate.sum()),
        'year_comparison_example_ids':selected.product_id.tolist(),
        'all_rating_counts_and_monthly_counts_reconcile':True,'input_hash_unchanged':True,
        'zero_review_month_mean_is_missing_not_zero':True,'png_files':figure_paths,
        'limitations':[
            'Counts are review records, not unique buyers, orders or sales; general and one-month reviews included.',
            'Low rating means 1 or 2 stars, not text sentiment; 1-3 star counts are also exported.',
            'Global comparator is review-weighted; Wilson intervals are descriptive and ignore linked-review dependence.',
            'Review launch date/product availability unknown. Zeros do not prove product was offered with no demand.',
            '2025-05-31 has no source reviews; true zero versus collection gap unresolved. Calendar denominator retained.',
            '2026-10 ends on day 13 at 16:49:47; daily averages include that partial day. Excluded from repeat comparison.',
            'Only one complete calendar year. Repeated-pattern flags are exploration heuristics, not confirmed seasonality.',
            'Calendar-year winter combines Jan-Feb and Dec 2025, not one continuous winter season.',
            'Most-reviewed month uses raw counts; daily rates and all tied peak months are separately retained.',
        ]}
    (args.reports/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k not in ('png_files','limitations')},ensure_ascii=False,indent=2))
    print('TOP LOW RATING RATES')
    print(low[['product_id','product_name','review_count','mean_rating','low_1_2_count','low_1_2_pct']].head(6).to_string(index=False))
    print('REPEATED PATTERN CANDIDATES')
    print(season_summary[season_summary.repeated_pattern_candidate].sort_values('peak_month_share_2025_pct',ascending=False).head(8).to_string(index=False))


if __name__=='__main__':
    main()
