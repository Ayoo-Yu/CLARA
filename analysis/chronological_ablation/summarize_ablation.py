"""Pool chronological ablations using the manuscript's condition/zone weighting."""
from run_ablation import *
OUT=HERE/'summary';OUT.mkdir(exist_ok=True)
KEY=['zone_or_farm','price_id','predictor','horizon_steps','target_coverage','regime','method']
ROLL=['TUWR','TOWR','ARD','under_gap','over_gap']
BOOT=np.random.default_rng(20260911).integers(0,10,(10000,10))
SIGNS=np.array(list(itertools.product([-1.,1.],repeat=10)))

def csv(d,name):d.to_csv(OUT/name,index=False,encoding='utf8',float_format='%.17g')
def pool(frame,keys):
    d=frame.copy()
    for k in M:d[k]*=d.reliability_event_count if k in ROLL else d.event_count
    d=d.groupby(keys,as_index=False,sort=False)[M+['event_count','reliability_event_count']].sum(min_count=1)
    for k in M:d[k]/=d.reliability_event_count if k in ROLL else d.event_count
    d['below_floor']=(d.coverage<d.target_coverage-.02-1e-12).astype(float)
    d.loc[d.coverage.isna(),'below_floor']=np.nan
    return d
def ci(x):return tuple(float(v) for v in np.nanquantile(x,[.025,.975]))
def p_exact(diff):
    d=np.asarray(diff,float);return float(np.mean(abs((SIGNS*d).mean(1))>=abs(d.mean())-1e-14))
def zone_means(d,extra=[]):
    z=d.groupby(['zone_or_farm','price_id',*extra,'method'],as_index=False)[M+['below_floor']].mean()
    a=z.groupby(['zone_or_farm',*extra,'method'],as_index=False)[M+['below_floor']].mean();a['price_id']='ALL'
    return pd.concat([z,a],ignore_index=True)
def comparisons(z,extra=[]):
    rows=[]
    for key,g in z.groupby(['price_id',*extra],sort=False):
        if not isinstance(key,tuple):key=(key,)
        meta=dict(zip(['price_id',*extra],key));base=g[g.method.eq('CLARA')].set_index('zone_or_farm').reindex(fc.ZONES)
        for method,a in g.groupby('method',sort=False):
            a=a.set_index('zone_or_farm').reindex(fc.ZONES)
            row=dict(**meta,method=method,**a[M].mean().to_dict(),below_floor_pct=100*a.below_floor.mean())
            x=a.ERRF.to_numpy();b=base.ERRF.to_numpy()
            row['delta_ERRF_pct']=100*(np.nanmean(x)/np.nanmean(b)-1)
            row['delta_ERRF_pct_CI_low'],row['delta_ERRF_pct_CI_high']=ci(100*(np.nanmean(x[BOOT],axis=1)/np.nanmean(b[BOOT],axis=1)-1))
            for k,label in [('TUWR','TUWR_pp'),('below_floor','below_floor_pp')]:
                delta=a[k].to_numpy()-base[k].to_numpy();row['delta_'+label]=100*np.nanmean(delta)
                row['delta_'+label+'_CI_low'],row['delta_'+label+'_CI_high']=ci(100*np.nanmean(delta[BOOT],axis=1))
            if np.isfinite(x).all() and np.isfinite(b).all():
                row['delta_ERRF_exact_p']=p_exact(x-b)
                row['delta_below_floor_exact_p']=p_exact(a.below_floor-base.below_floor)
            for k in ['capacity','exceedance']:row['delta_'+k+'_contribution_pct']=100*(a[k].mean()-base[k].mean())/base.ERRF.mean()
            rows.append(row)
    return pd.DataFrame(rows)

def main():
    assert read(HERE/'EVALUATION_COMPLETE.json')['status']=='PASS'
    raw=[];screens=[];sub=[];traces=[];receipts={}
    for z in fc.ZONES:
        for s in fc.SEEDS:
            folder=HERE/'units'/f'{z}__seed{s}';receipt=read(folder/'COMPLETE.json')
            accepted_hashes={fc.sha256(Path(__file__).parent/'run_ablation.py'),read(Path(__file__).parent/'source_provenance.json')['run_ablation.py']['source_sha256']}
            assert receipt['status']=='PASS' and receipt['identity']['code'] in accepted_hashes
            for name,hsh in receipt['outputs'].items():assert fc.sha256(folder/name)==hsh
            raw.append(pd.read_parquet(folder/'seed_condition_metrics.parquet'))
            screens.append(pd.read_parquet(folder/'screening_counts.parquet'))
            sub.append(pd.read_parquet(folder/'subset_metrics.parquet'))
            traces.append(pd.read_parquet(folder/'paired_traces.parquet'))
            receipts[folder.name]=fc.sha256(folder/'COMPLETE.json')
    raw=pd.concat(raw,ignore_index=True);assert not raw.duplicated(KEY+['seed']).any()
    raw.to_parquet(OUT/'seed_condition_metrics.parquet',index=False)
    d=pool(raw,KEY);d.to_parquet(OUT/'condition_metrics.parquet',index=False)
    overall=d[d.regime.eq('overall')];assert overall.groupby(['price_id','method']).size().eq(2200).all()
    assert overall.groupby(['price_id','method']).event_count.sum().eq(18530160).all()
    assert overall.groupby(['price_id','method']).reliability_event_count.sum().eq(17427960).all()
    z=zone_means(d,['regime']);csv(z,'zone_means.csv')
    avg=z.groupby(['price_id','regime','method'],as_index=False)[M+['below_floor']].mean();csv(avg,'method_means.csv')
    paired=comparisons(z,['regime']);csv(paired,'paired_summary.csv')
    table5methods=METHODS[:7]
    table5=paired[paired.price_id.eq(PRIMARY)&paired.regime.eq('overall')&paired.method.isin(table5methods)].copy()
    table5['TUWR_pct']=table5.TUWR*100;table5['conditions']=2200
    table5['below_floor_count']=table5.method.map(overall[overall.price_id.eq(PRIMARY)].groupby('method').below_floor.sum())
    table5['order']=table5.method.map({m:i for i,m in enumerate(table5methods)})
    table5=table5.sort_values('order').drop(columns='order');csv(table5,'Table5_reference_price.csv')
    primary=[]
    tab=table5.set_index('method')
    for method,metric in [('GLOBAL_SCORE_SAME_SCREEN','ERRF'),('WIDTH_SAME_SCREEN','ERRF'),('NO_RWG','below_floor'),('NO_SCREEN','below_floor')]:
        r=tab.loc[method];effect='delta_ERRF_pct' if metric=='ERRF' else 'delta_below_floor_pp'
        primary.append(dict(method=method,metric=metric,effect=r[effect],CI_low=r[effect+'_CI_low'],CI_high=r[effect+'_CI_high'],p_unadjusted=r['delta_'+metric+'_exact_p']))
    primary=pd.DataFrame(primary);order=np.argsort(primary.p_unadjusted.to_numpy());adjusted=np.empty(4);running=0.
    for rank,i in enumerate(order):running=max(running,(4-rank)*primary.p_unadjusted.iloc[i]);adjusted[i]=min(1,running)
    primary['p_Holm_four']=adjusted;csv(primary,'primary_contrasts.csv')

    # Source screening rates are target-frequency-weighted within each condition,
    # followed by equal condition and zone weighting, matching all main metrics.
    scr=pd.concat(screens,ignore_index=True);sk=KEY[:-1];sc=[c for c in scr if c not in sk+['seed']]
    scr=scr.groupby(sk,as_index=False)[sc].sum()
    for name,num,den in [('excluded_fraction','excluded_candidate_count','candidate_options'),('empty_fraction','empty_screen_count','event_count'),('exact_empty_fraction','exact_empty_count','event_count'),('backoff_fraction','backoff_count','event_count'),('no_g_selects_full_ineligible_fraction','no_g_selects_full_ineligible_count','event_count')]:scr[name]=scr[num]/scr[den]
    scr['no_g_on_ineligible_coverage']=scr.no_g_full_ineligible_covered_sum/scr.no_g_selects_full_ineligible_count.replace(0,np.nan)
    scr['clara_on_no_g_ineligible_coverage']=scr.clara_on_no_g_full_ineligible_covered_sum/scr.no_g_selects_full_ineligible_count.replace(0,np.nan)
    scr['no_g_on_ineligible_gap']=scr.no_g_on_ineligible_coverage-scr.target_coverage
    scr['clara_on_no_g_ineligible_gap']=scr.clara_on_no_g_ineligible_coverage-scr.target_coverage
    scr.to_parquet(OUT/'screening_condition_counts.parquet',index=False)
    rates=[c for c in scr if c.endswith('_fraction') or c.endswith('_gap')]
    sz=scr.groupby(['zone_or_farm','price_id','regime'],as_index=False)[rates].mean()
    sa=sz.groupby(['zone_or_farm','regime'],as_index=False)[rates].mean();sa['price_id']='ALL';sz=pd.concat([sz,sa],ignore_index=True)
    csv(sz.groupby(['price_id','regime'],as_index=False)[rates].mean(),'screening_summary.csv')
    sub=pool(pd.concat(sub,ignore_index=True),KEY[:5]+['subset','method']);sub.to_parquet(OUT/'subset_condition_metrics.parquet',index=False)
    subz=zone_means(sub,['subset']);csv(comparisons(subz,['subset']),'no_backoff_subset_summary.csv')
    tr=pd.concat(traces,ignore_index=True);tk=KEY;tc=[c for c in tr if c not in tk+['seed']]
    tr=tr.groupby(tk,as_index=False)[tc].sum()
    tr['changed_fraction']=tr.changed_count/tr.event_count
    for k in ['ERRF','capacity','exceedance','covered','width']:tr['delta_'+k+'_contribution']=tr['delta_'+k+'_sum']/tr.event_count
    tr.to_parquet(OUT/'paired_condition_traces.parquet',index=False)
    vals=['changed_fraction']+[c for c in tr if c.endswith('_contribution')]
    tz=tr.groupby(['zone_or_farm','price_id','regime','method'],as_index=False)[vals].mean()
    ta=tz.groupby(['zone_or_farm','regime','method'],as_index=False)[vals].mean();ta['price_id']='ALL'
    csv(pd.concat([tz,ta]).groupby(['price_id','regime','method'],as_index=False)[vals].mean(),'decision_change_summary.csv')

    # Hindsight bounds use all matched target outcomes within the complete state,
    # including coverage and horizon in the group. They are never fitted policies.
    oracle=[];price_rows=[]
    for pid,g in z[z.regime.eq('overall')].groupby('price_id',sort=False):
        pivot=g.pivot(index='zone_or_farm',columns='method',values='ERRF').reindex(fc.ZONES);a=pivot.CLARA.to_numpy()
        for method in ['FEASIBLE_STATE_ORACLE','STATE_ORACLE','OUTCOME_ORACLE']:
            b=pivot[method].to_numpy();low,high=ci(100*(a[BOOT].mean(1)/b[BOOT].mean(1)-1))
            oracle.append(dict(price_id=pid,oracle=method,CLARA_ERRF=float(a.mean()),oracle_ERRF=float(b.mean()),gap_pct=100*(a.mean()/b.mean()-1),ci95_low=low,ci95_high=high))
        b=pivot.FIX_REFERENCE_PRICE.to_numpy();low,high=ci(100*(1-a[BOOT].mean(1)/b[BOOT].mean(1)))
        tw=g.pivot(index='zone_or_farm',columns='method',values='TUWR').reindex(fc.ZONES)
        price_rows.append(dict(price_id=pid,fixed_reference_ERRF=b.mean(),price_reselected_ERRF=a.mean(),reduction_pct=100*(1-a.mean()/b.mean()),ci95_low=low,ci95_high=high,fixed_reference_TUWR=tw.FIX_REFERENCE_PRICE.mean(),price_reselected_TUWR=tw.CLARA.mean(),TUWR_change_pp=100*(tw.CLARA.mean()-tw.FIX_REFERENCE_PRICE.mean())))
    oracle=pd.DataFrame(oracle);csv(oracle,'oracle_gaps.csv');csv(oracle,'TableS8_oracle_gaps.csv')
    csv(avg[avg.method.isin(['CLARA','WIDTH_ONLY','STATE_ORACLE','FEASIBLE_STATE_ORACLE','OUTCOME_ORACLE'])],'oracle_means.csv')
    prices=pd.DataFrame(price_rows);csv(prices,'price_reselection.csv')
    assert abs(prices.loc[prices.price_id.eq(PRIMARY),'reduction_pct'].iloc[0])<1e-10
    assert np.all(oracle.gap_pct>=-1e-8)

    # Verify against the sealed main comparison, independently of replay code.
    reference=pd.read_parquet(CHRONO/'summary/condition_metrics.parquet')
    # The sealed table uses a compact naming convention; inspect and map only.
    mapping={'zone':'zone_or_farm','horizon':'horizon_steps','coverage_target':'target_coverage','mean_errf':'ERRF','mean_capacity_cost':'capacity','mean_exceedance_cost':'exceedance','tuwr':'TUWR','towr':'TOWR','ard':'ARD'}
    ref=reference.rename(columns=mapping)
    key=['zone_or_farm','price_ratio','predictor','horizon_steps','target_coverage']
    a=overall[overall.method.eq('CLARA')].copy();a['price_ratio']=a.price_id.map({p.price_id:p.miss_weight/p.capacity_weight for p in cf.PRICES})
    b=ref[ref.method.eq('CLARA')]
    joined=a.merge(b,on=key,validate='one_to_one',suffixes=('_a','_b'))
    assert len(joined)==11000
    compared=[k for k in ['ERRF','capacity','exceedance','coverage','TUWR','TOWR','ARD','under_gap','over_gap'] if k+'_b' in joined]
    errors={k:float(np.nanmax(abs(joined[k+'_a']-joined[k+'_b']))) for k in compared}
    assert max(errors.values())<1e-10,errors
    report=dict(status='PASS',units=30,methods=METHODS,target_events_per_price_method=18530160,
                coverage_specific_conditions_per_price_method=2200,rolling_windows_per_price_method=17427960,
                pooling='Pool seed sums/counts; equal weight conditions within zone, then zones; average five prices equally.',
                bootstrap=dict(unit='held-out zone',replicates=10000,seed=20260911),
                original_main_results_never_used=True,sealed_chronological_main_metric_errors=errors,
                no_backoff='Use nonempty exact histories regardless of support; retain main source-selected shrinkage, uncertainty scores, and screening. Static only when exact history is empty.',
                oracle='Hindsight only; group by zone,seed,forecaster,horizon,target coverage and complete state; feasible oracle uses historical admissibility including minimum-violation recovery.',
                source_receipts=receipts,code_hashes={n:fc.sha256(Path(__file__).parent/n) for n in ['run_ablation.py','summarize_ablation.py']},
                outputs={p.name:fc.sha256(p) for p in OUT.iterdir() if p.suffix in ['.csv','.parquet']})
    save(OUT/'ANALYSIS_COMPLETE.json',report)
    print(table5[['method','ERRF','delta_ERRF_pct','below_floor_pct','TUWR_pct']].to_string(index=False))
    print(prices.to_string(index=False))
    print(oracle.to_string(index=False))

if __name__=='__main__':main()
