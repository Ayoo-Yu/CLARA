"""Aggregate every prespecified sensitivity setting without selecting target outcomes."""
from sensitivity_common import *
import argparse,datetime

def estimation():
    out=HERE/'summary';out.mkdir(exist_ok=True);figure=HERE/'figure_data';figure.mkdir(exist_ok=True)
    frames=[];receipts=[]
    for z in fc.ZONES:
        for s in fc.SEEDS:
            p=HERE/'estimation'/f'{z}_seed{s}';r=read(p/'COMPLETE.json')
            assert r['status']=='PASS' and r['identity']=={**base_identity(),'script':sha(CODE_DIR/'estimation_screening.py')}
            for n,h in r['outputs'].items():assert sha(p/n)==h
            frames.append(pd.read_parquet(p/'metrics.parquet'));receipts.append(r)
    data=pd.concat(frames,ignore_index=True);assert len(data)==429000
    data.to_parquet(out/'estimation_seed_metrics.parquet',index=False)
    keys=['zone','predictor','horizon','target_coverage','price_id','price_ratio','method']
    weighted=data[keys+['n','window_count']].copy()
    for m in EVENTS:weighted[m]=data[m]*data.n
    for m in WINDOWS:weighted[m]=data[m]*data.window_count
    cells=weighted.groupby(keys,as_index=False).sum()
    for m in EVENTS:cells[m]/=cells.n
    for m in WINDOWS:cells[m]/=cells.window_count
    assert len(cells)==143000
    assert cells.groupby(keys[:-1])[['n','window_count']].nunique().to_numpy().max()==1
    cells.to_parquet(out/'estimation_condition_metrics.parquet',index=False)
    metrics=list(EVENTS)+list(WINDOWS)
    zones=cells.groupby(['zone','price_id','method'],as_index=False)[metrics].mean()
    allzones=zones.groupby(['zone','method'],as_index=False)[metrics].mean();allzones['price_id']='ALL'
    zones=pd.concat([zones,allzones],ignore_index=True);zones.to_csv(out/'estimation_zone_means.csv',index=False)
    means=zones.groupby(['price_id','method'],as_index=False)[metrics].mean()
    means.to_csv(out/'estimation_means.csv',index=False)
    ref=means[means.method.eq('CLARA')].set_index('price_id')
    rows=[]
    for name in ESTIMATION:
        for r in means[means.method.eq(name)].itertuples():
            prefix,value=name.split('_');rows.append(dict(parameter=prefix,value=int(value),price_id=r.price_id,
                ERRF=r.mean_errf,TUWR_pct=100*r.tuwr,ERRF_change_pct=100*(r.mean_errf/ref.loc[r.price_id,'mean_errf']-1)))
    pd.DataFrame(rows).to_csv(figure/'parameter_sensitivity.csv',index=False)
    tol=means[means.method.str.startswith('TOL_')].copy()
    tol['multiplier']=tol.method.str.replace('TOL_','',regex=False).astype(float)
    tol['ERRF']=tol.mean_errf;tol['TUWR_pct']=100*tol.tuwr
    tol.to_csv(figure/'coverage_tolerances_all_prices.csv',index=False)
    tol[tol.price_id.eq('ALL')].sort_values('multiplier').to_csv(figure/'coverage_tolerances.csv',index=False)
    # Pointwise zone-cluster intervals, preserving paired conditional designs.
    ix=np.random.default_rng(20260925).integers(0,10,(10000,10));inference=[]
    zorder=list(fc.ZONES)
    for pid,g in zones.groupby('price_id'):
        for metric in ('mean_errf','tuwr'):
            pivot=g.pivot(index='zone',columns='method',values=metric).reindex(zorder);b=pivot.CLARA.to_numpy()
            for name in pivot.columns:
                if name=='CLARA':continue
                a=pivot[name].to_numpy()
                values=100*(a[ix].mean(axis=1)/b[ix].mean(axis=1)-1) if metric=='mean_errf' else 100*(a-b)[ix].mean(axis=1)
                point=100*(a.mean()/b.mean()-1) if metric=='mean_errf' else 100*(a-b).mean()
                lo,hi=np.quantile(values,[.025,.975])
                inference.append(dict(price_id=pid,method=name,metric=metric,change=point,ci95_low=lo,ci95_high=hi,n_zones=10,bootstrap_draws=10000,bootstrap_seed=20260925))
    intervals=pd.DataFrame(inference);intervals.to_csv(out/'estimation_intervals.csv',index=False)
    archive=[]
    for name in ESTIMATION:
        v=intervals[(intervals.price_id=='ALL')&(intervals.method==name)].set_index('metric')
        cost=v.loc['mean_errf'];risk=v.loc['tuwr']
        label=name.replace('NMIN_','Minimum count = ').replace('NU_','Shrinkage = ')
        archive.append({'Fixed parameter':label,'ERRF change (%) [95% interval]':f'{cost.change:+.4f} [{cost.ci95_low:+.4f}, {cost.ci95_high:+.4f}]',
            'TUWR change (pp) [95% interval]':f'{risk.change:+.4f} [{risk.ci95_low:+.4f}, {risk.ci95_high:+.4f}]'})
    pd.DataFrame(archive).to_csv(out/'Estimation_parameter_sensitivity.csv',index=False)
    tolrows=[]
    for pid in [*PIDS,'ALL']:
        row={'Price ratio':pid}
        for m in (.5,1.,2.):
            v=tol[(tol.price_id==pid)&(tol.multiplier==m)].iloc[0]
            row[f'ERRF {m:g}x']=v.ERRF;row[f'TUWR {m:g}x (%)']=v.TUWR_pct
        tolrows.append(row)
    pd.DataFrame(tolrows).to_csv(out/'Coverage_tolerance_sensitivity.csv',index=False)
    report=dict(status='PASS',fits=30,seed_rows=len(data),pooled_rows=len(cells),source_only=True,
        paired_targets_per_price=int(data[(data.method=='CLARA')&(data.price_id=='R01')].n.sum()),
        all_reference_policies_and_metrics_reproduced=True,settings=list(ESTIMATION),tolerances=[.5,1,2])
    save(out/'estimation_summary_validation.json',report);print(json.dumps(report),flush=True)

def ramp():
    out=HERE/'summary';out.mkdir(exist_ok=True);figure=HERE/'figure_data';figure.mkdir(exist_ok=True)
    outer=[];receipts=[]
    for z in fc.ZONES:
        for s in fc.SEEDS:
            p=HERE/'ramp'/f'{z}_seed{s}';r=read(p/'COMPLETE.json')
            assert r['status']=='PASS' and r['identity']=={**base_identity(),'script':sha(CODE_DIR/'ramp_validation.py')}
            for n,h in r['outputs'].items():assert sha(p/n)==h
            assert r['target_loaded'] is False and r['reference_012_statistics_exact']
            g=pd.read_csv(p/'validation_by_price.csv');g['outer_zone']=z;g['seed']=s;outer.append(g);receipts.append(r)
    data=pd.concat(outer,ignore_index=True);assert len(data)==900
    data.to_csv(out/'threshold_source_validation_by_outer.csv',index=False)
    cols=list(EVENTS)+list(WINDOWS)+['ramp_fraction','backoff_fraction','coverage_deficiency']
    curves=data.groupby(['tau','price_id'],as_index=False)[cols].mean()
    pooled=curves.groupby('tau',as_index=False)[cols].mean();pooled['price_id']='ALL'
    curves=pd.concat([curves,pooled],ignore_index=True)
    curves['ERRF']=curves.mean_errf;curves['TUWR_pct']=100*curves.tuwr
    curves['flagged_fraction']=curves.ramp_fraction
    curves.to_csv(figure/'threshold_all_prices.csv',index=False)
    plotted=curves[curves.price_id.eq('ALL')].sort_values('tau')
    plotted.to_csv(figure/'threshold_sensitivity.csv',index=False)
    archive=plotted[['tau','ERRF','TUWR_pct','ramp_fraction','backoff_fraction']].copy()
    archive.ramp_fraction*=100;archive.backoff_fraction*=100
    archive.columns=['Threshold (p.u.)','ERRF','TUWR (%)','Flagged (%)','Backoff (%)']
    archive.to_csv(out/'Power_threshold_source_validation.csv',index=False)
    report=dict(status='PASS',outer_seed_fits=30,inner_fits=90,thresholds=list(TAUS),
        validation_metric_rows=sum(r['metric_rows'] for r in receipts),target_data_loaded=False,
        source_cost_minimum_tau=float(plotted.loc[plotted.ERRF.idxmin(),'tau']),
        source_TUWR_minimum_tau=float(plotted.loc[plotted.TUWR_pct.idxmin(),'tau']),
        source_cost_range_percent=float(100*(plotted.ERRF.max()/plotted.ERRF.min()-1)),
        main_threshold_unchanged=.12,source_curve_points_not_independent_replicates=True)
    save(out/'ramp_summary_validation.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['estimation','ramp','all']);a=p.parse_args()
    if a.stage in ('estimation','all'):estimation()
    if a.stage in ('ramp','all'):ramp()
