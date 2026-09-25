"""Recompute all metrics from matched post-cutoff endpoints and decisions.

No published full-period averages are used as a matched-period comparator.
The input manifest and the exact per-fold files are hashed in the output audit.
"""
from __future__ import annotations
from forward_common import *
import argparse, datetime, itertools, time
from concurrent.futures import ProcessPoolExecutor, as_completed

METHODS=('Static','ACI','AgACI','EnbPI-RH','TSC','EEE','CLARA','CART','LinUCB')
PRICE_IDS=('R01','R02','Rref','R10','R20')
EVENT_METRICS=('mean_errf','mean_capacity_cost','mean_exceedance_cost','coverage','width','interval_score')
WINDOW_METRICS=('tuwr','towr','ard','under_gap','over_gap')

def selector_files(z,s):
    return {'CLARA':RUN/'clara'/z/f'seed{s}'/'decisions.npz',
            'CART':RUN/'cart_results'/z/f'target_seed{s}'/'decisions.npz',
            'LinUCB':RUN/'linucb'/z/f'target_seed{s}'/'decisions.npz'}

def one(task):
    z,s=task
    dest=RUN/'summary/seed_cells'/f'{z}_seed{s}.parquet'
    dest.parent.mkdir(parents=True,exist_ok=True)
    selpaths=selector_files(z,s)
    identity={m:sha256(p) for m,p in selpaths.items()}
    identity['summary_code']=sha256(Path(__file__))
    identity['metrics_and_loading_code']=sha256(Path(__file__).resolve().parent / 'forward_common.py')
    identity['validated_dataset']=sha256(RUN/'DATA_READY.json')
    identity['TSC_selection_scores']=sha256(RUN/'tsc_zone_scores.parquet')
    identity['target_datasets']={f'{p}-H{h:02d}':sha256(dataset_path(z,s,p,h,'target'))
                                 for p in PREDICTORS for h in HORIZONS}
    receipt_path=dest.with_suffix('.json')
    if dest.exists() and receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if receipt.get('inputs')==identity and receipt.get('sha256')==sha256(dest):
            return receipt
    sels={}
    for method,path in selpaths.items():
        with np.load(path,allow_pickle=False) as f: sels[method]={k:f[k] for k in f.files}
    ti=select_tsc({z}); rows=[]; timestamps=[]
    for p in PREDICTORS:
        for h in HORIZONS:
            d=load_stream(z,s,p,h,'target',ti); n=len(d['y']); key=f'{p}__H{h:02d}'
            assert n>=168 and np.all(d['issue_ns']>=1378339200000000000)
            timestamps.append({'predictor':p,'horizon':h,'n':n,'first_issue_ns':int(d['issue_ns'][0]),'last_issue_ns':int(d['issue_ns'][-1])})
            c_index=np.arange(11)[:,None]; t_index=np.arange(n)[None,:]
            for method in METHODS:
                if method in sels:
                    choice=sels[method][key]
                    assert choice.shape==(5,11,n) and np.all((choice>=0)&(choice<6))
                for pri,rho in enumerate(PRICES):
                    if method in sels:
                        lo=d['lower'][choice[pri],c_index,t_index]
                        up=d['upper'][choice[pri],c_index,t_index]
                    else:
                        ai=METHODS.index(method); lo=d['lower'][ai]; up=d['upper'][ai]
                    for ci,c in enumerate(COVERAGES):
                        vals=evaluation_metrics(lo[ci],up[ci],d['y'],d['center'],c,rho)
                        rows.append(dict(zone=z,seed=s,predictor=p,horizon=h,target_coverage=c,
                                         price_id=PRICE_IDS[pri],price_ratio=float(rho),method=method,
                                         width=float((up[ci]-lo[ci]).mean()),**vals))
    frame=pd.DataFrame(rows)
    assert len(frame)==9900
    assert frame.groupby(['predictor','horizon','target_coverage','price_id'])[['n','window_count']].nunique().to_numpy().max()==1
    np.testing.assert_allclose(frame.mean_errf,frame.mean_capacity_cost+frame.mean_exceedance_cost,atol=1e-12)
    np.testing.assert_allclose(frame.ard,frame.under_gap+frame.over_gap,atol=1e-12)
    assert np.isfinite(frame[list(EVENT_METRICS)+list(WINDOW_METRICS)]).all().all()
    frame.to_parquet(dest,index=False)
    receipt={'zone':z,'seed':s,'rows':len(frame),'inputs':identity,'sha256':sha256(dest),'target_streams':timestamps,
             'events_per_method_price':int(frame[(frame.method=='CLARA')&(frame.price_id=='R01')].n.sum()),'status':'PASS'}
    receipt_path.write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    return receipt

def paired_inference(cells,methods=METHODS):
    """Fixed-seed paired zone bootstrap, exact sign flips, separate metric families."""
    rng=np.random.default_rng(20260925)
    samples=rng.integers(0,10,size=(10000,10))
    signs=np.array(list(itertools.product([-1.,1.],repeat=10)))
    out=[]
    for scope in ('ALL',*PRICE_IDS):
        d=cells if scope=='ALL' else cells[cells.price_id.eq(scope)]
        means=d.groupby(['zone','method'])[['mean_errf','tuwr']].mean()
        for metric in ('mean_errf','tuwr'):
            tab=means[metric].unstack('method').reindex(ZONES)
            for baseline in methods:
                if baseline=='CLARA':continue
                a=tab.CLARA.to_numpy(); b=tab[baseline].to_numpy(); dif=a-b
                draws=dif[samples].mean(axis=1)
                q=np.quantile(draws,[.025,.975])
                p=float(np.mean(np.abs(signs@dif/10)>=abs(dif.mean())-1e-14))
                r=dict(scope=scope,metric=metric,baseline=baseline,clara_mean=float(a.mean()),baseline_mean=float(b.mean()),
                       difference=float(dif.mean()),ci_low=float(q[0]),ci_high=float(q[1]),sign_flip_p=p,
                       n_zones=10,bootstrap_draws=10000,bootstrap_seed=20260925)
                if metric=='mean_errf':
                    relative=100*(a[samples].mean(axis=1)/b[samples].mean(axis=1)-1)
                    rq=np.quantile(relative,[.025,.975])
                    r.update(relative_difference_pct=float(100*(a.mean()/b.mean()-1)),relative_ci_low_pct=float(rq[0]),relative_ci_high_pct=float(rq[1]))
                out.append(r)
    frame=pd.DataFrame(out)
    frame['holm_p']=np.nan
    for _,ids in frame.groupby(['scope','metric']).groups.items():
        index=np.asarray(list(ids)); order=np.argsort(frame.loc[index,'sign_flip_p'].to_numpy(),kind='stable')
        ordered=index[order]; ps=frame.loc[ordered,'sign_flip_p'].to_numpy()
        frame.loc[ordered,'holm_p']=np.minimum(1,np.maximum.accumulate(ps*np.arange(len(ps),0,-1)))
    return frame

def aggregate():
    folder=RUN/'summary'
    paths=[folder/'seed_cells'/f'{z}_seed{s}.parquet' for z in ZONES for s in SEEDS]
    d=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    assert len(d)==297000 and len(d.groupby(['zone','seed']))==30
    d.to_parquet(folder/'all_seed_metrics.parquet',index=False)
    keys=['zone','predictor','horizon','target_coverage','price_id','price_ratio','method']
    weighted=d[keys+['n','window_count']].copy()
    for m in EVENT_METRICS: weighted[m]=d[m]*d.n
    for m in WINDOW_METRICS: weighted[m]=d[m]*d.window_count
    cells=weighted.groupby(keys,as_index=False).sum()
    for m in EVENT_METRICS: cells[m]/=cells.n
    for m in WINDOW_METRICS: cells[m]/=cells.window_count
    assert len(cells)==99000
    cells.to_parquet(folder/'condition_metrics.parquet',index=False)
    grouping=['zone','predictor','horizon','target_coverage','price_id']
    assert cells.groupby(grouping)[['n','window_count']].nunique().to_numpy().max()==1
    cells['economic_rank']=cells.groupby(grouping).mean_errf.rank(method='average')
    minimum=cells.groupby(grouping).mean_errf.transform('min')
    cells['relative_excess_pct']=100*(cells.mean_errf/minimum-1)
    cells.to_parquet(folder/'ranked_condition_metrics.parquet',index=False)
    reportcols=list(EVENT_METRICS)+list(WINDOW_METRICS)+['economic_rank','relative_excess_pct']
    overall=cells.groupby('method')[reportcols].mean().sort_values('mean_errf').reset_index()
    overall.to_csv(folder/'overall_results.csv',index=False,encoding='utf-8-sig')
    for field,name in [('price_id','by_price'),('horizon','by_horizon'),('zone','by_zone'),('target_coverage','by_coverage'),('predictor','by_predictor')]:
        tab=cells.groupby([field,'method'],as_index=False)[reportcols].mean()
        tab.to_csv(folder/f'{name}.csv',index=False,encoding='utf-8-sig')
    inference=paired_inference(cells)
    inference.to_csv(folder/'paired_zone_inference.csv',index=False,encoding='utf-8-sig')
    counts=d.groupby(['method','price_id']).n.sum()
    assert counts.nunique()==1
    validation=dict(status='PASS',methods=9,price_scenarios=5,matched_events_per_method_price=int(counts.iloc[0]),
                    main_comparison_conditions=11000,conditions_by_price=2200,
                    source_fit_cutoff_utc='2013-09-05T00:00:00Z',evaluation_window='Same post-cutoff suffix for all methods',
                    metric_units={'ERRF':'CNY per hourly forecast on a 1 MW power base','coverage_and_window_rates':'fractions'},
                    inference='Paired bootstrap of 10 zones, 10000 replicates; exact 1024 sign flips; Holm within 8-baseline metric/scope family.',
                    seed_cells_sha256={p.name:sha256(p) for p in paths},protocol_sha256=sha256(RUN/'protocol.json'))
    (folder/'summary_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    print(overall[['method','mean_errf','tuwr','coverage','ard']].to_string(index=False),flush=True)
    print(json.dumps({k:v for k,v in validation.items() if k!='seed_cells_sha256'},indent=2),flush=True)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=3)
    ap.add_argument('--aggregate-only',action='store_true'); ap.add_argument('--outer'); ap.add_argument('--seed',type=int)
    args=ap.parse_args()
    if not (RUN/'DATA_READY.json').exists():raise RuntimeError('Validated candidate data required')
    if not args.aggregate_only:
        tasks=[(z,s) for z in ([args.outer] if args.outer else ZONES) for s in ([args.seed] if args.seed is not None else SEEDS)]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures=[pool.submit(one,t) for t in tasks]
            for i,f in enumerate(as_completed(futures),1):
                r=f.result(); print(i,len(tasks),r['zone'],r['seed'],r['events_per_method_price'],flush=True)
    if args.outer is None and args.seed is None:aggregate()

if __name__=='__main__':main()
