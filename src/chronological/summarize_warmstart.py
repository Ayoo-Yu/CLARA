"""Prespecified warm-initialization sensitivity on the same target forecasts."""
from forward_common import *
from summarize_chrono import EVENT_METRICS, WINDOW_METRICS, PRICE_IDS, METHODS, paired_inference
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

WARM='LinUCB (pre-period initialization)'

def one(task):
    z,s=task
    selected_path=RUN/'linucb_warm'/z/f'target_seed{s}'/'decisions.npz'
    with np.load(selected_path,allow_pickle=False) as f: choices={k:f[k] for k in f.files}
    rows=[];ti=select_tsc({z})
    for p in PREDICTORS:
        for h in HORIZONS:
            d=load_stream(z,s,p,h,'target',ti); n=len(d['y'])
            actions=choices[f'{p}__H{h:02d}']
            assert actions.shape==(5,11,n)
            for pri,rho in enumerate(PRICES):
                lower=d['lower'][actions[pri],np.arange(11)[:,None],np.arange(n)[None,:]]
                upper=d['upper'][actions[pri],np.arange(11)[:,None],np.arange(n)[None,:]]
                for ci,c in enumerate(COVERAGES):
                    values=evaluation_metrics(lower[ci],upper[ci],d['y'],d['center'],c,rho)
                    rows.append(dict(zone=z,seed=s,predictor=p,horizon=h,target_coverage=c,
                                     price_id=PRICE_IDS[pri],price_ratio=float(rho),method=WARM,
                                     width=float(np.mean(upper[ci]-lower[ci])),**values))
    result=pd.DataFrame(rows)
    assert len(result)==1100
    folder=RUN/'summary/warm_initialization/seed_cells';folder.mkdir(parents=True,exist_ok=True)
    result.to_parquet(folder/f'{z}_seed{s}.parquet',index=False)
    return {'zone':z,'seed':s,'rows':len(result),'decision_sha256':sha256(selected_path)}

def aggregate():
    folder=RUN/'summary/warm_initialization'
    frames=[pd.read_parquet(folder/'seed_cells'/f'{z}_seed{s}.parquet') for z in ZONES for s in SEEDS]
    d=pd.concat(frames,ignore_index=True)
    assert len(d)==33000
    primary=pd.read_parquet(RUN/'summary/all_seed_metrics.parquet')
    keys=['zone','seed','predictor','horizon','target_coverage','price_id']
    matched=d.merge(primary[primary.method.eq('CLARA')][keys+['n','window_count']],on=keys,suffixes=('','_primary'),validate='one_to_one')
    assert np.array_equal(matched.n,matched.n_primary) and np.array_equal(matched.window_count,matched.window_count_primary)
    d.to_parquet(folder/'all_seed_metrics.parquet',index=False)
    cellkeys=['zone','predictor','horizon','target_coverage','price_id','price_ratio','method']
    sums=d[cellkeys+['n','window_count']].copy()
    for m in EVENT_METRICS:sums[m]=d[m]*d.n
    for m in WINDOW_METRICS:sums[m]=d[m]*d.window_count
    cells=sums.groupby(cellkeys,as_index=False).sum()
    for m in EVENT_METRICS:cells[m]/=cells.n
    for m in WINDOW_METRICS:cells[m]/=cells.window_count
    cells.to_parquet(folder/'condition_metrics.parquet',index=False)
    primary_cells=pd.read_parquet(RUN/'summary/condition_metrics.parquet')
    together=pd.concat([primary_cells,cells],ignore_index=True)
    metrics=list(EVENT_METRICS)+list(WINDOW_METRICS)
    together.groupby('method')[metrics].mean().to_csv(folder/'comparison_all_prices.csv',encoding='utf-8-sig')
    together.groupby(['price_id','method'])[metrics].mean().to_csv(folder/'comparison_by_price.csv',encoding='utf-8-sig')
    # The secondary analysis includes both LinUCB versions and adjusts over all
    # nine baseline contrasts within each metric/scope; do not replace primary.
    comparisons=paired_inference(together,(*METHODS,WARM))
    comparisons.to_csv(folder/'paired_zone_inference_all_versions.csv',index=False,encoding='utf-8-sig')
    audit={'status':'PASS','matched_target_records_and_windows':True,'warm_rows':len(d),'warm_conditions':len(cells),
           'reported_versions':['LinUCB',WARM],'secondary_Holm_baselines':9,
           'addendum_sha256':sha256(RUN/'protocol_warmstart_addendum.json'),
           'primary_condition_sha256':sha256(RUN/'summary/condition_metrics.parquet')}
    (folder/'validation.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(together.groupby('method')[['mean_errf','tuwr']].mean().to_string(),flush=True)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--aggregate-only',action='store_true');ap.add_argument('--cells-only',action='store_true')
    args=ap.parse_args()
    if not args.aggregate_only:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fs=[pool.submit(one,(z,s)) for z in ZONES for s in SEEDS]
            for i,f in enumerate(as_completed(fs),1):print(i,30,f.result(),flush=True)
    if not args.cells_only:aggregate()

if __name__=='__main__':main()
