"""Recompute commercial evaluation metrics from released candidate intervals.

Run with --archive pointing to the extracted commercial_candidate_archive folder.
This evaluates saved decision paths; it does not refit the forecasting models.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ACTIONS=['Static','ACI','AgACI','EnbPI-RH','TSC','EEE']
CHOICE_METHODS=['CLARA_frozen','CLARA','CLARA_cost_update','CLARA_coverage_update','CART','LinUCB']
METHODS=ACTIONS+CHOICE_METHODS
METRICS=['ERRF','capacity','exceedance','coverage','width','TUWR','TOWR','ARD','under_gap','over_gap']
CONDITIONS=['farm_id','price_id','predictor','horizon_steps','target_coverage','regime','method']

def write_frame(p,frame):
    p.parent.mkdir(parents=True,exist_ok=True)
    table=pa.Table.from_pandas(frame,preserve_index=False).replace_schema_metadata(None)
    pq.write_table(table,p,compression='zstd',compression_level=9)

def evaluate_shard(folder):
    manifest=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    pricing=json.loads((folder/'price_configs.json').read_text(encoding='utf-8'))['prices']
    decisions={}
    for price in pricing:
        with np.load(folder/'choices'/(price['price_id']+'.npz'),allow_pickle=False) as z:
            decisions[price['price_id']]=z['selected'].copy()
        assert decisions[price['price_id']].shape==(manifest['counts']['test'],6)
    rows=[]
    for predictor in manifest['predictors']:
        for horizon in manifest['horizon_steps']:
            stream=folder/'streams'/f'{predictor}_h{horizon:02d}'
            f=pd.read_parquet(stream/'base_test.parquet').sort_values(['issue_tick','target_coverage'],kind='mergesort').reset_index(drop=True)
            tsc={}
            for key in {p['tsc_configuration'] for p in pricing}:
                q=pd.read_parquet(stream/f'{key}_test.parquet').set_index('row_id').reindex(f.row_id)
                assert q.notna().all().all()
                tsc[key]=q[['lower','upper']].to_numpy(np.float64)
            y=f.target.to_numpy(np.float64)[:,None]
            center=f.base_center.to_numpy(np.float64)[:,None]
            for price in pricing:
                lo=np.column_stack([tsc[price['tsc_configuration']][:,0] if a=='TSC' else f[a+'__lower'].to_numpy(np.float64) for a in ACTIONS])
                hi=np.column_stack([tsc[price['tsc_configuration']][:,1] if a=='TSC' else f[a+'__upper'].to_numpy(np.float64) for a in ACTIONS])
                selected=np.column_stack([np.broadcast_to(np.arange(6,dtype=np.int8),(len(f),6)),decisions[price['price_id']][f.replay_order.to_numpy()]])
                lower=np.take_along_axis(lo,selected,axis=1)
                upper=np.take_along_axis(hi,selected,axis=1)
                pi=float(price['capacity_weight']);kappa=float(price['exceedance_weight'])
                capacity_up=.25*np.maximum(upper-center,0)*pi
                capacity_down=.25*np.maximum(center-lower,0)*pi
                exceed_up=.25*np.maximum(y-upper,0)*kappa
                exceed_down=.25*np.maximum(lower-y,0)*kappa
                errf=capacity_up+capacity_down+exceed_up+exceed_down
                covered=(lower<=y)&(y<=upper)
                for coverage,indices in f.groupby('target_coverage',sort=True).indices.items():
                    indices=np.asarray(indices)
                    ticks=f.issue_tick.to_numpy()[indices]
                    assert (np.diff(ticks)==1).all(), 'Incomplete chronological condition stream'
                    n=len(indices);nr=n-671;assert nr>0
                    cov=covered[indices].astype(np.float64)
                    prefix=np.vstack([np.zeros((1,len(METHODS))),np.cumsum(cov,axis=0)])
                    gap=(prefix[672:]-prefix[:-672])/672-float(coverage)
                    tolerance=1.96*np.sqrt(float(coverage)*(1-float(coverage))/672)
                    regimes=f.ramp_state.to_numpy()[indices]
                    for regime in ['overall','ordinary','ramp']:
                        mask=np.ones(n,dtype=bool) if regime=='overall' else regimes==regime
                        keep=indices[mask];window_mask=mask[671:]
                        n_regime=int(mask.sum());nr_regime=int(window_mask.sum())
                        assert n_regime>0 and nr_regime>0
                        g=gap[window_mask]
                        means=[errf[keep].mean(0),(capacity_up[keep]+capacity_down[keep]).mean(0),
                            (exceed_up[keep]+exceed_down[keep]).mean(0),cov[mask].mean(0),(upper[keep]-lower[keep]).mean(0),
                            (g < -tolerance).mean(0),(g > tolerance).mean(0),np.abs(g).mean(0),
                            np.maximum(-g,0).mean(0),np.maximum(g,0).mean(0)]
                        for j,method in enumerate(METHODS):
                            row={'farm_id':manifest['farm_id'],'seed':manifest['seed'],'price_id':price['price_id'],
                                'predictor':predictor,'horizon_steps':horizon,'target_coverage':float(coverage),
                                'regime':regime,'method':method,'event_count':n_regime,'reliability_count':nr_regime}
                            row.update({m:float(means[k][j]) for k,m in enumerate(METRICS)})
                            rows.append(row)
            print(json.dumps({'farm':manifest['farm_id'],'seed':manifest['seed'],'stream':stream.name,'status':'evaluated'}),flush=True)
    return pd.DataFrame(rows)

def aggregate(seed_cells):
    sums=seed_cells.copy()
    for name in METRICS:sums[name]*=sums['event_count' if name in METRICS[:5] else 'reliability_count']
    conditions=sums.groupby(CONDITIONS,as_index=False)[METRICS+['event_count','reliability_count']].sum()
    for name in METRICS:conditions[name]/=conditions['event_count' if name in METRICS[:5] else 'reliability_count']
    farm_price=conditions.groupby(['farm_id','price_id','regime','method'],as_index=False)[METRICS].mean()
    farm_all=farm_price.groupby(['farm_id','regime','method'],as_index=False)[METRICS].mean();farm_all['price_id']='ALL'
    means=pd.concat([farm_price,farm_all],ignore_index=True)
    pooled=means.groupby(['price_id','regime','method'],as_index=False)[METRICS].mean();pooled['farm_id']='Pooled'
    means=pd.concat([means,pooled],ignore_index=True)
    return conditions,means

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--farm',choices=['FarmA','FarmB']);ap.add_argument('--seed',type=int);args=ap.parse_args()
    paths=[p.parent for p in sorted(args.archive.glob('Farm*/seed*/manifest.json'))]
    if args.farm:paths=[p for p in paths if p.parent.name==args.farm]
    if args.seed is not None:paths=[p for p in paths if p.name==f'seed{args.seed}']
    if not paths:raise SystemExit('No complete extracted shards found')
    results=[]
    for p in paths:
        cache=args.output/'shards'/f'{p.parent.name}_{p.name}.parquet'
        if cache.exists():
            cached=pd.read_parquet(cache)
            if 'regime' in cached:results.append(cached);continue
        f=evaluate_shard(p);write_frame(cache,f);results.append(f)
    seed_cells=pd.concat(results,ignore_index=True)
    conditions,means=aggregate(seed_cells)
    write_frame(args.output/'seed_cells.parquet',seed_cells)
    write_frame(args.output/'condition_metrics.parquet',conditions)
    write_frame(args.output/'means.parquet',means)
    means.to_csv(args.output/'means.csv',index=False)
    print(means[(means.farm_id=='Pooled')&(means.price_id=='ALL')&(means.regime=='overall')][['method','ERRF','TUWR']].to_string(index=False),flush=True)

if __name__=='__main__':main()
