"""Chronological candidate diagnostics and the original forecaster-rule transfer test."""
from pathlib import Path
import sys, json, hashlib
from concurrent.futures import ProcessPoolExecutor,as_completed
import os
CODE_DIR=Path(__file__).resolve().parent
REPOSITORY=CODE_DIR.parents[1]
sys.path.insert(0,str(REPOSITORY/'src/chronological'))
from _paths import DATA_ROOT as CHRONO
import forward_common as C
import clara_forward as F
import numpy as np
import pandas as pd
OUT=Path(os.environ.get('CLARA_DISCOVERY_OUTPUT',str(REPOSITORY/'outputs/chronological_discovery')))
OUT.mkdir(exist_ok=True,parents=True)
FIG4=OUT/'figure04/data';FIG4.mkdir(parents=True,exist_ok=True)
FIGS=OUT/'figureS1/data';FIGS.mkdir(parents=True,exist_ok=True)
PID=['R01','R02','R05P6179775281','R10','R20']
NAMES=['Static','ACI','AgACI','EnbPI-RH','TSC','EEE']

def fit(task):
 z,s=task; bundle=F.load_bundle(CHRONO/'clara'/z/f'seed{s}')
 count=bundle.stats.arrays['event_count'].sum(axis=0)
 cap=bundle.stats.arrays['capacity_sum'].sum(axis=1).T
 miss=bundle.stats.arrays['miss_sum'].sum(axis=1).T
 rules=[];empties=[]
 for rho in C.PRICES:
  loss=C.PI*(cap+rho*miss)
  bypred=[];ep=[]
  for pi in range(4):
   ix=slice(pi*1650,(pi+1)*1650);n=count[ix];values=loss[ix]
   means=np.divide(values,n[:,None],out=np.zeros_like(values),where=n[:,None]>0)
   empty=n==0;means[empty]=values.sum(axis=0)/n.sum()
   choices=np.argmax(means<=means.min(axis=1)[:,None]+1e-12,axis=1)
   bypred.append(choices);ep.append(empty)
  rules.append(bypred);empties.append(ep)
 dest=OUT/'rules';dest.mkdir(exist_ok=True)
 np.savez_compressed(dest/f'{z}_seed{s}.npz',rules=np.asarray(rules,dtype=np.int8),empty=np.asarray(empties),edges=bundle.width_edges)

def one(task):
 z,s=task
 with np.load(OUT/'rules'/f'{z}_seed{s}.npz') as f:rule=f['rules'];empty=f['empty'];edges=f['edges']
 rows=[];ti=C.select_tsc({z})
 with np.load(CHRONO/'clara'/z/f'seed{s}/decisions.npz') as f:checks={k:f[k] for k in f.files}
 reference=pd.read_parquet(CHRONO/f'summary/seed_cells/{z}_seed{s}.parquet')
 reference=reference[reference.method.eq('CLARA')].set_index(['predictor','horizon','target_coverage','price_id'])
 for p in C.PREDICTORS:
  for h in C.HORIZONS:
   d=C.load_stream(z,s,p,h,'target',ti);code=F.state_codes(d,p,h,edges);rest=code%1650
   capacity,miss,_=C.endpoint_components(d['lower'],d['upper'],d['center'][None,None,:],d['y'][None,None,:])
   cidx=np.arange(11)[:,None];tidx=np.arange(len(d['y']))[None,:]
   for pri,rho in enumerate(C.PRICES):
    loss=C.PI*(capacity+rho*miss)
    own=checks[f'{p}__H{h:02d}'][pri]
    chk=loss[own,cidx,tidx].mean(axis=1)
    key=['R01','R02','Rref','R10','R20'][pri]
    expected=[reference.loc[(p,h,c,key),'mean_errf'] for c in C.COVERAGES]
    np.testing.assert_allclose(chk,expected,atol=1e-12,rtol=0)
    for ip,source in enumerate(C.PREDICTORS):
     choice=rule[pri,ip,rest];real=loss[choice,cidx,tidx]
     for ci,c in enumerate(C.COVERAGES):
      rows.append(dict(zone=z,seed=s,price_id=PID[pri],target_predictor=p,source_predictor=source,
                       horizon=h,target_coverage=c,n=len(d['y']),ERRF=real[ci].mean(),fallback_n=int(empty[pri,ip,rest[ci]].sum())))
 dest=OUT/'transfer_cells';dest.mkdir(exist_ok=True)
 pd.DataFrame(rows).to_parquet(dest/f'{z}_seed{s}.parquet',index=False)
 return f'{z}/{s}'

def aggregate():
 raw=pd.concat([pd.read_parquet(p) for p in sorted((OUT/'transfer_cells').glob('*.parquet'))],ignore_index=True)
 assert len(raw)==132000
 keys=['zone','price_id','target_predictor','source_predictor','horizon','target_coverage']
 raw['loss_sum']=raw.ERRF*raw.n
 cell=raw.groupby(keys,as_index=False)[['loss_sum','n','fallback_n']].sum();cell['ERRF']=cell.loss_sum/cell.n
 cell.to_parquet(OUT/'transfer_condition_metrics.parquet',index=False)
 zone=cell.groupby(keys[:4],as_index=False).ERRF.mean()
 allzone=zone.groupby(['zone','target_predictor','source_predictor'],as_index=False).ERRF.mean();allzone['price_id']='ALL'
 zone=pd.concat([zone,allzone],ignore_index=True);zone.to_csv(OUT/'transfer_zone_means.csv',index=False)
 draws=np.random.default_rng(20260925).integers(0,10,(10000,10));rows=[]
 for pid in ['ALL',*PID]:
  for target in C.PREDICTORS:
   q=zone[(zone.price_id==pid)&(zone.target_predictor==target)].pivot(index='zone',columns='source_predictor',values='ERRF').reindex(C.ZONES)
   own=q[target].to_numpy()
   for source in C.PREDICTORS:
    val=q[source].to_numpy();diff=val-own
    ci=np.quantile(100*(val[draws].mean(axis=1)/own[draws].mean(axis=1)-1),[.025,.975])
    rows.append(dict(price_id=pid,target_predictor=target,source_predictor=source,ERRF=val.mean(),matched_ERRF=own.mean(),
                     change_pct=100*(val.mean()/own.mean()-1),ci95_low=ci[0],ci95_high=ci[1],
                     zones_increase=int((diff>1e-12).sum()),zones_decrease=int((diff< -1e-12).sum())))
 table=pd.DataFrame(rows);table.to_csv(FIG4/'conditional_forecaster_transfer_all_prices.csv',index=False)
 data=pd.read_parquet(CHRONO/'summary/condition_metrics.parquet');data=data[data.method.isin(NAMES)].copy()
 data.price_id=data.price_id.replace({'Rref':PID[2]})
 means=data.groupby(['price_id','target_coverage','method'],as_index=False).mean_errf.mean()
 best=means.loc[means.groupby(['price_id','target_coverage']).mean_errf.idxmin()].rename(columns={'method':'lowest_cost_candidate'})
 best.to_csv(FIG4/'lowest_candidate_by_price_coverage.csv',index=False)
 keys=['zone','price_id','predictor','horizon','target_coverage']
 data['relative_excess']=100*(data.mean_errf/data.groupby(keys).mean_errf.transform('min')-1)
 stats=[]
 for method,q in data.groupby('method'):
  v=q.relative_excess.to_numpy();s=np.quantile(v,[0,.25,.5,.75,1]);stats.append(dict(method=method,minimum=s[0],q1=s[1],median=s[2],q3=s[3],maximum=s[4]))
 pd.DataFrame(stats).to_csv(FIG4/'candidate_cost_distribution.csv',index=False)
 keys=['zone','predictor','horizon','target_coverage','method']
 invariant=data.groupby(keys)[['coverage','tuwr']].nunique();assert invariant.to_numpy().max()==1
 reliable=data.drop_duplicates(keys).copy();assert len(reliable)==13200
 reliable['overall_gap_pp']=100*(reliable.coverage-reliable.target_coverage);reliable['TUWR_pct']=100*reliable.tuwr
 reliable.to_csv(FIG4/'all_candidate_overall_and_rolling_reliability.csv',index=False)
 off=table[(table.price_id=='ALL')&(table.source_predictor!=table.target_predictor)]
 (OUT/'discovery_summary.json').write_text(json.dumps(dict(status='PASS',fit_source='Chronological prefix only; outer target excluded',
  transfer_rule='Event-weighted six-action mean ERRF within exact state; empty state falls back to that source forecaster mean; no CLARA screening/shrinkage',
  retained_target_state_categories=True,primary_clara_metrics_verified=True,seed_rows=len(raw),pooled_cells=len(cell),
  transfer_min_pct=off.change_pct.min(),transfer_max_pct=off.change_pct.max(),
  increase_cells=int((off.change_pct>0).sum()),negative_cells=int((off.change_pct<0).sum()),candidate_reliability_cells=len(reliable)),indent=2),encoding='utf8')

def chronology():
 z='zone1';s=0;p='GBR';h=6;ci=8;c=.9;pri=2
 d=C.load_stream(z,s,p,h,'target',C.select_tsc({z}))
 with np.load(CHRONO/'clara'/z/f'seed{s}/decisions.npz') as f:choice=f[f'{p}__H{h:02d}'][pri,ci]
 lower=d['lower'][:,ci,:];upper=d['upper'][:,ci,:];y=d['y'];n=len(y)
 rows=[]
 for method,a in [('Static',np.zeros(n,dtype=int)),('EnbPI-RH',np.full(n,3)),('CLARA',choice)]:
  cover=(lower[a,np.arange(n)]<=y)&(y<=upper[a,np.arange(n)])
  rolling=pd.Series(cover).rolling(168,min_periods=168).mean().to_numpy();delta=1.96*np.sqrt(c*(1-c)/168)
  tuwr=np.where(np.isfinite(rolling),(rolling<c-delta).astype(float),np.nan)
  rows.append(pd.DataFrame(dict(issue_timestamp=pd.to_datetime(d['issue_ns']),method=method,covered=cover.astype(int),rolling_coverage=rolling,TUWR=tuwr)))
 df=pd.concat(rows,ignore_index=True);df.to_csv(FIGS/'chronology.csv',index=False)
 df.groupby('method')[['covered','TUWR']].mean().to_csv(FIGS/'summary.csv')
 (FIGS/'receipt.json').write_text(json.dumps(dict(status='PASS',per_method_forecasts=n,coverage=.9,
  first_issue=str(pd.to_datetime(d['issue_ns'][0])),first_complete_168h_window=str(pd.to_datetime(d['issue_ns'][167])),
  post_cutoff_only=True),indent=2),encoding='utf8')

if __name__=='__main__':
 tasks=[(z,s) for z in C.ZONES for s in C.SEEDS]
 for t in tasks:fit(t)
 (OUT/'transfer_fit_seal.json').write_text(json.dumps({p.name:C.sha256(p) for p in sorted((OUT/'rules').glob('*.npz'))},indent=2),encoding='utf8')
 with ProcessPoolExecutor(max_workers=2) as pool:
  fs=[pool.submit(one,t) for t in tasks]
  for i,f in enumerate(as_completed(fs),1):print(i,30,f.result(),flush=True)
 aggregate();chronology();print('COMPLETE',flush=True)
