"""Separate three hindsight estimands on the unchanged six-action target archive."""
from pathlib import Path
import os,sys,json,hashlib,time,importlib.util,argparse
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor,as_completed
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT/'analysis/complete_ablation_20260909'))
import runtime as rt
spec=importlib.util.spec_from_file_location('oracle_gev',rt.GEOLD/'evaluate.py')
gev=importlib.util.module_from_spec(spec);spec.loader.exec_module(gev)
V38=ROOT/'analysis/chapter4_original_structure_v38_20260909'
METHODS=['CLARA','STATE_ORACLE','FEASIBLE_STATE_ORACLE','OUTCOME_ORACLE']
M=rt.METRICS;F=rt.FIELDS
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def allowed_set(d,e,thresholds):
 eps=float(thresholds['coverage_shortfall_epsilon']);tau=float(thresholds['tuwr_upper'])
 target=d.target_coverage.to_numpy(float)[:,None];warm=d.rolling_state.ne('cold_start').to_numpy()[:,None]
 basic=(e['coverage']>=target-eps)&((~warm)|(np.isfinite(e['tuwr'])&(e['tuwr']<=tau)))
 v=np.maximum(np.maximum(0,target-eps-e['coverage'])/eps,np.where(warm,np.maximum(0,e['tuwr']-tau)/tau,0))
 v=np.where(warm&~np.isfinite(e['tuwr']),np.inf,v)
 empty=~basic.any(1);allowed=basic.copy();allowed[empty]=v[empty]<=v[empty].min(1)[:,None]+1e-12
 selected,trace=rt.agreed.select(d,e,thresholds,'DIRECTIONAL_REPAIR',1.)
 assert allowed[np.arange(len(d)),selected].all()
 np.testing.assert_array_equal(rt.agreed.choose(e,allowed),selected)
 return allowed,selected

def worker(task):
 zone,seed=task;uid=f'{zone}__seed{seed}';dest=HERE/'oracle'/uid;dest.mkdir(parents=True,exist_ok=True)
 identity={'plan':sha(HERE/'PLAN.md'),'code':sha(__file__),'accepted_policies':sha(rt.GEOLD/'policies_frozen.json')}
 if (dest/'manifest.json').exists():
  man=read(dest/'manifest.json');assert man['status']=='PASS' and man['identity']==identity
  for n,h in man['outputs'].items():assert sha(dest/n)==h
  return {'unit':uid,'status':'REUSED'}
 start=time.monotonic();policies={};allowed={};choices={};source_refs={}
 for pid,_ in rt.PRICES:
  d,e,audit,man=rt.original_ge(zone,seed,pid)
  allowed[pid],choices[pid]=allowed_set(d,e,audit['adaptive_guardrail_thresholds']);policies[pid]=d
  path=rt.GEOLD/'policies'/uid/pid/'state_decisions.parquet'
  old=pd.read_parquet(path).sort_values('full_state_code');np.testing.assert_array_equal(choices[pid],old[rt.agreed.PRIMARY])
  source_refs[pid]=sha(path)
 target,_,_,_=gev.expected_sources(zone,seed)
 tp=rt.ge.FACTROOT/'fold_width_thresholds.parquet';assert sha(tp)==next(iter(target['stream_audits'].values()))['leaf_sha256']['width_thresholds']
 thresholds=pd.read_parquet(tp);records=[];state_rows=[]
 for p in rt.ge.PREDICTORS:
  for h in rt.ge.HORIZONS:
   facts,_=gev.load_stream(zone,seed,p,h,target,thresholds)
   state_ix=pd.MultiIndex.from_frame(d[F]).get_indexer(pd.MultiIndex.from_frame(facts[F]));assert (state_ix>=0).all()
   counts=np.bincount(state_ix,minlength=len(d));present=counts>0
   for pid,rho in rt.PRICES:
    theta=[3.56,3.56,20. if pid==rt.rt.MAIN_ID else 3.56*rho,20. if pid==rt.rt.MAIN_ID else 3.56*rho]
    cand=gev.event_arrays(facts,np.tile(np.arange(6),(len(facts),1)),theta)
    costs=np.stack([np.bincount(state_ix,weights=cand[:,a,0],minlength=len(d)) for a in range(6)],axis=1)
    # Same-size groups make minimizing sums equivalent to minimizing group means.
    unrestricted=costs.argmin(1);feasible=np.where(allowed[pid],costs,np.inf).argmin(1)
    selected=choices[pid]
    assert np.all(costs[present,unrestricted[present]]<=costs[present,feasible[present]]+1e-9)
    assert np.all(costs[present,feasible[present]]<=costs[present,selected[present]]+1e-9)
    ch=np.column_stack([selected[state_ix],unrestricted[state_ix],feasible[state_ix],cand[:,:,0].argmin(1)])
    values=np.take_along_axis(cand,ch[:,:,None],axis=1)
    for c,g in facts.groupby('target_coverage',sort=False):
     at=g.index.to_numpy();roll=gev.rolling(values[at,:,3],c);n=len(at);nr=max(0,n-167)
     avg=np.concatenate([values[at].mean(0),np.nansum(roll,axis=0)/nr],axis=1)
     assert avg[3,0]<=avg[1,0]+1e-10<=avg[2,0]+2e-10<=avg[0,0]+3e-10
     for j,m in enumerate(METHODS):records.append(dict(zone_or_farm=zone,seed=seed,price_id=pid,predictor=p,horizon_steps=h,target_coverage=float(c),regime='overall',method=m,event_count=n,reliability_event_count=nr,**dict(zip(M,avg[j]))))
    st=d.loc[present,['full_state_code']+F].copy();st['zone']=zone;st['seed']=seed;st['horizon_steps']=h;st['price_id']=pid
    st['event_count']=counts[present];st['CLARA']=selected[present];st['STATE_ORACLE']=unrestricted[present];st['FEASIBLE_STATE_ORACLE']=feasible[present]
    st['eligible_count']=allowed[pid][present].sum(1)
    for a,name in enumerate(rt.ge.ACTIONS):st[name+'__ERRF']=costs[present,a]/counts[present]
    state_rows.append(st)
 records=pd.DataFrame(records)
 key=['price_id','predictor','horizon_steps','target_coverage','regime','method']
 old=pd.read_parquet(V38/'targeted_checks/evaluation'/uid/'cell_metrics.parquet');old=old[old.method.isin(['CLARA','OUTCOME_ORACLE']) & old.regime.eq('overall')].set_index(key).sort_index()
 got=records[records.method.isin(['CLARA','OUTCOME_ORACLE'])].set_index(key).reindex(old.index)
 np.testing.assert_array_equal(got[['event_count','reliability_event_count']],old[['event_count','reliability_event_count']])
 error=float(np.nanmax(abs(got[M].to_numpy()-old[M].to_numpy())));assert error<1e-10
 records.to_parquet(dest/'cell_metrics.parquet',index=False);pd.concat(state_rows,ignore_index=True).to_parquet(dest/'state_choices.parquet',index=False)
 save(dest/'manifest.json',{'status':'PASS','identity':identity,'source_policies':source_refs,'accepted_CLARA_and_sample_oracle_max_error':error,'outputs':{p.name:sha(p) for p in dest.glob('*.parquet')},'seconds':time.monotonic()-start})
 return {'unit':uid,'status':'PASS','seconds':round(time.monotonic()-start,1)}

def summarize():
 out=HERE/'summary';out.mkdir(exist_ok=True);frames=[];receipts=[]
 for zone in rt.ge.ZONES:
  for seed in range(3):
   folder=HERE/'oracle'/f'{zone}__seed{seed}';man=read(folder/'manifest.json');assert man['status']=='PASS'
   assert man['identity']['code']==sha(__file__)
   for n,h in man['outputs'].items():assert sha(folder/n)==h
   frames.append(pd.read_parquet(folder/'cell_metrics.parquet'));receipts.append({'unit':folder.name,'error':man['accepted_CLARA_and_sample_oracle_max_error']})
 d=pd.concat(frames,ignore_index=True);key=['zone_or_farm','price_id','predictor','horizon_steps','target_coverage','method']
 for m in M:d[m]*=d['event_count' if m in M[:5] else 'reliability_event_count']
 d=d.groupby(key,as_index=False)[M+['event_count','reliability_event_count']].sum(min_count=1)
 for m in M:d[m]/=d['event_count' if m in M[:5] else 'reliability_event_count']
 d.to_parquet(out/'oracle_conditions.parquet',index=False)
 z=d.groupby(['zone_or_farm','price_id','method'],as_index=False)[M].mean()
 a=z.groupby(['zone_or_farm','method'],as_index=False)[M].mean();a['price_id']='ALL';z=pd.concat([z,a],ignore_index=True)
 z.to_csv(out/'oracle_zone_means.csv',index=False)
 avg=z.groupby(['price_id','method'],as_index=False)[M].mean();avg.to_csv(out/'oracle_means.csv',index=False)
 rng=np.random.default_rng(2026091021);ix=rng.integers(0,10,(5000,10));rows=[]
 for pid,g in z.groupby('price_id'):
  p=g.pivot(index='zone_or_farm',columns='method',values='ERRF').reindex(rt.ge.ZONES)
  for m in METHODS[1:]:
   a=p.CLARA.to_numpy();b=p[m].to_numpy();v=100*(a[ix].mean(1)/b[ix].mean(1)-1);lo,hi=np.quantile(v,[.025,.975])
   rows.append(dict(price_id=pid,oracle=m,CLARA_ERRF=a.mean(),oracle_ERRF=b.mean(),gap_pct=100*(a.mean()/b.mean()-1),ci95_low=lo,ci95_high=hi))
 pd.DataFrame(rows).to_csv(out/'oracle_gaps.csv',index=False)
 save(out/'oracle_manifest.json',{'status':'PASS','code':sha(__file__),'plan':sha(HERE/'PLAN.md'),'reproduction':receipts,'estimands':METHODS,'complete_price_grid':True})
 print(pd.DataFrame(rows).to_string(index=False),flush=True)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);ap.add_argument('--pilot',action='store_true');ap.add_argument('--summarize',action='store_true');a=ap.parse_args()
 if a.summarize:summarize()
 else:
  tasks=[(z,s) for z in rt.ge.ZONES for s in range(3)]
  if a.pilot:tasks=tasks[:1]
  with ProcessPoolExecutor(max_workers=a.workers) as pool:
   for f in as_completed([pool.submit(worker,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
