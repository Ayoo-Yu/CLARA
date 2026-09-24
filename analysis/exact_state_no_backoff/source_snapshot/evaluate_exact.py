"""Evaluate sealed exact-state decisions against the current six-action archive."""
from pathlib import Path
import os, sys, json, hashlib, time, importlib.util, argparse
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[k]='1'
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
GE=ROOT/'analysis/guardrail_redesign_gefcom_20260908'
sys.path.insert(0,str(GE))
import gefcom_guardrail_runtime as ge
spec=importlib.util.spec_from_file_location('archive_evaluator',GE/'evaluate.py')
gev=importlib.util.module_from_spec(spec);spec.loader.exec_module(gev)
METHODS=['CLARA','NO_BACKOFF_EXACT']
METRICS=ge.METRICS
KEY=['zone_or_farm','price_id','predictor','horizon_steps','target_coverage','regime','method']

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

_verified={}
def checked(path,expected):
    p=Path(path);s=p.stat();key=(str(p.resolve()),s.st_size,s.st_mtime_ns,expected)
    if key not in _verified:
        actual=sha(p);assert actual==expected,(str(p),actual,expected)
        _verified[key]=True
    return expected
gev.checked=checked

def source_contract():
    manifest=ROOT/'analysis/ablation_integration_v47_20260911/data_manifest.json'
    m=read(manifest);assert m['status']=='PASS' and m['CLARA_identical_across_archives']
    for path,h in m['source_hashes'].items():checked(path,h)
    return manifest,m

def metrics(values,roll,positions,valid_mask):
    n=len(positions);nr=int(valid_mask.sum())
    sums=np.concatenate([values[positions].sum(0),np.nansum(roll,axis=0)],axis=1)
    den=np.array([n]*5+[nr]*5)
    return np.divide(sums,den,out=np.full_like(sums,np.nan),where=den>0),n,nr

def worker(task):
    zone,seed=task;uid=f'{zone}__seed{seed}';dest=HERE/'evaluation'/uid
    identity={'freeze_sha256':sha(HERE/'policies_frozen.json'),'code_sha256':sha(__file__),'plan_sha256':sha(HERE/'PLAN.md')}
    if (dest/'manifest.json').exists():
        m=read(dest/'manifest.json');assert m['status']=='PASS' and m['identity']==identity
        for n,h in m['outputs'].items():checked(dest/n,h)
        return {'unit':uid,'status':'REUSED'}
    frozen=read(HERE/'policies_frozen.json');assert frozen['status']=='PASS'
    checked(HERE/'PLAN.md',frozen['plan'])
    checked(HERE/'build_policies.py',frozen['fit_code'])
    start=time.monotonic();dest.mkdir(parents=True,exist_ok=True)
    v47path,v47=source_contract()
    folder=HERE/'policies'/uid
    checked(folder/'manifest.json',frozen['policy_parents'][str((folder/'manifest.json').relative_to(HERE))])
    pm=read(folder/'manifest.json');assert pm['status']=='PASS' and not pm['test_outcomes_used_in_fit']
    assert pm['identity']==frozen['identity']
    for name,h in pm['outputs'].items():checked(folder/name,h)
    decisions=pd.read_parquet(folder/'decisions.parquet')
    assert not decisions.duplicated(['price_id','full_state_code']).any()
    for name in METHODS:
        assert decisions[name].notna().all() and decisions[name].between(0,5).all()
    policies={pid:g.sort_values('full_state_code').reset_index(drop=True) for pid,g in decisions.groupby('price_id')}
    target,_,_,inputs=gev.expected_sources(zone,seed)
    threshold_path=ge.FACTROOT/'fold_width_thresholds.parquet'
    checked(threshold_path,next(iter(target['stream_audits'].values()))['leaf_sha256']['width_thresholds'])
    thresholds=pd.read_parquet(threshold_path)
    records=[];subsets=[];traces=[];receipts=[]
    for predictor in ge.PREDICTORS:
        for horizon in ge.HORIZONS:
            f,source_ids=gev.load_stream(zone,seed,predictor,horizon,target,thresholds);inputs.update(source_ids)
            receipts.append({'predictor':predictor,'horizon':horizon,'event_count':len(f),'event_signature':gev.signature(f.event_id)})
            for pid,rho in ge.PRICES:
                q=policies[pid]
                ix=pd.MultiIndex.from_frame(q[ge.FIELDS]).get_indexer(pd.MultiIndex.from_frame(f[ge.FIELDS]));assert (ix>=0).all()
                choices=q[METHODS].to_numpy(int)[ix]
                theta=[3.56,3.56,20. if pid=='R05P6179775281' else 3.56*rho,20. if pid=='R05P6179775281' else 3.56*rho]
                values=gev.event_arrays(f,choices,theta)
                backoff=q.was_backoff.to_numpy(bool)[ix]
                zero=q.exact_count.to_numpy(int)[ix]==0
                assert np.all(choices[zero,1]==0), 'Zero-history states must use the author-specified Static action'
                assert np.all(backoff[zero]), 'A zero-history state cannot meet the original support gate'
                np.testing.assert_array_equal(choices[~backoff,0],choices[~backoff,1])
                for coverage,g in f.groupby('target_coverage',sort=False):
                    pos=g.index.to_numpy();rolled=gev.rolling(values[pos,:,3],coverage);valid=np.arange(len(g))>=167
                    common={'zone_or_farm':zone,'seed':seed,'price_id':pid,'predictor':predictor,'horizon_steps':horizon,'target_coverage':float(coverage)}
                    for regime in ['overall','ordinary','ramp']:
                        mask=np.ones(len(g),bool) if regime=='overall' else g.ramp_state.eq(regime).to_numpy()
                        result,n,nr=metrics(values,rolled[mask],pos[mask],valid[mask])
                        for j,method in enumerate(METHODS):
                            records.append({**common,'regime':regime,'method':method,'event_count':n,'reliability_event_count':nr,**dict(zip(METRICS,result[j]))})
                    for label,mask in [('supported',~backoff[pos]),('sparse_nonempty',backoff[pos]&~zero[pos]),('nonempty',~zero[pos]),('zero',zero[pos]),('backoff_required',backoff[pos])]:
                        result,n,nr=metrics(values,rolled[mask],pos[mask],valid[mask])
                        for j,method in enumerate(METHODS):
                            subsets.append({**common,'subset':label,'method':method,'event_count':n,'reliability_event_count':nr,**dict(zip(METRICS,result[j]))})
                    changed=choices[pos,0]!=choices[pos,1]
                    traces.append({**common,'event_count':len(pos),'changed_count':int(changed.sum()),'backoff_count':int(backoff[pos].sum()),'zero_count':int(zero[pos].sum()),
                                   'changed_supported_count':int((changed&~backoff[pos]).sum()),'changed_backoff_count':int((changed&backoff[pos]).sum()),
                                   'delta_ERRF_sum':float((values[pos,1,0]-values[pos,0,0]).sum()),'delta_capacity_sum':float((values[pos,1,1]-values[pos,0,1]).sum()),'delta_exceedance_sum':float((values[pos,1,2]-values[pos,0,2]).sum())})
    records=pd.DataFrame(records);subsets=pd.DataFrame(subsets);traces=pd.DataFrame(traces)
    partition=subsets.pivot(index=['price_id','predictor','horizon_steps','target_coverage','method'],columns='subset',values='event_count')
    total=records[records.regime.eq('overall')].set_index(['price_id','predictor','horizon_steps','target_coverage','method']).event_count.reindex(partition.index)
    np.testing.assert_array_equal(partition.supported+partition.sparse_nonempty+partition.zero,total)
    np.testing.assert_array_equal(partition.nonempty+partition.zero,total)
    np.testing.assert_array_equal(partition.backoff_required,partition.sparse_nonempty+partition.zero)
    reference=ROOT/'analysis/chapter4_original_structure_v38_20260909/targeted_checks/evaluation'/uid/'cell_metrics.parquet'
    ref=pd.read_parquet(reference)
    k=['price_id','predictor','horizon_steps','target_coverage','regime']
    a=records[records.method.eq('CLARA')].set_index(k).sort_index();b=ref[ref.method.eq('CLARA')].set_index(k).reindex(a.index)
    np.testing.assert_array_equal(a[['event_count','reliability_event_count']],b[['event_count','reliability_event_count']])
    max_error=float(np.nanmax(abs(a[METRICS].to_numpy()-b[METRICS].to_numpy())))
    np.testing.assert_allclose(a[METRICS],b[METRICS],rtol=0,atol=1e-10,equal_nan=True)
    np.testing.assert_allclose(records.ERRF,records.capacity+records.exceedance,rtol=0,atol=1e-11)
    assert records[records.regime.eq('overall')].groupby(['price_id','method']).event_count.sum().eq(sum(s['event_count'] for s in receipts)).all()
    for name,frame in [('cell_metrics.parquet',records),('subset_metrics.parquet',subsets),('decision_changes.parquet',traces)]:frame.to_parquet(dest/name,index=False)
    m={'status':'PASS','identity':identity,'unit':uid,'events_per_price':sum(s['event_count'] for s in receipts),'source_files':inputs,'streams':receipts,
       'v47_manifest_sha256':sha(v47path),'CLARA_archive_max_error':max_error,'reference_sha256':sha(reference),
       'same_events_for_both_methods':True,'seconds':time.monotonic()-start,
       'outputs':{n:sha(dest/n) for n in ['cell_metrics.parquet','subset_metrics.parquet','decision_changes.parquet']}}
    save(dest/'manifest.json',m)
    return {k:m[k] for k in ['unit','status','seconds','CLARA_archive_max_error']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--pilot',action='store_true');args=ap.parse_args()
    assert read(HERE/'policies_frozen.json')['status']=='PASS'
    tasks=[(z,s) for z in ge.ZONES for s in range(3)]
    if args.pilot:tasks=tasks[:1]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
    if not args.pilot:
        ms=[read(HERE/'evaluation'/f'{z}__seed{s}'/'manifest.json') for z,s in tasks]
        assert sum(m['events_per_price'] for m in ms)==23024760
        save(HERE/'evaluation_manifest.json',{'status':'PASS','units':30,'events_per_method_price':23024760,'prices':5,
             'freeze_sha256':sha(HERE/'policies_frozen.json'),'code_sha256':sha(__file__),
             'maximum_CLARA_error':max(m['CLARA_archive_max_error'] for m in ms)})

if __name__=='__main__':main()
