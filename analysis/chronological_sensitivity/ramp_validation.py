"""Prespecified six-threshold source-zone validation, entirely before T0."""
from sensitivity_common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def load_source_with_proxy(zone,seed,p,h,ti):
    d=fc.load_stream(zone,seed,p,h,'source',ti);cf._verify_source_times(d)
    path=fc.BASE/p/fc.stream_id(zone,seed,p,h)/'base_predictions.parquet'
    raw=pd.read_parquet(path,columns=['issue_timestamp','base_center','feature_target_lag1'])
    issues=raw.issue_timestamp.to_numpy(dtype='datetime64[ns]').astype(np.int64)
    index=pd.Index(issues).get_indexer(d['issue_ns']);assert (index>=0).all()
    center=raw.base_center.to_numpy(float)[index]
    np.testing.assert_array_equal(center,d['center'])
    proxy=np.abs(center-raw.feature_target_lag1.to_numpy(float)[index])
    np.testing.assert_array_equal(proxy>=.12,d['ramp'].astype(bool))
    d['ramp_proxy']=proxy
    return d,sha(path)

def subset_stats(base,arrays,best,train):
    ix=np.array([base.source_zones.index(z) for z in train])
    sliced={k:(v[:,ix,:].copy() if v.ndim==3 else v[ix,:].copy()) for k,v in arrays.items()}
    cb=best[:,:,ix,:].copy()
    _,digest=cf.original._statistics_array_content_identity(sliced,cb)
    audit={**base.audit,'source_zones':train,'sufficient_statistics_content_sha256':digest,
           'sensitivity_inner_training_zones':train,'source_event_count':int(sliced['event_count'].sum())}
    return replace(base,source_zones=tuple(train),arrays=sliced,cart_best_count=cb,audit=audit,
                   stream_audit=base.stream_audit[base.stream_audit.source_zone.isin(train)].copy())

def worker(task):
    z,s=task;start=time.perf_counter();out=HERE/'ramp'/f'{z}_seed{s}';out.mkdir(parents=True,exist_ok=True)
    ident={**base_identity(),'script':sha(__file__)}
    if (out/'COMPLETE.json').exists():
        r=read(out/'COMPLETE.json');assert r['identity']==ident
        for name,digest in r['outputs'].items():assert sha(out/name)==digest
        return r
    parent=CHRONO/'clara'/z/f'seed{s}';bundle=cf.load_bundle(parent);base=bundle.stats
    sources=list(base.source_zones);assert z not in sources and len(sources)==9
    ti=bundle.audit['tsc_configuration_index'];streams={};rawhashes={};codes={}
    arrays=[cf.new_arrays(9) for _ in TAUS];best=[np.zeros((5,6,9,6600),np.int64) for _ in TAUS]
    for zi,source in enumerate(sources):
        for p in fc.PREDICTORS:
            for h in fc.HORIZONS:
                d,rawhash=load_source_with_proxy(source,s,p,h,ti);key=(source,p,h)
                streams[key]=d;rawhashes[fc.stream_id(source,s,p,h)]=rawhash
                code=cf.state_codes(d,p,h,bundle.width_edges);codes[key]=[]
                for i,tau in enumerate(TAUS):
                    adjusted=code+15*((d['ramp_proxy']>=tau).astype(np.int64)-d['ramp'])[None,:]
                    codes[key].append(adjusted)
                    cf.add_stream_statistics(arrays[i],best[i],zi,d,adjusted)
    for key,value in base.arrays.items():np.testing.assert_array_equal(arrays[2][key],value)
    np.testing.assert_array_equal(best[2],base.cart_best_count)
    rows=[];inner_receipts=[]
    for inner in range(3):
        valid=sources[inner::3];train=[q for q in sources if q not in valid]
        assert len(train)==6 and len(valid)==3 and not set(train)&set(valid)
        assert z not in train+valid
        policies=[];backoffs=[];audits=[]
        for i,tau in enumerate(TAUS):
            stats=subset_stats(base,arrays[i],best[i],train)
            choice,back,audit=fit_choices(stats)
            policies.append(choice);backoffs.append(back);audits.append(audit)
        policies=np.stack(policies);backoffs=np.stack(backoffs)
        np.savez_compressed(out/f'inner{inner}_state_choices.npz',choices=policies,backoff=backoffs)
        save(out/f'inner{inner}_FIT_FROZEN.json',dict(status='PASS',identity=ident,training_zones=train,
             validation_zones=valid,outer_heldout=z,source_cutoff='2013-09-05 00:00 UTC',
             target_outcomes_used=False,inner_validation_outcomes_used_in_cost_coverage_fit=False,
             common_outer_source_width_and_TSC=True,choices_sha256=sha(out/f'inner{inner}_state_choices.npz')))
        for source in valid:
            for p in fc.PREDICTORS:
                for h in fc.HORIZONS:
                    key=(source,p,h);d=streams[key];ix=np.arange(len(d['y']))
                    for i,tau in enumerate(TAUS):
                        code=codes[key][i];choice=policies[i][:,code];bo=backoffs[i][:,code]
                        flagged=float(np.mean(d['ramp_proxy']>=tau))
                        for pi,pid in enumerate(PIDS):
                            for ci,c in enumerate(fc.COVERAGES):
                                act=choice[pi,ci]
                                m=metric_values(d,d['lower'][act,ci,ix],d['upper'][act,ci,ix],ci,pi)
                                rows.append(dict(outer_zone=z,seed=s,inner_fold=inner,validation_zone=source,
                                    predictor=p,horizon=h,target_coverage=c,tau=tau,price_id=pid,
                                    price_ratio=float(fc.PRICES[pi]),ramp_fraction=flagged,
                                    backoff_fraction=float(np.mean(bo[pi,ci]>0)),
                                    coverage_deficiency=max(0.,float(c)-m['coverage']),**m))
        inner_receipts.append(dict(inner=inner,training=train,validation=valid,
             fitted_policy_sha256=sha(out/f'inner{inner}_state_choices.npz')))
        print(f'ramp {z} seed{s} inner{inner} complete {time.perf_counter()-start:.1f}s',flush=True)
    frame=pd.DataFrame(rows);assert len(frame)==9*20*11*5*6
    frame.to_parquet(out/'validation_metrics.parquet',index=False)
    cols=list(EVENTS)+list(WINDOWS)+['ramp_fraction','backoff_fraction','coverage_deficiency']
    frame.groupby(['tau','price_id'],as_index=False)[cols].mean().to_csv(out/'validation_by_price.csv',index=False)
    report=dict(status='PASS',zone=z,seed=s,identity=ident,source_only=True,target_loaded=False,
        reference_012_statistics_exact=True,source_zones=sources,inner_folds=inner_receipts,
        base_prediction_hashes=rawhashes,metric_rows=len(frame),seconds=time.perf_counter()-start,
        outputs={p.name:sha(p) for p in out.iterdir() if p.suffix in ('.npz','.parquet','.csv') or p.name.endswith('_FROZEN.json')})
    save(out/'COMPLETE.json',report);return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    tasks=[('zone1',0)] if a.pilot else [(z,s) for z in fc.ZONES for s in fc.SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for done in as_completed([pool.submit(worker,t) for t in tasks]):
            r=done.result();print(r['zone'],r['seed'],r['status'],round(r['seconds'],2),flush=True)
