"""Prepare cutoff-limited source histories without changing original calibration.

Generate all configurations on the complete original test timeline. Compute
feedback-based states before slicing at T0; the target slice never resets them.
The four original candidate streams are independently compared to sealed inputs.
"""
from __future__ import annotations
from forward_common import *
import argparse, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from conformal_forward_v2 import evaluate_conformal_grid_compact, build_feedback_schedule


def ns(series):
    return series.to_numpy(dtype='datetime64[ns]').astype(np.int64)


def quantiles(frame):
    return (np.stack([frame[f'q_{(1-c)/2:.4f}'].to_numpy(float) for c in COVERAGES]),
            np.stack([frame[f'q_{(1+c)/2:.4f}'].to_numpy(float) for c in COVERAGES]))


def rolling_summaries(labels, covered, coverage, ends, queries):
    """Exact current 24 h diagnostics over 168 h, preserving numeric boundaries."""
    n=len(labels); ends=np.asarray(ends); queries=np.asarray(queries)
    start=np.maximum(0,ends-168); cutoff=queries-168*HOUR_NS
    available_count=ends-np.minimum(ends,np.searchsorted(labels,cutoff,'left'))
    valid=(available_count>=168)&(labels[np.minimum(start,n-1)]<=cutoff+HOUR_NS)
    cp=np.r_[0.,np.cumsum(covered,dtype=float)]
    gap=np.full(len(ends),np.nan); tuwr=gap.copy(); towr=gap.copy(); ard=gap.copy()
    if n>=24:
        wgap=(cp[24:]-cp[:-24])/24-coverage
        tol=1.96*np.sqrt(coverage*(1-coverage)/24)
        a=start[valid]; b=ends[valid]-23
        gap[valid]=(cp[ends[valid]]-cp[a])/168-coverage
        for dest,values in ((tuwr,wgap < -tol),(towr,wgap > tol)):
            pref=np.r_[0.,np.cumsum(values,dtype=float)]
            dest[valid]=(pref[b]-pref[a])/145
        if n>=168:
            means=np.mean(np.abs(np.lib.stride_tricks.sliding_window_view(wgap,145)),axis=-1)
            ard[valid]=means[a]
    state=np.zeros(len(ends),dtype=np.int8); state[valid]=4
    state[valid&(ard>.08)]=3
    state[valid&((towr>.50)|(gap>.05))]=2
    state[valid&((tuwr>.35)|(gap<-.05))]=1
    return gap,tuwr,towr,ard,state


def compare_original(record, issue, lower, upper, y):
    z,s,p,h=record; directory=BUNDLES/p/stream_id(z,s,p,h)
    ef=directory/'event_core.parquet'; cf=directory/'candidate_intervals.parquet'
    ev=pd.read_parquet(ef,columns=['event_id','issue_timestamp','target_coverage'])
    ca=pd.read_parquet(cf,columns=['event_id','action','candidate_lower','candidate_upper'])
    joined=ev.merge(ca,on='event_id',validate='one_to_many'); report=[]
    for ai,action in zip(MAIN_CONFIG_INDICES,ACTIONS[:4]):
        max_error=0.; coverage_changes=0; bitwise=True
        for ci,c in enumerate(COVERAGES):
            q=joined[(joined.action==action)&np.isclose(joined.target_coverage,c)].sort_values('issue_timestamp')
            np.testing.assert_array_equal(ns(q.issue_timestamp),issue)
            old=q[['candidate_lower','candidate_upper']].to_numpy(float)
            new=np.column_stack([lower[ai,ci],upper[ai,ci]])
            error=float(np.max(np.abs(old-new))); max_error=max(max_error,error)
            coverage_changes+=int(np.count_nonzero(((old[:,0]<=y)&(y<=old[:,1]))!=((new[:,0]<=y)&(y<=new[:,1]))))
            bitwise=bitwise and np.array_equal(old,new)
            np.testing.assert_allclose(old,new,atol=1e-12,rtol=0)
        assert coverage_changes==0,(action,coverage_changes)
        report.append(dict(action=action,maximum_endpoint_difference=max_error,
                           coverage_changes=coverage_changes,bitwise_equal=bool(bitwise)))
    return report,dict(event_core_sha256=sha256(ef),candidate_intervals_sha256=sha256(cf))


def generate(raw, manifest, record, compare=True):
    t0=int(json.loads((RUN/'chrono_cutoff.json').read_text())['t0_ns'])
    raw=raw.sort_values('issue_timestamp').reset_index(drop=True)
    calib=raw[raw.split.eq('calibration')].copy()
    test=raw[raw.split.eq('test')].copy()
    issue=ns(test.issue_timestamp); labels=ns(test.label_timestamp); avail=ns(test.label_available_timestamp)
    avail=np.where(avail==np.iinfo(np.int64).min,np.iinfo(np.int64).max,avail)
    warm=calib[ns(calib.label_available_timestamp)<=issue[0]].copy()
    assert len(warm)>0 and len(test)>168
    assert ns(warm.label_available_timestamp).max()<=issue[0]
    assert np.all(ns(warm.label_timestamp)<issue[0])
    lo,up=quantiles(test); wlo,wup=quantiles(warm)
    order,ends,_=build_feedback_schedule(issue,labels,avail)
    y=test.target.to_numpy(float); center=test.base_center.to_numpy(float)
    lower,upper=evaluate_conformal_grid_compact(contracts=CONTRACTS,coverages=COVERAGES,
        calibration_target=warm.target.to_numpy(float),calibration_lower=wlo,calibration_upper=wup,
        calibration_center=warm.base_center.to_numpy(float),calibration_label_ns=ns(warm.label_timestamp),
        test_target=y,test_lower=lo,test_upper=up,test_center=center,test_issue_ns=issue,
        test_label_ns=labels,feedback_order=order,matured_ends=ends,cadence_hours=1.)[:2]
    checks,hashes=compare_original(record,issue,lower,upper,y) if compare else ([],{})
    lower=np.concatenate([lower,lower[MAIN_CONFIG_INDICES].mean(axis=0,keepdims=True)])
    upper=np.concatenate([upper,upper[MAIN_CONFIG_INDICES].mean(axis=0,keepdims=True)])
    covered=(lower<=y)&(y<=upper)
    states=np.zeros((len(COVERAGES),len(y)),np.int8)
    ct=np.full(lower.shape,np.nan); ca=np.full(lower.shape,np.nan)
    release_ends=np.minimum(np.searchsorted(avail,issue,'right'),np.searchsorted(labels,issue,'left'))
    feedback_ends=np.arange(1,len(y)+1)
    for ci,c in enumerate(COVERAGES):
        states[ci]=rolling_summaries(labels,covered[0,ci],c,release_ends,issue)[-1]
        for ai in range(20):
            _,tu,_,ard,_=rolling_summaries(labels,covered[ai,ci],c,feedback_ends,avail)
            ct[ai,ci]=np.where(states[ci]!=0,tu,np.nan)
            ca[ai,ci]=np.where(states[ci]!=0,ard,np.nan)
    source=(issue<t0)&(avail<=t0)&(labels<t0)
    # Keep the original complete-feedback analysis tail rule for every method.
    target=(issue>=t0)&(avail<=issue[-1])
    assert source.any() and target.any() and not (source&target).any()
    all_data=dict(issue_ns=issue,label_ns=labels,available_ns=avail,y=y,center=center,
        raw_width=up-lo,ramp=(np.abs(center-test.feature_target_lag1.to_numpy(float))>=.12).astype(np.int8),
        rolling_state=states,lower=lower,upper=upper,candidate_tuwr=ct,candidate_ard=ca)
    result={}
    for split,mask in [('source',source),('target',target)]:
        d={k:v[...,mask] for k,v in all_data.items()}
        d.update(source_cutoff_ns=np.int64(t0),source_available_max=np.int64(avail[mask].max()),
            initialization_available_max=np.int64(ns(warm.label_available_timestamp).max()),
            initialization_n=np.int64(len(warm)),boundary_row=np.int64(manifest['causal_split']['test_boundary_index']),
            full_generation_n=np.int64(len(y)),target_feedback_tail_removed=np.int64(np.count_nonzero(avail>issue[-1])),
            original_test_start_ns=np.int64(issue[0]),source_prefix_issue_count=np.int64(np.count_nonzero(issue<t0)),
            source_immature_excluded=np.int64(np.count_nonzero((issue<t0)&~source)))
        assert d['lower'].shape==(20,11,len(d['y']))
        assert np.isfinite(d['lower']).all() and np.isfinite(d['upper']).all()
        assert np.all(d['lower']<=d['upper']) and np.all(d['issue_ns']<d['label_ns'])
        assert np.all(d['label_ns']<d['available_ns'])
        np.testing.assert_array_equal(d['lower'][19],d['lower'][MAIN_CONFIG_INDICES].mean(axis=0))
        np.testing.assert_array_equal(d['upper'][19],d['upper'][MAIN_CONFIG_INDICES].mean(axis=0))
        if split=='source':
            assert d['issue_ns'].max()<t0 and d['available_ns'].max()<=t0 and d['label_ns'].max()<t0
        else:
            assert d['issue_ns'].min()>=t0
        result[split]=d
    return result,checks,hashes


def one(record):
    z,s,p,h=record; start=time.perf_counter(); sid=stream_id(z,s,p,h)
    src=BASE/p/sid/'base_predictions.parquet'
    identity=dict(base_sha256=sha256(src),code_sha256=sha256(__file__),
        conformal_sha256=sha256(RUN/'conformal_forward_v2.py'),cutoff_sha256=sha256(RUN/'chrono_cutoff.json'))
    paths=[dataset_path(z,s,p,h,k) for k in ('source','target')]
    metas=[]
    for path in paths:
        if not path.exists() or not path.with_suffix('.json').exists():break
        meta=json.loads(path.with_suffix('.json').read_text(encoding='utf8'))
        if any(meta.get(k)!=v for k,v in identity.items()) or sha256(path)!=meta['output_sha256']:break
        metas.append(meta)
    if len(metas)==2:return dict(stream=sid,resumed=True,seconds=time.perf_counter()-start,splits=dict(zip(('source','target'),metas)))
    manifest=json.loads(src.with_name('manifest.json').read_text(encoding='utf8'))
    assert manifest['split']=={'train_ratio':.6,'calibration_ratio':.2}
    data,checks,hashes=generate(pd.read_parquet(src),manifest,record)
    info=dict(stream=sid,original_candidate_checks=checks,splits={})
    for split,path in zip(('source','target'),paths):
        d=data[split]; path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.pending.npz'); np.savez_compressed(temp,**d); temp.replace(path)
        meta={k:int(v) for k,v in d.items() if np.ndim(v)==0}
        meta.update(identity);meta.update(hashes)
        meta.update(n=len(d['y']),first_issue=str(pd.Timestamp(d['issue_ns'][0])),
            last_issue=str(pd.Timestamp(d['issue_ns'][-1])),last_feedback=str(pd.Timestamp(d['available_ns'][-1])),
            output_sha256=sha256(path),original_candidate_checks=checks,
            state_history='Computed on full original test timeline before cutoff slicing',
            initialization='Original full calibration segment; same as sealed original candidates')
        path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf8');info['splits'][split]=meta
    info['seconds']=time.perf_counter()-start
    return info


def tsc_scores():
    records=[]; configs=conformal_grid(CONTRACTS)
    t0=int(json.loads((RUN/'chrono_cutoff.json').read_text())['t0_ns'])
    for z in ZONES:
        cs=np.zeros(19); ms=cs.copy(); cv=cs.copy(); gaps=cs.copy(); ts=cs.copy(); tc=cs.copy(); n=0
        for zz,s,p,h in streams(zones=[z]):
            d=load_all_configs(z,s,p,h,'source')
            assert np.all(d['issue_ns']<t0) and np.all(d['label_ns']<t0) and np.all(d['available_ns']<=t0)
            cap,miss,covered=endpoint_components(d['lower'][:19],d['upper'][:19],d['center'],d['y'])
            cs+=cap.sum(axis=(1,2));ms+=miss.sum(axis=(1,2));cv+=covered.sum(axis=(1,2))
            gaps+=(covered-COVERAGES[None,:,None]).sum(axis=(1,2))
            tu=d['candidate_tuwr'][:19];ts+=np.nansum(tu,axis=(1,2));tc+=np.isfinite(tu).sum(axis=(1,2))
            n+=len(d['y'])*len(COVERAGES)
        for i,cfg in enumerate(configs):
            records.append(dict(inner_validation_zone=z,configuration_id=cfg.configuration_id,complexity_rank=i,
                mean_errf=PI*(cs[i]+REFERENCE_PRICE*ms[i])/n,mean_coverage_gap=gaps[i]/n,
                mean_tuwr=ts[i]/tc[i],event_count=n,mean_coverage=cv[i]/n,
                capacity_sum=cs[i],miss_sum=ms[i],tuwr_sum=ts[i],tuwr_count=tc[i],source_cutoff_ns=t0))
        print('TSC cutoff-limited source scores',z,flush=True)
    pd.DataFrame(records).to_parquet(RUN/'tsc_zone_scores.parquet',index=False)
    choices=[]
    for outer in ZONES:
        for inner in (None,)+tuple(z for z in ZONES if z!=outer):
            excluded={outer} if inner is None else {outer,inner}
            choices.append(dict(outer=outer,inner=inner,index=select_tsc(excluded),**tsc_selection_details(excluded)))
    pd.DataFrame(choices).to_csv(RUN/'tsc_choices.csv',index=False)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--all',action='store_true')
    parser.add_argument('--scores-only',action='store_true');args=parser.parse_args()
    if args.scores_only:tsc_scores();return
    if not args.smoke and not args.all:parser.error('Use --smoke or explicitly --all')
    todo=[('zone1',0,'GBR',1)] if args.smoke else streams()
    if args.smoke:
        out=one(todo[0]);print(json.dumps(out,indent=2),flush=True)
        (RUN/'candidate_smoke.json').write_text(json.dumps(dict(status='PASS',**out),indent=2),encoding='utf8')
        return
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(one,r):r for r in todo}
        for i,f in enumerate(as_completed(futures),1):
            try:out=f.result()
            except Exception:
                print('FAILED',futures[f],traceback.format_exc(),flush=True);raise
            with (RUN/'data_progress.jsonl').open('a',encoding='utf8') as log:log.write(json.dumps(out)+'\n')
            print(f'{i}/{len(todo)} {out["stream"]} {out["seconds"]:.1f}s',flush=True)
    tsc_scores()


if __name__=='__main__':main()
