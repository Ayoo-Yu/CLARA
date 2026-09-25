"""Independent add-on: preserve issued but immature pre-cutoff observations.

This does not change source fitting or the main target evaluation. It enables
continuous warm-start replay by restoring the pending feedback queue at T0.
"""
from forward_common import *
from prepare_chrono import ns, quantiles, rolling_summaries
from conformal_forward_v2 import evaluate_conformal_grid_compact,build_feedback_schedule
import argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed


def one(record):
    z,s,p,h=record;sid=stream_id(z,s,p,h);start=time.perf_counter()
    t0=int(json.loads((RUN/'chrono_cutoff.json').read_text())['t0_ns'])
    src=BASE/p/sid/'base_predictions.parquet'
    path=dataset_path(z,s,p,h,'prefix_pending');path.parent.mkdir(parents=True,exist_ok=True)
    identity=dict(code_sha256=sha256(__file__),base_sha256=sha256(src),
        source_dataset_sha256=sha256(dataset_path(z,s,p,h,'source')),
        target_dataset_sha256=sha256(dataset_path(z,s,p,h,'target')),
        conformal_sha256=sha256(RUN/'conformal_forward_v2.py'),
        prepare_chrono_sha256=sha256(Path(__file__).resolve().parent / 'prepare_chrono.py'),cutoff_sha256=sha256(RUN/'chrono_cutoff.json'))
    if path.exists() and path.with_suffix('.json').exists():
        prior=json.loads(path.with_suffix('.json').read_text())
        if all(prior.get(k)==v for k,v in identity.items()) and sha256(path)==prior['output_sha256']:
            return dict(stream=sid,resumed=True,seconds=time.perf_counter()-start,**prior)
    raw=pd.read_parquet(src).sort_values('issue_timestamp')
    calib=raw[raw.split.eq('calibration')]
    test=raw[raw.split.eq('test')];test=test[ns(test.issue_timestamp)<t0]
    issue=ns(test.issue_timestamp);labels=ns(test.label_timestamp);avail=ns(test.label_available_timestamp)
    assert len(issue)==681 and np.all(issue<t0)
    warm=calib[ns(calib.label_available_timestamp)<=issue[0]]
    lo,up=quantiles(test);wlo,wup=quantiles(warm)
    order,ends,_=build_feedback_schedule(issue,labels,avail)
    y=test.target.to_numpy(float);center=test.base_center.to_numpy(float)
    lower,upper=evaluate_conformal_grid_compact(contracts=CONTRACTS,coverages=COVERAGES,
        calibration_target=warm.target.to_numpy(float),calibration_lower=wlo,calibration_upper=wup,
        calibration_center=warm.base_center.to_numpy(float),calibration_label_ns=ns(warm.label_timestamp),
        test_target=y,test_lower=lo,test_upper=up,test_center=center,test_issue_ns=issue,
        test_label_ns=labels,feedback_order=order,matured_ends=ends,cadence_hours=1.)[:2]
    lower=np.concatenate([lower,lower[MAIN_CONFIG_INDICES].mean(axis=0,keepdims=True)])
    upper=np.concatenate([upper,upper[MAIN_CONFIG_INDICES].mean(axis=0,keepdims=True)])
    covered=(lower<=y)&(y<=upper);states=np.zeros((11,len(y)),np.int8)
    ct=np.full(lower.shape,np.nan);ca=np.full(lower.shape,np.nan)
    release_ends=np.minimum(np.searchsorted(avail,issue,'right'),np.searchsorted(labels,issue,'left'))
    feedback_ends=np.arange(1,len(y)+1)
    for ci,c in enumerate(COVERAGES):
        states[ci]=rolling_summaries(labels,covered[0,ci],c,release_ends,issue)[-1]
        for ai in range(20):
            _,tu,_,ard,_=rolling_summaries(labels,covered[ai,ci],c,feedback_ends,avail)
            ct[ai,ci]=np.where(states[ci]!=0,tu,np.nan);ca[ai,ci]=np.where(states[ci]!=0,ard,np.nan)
    fields=dict(issue_ns=issue,label_ns=labels,available_ns=avail,y=y,center=center,
        raw_width=up-lo,ramp=(np.abs(center-test.feature_target_lag1.to_numpy(float))>=.12).astype(np.int8),
        rolling_state=states,lower=lower,upper=upper,candidate_tuwr=ct,candidate_ard=ca)
    matured=(labels<t0)&(avail<=t0);pending=~matured
    assert pending.sum()==h
    prior_source=load_all_configs(z,s,p,h,'source')
    for key,array in fields.items():np.testing.assert_array_equal(array[...,matured],prior_source[key])
    d={key:array[...,pending] for key,array in fields.items()}
    d.update(source_cutoff_ns=np.int64(t0),initialization_available_max=np.int64(ns(warm.label_available_timestamp).max()),
        initialization_n=np.int64(len(warm)),original_test_start_ns=np.int64(issue[0]),
        full_generation_n=np.int64(len(y)),source_available_max=np.int64(avail[pending].max()))
    assert np.all(d['issue_ns']<t0) and np.all((d['available_ns']>t0)|(d['label_ns']>=t0))
    target=load_all_configs(z,s,p,h,'target')
    combined=np.sort(np.r_[prior_source['issue_ns'],d['issue_ns'],target['issue_ns']])
    assert len(np.unique(combined))==len(combined) and np.all(np.diff(combined)==HOUR_NS)
    # Original four sealed actions provide an independent endpoint check for
    # every restored pending observation, beyond the source-prefix invariance.
    bd=BUNDLES/p/sid
    ev=pd.read_parquet(bd/'event_core.parquet',columns=['event_id','issue_timestamp','target_coverage'])
    cs=pd.read_parquet(bd/'candidate_intervals.parquet',columns=['event_id','action','candidate_lower','candidate_upper'])
    ev=ev[(ns(ev.issue_timestamp)>=d['issue_ns'][0])&(ns(ev.issue_timestamp)<t0)]
    joined=ev.merge(cs,on='event_id',validate='one_to_many');max_error=0.
    for ai,action in zip(MAIN_CONFIG_INDICES,ACTIONS[:4]):
        for ci,c in enumerate(COVERAGES):
            old=joined[(joined.action==action)&np.isclose(joined.target_coverage,c)].sort_values('issue_timestamp')
            np.testing.assert_array_equal(ns(old.issue_timestamp),d['issue_ns'])
            expected=old[['candidate_lower','candidate_upper']].to_numpy(float)
            actual=np.column_stack([d['lower'][ai,ci],d['upper'][ai,ci]])
            np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0)
            max_error=max(max_error,float(abs(actual-expected).max()))
    temp=path.with_suffix('.pending.npz');np.savez_compressed(temp,**d);temp.replace(path)
    meta=dict(identity,n=int(pending.sum()),source_mature_n=int(matured.sum()),
        t0_ns=t0,first_issue=str(pd.Timestamp(d['issue_ns'][0])),last_issue=str(pd.Timestamp(d['issue_ns'][-1])),
        first_feedback=str(pd.Timestamp(d['available_ns'][0])),last_feedback=str(pd.Timestamp(d['available_ns'][-1])),
        prefix_only_source_bitwise_equal=True,original_pending_max_error=max_error,
        output_sha256=sha256(path),combined_complete_timeline_n=len(combined),
        purpose='Warm-start pending feedback only; excluded from source fitting and primary evaluation')
    path.with_suffix('.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    return dict(stream=sid,seconds=time.perf_counter()-start,**meta)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=4);ap.add_argument('--smoke',action='store_true')
    args=ap.parse_args()
    if not (RUN/'DATA_READY.json').exists():raise RuntimeError('Main source/target datasets must pass audit first')
    tasks=[('zone1',0,'GBR',1),('zone1',1,'MLP',24)] if args.smoke else streams()
    rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(one,t) for t in tasks]
        for i,f in enumerate(as_completed(futures),1):
            r=f.result();rows.append(r)
            with (RUN/'pending_progress.jsonl').open('a',encoding='utf8') as log:log.write(json.dumps(r)+'\n')
            if i%20==0 or args.smoke:print(i,len(tasks),r['stream'],f"{r['seconds']:.2f}s",flush=True)
    result=dict(status='PASS',streams=len(rows),pending_time_records=sum(r['n'] for r in rows),
        pending_events=sum(r['n'] for r in rows)*11,source_recomputed_bitwise_equal=True,
        original_four_pending_max_error=max(r['original_pending_max_error'] for r in rows),
        complete_timeline_unique_and_contiguous=True,code_sha256=sha256(__file__),
        main_data_ready_sha256=sha256(RUN/'DATA_READY.json'),
        metadata_identity=hashlib.sha256(json.dumps(sorted(rows,key=lambda r:r['stream']),sort_keys=True).encode()).hexdigest())
    if not args.smoke:
        assert len(rows)==600 and result['pending_time_records']==5520
        (RUN/'prefix_pending_identity_manifest.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
        (RUN/'PREFIX_PENDING_READY.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    else:(RUN/'prefix_pending_smoke.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
