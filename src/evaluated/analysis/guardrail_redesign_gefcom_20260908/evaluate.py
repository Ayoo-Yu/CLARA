"""Full sealed-target replay of the prespecified GEFCom guardrail policies."""
from gefcom_guardrail_runtime import *
import argparse,time,traceback
from concurrent.futures import ProcessPoolExecutor,as_completed

def signature(ids):
    assert ids.notna().all() and not ids.duplicated().any()
    raw=json.dumps(sorted(ids.astype(str).tolist()),ensure_ascii=False,sort_keys=True,separators=(',',':'),default=str).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()

def load_stream(zone,seed,p,h,target_manifest,thresholds):
    audit=target_manifest['stream_audits'][f'H{h:02d}__{p}']
    assert audit['status']=='PASS' and audit['heldout_zone_used_only_as_target']
    paths={
        'source_fact':FACTROOT/'source_facts'/f'predictor={p}'/f'zone={zone}'/f'horizon={h:02d}'/f'seed={seed}'/'facts.parquet',
        'event_core':BUNDLES/p/f'{p}-H{h:02d}-{zone}-S{seed}'/'event_core.parquet',
        'endpoint_addon':RUN/'endpoint_units'/f'{zone}__seed{seed}__H{h:02d}'/'endpoint_addons.parquet',
    }
    for k,path in paths.items():checked(path,audit['leaf_sha256'][k])
    cols=['event_id','issue_timestamp',*FIELDS[:-1],'raw_width_value',*[a+'__candidate_'+edge for a in ACTIONS[:4] for edge in ['lower','upper']]]
    f=pd.read_parquet(paths['source_fact'],columns=cols)
    a=pd.read_parquet(paths['endpoint_addon'],filters=[('predictor','==',p)],columns=['event_id',*[a+'__candidate_'+edge for a in ACTIONS[4:] for edge in ['lower','upper']]])
    f=f.merge(a,on='event_id',how='inner',validate='one_to_one')
    assert len(f)==len(a)==audit['target_event_count']
    assert signature(f.event_id)==audit['event_signature']['canonical_sorted_event_id_sha256']
    core=pd.read_parquet(paths['event_core'],columns=['event_id','target','base_center'])
    f=f.merge(core,on='event_id',how='left',validate='one_to_one')
    assert f[['target','base_center']].notna().all().all()
    t=thresholds[(thresholds.outer_heldout_zone==zone)&(thresholds.seed==seed)&(thresholds.predictor==p)]
    assert not t.heldout_zone_in_threshold.any()
    keys=['predictor','horizon_group','target_coverage']
    f=f.merge(t[keys+['raw_width_q33','raw_width_q67']],on=keys,how='left',validate='many_to_one')
    assert f[['raw_width_q33','raw_width_q67']].notna().all().all()
    f['raw_width_state']=np.where(f.raw_width_value<=f.raw_width_q33,'narrow',np.where(f.raw_width_value<=f.raw_width_q67,'medium','wide'))
    f=f.sort_values(['target_coverage','issue_timestamp','event_id'],kind='mergesort').reset_index(drop=True)
    assert set(f.target_coverage)==set(COVERAGES)
    for c,g in f.groupby('target_coverage',sort=False):
        cid=f'H{h:02d}__{p}__C'+format(c,'.12g').replace('.','p')
        ref=target_manifest['partition_records'][cid]
        assert len(g)==ref['event_count'] and signature(g.event_id)==ref['canonical_sorted_event_id_sha256']
    return f,{str(path.relative_to(WORK)):audit['leaf_sha256'][k] for k,path in paths.items()}

def rolling(covered,c):
    n,m=covered.shape
    out=np.full((n,m,5),np.nan)
    if n<168:return out
    cs=np.vstack([np.zeros((1,m),np.int64),np.cumsum(covered,dtype=np.int64,axis=0)])
    gap=(cs[168:]-cs[:-168])/168-float(c)
    tol=1.96*np.sqrt(c*(1-c)/168)
    out[167:,:,0]=gap < -tol
    out[167:,:,1]=gap > tol
    out[167:,:,2]=abs(gap)
    out[167:,:,3]=np.maximum(-gap,0)
    out[167:,:,4]=np.maximum(gap,0)
    return out

def event_arrays(f,choice,theta):
    low=f[[a+'__candidate_lower' for a in ACTIONS]].to_numpy(float)
    up=f[[a+'__candidate_upper' for a in ACTIONS]].to_numpy(float)
    assert np.isfinite(low).all() and np.isfinite(up).all() and (low<=up).all()
    y=f.target.to_numpy(float)[:,None];center=f.base_center.to_numpy(float)[:,None]
    pu,pdwn,ku,kd=theta
    cap=pu*np.maximum(up-center,0)+pdwn*np.maximum(center-low,0)
    miss=ku*np.maximum(y-up,0)+kd*np.maximum(low-y,0)
    candidates=[cap+miss,cap,miss,((y>=low)&(y<=up)).astype(float),up-low]
    return np.stack([np.take_along_axis(v,choice,axis=1) for v in candidates],axis=2)

def expected_sources(zone,seed):
    agg=readj(AGG/'manifest.json');entry=agg['input_replay_parents'][f'{zone}__seed{seed}']
    target_path=BASE/'replay_parents'/f'{zone}__seed{seed}'/'target_input_manifest.json'
    refs={};counts={};identities={}
    for pid,_ in PRICES:
        key=MODE+'__'+pid;rec=entry['policy_evaluation_children'][key]
        path=BASE/rec['manifest_relative_path'];checked(path,rec['manifest_sha256'])
        m=readj(path);assert m['status']=='PASS'
        checked(target_path,m['target_input_manifest_sha256'])
        fpath=BASE/'fit_parents'/f'{zone}__seed{seed}'/'children'/pid/'manifest.json'
        checked(fpath,m['fit_child_manifest_sha256'])
        for k in ['diagnostic_metrics','action_counts']:
            ff=path.parent/m['files'][k]['relative_path'];checked(ff,m['files'][k]['sha256'])
            identities[str(ff.relative_to(WORK))]=m['files'][k]['sha256']
        d=pd.read_parquet(path.parent/'diagnostic_metrics.parquet')
        refs[pid]=d.set_index(['predictor','horizon_steps','target_coverage','regime','method']).sort_index()
        counts[pid]=pd.read_parquet(path.parent/'action_counts.parquet')
    return readj(target_path),refs,counts,identities

def worker(task):
    zone,seed=task;uid=f'{zone}__seed{seed}';dest=HERE/'evaluation'/uid
    identity=dict(policy_freeze_sha256=sha(HERE/'policies_frozen.json'),runner_sha256=sha(__file__))
    if (dest/'manifest.json').exists():
        m=readj(dest/'manifest.json');assert m['status']=='PASS' and m['identity']==identity
        for n,v in m['outputs'].items():checked(dest/n,v)
        return dict(unit=uid,status='REUSED')
    start=time.monotonic();dest.mkdir(parents=True,exist_ok=True)
    try:
        frozen=readj(HERE/'policies_frozen.json')
        assert frozen['status']=='PASS' and frozen['plan_sha256']==sha(HERE/'PLAN.md')
        assert frozen['runtime_sha256']==sha(HERE/'gefcom_guardrail_runtime.py') and frozen['agreed_policy_sha256']==sha(EXTERNAL/'policies.py')
        checked(AGG/'manifest.json',frozen['aggregate_manifest_sha256'])
        target,refs,refcounts,inputs=expected_sources(zone,seed)
        tp=FACTROOT/'fold_width_thresholds.parquet'
        assert len({v['leaf_sha256']['width_thresholds'] for v in target['stream_audits'].values()})==1
        checked(tp,next(iter(target['stream_audits'].values()))['leaf_sha256']['width_thresholds'])
        thresholds=pd.read_parquet(tp)
        policies={};audits={};prices={}
        for pid,_ in PRICES:
            pf=policy_dir(zone,seed,pid);pm=readj(pf/'manifest.json')
            for n,v in pm['outputs'].items():checked(pf/n,v)
            policies[pid]=pd.read_parquet(pf/'state_decisions.parquet')
            af=pd.read_parquet(pf/'constraint_audit.parquet')
            aa=np.full((len(policies[pid]),len(METHODS),len(AUDIT_FIELDS)),np.nan)
            for j,m in enumerate(METHODS):
                if m.startswith(tuple(agreed.FAMILIES)):
                    aa[:,j]=af[af.method==m].set_index('full_state_code').reindex(policies[pid].full_state_code)[AUDIT_FIELDS].to_numpy(float)
            audits[pid]=aa;prices[pid]=pm['price']['theta']
        out=[];acrows=[];stream_receipts=[];max_error=0.;checks=0
        method_indices=np.array([METHODS.index(m) for m in OFFICIAL])
        official_order=[OFFICIAL[m] for m in OFFICIAL]
        for h in HORIZONS:
            for p in PREDICTORS:
                f,source_ids=load_stream(zone,seed,p,h,target,thresholds);inputs.update(source_ids)
                counts_check=[]
                for pid,_ in PRICES:
                    pol=policies[pid]
                    ix=pd.MultiIndex.from_frame(pol[FIELDS]).get_indexer(pd.MultiIndex.from_frame(f[FIELDS]));assert (ix>=0).all()
                    choices=pol[METHODS].to_numpy(int)[ix]
                    values=event_arrays(f,choices,prices[pid])
                    for c,g in f.groupby('target_coverage',sort=False):
                        pos=g.index.to_numpy();full_roll=rolling(values[pos,:,3],c)
                        for regime in REGIMES:
                            mask=np.ones(len(g),bool) if regime=='overall' else g.ramp_state.eq(regime).to_numpy()
                            at=pos[mask];n=len(at)
                            if n==0:continue
                            nr=max(0,len(g)-167) if regime=='overall' else int((np.arange(len(g))[mask]>=167).sum())
                            legacy=full_roll if regime=='overall' else rolling(values[at,:,3],c)
                            nlegacy=max(0,n-167)
                            means=np.concatenate([values[at].mean(0),np.nansum(full_roll[mask],axis=0)/nr if nr else np.full((len(METHODS),5),np.nan)],axis=1)
                            lm=np.nansum(legacy,axis=0)/nlegacy if nlegacy else np.full((len(METHODS),5),np.nan)
                            np.testing.assert_allclose(means[:,0],means[:,1]+means[:,2],rtol=0,atol=1e-12)
                            np.testing.assert_allclose(means[:,7],means[:,8]+means[:,9],rtol=0,atol=1e-12,equal_nan=True)
                            trace=np.nan_to_num(audits[pid][ix[at]],nan=0).mean(0)
                            for j,m in enumerate(METHODS):
                                row=dict(zone_or_farm=zone,seed=seed,price_id=pid,predictor=p,horizon_steps=h,target_coverage=float(c),regime=regime,method=m,event_count=n,reliability_event_count=nr,legacy_reliability_event_count=nlegacy)
                                row.update(dict(zip(METRICS,means[j])))
                                row.update({'legacy_'+k:lm[j,z] for z,k in enumerate(METRICS[5:])})
                                row.update(dict(zip(AUDIT_FIELDS,trace[j])))
                                out.append(row)
                            r=refs[pid].loc[(p,h,float(c),regime)].reindex(official_order)
                            assert (r.event_count==n).all() and (r.rolling_window_count==nlegacy).all()
                            raw=r[['errf_sum','reserve_up_sum','reserve_down_sum','miss_upper_sum','miss_lower_sum','covered_sum','width_sum']].to_numpy(float)/n
                            expected=np.column_stack([raw[:,0],prices[pid][0]*raw[:,1]+prices[pid][1]*raw[:,2],prices[pid][2]*raw[:,3]+prices[pid][3]*raw[:,4],raw[:,5],raw[:,6]])
                            errors=np.abs(means[method_indices,:5]-expected)
                            max_error=max(max_error,float(np.nanmax(errors)))
                            np.testing.assert_allclose(means[method_indices,:5],expected,rtol=0,atol=1e-10)
                            np.testing.assert_allclose(lm[method_indices,:3],r[['TUWR','TOWR','ARD']].to_numpy(float),rtol=0,atol=1e-10,equal_nan=True)
                            if regime=='overall':np.testing.assert_allclose(means[method_indices,5:8],lm[method_indices,:3],rtol=0,atol=1e-12,equal_nan=True)
                            checks+=len(OFFICIAL)
                            for m in [ORIGINAL,PRIMARY]:
                                j=METHODS.index(m);nc=np.bincount(choices[at,j],minlength=6)
                                acrows.extend([dict(zone_or_farm=zone,seed=seed,price_id=pid,predictor=p,horizon_steps=h,target_coverage=float(c),regime=regime,method=m,selected_action=a,event_count=int(v)) for a,v in zip(ACTIONS,nc)])
                                if m==ORIGINAL and regime!='overall':
                                    rr=refcounts[pid]
                                    rr=rr[(rr.predictor==p)&(rr.horizon_steps==h)&(rr.target_coverage==c)&(rr.ramp_state==regime)&(rr.method=='CLARA_6A')]
                                    ref=rr.set_index('selected_action').action_count.reindex(ACTIONS,fill_value=0).to_numpy(int)
                                    np.testing.assert_array_equal(nc,ref)
                stream_receipts.append(dict(predictor=p,horizon_steps=h,event_count=len(f),event_signature=signature(f.event_id)))
            log('horizon_complete',unit=uid,horizon=h)
        cells=pd.DataFrame(out);assert len(cells)==5*220*len(METHODS)*3
        cells.to_parquet(dest/'cell_metrics.parquet',index=False)
        pd.DataFrame(acrows).to_parquet(dest/'action_counts.parquet',index=False)
        total=sum(v['event_count'] for v in stream_receipts)
        assert total==target['combined_event_signature']['event_count']
        receipt=dict(status='PASS',unit=uid,identity=identity,completed_at=stamp(),events_per_price=total,streams=stream_receipts,source_files=inputs,original_metric_cell_checks=checks,original_metric_max_error=max_error,original_actions_exact=True,legacy_regime_metrics_reproduced=True,full_time_regime_metrics_separately_retained=True,elapsed_seconds=time.monotonic()-start,outputs={n:sha(dest/n) for n in ['cell_metrics.parquet','action_counts.parquet']})
        writej(dest/'manifest.json',receipt)
        return {k:receipt[k] for k in ['unit','status','events_per_price','original_metric_cell_checks','original_metric_max_error','elapsed_seconds']}
    except Exception:
        writej(dest/'exception.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    tasks=[(z,s) for z in ZONES for s in range(3)]
    if a.pilot:tasks=[('zone1',0)]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for done in as_completed([pool.submit(worker,t) for t in tasks]):log('unit_complete',**done.result())
    if not a.pilot:
        receipts=[readj(HERE/'evaluation'/f'{z}__seed{s}'/'manifest.json') for z,s in tasks]
        assert sum(v['events_per_price'] for v in receipts)==23024760
        writej(HERE/'evaluation_manifest.json',dict(status='PASS',units=30,events_per_method_price=23024760,completed_at=stamp(),policy_freeze_sha256=sha(HERE/'policies_frozen.json'),runner_sha256=sha(__file__),max_original_metric_error=max(v['original_metric_max_error'] for v in receipts)))

if __name__=='__main__':main()
