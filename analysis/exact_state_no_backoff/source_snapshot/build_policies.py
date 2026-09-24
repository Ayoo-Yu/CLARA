"""Source-only policies: disable support-driven broadening for nonempty states."""
from audit_support import *
import os, argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
core=rt.gc
NAMES={'risk_mean':'risk_mean','risk_parent':'risk_parent_mean','risk_shrunken':'risk_shrunken_mean','risk_se':'risk_zone_cluster_se','risk_score':'risk_score','coverage':'coverage_point','tuwr':'tuwr_point','ard':'ard_point'}

def matrices(f):
    f=f.sort_values(['full_state_code','action_order']).reset_index(drop=True)
    return {k:f[v].to_numpy(float).reshape(-1,6) for k,v in NAMES.items()}

def identity():
    return {'plan_sha256':sha(HERE/'PLAN.md'),'fit_code_sha256':sha(__file__),'support_audit_sha256':sha(HERE/'support_audit.json'),'accepted_freeze_sha256':sha(rt.GEOLD/'policies_frozen.json'),'selection_module_sha256':sha(Path(rt.agreed.__file__))}

def forced_profile(prepared,full,levels):
    profile=core.adaptive_v4.stitch_profile(prepared.profiles,full.selected_n_min.to_numpy())
    mapping=prepared.mappings[0]
    exact=levels[0]
    n=exact['event_count'][:,mapping].sum(0)
    nz=(exact['event_count'][:,mapping]>0).sum(0)
    use=n>0
    profile.loc[use,'support_backoff_level']=0
    profile.loc[use,'support_backoff_name']='exact'
    profile.loc[use,'selected_group_id']=mapping[use]
    profile.loc[use,'support_n']=n[use]
    profile.loc[use,'support_source_zones']=nz[use]
    # Audit metadata only: retain the original bootstrap-support definition.
    minzones=prepared.contracts.protocol['support_and_backoff_contract']['minimum_source_zones']
    gc=prepared.contracts.protocol['guardrail_contract']
    minspan=float(gc['dependence_robust_bounds']['primary_block_hours'])*float(gc['bootstrap_support_requirements']['minimum_time_span_in_primary_blocks'])
    for i in np.flatnonzero(use):
        gid=mapping[i];cold=bool(profile.loc[i,'cold_start_guardrail'])
        prefix='' if cold else 'guardrail_'
        cc=exact['event_count' if cold else 'guardrail_count'][:,gid]
        present=cc>0
        ok=int(present.sum())>=minzones
        if ok:
            span=(exact[prefix+'issue_max_us'][present,gid].astype(float)-exact[prefix+'issue_min_us'][present,gid].astype(float))/3_600_000_000
            ok=float(span.min())>=minspan
        profile.loc[i,'bootstrap_supported']=ok
    return profile,n,nz

def worker(task):
    zone,seed=task;uid=f'{zone}__seed{seed}';dest=HERE/'policies'/uid
    ident=identity()
    if (dest/'manifest.json').exists():
        m=read(dest/'manifest.json');assert m['identity']==ident and m['status']=='PASS'
        for name,h in m['outputs'].items():assert sha(dest/name)==h
        return {'unit':uid,'status':'REUSED'}
    dest.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    cache=ROOT/'analysis/reviewer_closure_20260905/policy/parents'/uid/'source_cache'
    cm=read(cache/'manifest.json');stats,_=closure.read_source_cache(cache,cm['source_identity'])
    accepted=read(rt.GEOLD/'policies_frozen.json')
    assert cm['source_identity']['endpoint_root_manifest_sha256']==accepted['endpoint_root_sha256']
    contracts=core.six_action_contracts(core.load_frozen_contracts());config=read(core.FOUR_VERSION_CONFIG_PATH)
    prepared=core.prepare_clara_fit(stats,contracts=contracts,v4_config=config)
    records=[];regression=[];all_evidence=[];all_traces=[]
    freqmeta=pd.read_parquet(HERE/'exact_support_by_state.parquet',filters=[('zone','==',zone),('seed','==',seed)]).sort_values('full_state_code')
    for price in stats.prices:
        # Rebuild the full source fit, including adaptive parameters, from sealed sufficient statistics.
        fitted=core.fit_price_conditioned_clara(prepared,price=price,v4_config=config)
        fd=fitted.decisions.sort_values('full_state_code').reset_index(drop=True)
        fe=matrices(fitted.action_evidence)
        old,oe,audit,om=rt.original_ge(zone,seed,price.price_id)
        for field in ['full_state_code','selected_n_min','selected_nu','support_backoff_level']:
            np.testing.assert_array_equal(fd[field],old[field])
        maxerr=0.
        for k in NAMES:
            np.testing.assert_allclose(fe[k],oe[k],atol=0,rtol=0,equal_nan=True)
            maxerr=max(maxerr,float(np.nanmax(abs(fe[k]-oe[k]))))
        full,fulltrace=rt.agreed.select(fd,fe,audit['adaptive_guardrail_thresholds'],'DIRECTIONAL_REPAIR',1.)
        accepted_dir=rt.GEOLD/'policies'/uid/price.price_id
        accepted_manifest=read(accepted_dir/'manifest.json')
        assert accepted_manifest['source_manifest_sha256']==sha(rt.GEBASE/'fit_parents'/uid/'children'/price.price_id/'manifest.json')
        assert sha(accepted_dir/'state_decisions.parquet')==accepted_manifest['outputs']['state_decisions.parquet']
        oldchoices=pd.read_parquet(accepted_dir/'state_decisions.parquet').sort_values('full_state_code')
        np.testing.assert_array_equal(full,oldchoices[rt.agreed.PRIMARY])

        levels=core.landscape_accelerator.build_level_aggregates(stats.arrays_for_price(price),list(prepared.mappings),list(prepared.group_frames))
        profile,n,nz=forced_profile(prepared,fd,levels)
        frozen_nu=fd.selected_nu.to_numpy()
        # Calling the same existing evidence builder preserves the shrinkage/SE arithmetic exactly.
        evidence_by_nu={}
        for nu in sorted(set(frozen_nu.tolist())):
            cfg=core.adaptive_v4.parameterized_contracts(contracts,n_min=int(config['fixed_support']['n_min']),nu=int(nu))
            evidence_by_nu[nu]=core.landscape_accelerator.compute_cache_evidence(profile=profile,contracts=cfg,mappings=list(prepared.mappings),levels=levels)
        oldactions=core.adaptive_v4.ACTIONS
        try:
            core.adaptive_v4.ACTIONS=core.SIX_ACTIONS
            evidence=core.adaptive_v4.stitch_evidence(evidence_by_nu,frozen_nu)
        finally:core.adaptive_v4.ACTIONS=oldactions
        ee=matrices(evidence)
        for k in ['risk_mean','risk_parent','risk_shrunken','risk_se','risk_score','coverage']:
            assert np.isfinite(ee[k]).all(),(uid,price.price_id,k)
        warm=fd.rolling_state.ne('cold_start').to_numpy()
        assert np.isfinite(ee['tuwr'][warm]).all() and np.isfinite(ee['ard'][warm]).all()
        positive=n>0
        poschoice,postrace=rt.agreed.select(fd.loc[positive].reset_index(drop=True),{k:v[positive] for k,v in ee.items()},audit['adaptive_guardrail_thresholds'],'DIRECTIONAL_REPAIR',1.)
        chosen=np.zeros(len(fd),dtype=int)  # User-specified Static for wholly absent exact history.
        chosen[positive]=poschoice
        trace={k:np.full(len(fd),np.nan) for k in postrace}
        for k,v in postrace.items():trace[k][positive]=v
        unchanged=fd.support_backoff_level.eq(0).to_numpy()
        for k in NAMES:np.testing.assert_allclose(ee[k][unchanged],fe[k][unchanged],atol=0,rtol=0,equal_nan=True)
        np.testing.assert_array_equal(chosen[unchanged],full[unchanged])
        np.testing.assert_array_equal(profile.loc[n>0,'support_backoff_level'],0)
        np.testing.assert_array_equal(profile.loc[n>0,'support_n'],n[n>0])
        q=fd[['full_state_code']+rt.FIELDS].copy()
        q['price_id']=price.price_id;q['CLARA']=full;q['NO_BACKOFF_EXACT']=chosen
        q['exact_count']=n;q['exact_zone_count']=nz;q['exact_guard_count']=levels[0]['guardrail_count'][:,prepared.mappings[0]].sum(0)
        q['was_backoff']=fd.support_backoff_level.gt(0).to_numpy();q['original_backoff_level']=fd.support_backoff_level
        q['exact_zero']=n==0;q['zero_static_fallback']=n==0;q['one_zone_se_omitted']=nz==1
        q['ablated_backoff_level']=np.where(positive,0,-1)
        q['selected_n_min']=fd.selected_n_min;q['selected_nu']=frozen_nu
        np.testing.assert_array_equal(freqmeta.full_state_code,q.full_state_code)
        q['target_event_count_one_price']=freqmeta.target_event_count_one_price.to_numpy()
        records.append(q)
        zero_rows=np.repeat(n==0,6)
        # Zero-history rows have a default action, not fabricated exact-state scores.
        for column in list(NAMES.values())+['coverage_lcb','tuwr_ucb','ard_ucb']:
            evidence.loc[zero_rows,column]=np.nan
        evidence.loc[zero_rows,'support_backoff_level']=-1
        evidence.loc[zero_rows,'support_backoff_name']='zero_history_Static'
        evidence.loc[zero_rows,'selected_group_id']=-1
        evidence.loc[zero_rows,'support_n']=0
        evidence.loc[zero_rows,'support_source_zones']=0
        evidence.loc[zero_rows,'bootstrap_supported']=False
        evidence.loc[zero_rows,'action_evidence_level']='ZERO_HISTORY_STATIC_DEFAULT'
        evidence.loc[zero_rows,'action_bound_method']='NOT_ESTIMATED'
        evidence['zero_static_fallback']=zero_rows
        evidence.insert(0,'price_id',price.price_id);all_evidence.append(evidence)
        tr=pd.DataFrame(trace);tr['price_id']=price.price_id;tr['full_state_code']=fd.full_state_code;all_traces.append(tr)
        assert (chosen[n==0]==0).all()
        regression.append({'price_id':price.price_id,'full_evidence_max_abs_error':maxerr,'full_parameters_exact':True,'accepted_CLARA_choices_exact':True,'unchanged_states_evidence_and_choices_exact':True,'zero_history_Static_exact':True,'new_target_outcomes_used':False,'source_fit_manifest_sha256':sha(rt.GEBASE/'fit_parents'/uid/'children'/price.price_id/'manifest.json'),'accepted_policy_manifest_sha256':sha(accepted_dir/'manifest.json')})
    out=pd.concat(records,ignore_index=True)
    assert len(out)==33000 and not out.duplicated(['price_id','full_state_code']).any()
    assert out[['CLARA','NO_BACKOFF_EXACT']].isin(range(6)).all().all()
    out.to_parquet(dest/'decisions.parquet',index=False)
    pd.concat(all_evidence,ignore_index=True).to_parquet(dest/'action_evidence.parquet',index=False)
    pd.concat(all_traces,ignore_index=True).to_parquet(dest/'constraint_audit.parquet',index=False)
    save(dest/'manifest.json',{'status':'PASS','identity':ident,'heldout_zone':zone,'seed':seed,'source_zones':list(stats.source_zones),'source_cache_manifest_sha256':sha(cache/'manifest.json'),'source_arrays_sha256':sha(cache/'arrays.npz'),'source_statistic_identity_sha256':cm['source_statistic_identity_sha256'],'endpoint_root_sha256':accepted['endpoint_root_sha256'],'test_outcomes_used_in_fit':False,'outputs':{p.name:sha(p) for p in dest.glob('*.parquet')},'regression':regression,'seconds':time.monotonic()-start})
    return {'unit':uid,'status':'PASS','seconds':round(time.monotonic()-start,1)}

def freeze():
    parents={};ident=identity();regs=[]
    for zone in rt.ge.ZONES:
        for seed in range(3):
            p=HERE/'policies'/f'{zone}__seed{seed}'/'manifest.json';m=read(p)
            assert m['status']=='PASS' and m['identity']==ident
            assert not m['test_outcomes_used_in_fit']
            for n,h in m['outputs'].items():assert sha(p.parent/n)==h
            parents[str(p.relative_to(HERE))]=sha(p);regs.extend(m['regression'])
    assert len(regs)==150
    save(HERE/'policies_frozen.json',{'status':'PASS','policy_parents':parents,'identity':ident,'plan':sha(HERE/'PLAN.md'),'fit_code':sha(__file__),'target_metrics_evaluated':False,'fit_units':150,'state_price_rows':990000,'full_evidence_max_abs_error':max(r['full_evidence_max_abs_error'] for r in regs),'all_original_parameters_and_accepted_choices_exact':True,'zero_history_handling':'Static action index 0, explicitly requested by user before outcomes','methods':['CLARA','NO_BACKOFF_EXACT']})
    print('ALL_POLICIES_FROZEN',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=3);ap.add_argument('--pilot',action='store_true');ap.add_argument('--seal',action='store_true');args=ap.parse_args()
    if args.seal:freeze()
    else:
        tasks=[(z,s) for z in rt.ge.ZONES for s in range(3)]
        if args.pilot:tasks=tasks[:1]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for f in as_completed([pool.submit(worker,t) for t in tasks]):print(json.dumps(f.result()),flush=True)
