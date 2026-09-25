"""Chronological-source ablations and explicitly nondeployable hindsight bounds.

All fitted quantities come from sealed CHRONO source sufficient statistics.
The original analyses are consulted for definitions only, never result inputs.
"""
from pathlib import Path
import os,sys,json,time,copy,argparse,itertools
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor,as_completed
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
REPOSITORY=HERE.parents[1]
sys.path.insert(0,str(REPOSITORY/'src/chronological'))
from _paths import DATA_ROOT as CHRONO, EVALUATED as ROOT
HERE=Path(os.environ.get('CLARA_ABLATION_OUTPUT',str(REPOSITORY/'outputs/chronological_ablation')))
HERE.mkdir(parents=True,exist_ok=True)
import forward_common as fc
import clara_forward as cf
core=cf.original
METHODS=['CLARA','GLOBAL_SCORE_SAME_SCREEN','WIDTH_SAME_SCREEN','NO_G','NO_RWG','NO_SCREEN','NO_BACKOFF_EXACT','FIX_REFERENCE_PRICE','WIDTH_ONLY','STATE_ORACLE','FEASIBLE_STATE_ORACLE','OUTCOME_ORACLE']
M=['ERRF','capacity','exceedance','coverage','width','TUWR','TOWR','ARD','under_gap','over_gap','IS']
PRIMARY='R05P6179775281'
EV_NAMES={'risk_mean':'risk_mean','risk_parent':'risk_parent_mean','risk_shrunken':'risk_shrunken_mean','risk_se':'risk_zone_cluster_se','risk_score':'risk_score','coverage':'coverage_point','tuwr':'tuwr_point','ard':'ard_point'}

def save(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=str),encoding='utf8')
def read(p):return json.loads(p.read_text(encoding='utf8'))
def matrices(frame):
    f=frame.sort_values(['full_state_code','action_order']).reset_index(drop=True)
    return {k:f[v].to_numpy(float).reshape(6600,6) for k,v in EV_NAMES.items()}
def variant_contract(contracts,removed):
    protocol=copy.deepcopy(contracts.protocol);levels=[];seen=set()
    for level in protocol['support_and_backoff_contract']['levels']:
        fields=tuple(f for f in level['fields'] if f not in removed)
        if fields in seen:continue
        seen.add(fields);levels.append(dict(level=len(levels),name='retained_'+'_'.join(fields) if fields else 'global',fields=list(fields)))
    protocol['support_and_backoff_contract']['levels']=levels
    return replace(contracts,protocol=protocol)

def exact_profile(prepared,full,levels):
    profile=core.adaptive_v4.stitch_profile(prepared.profiles,full.selected_n_min.to_numpy())
    mapping=prepared.mappings[0];exact=levels[0]
    n=exact['event_count'][:,mapping].sum(0);nz=(exact['event_count'][:,mapping]>0).sum(0);use=n>0
    profile.loc[use,'support_backoff_level']=0;profile.loc[use,'support_backoff_name']='exact'
    profile.loc[use,'selected_group_id']=mapping[use];profile.loc[use,'support_n']=n[use];profile.loc[use,'support_source_zones']=nz[use]
    minzones=prepared.contracts.protocol['support_and_backoff_contract']['minimum_source_zones']
    gc=prepared.contracts.protocol['guardrail_contract']
    minspan=float(gc['dependence_robust_bounds']['primary_block_hours'])*float(gc['bootstrap_support_requirements']['minimum_time_span_in_primary_blocks'])
    for i in np.flatnonzero(use):
        gid=mapping[i];cold=bool(profile.loc[i,'cold_start_guardrail']);prefix='' if cold else 'guardrail_'
        cc=exact['event_count' if cold else 'guardrail_count'][:,gid];present=cc>0;ok=int(present.sum())>=minzones
        if ok:
            span=(exact[prefix+'issue_max_us'][present,gid].astype(float)-exact[prefix+'issue_min_us'][present,gid].astype(float))/3_600_000_000
            ok=float(span.min())>=minspan
        profile.loc[i,'bootstrap_supported']=ok
    return profile,n,nz

def fit_unit(z,s,out):
    folder=CHRONO/'clara'/z/f'seed{s}';done=read(folder/'COMPLETE.json')
    bundle=cf.load_bundle(folder);assert z not in bundle.stats.source_zones and bundle.stats.heldout_zone==z
    assert bundle.audit['all_source_feedback_precedes_test']
    contracts,config=cf.contracts_and_config()
    prepared=core.prepare_clara_fit(bundle.stats,contracts=contracts,v4_config=config,require_formal_identity=False)
    variants={name:core.prepare_clara_fit(bundle.stats,contracts=variant_contract(contracts,removed),v4_config=config,require_formal_identity=False)
              for name,removed in {'NO_G':['rolling_state'],'NO_RWG':['ramp_state','raw_width_state','rolling_state'],'GLOBAL':list(cf.STATE_FIELDS)}.items()}
    accepted=pd.read_parquet(folder/'clara_action_evidence.parquet')
    accepted_d=pd.read_parquet(folder/'clara_decisions.parquet')
    selections=np.load(folder/'clara_selected_actions.npy');results=[];records=[];audits=[]
    for pi,price in enumerate(cf.PRICES):
        fit=core.fit_price_conditioned_clara(prepared,price=price,v4_config=config)
        d=fit.decisions.sort_values('full_state_code').reset_index(drop=True);e=matrices(fit.action_evidence)
        ae=matrices(accepted[accepted.price_id.eq(price.price_id)])
        for k in EV_NAMES:np.testing.assert_array_equal(e[k],ae[k])
        ad=accepted_d[accepted_d.price_id.eq(price.price_id)].sort_values('full_state_code')
        for k in ['selected_n_min','selected_nu','support_backoff_level']:np.testing.assert_array_equal(d[k],ad[k])
        thresholds=fit.audit['adaptive_guardrail_thresholds'];full,guard=cf.directional_repair(d,e,thresholds)
        np.testing.assert_array_equal(full,selections[pi])
        chosen={'CLARA':full,'NO_SCREEN':cf.choose_by_score(e,np.ones((6600,6),bool))}
        parameters=[]
        for name,prep in variants.items():
            vf=core.fit_price_conditioned_clara(prep,price=price,v4_config=config);vd=vf.decisions.sort_values('full_state_code').reset_index(drop=True);ve=matrices(vf.action_evidence)
            if name=='GLOBAL':
                np.testing.assert_allclose(ve['risk_score'],np.repeat(ve['risk_score'][:1],6600,axis=0),rtol=0,atol=1e-11)
                ge=dict(e);ge['risk_score']=ve['risk_score'];chosen['GLOBAL_SCORE_SAME_SCREEN']=cf.choose_by_score(ge,guard['eligible'])
            else:
                chosen[name],_=cf.directional_repair(vd,ve,thresholds)
                fields=[f for f in cf.STATE_FIELDS if f not in (['rolling_state'] if name=='NO_G' else ['ramp_state','raw_width_state','rolling_state'])]
                check=vd[fields].copy();check['cold']=vd.rolling_state.eq('cold_start');check['choice']=chosen[name]
                assert check.groupby(fields+['cold']).choice.nunique().max()==1
            vp=vd[['full_state_code','selected_n_min','selected_nu','support_backoff_level']].copy();vp['variant']=name;vp['price_id']=price.price_id;parameters.append(vp)
        levels=core.landscape_accelerator.build_level_aggregates(bundle.stats.arrays_for_price(price),list(prepared.mappings),list(prepared.group_frames))
        profile,n,nz=exact_profile(prepared,d,levels);nu=d.selected_nu.to_numpy();evidence_by_nu={}
        for strength in sorted(set(nu.tolist())):
            cfg=core.adaptive_v4.parameterized_contracts(contracts,n_min=int(config['fixed_support']['n_min']),nu=int(strength))
            evidence_by_nu[strength]=core.landscape_accelerator.compute_cache_evidence(profile=profile,contracts=cfg,mappings=list(prepared.mappings),levels=levels)
        oldactions=core.adaptive_v4.ACTIONS
        try:
            core.adaptive_v4.ACTIONS=core.SIX_ACTIONS
            exact_evidence=core.adaptive_v4.stitch_evidence(evidence_by_nu,nu)
        finally:core.adaptive_v4.ACTIONS=oldactions
        ee=matrices(exact_evidence);positive=n>0;exact=np.zeros(6600,int)
        exact[positive],_=cf.directional_repair(d.loc[positive].reset_index(drop=True),{k:v[positive] for k,v in ee.items()},thresholds)
        unchanged=d.support_backoff_level.eq(0).to_numpy()
        for k in EV_NAMES:np.testing.assert_array_equal(ee[k][unchanged],e[k][unchanged])
        np.testing.assert_array_equal(exact[unchanged],full[unchanged]);assert (exact[~positive]==0).all()
        chosen['NO_BACKOFF_EXACT']=exact
        rec=d[['full_state_code',*cf.STATE_FIELDS,'selected_n_min','selected_nu','support_backoff_level']].copy()
        rec['price_id']=price.price_id;rec['exact_count']=n;rec['exact_zones']=nz
        for name,value in chosen.items():rec[name]=value
        records.append(rec)
        results.append(dict(choices=chosen,e=e,guard=guard,exact_count=n,backoff=d.support_backoff_level.to_numpy(),states=d,price=price))
        audits.append(dict(price_id=price.price_id,thresholds=thresholds,full_evidence_and_decisions_exact=True,source_available_max=bundle.audit['source_available_max_ns'],source_cutoff_ns=bundle.audit['source_cutoff_ns'],empty_exact_states=int((n==0).sum()),no_backoff_static_only_when_empty=True))
        pd.concat(parameters).to_parquet(out/f'variant_parameters_{price.price_id}.parquet',index=False)
    pd.concat(records).to_parquet(out/'state_policies.parquet',index=False)
    return bundle,done,results,audits

def rolling(covered,c):
    n,m=covered.shape;out=np.full((n,m,5),np.nan)
    cs=np.vstack([np.zeros((1,m)),np.cumsum(covered,axis=0)])
    wc=(cs[168:]-cs[:-168])/168;gap=wc-c;tol=1.96*np.sqrt(c*(1-c)/168)
    out[167:,:,0]=wc<c-tol;out[167:,:,1]=wc>c+tol;out[167:,:,2]=abs(gap)
    out[167:,:,3]=np.maximum(-gap,0);out[167:,:,4]=np.maximum(gap,0);return out

def summarize_values(v,roll,mask):
    n=int(mask.sum());nr=int((np.arange(len(mask))[mask]>=167).sum())
    # v shape time x methods x [cost,capacity,miss,coverage,width,IS]
    a=np.concatenate([v[mask,:,:5].mean(0),np.nansum(roll[mask],axis=0)/nr if nr else np.full((v.shape[1],5),np.nan),v[mask,:,5].mean(0)[:,None]],axis=1)
    return n,nr,a

def worker(task):
    z,s=task;out=HERE/'units'/f'{z}__seed{s}';out.mkdir(parents=True,exist_ok=True);began=time.perf_counter()
    identity=dict(code=fc.sha256(__file__),chrono_data=fc.sha256(CHRONO/'DATA_READY.json'),clara_policy=fc.sha256(CHRONO/'clara'/z/f'seed{s}'/'COMPLETE.json'))
    marker=out/'COMPLETE.json'
    if marker.exists():
        r=read(marker);assert r['identity']==identity
        for name,hsh in r['outputs'].items():assert fc.sha256(out/name)==hsh
        return r
    bundle,main_meta,models,fit_audit=fit_unit(z,s,out)
    metrics=[];screening=[];subsets=[];traces=[];oracle_groups=[];main_max=0.
    with np.load(CHRONO/'clara'/z/f'seed{s}'/'decisions.npz') as f:accepted_choices={k:f[k] for k in f.files}
    for p in fc.PREDICTORS:
        for h in fc.HORIZONS:
            d=fc.load_stream(z,s,p,h,'target',main_meta['tsc_index']);codes=cf.state_codes(d,p,h,bundle.width_edges);n=len(d['y'])
            cap,miss,covered=cf.stream_components(d);width=d['upper']-d['lower'];reference=models[2]['choices']['CLARA'][codes]
            for pi,model in enumerate(models):
                price=model['price'];pid=price.price_id;ratio=float(price.miss_weight/price.capacity_weight)
                costs=price.capacity_weight*cap+price.miss_weight*miss
                flat=codes.reshape(-1);counts=np.bincount(flat,minlength=6600);present=counts>0
                costs_group=np.stack([np.bincount(flat,weights=costs[a].reshape(-1),minlength=6600) for a in range(6)],axis=1)
                so=costs_group.argmin(1);fo=np.where(model['guard']['eligible'],costs_group,np.inf).argmin(1);full=model['choices']['CLARA']
                assert np.all(costs_group[present,so[present]]<=costs_group[present,fo[present]]+1e-9)
                assert np.all(costs_group[present,fo[present]]<=costs_group[present,full[present]]+1e-9)
                choices={k:a[codes] for k,a in model['choices'].items()}
                ev={k:v[codes].reshape(-1,6) for k,v in model['e'].items()};ev['risk_score']=np.moveaxis(width,0,-1).reshape(-1,6)
                choices['WIDTH_SAME_SCREEN']=cf.choose_by_score(ev,model['guard']['eligible'][codes].reshape(-1,6)).reshape(11,n)
                choices.update(FIX_REFERENCE_PRICE=reference,WIDTH_ONLY=width.argmin(0),STATE_ORACLE=so[codes],FEASIBLE_STATE_ORACLE=fo[codes],OUTCOME_ORACLE=costs.argmin(0))
                np.testing.assert_array_equal(choices['CLARA'],accepted_choices[f'{p}__H{h:02d}'][pi])
                for name in ['CLARA','GLOBAL_SCORE_SAME_SCREEN','WIDTH_SAME_SCREEN','FEASIBLE_STATE_ORACLE']:
                    assert np.take_along_axis(model['guard']['eligible'][codes],choices[name][...,None],axis=2).all()
                for ci,c in enumerate(fc.COVERAGES):
                    base=dict(zone_or_farm=z,seed=s,price_id=pid,predictor=p,horizon_steps=h,target_coverage=float(c))
                    ch=np.stack([choices[m][ci] for m in METHODS],axis=1);times=np.arange(n)[:,None]
                    v=np.stack([costs[ch,ci,times],price.capacity_weight*cap[ch,ci,times],price.miss_weight*miss[ch,ci,times],covered[ch,ci,times],width[ch,ci,times],width[ch,ci,times]+2/(1-c)*miss[ch,ci,times]],axis=2)
                    roll=rolling(v[:,:,3],float(c));code=codes[ci];positive=model['exact_count'][code]>0
                    for reg in ['overall','ordinary','ramp']:
                        mask=np.ones(n,bool) if reg=='overall' else d['ramp']==(reg=='ramp')
                        if not mask.any():continue
                        nn,nr,means=summarize_values(v,roll,mask)
                        for mi,m in enumerate(METHODS):metrics.append(dict(**base,regime=reg,method=m,event_count=nn,reliability_event_count=nr,**dict(zip(M,means[mi]))))
                        no_g=choices['NO_G'][ci];nog_excluded=~model['guard']['eligible'][code,no_g]
                        screening.append(dict(**base,regime=reg,event_count=nn,candidate_options=6*nn,
                            excluded_candidate_count=int((~model['guard']['admissible'][code[mask]]).sum()),empty_screen_count=int(model['guard']['guardrail_empty_fallback'][code[mask]].sum()),
                            exact_empty_count=int((~positive[mask]).sum()),backoff_count=int((model['backoff'][code[mask]]>0).sum()),
                            no_g_selects_full_ineligible_count=int(nog_excluded[mask].sum()),
                            no_g_full_ineligible_covered_sum=float(v[mask&nog_excluded,METHODS.index('NO_G'),3].sum()),
                            clara_on_no_g_full_ineligible_covered_sum=float(v[mask&nog_excluded,0,3].sum())))
                        for m in METHODS[1:9]:
                            mi=METHODS.index(m);delta=v[mask,mi,:5]-v[mask,0,:5]
                            traces.append(dict(**base,regime=reg,method=m,event_count=nn,changed_count=int((ch[mask,mi]!=ch[mask,0]).sum()),
                                **dict(zip(['delta_ERRF_sum','delta_capacity_sum','delta_exceedance_sum','delta_covered_sum','delta_width_sum'],delta.sum(0)))))
                    for subset,mask in [('nonempty',positive),('empty',~positive),('sparse_nonempty',positive&(model['backoff'][code]>0)),('supported',model['backoff'][code]==0)]:
                        if not mask.any():continue
                        nn,nr,means=summarize_values(v,roll,mask)
                        for m in ['CLARA','NO_BACKOFF_EXACT']:
                            subsets.append(dict(**base,subset=subset,method=m,event_count=nn,reliability_event_count=nr,**dict(zip(M,means[METHODS.index(m)]))))
                    # Independent scalar metric check on accepted CLARA (same target and window definition).
                    action=choices['CLARA'][ci];at=np.arange(n)
                    expected=fc.evaluation_metrics(d['lower'][action,ci,at],d['upper'][action,ci,at],d['y'],d['center'],float(c),ratio)
                    actual=np.r_[v[:,0,:5].mean(0),np.nanmean(roll[:,0],axis=0),v[:,0,5].mean()]
                    ex=np.array([expected[k] for k in ['mean_errf','mean_capacity_cost','mean_exceedance_cost','coverage']]+[float(width[action,ci,at].mean())]+[expected[k] for k in ['tuwr','towr','ard','under_gap','over_gap','interval_score']])
                    np.testing.assert_allclose(actual,ex,rtol=0,atol=1e-10);main_max=max(main_max,float(abs(actual-ex).max()))
                state=model['states'].loc[present,['full_state_code',*cf.STATE_FIELDS]].copy()
                state['zone_or_farm']=z;state['seed']=s;state['price_id']=pid;state['horizon_steps']=h;state['event_count']=counts[present]
                state['CLARA']=full[present];state['STATE_ORACLE']=so[present];state['FEASIBLE_STATE_ORACLE']=fo[present]
                state['eligible_count']=model['guard']['eligible'][present].sum(1)
                for ai,a in enumerate(fc.ACTIONS):state[a+'__mean_ERRF']=costs_group[present,ai]/counts[present]
                oracle_groups.append(state)
    frames={'seed_condition_metrics.parquet':pd.DataFrame(metrics),'screening_counts.parquet':pd.DataFrame(screening),'subset_metrics.parquet':pd.DataFrame(subsets),'paired_traces.parquet':pd.DataFrame(traces),'oracle_state_groups.parquet':pd.concat(oracle_groups,ignore_index=True)}
    for name,frame in frames.items():frame.to_parquet(out/name,index=False)
    f=frames['seed_condition_metrics.parquet'];np.testing.assert_allclose(f.ERRF,f.capacity+f.exceedance,rtol=0,atol=1e-10)
    np.testing.assert_allclose(f.ARD,f.under_gap+f.over_gap,rtol=0,atol=1e-10,equal_nan=True)
    assert f[f.regime.eq('overall')].groupby(['price_id','method']).event_count.sum().eq(617672).all()
    record=dict(status='PASS',zone=z,seed=s,identity=identity,fit_audit=fit_audit,main_decisions_exact=True,main_metric_max_error=main_max,
                source_bundle_identity=bundle.audit['identity_sha256'],test_data_used_only_for_evaluation_and_labeled_oracles=True,
                methods=METHODS,seconds=time.perf_counter()-began,outputs={p.name:fc.sha256(p) for p in out.glob('*.parquet')})
    save(marker,record);return record

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    tasks=[('zone1',0)] if a.pilot else [(z,s) for z in fc.ZONES for s in fc.SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for i,f in enumerate(as_completed([pool.submit(worker,t) for t in tasks]),1):
            r=f.result();print(i,len(tasks),r['zone'],r['seed'],r['status'],round(r['seconds'],2),flush=True)
    if not a.pilot:save(HERE/'EVALUATION_COMPLETE.json',dict(status='PASS',units=30,code_sha256=fc.sha256(__file__),chrono_DATA_READY_sha256=fc.sha256(CHRONO/'DATA_READY.json'),methods=METHODS))

if __name__=='__main__':main()
