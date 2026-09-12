"""Frozen-policy mechanism replay over the complete external five-price grid."""
from pathlib import Path
import os
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[name] = '1'
import sys
import ast
import json
import time
import traceback
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
WORK = HERE.parents[1]
SOURCE = WORK/'analysis/all_method_price_evidence_20260905'
sys.path.insert(0,str(SOURCE))
import price_evidence_runtime as rt
from vectorized_external_evidence import evidence_for_profile
core = rt.core
core.ACTIONS = rt.ACTIONS
FIELDS = list(core.STATE_FIELDS)
ACTIONS = list(rt.ACTIONS)
TSC = 'TunedSingleConformal_Local'
T = ACTIONS.index(TSC)
PHASES = ['TSC','raw','shrink','score','CLARA']
COUNTS = ['event_count','reliability_event_count']
METRICS = ['mean_errf','mean_reserve','mean_miss','empirical_coverage','average_width','TUWR','TOWR','ARD']
SUM_NAMES = ['errf_sum','reserve_sum','miss_sum','covered_sum','width_sum','tuwr_sum','towr_sum','ard_sum']
ARRAY_NAMES = ['errf','reserve','miss','covered','width','tuwr','towr','ard']

def writej(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def log(stage,**kwargs):
    print(json.dumps(dict(stage=stage,**kwargs),ensure_ascii=False,default=str),flush=True)

def load_stream(farm,seed,predictor,h,split,pid):
    folder=SOURCE/'external_streams'/f'{farm}-S{seed}-{predictor}-H{h:02d}'
    man=rt.readj(folder/'manifest.json'); assert man['status']=='PASS'
    config=rt.readj(SOURCE/'tsc_adaptation/selected_configurations.json')[pid][farm]
    names=[f'base_{split}.parquet',f'tsc_{split}_{config["configuration_id"]}.parquet']
    for name in names:assert rt.sha(folder/name)==man['outputs'][name]
    b,t=(pd.read_parquet(folder/name) for name in names)
    assert b.event_id.equals(t.event_id)
    f=pd.concat([b,t.drop(columns='event_id')],axis=1)
    theta=rt.theta_for(pid)
    for a in ACTIONS:
        values=core._metric_arrays_from_endpoints(f,f[a+'__candidate_lower'].to_numpy(float),f[a+'__candidate_upper'].to_numpy(float),theta=theta)
        f[a+'__errf']=values['errf']
    return f

def fast_statistics(facts, fields):
    # Same means, missing-value masks and UTC-day standard errors as the sealed estimator.
    fields=list(fields)
    cols=[a+'__'+m for a in ACTIONS for m in ['errf','covered','tuwr_indicator','ard_value']]
    f=facts[fields+['event_id','issue_timestamp']+cols].copy()
    f['_utc_day']=core._utc_day(f.issue_timestamp)
    if not fields:f['_global_key']=0;fields=['_global_key']
    rename={}
    for a in ACTIONS:
        for old,new in [('errf','errf_mean'),('covered','covered_mean'),('tuwr_indicator','tuwr_mean'),('ard_value','ard_mean')]:
            rename[a+'__'+old]=a+'__'+new
        guard=a+'__covered_guardrail_mean'
        f[guard]=f[a+'__covered'].where(f[a+'__tuwr_indicator'].notna()&f[a+'__ard_value'].notna())
        cols.append(guard)
    gb=f.groupby(fields,sort=True,dropna=False)
    means=gb[cols].mean().rename(columns=rename)
    means['event_count']=gb.size()
    ec=[a+'__errf' for a in ACTIONS]
    daily=f.groupby(fields+['_utc_day'],sort=True,dropna=False)[ec].mean()
    ds=daily.groupby(level=list(range(len(fields))),sort=True,dropna=False)
    se=(ds.std()/np.sqrt(ds.count().clip(lower=1))).fillna(0)
    se.columns=[a+'__risk_day_cluster_se' for a in ACTIONS]
    return means.join(se).reset_index()

def unscreened(evidence,metric,tolerance):
    candidates=evidence[metric]<=np.min(evidence[metric],axis=1,keepdims=True)+tolerance
    coverage=evidence['coverage']
    best=np.max(np.where(candidates,coverage,-np.inf),axis=1)
    candidates &= coverage>=best[:,None]-tolerance
    for name in ['tuwr','ard']:
        v=evidence[name];finite=np.isfinite(v)
        count=candidates.sum(1);fc=(candidates&finite).sum(1)
        assert not ((fc>0)&(fc<count)).any()
        applies=(fc==count)&(count>0)
        best=np.min(np.where(candidates&finite,v,np.inf),axis=1)
        candidates &= (~applies[:,None]) | (finite&(v<=best[:,None]+tolerance))
    assert candidates.any(1).all()
    return np.argmax(candidates,axis=1)

def reconstruct_evidence(farm,seed,pid,dest):
    unit=SOURCE/'external_price_runs/units'/f'{farm}-S{seed}-{pid}'
    man=rt.readj(unit/'manifest.json');assert man['status']=='PASS'
    for name in ['clara_state_decisions.parquet','cell_metrics.parquet','action_counts.parquet','audits.json']:
        assert rt.sha(unit/name)==man['outputs'][name]
    decisions=pd.read_parquet(unit/'clara_state_decisions.parquet')
    states=decisions[FIELDS].copy()
    contracts=rt.load_frozen_contracts()
    facts=pd.concat([load_stream(farm,seed,p,h,'adaptation',pid) for p in rt.PREDICTORS for h in rt.HORIZONS],ignore_index=True)
    assert len(facts)==man['adaptation_events'] and not facts.event_id.duplicated().any()
    support=[tuple(l['fields']) for l in contracts.support_levels]
    tables=[fast_statistics(facts,f) for f in support]
    mapped=[core._map_level_statistics(states,t,fields=f) for f,t in zip(support,tables)]
    chosen=decisions.support_backoff_level.to_numpy(int)
    for n in decisions.selected_n_min.unique():
        mask=decisions.selected_n_min.eq(n).to_numpy()
        assert np.array_equal(core._selected_profile(mapped,n_min=int(n))[mask],chosen[mask])
    beta=float(contracts.protocol['landscape_contract']['beta'])
    evidence=evidence_for_profile(states=states,mapped_levels=mapped,selected_levels=chosen,nu=decisions.selected_nu.to_numpy(int),beta=beta)
    # Independently check scalar group means/SE on predetermined evenly spaced states.
    idx=np.unique(np.linspace(0,len(states)-1,12,dtype=int))
    for i in idx:
        level=chosen[i];fs=support[level];mask=np.ones(len(facts),dtype=bool)
        for field in fs:mask &= facts[field].eq(states.iloc[i][field]).to_numpy()
        sub=facts.loc[mask]
        for a in ACTIONS:
            daily=sub.groupby(core._utc_day(sub.issue_timestamp))[a+'__errf'].mean()
            se=daily.std()/np.sqrt(len(daily)) if len(daily)>1 else 0.0
            assert abs(mapped[level].iloc[i][a+'__errf_mean']-sub[a+'__errf'].mean())<1e-12
            assert abs(mapped[level].iloc[i][a+'__risk_day_cluster_se']-se)<1e-12
    slow=core._evidence_for_profile(states=states.iloc[idx].reset_index(drop=True),
        mapped_levels=[m.iloc[idx].reset_index(drop=True) for m in mapped],selected_levels=chosen[idx],
        nu=decisions.selected_nu.to_numpy(int)[idx],beta=beta)
    for key in evidence:np.testing.assert_allclose(evidence[key][idx],slow[key],rtol=0,atol=1e-12,equal_nan=True)
    thresholds=rt.readj(unit/'audits.json')['thresholds']
    tol=float(contracts.protocol['selection_contract']['score_tie_tolerance'])
    selected,empty,safe=core._select_local_actions(states=states,evidence=evidence,thresholds=thresholds,tolerance=tol)
    assert np.array_equal(np.asarray(ACTIONS)[selected],decisions.selected_action.to_numpy())
    assert np.array_equal(empty,decisions.guardrail_empty_fallback.to_numpy())
    trace=decisions.copy();trace['state_id']=np.arange(len(trace))
    trace['TSC_safe']=safe[:,T]
    trace['TSC_fit_coverage_gap']=evidence['coverage'][:,T]-states.target_coverage.to_numpy()
    trace['TSC_fit_TUWR']=evidence['tuwr'][:,T];trace['TSC_fit_ARD']=evidence['ard'][:,T]
    trace['TSC_best_score']=evidence['risk_score'][:,T]<=evidence['risk_score'].min(1)+tol
    trace['fit_cost_delta']=evidence['risk_mean'][np.arange(len(trace)),selected]-evidence['risk_mean'][:,T]
    trace['TSC_fit_cost']=evidence['risk_mean'][:,T]
    trace['reason']=np.select([selected==T,empty,~safe[:,T]],
        ['retain_TSC','empty_screen_switch','TSC_fails_screen_switch'],default='TSC_passes_screen_switch')
    trace['historical_TSC_coverage']=np.select([
        trace.TSC_fit_coverage_gap < -float(thresholds['coverage_shortfall_epsilon']),
        trace.TSC_fit_coverage_gap > float(thresholds['coverage_shortfall_epsilon'])],
        ['below_target_band','above_target_band'],default='within_target_band')
    for phase,metric in [('raw','risk_mean'),('shrink','risk_shrunken'),('score','risk_score')]:trace[phase]=unscreened(evidence,metric,tol)
    trace['CLARA']=selected;trace['TSC']=T
    trace.to_parquet(dest/'state_trace.parquet',index=False)
    pd.concat([states.assign(action=a,**{k:v[:,j] for k,v in evidence.items()}) for j,a in enumerate(ACTIONS)],ignore_index=True).to_parquet(dest/'historical_action_evidence.parquet',index=False)
    return trace,thresholds

def metrics_arrays(facts,pid):
    theta=rt.theta_for(pid)
    arrays=[core._metric_arrays_from_endpoints(facts,facts[a+'__candidate_lower'].to_numpy(float),facts[a+'__candidate_upper'].to_numpy(float),theta=theta) for a in ACTIONS]
    return {k:np.column_stack([a[k] for a in arrays]) for k in ['errf','reserve','miss','covered','width']}

def rolling(facts,covered):
    shape=covered.shape
    result={k:np.full(shape,np.nan) for k in ['tuwr','towr','ard']}
    for c,group in facts.groupby('target_coverage',sort=True):
        ordered=group.sort_values(['issue_timestamp','event_id'],kind='mergesort').index.to_numpy()
        n=int(np.ceil(168*60/float(group.nominal_cadence_minutes.iloc[0])))
        assert n==672
        cs=np.vstack([np.zeros((1,shape[1])),np.cumsum(covered[ordered],axis=0)])
        cov=(cs[n:]-cs[:-n])/n;gap=cov-float(c)
        tolerance=1.96*np.sqrt(c*(1-c)/n)
        at=ordered[n-1:]
        result['tuwr'][at]=(gap < -tolerance)
        result['towr'][at]=(gap > tolerance)
        result['ard'][at]=abs(gap)
    return result

def summarize(facts,arrays,selections,trace,indices,farm,seed,pid,predictor,h,expected):
    cells=[];states=[];blocks=[];reason_blocks=[];decomposition=[]
    n=len(facts);nph=len(PHASES)
    selected={k:np.column_stack([v[np.arange(n),selections[ph]] for ph in PHASES]) for k,v in arrays.items()}
    selected.update(rolling(facts,selected['covered']))
    base=facts[['target_coverage','ramp_state','issue_timestamp']].copy()
    base['state_id']=indices
    base['reason']=trace.reason.to_numpy()[indices]
    base['historical_TSC_coverage']=trace.historical_TSC_coverage.to_numpy()[indices]
    base['fit_cost_delta']=trace.fit_cost_delta.to_numpy()[indices]
    base['TSC_fit_cost']=trace.TSC_fit_cost.to_numpy()[indices]
    labels=dict(zone_or_farm=farm,seed=seed,price_id=pid,predictor=predictor,horizon_steps=h)
    # Exact per-cell and action-count regression precedes retaining any diagnostic.
    for j,phase in enumerate(PHASES):
        frame=base.copy();frame['event_count']=1
        frame['reliability_event_count']=np.isfinite(selected['tuwr'][:,j]).astype(int)
        for name,a in zip(SUM_NAMES,ARRAY_NAMES):frame[name]=np.nan_to_num(selected[a][:,j],nan=0).astype(float)
        sums=COUNTS+SUM_NAMES
        for regime in ['overall','ordinary','ramp']:
            part=frame if regime=='overall' else frame[frame.ramp_state.eq(regime)]
            grouped=part.groupby('target_coverage')[sums].sum()
            for c,row in grouped.iterrows():
                record={**labels,'target_coverage':c,'phase':phase,'regime':regime,**row.to_dict()}
                for name,num in zip(METRICS,SUM_NAMES):record[name]=row[num]/row['reliability_event_count' if name in ['TUWR','TOWR','ARD'] else 'event_count']
                cells.append(record)
                if phase in ['CLARA','TSC']:
                    ref=expected.loc[(predictor,h,c,'CLARA_6A_Local' if phase=='CLARA' else TSC,regime)]
                    for name in COUNTS+METRICS:
                        assert abs(record[name]-ref[name])<1e-12,(farm,seed,pid,predictor,h,c,phase,regime,name,record[name],ref[name])
        ss=frame.groupby(['state_id','target_coverage'])[sums+['fit_cost_delta','TSC_fit_cost']].sum().reset_index()
        for k,v in labels.items():ss[k]=v
        ss['phase']=phase;states.append(ss)
        issue=pd.to_datetime(frame.issue_timestamp)
        if issue.dt.tz is None:issue=issue.dt.tz_localize('Asia/Shanghai')
        issue=issue.dt.tz_convert('UTC')
        for hours in [24,168]:
            frame['block_start_utc']=issue.dt.floor(f'{hours}h')
            bb=frame.groupby(['target_coverage','block_start_utc'])[sums].sum().reset_index()
            for k,v in labels.items():bb[k]=v
            bb['phase']=phase;bb['block_hours']=hours;blocks.append(bb)
            if phase in ['TSC','CLARA']:
                rb=frame.groupby(['target_coverage','block_start_utc','reason'])[sums].sum().reset_index()
                for k,v in labels.items():rb[k]=v
                rb['phase']=phase;rb['block_hours']=hours;reason_blocks.append(rb)
    # Within-stream random-proportion accounting; all candidates retain the same histories.
    for c,group in facts.groupby('target_coverage'):
        at=group.index.to_numpy();chosen=selections['CLARA'][at]
        shares=np.bincount(chosen,minlength=len(ACTIONS))/len(at)
        row={**labels,'target_coverage':c,'event_count':len(at),'TSC_share':shares[T]}
        for name in ['errf','reserve','miss','covered','width']:
            v=arrays[name][at];cl=v[np.arange(len(at)),chosen].mean();ts=v[:,T].mean();mix=shares@v.mean(axis=0)
            row.update({name+'_CLARA':cl,name+'_TSC':ts,name+'_mixture':mix,name+'_composition':mix-ts,name+'_alignment':cl-mix})
        delta=arrays['errf'][at,chosen]-arrays['errf'][at,T]
        assert np.all(abs(delta[chosen==T])<1e-12)
        row['beneficial_switch_count']=int((delta < -1e-12).sum());row['costly_switch_count']=int((delta>1e-12).sum())
        row['switch_saving_sum']=float(-delta[delta<0].sum());row['switch_cost_sum']=float(delta[delta>0].sum())
        assert abs((row['switch_cost_sum']-row['switch_saving_sum'])/len(at)-(row['errf_CLARA']-row['errf_TSC']))<1e-12
        decomposition.append(row)
    return pd.DataFrame(cells),pd.concat(states,ignore_index=True),pd.concat(blocks,ignore_index=True),pd.concat(reason_blocks,ignore_index=True),pd.DataFrame(decomposition)

def run_unit(task):
    farm,seed,pid=task;uid=f'{farm}-S{seed}-{pid}';dest=HERE/'units'/uid;dest.mkdir(parents=True,exist_ok=True)
    started=time.monotonic();script_hash=rt.sha(__file__)
    done=dest/'manifest.json'
    if done.exists():
        man=rt.readj(done)
        assert man['script_sha256']==script_hash and man['status']=='PASS'
        assert all(rt.sha(dest/k)==v for k,v in man['outputs'].items())
        return dict(unit=uid,status='REUSED')
    try:
        log('historical_evidence',unit=uid)
        trace,thresholds=reconstruct_evidence(farm,seed,pid,dest)
        log('frozen_actions_reproduced',unit=uid,seconds=round(time.monotonic()-started,1))
        source=SOURCE/'external_price_runs/units'/uid
        expected=pd.read_parquet(source/'cell_metrics.parquet').set_index(['predictor','horizon_steps','target_coverage','method','regime'])
        official_actions=pd.read_parquet(source/'action_counts.parquet')
        official_actions=official_actions[official_actions.method.eq('CLARA_6A_Local')]
        index=pd.MultiIndex.from_frame(trace[FIELDS]);assert index.is_unique
        outputs=[dest/'state_trace.parquet',dest/'historical_action_evidence.parquet']
        total=0
        for predictor in rt.PREDICTORS:
            for h in rt.HORIZONS:
                facts=load_stream(farm,seed,predictor,h,'final',pid).reset_index(drop=True)
                ix=index.get_indexer(pd.MultiIndex.from_frame(facts[FIELDS]));assert (ix>=0).all()
                selections={ph:trace[ph].to_numpy(int)[ix] for ph in PHASES}
                selected=np.asarray(ACTIONS)[selections['CLARA']]
                counts=facts[['target_coverage','ramp_state']].copy();counts['selected_action']=selected
                for regime in ['overall','ordinary','ramp']:
                    sub=counts if regime=='overall' else counts[counts.ramp_state.eq(regime)]
                    actual=sub.groupby(['target_coverage','selected_action']).size().sort_index()
                    ref=official_actions[official_actions.predictor.eq(predictor)&official_actions.horizon_steps.eq(h)&official_actions.regime.eq(regime)].set_index(['target_coverage','selected_action']).event_count.sort_index()
                    assert actual.index.equals(ref.index) and np.array_equal(actual,ref)
                arrays=metrics_arrays(facts,pid)
                result=summarize(facts,arrays,selections,trace,ix,farm,seed,pid,predictor,h,expected)
                for suffix,frame in zip(['cells','states','blocks','reason_blocks','decomposition'],result):
                    path=dest/f'{predictor}-H{h:02d}_{suffix}.parquet';frame.to_parquet(path,index=False);outputs.append(path)
                total+=len(facts)
        assert total==rt.readj(source/'manifest.json')['final_events']
        man=dict(status='PASS',unit=uid,script_sha256=script_hash,events=total,states=len(trace),seconds=time.monotonic()-started,
            action_counts_exact=True,official_metrics_reproduced=True,thresholds=thresholds,
            source_manifest_sha256=rt.sha(source/'manifest.json'),outputs={p.name:rt.sha(p) for p in outputs})
        writej(done,man)
        return {k:man[k] for k in ['unit','status','events','seconds']}
    except Exception:
        writej(dest/'exception.json',dict(status='FAILED',traceback=traceback.format_exc()));raise

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=2)
    args=ap.parse_args()
    tasks=[(f,s,p) for f in rt.FARMS for s in range(3) for p,r in rt.PRICES]
    if args.pilot:tasks=[('FarmB',0,rt.MAIN_ID)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(run_unit,t):t for t in tasks}
        for future in as_completed(futures):log('unit_complete',**future.result())
    if not args.pilot:
        writej(HERE/'replay_manifest.json',dict(status='PASS',units=len(tasks),script_sha256=rt.sha(__file__),plan_sha256=rt.sha(HERE/'PLAN.md')))

if __name__=='__main__':main()
