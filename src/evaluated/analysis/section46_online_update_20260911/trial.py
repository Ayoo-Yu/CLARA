"""Causal cumulative CLARA replay, isolated from all accepted outputs."""
from pathlib import Path
import os, sys, json, time, argparse, traceback, hashlib
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
    os.environ[k] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT/'analysis/section46_failure_diagnosis_20260911'))
import diagnose as prior
rt = prior.rt
from engine import add_rows, evidence, choose_many, run_batches
from source_tuning_facts import _CompactRollingSummaries
F = prior.F
A = prior.A
M = prior.M
EK = ['risk_mean','risk_parent','risk_shrunken','risk_se','risk_score','coverage','tuwr','ard']
METHODS = ['CLARA','CLARA_online','Cost_update_only','Reliability_update_only']
S = prior.S


def save(p, x): prior.save(p, x)
def read(p): return prior.read(p)
def log(**kw): prior.log(**kw)
def ns(s): return s.to_numpy(dtype='datetime64[ns]').astype(np.int64)


def reliability(f):
    """Same per-feedback historical summaries as original code; no new windows."""
    n = len(f)
    out = np.full((n, 6, 2), np.nan)
    max_error = 0.0
    for c, group in f.groupby('target_coverage', sort=True):
        pos = group.sort_values(['issue_timestamp','event_id'], kind='mergesort').index.to_numpy()
        cov = f.loc[pos, [a+'__covered' for a in A]].to_numpy(float)
        warm = f.loc[pos,'rolling_state'].ne('cold_start').to_numpy()
        label = ns(f.loc[pos,'label_timestamp'])
        avail = ns(f.loc[pos,'label_available_timestamp'])
        assert (np.diff(label)>0).all() and (np.diff(avail)>0).all()
        assert (avail > label).all()
        prefix = np.vstack([np.zeros((1,6)), np.cumsum(cov, axis=0)])
        gap = (prefix[96:] - prefix[:-96])/96 - c
        tol = 1.96*np.sqrt(c*(1-c)/96)
        vals = np.stack([gap < -tol, np.abs(gap)], axis=2)
        vp = np.concatenate([np.zeros((1,6,2)), np.cumsum(vals, axis=0)])
        end = np.arange(1,len(pos)+1)
        good = np.flatnonzero(warm)
        assert (end[good]>=672).all()
        start = end[good] - 672
        cutoff = avail[good] - int(168*3600e9)
        assert (label[start]>=cutoff).all()
        assert (label[start]<=cutoff+int(900e9)).all()
        out[pos[good]] = (vp[end[good]-96+1] - vp[start])/577
        # Original implementation independently verifies early, middle and late rows.
        for a in range(6):
            summary = _CompactRollingSummaries(label_ns=label, covered=cov[:,a], target_coverage=float(c), cadence_minutes=15.)
            for row in good[np.linspace(0,len(good)-1,3,dtype=int)]:
                old = summary.summarize(end_exclusive=int(row+1), query_ns=int(avail[row]))
                np.testing.assert_allclose(out[pos[row],a],old[3:5],atol=2e-12,rtol=0)
    for j,a in enumerate(A):
        for k,suffix in enumerate(['tuwr_indicator','ard_value']):
            old = f[a+'__'+suffix].to_numpy(float)
            valid = np.isfinite(old)
            if valid.any():
                assert np.array_equal(valid,np.isfinite(out[:,j,k]))
                err = np.max(np.abs(old[valid]-out[valid,j,k]))
                max_error=max(max_error,float(err));assert err<2e-12
    assert np.isfinite(out).all(axis=(1,2)).tolist() == f.rolling_state.ne('cold_start').tolist()
    return out, max_error


def load_data(farm, seed, pid, d):
    parts = {'adaptation':[], 'final':[]}
    values = {'adaptation':[], 'final':[]}
    metric_parts=[]
    source=[]
    index = pd.MultiIndex.from_frame(d[F])
    reliability_error=0.
    for p in rt.rt.PREDICTORS:
        for h in rt.rt.HORIZONS:
            for split in ['adaptation','final']:
                f = rt.fr.load_stream(farm,seed,p,h,split,pid)
                state_ids = index.get_indexer(pd.MultiIndex.from_frame(f[F]));assert (state_ids>=0).all()
                rel,err = reliability(f);reliability_error=max(err,reliability_error)
                facts=np.stack([f[[a+'__errf' for a in A]].to_numpy(float),f[[a+'__covered' for a in A]].to_numpy(float),rel[:,:,0],rel[:,:,1]],axis=2)
                cols=['event_id','issue_timestamp','label_timestamp','label_available_timestamp','predictor','horizon_steps','target_coverage']
                meta=f[cols].copy();meta['state_id']=state_ids
                parts[split].append(meta);values[split].append(facts)
                if split=='final':
                    arrays=[rt.rt.core._metric_arrays_from_endpoints(f,f[a+'__candidate_lower'].to_numpy(float),f[a+'__candidate_upper'].to_numpy(float),theta=rt.rt.theta_for(pid)) for a in A]
                    metric_parts.append(np.stack([np.column_stack([v[k] for v in arrays]) for k in ['errf','reserve','miss','covered','width']],axis=2))
                source.append(dict(predictor=p,horizon_steps=h,split=split,n=len(f),min_issue=str(f.issue_timestamp.min()),max_issue=str(f.issue_timestamp.max()),max_available=str(f.label_available_timestamp.max())))
    out={}
    for split in ['adaptation','final']:
        f=pd.concat(parts[split],ignore_index=True)
        assert not f.event_id.duplicated().any()
        order=f.sort_values(['issue_timestamp','event_id'],kind='mergesort').index.to_numpy()
        out[split]=(f.iloc[order].reset_index(drop=True),np.concatenate(values[split])[order])
        if split=='final':out['metrics']=np.concatenate(metric_parts)[order]
    out['source']=source;out['reliability_error']=reliability_error
    return out


def layout(d, nday):
    maps=[];offset=0
    for level in rt.rt.load_frozen_contracts().support_levels:
        fields=list(level['fields'])
        if fields:
            group=d[fields].drop_duplicates().reset_index(drop=True)
            ix=pd.MultiIndex.from_frame(group).get_indexer(pd.MultiIndex.from_frame(d[fields]))
        else:ix=np.zeros(len(d),np.int64)
        maps.append(ix+offset);offset+=int(ix.max())+1
    mapping=np.column_stack(maps).astype(np.int64)
    stats=(np.zeros(offset,np.int64),np.zeros(offset,np.int64),np.zeros((offset,6,5)),
           np.zeros((offset,nday),np.int64),np.zeros((offset,nday,6)),np.zeros(offset,np.int64),np.zeros((offset,6)),np.zeros((offset,6)))
    return mapping,stats


def batch_verify(d, mapping, nmin, nu, cold, stats, meta, facts, day_base):
    # Independent pandas group means/day SE and existing vectorized estimator.
    frame=d.iloc[meta.state_id.to_numpy()][F].reset_index(drop=True).copy()
    frame['event_id']=meta.event_id.to_numpy();frame['issue_timestamp']=meta.issue_timestamp.to_numpy()
    for j,a in enumerate(A):
        for k,suffix in enumerate(['errf','covered','tuwr_indicator','ard_value']):frame[a+'__'+suffix]=facts[:,j,k]
    levels=[]
    for level in rt.rt.load_frozen_contracts().support_levels:
        fields=level['fields']
        levels.append(rt.rt.core._map_level_statistics(d,rt.fr.fast_statistics(frame,fields),fields=fields))
    counts=np.stack([l.event_count.to_numpy() for l in levels],axis=1)
    selected=(counts>=nmin[:,None]).argmax(1)
    expected=rt.fr.evidence_for_profile(states=d,mapped_levels=levels,selected_levels=selected,nu=nu,beta=1.)
    actual,actual_levels=evidence(np.arange(len(d)),mapping,nmin,nu,cold,stats[0],stats[1],stats[2],stats[5],stats[3],stats[4])
    np.testing.assert_array_equal(selected,actual_levels)
    largest=0.
    for j,key in enumerate(EK):
        np.testing.assert_allclose(actual[:,:,j],expected[key],rtol=0,atol=3e-10,equal_nan=True)
        largest=max(largest,float(np.nanmax(abs(actual[:,:,j]-expected[key]))))
    return largest


def summarize_unit(dest, farm, seed, pid, d, frozen, f, metrics, selected, pred, trace):
    records=[];blocks=[];diagnostics=[]
    uid=dest.name
    old=read(ROOT/'analysis/section46_failure_diagnosis_20260911/units'/uid/'base_manifest.json')
    thresholds=old['thresholds'];eps=thresholds['coverage_shortfall_epsilon'];tau=thresholds['tuwr_upper']
    for (p,h,c),g in f.groupby(['predictor','horizon_steps','target_coverage'],sort=True):
        pos=g.index.to_numpy();n=len(pos);nr=n-671
        vals=np.take_along_axis(metrics[pos],selected[pos,:,None],axis=1)
        vals=np.concatenate([vals,prior.rolling(vals[:,:,3],float(c))],axis=2)
        clean=np.nan_to_num(vals,nan=0)
        key=dict(zone_or_farm=farm,seed=seed,price_id=pid,predictor=p,horizon_steps=h,target_coverage=float(c))
        den=np.array([n]*5+[nr]*5)
        for j,method in enumerate(METHODS):records.append(key|dict(method=method,event_count=n,reliability_event_count=nr,**dict(zip(M,clean[:,j].sum(0)/den))))
        # Blocks anchored to UTC epoch, same existing physical-block aggregation.
        # Exact dates saved so blocks can be paired across prices/conditions/seeds.
        utc_ns=ns(g.issue_timestamp)-int(8*3600e9)
        block=utc_ns//int(168*3600e9)
        ids=f.loc[pos,'state_id'].to_numpy(int)
        oldpred=frozen[ids]
        oldsel=selected[pos,0];newsel=selected[pos,1]
        rr=np.arange(n)
        changed=newsel!=oldsel
        actual=metrics[pos,newsel,0]-metrics[pos,oldsel,0]
        ppnew=pred[pos,newsel,0]-pred[pos,oldsel,0]
        ppold=oldpred[rr,newsel,2]-oldpred[rr,oldsel,2]
        prediction_actual=metrics[pos,:,0]
        errors=(pred[pos,:,0]-prediction_actual)**2
        errorsold=(oldpred[:,:,2]-prediction_actual)**2
        old_tsc_ok=(oldpred[:,4,5]>=c-eps)&((d.rolling_state.to_numpy()[ids]=='cold_start')|(oldpred[:,4,6]<=tau))
        new_tsc_ok=(pred[pos,4,1]>=c-eps)&((d.rolling_state.to_numpy()[ids]=='cold_start')|(pred[pos,4,2]<=tau))
        for quarter in [0,1,2,3,4]:
            mask=np.ones(n,bool) if quarter==0 else np.minimum(3,np.arange(n)*4//n)+1==quarter
            nn=int(mask.sum())
            diagnostics.append(key|dict(quarter=quarter,n=nn,changed=int(changed[mask].sum()),
                new_backoff=int((trace[pos,0][mask]>0).sum()),old_backoff=int((d.support_backoff_level.to_numpy()[ids][mask]>0).sum()),
                new_empty=int(trace[pos,1][mask].sum()),old_empty=int(trace[pos,2][mask].sum()),
                old_cost_mse=float(errorsold[mask].mean()),new_cost_mse=float(errors[mask].mean()),
                old_coverage_mse=float(((oldpred[:,:,5]-metrics[pos,:,3])**2)[mask].mean()),
                new_coverage_mse=float(((pred[pos,:,1]-metrics[pos,:,3])**2)[mask].mean()),
                old_tsc_excluded=int((~old_tsc_ok[mask]).sum()),new_tsc_excluded=int((~new_tsc_ok[mask]).sum()),
                actual_cost_change=float(actual[mask].sum()),predicted_change_old=float(ppold[mask].sum()),predicted_change_new=float(ppnew[mask].sum()),
                cost_decrease=int((actual[mask]<-1e-12).sum()),cost_increase=int((actual[mask]>1e-12).sum())))
        for bid in np.unique(block):
            mask=block==bid;nn=int(mask.sum());nnr=int((mask&(np.arange(n)>=671)).sum())
            for j,method in enumerate(METHODS):
                blocks.append(key|dict(block_id=int(bid),method=method,n=nn,nr=nnr,**dict(zip(M,clean[mask,j].sum(0)))))
    cells=pd.DataFrame(records);b=pd.DataFrame(blocks);dg=pd.DataFrame(diagnostics)
    archive=pd.read_parquet(ROOT/'analysis/section46_failure_diagnosis_20260911/units'/uid/'base_cells.parquet')
    keys=['predictor','horizon_steps','target_coverage']
    current=cells[cells.method.eq('CLARA')].set_index(keys)
    expected=archive[archive.method.eq('CLARA')].set_index(keys).reindex(current.index)
    err=float(abs(current[M].to_numpy()-expected[M].to_numpy()).max());assert err<2e-10
    # Cost-change contributions reconcile with full-condition online minus frozen metrics.
    change=dg[dg.quarter.eq(0)].set_index(keys)
    joint=cells[cells.method.eq('CLARA_online')].set_index(keys).reindex(current.index)
    np.testing.assert_allclose(change.actual_cost_change/change.n,(joint.ERRF-current.ERRF).reindex(change.index),atol=2e-12,rtol=0)
    cells.to_parquet(dest/'cells.parquet',index=False);b.to_parquet(dest/'blocks_168h.parquet',index=False);dg.to_parquet(dest/'diagnostics.parquet',index=False)
    return err


def worker(task):
    farm,seed,pid,verify=task;uid=f'{farm}-S{seed}-{pid}';start=time.monotonic()
    dest=HERE/'units'/uid;dest.mkdir(parents=True,exist_ok=True)
    if (dest/'manifest.json').exists():
        man=read(dest/'manifest.json');assert man['status']=='PASS';assert man['engine_sha256']==rt.sha(HERE/'engine.py');assert man['plan_sha256']==rt.sha(HERE/'PLAN.md')
        return dict(unit=uid,status='REUSED')
    d,e,_=rt.original_farm(farm,seed,pid)
    frozen=np.stack([e[k] for k in EK],axis=2)
    th=read(S/'external_price_runs/units'/uid/'audits.json')['thresholds']
    nmin=d.selected_n_min.to_numpy(int);nu=d.selected_nu.to_numpy(float);cold=d.rolling_state.eq('cold_start').to_numpy();coverage=d.target_coverage.to_numpy(float)
    log(unit=uid,stage='loading_checked_candidate_streams')
    data=load_data(farm,seed,pid,d)
    am,af=data['adaptation'];fm,ff=data['final'];metrics=data['metrics']
    issue=ns(fm.issue_timestamp);label=ns(fm.label_timestamp);available=ns(fm.label_available_timestamp)
    assert ns(am.label_available_timestamp).max()<=issue.min() and ns(am.label_timestamp).max()<issue.min()
    aday=(ns(am.issue_timestamp)-int(8*3600e9))//int(86400e9)
    fday=(issue-int(8*3600e9))//int(86400e9)
    day_base=int(min(aday.min(),fday.min()));aday-=day_base;fday-=day_base
    mapping,stats=layout(d,int(max(aday.max(),fday.max())+1))
    add_rows(am.state_id.to_numpy(int),aday,af,0,len(am),mapping,*stats)
    initial,ilevels=evidence(np.arange(len(d)),mapping,nmin,nu,cold,stats[0],stats[1],stats[2],stats[5],stats[3],stats[4])
    np.testing.assert_array_equal(ilevels,d.support_backoff_level)
    np.testing.assert_allclose(initial,frozen,atol=3e-10,rtol=0,equal_nan=True)
    initialerr=float(np.nanmax(abs(initial-frozen)))
    choices,_=choose_many(np.arange(len(d)),initial,coverage,cold,th['coverage_shortfall_epsilon'],th['tuwr_upper'])
    original,_=rt.agreed.select(d,e,th,'DIRECTIONAL_REPAIR',1.)
    np.testing.assert_array_equal(choices,original)
    log(unit=uid,stage='adaptation_evidence_verified',max_error=initialerr,events=len(fm))
    feedback_order=np.lexsort((label,available)).astype(np.int64)
    audit={}
    if verify:
        # Intermediate evidence and nonanticipation checks use a copy of initial statistics.
        cutoff=issue.min()+int(14*86400e9)
        stop=int(np.searchsorted(issue,cutoff,side='left'))
        copied=tuple(x.copy() for x in stats)
        short=run_batches(issue,label,available,fm.state_id.to_numpy(int),fday,ff,feedback_order,mapping,nmin,nu,cold,coverage,frozen,th['coverage_shortfall_epsilon'],th['tuwr_upper'],copied,stop)
        ptr=short[3];mature=feedback_order[:ptr]
        bm=pd.concat([am,fm.iloc[mature]],ignore_index=True);bf=np.concatenate([af,ff[mature]])
        audit['intermediate_batch_evidence_max_error']=batch_verify(d,mapping,nmin,nu,cold,copied,bm,bf,day_base)
        del bm,bf,copied
        altered=ff.copy();not_mature=np.ones(len(ff),bool);not_mature[mature]=False
        altered[not_mature,:,0]+=1000.;altered[not_mature,:,1]=1.-altered[not_mature,:,1]
        copied=tuple(x.copy() for x in stats)
        other=run_batches(issue,label,available,fm.state_id.to_numpy(int),fday,altered,feedback_order,mapping,nmin,nu,cold,coverage,frozen,th['coverage_shortfall_epsilon'],th['tuwr_upper'],copied,stop)
        np.testing.assert_array_equal(short[0],other[0]);np.testing.assert_allclose(short[1],other[1],atol=0,rtol=0,equal_nan=True)
        audit['future_outcome_perturbation_no_effect']=True
        del altered,other,short,copied
    chosen,pred,trace,feedback_count=run_batches(issue,label,available,fm.state_id.to_numpy(int),fday,ff,feedback_order,mapping,nmin,nu,cold,coverage,frozen,th['coverage_shortfall_epsilon'],th['tuwr_upper'],stats)
    np.testing.assert_array_equal(chosen[:,0],original[fm.state_id.to_numpy(int)])
    assert feedback_count==int((available<=issue.max()).sum())
    np.savez_compressed(dest/'choices.npz',selected=chosen,issue=issue,state_id=fm.state_id.to_numpy(np.int32),trace=trace)
    metricerr=summarize_unit(dest,farm,seed,pid,d,frozen,fm,metrics,chosen,pred,trace)
    audit.update(future_feedback_violations=0,same_issue_feedback_violations=0,all_six_candidates_updated=True,adaptation_records=len(am),matured_final_records=feedback_count,pending_final_records=len(fm)-feedback_count,final_drain=False)
    save(dest/'manifest.json',dict(status='PASS',unit=uid,thresholds=th,audit=audit,source_inventory=data['source'],
        initial_evidence_max_error=initialerr,historical_reliability_max_error=data['reliability_error'],frozen_metric_max_error=metricerr,
        outputs={name:rt.sha(dest/name) for name in ['cells.parquet','blocks_168h.parquet','diagnostics.parquet','choices.npz']},
        engine_sha256=rt.sha(HERE/'engine.py'),trial_sha256=rt.sha(__file__),plan_sha256=rt.sha(HERE/'PLAN.md'),seconds=time.monotonic()-start))
    result=dict(unit=uid,status='PASS',seconds=round(time.monotonic()-start,1))
    log(**result);return result


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--limit',type=int);ap.add_argument('--farm');ap.add_argument('--price');a=ap.parse_args()
    tasks=[(f,s,p,s==0 and p=='R01') for f in rt.rt.FARMS for s in range(3) for p,_ in rt.rt.PRICES if (not a.farm or f==a.farm) and (not a.price or p==a.price)]
    if a.limit:tasks=tasks[:a.limit]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for future in as_completed([pool.submit(worker,t) for t in tasks]):future.result()
