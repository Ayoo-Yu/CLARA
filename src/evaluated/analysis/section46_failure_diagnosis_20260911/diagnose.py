"""Read-only replay of accepted local-farm policies and matched diagnostic choices."""
from pathlib import Path
import os, sys, json, time, argparse, inspect, hashlib
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT/'analysis/complete_ablation_20260909'))
import runtime as rt
sys.path.insert(0,str(ROOT/'analysis/chapter4_evidence_explanation_20260910'))
from oracle_audit import allowed_set
import s09_linucb_batch_accelerator as original_lin
F=rt.FIELDS; A=list(rt.rt.ACTIONS); T=A.index('TunedSingleConformal_Local')
M=['ERRF','capacity','exceedance','coverage','width','TUWR','TOWR','ARD','under_gap','over_gap']
S=ROOT/'analysis/all_method_price_evidence_20260905'
PHASES=['CLARA','COST_SCORE','RAW_COST','SHRUNK_COST','NO_SE','NO_SHRINK']+A

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def log(**kw):print(json.dumps(kw,ensure_ascii=False,default=str),flush=True)

def phase_choices(d,e,thresholds):
    allowed,clara=allowed_set(d,e,thresholds)
    def select(metric,mask):
        ee=e.copy();ee['risk_score']=metric
        return rt.agreed.choose(ee,mask)
    allok=np.ones_like(allowed)
    out=[clara,select(e['risk_score'],allok),select(e['risk_mean'],allok),
         select(e['risk_shrunken'],allok),select(e['risk_shrunken'],allowed),
         select(e['risk_mean']+e['risk_se'],allowed)]
    out += [np.full(len(d),j,dtype=int) for j in range(6)]
    return np.column_stack(out),allowed

def rolling(covered,c,n=672):
    out=np.full((len(covered),covered.shape[1],5),np.nan)
    cs=np.vstack([np.zeros((1,covered.shape[1])),np.cumsum(covered,axis=0)])
    gap=(cs[n:]-cs[:-n])/n-c;tol=1.96*np.sqrt(c*(1-c)/n)
    out[n-1:]=np.stack([gap < -tol,gap > tol,abs(gap),np.maximum(-gap,0),np.maximum(gap,0)],axis=2)
    return out

def thin(f):
    cols=list(dict.fromkeys(F+['event_id','issue_timestamp','label_timestamp','label_available_timestamp']+[a+'__errf' for a in A]))
    return f[cols].copy()

def local_support(farm,seed,d):
    parts=[];inventory=[]
    for p in rt.rt.PREDICTORS:
        for h in rt.rt.HORIZONS:
            path=S/'external_streams'/f'{farm}-S{seed}-{p}-H{h:02d}'/'base_adaptation.parquet'
            f=pd.read_parquet(path,columns=F+['issue_timestamp'])
            parts.append(f)
            inventory.append(dict(predictor=p,horizon_steps=h,n=len(f),start=str(f.issue_timestamp.min()),end=str(f.issue_timestamp.max()),times=f.issue_timestamp.nunique()))
    f=pd.concat(parts,ignore_index=True);f['_day']=pd.to_datetime(f.issue_timestamp).dt.floor('D')
    levels=[list(x['fields']) for x in rt.rt.load_frozen_contracts().support_levels]
    result={};sel=d.support_backoff_level.to_numpy(int)
    ns=[];days=[]
    for fields in levels:
        if not fields:
            ns.append(np.full(len(d),len(f),float));days.append(np.full(len(d),f._day.nunique(),float))
        else:
            g=f.groupby(fields,dropna=False).agg(n=('_day','size'),days=('_day','nunique'))
            index=pd.MultiIndex.from_frame(d[fields]) if len(fields)>1 else pd.Index(d[fields[0]])
            g=g.reindex(index).fillna(0)
            ns.append(g.n.to_numpy(float));days.append(g.days.to_numpy(float))
    result['fit_n_full']=ns[0];result['fit_days_full']=days[0]
    result['fit_n_used']=np.stack(ns)[sel,np.arange(len(d))]
    result['fit_days_used']=np.stack(days)[sel,np.arange(len(d))]
    return result,inventory

def make_lin_engine():
    source=inspect.getsource(original_lin.run_linucb_batch_equivalent)
    changes=[
      ('    selected_output = np.empty(len(final_facts), dtype=object)',
       '    selected_output = np.empty(len(final_facts), dtype=object)\n    trace_output = np.empty((len(final_facts), 9), dtype=np.float64)'),
      ('        scores = np.empty((len(batch), len(core.ACTIONS)), dtype=np.float64)',
       '        scores = np.empty((len(batch), len(core.ACTIONS)), dtype=np.float64)\n        all_means = np.empty_like(scores)\n        all_bonuses = np.empty_like(scores)'),
      ('        minimum = scores.min(axis=1)',
       '            all_means[:, action_index] = mean_cost\n            all_bonuses[:, action_index] = float(exploration_alpha) * np.sqrt(variance)\n        minimum = scores.min(axis=1)'),
      ('            selected_output[output_positions] = selected_actions[evaluation_mask]',
       '            selected_output[output_positions] = selected_actions[evaluation_mask]\n            rr = np.arange(len(batch))[evaluation_mask]\n            cc = batch.loc[evaluation_mask, "_clara_index"].to_numpy(dtype=np.int64)\n            ss = selected_index[evaluation_mask]\n            greedy = (all_means <= all_means.min(axis=1)[:, None] + 1e-12).argmax(axis=1)\n            uu = np.array([updates[a] for a in core.ACTIONS])\n            trace_output[output_positions] = np.column_stack([all_means[rr,ss],all_means[rr,cc],all_means[rr,4],all_bonuses[rr,ss],all_bonuses[rr,cc],greedy[evaluation_mask],uu[ss],uu[cc],np.full(len(rr),feedback_count)])'),
      ('    return selected_output.astype(str), audit','    return selected_output.astype(str), audit, trace_output')]
    for a,b in changes:
        assert source.count(a)==1,a
        source=source.replace(a,b)
    scope=dict(original_lin.__dict__)
    exec(compile(source,str(HERE/'diagnose.py')+'::observation_only', 'exec'),scope)
    return scope['run_linucb_batch_equivalent']

def lin_replay(farm,seed,pid,d,e,choices,dest):
    path=dest/'lin_trace.npz'
    if path.exists():
        z=np.load(path);return z['selected'],z['trace']
    parts=[];adapt=[];slices={};cursor=0;ixd=pd.MultiIndex.from_frame(d[F])
    for p in rt.rt.PREDICTORS:
        for h in rt.rt.HORIZONS:
            f=rt.fr.load_stream(farm,seed,p,h,'final',pid)
            # The original global replay receives each stream in its archived row order.
            ix=ixd.get_indexer(pd.MultiIndex.from_frame(f[F]));assert (ix>=0).all()
            x=thin(f);x['_clara_index']=choices[ix,0];parts.append(x)
            slices[(p,h)]=(cursor,cursor+len(x));cursor+=len(x)
            adapt.append(thin(rt.fr.load_stream(farm,seed,p,h,'adaptation',pid)))
    final=pd.concat(parts,ignore_index=True);adapt=pd.concat(adapt,ignore_index=True)
    del parts
    config=read(rt.rt.S09/'configs/s09_minimum_external_evaluation_v1.json')['linucb']['configuration']
    log(unit=dest.name,stage='lin_replay',final=len(final),adaptation=len(adapt))
    old=original_lin.core.ACTIONS;original_lin.core.ACTIONS=tuple(A)
    try:
        selected,audit,trace=make_lin_engine()(final_facts=final,adaptation_facts=adapt,contracts=rt.rt.load_frozen_contracts(),**config)
    finally:original_lin.core.ACTIONS=old
    archived=read(S/'external_price_runs/units'/dest.name/'audits.json')['linucb']
    for k in ['event_count','adaptation_event_count','final_event_count','feedback_count_before_terminal_drain','pending_feedback_count','future_feedback_violation_count','within_issue_feedback_use_count','action_update_count']:
        assert audit[k]==archived[k],(k,audit[k],archived[k])
    selected=pd.Categorical(selected,categories=A).codes
    np.savez_compressed(path,selected=selected,trace=trace)
    save(dest/'lin_audit.json',audit|{'matched_archived_audit':True,'observation_only_instrumentation':True})
    return selected,trace

def worker(task):
    farm,seed,pid,do_lin=task;start=time.monotonic();uid=f'{farm}-S{seed}-{pid}'
    dest=HERE/'units'/uid;dest.mkdir(parents=True,exist_ok=True)
    suffix='lin' if do_lin else 'base'
    if (dest/f'{suffix}_manifest.json').exists():
        man=read(dest/f'{suffix}_manifest.json');assert man['status']=='PASS';return dict(unit=uid,status='REUSED',mode=suffix)
    d,e,_=rt.original_farm(farm,seed,pid)
    thresholds=read(S/'external_price_runs/units'/uid/'audits.json')['thresholds']
    choices,allowed=phase_choices(d,e,thresholds)
    old=pd.read_parquet(rt.FAOLD/'policies'/uid/'state_decisions.parquet').sort_values('state_id')
    np.testing.assert_array_equal(choices[:,0],old[rt.agreed.PRIMARY]);assert d[F].equals(old[F].reset_index(drop=True))
    support,inventory=local_support(farm,seed,d)
    phases=PHASES.copy();lin_all=lin_trace=None
    if do_lin:
        lin_all,lin_trace=lin_replay(farm,seed,pid,d,e,choices,dest);phases += ['LinUCB']
    index=pd.MultiIndex.from_frame(d[F]);rows=[];cats=[];state=[];counts=[];cursor=0
    warm=d.rolling_state.ne('cold_start').to_numpy()[:,None]
    cov_ok=e['coverage']>=d.target_coverage.to_numpy()[:,None]-thresholds['coverage_shortfall_epsilon']
    tw_ok=(~warm)|(np.isfinite(e['tuwr'])&(e['tuwr']<=thresholds['tuwr_upper']))
    basic=cov_ok&tw_ok
    for p in rt.rt.PREDICTORS:
        for h in rt.rt.HORIZONS:
            f=rt.fr.load_stream(farm,seed,p,h,'final',pid)
            ix=index.get_indexer(pd.MultiIndex.from_frame(f[F]));assert (ix>=0).all()
            a=[rt.rt.core._metric_arrays_from_endpoints(f,f[v+'__candidate_lower'].to_numpy(float),f[v+'__candidate_upper'].to_numpy(float),theta=rt.rt.theta_for(pid)) for v in A]
            cand=np.stack([np.column_stack([v[k] for v in a]) for k in ['errf','reserve','miss','covered','width']],axis=2)
            chosen=choices[ix]
            trace=None
            if do_lin:
                chosen=np.column_stack([chosen,lin_all[cursor:cursor+len(f)]])
                trace=lin_trace[cursor:cursor+len(f)]
            cursor+=len(f)
            vals=np.take_along_axis(cand,chosen[:,:,None],axis=1)
            for c,g in f.groupby('target_coverage',sort=True):
                pos=g.sort_values(['issue_timestamp','event_id'],kind='mergesort').index.to_numpy()
                ids=ix[pos];act=chosen[pos];n=len(pos);nr=n-671;assert nr>0
                value=np.concatenate([vals[pos],rolling(vals[pos,:,3],float(c))],axis=2)
                clean=np.nan_to_num(value,nan=0);valid=np.arange(n)>=671
                key=dict(zone_or_farm=farm,seed=seed,price_id=pid,predictor=p,horizon_steps=h,target_coverage=float(c))
                den=np.array([n]*5+[nr]*5)
                for j,method in enumerate(phases):rows.append(key|dict(method=method,event_count=n,reliability_event_count=nr,**dict(zip(M,clean[:,j].sum(0)/den))))
                # Actual outcomes by full state, for descriptive history-to-test calibration checks.
                unique=np.unique(ids)
                for st in unique:
                    at=ids==st;nn=int(at.sum());nnr=int(valid[at].sum())
                    rec=key|dict(state_id=int(st),n=nn,nr=nnr,test_days=pd.to_datetime(g.loc[pos[at],'issue_timestamp']).dt.floor('D').nunique())
                    rec.update({k:d.iloc[st][k] for k in ['ramp_state','rolling_state','raw_width_state','selected_n_min','selected_nu','support_backoff_level']})
                    rec.update({k:v[st] for k,v in support.items()})
                    rec.update({k:float(e[k][st,choices[st,0]]) for k in ['risk_mean','risk_shrunken','risk_se']})
                    for j,name in enumerate(A):
                        for mi,m in enumerate(M[:5]):rec[name+'__'+m]=float(cand[pos[at],j,mi].mean())
                        rec[name+'__fit_cost']=float(e['risk_shrunken'][st,j]);rec[name+'__fit_coverage']=float(e['coverage'][st,j])
                        rec[name+'__fit_TUWR']=float(e['tuwr'][st,j])
                    rec['clara_action']=A[choices[st,0]];rec['empty']=not basic[st].any()
                    state.append(rec)
                # Full-denominator paired contributions; rolling windows are never restarted for categories.
                for baseline,bj in [('TSC',6+T)]+([('LinUCB',len(phases)-1)] if do_lin else []):
                    ba=act[:,bj];ca=act[:,0];rr=np.arange(n)
                    bc=cov_ok[ids,ba];bt=tw_ok[ids,ba];anyok=basic[ids].any(1)
                    reason=np.select([ca==ba,~anyok,~bc&~bt,~bc,~bt],['same_action','no_eligible','coverage_and_TUWR','coverage_only','TUWR_only'],default='both_eligible')
                    quarter=np.minimum(3,np.arange(n)*4//n)+1
                    axes={'reason':reason,'support_level':d.support_backoff_level.to_numpy()[ids].astype(str),'rolling':d.rolling_state.to_numpy()[ids],
                          'width':d.raw_width_state.to_numpy()[ids],'regime':d.ramp_state.to_numpy()[ids],'quarter':quarter.astype(str),
                          'chosen_action':np.asarray(A)[ca], 'support_reason':np.char.add(np.char.add(d.support_backoff_level.to_numpy()[ids].astype(str),'|'),reason)}
                    diff=clean[:,0]-clean[:,bj]
                    pc=e['risk_shrunken'][ids,ca];pb=e['risk_shrunken'][ids,ba];actual=diff[:,0];pred=pc-pb
                    extra={'fit_difference':pred,'fit_raw_difference':e['risk_mean'][ids,ca]-e['risk_mean'][ids,ba],
                           'fit_SE_difference':e['risk_se'][ids,ca]-e['risk_se'][ids,ba],
                           'clara_squared_error':(pred-actual)**2,'clara_absolute_error':abs(pred-actual),
                           'fit_baseline_gap':e['coverage'][ids,ba]-c,'actual_baseline_gap':value[:,bj,3]-c,
                           'fit_clara_gap':e['coverage'][ids,ca]-c,'actual_clara_gap':value[:,0,3]-c,
                           'fit_baseline_TUWR':np.nan_to_num(e['tuwr'][ids,ba]),'fit_n_used':support['fit_n_used'][ids],
                           'fit_days_used':support['fit_days_used'][ids], 'changed':(ca!=ba).astype(float),
                           'cost_win':(actual < -1e-12).astype(float),'cost_loss':(actual > 1e-12).astype(float)}
                    if baseline=='LinUCB':
                        tr=trace[pos];predlin=tr[:,1]-tr[:,0]
                        extra.update(lin_predicted_difference=predlin,lin_squared_error=(predlin-actual)**2,
                                     lin_absolute_error=abs(predlin-actual),lin_selected_updates=tr[:,6],lin_clara_updates=tr[:,7],
                                     lin_feedback_count=tr[:,8],lin_exploration_changed=(tr[:,5]!=ba).astype(float))
                    for axis,labels in axes.items():
                        for label in np.unique(labels):
                            at=labels==label
                            rec=key|dict(baseline=baseline,axis=axis,category=str(label),n=int(at.sum()),nr=int(valid[at].sum()),total_n=n,total_nr=nr)
                            rec.update({m+'_sum':float(v) for m,v in zip(M,diff[at].sum(0))})
                            rec.update({m+'_sum':float(v[at].sum()) for m,v in extra.items()})
                            cats.append(rec)
                    if baseline=='LinUCB':
                        for action in range(6):counts.append(key|dict(selected_action=A[action],event_count=int((ba==action).sum())))
            log(unit=uid,stage='stream',predictor=p,horizon=h,mode=suffix)
    rows=pd.DataFrame(rows);cats=pd.DataFrame(cats);state=pd.DataFrame(state)
    old=pd.read_parquet(rt.FAOLD/'evaluation'/uid/'cell_metrics.parquet');old=old[old.regime.eq('overall')&old.method.isin([rt.agreed.PRIMARY,'TSC','NO_SCREEN'])].copy()
    old.method=old.method.replace({rt.agreed.PRIMARY:'CLARA','TSC':A[T],'NO_SCREEN':'COST_SCORE'})
    keys=['predictor','horizon_steps','target_coverage','method']
    aa=rows.set_index(keys);bb=old.set_index(keys)
    err=float(np.max(abs(aa.reindex(bb.index)[M].to_numpy()-bb[M].to_numpy())));assert err<1e-10
    if do_lin:
        archive=pd.read_parquet(S/'external_price_runs/units'/uid/'cell_metrics.parquet')
        archive=archive[archive.method.eq('LinUCB_6A_Local')&archive.regime.eq('overall')]
        k=keys[:-1];a=rows[rows.method.eq('LinUCB')].set_index(k).sort_index();b=archive.set_index(k).reindex(a.index)
        mm={'ERRF':'mean_errf','capacity':'mean_reserve','exceedance':'mean_miss','coverage':'empirical_coverage','width':'average_width','TUWR':'TUWR','TOWR':'TOWR','ARD':'ARD'}
        linerr=max(float(np.max(abs(a[x]-b[y]))) for x,y in mm.items());assert linerr<1e-10,linerr
        ac=pd.read_parquet(S/'external_price_runs/units'/uid/'action_counts.parquet');ac=ac[ac.method.eq('LinUCB_6A_Local')&ac.regime.eq('overall')]
        ck=k+['selected_action'];ref=ac.set_index(ck).event_count;got=pd.DataFrame(counts).set_index(ck).event_count
        assert (got.reindex(ref.index)==ref).all();assert got.sum()==ref.sum()
    rows.to_parquet(dest/f'{suffix}_cells.parquet',index=False);cats.to_parquet(dest/f'{suffix}_categories.parquet',index=False);state.to_parquet(dest/f'{suffix}_states.parquet',index=False)
    save(dest/f'{suffix}_manifest.json',dict(status='PASS',unit=uid,mode=suffix,thresholds=thresholds,accepted_policy_exact=True,
         accepted_metric_max_error=err,lin_metric_max_error=linerr if do_lin else None,adaptation_inventory=inventory,
         code_sha256=rt.sha(__file__),seconds=time.monotonic()-start))
    return dict(unit=uid,status='PASS',mode=suffix,seconds=round(time.monotonic()-start,1))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=2);ap.add_argument('--lin',action='store_true');ap.add_argument('--limit',type=int);a=ap.parse_args()
    tasks=[(f,s,p,a.lin) for f in rt.rt.FARMS for s in range(3) for p,_ in rt.PRICES]
    if a.limit:tasks=tasks[:a.limit]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):log(**f.result())
