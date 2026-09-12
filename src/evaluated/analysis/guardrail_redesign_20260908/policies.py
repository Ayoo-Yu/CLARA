"""Prespecified factorial guardrail policies; original score and tie order."""
from pathlib import Path
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import json,sys,hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
SOURCE=WORK/'analysis/clara_tsc_deep_audit_20260908'
PRICE=WORK/'analysis/all_method_price_evidence_20260905'
sys.path.insert(0,str(PRICE))
import price_evidence_runtime as rt
core=rt.core
core.ACTIONS=rt.ACTIONS
FIELDS=list(core.STATE_FIELDS)
ACTIONS=list(rt.ACTIONS)
FAMILIES=['ORIGINAL','FALLBACK_REPAIR','DIRECTIONAL','DIRECTIONAL_REPAIR']
SCALES=[.5,1.,2.]
def name(family,scale):return family+'_S'+{.5:'05',1.:'10',2.:'20'}[scale]
METHODS=['TSC','EEE','NO_SCREEN']+[name(f,s) for f in FAMILIES for s in SCALES]
PRIMARY=name('DIRECTIONAL_REPAIR',1.)
ORIGINAL=name('ORIGINAL',1.)

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def writej(p,data):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(data,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def choose(e,allowed,tol=1e-12):
    assert allowed.any(1).all()
    candidates=allowed.copy()
    risk=e['risk_score']
    best=np.min(np.where(candidates,risk,np.inf),axis=1)
    candidates &= risk <= best[:,None]+tol
    coverage=e['coverage']
    best=np.max(np.where(candidates,coverage,-np.inf),axis=1)
    candidates &= coverage >= best[:,None]-tol
    for values in [e['tuwr'],e['ard']]:
        finite=np.isfinite(values);count=candidates.sum(1);fc=(candidates&finite).sum(1)
        assert not ((fc>0)&(fc<count)).any()
        applies=(fc==count)&(count>0)
        best=np.min(np.where(candidates&finite,values,np.inf),axis=1)
        candidates &= (~applies[:,None]) | (finite&(values<=best[:,None]+tol))
    assert candidates.any(1).all()
    return candidates.argmax(1)

def select(states,e,thresholds,family,scale):
    eps=float(thresholds['coverage_shortfall_epsilon'])*scale
    tau=float(thresholds['tuwr_upper'])*scale
    ard_limit=float(thresholds['ard_upper'])
    target=states.target_coverage.to_numpy(float)[:,None]
    warm=states.rolling_state.ne('cold_start').to_numpy()[:,None]
    cv=e['coverage'] >= target-eps
    tv=(~warm) | (np.isfinite(e['tuwr'])&(e['tuwr']<=tau))
    av=(~warm) | (np.isfinite(e['ard'])&(e['ard']<=ard_limit))
    basic=cv&tv
    full=basic&av
    uses_ard=family in ['ORIGINAL','FALLBACK_REPAIR']
    repairs=family in ['FALLBACK_REPAIR','DIRECTIONAL_REPAIR']
    initial=full if uses_ard else basic
    empty=~initial.any(1)
    basic_empty=~basic.any(1)
    excess_cov=np.maximum(0,target-eps-e['coverage'])/eps
    excess_tuwr=np.where(warm,np.maximum(0,e['tuwr']-tau)/tau,0)
    excess_tuwr=np.where(warm&~np.isfinite(e['tuwr']),np.inf,excess_tuwr)
    violation=np.maximum(excess_cov,excess_tuwr)
    eligible=initial.copy()
    if repairs:
        recover=empty & ~basic_empty
        eligible[recover]=basic[recover]
        none=empty & basic_empty
        minimum=np.min(violation,axis=1)
        eligible[none]=violation[none] <= minimum[none,None]+1e-12
    else:
        eligible[empty]=True
    selected=choose(e,eligible)
    rows=np.arange(len(states))
    if repairs:
        assert basic[rows,selected][~basic_empty].all()
        np.testing.assert_allclose(violation[rows,selected][basic_empty],violation.min(1)[basic_empty],rtol=0,atol=1e-12)
    audit={'initial_empty':empty,'basic_empty':basic_empty,'selected_basic_failure':~basic[rows,selected],
           'selected_full_failure':~full[rows,selected],'selected_max_excess':violation[rows,selected],
           'selected_ARD_failure':~av[rows,selected]}
    return selected,audit

def freeze():
    receipts=[]
    for farm in rt.FARMS:
        for seed in range(3):
            for pid,_ in rt.PRICES:
                uid=f'{farm}-S{seed}-{pid}';src=SOURCE/'units'/uid;dest=HERE/'policies'/uid
                man=json.loads((src/'manifest.json').read_text(encoding='utf-8'));assert man['status']=='PASS'
                for f in ['historical_action_evidence.parquet','state_trace.parquet']:
                    assert sha(src/f)==man['outputs'][f]
                trace=pd.read_parquet(src/'state_trace.parquet')
                history=pd.read_parquet(src/'historical_action_evidence.parquet')
                history=history.merge(trace[FIELDS+['state_id']],on=FIELDS,validate='many_to_one')
                e={k:history.pivot(index='state_id',columns='action',values=k).reindex(index=trace.state_id,columns=ACTIONS).to_numpy(float) for k in ['risk_mean','risk_score','coverage','tuwr','ard']}
                states=trace[FIELDS];policy=trace[FIELDS+['state_id']].copy()
                policy['TSC']=ACTIONS.index('TunedSingleConformal_Local');policy['EEE']=ACTIONS.index('EqualEndpointEnsemble')
                policy['NO_SCREEN']=choose(e,np.ones(e['risk_score'].shape,bool))
                audits=[]
                for family in FAMILIES:
                    for scale in SCALES:
                        method=name(family,scale)
                        choice,audit=select(states,e,man['thresholds'],family,scale)
                        policy[method]=choice
                        item=pd.DataFrame(audit);item['method']=method;item['state_id']=trace.state_id.to_numpy();audits.append(item)
                        if family=='ORIGINAL' and scale==1:
                            assert np.array_equal(np.asarray(ACTIONS)[choice],trace.selected_action)
                            original,empty,_=core._select_local_actions(states=states,evidence=e,thresholds=man['thresholds'],tolerance=1e-12)
                            assert np.array_equal(choice,original) and np.array_equal(audit['initial_empty'],empty)
                        if family=='FALLBACK_REPAIR':
                            original,_=select(states,e,man['thresholds'],'ORIGINAL',scale)
                            assert np.array_equal(choice[~audit['initial_empty']],original[~audit['initial_empty']])
                dest.mkdir(parents=True,exist_ok=True)
                policy.to_parquet(dest/'state_decisions.parquet',index=False)
                pd.concat(audits,ignore_index=True).to_parquet(dest/'constraint_audit.parquet',index=False)
                receipt=dict(unit=uid,states=len(policy),original_actions_exact=True,repair_properties_pass=True,
                             source_manifest_sha256=sha(src/'manifest.json'),outputs={f:sha(dest/f) for f in ['state_decisions.parquet','constraint_audit.parquet']})
                writej(dest/'manifest.json',dict(status='PASS',**receipt));receipts.append(receipt)
    writej(HERE/'policies_frozen.json',dict(status='PASS',created_at=rt.stamp(),units=receipts,plan_sha256=sha(HERE/'PLAN.md'),policy_script_sha256=sha(__file__),methods=METHODS,primary_method=PRIMARY,new_test_metrics_read=False))
    print(json.dumps(dict(status='PASS',units=len(receipts),methods=len(METHODS),primary=PRIMARY)),flush=True)

if __name__=='__main__':freeze()
