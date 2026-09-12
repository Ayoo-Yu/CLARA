"""Isolated GEFCom validation, sharing the agreed commercial policy exactly."""
from pathlib import Path
import os,sys,json,hashlib
from datetime import datetime,timezone
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
    os.environ[k]='1'
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
WORK=HERE.parents[1]
EXTERNAL=WORK/'analysis/guardrail_redesign_20260908'
sys.path.insert(0,str(EXTERNAL))
import policies as agreed
EXP=WORK/'workspace/0427/0620/提交版本0703 Energy/一审/一审修订实验'
RUN=EXP/'测试CLARA/results_raw/gefcom_six_action_price_full_v1'
BASE=RUN/'selector_v1'
AGG=BASE/'aggregate_v1'
FACTROOT=EXP/'06_基线实现与训练区调参/results_raw/nested_source_selection_v1'
BUNDLES=EXP/'03_基础预测与候选区间重建/results_raw/full_rebuild_v1/bundles'
ACTIONS=['Static','ACI','AgACI','EnbPI_RH','TunedSingleConformal','EqualEndpointEnsemble']
FIXED=['FixedStatic','FixedACI','FixedAgACI','FixedEnbPI_RH','TSC','EEE']
METHODS=FIXED+['NO_SCREEN']+[agreed.name(f,s) for f in agreed.FAMILIES for s in agreed.SCALES]
PRIMARY=agreed.PRIMARY
ORIGINAL=agreed.ORIGINAL
FIELDS=['predictor','horizon_group','target_coverage','ramp_state','rolling_state','raw_width_state']
ZONES=[f'zone{i}' for i in range(1,11)]
PREDICTORS=['Ridge','GBR','MLP','QRLSTM']
HORIZONS=[1,3,6,12,24]
COVERAGES=[.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99]
PRICES=agreed.rt.PRICES
MODE='RESELECT_EACH_RATIO'
METRICS=['ERRF','capacity','exceedance','coverage','width','TUWR','TOWR','ARD','under_gap','over_gap']
AUDIT_FIELDS=['initial_empty','basic_empty','selected_basic_failure','selected_full_failure','selected_ARD_failure']
OFFICIAL={ORIGINAL:'CLARA_6A','TSC':'TunedSingleConformal','EEE':'EqualEndpointEnsemble',**{a:a for a in FIXED[:4]}}
REGIMES=['overall','ordinary','ramp']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def readj(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def writej(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def stamp():return datetime.now(timezone.utc).isoformat()
def log(stage,**kw):print(json.dumps(dict(at=stamp(),stage=stage,**kw),ensure_ascii=False,default=str),flush=True)
def checked(path,expected):
    actual=sha(path)
    assert actual==expected,(str(path),actual,expected)
    return actual
def policy_dir(zone,seed,pid):return HERE/'policies'/f'{zone}__seed{seed}'/pid

def freeze():
    receipts=[]
    aggregate=readj(AGG/'manifest.json');assert aggregate['status']=='PASS'
    root_path=RUN/'endpoint_units/root_manifest__B_UNIFIED_FIVE_HORIZON_NEW_IDENTITY.json'
    checked(root_path,aggregate['endpoint_root_sha256'])
    mono_count=0
    for zone in ZONES:
        for seed in range(3):
            for pid,_ in PRICES:
                src=BASE/'fit_parents'/f'{zone}__seed{seed}'/'children'/pid
                man=readj(src/'manifest.json');assert man['status']=='PASS'
                assert man['heldout_zone']==zone and man['seed']==seed and man['price_id']==pid
                for k in ['clara_action_evidence','clara_decisions','clara_audit']:
                    checked(src/man['files'][k]['relative_path'],man['files'][k]['sha256'])
                aud=readj(src/'clara_audit.json')
                assert aud['status']=='PASS' and not aud['heldout_zone_used_in_fit'] and zone not in aud['source_zones']
                d=pd.read_parquet(src/'clara_decisions.parquet').sort_values('full_state_code').reset_index(drop=True)
                e=pd.read_parquet(src/'clara_action_evidence.parquet').sort_values(['full_state_code','action_order']).reset_index(drop=True)
                n=len(d);assert n==6600 and len(e)==6*n
                assert np.array_equal(e.full_state_code.to_numpy().reshape(n,6),np.broadcast_to(d.full_state_code.to_numpy()[:,None],(n,6)))
                assert (e.action.to_numpy().reshape(n,6)==np.asarray(ACTIONS)[None,:]).all()
                mats={k:e[v].to_numpy(float).reshape(n,6) for k,v in {'risk_score':'risk_score','coverage':'coverage_point','tuwr':'tuwr_point','ard':'ard_point'}.items()}
                cold=e.cold_start_guardrail.to_numpy(bool).reshape(n,6)
                assert np.array_equal(cold,np.broadcast_to(d.rolling_state.eq('cold_start').to_numpy()[:,None],(n,6)))
                out=d[FIELDS+['full_state_code']].copy()
                for j,m in enumerate(FIXED):out[m]=j
                out['NO_SCREEN']=agreed.choose(mats,np.ones((n,6),bool))
                audits=[]
                for family in agreed.FAMILIES:
                    for scale in agreed.SCALES:
                        m=agreed.name(family,scale)
                        choice,trace=agreed.select(d,mats,aud['adaptive_guardrail_thresholds'],family,scale)
                        out[m]=choice
                        af=pd.DataFrame(trace);af['method']=m;af['full_state_code']=d.full_state_code;audits.append(af)
                        if m==ORIGINAL:
                            assert np.array_equal(np.asarray(ACTIONS)[choice],d.selected_action)
                            assert np.array_equal(trace['initial_empty'],d.guardrail_empty_fallback)
                            np.testing.assert_array_equal(e.guardrail_pass.to_numpy(bool).reshape(n,6).any(1),~trace['initial_empty'])
                rows=np.arange(n)
                scores=np.column_stack([mats['risk_score'][rows,out[agreed.name('DIRECTIONAL_REPAIR',s)]] for s in agreed.SCALES])
                assert (np.diff(scores,axis=1)<=1e-12).all()
                mono_count+=n
                dest=policy_dir(zone,seed,pid);dest.mkdir(parents=True,exist_ok=True)
                out.to_parquet(dest/'state_decisions.parquet',index=False)
                pd.concat(audits,ignore_index=True).to_parquet(dest/'constraint_audit.parquet',index=False)
                receipt=dict(zone=zone,seed=seed,price_id=pid,states=n,status='PASS',source_manifest_sha256=sha(src/'manifest.json'),thresholds=aud['adaptive_guardrail_thresholds'],price=aud['price'],outputs={f:sha(dest/f) for f in ['state_decisions.parquet','constraint_audit.parquet']})
                writej(dest/'manifest.json',receipt);receipts.append(receipt)
            log('policies_frozen',zone=zone,seed=seed)
    assert len(receipts)==150
    writej(HERE/'policies_frozen.json',dict(status='PASS',created_at=stamp(),plan_sha256=sha(HERE/'PLAN.md'),runtime_sha256=sha(__file__),agreed_policy_sha256=sha(EXTERNAL/'policies.py'),aggregate_manifest_sha256=sha(AGG/'manifest.json'),endpoint_root_sha256=sha(root_path),units=receipts,methods=METHODS,primary=PRIMARY,all_original_decisions_exact=True,monotone_fitted_score_states=mono_count,new_test_metrics_read=False))
    log('freeze_complete',units=150,states=mono_count,methods=len(METHODS))

if __name__=='__main__':freeze()
