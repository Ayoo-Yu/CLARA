"""Read-only access to sealed chronological data and current CLARA mathematics."""
from pathlib import Path
import sys,os,json,copy,hashlib,time
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
CODE_DIR=Path(__file__).resolve().parent
REPO=CODE_DIR.parents[1]
sys.path.insert(0,str(REPO/'src/chronological'))
from _paths import DATA_ROOT as CHRONO
HERE=Path(os.environ.get('CLARA_SENSITIVITY_OUTPUT',str(REPO/'outputs/chronological_sensitivity')))
HERE.mkdir(exist_ok=True,parents=True)
import numpy as np
import pandas as pd
import forward_common as fc
import clara_forward as cf
from dataclasses import replace
PIDS=('R01','R02','Rref','R10','R20')
ESTIMATION=tuple([f'NMIN_{n}' for n in (10,20,30,50)]+[f'NU_{n}' for n in (0,10,30,60,100)])
TAUS=(.08,.10,.12,.14,.16,.18)
EVENTS=('mean_errf','mean_capacity_cost','mean_exceedance_cost','coverage','width','interval_score')
WINDOWS=('tuwr','towr','ard','under_gap','over_gap')
sha=fc.sha256
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str),encoding='utf8')
def variant_config(name):
    contracts,config=cf.contracts_and_config();config=copy.deepcopy(config)
    if name.startswith('NMIN_'):
        n=int(name.split('_')[1]);config['fixed_support']['n_min']=n
        config['adaptive_support']['n_min_candidates']=[n]
        config['adaptive_support']['n_min_no_candidate_rule']=n
    elif name.startswith('NU_'):
        n=int(name.split('_')[1]);config['adaptive_support']['nu_candidates']=[n]
        config['adaptive_support']['nu_rules']=[dict(minimum_ratio=0.,nu=n)]
    elif name!='CLARA':raise ValueError(name)
    return contracts,config
def fit_choices(stats,name='CLARA'):
    contract,config=variant_config(name)
    prepared=cf.original.prepare_clara_fit(stats,contracts=contract,v4_config=config,require_formal_identity=False)
    choices=[];backs=[];records=[]
    for price in cf.PRICES:
        model=cf.original.fit_price_conditioned_clara(prepared,price=price,v4_config=config)
        ev=model.action_evidence.sort_values(['full_state_code','action_order'],kind='mergesort')
        mats={k:ev[col].to_numpy().reshape(6600,6) for k,col in [('risk_score','risk_score'),('coverage','coverage_point'),('tuwr','tuwr_point'),('ard','ard_point')]}
        assert np.isfinite(mats['risk_score']).all() and np.isfinite(mats['coverage']).all()
        warm=stats.states.rolling_state.ne('cold_start').to_numpy()
        assert np.isfinite(mats['tuwr'][warm]).all() and np.isfinite(mats['ard'][warm]).all()
        choice,guard=cf.directional_repair(stats.states,mats,model.audit['adaptive_guardrail_thresholds'])
        dec=model.decisions.sort_values('full_state_code')
        if name.startswith('NMIN_'):assert dec.selected_n_min.eq(int(name.split('_')[1])).all()
        if name.startswith('NU_'):assert dec.selected_nu.eq(int(name.split('_')[1])).all()
        choices.append(choice);backs.append(dec.support_backoff_level.to_numpy())
        records.append(dict(price_id=price.price_id,n_min_counts=model.audit['n_min_counts'],nu_counts=model.audit['nu_counts'],
                            empty_states=int(guard['guardrail_empty_fallback'].sum()),thresholds=model.audit['adaptive_guardrail_thresholds']))
    return np.stack(choices),np.stack(backs),records
def metric_values(d,lo,up,ci,pi):
    m=fc.evaluation_metrics(lo,up,d['y'],d['center'],fc.COVERAGES[ci],fc.PRICES[pi])
    m['width']=float(np.mean(up-lo));return m
def base_identity():
    return dict(protocol=sha(CHRONO/'protocol.json'),data_ready=sha(CHRONO/'DATA_READY.json'),
                fitter=sha(cf.__file__),common=sha(fc.__file__),
                math=sha(cf.original.__file__),sensitivity_common=sha(__file__),plan=sha(CODE_DIR/'protocol.json'))
