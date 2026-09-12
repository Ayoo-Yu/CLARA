"""Prespecified component ablations; no target outcomes used in fitting."""
from pathlib import Path
import os, sys, json, hashlib, time
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
EXP = ROOT/'workspace/0427/0620/提交版本0703 Energy/一审/一审修订实验'
GEOLD = ROOT/'analysis/guardrail_redesign_gefcom_20260908'
FAOLD = ROOT/'analysis/guardrail_redesign_20260908'
DEEP = ROOT/'analysis/clara_tsc_deep_audit_20260908'
for p in (GEOLD, DEEP, ROOT/'analysis/all_method_price_evidence_20260905', EXP/'测试CLARA/scripts'):
    sys.path.insert(0,str(p))
import gefcom_guardrail_runtime as ge
import replay_mechanisms as fr
import gefcom_six_action_selector_core_v1 as gc
agreed = ge.agreed
rt = agreed.rt
FIELDS = ge.FIELDS
TASK = FIELDS[:3]
LEVEL_FIELDS = [TASK, TASK[1:], TASK[2:], []]
METHODS = ['CLARA','NO_CONTEXT','NO_SHRINKAGE','NO_SE_PENALTY','NO_CONSTRAINTS','NO_VIOLATION_PRIORITY']
LABELS = ['Full CLARA','Without context grouping','Without risk shrinkage','Without uncertainty penalty','Without coverage constraints','Without violation priority']
CN = ['完整CLARA','去除上下文分组','去除风险收缩','去除标准误差惩罚','去除覆盖约束','去除违反程度优先规则']
METRICS = ge.METRICS
PRICES = ge.PRICES
GEBASE = ge.BASE
SOURCE_FIELDS = ['capacity_sum','miss_sum','covered_sum','guard_covered_sum','tuwr_sum','ard_sum']

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def log(stage,**kw):print(json.dumps(dict(stage=stage,**kw),ensure_ascii=False,default=str),flush=True)
def checked(p,h): assert sha(p)==h,str(p)

def states():
    return pd.read_parquet(GEBASE/'fit_parents/zone1__seed0/children/R01/clara_decisions.parquet').sort_values('full_state_code').reset_index(drop=True)

def groups(d):
    return [d[f].drop_duplicates().sort_values(f).reset_index(drop=True) if f else pd.DataFrame({'global':[0]}) for f in LEVEL_FIELDS]

def mapping(frame,g,fields):
    if not fields:return np.zeros(len(frame),int)
    ix=pd.MultiIndex.from_frame(g).get_indexer(pd.MultiIndex.from_frame(frame[fields]));assert (ix>=0).all()
    return ix

def evidence(levels,selected,nu,d,cold_nan):
    n=len(d);rows=np.arange(n);nu=np.broadcast_to(np.asarray(nu,float),(n,))
    raw=np.stack([l['mean'] for l in levels]);se=np.stack([l['se'] for l in levels])
    parent=levels[-1]['mean'].copy();shrunk=[None]*len(levels);parents=[None]*len(levels)
    shrunk[-1]=parent.copy();parents[-1]=parent.copy()
    for k in range(len(levels)-2,-1,-1):
        parents[k]=parent.copy();count=levels[k]['n'][:,None]
        denominator=count+nu[:,None]
        parent=np.divide(count*raw[k]+nu[:,None]*parent,denominator,out=raw[k].copy(),where=denominator>0)
        shrunk[k]=parent.copy()
    cold=d.rolling_state.eq('cold_start').to_numpy()
    def take(field):return np.stack([l[field] for l in levels])[selected,rows]
    e={'risk_mean':raw[selected,rows],'risk_parent':np.stack(parents)[selected,rows],
       'risk_shrunken':np.stack(shrunk)[selected,rows],'risk_se':se[selected,rows],
       'coverage':np.where(cold[:,None],take('cov_all'),take('cov_guard')),
       'tuwr':take('tuwr'),'ard':take('ard')}
    e['risk_score']=e['risk_shrunken']+e['risk_se']
    if cold_nan:
        e['tuwr'][cold]=np.nan;e['ard'][cold]=np.nan
    return e

def reduced_fit(levels,d,cold_nan):
    profiles={};ratios={}
    for count in (10,20,30,50):
        selected=np.full(len(d),-1,int)
        for k,l in enumerate(levels):
            use=(selected<0)&(l['n']>=count)&(l['zones']>=(3 if cold_nan else 1))
            selected[use]=k
        assert (selected>=0).all()
        e=evidence(levels,selected,30,d,cold_nan)
        profiles[count]=selected;ratios[count]=rt.core._top_two_margin_ratio(e)
    nmin=np.full(len(d),50,int);decided=np.zeros(len(d),bool)
    for count in (10,20,30,50):
        use=~decided&(ratios[count]>=1);nmin[use]=count;decided[use]=True
    selected=np.array([profiles[n][i] for i,n in enumerate(nmin)])
    diagnostic=evidence(levels,selected,30,d,cold_nan)
    ratio=np.nanmedian(abs(diagnostic['risk_mean']-diagnostic['risk_parent']),axis=1)/np.maximum(np.nanmedian(diagnostic['risk_se'],axis=1),1e-12)
    nu=np.select([ratio>=2,ratio>=1,ratio>=.5,ratio>=.25],[0,10,30,60],default=100)
    return evidence(levels,selected,nu,d,cold_nan),dict(context_n_min=nmin,context_nu=nu,context_level=selected+3)

def variants(d,e,reduced,thresholds):
    choices={};audits=[]
    for method in METHODS:
        current={k:v.copy() for k,v in (reduced if method=='NO_CONTEXT' else e).items()}
        if method=='NO_SHRINKAGE':current['risk_score']=current['risk_mean']+current['risk_se']
        if method=='NO_SE_PENALTY':current['risk_score']=current['risk_shrunken'].copy()
        family='DIRECTIONAL' if method=='NO_VIOLATION_PRIORITY' else 'DIRECTIONAL_REPAIR'
        choice,trace=agreed.select(d,current,thresholds,family,1.)
        if method=='NO_CONSTRAINTS':choice=agreed.choose(current,np.ones(current['risk_score'].shape,bool))
        choices[method]=choice
        q=pd.DataFrame(trace);q['method']=method;q['state_id']=np.arange(len(d));audits.append(q)
    # All remaining dynamic categories are irrelevant after coarse grouping; cold-start is availability.
    q=d[TASK].copy();q['cold']=d.rolling_state.eq('cold_start');q['choice']=choices['NO_CONTEXT']
    assert q.groupby(TASK+['cold']).choice.nunique().max()==1
    return pd.DataFrame(choices),pd.concat(audits,ignore_index=True)

def original_ge(zone,seed,pid):
    p=GEBASE/'fit_parents'/f'{zone}__seed{seed}'/'children'/pid
    man=read(p/'manifest.json');assert man['status']=='PASS'
    for key in ['clara_decisions','clara_action_evidence','clara_audit']:
        checked(p/man['files'][key]['relative_path'],man['files'][key]['sha256'])
    d=pd.read_parquet(p/'clara_decisions.parquet').sort_values('full_state_code').reset_index(drop=True)
    f=pd.read_parquet(p/'clara_action_evidence.parquet').sort_values(['full_state_code','action_order']).reset_index(drop=True)
    cols={'risk_mean':'risk_mean','risk_parent':'risk_parent_mean','risk_shrunken':'risk_shrunken_mean','risk_se':'risk_zone_cluster_se','risk_score':'risk_score','coverage':'coverage_point','tuwr':'tuwr_point','ard':'ard_point'}
    e={k:f[v].to_numpy(float).reshape(len(d),6) for k,v in cols.items()}
    aud=read(p/'clara_audit.json')
    assert zone not in aud['source_zones'] and not aud['heldout_zone_used_in_fit']
    return d,e,aud,man

def original_farm(farm,seed,pid):
    p=DEEP/'units'/f'{farm}-S{seed}-{pid}';man=read(p/'manifest.json')
    assert man['status']=='PASS'
    for f in ['state_trace.parquet','historical_action_evidence.parquet']:checked(p/f,man['outputs'][f])
    d=pd.read_parquet(p/'state_trace.parquet').sort_values('state_id').reset_index(drop=True)
    h=pd.read_parquet(p/'historical_action_evidence.parquet').merge(d[FIELDS+['state_id']],on=FIELDS,validate='many_to_one')
    e={k:h.pivot(index='state_id',columns='action',values=k).reindex(index=d.state_id,columns=rt.ACTIONS).to_numpy(float)
       for k in ['risk_mean','risk_parent','risk_shrunken','risk_se','risk_score','coverage','tuwr','ard']}
    return d,e,man
