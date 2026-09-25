"""Shared inputs for the 2026-09-25 cutoff-limited frozen transfer check.

Keep original 60/20/20 forecaster/calibrator initialization. Source fitting
uses the initial 28-day prefix of the original test period; held-out evaluation
starts at a prespecified common cutoff. Original artifacts remain read-only.
"""
from __future__ import annotations
import os
for _key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[_key]='1'
from pathlib import Path
import sys, json, hashlib
from functools import lru_cache
import numpy as np
import pandas as pd

from _paths import DATA_ROOT as RUN, EVALUATED as ROOT, CODE_ROOT
EXP = ROOT/'workspace/0427/0620/提交版本0703 Energy/一审/一审修订实验'
AUTH = EXP/'_权威代码/code'
BASE = Path(os.environ.get('CLARA_BASE_ROOT', str(EXP/'03_基础预测与候选区间重建/results_raw/full_rebuild_v1/base')))
BUNDLES = Path(os.environ.get('CLARA_CANDIDATE_ROOT', str(EXP/'03_基础预测与候选区间重建/results_raw/full_rebuild_v1/bundles')))
sys.path.insert(0,str(AUTH))
from clara_event_contract import load_frozen_contracts
from baseline_conformal import conformal_grid

CONTRACTS=load_frozen_contracts()
COVERAGES=np.array([.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,.99])
PREDICTORS=('Ridge','GBR','MLP','QRLSTM')
HORIZONS=(1,3,6,12,24)
SEEDS=(0,1,2)
ZONES=tuple(f'zone{i}' for i in range(1,11))
PRICES=np.array([1.,2.,20/3.56,10.,20.])
PI=3.56
REFERENCE_PRICE=20/3.56
ACTIONS=('Static','ACI','AgACI','EnbPI_RH','TunedSingleConformal','EqualEndpointEnsemble')
MAIN_CONFIG_INDICES=[0,3,6,12]
ROLLING_STATES=('cold_start','undercoverage_pressure','overcoverage_pressure','volatile','stable')
RAMP_STATES=('ordinary','ramp')
HOUR_NS=3_600_000_000_000

def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def stream_id(zone,seed,predictor,horizon):
    return f'{predictor}-H{int(horizon):02d}-{zone}-S{int(seed)}'

def dataset_path(zone,seed,predictor,horizon,split):
    return RUN/'datasets'/split/f'{stream_id(zone,seed,predictor,horizon)}.npz'

def load_stream(zone,seed,predictor,horizon,split,tsc_index=0):
    split={'fit':'source','test':'target'}.get(split,split)
    with np.load(dataset_path(zone,seed,predictor,horizon,split),allow_pickle=False) as f:
        d={k:f[k] for k in f.files}
    inds=MAIN_CONFIG_INDICES+[int(tsc_index or 0),19]
    for key in ('lower','upper','candidate_tuwr','candidate_ard'):
        d[key]=d[key][inds]
    d.update(zone=zone,seed=int(seed),predictor=predictor,horizon=int(horizon),split=split)
    return d

def load_all_configs(zone,seed,predictor,horizon,split):
    with np.load(dataset_path(zone,seed,predictor,horizon,split),allow_pickle=False) as f:
        return {k:f[k] for k in f.files}

@lru_cache(maxsize=100)
def _select_tsc(excluded):
    from baseline_selection import select_configuration
    d=pd.read_parquet(RUN/'tsc_zone_scores.parquet')
    d=d[~d.inner_validation_zone.isin(excluded)]
    out=select_configuration(contracts=CONTRACTS,validation_scores=d).as_record()
    configs=conformal_grid(CONTRACTS)
    index=next(i for i,c in enumerate(configs) if c.configuration_id==out['selected_configuration_id'])
    return index,out

def select_tsc(excluded_zones):
    return _select_tsc(tuple(sorted(excluded_zones)))[0]

def tsc_selection_details(excluded_zones):
    return _select_tsc(tuple(sorted(excluded_zones)))[1]

def streams(zones=ZONES,seeds=SEEDS):
    return [(z,s,p,h) for z in zones for s in seeds for p in PREDICTORS for h in HORIZONS]

def endpoint_components(lower,upper,center,y):
    capacity=np.maximum(upper-center,0.)+np.maximum(center-lower,0.)
    miss=np.maximum(y-upper,0.)+np.maximum(lower-y,0.)
    return capacity,miss,(lower<=y)&(y<=upper)

def evaluation_metrics(lower,upper,y,center,coverage,price):
    """One complete chronological series; reported windows are 168 hourly points."""
    cap,miss,covered=endpoint_components(lower,upper,center,y)
    prefix=np.r_[0,np.cumsum(covered,dtype=np.float64)]
    wc=(prefix[168:]-prefix[:-168])/168 if len(y)>=168 else np.array([])
    delta=1.96*np.sqrt(coverage*(1-coverage)/168)
    return dict(n=len(y),mean_errf=float(PI*(cap.mean()+price*miss.mean())),
                mean_capacity_cost=float(PI*cap.mean()),mean_exceedance_cost=float(PI*price*miss.mean()),
                coverage=float(covered.mean()),tuwr=float(np.mean(wc<coverage-delta)),
                towr=float(np.mean(wc>coverage+delta)),ard=float(np.mean(np.abs(wc-coverage))),
                under_gap=float(np.mean(np.maximum(coverage-wc,0))),
                over_gap=float(np.mean(np.maximum(wc-coverage,0))),
                interval_score=float(np.mean(upper-lower+2/(1-coverage)*miss)),
                window_count=len(wc))
