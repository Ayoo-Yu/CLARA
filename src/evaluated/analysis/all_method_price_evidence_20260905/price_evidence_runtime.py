"""Isolated price-extension paths and reproducibility helpers (unique module name)."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

ROOT = Path(__file__).resolve().parent
WORK = ROOT.parents[1]
EXP = WORK / 'workspace/0427/0620/提交版本0703 Energy/一审/一审修订实验'
TEST = EXP / '测试CLARA'
S09 = EXP / '09_商业场站外部验证'
for _path in [EXP / '_权威代码/code', S09 / 'scripts', TEST / 'scripts']:
    sys.path.insert(0, str(_path))

import numpy as np
import pandas as pd
from clara_errf import ErrfTheta
from clara_event_contract import load_frozen_contracts
import s09_minimum_external_core as core
import run_extended_action_external as extended

PRICES = [('R01', 1.0), ('R02', 2.0), ('R05P6179775281', 20.0/3.56), ('R10', 10.0), ('R20', 20.0)]
MAIN_ID = PRICES[2][0]
ACTIONS = extended.SIX_ACTIONS
FARMS = ['FarmA', 'FarmB']
PREDICTORS = ['Ridge', 'GBR', 'MLP', 'QRLSTM']
HORIZONS = [4, 24, 48, 96]
COVERAGES = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]
METHODS = ['CLARA_6A_Local', 'CART_6A_Local', 'LinUCB_6A_Local', *ACTIONS]
assert len(METHODS) == 9 and len(set(METHODS)) == 9

def stamp():
    return datetime.now(timezone.utc).isoformat()

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for part in iter(lambda: handle.read(2**20), b''):
            h.update(part)
    return h.hexdigest()

def log(stage, **kwargs):
    print(json.dumps(dict(at=stamp(), stage=stage, **kwargs), ensure_ascii=False, default=str), flush=True)

def readj(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def writej(path, value):
    core.write_json_atomic(value, Path(path))

def writep(path, value):
    core.write_parquet_atomic(value, Path(path))

def theta_for(price_id):
    ratio = dict(PRICES)[price_id]
    kappa = 20.0 if price_id == MAIN_ID else 3.56 * ratio
    return ErrfTheta(kappa_plus=kappa, kappa_minus=kappa, theta_id='external_price_extension_' + price_id)

def input_identities():
    paths = {
        'plan': ROOT / 'PLAN.md',
        's09_protocol': S09 / 'configs/s09_minimum_external_evaluation_v1.json',
        'adaptive_protocol': TEST / 'configs/test_clara_four_versions_v1.json',
        'matched_protocol': TEST / 'configs/test_clara_matched_six_action_selectors_v1.json',
        's09_core': S09 / 'scripts/s09_minimum_external_core.py',
        'extended_runner': TEST / 'scripts/run_extended_action_external.py',
        'selector_adapter': TEST / 'scripts/extended_selector_adapters.py',
        'linucb_accelerator': S09 / 'scripts/s09_linucb_batch_accelerator.py',
        'original_selected_tsc': S09 / 'results_raw/local_conformal_selection_v1/selected_configurations.json',
    }
    return {k: sha(v) for k,v in paths.items()}
