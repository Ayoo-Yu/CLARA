"""Recompute Table 5 and its 10,000-resample intervals from released conditions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
d=pd.read_parquet(HERE/'summary/condition_metrics.parquet')
q=d[(d.price_id=='R05P6179775281')&(d.regime=='overall')]
table=pd.read_csv(HERE/'summary/Table5_reference_price.csv').set_index('method')
metrics=['ERRF','capacity','exceedance','coverage','width','TUWR','TOWR','ARD','under_gap','over_gap','IS']
means=q.groupby('method')[metrics].mean()
np.testing.assert_allclose(means.reindex(table.index),table[metrics],atol=1e-11,rtol=0)
zones=[f'zone{i}' for i in range(1,11)]
cost=q.groupby(['zone_or_farm','method']).ERRF.mean().unstack().reindex(zones)
draw=np.random.default_rng(20260911).integers(0,10,(10000,10))
base=cost.CLARA.to_numpy()
for m,r in table.iterrows():
    x=cost[m].to_numpy()
    estimate=100*(x.mean()/base.mean()-1)
    ci=np.quantile(100*(x[draw].mean(1)/base[draw].mean(1)-1),[.025,.975])
    np.testing.assert_allclose(estimate,r.delta_ERRF_pct,atol=1e-10,rtol=0)
    np.testing.assert_allclose(ci,[r.delta_ERRF_pct_CI_low,r.delta_ERRF_pct_CI_high],atol=1e-10,rtol=0)
np.testing.assert_allclose(d.ERRF,d.capacity+d.exceedance,atol=1e-11,rtol=0)
out=ROOT/'outputs/chronological_ablation_verification';out.mkdir(parents=True,exist_ok=True)
result=dict(status='PASS',table_rows=len(table),all_condition_rows=len(d),zone_bootstrap_resamples=10000,bootstrap_seed=20260911,scope='Released condition summaries and paired zone-bootstrap intervals; no model retraining')
(out/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
print(json.dumps(result,indent=2))
