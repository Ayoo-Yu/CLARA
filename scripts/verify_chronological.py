"""Check released chronological evidence and replay one matched target stream."""
from pathlib import Path
import argparse,json,os,sys
import numpy as np
import pandas as pd

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,help='Extracted gefcom_chronological_v1_1_0 directory')
    p.add_argument('--zone',default='zone1');p.add_argument('--seed',type=int,default=0)
    p.add_argument('--predictor',default='GBR');p.add_argument('--horizon',type=int,default=1)
    p.add_argument('--refit-source',action='store_true',help='Also refit all five policies from the released source sufficient statistics')
    p.add_argument('--output',type=Path,default=Path('outputs/chronological_verification'))
    a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    if a.data_root:os.environ['CLARA_CHRONO_ROOT']=str(a.data_root.resolve())
    sys.path.insert(0,str(repo/'src/chronological'))
    import forward_common as C
    import clara_forward as F
    if not (C.RUN/'DATA_READY.json').exists():
        p.error('Extract chronological evidence and candidate archives first; see src/chronological/README.md')
    folder=C.RUN/'clara'/a.zone/f'seed{a.seed}'
    bundle=F.load_bundle(folder)
    assert a.zone not in bundle.stats.source_zones
    assert bundle.audit['all_source_feedback_precedes_test']
    evidence=pd.read_parquet(folder/'clara_action_evidence.parquet')
    states=pd.read_parquet(folder/'clara_decisions.parquet')
    audit=json.loads((folder/'clara_audit.json').read_text())
    saved=np.load(folder/'clara_selected_actions.npy')
    if a.refit_source:
        rebuilt=F.fit_policies(bundle)
        np.testing.assert_array_equal(rebuilt.selected_actions,saved)
        for field in ['risk_score','coverage_point','tuwr_point','ard_point']:
            np.testing.assert_allclose(rebuilt.evidence[field],evidence[field],atol=1e-12,rtol=0,equal_nan=True)
    chosen=[]
    for i,entry in enumerate(audit['per_price']):
        pid=entry['price']['price_id']
        st=states[states.price_id.eq(pid)].sort_values('full_state_code')
        ev=evidence[evidence.price_id.eq(pid)].sort_values(['full_state_code','action_order'])
        fields={'risk_score':'risk_score','coverage':'coverage_point','tuwr':'tuwr_point','ard':'ard_point'}
        e={k:ev[v].to_numpy().reshape(6600,6) for k,v in fields.items()}
        pick,_=F.directional_repair(st,e,entry['thresholds'])
        np.testing.assert_array_equal(pick,saved[i]);chosen.append(pick)
    ti=C.select_tsc({a.zone});d=C.load_stream(a.zone,a.seed,a.predictor,a.horizon,'target',ti)
    assert np.all(d['issue_ns']>=1378339200000000000)
    ids=F.state_codes(d,a.predictor,a.horizon,bundle.width_edges)
    replay=np.asarray(chosen)[:,ids]
    with np.load(folder/'decisions.npz') as f:
        np.testing.assert_array_equal(replay,f[f'{a.predictor}__H{a.horizon:02d}'])
    ref=pd.read_parquet(C.RUN/'summary/seed_cells'/f'{a.zone}_seed{a.seed}.parquet')
    ref=ref[(ref.method=='CLARA')&(ref.predictor==a.predictor)&(ref.horizon==a.horizon)]
    errors=[]
    for pi,rho in enumerate(C.PRICES):
        for ci,c in enumerate(C.COVERAGES):
            lo=d['lower'][replay[pi,ci],ci,np.arange(len(d['y']))]
            up=d['upper'][replay[pi,ci],ci,np.arange(len(d['y']))]
            metrics=C.evaluation_metrics(lo,up,d['y'],d['center'],c,rho)
            pid=['R01','R02','Rref','R10','R20'][pi]
            old=ref[(ref.price_id==pid)&(ref.target_coverage==c)].iloc[0]
            for key in ['mean_errf','coverage','tuwr','ard']:
                err=abs(metrics[key]-old[key]);errors.append(err)
                np.testing.assert_allclose(metrics[key],old[key],atol=1e-12,rtol=0)
    result=dict(status='PASS',zone=a.zone,seed=a.seed,predictor=a.predictor,horizon=a.horizon,
        state_price_choices=33000,target_stream_choices=int(replay.size),target_forecasts=len(d['y']),
        source_sufficient_statistics_checksum='PASS',outer_zone_excluded=True,
        target_after_cutoff=True,maximum_metric_difference=max(errors),
        refitted_from_source_statistics=a.refit_source,
        scope='Fitted-policy reconstruction and target replay; this does not retrain base forecasts.')
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
