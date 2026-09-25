from __future__ import annotations
from forward_common import *
import argparse,time
from concurrent.futures import ProcessPoolExecutor,as_completed
from clara_forward import build_source_stats,fit_policies,replay_stream,save_policies

def one(task):
    z,s=task; start=time.perf_counter()
    out=RUN/'clara'/z/f'seed{s}'; out.mkdir(parents=True,exist_ok=True)
    done=out/'COMPLETE.json'
    identity={name:sha256((Path(__file__).parent if name.endswith('.py') else RUN)/name) for name in ('protocol.json','forward_common.py','clara_forward.py','DATA_READY.json','tsc_zone_scores.parquet')}
    if done.exists():
        previous=json.loads(done.read_text(encoding='utf8'))
        if previous.get('input_identity')!=identity:
            raise RuntimeError(f'Existing CLARA checkpoint has a different input identity: {done}')
        if previous['decision_sha256']!=sha256(out/'decisions.npz'):
            raise RuntimeError(f'CLARA checkpoint decision hash mismatch: {done}')
        return previous
    ti=select_tsc({z})
    target_cutoff=min(int(load_stream(z,s,p,h,'target',ti)['issue_ns'][0]) for p in PREDICTORS for h in HORIZONS)
    bundle=build_source_stats(z,s,ti,load_stream,cutoff_ns=target_cutoff)
    model=fit_policies(bundle)
    evidence=model.evidence
    warm=evidence.rolling_state.ne('cold_start')
    finite_audit={
        'all_risk_scores_finite':bool(np.isfinite(evidence.risk_score).all()),
        'all_coverage_estimates_finite':bool(np.isfinite(evidence.coverage_point).all()),
        'all_warm_TUWR_estimates_finite':bool(np.isfinite(evidence.loc[warm,'tuwr_point']).all()),
        'all_warm_ARD_estimates_finite':bool(np.isfinite(evidence.loc[warm,'ard_point']).all()),
    }
    if not all(finite_audit.values()):
        raise RuntimeError(f'Inspect incomplete fitted evidence before replay: {finite_audit}')
    save_policies(model,out)
    decisions={}
    for p in PREDICTORS:
        for h in HORIZONS:
            d=load_stream(z,s,p,h,'target',ti)
            a=replay_stream(model,d,p,h)
            assert a.shape==(5,11,len(d['y'])) and np.all((a>=0)&(a<6))
            decisions[f'{p}__H{h:02d}']=a.astype(np.int8)
    np.savez_compressed(out/'decisions.npz',**decisions)
    record=dict(zone=z,seed=s,tsc_index=ti,source_time_audit='PASS',seconds=time.perf_counter()-start,
                source_identity=bundle.audit['identity_sha256'],decision_sha256=sha256(out/'decisions.npz'),
                evidence_finite=finite_audit,protocol_sha256=sha256(RUN/'protocol.json'),
                fitter_sha256=sha256(Path(__file__).resolve().parent / 'clara_forward.py'),input_identity=identity)
    done.write_text(json.dumps(record,indent=2),encoding='utf8')
    return record

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--workers',type=int,default=2);ap.add_argument('--outer');ap.add_argument('--seed',type=int)
    args=ap.parse_args()
    if not (RUN/'DATA_READY.json').exists(): raise RuntimeError('Validated full data not ready')
    tasks=[(z,s) for z in ([args.outer] if args.outer else ZONES) for s in ([args.seed] if args.seed is not None else SEEDS)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fs=[pool.submit(one,t) for t in tasks]
        for i,f in enumerate(as_completed(fs),1):
            r=f.result();print(i,len(tasks),json.dumps(r),flush=True)

if __name__=='__main__': main()
