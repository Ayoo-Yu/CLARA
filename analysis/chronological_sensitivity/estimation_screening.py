"""One-parameter refits and screening sensitivity on chronological held-out suffix."""
from sensitivity_common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse

def worker(task):
    z,s=task;start=time.perf_counter();out=HERE/'estimation'/f'{z}_seed{s}';out.mkdir(parents=True,exist_ok=True)
    ident={**base_identity(),'script':sha(__file__)}
    if (out/'COMPLETE.json').exists():
        r=read(out/'COMPLETE.json');assert r['identity']==ident
        for n,h in r['outputs'].items():assert sha(out/n)==h
        return r
    parent=CHRONO/'clara'/z/f'seed{s}';bundle=cf.load_bundle(parent)
    assert z not in bundle.stats.source_zones and bundle.audit['all_source_feedback_precedes_test']
    temporal=bundle.stats.stream_audit
    assert (temporal.available_max_ns<=temporal.test_cutoff_ns).all() and (temporal.label_max_ns<temporal.test_cutoff_ns).all()
    policies={'CLARA':np.load(parent/'clara_selected_actions.npy')};params={}
    # Explicit unchanged refit protects against accidental algorithm-version drift.
    ref,_,audit=fit_choices(bundle.stats)
    np.testing.assert_array_equal(ref,policies['CLARA']);params['CLARA']=audit
    for name in ESTIMATION:
        policies[name],_,params[name]=fit_choices(bundle.stats,name)
    ev=pd.read_parquet(parent/'clara_action_evidence.parquet')
    old_audit=read(parent/'clara_audit.json')
    for multiplier in (.5,1.,2.):
        values=[]
        for pi,price in enumerate(cf.PRICES):
            x=ev[ev.price_id.eq(price.price_id)].sort_values(['full_state_code','action_order'],kind='mergesort')
            mats={k:x[col].to_numpy().reshape(6600,6) for k,col in [('risk_score','risk_score'),('coverage','coverage_point'),('tuwr','tuwr_point'),('ard','ard_point')]}
            thresholds=copy.deepcopy(old_audit['per_price'][pi]['thresholds'])
            for key in ('coverage_shortfall_epsilon','tuwr_upper'):thresholds[key]*=multiplier
            values.append(cf.directional_repair(bundle.stats.states,mats,thresholds)[0])
        policies[f'TOL_{multiplier:g}']=np.stack(values)
    np.testing.assert_array_equal(policies['TOL_1'],policies['CLARA'])
    np.savez_compressed(out/'state_choices.npz',**policies)
    save(out/'fit_parameters.json',params)
    # Source fitting is complete before any held-out outcome is loaded.
    save(out/'FIT_FROZEN.json',dict(status='PASS',identity=ident,parent_source_audit=sha(parent/'source_audit.json'),
         source_zones=list(bundle.stats.source_zones),width_edges_sha256=bundle.audit['width_edges_sha256'],
         choices_sha256=sha(out/'state_choices.npz'),target_outcomes_used_in_fit=False))
    rows=[];ti=bundle.audit['tsc_configuration_index'];names=list(policies)
    for p in fc.PREDICTORS:
        for h in fc.HORIZONS:
            d=fc.load_stream(z,s,p,h,'target',ti);codes=cf.state_codes(d,p,h,bundle.width_edges);ix=np.arange(len(d['y']))
            for name,policy in policies.items():
                choice=policy[:,codes]
                for pi,pid in enumerate(PIDS):
                    for ci,c in enumerate(fc.COVERAGES):
                        act=choice[pi,ci]
                        m=metric_values(d,d['lower'][act,ci,ix],d['upper'][act,ci,ix],ci,pi)
                        rows.append(dict(zone=z,seed=s,predictor=p,horizon=h,target_coverage=c,price_id=pid,
                                         price_ratio=float(fc.PRICES[pi]),method=name,**m))
    frame=pd.DataFrame(rows);assert len(frame)==len(names)*1100
    ref=pd.read_parquet(CHRONO/'summary/seed_cells'/f'{z}_seed{s}.parquet').query("method=='CLARA'")
    keys=['zone','seed','predictor','horizon','target_coverage','price_id']
    joined=frame[frame.method.eq('CLARA')].merge(ref,on=keys,validate='one_to_one',suffixes=('_new','_main'))
    for m in (*EVENTS,*WINDOWS,'n','window_count'):np.testing.assert_allclose(joined[m+'_new'],joined[m+'_main'],rtol=0,atol=1e-12)
    frame.to_parquet(out/'metrics.parquet',index=False)
    report=dict(status='PASS',zone=z,seed=s,identity=ident,methods=names,metric_rows=len(frame),
                baseline_reproduced=True,source_only=True,seconds=time.perf_counter()-start,
                outputs={n:sha(out/n) for n in ('metrics.parquet','state_choices.npz','fit_parameters.json','FIT_FROZEN.json')})
    save(out/'COMPLETE.json',report);return report

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pilot',action='store_true');ap.add_argument('--workers',type=int,default=2);a=ap.parse_args()
    tasks=[('zone1',0)] if a.pilot else [(z,s) for z in fc.ZONES for s in fc.SEEDS]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(worker,t) for t in tasks]):
            r=f.result();print(r['zone'],r['seed'],r['status'],round(r['seconds'],2),flush=True)
