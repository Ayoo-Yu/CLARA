"""Audit source-only exact state support and target state frequencies, no outcomes."""
from pathlib import Path
import sys,json,hashlib,importlib.util,time
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT/'analysis/complete_ablation_20260909'))
import runtime as rt
spec=importlib.util.spec_from_file_location('closure_builder',ROOT/'analysis/reviewer_closure_20260905/policy/build_policy_variants.py')
closure=importlib.util.module_from_spec(spec);spec.loader.exec_module(closure)

def sha(p):return rt.sha(p)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')

def audit():
    freeze=read(rt.GEOLD/'policies_frozen.json')
    assert freeze['status']=='PASS'
    assert sha(rt.ge.AGG/'manifest.json')==freeze['aggregate_manifest_sha256']
    assert sha(Path(rt.agreed.__file__))==freeze['agreed_policy_sha256']
    v47=ROOT/'analysis/ablation_integration_v47_20260911/data_manifest.json'
    latest=read(v47)
    assert latest['CLARA_identical_across_archives'] and latest['source_metrics_exactly_reproduced']
    resolved={}
    for old,h in latest['source_hashes'].items():
        rel=old.replace('\\','/').split('/analysis/',1)[1]
        p=ROOT/'analysis'/rel
        assert sha(p)==h
        resolved[str(p)]=h
    rows=[];details=[];inventory=[]
    for zone in rt.ge.ZONES:
        for seed in range(3):
            uid=f'{zone}__seed{seed}'
            cache=ROOT/'analysis/reviewer_closure_20260905/policy/parents'/uid/'source_cache'
            cm=read(cache/'manifest.json');stats,_=closure.read_source_cache(cache,cm['source_identity'])
            assert tuple(stats.actions)==tuple(rt.ge.ACTIONS) and zone not in stats.source_zones
            assert cm['source_identity']['endpoint_root_manifest_sha256']==freeze['endpoint_root_sha256']
            counts=stats.arrays['event_count'].sum(0)
            zones=(stats.arrays['event_count']>0).sum(0)
            guards=stats.arrays['guardrail_count'].sum(0)
            cold=stats.states.rolling_state.eq('cold_start').to_numpy()
            mask={'exact_n_zero':counts==0,'exact_one_zone':zones==1,'exact_two_zones':zones==2,'warm_guard_zero':(~cold)&(guards==0),'exact_n_positive_lt10':(counts>0)&(counts<10)}
            inventory.append({'unit':uid,'source_cache':str(cache),'source_manifest_sha256':sha(cache/'manifest.json'),'source_arrays_sha256':sha(cache/'arrays.npz'),'source_statistic_identity_sha256':cm['source_statistic_identity_sha256'],'actions':list(stats.actions),'states':len(stats.states),'heldout_zone_absent':True})
            detail=stats.states.copy();detail['zone']=zone;detail['seed']=seed;detail['exact_count']=counts;detail['exact_zone_count']=zones;detail['exact_guard_count']=guards
            allfreq=None
            for pid,_ in rt.PRICES:
                child=rt.GEBASE/'replay_parents'/uid/'children'/('RESELECT_EACH_RATIO__'+pid)
                m=read(child/'manifest.json')
                record=m['files']['clara_state_counts'];p=child/record['relative_path'];assert sha(p)==record['sha256']
                df=pd.read_parquet(p,columns=['full_state_code','event_count'])
                freq=df.groupby('full_state_code').event_count.sum().reindex(stats.states.full_state_code,fill_value=0).to_numpy()
                if allfreq is None:allfreq=freq.copy()
                else:np.testing.assert_array_equal(freq,allfreq)
                d,_,_,_=rt.original_ge(zone,seed,pid)
                np.testing.assert_array_equal(d.full_state_code,stats.states.full_state_code)
                row=dict(zone=zone,seed=seed,price_id=pid,target_events=int(freq.sum()),target_used_states=int((freq>0).sum()),original_backoff_events=int(freq[d.support_backoff_level.to_numpy()>0].sum()),target_state_file_sha256=sha(p))
                for name,mm in mask.items():row[name+'_states']=int(mm.sum());row[name+'_used_states']=int((mm&(freq>0)).sum());row[name+'_events']=int(freq[mm].sum())
                rows.append(row)
            detail['target_event_count_one_price']=allfreq;details.append(detail)
            print(uid,rows[-1]['exact_n_zero_events'],rows[-1]['exact_one_zone_events'],rows[-1]['warm_guard_zero_events'],flush=True)
    pd.DataFrame(rows).to_csv(HERE/'support_audit_by_unit_price.csv',index=False)
    pd.concat(details).to_parquet(HERE/'exact_support_by_state.parquet',index=False)
    pd.DataFrame(inventory).to_json(HERE/'source_inventory.json',orient='records',indent=2,force_ascii=False)
    totals=pd.DataFrame(rows).select_dtypes('number').drop(columns=['seed']).sum().to_dict()
    result={'status':'PASS','uses_target_outcomes':False,'purpose':'Only state frequency and source support audit before boundary rule selection','latest_accepted_manifest_sha256':sha(v47),'latest_accepted_source_hashes_verified':resolved,'latest_baseline':latest['full_baseline'],'accepted_policy_freeze_sha256':sha(rt.GEOLD/'policies_frozen.json'),'endpoint_root_sha256':freeze['endpoint_root_sha256'],'totals_across_five_prices':totals,'files':{p.name:sha(p) for p in [HERE/'support_audit_by_unit_price.csv',HERE/'exact_support_by_state.parquet',HERE/'source_inventory.json']}}
    save(HERE/'support_audit.json',result)
    print(json.dumps(result,ensure_ascii=True,indent=2),flush=True)

if __name__=='__main__':audit()
