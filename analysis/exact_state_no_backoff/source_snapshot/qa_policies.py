"""Post-build policy QA only; no target outcomes or performance metrics."""
from build_policies import *

def run():
    frozen=read(HERE/'policies_frozen.json');assert frozen['status']=='PASS'
    checks=[];changes=[];manual=[]
    for zone in rt.ge.ZONES:
        for seed in range(3):
            uid=f'{zone}__seed{seed}';dest=HERE/'policies'/uid
            assert sha(dest/'manifest.json')==frozen['policy_parents'][str((dest/'manifest.json').relative_to(HERE))]
            m=read(dest/'manifest.json')
            for n,h in m['outputs'].items():assert sha(dest/n)==h
            q=pd.read_parquet(dest/'decisions.parquet')
            e=pd.read_parquet(dest/'action_evidence.parquet')
            assert len(q)==33000 and len(e)==198000
            z=q.exact_zero.to_numpy();assert (q.loc[z,'NO_BACKOFF_EXACT']==0).all()
            assert q.loc[~z,'ablated_backoff_level'].eq(0).all()
            assert q.loc[z,'ablated_backoff_level'].eq(-1).all()
            assert q.loc[~q.was_backoff,'CLARA'].equals(q.loc[~q.was_backoff,'NO_BACKOFF_EXACT'])
            cache=ROOT/'analysis/reviewer_closure_20260905/policy/parents'/uid/'source_cache'
            cm=read(cache/'manifest.json');stats,_=closure.read_source_cache(cache,cm['source_identity'])
            contract=core.six_action_contracts(core.load_frozen_contracts())
            for price in stats.prices:
                d=q[q.price_id.eq(price.price_id)].sort_values('full_state_code').reset_index(drop=True)
                f=e[e.price_id.eq(price.price_id)].sort_values(['full_state_code','action_order']).reset_index(drop=True)
                positive=d.exact_count.gt(0).to_numpy()
                assert f.loc[np.repeat(~positive,6),list(NAMES.values())].isna().all().all()
                assert f.loc[np.repeat(positive,6),'support_n'].to_numpy().tolist()==np.repeat(d.exact_count[positive].to_numpy(),6).tolist()
                assert f.loc[np.repeat(d.exact_zone_count.eq(1).to_numpy(),6),'risk_zone_cluster_se'].eq(0).all()
                changed=d.CLARA.ne(d.NO_BACKOFF_EXACT)
                for subset,mask in {'all':np.ones(len(d),bool),'nonempty':positive,'zero':~positive,'nonempty_was_backoff':positive&d.was_backoff.to_numpy()}.items():
                    changes.append({'zone':zone,'seed':seed,'price_id':price.price_id,'subset':subset,'events':int(d.loc[mask,'target_event_count_one_price'].sum()),'changed_choice_events':int(d.loc[mask&changed,'target_event_count_one_price'].sum())})
                # Direct scalar reconstruction from exact source-statistic columns,
                # independently of the level-aggregation and evidence-builder helpers.
                eligible=d.index[positive & d.was_backoff.to_numpy() & d.target_event_count_one_price.gt(0).to_numpy()].tolist()
                if not eligible:eligible=d.index[positive].tolist()
                selected=eligible[::max(1,len(eligible)//3)][:3]
                arrays=stats.arrays_for_price(price)
                for i in selected:
                    state=stats.states.iloc[i];nu=float(d.loc[i,'selected_nu'])
                    parent=None;exactraw=None;exactse=None
                    for lev in reversed(contract.support_levels):
                        mask=np.ones(len(stats.states),bool)
                        for fld in lev['fields']:mask&=stats.states[fld].eq(state[fld]).to_numpy()
                        counts=arrays['event_count'][:,mask].sum(1)
                        sums=arrays['errf_sum'][:,:,mask].sum(2)
                        present=counts>0
                        means=sums[:,present]/counts[present][None,:]
                        raw=means.mean(1)
                        if parent is None:parent=raw
                        else:parent=(counts.sum()*raw+nu*parent)/(counts.sum()+nu)
                        if lev['level']==0:
                            exactraw=raw
                            exactse=means.std(1,ddof=1)/np.sqrt(present.sum()) if present.sum()>1 else np.zeros(6)
                    got=f.iloc[i*6:(i+1)*6]
                    np.testing.assert_allclose(got.risk_mean,exactraw,atol=1e-11,rtol=0)
                    np.testing.assert_allclose(got.risk_shrunken_mean,parent,atol=1e-11,rtol=0)
                    np.testing.assert_allclose(got.risk_zone_cluster_se,exactse,atol=1e-11,rtol=0)
                    np.testing.assert_allclose(got.risk_score,parent+exactse,atol=1e-11,rtol=0)
                    manual.append({'unit':uid,'price_id':price.price_id,'full_state_code':int(state.full_state_code),'source_count':int(d.loc[i,'exact_count']),'source_zones':int(d.loc[i,'exact_zone_count']),'manual_scalar_reconstruction':'PASS'})
            checks.append({'unit':uid,'rows':len(q),'source_hashes':'PASS','zero_Static':'PASS','exact_supported_choice_unchanged':'PASS','nonempty_level_zero':'PASS','zero_evidence_unavailable':'PASS'})
    pd.DataFrame(checks).to_csv(HERE/'policy_qa_by_unit.csv',index=False)
    pd.DataFrame(changes).to_csv(HERE/'choice_change_counts.csv',index=False)
    pd.DataFrame(manual).to_csv(HERE/'manual_source_reconstruction.csv',index=False)
    save(HERE/'policy_qa.json',{'status':'PASS','frozen_policy_sha256':sha(HERE/'policies_frozen.json'),'new_target_outcomes_used':False,'units':len(checks),'manual_scalar_source_reconstruction_cases':len(manual),'full_reference_max_evidence_error':frozen['full_evidence_max_abs_error'],'files':{p.name:sha(p) for p in [HERE/'policy_qa_by_unit.csv',HERE/'choice_change_counts.csv',HERE/'manual_source_reconstruction.csv']}})
    print('POLICY_QA_PASS',len(checks),len(manual),flush=True)

if __name__=='__main__':run()
