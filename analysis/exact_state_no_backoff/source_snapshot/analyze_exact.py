"""Paired analysis of newly frozen exact-state ablation, preserving manuscript weighting."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OUT=HERE/'summary'
MET=['ERRF','capacity','exceedance','coverage','width','TUWR','TOWR','ARD','under_gap','over_gap']
METHODS=['CLARA','NO_BACKOFF_EXACT']
KEY=['zone_or_farm','price_id','predictor','horizon_steps','target_coverage','regime','method']
ZONES=[f'zone{i}' for i in range(1,11)]
MAIN='R05P6179775281'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def csv(d,name):d.to_csv(OUT/name,index=False,encoding='utf-8-sig',float_format='%.17g')
def save(p,obj):Path(p).write_text(json.dumps(obj,ensure_ascii=False,indent=2,default=str),encoding='utf-8')
def pool(frame,keys):
    d=frame.copy()
    for k in MET:d[k]*=d.event_count if k in MET[:5] else d.reliability_event_count
    d=d.groupby(keys,as_index=False)[MET+['event_count','reliability_event_count']].sum(min_count=1)
    for k in MET:d[k]/=d.event_count if k in MET[:5] else d.reliability_event_count
    return d

def paired_summary(d,subset='all'):
    out=[]
    for pid in ['R01','R02',MAIN,'R10','R20','ALL']:
        g=d if pid=='ALL' else d[d.price_id.eq(pid)]
        pairs=g.pivot(index=KEY[:5],columns='method',values='event_count')
        assert (pairs.CLARA==pairs.NO_BACKOFF_EXACT).all()
        zp=g.groupby(['zone_or_farm','price_id','method'])[MET+['below_floor']].mean()
        z=zp.groupby(['zone_or_farm','method']).mean()
        a=z.xs('NO_BACKOFF_EXACT',level='method').reindex(ZONES)
        b=z.xs('CLARA',level='method').reindex(ZONES)
        rng=np.random.default_rng(20260911);draws=rng.integers(0,10,(5000,10))
        for method,t in [('CLARA',b),('NO_BACKOFF_EXACT',a)]:
            row={'price_id':pid,'subset':subset,'method':method,**t.mean().to_dict()}
            valid=g[g.method.eq(method)]
            row.update(conditions_with_cost=int(valid.ERRF.notna().sum()),conditions_with_TUWR=int(valid.TUWR.notna().sum()),
                       event_count=int(valid.event_count.sum()),below_floor_count=int(valid.below_floor.fillna(False).sum()))
            row['below_floor_pct']=100*row.pop('below_floor')
            row['delta_ERRF_pct']=100*(t.ERRF.mean()/b.ERRF.mean()-1)
            row['delta_TUWR_pp']=100*(t.TUWR.mean()-b.TUWR.mean())
            row['delta_below_floor_pp']=row['below_floor_pct']-100*b.below_floor.mean()
            for field in ['capacity','exceedance']:
                row['delta_'+field+'_contribution_pct']=100*(t[field].mean()-b[field].mean())/b.ERRF.mean()
            if subset!='zero':
                bootrel=100*(np.nanmean(t.ERRF.to_numpy()[draws],axis=1)/np.nanmean(b.ERRF.to_numpy()[draws],axis=1)-1)
                boottuwr=100*np.nanmean((t.TUWR.to_numpy()-b.TUWR.to_numpy())[draws],axis=1)
                for name,values in [('ERRF_change_pct',bootrel),('TUWR_change_pp',boottuwr)]:
                    row[name+'_CI_low'],row[name+'_CI_high']=map(float,np.nanquantile(values,[.025,.975]))
            else:
                for name in ['ERRF_change_pct','TUWR_change_pp']:
                    row[name+'_CI_low']=np.nan;row[name+'_CI_high']=np.nan
            row['zones_with_higher_ERRF']=int((t.ERRF>b.ERRF+1e-12).sum())
            out.append(row)
    return pd.DataFrame(out)

def main():
    assert read(HERE/'evaluation_manifest.json')['status']=='PASS'
    OUT.mkdir(exist_ok=True)
    raw=[];sub=[];tr=[];receipts={}
    for z in ZONES:
        for seed in range(3):
            folder=HERE/'evaluation'/f'{z}__seed{seed}';m=read(folder/'manifest.json')
            assert m['status']=='PASS'
            for name,h in m['outputs'].items():assert sha(folder/name)==h
            raw.append(pd.read_parquet(folder/'cell_metrics.parquet'))
            sub.append(pd.read_parquet(folder/'subset_metrics.parquet'))
            tr.append(pd.read_parquet(folder/'decision_changes.parquet'))
            receipts[folder.name]=sha(folder/'manifest.json')
    raw=pd.concat(raw,ignore_index=True);raw.to_parquet(OUT/'seed_condition_metrics.parquet',index=False)
    assert not raw.duplicated(KEY+['seed']).any()
    d=pool(raw,KEY);d['below_floor']=(d.coverage < d.target_coverage-.02-1e-12).astype(float)
    d.loc[d.coverage.isna(),'below_floor']=np.nan
    d.to_parquet(OUT/'condition_metrics.parquet',index=False)
    overall=d[d.regime.eq('overall')]
    assert overall.groupby('method').size().eq(11000).all()
    assert overall.groupby(['price_id','method']).event_count.sum().eq(23024760).all()
    # Latest v47 source is read under a verified content hash, not selected by folder date.
    v47=read(ROOT/'analysis/ablation_integration_v47_20260911/data_manifest.json')
    refpath=ROOT/'analysis/chapter4_original_structure_v38_20260909/targeted_checks/summary/condition_metrics.parquet'
    assert sha(refpath)==v47['source_hashes'][str(refpath)]
    ref=pd.read_parquet(refpath);ref=ref[ref.method.eq('CLARA')]
    a=d[d.method.eq('CLARA')].set_index(KEY).sort_index();b=ref.set_index(KEY).reindex(a.index)
    np.testing.assert_allclose(a[MET],b[MET],atol=1e-10,rtol=0,equal_nan=True)
    np.testing.assert_array_equal(a[['event_count','reliability_event_count']],b[['event_count','reliability_event_count']])
    summaries=[paired_summary(overall)]
    sub=pool(pd.concat(sub,ignore_index=True),KEY[:5]+['subset','method'])
    sub['below_floor']=(sub.coverage < sub.target_coverage-.02-1e-12).astype(float)
    sub.loc[sub.coverage.isna(),'below_floor']=np.nan
    sub.to_parquet(OUT/'subset_condition_metrics.parquet',index=False)
    for subset,g in sub.groupby('subset',sort=False):
        summaries.append(paired_summary(g,subset))
    summaries=pd.concat(summaries,ignore_index=True);csv(summaries,'paired_summary.csv')
    for field in ['ERRF','capacity','exceedance','TUWR','ARD']:
        val=summaries.query("subset == 'all' and method == 'CLARA' and price_id == 'ALL'")[field].iloc[0]
        assert abs(val-v47['full_baseline'][field])<1e-10,(field,val)
    traces=pd.concat(tr,ignore_index=True)
    tracekeys=KEY[:5]
    traces=traces.groupby(tracekeys,as_index=False)[[c for c in traces if c not in tracekeys+['seed']]].sum()
    traces['changed_frequency']=traces.changed_count/traces.event_count
    traces['changed_supported_frequency']=traces.changed_supported_count/traces.event_count
    for field in ['ERRF','capacity','exceedance']:traces['delta_'+field+'_contribution']=traces['delta_'+field+'_sum']/traces.event_count
    traces.to_parquet(OUT/'condition_decision_changes.parquet',index=False)
    # Attribute costs on the full, condition-balanced population; subset-relative
    # averages alone cannot supply additive contributions to the headline effect.
    fullbase=overall[overall.method.eq('CLARA')].set_index(KEY[:5])
    components=[]
    for subset in ['supported','sparse_nonempty','zero']:
        g=sub[sub.subset.eq(subset)]
        a=g[g.method.eq('NO_BACKOFF_EXACT')].set_index(KEY[:5]).reindex(fullbase.index)
        b=g[g.method.eq('CLARA')].set_index(KEY[:5]).reindex(fullbase.index)
        x=fullbase.reset_index()[KEY[:5]].copy();x['subset']=subset
        for field in ['ERRF','capacity','exceedance']:
            x['delta_'+field+'_contribution']=((a[field]-b[field])*a.event_count/fullbase.event_count).fillna(0).to_numpy()
        components.append(x)
    components=pd.concat(components,ignore_index=True)
    totals=components.groupby(KEY[:5])[['delta_ERRF_contribution','delta_capacity_contribution','delta_exceedance_contribution']].sum().sort_index()
    expected=traces.set_index(KEY[:5])[totals.columns].reindex(totals.index)
    np.testing.assert_allclose(totals,expected,rtol=0,atol=1e-10)
    components.to_parquet(OUT/'condition_cost_attribution.parquet',index=False)
    attribution=[]
    for pid in ['R01','R02',MAIN,'R10','R20','ALL']:
        c=components if pid=='ALL' else components[components.price_id.eq(pid)]
        denominator=fullbase.ERRF.mean() if pid=='ALL' else fullbase.xs(pid,level='price_id').ERRF.mean()
        for subset,g in c.groupby('subset'):
            attribution.append({'price_id':pid,'subset':subset,**{field:float(g[field].mean()) for field in totals.columns},
                                **{field+'_pct':100*float(g[field].mean())/denominator for field in totals.columns}})
    csv(pd.DataFrame(attribution),'cost_attribution.csv')
    changes=traces.groupby('price_id',as_index=False).agg(events=('event_count','sum'),changed=('changed_count','sum'),
            zero=('zero_count','sum'),backoff=('backoff_count','sum'),changed_supported=('changed_supported_count','sum'),
            changed_condition_mean=('changed_frequency','mean'),delta_ERRF_condition_mean=('delta_ERRF_contribution','mean'))
    changes['changed_event_pct']=100*changes.changed/changes.events
    csv(changes,'decision_changes.csv')
    event_sub=pool(sub, ['price_id','subset','method']);csv(event_sub,'subset_event_weighted_metrics.csv')
    regime=d.groupby(['price_id','regime','method'],as_index=False)[MET].mean();csv(regime,'regime_means.csv')
    save(OUT/'analysis_manifest.json',{'status':'PASS','code_sha256':sha(__file__),'freeze_sha256':sha(HERE/'policies_frozen.json'),
         'evaluation_manifest_sha256':sha(HERE/'evaluation_manifest.json'),'evaluation_receipts':receipts,
         'CLARA_latest_v47_verified':True,'same_events':True,'no_new_candidate_training':True,
         'bootstrap':{'unit':'whole held-out zone','replicates':5000,'seed':20260911,'intervals':'pointwise exploratory'},
         'outputs':{p.name:sha(p) for p in OUT.iterdir() if p.suffix in ['.csv','.parquet']}})
    print(summaries.query("subset in ['all','nonempty','sparse_nonempty','zero'] and price_id in ['ALL','R05P6179775281']")
          [['price_id','subset','method','ERRF','delta_ERRF_pct','TUWR','below_floor_pct','event_count']].to_string(index=False))

if __name__=='__main__':main()
