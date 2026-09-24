"""Recompute archived paired ablation summaries; no fit or event replay."""
from pathlib import Path
import argparse, importlib.util, json, sys
sys.dont_write_bytecode = True
import numpy as np
import pandas as pd


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path,
                        default=here.parents[1]/'results/revision_20260924/exact_state_no_backoff')
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('exact_analysis', here/'source_snapshot/analyze_exact.py')
    analysis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analysis)
    d = pd.read_parquet(args.results/'condition_metrics.parquet')
    overall = d[d.regime.eq('overall')]
    assert overall.groupby('method').size().eq(11000).all()
    assert overall.groupby(['price_id', 'method']).event_count.sum().eq(23024760).all()
    summaries = [analysis.paired_summary(overall)]
    subsets = pd.read_parquet(args.results/'subset_condition_metrics.parquet')
    for subset, group in subsets.groupby('subset', sort=False):
        summaries.append(analysis.paired_summary(group, subset))
    got = pd.concat(summaries, ignore_index=True)
    expected = pd.read_csv(args.results/'paired_summary.csv')
    keys = ['price_id', 'subset', 'method']
    got = got.set_index(keys).sort_index()
    expected = expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(got.index, expected.index)
    assert set(got.columns) == set(expected.columns)
    expected = expected[got.columns]
    np.testing.assert_allclose(got.to_numpy(float), expected.to_numpy(float),
                               atol=1e-10, rtol=1e-12, equal_nan=True)
    raw = pd.read_parquet(args.results/'seed_condition_metrics.parquet')
    assert not raw.duplicated(analysis.KEY + ['seed']).any()
    pooled = analysis.pool(raw, analysis.KEY).set_index(analysis.KEY).sort_index()
    current = d.set_index(analysis.KEY).reindex(pooled.index)
    columns = analysis.MET + ['event_count', 'reliability_event_count']
    np.testing.assert_allclose(pooled[columns].to_numpy(float), current[columns].to_numpy(float),
                               atol=1e-10, rtol=1e-12, equal_nan=True)
    print(json.dumps({'status': 'PASS', 'paired_summary_rows':len(got),
                      'seed_metric_rows':len(raw), 'pooled_condition_rows':len(d),
                      'bootstrap_replicates':5000, 'event_replay_performed':False,
                      'policy_rebuild_performed':False}))


if __name__ == '__main__':
    main()
