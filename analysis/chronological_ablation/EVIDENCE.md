# Chronological ablation evidence

All values use the sealed chronological protocol and the same post-T0 target records. GEFCom intervals use 10,000 paired zone bootstrap draws. The main CLARA metrics reproduce the sealed main table within 1.44e-13.

## Reference-price Table 5

| Method | ERRF | Change (%) [95% interval] | Below-floor conditions (%) | TUWR (%) |
|---|---:|---:|---:|---:|
| CLARA | 2.5655 | +0.00 [+0.00, +0.00] | 1.68 | 7.44 |
| GLOBAL_SCORE_SAME_SCREEN | 2.6032 | +1.47 [+1.40, +1.53] | 0.32 | 5.93 |
| WIDTH_SAME_SCREEN | 2.6511 | +3.33 [+3.27, +3.40] | 21.59 | 20.64 |
| NO_G | 2.5535 | -0.47 [-0.55, -0.40] | 3.32 | 9.13 |
| NO_RWG | 2.5406 | -0.97 [-1.10, -0.85] | 9.09 | 12.61 |
| NO_SCREEN | 2.5512 | -0.56 [-0.65, -0.47] | 10.73 | 11.87 |
| NO_BACKOFF_EXACT | 2.5703 | +0.19 [+0.15, +0.22] | 1.36 | 7.12 |

## Screening and no-back-off explanation

At the reference price, the screen excludes 21.36% of candidate options; all six fail for 4.03% of forecasts (equal-condition averages). Exact history is empty for 1.67%; back-off is used for 14.00%.
On the common nonempty-history subset, removing back-off changes width from 0.453135 to 0.454316, capacity cost from 1.616098 to 1.620299, and exceedance cost from 0.953068 to 0.952280. ERRF changes by +0.1329% [0.1051, 0.1649]. This supports the capacity-versus-exceedance explanation; it does not imply that back-off improves every coverage metric.

## Price reselection

Fixed reference-price choices cost 2.922303; reselection costs 2.853116, a 2.3676% reduction [2.2755, 2.4599]. TUWR changes from 7.4426% to 10.2306%. See price_reselection.csv for the complete price grid.

## Oracle bounds (five-price means)

| Oracle | ERRF | CLARA gap (%) [95% interval] |
|---|---:|---:|
| FEASIBLE_STATE_ORACLE | 2.817666 | 1.258 [1.167, 1.357] |
| STATE_ORACLE | 2.781326 | 2.581 [2.336, 2.837] |
| OUTCOME_ORACLE | 2.492376 | 14.474 [13.395, 15.677] |

State-oracle groups contain the same zone, seed, forecaster, horizon, target coverage and complete state. Feasibility is defined by the source-fitted CLARA eligible set, including minimum-violation recovery. These are hindsight bounds and must not be ranked as deployable methods.

## Data files

- Table5_reference_price.csv: main table values and paired intervals.
- TableS8_oracle_gaps.csv: every price and the five-price average.
- condition_metrics.parquet: seed-pooled, condition-level data for figures (filter regime=overall for Fig. 5).
- screening_summary.csv, no_backoff_subset_summary.csv, decision_change_summary.csv: mechanism evidence.
- ANALYSIS_COMPLETE.json: data hashes, inference settings and independent sealed-main checks.

The old numerical result archives were not used in fitting or evaluation. No CHRONO data/model or Word files were changed.
