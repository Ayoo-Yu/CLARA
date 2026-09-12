# Commercial candidate archive

This archive contains the complete candidate-interval records used in the revised local-adaptation evaluation: two anonymized farms, three seeds, four forecasters, four lead times, eleven target coverages and five prices. It includes the adaptation records, test observations, all six candidate intervals, the initial selection parameters, and the evaluated decision paths. It supports replay of selection and recomputation of evaluation metrics. Raw SCADA/NWP inputs and the forecasting-model training records are outside this archive.

`CLARA` denotes joint sequential updating of candidate cost and coverage evidence. `CLARA_frozen`, `CLARA_cost_update` and `CLARA_coverage_update` are the three controls. The nine principal methods consist of CLARA, CART, LinUCB and the six standalone candidate methods.

## Files and joins

Each independent release shard extracts below `commercial_candidate_archive/FarmA/seed0`, with the corresponding farm and seed for the other five shards. Every shard contains all five prices and all forecasting conditions for that farm/seed.

| File | Content |
|---|---|
| `manifest.json` | Schema, action and method orders, record counts, and checksums of all data files in the shard. |
| `price_configs.json` | Five price ratios, cost weights and the TSC configuration used at each price. |
| `state_map.parquet` | The 6,600 complete state definitions indexed by `state_id`. |
| `streams/<predictor>_h<steps>/base_adaptation.parquet` | Adaptation observations, median forecasts, context and the five price-invariant candidate intervals. |
| `streams/<predictor>_h<steps>/base_test.parquet` | The corresponding complete test records. |
| `streams/<predictor>_h<steps>/TSC<id>_<split>.parquet` | TSC endpoints for one selected configuration; join to the matching base file on `row_id`. |
| `policies/<price_id>.npz` | Initial parameters and candidate evidence, indexed by `state_id`. |
| `policies/<price_id>_cart_statistics.parquet` | The evaluated CART fitting statistics, with the farm identifier removed. |
| `choices/<price_id>.npz` | Six decision-path columns indexed by the test `replay_order`. |

TSC configurations are stored once per distinct configuration rather than repeated at every price. `price_configs.json` supplies the exact correspondence. EEE remains the separate endpoint mean of Static, ACI, AgACI and EnbPI-RH. All power values and endpoints remain float64; no rounding, clipping, rescaling or outcome-dependent row removal is applied during anonymization.

## Power and cost units

`target`, `base_center`, candidate endpoints and interval width are in the original stored/model power-base p.u. These are the numerical values used by the evaluated models; they must not be divided again by a reported farm capacity. The stored model power base and a farm's reported nameplate capacity need not be identical. Neither physical capacity nor its identifying mapping is released.

ERRF uses the evaluated normalized 1 MW costing basis and a 0.25 h sampling interval. For lower/upper bounds L/U, median scheduling reference b and observation y, the symmetric-price calculation is:

```text
capacity = 0.25 * pi * (max(U - b, 0) + max(b - L, 0))
exceedance = 0.25 * kappa * (max(y - U, 0) + max(L - y, 0))
ERRF = capacity + exceedance
```

Both capacity weights are 3.56. Both exceedance weights are 3.56 times the price ratio, with the reference scenario represented by the exact weight 20. Ratios are 1, 2, 20/3.56, 10 and 20. The weights are comparison coefficients; the resulting normalized cost is not a disclosed farm invoice.

## Base-record fields

| Field | Type and meaning |
|---|---|
| `row_id` | Integer key unique within a stream and split; preserves the source record order without releasing source identifiers. |
| `replay_order` | Zero-based global order within a farm/seed/split, preserving the chronological order and within-batch tie order used by the evaluated replay. |
| `state_id` | Integer row index in `state_map.parquet`. |
| `issue_tick` | Forecast time as an integer number of 15-minute ticks from an undisclosed, farm-specific origin. |
| `label_tick` | Forecast target time on the same relative scale. |
| `available_tick` | Time when the outcome becomes available; equals `label_tick + 1` in these archives. |
| `utc_day_group` | Consecutive, zero-based grouping of the original UTC forecast dates. |
| `utc_week_group` | Consecutive, zero-based grouping of the original UTC 168 h blocks. |
| `target` | Observed power at the target time, in stored/model power-base p.u. |
| `base_center` | Uncalibrated median forecast and common scheduling reference, in the same p.u. |
| `target_coverage` | One of 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95 or 0.99. |
| `horizon_group` | The original lead-time category used by the state. |
| `ramp_state` | Original forecast-time power-change category: `ordinary` or `ramp`. |
| `rolling_state` | Original recent-coverage category: `cold_start`, `undercoverage_pressure`, `overcoverage_pressure`, `volatile` or `stable`. |
| `raw_width_state` | Original raw-interval category: `narrow`, `medium` or `wide`. |
| `<action>__lower`, `<action>__upper` | Complete endpoints for Static, ACI, AgACI, EnbPI-RH or EEE. |

Stream filenames contain lead times in 15-minute steps: 4, 24, 48 and 96 correspond to 1, 6, 12 and 24 h. Forecasters are Ridge, GBR, MLP and QRLSTM. Actual calendar dates, original event identifiers and the mapping from farm labels to sites are not included.

The two farms have independent time origins. Relative ticks cannot be used to align the farms to an actual shared calendar. UTC day/week memberships are retained for equivalent statistical grouping, but their original dates and absolute group numbers are removed. Use the provided group fields; flooring relative ticks alone does not in general recover the original UTC boundaries. Chronological replay should sort all streams together by `replay_order`, rather than by the stored Parquet row order. Parquet rows are sorted to improve compression.

## Initial policy arrays and candidate order

The candidate axis is always:

```text
0 Static; 1 ACI; 2 AgACI; 3 EnbPI-RH; 4 TSC; 5 EEE
```

| Array | Meaning |
|---|---|
| `n_min` | Adaptation-selected minimum historical count for every state; fixed during test replay. |
| `nu` | Adaptation-selected shrinkage parameter for every state; fixed during test replay. |
| `initial_support_level` | Historical support level used at the start of testing. |
| `frozen_evidence` | Float64 array with dimensions state × candidate × evidence field. |
| `cart_choice` | Evaluated fixed CART decision for each complete state, encoded by candidate index. |
| `coverage_shortfall` | Prescribed empirical coverage-shortfall tolerance for this price. |
| `tuwr_limit` | Prescribed historical TUWR limit for this price. |

The evidence-field axis is `risk_mean`, `risk_parent`, `risk_shrunken`, `risk_se`, `risk_score`, `coverage`, `tuwr`, `ard`. Missing historical diagnostics remain NaN where the original history is incomplete. These are initial historical estimates, not future guarantees.

The hierarchy relaxes recent coverage, raw width, power-change category, forecaster, lead-time group and finally exact target coverage. Sequential local adaptation retains the initial `n_min`, `nu` and coverage thresholds while refreshing costs and coverage from eligible outcomes. Eligibility requires `label_tick < issue_tick` and `available_tick <= issue_tick`. Every decision in a batch uses the same eligible history.

Historical candidate TUWR and ARD can be regenerated from the complete adaptation/test streams and original recent-history flags, using the evaluation implementation. They are not duplicated in the release files. The historical state diagnostics use a 168 h history of eligible observations with 24 h internal coverage windows. Test-summary diagnostics instead use full 168 h rolling windows.

## Saved choices

`choices/<price_id>.npz` contains an integer `selected` array. Its row index is the test `replay_order`, and its six columns are:

```text
CLARA_frozen, CLARA, CLARA_cost_update, CLARA_coverage_update, CART, LinUCB
```

The standalone candidate methods always use their own column of endpoints. The first four choice columns share all candidates and differ only in which historical evidence is refreshed. The reference CLARA result uses the joint-update `CLARA` column. CART uses a fixed tree; LinUCB retains its evaluated delayed, selected-action feedback rule. Initial CART sufficient statistics are supplied so its evaluated fixed mapping can be reconstructed without fitting on test observations.

## Recomputed summaries

`recompute_metrics.py` recomputes all nine methods and the three CLARA controls from the extracted endpoints, observations and saved choices. It also reports `overall`, `ordinary` and `ramp` conditions. Rolling metrics are computed on the complete chronological sequence before filtering window end times by regime.

TUWR and TOWR are fractions of complete 672-record test windows below/above the band `c +/- 1.96*sqrt(c*(1-c)/672)`. ARD, `under_gap` and `over_gap` are coverage gaps in fraction units. Multiply window rates and gaps by 100 for percentage or percentage-point presentation as appropriate. The tolerance band is a descriptive diagnostic, not a coverage guarantee for dependent observations.

`seed_cells.parquet` gives one row per farm, seed, price, forecaster, lead, target coverage, regime and method. `event_count` is the denominator for ERRF, capacity, exceedance, empirical coverage and width. `reliability_count` is the denominator for rolling diagnostics. `condition_metrics.parquet` first pools seeds using those denominators; `means.parquet` and `means.csv` then give equal weight to forecasting conditions, prices and farms. The `ALL` price summary averages the five prices; `Pooled` averages the two farms. `VALIDATION.json` records the agreement with the evaluated results.

The small summary files in this directory can be read without downloading the candidate archive. They are regenerated from the complete release data rather than from a selected subset.
