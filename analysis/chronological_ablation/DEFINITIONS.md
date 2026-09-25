# Chronological GEFCom ablation and oracle definitions

Inputs are the sealed source sufficient statistics and matched target streams in `20260925-124953_renewable-experiment-log_gefcom-chronological-frozen`. Every source outcome satisfies `issue < T0`, `label < T0`, and `available <= T0`, where T0 is 2013-09-05 00:00 UTC. The held-out zone is excluded from fitted evidence. This extension does not change the candidate intervals, original forecasters, source width cutoffs, or the main CLARA result.

| Method | Definition |
|---|---|
| CLARA | Refit the complete model on the sealed source statistics; require exact agreement of parameters, evidence and choices with the sealed main model. |
| GLOBAL_SCORE_SAME_SCREEN | Rank the main model's eligible candidates using source-global mean loss plus its zone standard error; retain the main historical screen and minimum-violation set. |
| WIDTH_SAME_SCREEN | Rank the same eligible candidates by current interval width; retain the original tie-breaking rules. |
| NO_G | Remove recent coverage from all historical grouping levels, deduplicate identical levels, and refit adaptive support and shrinkage from source evidence. Cold-start availability remains a diagnostic availability condition. |
| NO_RWG | Remove ramp, width and recent coverage from grouping; refit the resulting hierarchy and source-selected parameters. |
| NO_SCREEN | Rank all candidates by the complete model's uncertainty-adjusted conditional cost score. |
| NO_BACKOFF_EXACT | Use nonempty exact-state histories even below count or zone-support requirements. Retain the complete source-selected shrinkage strength, recursive parent shrinkage, standard-error adjustment and coverage screen. Use Static only as the empty-history default. |
| FIX_REFERENCE_PRICE | Retain actions selected at the reference price for every price scenario and revalue their identical intervals. |
| WIDTH_ONLY | Select the narrowest currently available interval, without a coverage screen. |
| STATE_ORACLE | After outcomes are known, choose the minimum mean-cost candidate within each zone–seed–forecaster–horizon–coverage–complete-state group. |
| FEASIBLE_STATE_ORACLE | Apply the same hindsight comparison within the complete model's historical eligible set, including its minimum-violation set. |
| OUTCOME_ORACLE | Choose the minimum realized-loss candidate separately for each forecast. |

All three oracles are hindsight diagnostics and are excluded from deployable-method rankings. Their fitting input includes target outcomes by definition; all other choices are fixed before target losses are evaluated.

Seed sums/counts are pooled within each forecasting condition. Event quantities use forecast counts; rolling diagnostics use complete 168-hour windows. Ordinary/ramp windows are assigned by their endpoints without concatenating nonconsecutive forecasts. Conditions, zones and price scenarios receive equal weight at their respective aggregation levels. The coverage-floor statistic uses target coverage minus 0.02 after seed pooling. All new GEFCom intervals use 10,000 paired whole-zone bootstrap samples; the four prespecified primary Table 5 tests use exact zone sign assignments and Holm correction.

`units/` contains source-fitted policies, fit-parameter records, per-seed condition metrics, screening counts, paired differences and explicit oracle groups. `summary/` supplies the pooled figure/table data and inferential results. `ANALYSIS_COMPLETE.json` records hashes and comparison with the sealed main result.
