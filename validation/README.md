# Reproduction checks

The JSON receipts describe the scope of each executed check. They are not evidence that every possible training configuration was rerun.

- GEFCom: all 150 fitted policies, comprising 990,000 state decisions, were reproduced with the evaluated selector. The complete 11,000-condition nine-method summaries reproduce the reported cost and ranking values.
- Base forecasts and candidates: the complete zone 1 / Ridge / 1 h / seed 0 stream was regenerated from the evaluated processed input. Its base quantiles and all six candidate endpoint sequences at 90% coverage agree with the saved experiment to floating-point precision. This check does not establish equivalence for every forecaster or input transformation.
- Sequential local selection: the complete FarmA and FarmB seed-0 test sequences at the reference price were replayed from anonymous inputs. All four CLARA update controls have exactly the archived selected actions. Initial adaptation evidence agrees to floating-point precision.
- Commercial metrics: `data/commercial/VALIDATION.json` covers all six farm/seed shards and all 30 farm/seed/price units, including nine principal methods and three controls. These summaries were recalculated from released endpoints, observations and saved choices, with a maximum metric difference of approximately 2.14e-13.
- Main result figures: figures 4–8 were regenerated and compared pixel by pixel with the final source PNGs using the checked font environment. Figure 5 was also rerun with the pinned NumPy 1.26.4 / Matplotlib 3.10.6 environment.

The commercial privacy check is recorded in `data/commercial/PRIVACY_CHECK.json`. Raw commercial predictor training and a complete fresh retraining of all forecasters are outside these release-validation checks.
