# Chronological GEFCom evaluation (v1.1.0)

The base forecasting split remains 60% training, 20% initial calibration and 20% chronological candidate release. T0 is 2013-09-05 00:00 UTC: the latest stream start rounded up to a UTC day, plus 28 days. Source fitting uses issue<T0, label<T0 and feedback-availability<=T0. Held-out-zone forecasts with issue>=T0 form the evaluation period. Width cutoffs, TSC choice, n_min, nu and baseline hyperparameters are fitted only from eligible source records. The held-out zone is excluded from this fitting.

## Download and verify

From release v1.1.0, extract `gefcom_chronological_evidence_v1_1_0.zip` and all ten `gefcom_chronological_zoneN_v1_1_0.zip` archives into `assets/`. They merge into `assets/gefcom_chronological_v1_1_0/`. Zone1 plus the evidence archive suffices for the default replay check. The archives contain prediction/candidate outputs, observed targets needed for scoring, fitted sufficient statistics and decisions; they do not include commercial raw measurements or the original wind-track input archive. Check each downloaded archive against `SHA256SUMS.txt`.

```bash
python scripts/verify_gefcom.py
python scripts/verify_chronological.py --refit-source
python src/chronological/clara_verify.py
python src/chronological/linucb_warmstart.py --smoke
```

Use `--data-root` on the verification command, or set `CLARA_CHRONO_ROOT`, to locate data elsewhere. Paths are portable; the evaluated mathematical helpers remain in `src/evaluated/`.

## Refit and replay

For a new run, use a separate working copy of the extracted assets. Keep `datasets/`, `tsc_zone_scores.parquet`, readiness/parameter JSON files and the cutoff. The fitted `clara/`, `cart_results/`, `linucb/` and `linucb_warm/` trees are reference outputs: move them aside in the working copy before refitting, so checkpoint identity guards do not mix source versions. Set `CLARA_CHRONO_ROOT` to that working copy.

```bash
python src/chronological/run_clara.py --outer zone1 --seed 0 --workers 1
python src/chronological/baseline_batch.py --mode cart-outer --zones zone1 --workers 1
python src/chronological/baseline_batch.py --mode linucb-outer --zones zone1 --workers 1 --threads 2
python src/chronological/linucb_warmstart.py --zone zone1 --seed 0 --threads 2
```

The primary LinUCB baseline starts from its prior at T0. The additional initialization sensitivity replays the held-out site's prefix and preserves observations still pending at T0; those affect later decisions only when their feedback becomes available. Tuning still excludes the held-out zone. This initialization is retrospective model initialization, not a claim that those fitted hyperparameters were historically deployed before T0.

`prepare_chrono.py` and `prepare_pending.py` regenerate the 19 conformal configurations plus EEE from original 60/20/20 base predictions. Set `CLARA_BASE_ROOT` to the original-style base-prediction tree and `CLARA_CANDIDATE_ROOT` to the original four-candidate audit tree. Generate these using the released base-forecast/candidate code and separately obtained official GEFCom inputs. Preparation intentionally requires those numerical-reproduction fixtures; downloaded chronological candidate assets bypass this expensive step. The packaged release was checked by fitted-policy reconstruction and event replay, not by retraining all 600 forecasting streams.

## Analysis

Current main results are in `results/gefcom/`, warm-initialization summaries in its `warm_initialization/` subfolder, and ablations/oracles in `analysis/chronological_ablation/summary/`. Recomputing ablations requires all chronological assets. Extract the optional `gefcom_chronological_ablation_v1_1_0.zip` into `outputs/` to use its recorded per-fold summaries, then run `analysis/chronological_ablation/summarize_ablation.py`. Set `CLARA_ABLATION_OUTPUT` to choose a different output path.

The bootstrap unit is a zone, with 10,000 resamples. Primary comparisons use seed 20260925; ablations and oracle summaries use seed 20260911. State and outcome oracles use target hindsight and are diagnostic lower bounds, never deployed methods. Source preparation and the target cutoff are identical across the main results and all policy analyses.
