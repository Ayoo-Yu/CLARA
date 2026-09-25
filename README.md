# CLARA

CLARA selects a complete wind-power prediction interval from six calibration methods using historical cost estimates and coverage requirements. This repository accompanies **CLARA: An Interpretable Contextual Decision Agent for Cost- and Reliability-Aware Calibration of Wind Power Prediction Intervals**, by Zhongze Yu and Zhenqing Liu.

[Repository](https://github.com/Ayoo-Yu/CLARA) · [Version 1.1.0 and data downloads](https://github.com/Ayoo-Yu/CLARA/releases/tag/v1.1.0)

The evaluated candidates are Static, ACI, AgACI, EnbPI-RH, TSC and EEE. The principal comparison includes these six standalone methods, CLARA, CART and LinUCB. The experiments use five exceedance-to-capacity price ratios: 1, 2, 20/3.56, 10 and 20.

Two evaluation protocols are kept distinct. In the GEFCom cross-zone experiment, all selector fitting and tuning use feedback available by 2013-09-05 00:00 UTC from the other nine zones. Evaluation uses only later forecasts in the excluded zone; CLARA's risk estimates and selection mapping remain fixed. In the commercial-farm experiment, an initial local adaptation history is followed by causal updating of all six candidates' cost and coverage estimates. Forecasting and candidate calibration histories follow their specified availability rules in both experiments.

Version 1.1.0 updates all GEFCom policy analyses to this chronological split, including baselines, ablations, oracle comparisons and the pre-period initialization sensitivity. Version 1.0.0 preserves the earlier full-period source analysis. Commercial-farm results are unchanged. See `DATA_ACCESS.md` for software and dataset reuse terms.

## Contents

| Path | Purpose |
|---|---|
| `src/evaluated/` | Evaluated forecasting, calibration, selection and analysis implementations. |
| `configs/` and `configs/fitted/` | Experiment settings and the exact fitted state, price and baseline configurations. |
| `scripts/` | Portable verification and commercial replay entry points. |
| `results/gefcom/` | Complete condition-level main-comparison summaries and a fitted-policy example. |
| `data/commercial/` | Anonymous summaries, data dictionary and metric-recomputation program. |
| `benchmarks/query/` | All measured current-policy query calls, fixtures and a portable correctness check. |
| `analysis/chronological_ablation/` | Current ablation, oracle and price-reselection code and condition summaries. |
| `src/chronological/` | Chronological preparation, fitting, delayed-feedback replay and warm initialization. |
| `results/gefcom/summary/` | Current chronological main-comparison tables; current ablations are in `analysis/chronological_ablation/summary/`. |
| `figures/current/` | Current main Figures 1–9 and Supplementary Figure S1, with an authoritative asset index. |
| `plotting/current/` and `results/current_figure_data/` | Current result-figure programs and their numerical inputs. |

Earlier presentation exports remain in legacy folders for traceability. They
use older numbering; follow the current paths above for the revised manuscript.

The current GEFCom candidate records and fitted evidence are separate v1.1.0 release assets; see `src/chronological/README.md`. Commercial assets remain in v1.0.0 and are unchanged. Extract all six commercial shards into the same directory; each contains a different `commercial_candidate_archive/FarmA|FarmB/seed0|seed1|seed2` subtree. Do not concatenate their files or alter the relative time indices.

Download the v1.1.0 chronological archives described below and keep the commercial archives from v1.0.0. `SHA256SUMS.txt` records archive digests. The ten GEFCom zone archives merge with the evidence archive into one dataset directory. The optional ablation archive contains the complete per-fold records used for its published summaries.

## Environment

Use Python 3.12 in a separate environment:

```bash
python -m venv .venv
# Windows PowerShell
.venv/Scripts/Activate.ps1
# macOS/Linux instead: source .venv/bin/activate
python -m pip install -r requirements-figures.txt
```

Run commands from the repository root. Main result figures use Times New Roman when installed. Font files are not distributed; systems without it use a serif fallback, so text layout can differ. Windows users should extract to a short directory such as `C:/CLARA` because the evaluated source layout retains nested experiment directories.

## Prepare GEFCom inputs and generate candidate intervals

Obtain the official data archive from the [competition authors' data page](https://blog.drhongtao.com/2017/03/gefcom2014-load-forecasting-data.html), and extract its `GEFCom2014-W_V2.zip` wind-track archive. Raw competition files are obtained separately and are not redistributed in this repository.

```bash
python scripts/prepare_gefcom.py --archive /path/to/GEFCom2014-W_V2.zip --output data/gefcom
python scripts/run_forecast_candidates.py --data data/gefcom/gefcom2014_zone1_processed.csv --zone zone1 --predictor Ridge --horizon 1 --seed 0 --coverage 0.9 --output outputs/zone1_Ridge_h1_seed0
```

The converter reconstructs all ten zones and records the retained rows and preprocessing in `conversion_report.json`. It preserves the study's original row set and split boundaries, including one early omitted timestamp and six terminal hours without target data. The lag feature is calculated after chronological sorting. Original training and prediction input matrices were checked for all ten zones and five horizons; their largest numerical difference was 8.05e-16. A complete raw-data-to-Ridge-to-six-candidate run was also checked against the saved experiment.

The archived preprocessing linearly interpolates 115 missing targets in zones 1–3 during calibration and testing. Some interpolated values depend on later observations; their effect on the reported results has not been quantified. The converter reproduces that preprocessing, and `tests/validation/gefcom_raw_conversion.json` records its availability limitation. These numerical-reproduction checks do not certify complete causal availability of the GEFCom inputs.

`run_forecast_candidates.py` supports Ridge, GBR, MLP and QRLSTM; horizons of 1, 3, 6, 12 and 24 h; and seeds 0, 1 and 2. Omit `--coverage` to construct the complete eleven-coverage grid for that stream. The command trains the original predictor and constructs all six candidates without fitting a selector.

## Reproduce the numerical results

```bash
python scripts/verify_gefcom.py
```

This checks all 11,000 price-specific forecasting conditions and 33,000 state choices in the bundled chronological policy example. Current CLARA means are ERRF **2.8531**, economic rank **3.27**, relative excess cost **2.41%**, and TUWR **10.23%**. Seeds are pooled within conditions; conditions and zones receive equal weight.

After extracting the v1.1.0 GEFCom evidence and zone1 candidate archives:

```bash
python scripts/verify_chronological.py --data-root assets/gefcom_chronological_v1_1_0 --refit-source
```

This independently reconstructs the fitted selector and checks its target actions and metrics against recorded outputs. Full fitting, candidate regeneration and delayed-feedback replay entry points are documented in `src/chronological/README.md`. The checks do not retrain all base forecasters.

For commercial data, extract the six candidate archives, then run:

```bash
python data/commercial/recompute_metrics.py --archive commercial_candidate_archive --output outputs/commercial_metrics
```

The complete re-evaluation has mean ERRF **0.6522** and TUWR **10.47%** for joint-update CLARA. All nine principal methods and three update controls are retained. Regime diagnostics are calculated on complete chronological windows before assigning the window endpoints to ordinary or predicted-ramp conditions. The input schema and averaging rules are described in `data/commercial/DATA_DICTIONARY.md`.

Replay the actual local update and selection algorithm for one farm/seed/price:

```bash
python scripts/replay_commercial.py --archive commercial_candidate_archive/FarmA/seed0 --price rho_reference --output outputs/replay_FarmA_seed0_reference
```

The replay regenerates candidate losses and historical reliability from the anonymous endpoints and observations, checks initial fitted evidence, executes the evaluated update functions and compares every selected action against the archived decision paths. Other prices are `rho_1`, `rho_2`, `rho_10` and `rho_20`. Add `--max-test-days 2` for a short test prefix while retaining the complete adaptation history. Replaying saved candidate outputs does not retrain the commercial forecasting models; the raw training measurements are excluded from release.

## Current fitted settings, timings and ablation

`configs/fitted/` records the 990,000 GEFCom and 198,000 commercial state-price
configurations, state definitions, price-dependent thresholds and selected
baseline settings. These are the fitted settings used in the reported results.

```bash
python benchmarks/query/benchmark_query.py --repository-root .
python analysis/chronological_ablation/verify_results.py
```

The first command checks 6,600 state mappings, 220 choices and 440 interval
endpoints without collecting new timings. The archived 1,920 calls support
the current Table S6; the median single-query time is 6.14 ms for lookup and
interval retrieval, or 9.01 ms including width classification. Hardware,
background load, measurement scope and a command for new measurements are
documented in `benchmarks/query/README.md`.

The second command checks the current chronological ablation summaries. The no-backoff variant uses nonempty exact-state histories and chooses Static only when that history is empty. All current GEFCom uncertainty intervals use 10,000 zone-bootstrap draws. Historical v1.0.0 ablation files are retained under their explicit legacy path.

## Reproduce the current result figures

```bash
python plotting/current/figure_04.py
python plotting/current/figure_05.py
python plotting/current/figure_06.py
python plotting/current/figure_07.py
python plotting/current/figure_08.py
python plotting/current/figure_09.py
python plotting/current/figure_S1.py
```

These programs write to `outputs/current/`. All six updated plots (Figures 4–8 and S1) were executed with the included inputs and matched the approved PNGs byte for byte. Figure 9 is unchanged. Figures 1–3 are editable mechanism artwork. See `figures/current/figure_index.csv` and `plotting/current/reproduction_qa.json` for current paths and checksums.

## Scope and interpretation

ERRF is a normalized interval-dependent cost measure, not a farm's actual settlement invoice. TUWR measures the proportion of rolling windows with coverage below the specified descriptive tolerance. Lower ERRF and lower TUWR concern different aspects of the comparison; CLARA does not minimize every metric at every price.

The executable checks separate state-choice reproduction, numerical aggregation, candidate-level metric reproduction and sequential decision replay. The original experiment modules are supplied for inspection and further work. Full fresh-environment retraining of every base predictor and every original analysis was not repeated as part of preparing this release.

Data access, commercial-output permissions and third-party rights are described in `DATA_ACCESS.md`. The authors' software uses the MIT license; this does not license the commercial dataset. Citation metadata is in `CITATION.cff`.
