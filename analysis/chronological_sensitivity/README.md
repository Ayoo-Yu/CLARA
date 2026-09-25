# Chronological source-only sensitivity

The numerical methods and grids match the v1.1.0 chronological experiment. Input assets are selected by `CLARA_CHRONO_ROOT` (see `src/chronological/README.md`); outputs go to `outputs/chronological_sensitivity/` or `CLARA_SENSITIVITY_OUTPUT`. For a new output directory:

```bash
python analysis/chronological_sensitivity/estimation_screening.py --pilot --workers 1
python analysis/chronological_sensitivity/ramp_validation.py --pilot --workers 1
```

Omitting `--pilot` evaluates all 30 zone/seed folds. The estimation/screening analysis uses the released source sufficient statistics and target candidates. **The ramp-threshold analysis additionally needs the original base-prediction files including `feature_target_lag1`, located with `CLARA_BASE_ROOT`. That predictor-input feature is not contained in the chronological candidate archives.** Regenerate it from official GEFCom inputs using the released forecasting preparation code; candidate-only assets alone cannot rerun this analysis. Published source-validation summaries and the plot inputs are included.

These scripts refit each prespecified setting from source data and keep target records out of the ramp-threshold validation. Do not reuse completed directories from the original workstation without adapting their explicit code-identity receipts; use a new output directory. The portable adapter was checked by source-baseline refitting and unchanged choices, not by rerunning every sensitivity setting during release preparation.
