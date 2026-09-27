# Current manuscript figures

This directory is the authoritative figure set for the 2026-09-27 manuscript: main Figures 1–9 and Supplementary Figure S1. `figure_index.csv` records the current files, SHA-256 hashes and earlier release numbering. The approved exports were matched to active images in the manuscript and supplementary documents; deleted revision images were excluded.

Version 1.1.1 updates presentation and notation, and supplies the two Figure 7 cases currently discussed in the manuscript. It retains the v1.1.0 chronological evaluation and all aggregate experimental results. Figure 9 is unchanged. See `release_v1_1_1_changes.json` for the individual changes.

Figures 1–3 are editable mechanism artwork. SVG preserves editable text; EMF companions preserve the approved appearance in Word. Figures 4–8 and S1 have portable plotting scripts and numerical inputs. Figure 4 retains its raster density layers inside the SVG/PDF/EMF; it is a mixed vector/raster figure.

Run `python plotting/current/figure_04.py` through `figure_09.py`, or `python plotting/current/figure_S1.py`, from any working directory. The scripts read `results/current_figure_data/` and write to `outputs/current/`; they never overwrite the authoritative assets. Install `requirements-figures.txt`. Times New Roman must be installed for the approved typography; proprietary font files are not distributed. Font substitution can change text layout.

Figure 7 uses the 2013-09-12 06:00 UTC cost-trade-off case and the 2013-09-06 06:00 UTC coverage-screening case. Both are 6 h GBR forecasts at 90% target coverage in held-out GEFCom zone 1, at the reference price ratio 20/3.56. Candidate-level inputs contain interval bounds, the subsequent observation, historical cost/coverage/TUWR, and CLARA and LinUCB choices. `results/current_figure_data/figure07/case_verification.json` records checks of these data. The optional `plotting/current/export_figure_07_emf.py` creates a Word-compatible EMF after Figure 7 is drawn; it requires Inkscape and lxml.

Figure 9 uses aggregate anonymous commercial-farm metrics only. No new commercial observations, timestamps or site identities are released. Figure 7 and S1 use public GEFCom evidence. Plotting reproduces saved evidence and does not retrain forecasting models or recompute experimental results.

All six changed numerical plotting scripts were executed. Their PNGs matched the approved assets byte for byte; see `plotting/current/reproduction_qa.json`. Earlier releases and their data archives are preserved.
