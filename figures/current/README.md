# Current manuscript figures

This directory is the authoritative figure set for the 2026-09-25 chronological revision: main Figures 1–9 and the sole Supplementary Figure S1. `figure_index.csv` maps the earlier release numbering to the current numbering and records SHA-256 hashes. Figures in legacy folders use older numbering.

Figures 4–8 and S1 use the v1.1.0 chronological evaluation; Figures 1–3 and 9 retain their prior approved artwork. SVG and PDF companions retain the approved style. Figures 1–3 are editable mechanism artwork, supplied as authoritative SVG/PNG rather than a claim of algorithmic drawing reproducibility.

Run `python plotting/current/figure_04.py` through `figure_09.py`, or `python plotting/current/figure_S1.py`, from any working directory. These scripts read `results/current_figure_data/` and write to `outputs/current/`; they never overwrite authoritative assets. Install the repository figure requirements; PyMuPDF is also required by Figures 7 and 8. The approved font is Times New Roman; no proprietary font files are distributed. A font substitution changes typography.

Figure 9 includes aggregate anonymous Farm A/Farm B metrics only; no commercial observations, timestamps or site identities are included. Figure 7 and S1 use the public GEFCom zone evidence. Plot scripts reproduce saved evidence; they do not rerun experiments.

Former data-split and other supplementary figures are not part of this current figure set. Reproduction validates plotting, data checks and exports, not a new statistical analysis.

All six updated numerical plotting scripts were executed in the release preparation environment. Their pixel comparisons with the approved assets are recorded in the plotting verification receipt. The approved figure files remain the publication assets. See `plotting/current/reproduction_qa.json`.
