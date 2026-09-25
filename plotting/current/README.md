# Current figure reproduction

From the repository root (or use an absolute script path):

```sh
python plotting/current/figure_04.py
python plotting/current/figure_05.py
python plotting/current/figure_06.py
python plotting/current/figure_07.py
python plotting/current/figure_08.py
python plotting/current/figure_09.py
python plotting/current/figure_S1.py
```

Scripts read repository-relative saved data and export to `outputs/current/figureNN/`. Dependencies: numpy, pandas, pyarrow, matplotlib, scipy, Pillow, and PyMuPDF. Times New Roman must be locally installed for the approved typography; no font binaries are included.

Figures 1–3 are editable mechanism artwork in `figures/current/`, not experimental-data plots. Current file numbering and legacy mappings are listed in `figures/current/figure_index.csv`. The legacy `plotting/figure_0N.py` entries refer to the older release.

`reproduction_qa.json` records comparison with the approved raster assets. It records the current chronological plot reruns; authoritative approved assets are preserved separately.
