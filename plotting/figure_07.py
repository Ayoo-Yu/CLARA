"""Reproduce manuscript Figure 7 from its numerical source data."""
from _fonts import preferred_font
FONT_FAMILY = preferred_font()
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FDATA = REPO / 'results' / 'figure_data'
DATA = FDATA / 'figure07'
OUT = REPO / 'outputs' / 'figure07'
OUT.mkdir(parents=True, exist_ok=True)
PARAM = pd.read_csv(DATA / 'parameter_sensitivity.csv').query("price_id == 'ALL'")
TOL = pd.read_csv(DATA / 'coverage_tolerances.csv').sort_values('multiplier')
RAMP = pd.read_csv(DATA / 'threshold_sensitivity.csv').sort_values('tau')
PRICE = pd.read_csv(DATA / 'price_adaptation_summary.csv')
TEAL, BLUE, ORANGE = '#8AB7BE', '#CAD9DE', '#C68865'
GRID = '#E2E6E8'
font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': [FONT_FAMILY], 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'text.color': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': .8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': False,
    'ytick.right': False, 'legend.frameon': False, 'legend.fontsize': 9,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
WIDTH_MM, HEIGHT_MM = 183, 184
fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)

def panel(bottom, letter, title, ymax=3.65):
    ax = fig.add_axes([.100, bottom, .775, .208])
    ax.set_ylim(0, ymax)
    ax.set_ylabel('Mean ERRF', labelpad=5)
    ax.set_axisbelow(True)
    ax.grid(axis='y', color=GRID, lw=.55)
    ax.tick_params(length=3.2, width=.7, pad=3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.text(-.080, 1.115, f'({letter})', transform=ax.transAxes,
            fontweight='bold', fontsize=11, va='baseline')
    ax.text(0, 1.115, title, transform=ax.transAxes, fontsize=11, va='baseline')
    return ax

def right_metric(ax, lim, ticks):
    right = ax.twinx()
    right.set_ylim(*lim)
    right.set_yticks(ticks)
    right.set_ylabel('TUWR (%)', labelpad=6)
    right.tick_params(length=3.2, width=.7, pad=3)
    right.spines['top'].set_visible(False)
    right.spines['left'].set_visible(False)
    right.spines['bottom'].set_visible(False)
    right.patch.set_visible(False)
    return right

cost_handle = Patch(facecolor=TEAL, edgecolor='black', linewidth=.65)
risk_handle = Line2D([], [], marker='o', color=ORANGE, markeredgecolor='black',
                     markeredgewidth=.55, lw=1.1, markersize=4.6)

# Reference A22 I09: compact full-width bars and outlined points in aligned rows.
# Separate groups are never connected across their unrelated parameter axes.
a = panel(.703, 'a', 'Estimation parameters and coverage tolerances')
ar = right_metric(a, (0, 20), [0, 5, 10, 15, 20])
groups = [
    ('Minimum count', PARAM.query("parameter == 'NMIN'").sort_values('value'), np.arange(4)),
    ('Shrinkage strength', PARAM.query("parameter == 'NU'").sort_values('value'), np.arange(5) + 4.8),
    ('Tolerance multiplier', TOL, np.arange(3) + 10.6),
]
positions, labels = [], []
for name, df, xx in groups:
    a.bar(xx, df.ERRF, width=.64, color=TEAL, edgecolor='black', lw=.7, zorder=2)
    ar.plot(xx, df.TUWR_pct, '-o', color=ORANGE, lw=1.1, markersize=4.7,
            markeredgecolor='black', markeredgewidth=.5, zorder=4)
    positions.extend(xx)
    labels.extend([f'{v:g}' for v in df['multiplier' if name.startswith('Tolerance') else 'value']])
    a.plot([xx[0] - .42, xx[-1] + .42], [-.175, -.175], color='black', lw=.65,
           transform=a.get_xaxis_transform(), clip_on=False)
    a.text(xx.mean(), -.205, name, transform=a.get_xaxis_transform(),
           ha='center', va='top', fontsize=9.5)
    if name == 'Tolerance multiplier':
        for x, val in zip(xx, df.TUWR_pct):
            ar.annotate(f'{val:.2f}', (x, val), xytext=(0, 7), textcoords='offset points',
                        ha='center', fontsize=8.5)
    else:
        spread = 100 * (df.ERRF.max() / df.ERRF.min() - 1)
        a.text(xx.mean(), 3.13, f'ERRF range {spread:.4f}%', ha='center', fontsize=8.5)
a.set_xlim(-.72, 13.32)
a.set_yticks([0, 1, 2, 3])
a.set_xticks(positions, labels)
a.legend([cost_handle, risk_handle], ['ERRF', 'TUWR'], loc='upper left',
         ncol=2, bbox_to_anchor=(0, 1.07), handlelength=1.3, columnspacing=1)

# Absolute outcomes at all six tested thresholds. No chosen threshold defines a zero.
b = panel(.391, 'b', 'Power-change threshold')
br = right_metric(b, (10.95, 11.67), [11.0, 11.2, 11.4, 11.6])
xx = np.arange(len(RAMP))
b.bar(xx, RAMP.ERRF, width=.58, color=TEAL, edgecolor='black', lw=.7, zorder=2)
br.plot(xx, RAMP.TUWR_pct, '-o', color=ORANGE, lw=1.15, markersize=5,
        markeredgecolor='black', markeredgewidth=.55, zorder=4)
for x, cost in zip(xx, RAMP.ERRF):
    b.annotate(f'{cost:.4f}' + ('\nLowest ERRF' if x == 5 else ''), (x, cost), xytext=(0, 5), textcoords='offset points',
               ha='center', fontsize=8.7)
risk_min = int(np.argmin(RAMP.TUWR_pct.to_numpy()))
cost_min = int(np.argmin(RAMP.ERRF.to_numpy()))
assert np.isclose(RAMP.iloc[risk_min].tau, .12)
assert np.isclose(RAMP.iloc[cost_min].tau, .18)
br.annotate('Lowest TUWR', (risk_min, RAMP.iloc[risk_min].TUWR_pct),
            xytext=(risk_min+.60, 11.025), ha='left', va='center', fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=.7, color='black'))
b.set_xlim(-.65, 5.70)
b.set_yticks([0, 1, 2, 3])
b.set_xticks(xx, [f'{v:.2f}' for v in RAMP.tau])
b.set_xlabel('Power-change threshold (p.u.)', labelpad=7)
b.legend([cost_handle, risk_handle], ['ERRF', 'TUWR'], loc='upper left',
         ncol=2, bbox_to_anchor=(0, 1.07), handlelength=1.3, columnspacing=1)

# Paired bars inherit the template's two-colour comparisons; their heights are cost.
c = panel(.087, 'c', 'Price adaptation', ymax=5.9)
c.spines['right'].set_visible(True)
xx = np.arange(6)
c.bar(xx-.18, PRICE.ERRF_fixed, width=.34, color=BLUE, edgecolor='black', lw=.7, zorder=2)
c.bar(xx+.18, PRICE.ERRF, width=.34, color=TEAL, edgecolor='black', lw=.7, zorder=2)
for x, row in zip(xx, PRICE.itertuples()):
    top = max(row.ERRF, row.ERRF_fixed)
    c.text(x, top+.13, f'{row.reduction_pct:.2f}%', ha='center', va='bottom',
           fontsize=9, fontweight='bold' if x == 5 else 'normal')
c.set_xlim(-.65, 5.70)
c.set_yticks([0, 1, 2, 3, 4, 5])
c.set_xticks(xx, ['1', '2', '20/3.56', '10', '20', 'All prices'])
c.set_xlabel('Price ratio', labelpad=7)
c.axvline(4.5, color='#AAB6BB', lw=.7, ls=(0, (3, 3)))
c.legend([Patch(facecolor=BLUE, edgecolor='black', linewidth=.65), cost_handle],
         ['Reference-price choices', 'Price-adapted choices'], ncol=2,
         loc='upper left', bbox_to_anchor=(0, 1.085), handlelength=1.3, columnspacing=1.5)

fig.canvas.draw()
for axis in fig.axes:
    assert axis.get_ylim()[0] >= 0
for suffix, dpi in [('pdf', 300), ('svg', 300), ('png', 300), ('tiff', 600)]:
    fig.savefig(OUT / f'Figure_7.{suffix}', dpi=dpi)
plt.close(fig)

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

metadata={'figure':7,'font':FONT_FAMILY,'input_directory':'results/figure_data/figure07','exports':[p.name for p in sorted(OUT.iterdir()) if p.is_file()]}
(OUT/'figure_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Figure 7 reproduced.')
