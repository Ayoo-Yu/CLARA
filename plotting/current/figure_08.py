"""Figure 8: retained estimation and threshold sensitivity panels.

Reference is a visual template only. All plotted values are saved CLARA results.
Metric identities retain consistent colours across rows; all bar axes start at zero.
Font: Times New Roman as explicitly requested by the author.
"""
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
import fitz

SCRIPT_DIR=Path(__file__).resolve().parent
REPO=SCRIPT_DIR.parents[1]
HERE=REPO/'results/current_figure_data/figure08'
OUTPUT=REPO/'outputs/current/figure08'
OUTPUT.mkdir(exist_ok=True,parents=True)
DATA = HERE
OUT = OUTPUT
OUT.mkdir(exist_ok=True)
PARAM = pd.read_csv(DATA / 'parameter_sensitivity.csv').query("price_id == 'ALL'")
TOL = pd.read_csv(DATA / 'coverage_tolerances.csv').sort_values('multiplier')
RAMP = pd.read_csv(DATA / 'threshold_sensitivity.csv').sort_values('tau')
TEAL, BLUE, ORANGE = '#8AB7BE', '#CAD9DE', '#C68865'
GRID = '#E2E6E8'
font_manager.findfont('Times New Roman', fallback_to_default=False)
plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'], 'font.size': 10,
    'axes.labelsize': 10, 'axes.titlesize': 11, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'axes.edgecolor': 'black', 'axes.labelcolor': 'black', 'text.color': 'black',
    'xtick.color': 'black', 'ytick.color': 'black', 'axes.linewidth': .8,
    'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': False,
    'ytick.right': False, 'legend.frameon': False, 'legend.fontsize': 9,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
    'figure.facecolor': 'white', 'savefig.facecolor': 'white',
})
WIDTH_MM, HEIGHT_MM = 183, 126.6
fig = plt.figure(figsize=(WIDTH_MM / 25.4, HEIGHT_MM / 25.4), dpi=300)

def panel(bottom, letter, title, ymax=3.65):
    # Preserve physical geometry of retained panels; remove the final row.
    ax = fig.add_axes([.100, (184 * bottom - 57.4) / HEIGHT_MM, .775, .208 * 184 / HEIGHT_MM])
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

# Approved layout: compact bars and outlined metric points in aligned rows.
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
risk_range=max(float(RAMP.TUWR_pct.max()-RAMP.TUWR_pct.min()),.2)
risk_low=float(RAMP.TUWR_pct.min())-.22*risk_range
risk_high=float(RAMP.TUWR_pct.max())+.32*risk_range
br = right_metric(b, (risk_low,risk_high), np.linspace(risk_low,risk_high,4))
from matplotlib.ticker import FormatStrFormatter
br.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
xx = np.arange(len(RAMP))
b.bar(xx, RAMP.ERRF, width=.58, color=TEAL, edgecolor='black', lw=.7, zorder=2)
br.plot(xx, RAMP.TUWR_pct, '-o', color=ORANGE, lw=1.15, markersize=5,
        markeredgecolor='black', markeredgewidth=.55, zorder=4)
for x, cost in zip(xx, RAMP.ERRF):
    b.annotate(f'{cost:.4f}' + ('\nLowest ERRF' if x == int(np.argmin(RAMP.ERRF.to_numpy())) else ''), (x, cost), xytext=(0, 5), textcoords='offset points',
               ha='center', fontsize=8.7)
risk_min = int(np.argmin(RAMP.TUWR_pct.to_numpy()))
cost_min = int(np.argmin(RAMP.ERRF.to_numpy()))
assert np.isfinite(RAMP[['ERRF','TUWR_pct']].to_numpy()).all()
br.annotate('Lowest TUWR', (risk_min, RAMP.iloc[risk_min].TUWR_pct),
            xytext=(min(risk_min+.6,4.4), risk_low+.08*risk_range), ha='left', va='center', fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=.7, color='black'))
b.set_xlim(-.65, 5.70)
b.set_yticks([0, 1, 2, 3])
b.set_xticks(xx, [f'{v:.2f}' for v in RAMP.tau])
b.set_xlabel('Power-change threshold (p.u.)', labelpad=7)
b.legend([cost_handle, risk_handle], ['ERRF', 'TUWR'], loc='upper left',
         ncol=2, bbox_to_anchor=(0, 1.07), handlelength=1.3, columnspacing=1)


# Preserve approved panel geometry; all values come from the new chronological analysis.
fig.canvas.draw()
assert len(fig.axes) == 4
for suffix, dpi in [('pdf', 300), ('svg', 300), ('png', 300), ('tiff', 600)]:
    fig.savefig(OUT / f'Figure_08.{suffix}', dpi=dpi)
plt.close(fig)
with fitz.open(OUT / 'Figure_08.pdf') as pdf:
    assert len(pdf) == 1
    assert len(pdf[0].get_images(full=True)) == 0
    assert all('Times' in f[3] for f in pdf[0].get_fonts(full=True))
    pdf[0].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False).save(OUT / 'Figure_08_pdf_preview.png')
def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()
manifest = {
    'status': 'AWAITING_VISUAL_QA',
    'scope': 'Recompute the same registered estimation, tolerance and source-validation threshold sensitivities under the adopted common chronological cutoff.',
    'reuse_level': 'Structural adaptation: approved layout and palette, all new cutoff-limited data and data-determined annotations',
    'dimensions_mm': [WIDTH_MM, HEIGHT_MM],
    'original_dimensions_mm': [183, 184],
    'physical_axes_height_mm': 38.272,
    'palette': {'ERRF': TEAL, 'TUWR': ORANGE},
    'font': 'Times New Roman',
    'panel_a': 'Five-price test means; fixed counts 10/20/30/50, shrinkage 0/10/30/60/100, joint tolerance multipliers 0.5/1/2; cost bars and right-axis TUWR.',
    'panel_b': 'Six source-zone validation thresholds 0.08 through 0.18 p.u.; excludes outer test zone; cost and TUWR minima marked separately.',
    'rows_retained': {'parameter_all_prices': len(PARAM), 'coverage_multipliers': len(TOL), 'validation_thresholds': len(RAMP)},
    'data_hashes': {p.name: sha(p) for p in DATA.glob('*.csv')},
    'output_hashes': {p.name: sha(p) for p in OUT.glob('Figure_08.*')},
}
(OUT / 'figure_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps(manifest, ensure_ascii=False))
