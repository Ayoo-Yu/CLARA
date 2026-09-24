"""Reproduce manuscript Figure 4 from its numerical source data."""
from _fonts import preferred_font
FONT_FAMILY = preferred_font()
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.patches import Rectangle
from matplotlib import font_manager

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FDATA = REPO / 'results' / 'figure_data'
ROOT = FDATA / 'figure04'
DATA = ROOT
OUT = REPO / 'outputs' / 'figure04'
OUT.mkdir(parents=True, exist_ok=True)

mpl.rcParams.update({
    'font.family': FONT_FAMILY,
    'font.size': 9,
    'axes.labelsize': 9,
    'axes.titlesize': 9.5,
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'text.color': 'black',
    'axes.labelcolor': 'black',
    'axes.edgecolor': 'black',
    'xtick.color': 'black',
    'ytick.color': 'black',
    'axes.linewidth': .7,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.top': True,
    'ytick.right': True,
    'svg.fonttype': 'none',
    'pdf.fonttype': 42,
    'savefig.facecolor': 'white',
})
font_path = font_manager.findfont(FONT_FAMILY, fallback_to_default=False)
ACTIONS = ['Static', 'ACI', 'AgACI', 'EnbPI-RH', 'TSC', 'EEE']
COLORS = {'Static': '#D99B68', 'ACI': '#87B79D', 'AgACI': '#7EACC7',
          'EnbPI-RH': '#ACB5BD', 'TSC': '#B29DC3', 'EEE': '#DDC47E'}
PRICES = ['R01', 'R02', 'R05P6179775281', 'R10', 'R20']
PRICE_LABELS = ['1', '2', '20/3.56', '10', '20']
COVERAGES = [.1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99]
FORECASTERS = ['GBR', 'MLP', 'QRLSTM', 'Ridge']

def style(ax, label, title, grid=None):
    ax.tick_params(length=3, width=.7, pad=3)
    row_y = .94 if label in ('a', 'b') else .525
    label_x = {'a': .043, 'b': .66, 'c': .043, 'd': .555}[label]
    title_artists[label] = fig.text(ax.get_position().x0, row_y, title,
                                   fontsize=9.5, ha='left', va='bottom')
    fig.text(label_x, row_y, f'({label})',
             weight='bold', fontsize=10, ha='left', va='bottom')
    if grid:
        ax.set_axisbelow(True)
        ax.grid(axis=grid, color='#DEDEDE', lw=.5)

# One wider discovery panel; each smaller panel supplies a distinct check.
fig = plt.figure(figsize=(7.40157480315, 5.62992125984))  # 188 by 143 mm
title_artists = {}
ax_a = fig.add_axes([.105, .64, .49, .275])
ax_b = fig.add_axes([.735, .64, .235, .275])
ax_c = fig.add_axes([.105, .13, .335, .37])
ax_d = fig.add_axes([.61, .13, .325, .37])

win = pd.read_csv(DATA / 'lowest_candidate_by_price_coverage.csv')
win['target_coverage'] = win.target_coverage.round(2)
matrix = win.pivot(index='price_id', columns='target_coverage', values='lowest_cost_candidate').reindex(index=PRICES, columns=COVERAGES)
assert matrix.shape == (5, 11) and matrix.notna().all().all()
for row, (_, series) in enumerate(matrix.iterrows()):
    values = list(series)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[end] == values[start]:
            end += 1
        method = values[start]
        ax_a.add_patch(Rectangle((start - .5, row - .5), end - start, 1,
                                facecolor=COLORS[method], edgecolor='white', lw=1))
        label = 'EnbPI-\nRH' if method == 'EnbPI-RH' else method
        ax_a.text((start + end - 1) / 2, row, label, ha='center', va='center',
                  fontsize=8 if end - start == 1 else 8.5, linespacing=.95)
        start = end
ax_a.set(xlim=(-.5, 10.5), ylim=(4.5, -.5), xlabel='Target coverage (%)', ylabel='Price ratio')
ax_a.set_xticks(range(11), ['10', '20', '30', '40', '50', '60', '70', '80', '90', '95', '99'])
ax_a.set_yticks(range(5), PRICE_LABELS)
style(ax_a, 'a', 'Lowest mean ERRF among all six candidates')

dist = pd.read_csv(DATA / 'candidate_cost_distribution.csv').set_index('method').reindex(ACTIONS)
stats = [dict(label=m, q1=dist.loc[m, 'q1'], med=dist.loc[m, 'median'], q3=dist.loc[m, 'q3'],
              whislo=dist.loc[m, 'minimum'], whishi=dist.loc[m, 'maximum'], fliers=[]) for m in ACTIONS]
boxes = ax_b.bxp(stats, positions=range(6), vert=False, widths=.55, patch_artist=True,
                  showfliers=False, manage_ticks=False,
                  boxprops=dict(linewidth=.7), medianprops=dict(color='black', linewidth=1.2),
                  whiskerprops=dict(color='black', linewidth=.7), capprops=dict(color='black', linewidth=.7))
for patch, method in zip(boxes['boxes'], ACTIONS):
    patch.set_facecolor(COLORS[method])
ax_b.set_yticks(range(6), ACTIONS)
ax_b.set(xlim=(-1, 62), ylim=(5.6, -.6), xlabel='Excess ERRF over cell minimum (%)')
ax_b.set_xticks([0, 20, 40, 60])
style(ax_b, 'b', 'Cost of retaining one candidate', grid='x')

t = pd.read_csv(DATA / 'conditional_forecaster_transfer_all_prices.csv')
t = t.loc[t.price_id.eq('ALL')].pivot(index='target_predictor', columns='source_predictor', values='change_pct').reindex(index=FORECASTERS, columns=FORECASTERS)
cmap_transfer = LinearSegmentedColormap.from_list('transfer_cost', ['#FFFFFF', '#B76453'])
im = ax_c.imshow(t, vmin=0, vmax=.4, cmap=cmap_transfer, aspect='auto', interpolation='nearest')
for i in range(4):
    for j in range(4):
        value = float(t.iloc[i, j])
        ax_c.text(j, i, '0' if i == j else f'+{value:.2f}', ha='center', va='center', fontsize=9)
ax_c.set_xticks(range(4), FORECASTERS, rotation=32, ha='right', rotation_mode='anchor')
ax_c.set_yticks(range(4), FORECASTERS)
ax_c.set(xlabel='Source forecaster', ylabel='Actual forecaster')
ax_c.set_xticks(np.arange(-.5, 4, 1), minor=True)
ax_c.set_yticks(np.arange(-.5, 4, 1), minor=True)
ax_c.grid(which='minor', color='white', linewidth=.7)
ax_c.tick_params(which='minor', length=0)
style(ax_c, 'c', 'Transferring a conditional rule')
cax_c = fig.add_axes([.452, .13, .012, .37])
cb_c = fig.colorbar(im, cax=cax_c, ticks=[0, .1, .2, .3, .4])
cb_c.set_label('ERRF increase (%)', labelpad=5)
cb_c.ax.tick_params(labelsize=8, length=2)

r = pd.read_csv(DATA / 'all_candidate_overall_and_rolling_reliability.csv')
assert len(r) == 13200
xmin, xmax, ymin, ymax = -15, 30, 0, 70
assert r.overall_gap_pp.between(xmin, xmax).all() and r.TUWR_pct.between(ymin, ymax).all()
cmap_counts = LinearSegmentedColormap.from_list('condition_count', ['#D8E5EB', '#6795AC', '#16415D'])
hb = ax_d.hexbin(r.overall_gap_pp, r.TUWR_pct, gridsize=(32, 25),
                 extent=(xmin, xmax, ymin, ymax), mincnt=1,
                 norm=LogNorm(vmin=1), cmap=cmap_counts, linewidths=0)
assert int(hb.get_array().sum()) == len(r)
ax_d.axvline(0, color='black', lw=.8, ls='--')
ax_d.set(xlim=(xmin, xmax), ylim=(ymin, ymax),
         xlabel='Overall coverage − target (pp)', ylabel='TUWR (%)')
ax_d.set_xticks([-10, 0, 10, 20, 30])
ax_d.set_yticks([0, 20, 40, 60])
style(ax_d, 'd', 'Overall versus rolling coverage', grid='both')
cax_d = fig.add_axes([.95, .13, .012, .37])
count_ticks = [10 ** k for k in range(5) if 10 ** k <= hb.get_array().max()]
cb_d = fig.colorbar(hb, cax=cax_d, ticks=count_ticks)
cb_d.ax.set_yticklabels([str(v) for v in count_ticks])
cb_d.ax.set_title('Count', fontsize=8, pad=7)
cb_d.ax.tick_params(labelsize=8, length=2)

for ax in (ax_c, ax_d):
    position = ax.get_position()
    ax.xaxis.set_label_coords(position.x0 + position.width / 2, .03,
                             transform=fig.transFigure)
    ax.xaxis.label.set_verticalalignment('baseline')

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
assert abs(title_artists['c'].get_window_extent(renderer).y0 -
           title_artists['d'].get_window_extent(renderer).y0) < .01
assert abs(ax_c.xaxis.label.get_transform().transform(ax_c.xaxis.label.get_position())[1] -
           ax_d.xaxis.label.get_transform().transform(ax_d.xaxis.label.get_position())[1]) < .01
outside = []
for artist in fig.findobj(mpl.text.Text):
    if not artist.get_visible() or not artist.get_text():
        continue
    box = artist.get_window_extent(renderer)
    if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
        outside.append(artist.get_text())
assert not outside, outside

stem = OUT / 'Figure_4'
fig.savefig(stem.with_suffix('.pdf'))
fig.savefig(stem.with_suffix('.svg'))
fig.savefig(stem.with_suffix('.tiff'), dpi=600, pil_kwargs={'compression': 'tiff_lzw'})
fig.savefig(stem.with_suffix('.png'), dpi=300)
plt.close(fig)

metadata={'figure':4,'font':FONT_FAMILY,'input_directory':'results/figure_data/figure04','exports':[p.name for p in sorted(OUT.iterdir()) if p.is_file()]}
(OUT/'figure_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Figure 4 reproduced.')
