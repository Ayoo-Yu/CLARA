"""Reproduce manuscript Figure 5 from its numerical source data."""
from _fonts import preferred_font
FONT_FAMILY = preferred_font()
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, to_hex
from PIL import Image
HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FDATA = REPO / 'results' / 'figure_data'
ROOT = REPO
OLD = FDATA / 'figure05' / 'main'
ORACLE = FDATA / 'figure05' / 'oracle'
EVID = FDATA / 'figure05' / 'ranking'
OUT = REPO / 'outputs' / 'figure05'
OUT.mkdir(parents=True, exist_ok=True)
DATA = OUT / 'source_data'
DATA.mkdir(exist_ok=True)
MAIN = ['CLARA', 'LinUCB', 'CART', 'TSC', 'EEE', 'Static', 'ACI', 'AgACI', 'EnbPI-RH']
METHODS = MAIN + ['WIDTH_ONLY', 'FEASIBLE_STATE_ORACLE']
LABEL = {'WIDTH_ONLY': 'Width-only', 'FEASIBLE_STATE_ORACLE': 'Feasible state oracle'}
ZONES = [f'zone{i}' for i in range(1, 11)]
PALETTE = {'CLARA':'#177E89', 'LinUCB':'#8865A2', 'CART':'#5678A5',
           'Static':'#D99B68', 'ACI':'#87B79D', 'AgACI':'#7EACC7',
           'EnbPI-RH':'#ACB5BD', 'TSC':'#B29DC3', 'EEE':'#DDC47E',
           'WIDTH_ONLY':'#9D9D9D', 'FEASIBLE_STATE_ORACLE':'#555555'}
TEAL = PALETTE['CLARA']
def tint(color, white=.62):
    return to_hex(np.asarray(to_rgb(color))*(1-white)+white)
def darken(color):
    return to_hex(np.asarray(to_rgb(color))*.70)
(OUT/'method_palette.json').write_text(json.dumps(PALETTE,indent=2),encoding='utf-8')

contract = {
    'conclusion': 'CLARA has the lowest mean ERRF among the implementable comparators; the figure also shows undercoverage, the constrained retrospective reference, paired zone differences, and stable first-place aggregate rankings.',
    'archetype': 'asymmetric quantitative composite: cost–undercoverage relationship, paired forest, and two compact ranking bars',
    'backend': 'python', 'dimensions_mm': [188, 164],
    'reference_adaptation': 'Style-only inheritance of compact aligned comparisons, restrained emphasis, and main-result plus supporting-panel composition from the supplied visual examples. No reference data or artwork reused.',
    'evidence': {'a': 'Mean ERRF versus TUWR, all 10 implementable methods and the feasible state oracle; shaded upper-right area has both higher ERRF and TUWR than CLARA', 'b': 'CLARA-relative differences for 9 comparators and the feasible oracle; all 10 zone estimates plus original pointwise paired-bootstrap 95% intervals', 'c': 'Mean ranks of all 9 main methods and all 10 single-zone omission estimates', 'd': 'Mean relative excess costs for all 9 main methods and all 10 single-zone omission estimates'},
    'statistics': 'Equal-weight aggregation over matched forecasting conditions and zones. Bootstrap intervals retain the existing analysis. The leave-one-zone-out range is descriptive sensitivity, not a confidence interval. No inferential test is implied by omission dots.',
    'data_accounting': 'No observations or price conditions removed. Tables 6 and 7 are preserved in full in supplementary tables; capacity, exceedance, ARD and width remain in the dedicated Section 4.3 analyses. The ranking universe is the nine main methods defined in Section 3.2.',
    'export': ['pdf with editable embedded TrueType text', 'svg with editable text', '600 dpi TIFF', '300 dpi PNG', 'source CSV files'],
    'style': 'Times New Roman, black axes and labels, inward ticks, light grids, direct labels, and bars with zero baselines. The six candidate colors match the accepted preceding figure exactly; selector colors follow the existing manuscript method colors.'
}
(OUT / 'figure_contract.json').write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding='utf-8')

source_files = [OLD/'main_overall.csv', OLD/'paired_main_comparisons.csv', OLD/'zone_reductions.csv', ORACLE/'oracle_means.csv', ORACLE/'oracle_gaps.csv', ORACLE/'oracle_zone_means.csv', EVID/'gefcom_ranks.csv', EVID/'gefcom_rank_cells.parquet']
means = pd.read_csv(OLD/'main_overall.csv').set_index('method').reindex(METHODS[:-1])
om = pd.read_csv(ORACLE/'oracle_means.csv')
means = pd.concat([means, om[(om.price_id == 'ALL') & (om.method == METHODS[-1])].set_index('method')]).reindex(METHODS)
assert means[['ERRF', 'TUWR', 'capacity', 'exceedance', 'ARD', 'width']].notna().all().all()
assert np.allclose(means.ERRF, means.capacity + means.exceedance)
assert f'{means.loc["CLARA", "ERRF"]:.4f}' == '2.8452'
means.to_csv(DATA/'mean_performance.csv')

ci = pd.read_csv(OLD/'paired_main_comparisons.csv').set_index('baseline')
og = pd.read_csv(ORACLE/'oracle_gaps.csv').query("price_id == 'ALL' and oracle == 'FEASIBLE_STATE_ORACLE'").iloc[0]
ci.loc[METHODS[-1], ['relative_pct', 'low', 'high', 'wins']] = [og.gap_pct, og.ci95_low, og.ci95_high, 0]
ci = ci.reindex(METHODS[1:])
assert ci[['relative_pct', 'low', 'high']].notna().all().all()
assert (ci.loc[METHODS[1:-1], 'high'] < 0).all()
ci.to_csv(DATA/'paired_relative_differences.csv')
zd = pd.read_csv(OLD/'zone_reductions.csv').rename(columns={'reduction_pct':'relative_pct'})
zd.relative_pct *= -1
oz = pd.read_csv(ORACLE/'oracle_zone_means.csv')
oz = oz[oz.price_id == 'ALL'].pivot(index='zone_or_farm', columns='method', values='ERRF').reindex(ZONES)
assert (oz.FEASIBLE_STATE_ORACLE > 0).all()
ov = 100*(oz.CLARA / oz.FEASIBLE_STATE_ORACLE - 1)
zd = pd.concat([zd, pd.DataFrame({'zone': ZONES, 'baseline': METHODS[-1], 'relative_pct': ov.to_numpy()})], ignore_index=True)
assert len(zd) == 100 and zd.relative_pct.notna().all()
zd.to_csv(DATA/'all_zone_relative_differences.csv', index=False)

ranks = pd.read_csv(EVID/'gefcom_ranks.csv')
ranks = ranks[(ranks.price_id == 'ALL') & ranks.zone_or_farm.isin(ZONES) & ranks.method.isin(MAIN)]
assert len(ranks) == 90 and ranks.groupby('method').size().eq(10).all()
overall = ranks.groupby('method')[['rank','excess']].mean().reindex(MAIN)
assert np.isclose(overall.loc['CLARA','rank'], 2.660409090909091)
assert np.isclose(overall.loc['CLARA','excess'], 1.645490861, atol=1e-7)
loo = pd.concat([ranks[ranks.zone_or_farm != z].groupby('method')[['rank','excess']].mean().assign(omitted_zone=z).reset_index() for z in ZONES], ignore_index=True)
assert len(loo) == 90
for z in ZONES:
    q = loo[loo.omitted_zone == z].set_index('method')
    assert q['rank'].idxmin() == q['excess'].idxmin() == 'CLARA'
overall.to_csv(DATA/'overall_ranking.csv')
loo.to_csv(DATA/'all_zone_omission_estimates.csv', index=False)
# Verify the full condition count without applying any new data exclusions.
cells = pd.read_parquet(EVID/'gefcom_rank_cells.parquet')
print('Rank-cell schema:', len(cells), list(cells.columns))

plt.rcParams.update({'font.family': FONT_FAMILY, 'font.size': 8.5,
    'axes.labelsize': 9, 'axes.titlesize': 9.5, 'axes.edgecolor': 'black',
    'axes.labelcolor': 'black', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
    'axes.linewidth': .65, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3, 'ytick.major.size': 0, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
width_mm = 188
height_mm = 164
fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4))
# Preserve physical panel heights; reclaim the external legend's 8 mm gap.
top_y = (.552 * 172 - 8) / height_mm
top_h = .382 * 172 / height_mm
bottom_y = .100 * 172 / height_mm
bottom_h = .315 * 172 / height_mm
a = fig.add_axes([.115, top_y, .425, top_h])
b = fig.add_axes([.690, top_y, .280, top_h])
c = fig.add_axes([.115, bottom_y, .397, bottom_h])
d = fig.add_axes([.628, bottom_y, .342, bottom_h], sharey=c)

def style(ax, letter, title, grid='x'):
    ax.set_axisbelow(True)
    ax.grid(axis=grid, color='#DDE2E5', lw=.5)
    ax.tick_params(axis='both', labelsize=8, pad=4)
    ax.tick_params(axis='x', top=True)
    ax.annotate(f'({letter})',(0,1),xycoords='axes fraction',xytext=(0,8),textcoords='offset points',weight='bold',fontsize=9.5,ha='left',va='bottom')
    ax.annotate(title,(0,1),xycoords='axes fraction',xytext=(19,8),textcoords='offset points',fontsize=9.5,ha='left',va='bottom')

style(a,'a','Cost–undercoverage balance','both')
a.set(xlim=(2.785,3.285),ylim=(-1.5,45),xlabel='Mean ERRF',ylabel='TUWR (%)')
a.set_xticks([2.8,2.9,3.0,3.1,3.2]);a.set_yticks([0,10,20,30,40])
cx,cy=means.loc['CLARA','ERRF'],100*means.loc['CLARA','TUWR']
a.fill_between([cx,3.285],[cy,cy],[45,45],facecolor='#EDF5F5',zorder=0)
a.plot([cx,3.285],[cy,cy],color='#77A4A8',ls=(0,(3,3)),lw=.7,zorder=1)
positions={
    'CLARA':(2.80,20.2,'left'),
    'LinUCB':(2.895,18.6,'center'),
    'CART':(2.965,16.6,'left'),
    'TSC':(3.025,4.5,'left'),
    'EEE':(3.045,8.0,'left'),
    'Static':(3.15,33.2,'center'),
    'ACI':(3.023,.2,'left'),
    'AgACI':(2.934,2.4,'right'),
    'EnbPI-RH':(3.137,20.2,'center'),
    'WIDTH_ONLY':(3.18,41.0,'center'),
    'FEASIBLE_STATE_ORACLE':(2.798,4.8,'left')}
for m in METHODS:
    x,y=means.loc[m,'ERRF'],100*means.loc[m,'TUWR']
    col=PALETTE[m]
    marker='D' if m in ['CLARA',METHODS[-1]] else '^' if m=='WIDTH_ONLY' else 'o'
    a.scatter(x,y,s=48 if m=='CLARA' else 33,marker=marker,facecolors='white' if m==METHODS[-1] else col,edgecolors=col if m==METHODS[-1] else 'white',linewidths=1.0 if m in ['CLARA',METHODS[-1]] else .7,zorder=5)
    if m in ['ACI','AgACI','TSC']:continue
    tx,ty,ha=positions[m]
    name='Feasible state\noracle' if m==METHODS[-1] else LABEL.get(m,m)
    a.annotate(name,(x,y),xytext=(tx,ty),ha=ha,va='center',fontsize=8.2,fontweight='bold' if m=='CLARA' else 'normal',
               arrowprops={'arrowstyle':'-','color':'#555555','lw':.5,'shrinkA':2,'shrinkB':4},zorder=6)

# Magnification separates three almost coincident low-undercoverage candidates.
ins=a.inset_axes([.085,.63,.43,.265])
ins.set(xlim=(2.973,3.006),ylim=(1.05,2.05))
ins.set_xticks([2.98,3.00]);ins.set_yticks([1.2,1.8])
ins.tick_params(axis='both',labelsize=6.4,length=2,pad=2)
ins.grid(color='#E4E4E4',lw=.4);ins.set_axisbelow(True)
ins.set_title('Low-TUWR candidates',fontsize=7.4,pad=3)
for m,off in [('ACI',(1,-8)),('AgACI',(-1,8)),('TSC',(0,8))]:
    xv,yv=means.loc[m,'ERRF'],100*means.loc[m,'TUWR']
    ins.plot(xv,yv,'o',ms=3.5,mfc=PALETTE[m],mec='white',mew=.5)
    ins.annotate(m,(xv,yv),xytext=off,textcoords='offset points',fontsize=7.0,ha='center',va='center')
a.text(3.008,1.45,'See inset',fontsize=7,va='center')

style(b,'b','Paired cost difference')
bm=METHODS[1:]
ys=np.arange(len(bm),dtype=float);ys[-1]+=.45
b.set(xlim=(-14.1,2.0),ylim=(10.2,-.65),xlabel='CLARA ERRF change (%)')
b.set_xticks([-12,-8,-4,0])
b.set_yticks(ys,[('Feasible state\noracle' if m==METHODS[-1] else LABEL.get(m,m)) for m in bm],fontsize=8)
b.axvline(0,color='black',lw=.65,ls='--')
b.axhline(8.65,color='#B7B7B7',ls='--',lw=.5)
for i,m in enumerate(bm):
    q=ci.loc[m]
    vals=zd[zd.baseline==m].set_index('zone').reindex(ZONES).relative_pct.to_numpy()
    offsets=np.linspace(-.19,.19,10)
    col=PALETTE[m]
    b.plot([vals.min(),vals.max()],[ys[i],ys[i]],color=tint(col),lw=.65,zorder=1)
    b.scatter(vals,ys[i]+offsets,s=9,color=tint(col,.50),edgecolors='none',zorder=2)
    b.errorbar(q.relative_pct,ys[i],xerr=[[q.relative_pct-q.low],[q.high-q.relative_pct]],fmt='D' if m==METHODS[-1] else 'o',ms=3.7,
               mfc='white' if m==METHODS[-1] else col,mec=darken(col),
               color=darken(col),elinewidth=1.1,capsize=2.5,capthick=.8,zorder=5)
b.legend(handles=[Line2D([],[],marker='o',ls='',mfc='#BBBBBB',mec='none',ms=3,label='Zone'),Line2D([],[],marker='o',color='#555555',lw=1,ms=3,label='Mean [95% CI]')],
         loc='upper left',bbox_to_anchor=(.035,.985),ncol=1,fontsize=7.3,frameon=True,facecolor='white',edgecolor='none',framealpha=1,borderaxespad=0,borderpad=.25,handlelength=1.2,columnspacing=.9,handletextpad=.4)

# The same method order is used for both aggregate ranking criteria.
rankorder=overall.sort_values('rank').index.tolist()
style(c,'c','Economic ranking')
style(d,'d','Relative excess cost')
c.set(xlim=(0,7.7),ylim=(8.65,-.65),xlabel='Mean rank')
d.set(xlim=(0,12.5),xlabel='Mean relative excess ERRF (%)')
c.set_xticks([0,2,4,6]);d.set_xticks([0,3,6,9,12])
c.set_yticks(range(9),rankorder,fontsize=8.5);d.tick_params(axis='y',labelleft=False)
for ax,metric in [(c,'rank'),(d,'excess')]:
    for i,m in enumerate(rankorder):
        col=PALETTE[m]
        q=loo[loo.method==m].set_index('omitted_zone').reindex(ZONES)[metric]
        v=overall.loc[m,metric]
        ax.barh(i,v,height=.59,color=col,edgecolor='white',linewidth=.45,zorder=2)
        ax.plot([q.min(),q.max()],[i,i],color='black',lw=.65,zorder=4)
        ax.scatter(q,i+np.linspace(-.17,.17,10),s=4.8,color=darken(col),edgecolors='white',linewidths=.15,zorder=5)
        ax.annotate(f'{v:.2f}',(max(v,q.max()),i),xytext=(5,0),textcoords='offset points',va='center',fontsize=8,fontweight='bold' if m=='CLARA' else 'normal')
from matplotlib.patches import Patch
fig.legend(handles=[Patch(facecolor='#B0B0B0',edgecolor='none',label='All 10 zones'),Line2D([],[],marker='o',ls='',mfc='#555555',mec='none',ms=3,label='After omitting one zone (10 estimates)')],
           loc='lower center',bbox_to_anchor=(.555,1.72/height_mm),ncol=2,fontsize=8,frameon=False,handletextpad=.6,columnspacing=1.7)

fig.canvas.draw()
renderer = fig.canvas.get_renderer()
legend = b.get_legend()
legend_box = legend.get_window_extent(renderer)
panel_box = b.get_window_extent(renderer)
assert panel_box.contains(legend_box.x0, legend_box.y0)
assert panel_box.contains(legend_box.x1, legend_box.y1)
# The legend is left of all marks in the two rows it occupies.
legend_data_right, legend_data_bottom = b.transData.inverted().transform((legend_box.x1, legend_box.y0))
for i, m in enumerate(bm):
    if ys[i] - .25 <= legend_data_bottom:
        row = zd[zd.baseline == m].relative_pct
        assert legend_data_right < min(row.min(), ci.loc[m, 'low']) - .15
outside = []
for artist in fig.findobj(matplotlib.text.Text):
    if not artist.get_visible() or not artist.get_text():
        continue
    box = artist.get_window_extent(renderer)
    if box.x0 < -1 or box.y0 < -1 or box.x1 > fig.bbox.width + 1 or box.y1 > fig.bbox.height + 1:
        outside.append(artist.get_text())
assert not outside, outside
layout_qa = {'legend_inside_panel_b': True, 'legend_clear_of_data': True,
             'panel_b_left_shift_mm': (.736 - .690) * width_mm,
             'inter_row_gap_reduction_mm': 8, 'physical_panel_heights_preserved': True,
             'outside_canvas_text': outside}
(OUT/'layout_checks.json').write_text(json.dumps(layout_qa, indent=2), encoding='utf-8')
fig.savefig(OUT/'Figure_5.pdf')
fig.savefig(OUT/'Figure_5.svg')
fig.savefig(OUT/'Figure_5.png', dpi=300)
fig.savefig(OUT/'Figure_5.tiff', dpi=600, pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)

metadata={'figure':5,'font':FONT_FAMILY,'input_directory':'results/figure_data/figure05','exports':[p.name for p in sorted(OUT.iterdir()) if p.is_file()]}
(OUT/'figure_metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Figure 5 reproduced.')
