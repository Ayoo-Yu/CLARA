from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb, to_hex
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
SCRIPT_DIR=Path(__file__).resolve().parent
REPO=SCRIPT_DIR.parents[1]
HERE=REPO/'results/current_figure_data/figure05'
OUTPUT=REPO/'outputs/current/figure05'
OUTPUT.mkdir(exist_ok=True,parents=True)
OUT = OUTPUT
DATA = HERE
MAIN = ['CLARA', 'LinUCB', 'CART', 'TSC', 'EEE', 'Static', 'ACI', 'AgACI', 'EnbPI-RH']
METHODS = MAIN+['WIDTH_ONLY', 'FEASIBLE_STATE_ORACLE']
LABEL = {'WIDTH_ONLY': 'Width-only', 'FEASIBLE_STATE_ORACLE': 'Feasible state oracle'}
ZONES = [f'zone{i}' for i in range(1,11)]
PALETTE=json.loads((DATA/'method_palette.json').read_text())
means=pd.read_csv(DATA/'mean_performance.csv').set_index('method')
overall=pd.read_csv(DATA/'overall_ranking.csv').set_index('method')
loo=pd.read_csv(DATA/'all_zone_omission_estimates.csv')
horizon=pd.read_csv(DATA/'lead_time_trajectories.csv')
ci=pd.read_csv(DATA/'paired_relative_differences.csv').set_index('baseline')
zd=pd.read_csv(DATA/'all_zone_relative_differences.csv')
assert len(horizon)==45
assert (horizon.TUWR_pct>0).all()
def tint(color,white=.62): return to_hex(np.asarray(to_rgb(color))*(1-white)+white)
def darken(color): return to_hex(np.asarray(to_rgb(color))*.70)
plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 8.5,
    'axes.labelsize': 9, 'axes.titlesize': 9.5, 'axes.edgecolor': 'black',
    'axes.labelcolor': 'black', 'text.color': 'black', 'xtick.color': 'black', 'ytick.color': 'black',
    'axes.linewidth': .65, 'xtick.direction': 'in', 'ytick.direction': 'in',
    'xtick.major.size': 3, 'ytick.major.size': 0, 'pdf.fonttype': 42,
    'svg.fonttype': 'none', 'savefig.facecolor': 'white'})
width_mm = 188
height_mm = 174
fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4))
a = fig.add_axes([.105, .563, .39, .365])
b = fig.add_axes([.620, .563, .29, .365])
c = fig.add_axes([.105, .142, .39, .318])
d = fig.add_axes([.620, .142, .29, .318])
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
ins.set(xlim=(2.962,2.995),ylim=(1.02,2.05))
ins.set_xticks([2.97,2.99]);ins.set_yticks([1.2,1.8])
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
         loc='upper left',bbox_to_anchor=(.015,.99),ncol=1,fontsize=6.8,frameon=False,borderaxespad=0,handlelength=1.2,columnspacing=.9,handletextpad=.4)

rankorder=overall.sort_values('rank').index.tolist()
heading_y=.460+(8/72)/(height_mm/25.4)
fig.text(.105,heading_y,'(c)',weight='bold',fontsize=9.5,va='bottom')
fig.text(.1407,heading_y,'Rank and relative excess cost',fontsize=9.5,va='bottom')
c.set(xlim=(1.1,8.15),ylim=(8.65,-.65),xlabel='Mean economic rank')
c.set_xticks([2,4,6,8])
c.set_yticks(range(9),rankorder,fontsize=7.8)
c.tick_params(axis='both',pad=4,top=True)
c.set_axisbelow(True);c.grid(axis='x',color='#DDE2E5',lw=.5)
c.axhspan(-.5,.5,color='#EDF5F5',zorder=0)
area_scale=20.0
for i,m in enumerate(rankorder):
    col=PALETTE[m]
    val=overall.loc[m]
    q=loo[loo.method.eq(m)]
    c.scatter(val['rank'],i,s=area_scale*val['excess'],color=col,edgecolors='white',linewidths=.7,zorder=3)
    # Horizontal intervals preserve rank sensitivity; thin concentric boundaries
    # preserve the excess-cost range under the same ten zone omissions.
    for value in [q.excess.min(),q.excess.max()]:
        c.scatter(val['rank'],i,s=area_scale*value,facecolors='none',edgecolors=darken(col),linewidths=.4,zorder=4)
    c.plot([q['rank'].min(),q['rank'].max()],[i,i],color='black',lw=.75,zorder=5)
    radius=np.sqrt(area_scale*max(val['excess'],q.excess.max()))/2
    for label,side in [(f"{val['rank']:.2f}",-1),(f"{val['excess']:.2f}%",1)]:
        c.annotate(label,(val['rank'],i),xytext=(side*(radius+3),0),textcoords='offset points',
                   ha='right' if side<0 else 'left',va='center',fontsize=7.4,
                   weight='bold' if m=='CLARA' else 'normal')
c.get_yticklabels()[0].set_weight('bold')
handles=[plt.scatter([],[],s=area_scale*v,color='#B4B4B4',edgecolors='none',label=f'{v}%') for v in [2,5,10]]
c.legend(handles=handles,title='Bubble area: relative excess cost',loc='upper center',bbox_to_anchor=(.50,-.19),
         ncol=3,fontsize=7,title_fontsize=7.4,frameon=False,handletextpad=.5,columnspacing=1.5)
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
costorder=means.loc[MAIN].sort_values('ERRF').index.tolist()
HORIZONS=[1,3,6,12,24]
costs=horizon.pivot(index='method',columns='horizon_steps',values='ERRF').loc[costorder,HORIZONS]
ranks=costs.rank(axis=0,method='min')
assert np.isfinite(ranks.to_numpy()).all()
style(d,'d','ERRF across lead times',grid='both')
d.grid(False)
cmap=LinearSegmentedColormap.from_list('economic_rank',['#79B1B7','#EAF3F3'],N=9)
norm=BoundaryNorm(np.arange(.5,10.5,1),9)
im=d.pcolormesh(np.arange(6)-.5,np.arange(10)-.5,ranks.to_numpy(),cmap=cmap,norm=norm,
                shading='flat',edgecolors='white',linewidth=.8,rasterized=False,antialiased=False)
d.set(xlim=(-.5,4.5),ylim=(8.5,-.5))
d.set_xticks(range(5),[str(h) for h in HORIZONS],fontsize=8)
d.set_yticks(range(9),costorder,fontsize=7.8)
d.tick_params(axis='both',length=0,pad=4)
d.set_xlabel('Forecast lead time (h)')
d.set_xticks(np.arange(-.5,5,1),minor=True)
d.set_yticks(np.arange(-.5,9,1),minor=True)
d.grid(which='minor',color='white',linewidth=.8)
d.tick_params(which='minor',length=0)
for i,m in enumerate(costorder):
    for j,h in enumerate(HORIZONS):
        rank=ranks.loc[m,h]
        d.text(j,i,f'{costs.loc[m,h]:.4f}',ha='center',va='center',fontsize=7.3,
               color='black',weight='bold' if m=='CLARA' else 'normal')
d.get_yticklabels()[0].set_weight('bold')
cax=fig.add_axes([.644,.055,.242,.010])
cb=fig.colorbar(im,cax=cax,orientation='horizontal',ticks=[1,3,5,7,9])
cb.ax.tick_params(labelsize=7,length=2,pad=2)
cb.outline.set_linewidth(.5)
cb.set_label('Cost rank within each lead time (1 = lowest)',fontsize=7.1,labelpad=2)
costs.to_csv(OUT/'figure5d_mean_ERRF.csv')
ranks.to_csv(OUT/'figure5d_cost_ranks.csv')


assert a.get_position().x0 == c.get_position().x0
assert a.get_position().x1 == c.get_position().x1
assert b.get_position().x0 == d.get_position().x0
assert b.get_position().x1 == d.get_position().x1
assert all(t.get_color() == 'black' for t in d.texts)
assert len(d.texts) == 47  # 45 values plus panel letter and title.
fig.savefig(OUT/'Figure_05.pdf')
fig.savefig(OUT/'Figure_05.svg')
fig.savefig(OUT/'Figure_05.png',dpi=300)
fig.savefig(OUT/'Figure_05.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
plt.close(fig)
print('Saved updated Figure 5: a/b retained, c combines both economic ranking metrics, d shows all 45 lead-time points.')
