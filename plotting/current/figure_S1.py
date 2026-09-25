"""Reproduce current Supplementary Figure S1 from the saved GEFCom chronology."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap
SCRIPT_DIR=Path(__file__).resolve().parent
REPO=SCRIPT_DIR.parents[1]
HERE=REPO/'results/current_figure_data/figureS1'
OUTPUT=REPO/'outputs/current/figureS1'
OUTPUT.mkdir(exist_ok=True,parents=True)
DATA = HERE
OUT = OUTPUT
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'Times New Roman','font.size':10,'axes.labelsize':11,'axes.titlesize':11,'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black','text.color':'black','axes.edgecolor':'black','axes.linewidth':.8,'xtick.direction':'in','ytick.direction':'in','xtick.top':True,'ytick.right':True,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none','legend.frameon':True,'legend.fancybox':False,'legend.edgecolor':'black','legend.framealpha':1,'savefig.facecolor':'white'})
ORDER=['CLARA','LinUCB','CART','TSC','EEE','Static','ACI','AgACI','EnbPI-RH','WIDTH_ONLY']
LABEL={'WIDTH_ONLY':'Width-only','OUTCOME_ORACLE':'Outcome oracle'}
COL=dict(zip(ORDER,['#177E89','#8865A2','#5678A5','#9B7048','#AB799B','#D07C30','#51A087','#DFB13C','#73AEC2','#9D9D9D']))
CAP='#537FA7';MISS='#D25D61'
HEAT=LinearSegmentedColormap.from_list('light_diverging',['#ECC1BB','#FFFFFF','#91BAD4'])
saved=[]
def read(n):return pd.read_csv(OUT/n)
def lab(m):return LABEL.get(m,m)
def style(ax,letter=None,grid='y'):
 ax.set_axisbelow(True)
 if grid:ax.grid(axis=grid,color='#DDDDDD',lw=.5)
 if letter:ax.text(0,1.045,'('+letter+')',transform=ax.transAxes,fontweight='bold',fontsize=12)
def save(fig,num):
 for ext in ['pdf','svg','png']:fig.savefig(OUT/f'Figure_{num}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.04)
 fig.savefig(OUT/f'Figure_{num}_preview.png',dpi=150,bbox_inches='tight',pad_inches=.04)
 plt.close(fig);saved.append(str(num))
def labels(ax,methods):ax.set_xticks(range(len(methods)),[lab(m) for m in methods],rotation=48,ha='right',rotation_mode='anchor',fontsize=9)
# Three complementary coverage summaries on the post-cutoff evaluation period.
c=pd.read_csv(DATA/'chronology.csv');c.issue_timestamp=pd.to_datetime(c.issue_timestamp)
methods=['Static','EnbPI-RH','CLARA'];summary=c[c.method.isin(methods)].groupby('method')[['covered','TUWR']].mean().reindex(methods)
fig=plt.figure(figsize=(7.5,3.9));gs=fig.add_gridspec(2,2,width_ratios=[1,2.1],hspace=.7,wspace=.35)
a=fig.add_subplot(gs[0,0]);b=fig.add_subplot(gs[:,1]);d=fig.add_subplot(gs[1,0])
a.bar(range(3),100*summary.covered,color=[COL[m] for m in methods],edgecolor='black',lw=.5);a.axhline(90,color='black',ls='--',lw=.8);a.set(ylim=(0,105),ylabel='Global coverage (%)');a.set_xticks(range(3),methods,fontsize=8);style(a,'a')
for i,v in enumerate(summary.covered):a.text(i,100*v+2,f'{100*v:.2f}',ha='center',fontsize=8)
for m in methods:
 z=c[(c.method==m)&c.issue_timestamp.between('2013-09-01','2013-10-01')];b.plot(z.issue_timestamp,100*z.rolling_coverage,lw=1.1,label=m,color=COL[m])
b.axhline(100*(.9-1.96*np.sqrt(.9*.1/168)),color='black',ls='--',lw=.8,label='Lower tolerance');b.axhline(90,color='black',ls=':',lw=.8);b.set(ylabel='168 h rolling coverage (%)',xlabel='Forecast date (2013)');style(b,'b','both');b.legend(fontsize=8,loc='lower right');b.xaxis.set_major_locator(mdates.DayLocator(interval=7));b.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
d.bar(range(3),100*summary.TUWR,color=[COL[m] for m in methods],edgecolor='black',lw=.5);d.set(ylabel='TUWR (%)',ylim=(0,25));d.set_xticks(range(3),methods,fontsize=8);style(d,'c')
for i,v in enumerate(summary.TUWR):d.text(i,100*v+.5,f'{100*v:.2f}',ha='center',fontsize=8)
save(fig,'S1')
