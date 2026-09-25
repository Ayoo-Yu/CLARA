"""Reproduce manuscript Figure 6 from its numerical source data."""
FONT_FAMILY = 'Times New Roman'
from pathlib import Path
import json, hashlib
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from matplotlib import font_manager as fm

SCRIPT_DIR=Path(__file__).resolve().parent
REPO=SCRIPT_DIR.parents[1]
HERE=REPO/'results/current_figure_data/figure06'
OUTPUT=REPO/'outputs/current/figure06'
OUTPUT.mkdir(exist_ok=True,parents=True)
FDATA=HERE
SRC=HERE
DATA=HERE;FIG=OUTPUT
mpl.rcParams.update({
    'font.family':FONT_FAMILY,'font.size':8,'text.color':'#000000',
    'axes.labelcolor':'#000000','axes.titlecolor':'#000000',
    'xtick.color':'#000000','ytick.color':'#000000',
    'axes.edgecolor':'#000000','axes.linewidth':.65,
    'xtick.major.width':.65,'ytick.major.width':.65,
    'xtick.major.size':2.5,'ytick.major.size':2.5,
    'xtick.direction':'out','ytick.direction':'out',
    'xtick.labelsize':7.6,'ytick.labelsize':7.6,
    'axes.labelsize':8.5,'axes.titlesize':9,
    'legend.fontsize':7.7,'legend.frameon':False,
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'savefig.facecolor':'white','figure.facecolor':'white',
    'axes.unicode_minus':True,
})
PAL=json.loads((FDATA/'method_palette.json').read_text())
if 'colors' in PAL:PAL=PAL['colors']
ALIASES={'Static':'Static','ACI':'ACI','AgACI':'AgACI','EnbPI_RH':'EnbPI-RH','TunedSingleConformal':'TSC','EqualEndpointEnsemble':'EEE'}
METHODS=list(ALIASES)
TEAL='#177E89';PURPLE='#8865A2';LOSS='#C68865';GRID='#DFE3E5'
PRICES=['R01','R02','R05P6179775281','R10','R20']
PRICE_LABELS=['1','2','20/3.56','10','20']
summary=pd.read_csv(DATA/'distribution_summary.csv').set_index('price_id')
allrow=summary.loc['ALL']
hfile=np.load(DATA/'histograms.npz');edges=hfile['edges'];x=(edges[1:]+edges[:-1])/2;dx=edges[1]-edges[0]
source_curves=[];qa={}

def prettify(ax,grid=True):
    ax.spines[['top','right']].set_visible(False)
    ax.set_axisbelow(True)
    if grid:ax.grid(color=GRID,lw=.35,zorder=0)

def title(ax,label,text):
    ax.text(0,1.04,label,transform=ax.transAxes,ha='left',va='bottom',fontweight='bold',fontsize=10)
    ax.text(.063,1.04,text,transform=ax.transAxes,ha='left',va='bottom',fontsize=9.2)

def fmt_tick(v,pos):
    return ('−' if v<0 else '')+f'{abs(v):g}'

W,H=183,174
fig=plt.figure(figsize=(W/25.4,H/25.4))

# Two-dimensional density and marginal histograms.
# Condition means are normalized by the corresponding fixed method's ERRF,
# so both component axes have the same interpretable percentage scale.
main=fig.add_axes([.073,.477,.405,.426])
top=fig.add_axes([.073,.910,.405,.047],sharex=main)
side=fig.add_axes([.485,.477,.054,.426],sharey=main)
raw=pd.read_parquet(SRC/'condition_means.parquet')
keys=['zone','price_id','predictor','horizon_steps','target_coverage']
c=raw[raw.method.eq('CLARA')].set_index(keys)
bins=np.arange(-50,40.00001,1)
gridx=(bins[1:]+bins[:-1])/2;xx,yy=np.meshgrid(gridx,gridx)
fixed_rows=[];marg_rows=[]
for a in METHODS:
    b=raw[raw.method.eq(a)].set_index(keys).reindex(c.index)
    vx=(100*(c.capacity-b.capacity)/b.ERRF).to_numpy()
    vy=(100*(c.exceedance-b.exceedance)/b.ERRF).to_numpy()
    assert len(vx)==11000 and np.isfinite(vx).all() and np.isfinite(vy).all()
    assert vx.min()>bins[0] and vx.max()<bins[-1] and vy.min()>bins[0] and vy.max()<bins[-1]
    col=PAL[ALIASES[a]]
    hist=np.histogram2d(vx,vy,bins=(bins,bins))[0].T
    density=gaussian_filter(hist.astype(float),1.35,mode='constant')
    ranked=np.sort(density.ravel())[::-1];cum=np.cumsum(ranked)/ranked.sum()
    lev=[float(ranked[np.searchsorted(cum,p)]) for p in [.85,.5]]
    # Every observation appears in a rasterized background cloud; contours are
    # descriptive mass regions, not confidence intervals.
    main.scatter(vx,vy,s=.7,c=col,alpha=.085,linewidths=0,rasterized=True,zorder=2)
    main.contourf(xx,yy,density,levels=[lev[0],density.max()+1],colors=[col],alpha=.10,zorder=3)
    main.contour(xx,yy,density,levels=lev,colors=[col],linewidths=[.6,1.05],zorder=4)
    hx=np.histogram(vx,bins=bins)[0]/len(vx);hy=np.histogram(vy,bins=bins)[0]/len(vy)
    top.stairs(hx,bins,color=col,lw=.8,fill=True,alpha=.38)
    side.stairs(hy,bins,color=col,lw=.8,fill=True,alpha=.38,orientation='horizontal')
    main.scatter(vx.mean(),vy.mean(),marker='o',s=25,facecolor=col,edgecolor='white',lw=.6,zorder=6)
    q=c.index.to_frame(index=False);q['baseline']=ALIASES[a];q['capacity_change_pct']=vx;q['exceedance_change_pct']=vy;fixed_rows.append(q)
    for k in range(len(gridx)):marg_rows.append(dict(method=ALIASES[a],bin_center=gridx[k],capacity_probability=hx[k],exceedance_probability=hy[k]))

main.set(xlim=(-50,40),ylim=(-50,40),xticks=[-40,-20,0,20,40],yticks=[-40,-20,0,20,40])
main.xaxis.set_major_formatter(FuncFormatter(fmt_tick));main.yaxis.set_major_formatter(FuncFormatter(fmt_tick))
prettify(main)
main.axhline(0,color='black',lw=.65,ls=(0,(3,3)),zorder=1)
main.axvline(0,color='black',lw=.65,ls=(0,(3,3)),zorder=1)
main.plot([-30,30],[30,-30],color='black',lw=.9,zorder=5)
main.text(-28,-22,'Lower total cost',fontsize=8,ha='center',va='center')
main.text(14,19,'Higher total cost',fontsize=8,rotation=-43.5,ha='center',va='center')
main.set_xlabel('Capacity cost change (%)',labelpad=3)
main.set_ylabel('Exceedance cost change (%)',labelpad=3)
top.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
side.tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
for ax in [top,side]:
    for s in ax.spines.values():s.set_visible(False)
fig.text(.073,.985,'a',fontweight='bold',fontsize=10,va='top')
fig.text(.097,.985,'Conditional cost trade-offs',fontsize=9.2,va='top')
legend=[Line2D([0],[0],color=PAL[ALIASES[a]],lw=1.3,marker='o',markersize=3.5,label=ALIASES[a]) for a in METHODS]
main.legend(handles=legend,ncol=2,loc='lower left',bbox_to_anchor=(.035,.035),columnspacing=1.1,handlelength=1.1,handletextpad=.5,borderaxespad=0)

# (b) A signed marginal distribution. The exact zero point mass is reported
# separately rather than smoothed into either economically signed group.
cart=fig.add_axes([.617,.477,.333,.426])
pos=sum(hfile[p+'_cart_positive'] for p in PRICES)/5
neg=sum(hfile[p+'_cart_negative'] for p in PRICES)/5
share=allrow.cart_share
sigma=.055/dx
dp=gaussian_filter1d(pos,sigma,mode='constant')/(share*dx)
dn=gaussian_filter1d(neg,sigma,mode='constant')/(share*dx)
# Reflect the kernel at zero within each sign, preserving each side's mass.
mid=len(x)//2
dp[mid:]+=dp[:mid][::-1];dp[:mid]=0
dn[:mid]+=dn[mid:][::-1];dn[mid:]=0
for v,d,col in [(x[x<0],dn[x<0],LOSS),(x[x>0],dp[x>0],TEAL)]:
    cart.fill_between(v,0,d,color=col,alpha=.48,lw=0)
    cart.plot(v,d,color=col,lw=1.0)
    source_curves.extend(dict(panel='b',group='loss' if col==LOSS else 'gain',price_id='ALL',x=float(xx0),density=float(dd)) for xx0,dd in zip(v,d))
cart.set_xscale('asinh',linear_width=.35)
cart.set_xlim(-70,70);cart.set_ylim(0,max(dp.max(),dn.max())*1.12)
cart.xaxis.set_major_locator(FixedLocator([-30,-3,-.3,0,.3,3,30]))
cart.xaxis.set_major_formatter(FuncFormatter(fmt_tick));cart.minorticks_off()
prettify(cart)
cart.axvline(0,color='black',lw=.7,ls=(0,(3,3)))
cart.set_xlabel('CART cost − CLARA cost',labelpad=3)
cart.set_ylabel('Density',labelpad=3)
fig.text(.617,.985,'b',fontweight='bold',fontsize=10,va='top')
fig.text(.641,.985,'Decision gains and losses',fontsize=9.2,va='top')
cart.text(.03,.95,'Higher cost',ha='left',va='top',transform=cart.transAxes,fontsize=7.7)
cart.text(.97,.95,'Lower cost',ha='right',va='top',transform=cart.transAxes,fontsize=7.7)
cart.text(.03,.87,f'{allrow.cart_loss_pct:.2f}%\nMean loss {allrow.loss_per_defeat:.4f}',ha='left',va='top',transform=cart.transAxes,fontsize=7.7,linespacing=1.5)
cart.text(.97,.87,f'{allrow.cart_win_pct:.2f}%\nMean gain {allrow.gain_per_win:.4f}',ha='right',va='top',transform=cart.transAxes,fontsize=7.7,linespacing=1.5)
fig.text(.7835,.934,f'Equal cost: {allrow.cart_tie_pct:.2f}%',ha='center',va='center',fontsize=7.7)

# (c) Directly reproduce the paired ridge structure in the author's PDF p12.
# Same disagreements, five price rows, no exploratory or uncertainty bonuses.
ridge=fig.add_axes([.105,.084,.875,.309])
ridge.set_xscale('asinh',linear_width=.35)
ridge.set_xlim(-80,70);ridge.set_ylim(-.2,5.05)
ridge.xaxis.set_major_locator(FixedLocator([-50,-10,-2,-.3,0,.3,2,10,50]))
ridge.xaxis.set_major_formatter(FuncFormatter(fmt_tick));ridge.minorticks_off()
ridge.set_yticks(range(5),PRICE_LABELS[::-1]);ridge.tick_params(axis='y',length=0,pad=8)
ridge.set_ylabel('Price ratio',labelpad=8)
ridge.yaxis.set_label_coords(-.078,.5)
ridge.set_xlabel('Cost-difference prediction error',labelpad=3)
prettify(ridge,False)
ridge.grid(axis='x',color=GRID,lw=.35,zorder=0)
ridge.axvline(0,color='black',lw=.7,ls=(0,(3,3)),zorder=7)
for j,pid in enumerate(PRICES):
    y0=4-j
    ridge.axhline(y0,color=GRID,lw=.4,zorder=0)
    for name,col,zorder in [('lin',PURPLE,2),('clara',TEAL,3)]:
        mass=hfile[pid+'_'+name+'_error']
        d=gaussian_filter1d(mass,.075/dx,mode='constant')/(mass.sum()*dx)
        curve=d/d.max()*.84
        ridge.fill_between(x,y0,y0+curve,color=col,alpha=.48,lw=0,zorder=zorder)
        ridge.plot(x,y0+curve,color=col,lw=.8,zorder=zorder+.1)
        source_curves.extend(dict(panel='c',group=name,price_id=pid,x=float(xx0),density=float(dd)) for xx0,dd in zip(x,d))
    r=summary.loc[pid]
    ridge.text(.82,y0+.39,f'{r.clara_RMSE:.2f}  /  {r.lin_RMSE:.2f}',transform=ridge.get_yaxis_transform(),fontsize=7.7,va='center')
fig.text(.073,.407,'c',ha='left',va='bottom',fontweight='bold',fontsize=10)
fig.text(.097,.407,'Estimating the cost of alternative choices',ha='left',va='bottom',fontsize=9.2)
ridge.text(.82,1.045,'RMSE: CLARA / LinUCB',transform=ridge.transAxes,fontsize=7.7,va='bottom')
ridge.legend(handles=[Patch(facecolor=TEAL,alpha=.55,label='CLARA'),Patch(facecolor=PURPLE,alpha=.55,label='LinUCB')],
             loc='lower left',bbox_to_anchor=(.53,1.02),ncol=2,handlelength=1,handletextpad=.45,columnspacing=1.0,borderaxespad=0)

fig.canvas.draw()
pd.concat(fixed_rows,ignore_index=True).to_csv(FIG/'panel_a_points.csv',index=False)
pd.DataFrame(marg_rows).to_csv(FIG/'panel_a_marginals.csv',index=False)
pd.DataFrame(source_curves).to_parquet(FIG/'density_curves.parquet',index=False)
qa.update({'status':'AWAITING_VISUAL_QA','width_mm':W,'height_mm':H,'font':'Times New Roman (author requirement)',
    'points_in_a':66000,'conditions_per_fixed_method':11000,'a_normalization':'Both component changes are divided by the same fixed-method mean ERRF within each matched prediction condition, then multiplied by 100.',
    'a_density':'One-percentage-point 2D bins, Gaussian smoothing sigma=1.35 bins, 50% and 85% density mass contours; all points are drawn, with no crop.',
    'b_density':'Signed weighted densities; Gaussian bandwidth 0.055 ERRF with reflection at zero. Exact ties shown separately. Full observed tails retained.',
    'c_density':'Gaussian bandwidth 0.075 ERRF, same bandwidth for both methods and every price, each curve peak-normalized for ridge display; full observed tails retained.',
    'bc_xscale':'asinh with linear_width=0.35; labels show original ERRF units. Nonlinear scale retains the center and full tails without a broken axis.',
    'statistics':'Descriptive distributions; no new significance claims. Exact means and RMSE use unsmoothed full data.',
    'no_random_subsampling':True,'all_text_black':all(t.get_color() in ['#000000','black',(0,0,0,1)] for t in fig.findobj(mpl.text.Text)),
})
fig.savefig(FIG/'Figure_06.png',dpi=400)
fig.savefig(FIG/'Figure_06_preview.png',dpi=300)
fig.savefig(FIG/'Figure_06.pdf',dpi=600)
fig.savefig(FIG/'Figure_06.svg',dpi=600)
fig.savefig(FIG/'Figure_06.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
(FIG/'figure_qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
plt.close(fig)
print(FIG/'Figure_06_preview.png')
