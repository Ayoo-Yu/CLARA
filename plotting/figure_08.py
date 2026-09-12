"""Reproduce manuscript Figure 8 from its numerical source data."""
from _fonts import preferred_font
FONT_FAMILY = preferred_font()
from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap

ROOT=Path(__file__).resolve().parents[1]
FDATA=ROOT/'results'/'figure_data'
OUT=ROOT/'outputs'/'figure08'
FIG=OUT; FIG.mkdir(parents=True,exist_ok=True)
OLD=FDATA/'figure08'/'expected'
SOURCE=FDATA/'figure08'/'means.csv'
data=pd.read_csv(SOURCE)
primary=['CLARA_online','TSC','LinUCB','CART','EEE','EnbPI-RH','ACI','Static','AgACI']
controls=['CLARA','Cost_update_only','Reliability_update_only']
methods=primary+controls
names={'CLARA_online':'CLARA','CLARA':'Frozen estimates','Cost_update_only':'Cost updates only',
       'Reliability_update_only':'Coverage updates only'}
pal=json.loads((FDATA/'method_palette.json').read_text())
colors={m:pal.get(names.get(m,m),'#A8B9BD') for m in methods}
colors.update(CLARA_online=pal['CLARA'],CLARA='#B0BABD',Cost_update_only='#B7A4C6',Reliability_update_only='#A8C9B8')
mpl.rcParams.update({
    'font.family':FONT_FAMILY,'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9,
    'xtick.labelsize':8,'ytick.labelsize':8.5,'legend.fontsize':8.3,
    'text.color':'black','axes.labelcolor':'black','axes.titlecolor':'black',
    'xtick.color':'black','ytick.color':'black','axes.edgecolor':'black',
    'axes.linewidth':.65,'xtick.major.width':.6,'ytick.major.width':.6,
    'xtick.major.size':2.3,'ytick.major.size':2.3,'xtick.direction':'out','ytick.direction':'out',
    'svg.fonttype':'none','pdf.fonttype':42,'ps.fonttype':42,
    'figure.facecolor':'white','savefig.facecolor':'white'})

WIDTH_MM,HEIGHT_MM=183,160
fig=plt.figure(figsize=(7.20472440945,6.29921259843))
def axes_mm(x,top,w,h):
    return fig.add_axes([x/WIDTH_MM,1-(top+h)/HEIGHT_MM,w/WIDTH_MM,h/HEIGHT_MM])
def text_mm(x,top,s,**kw):
    return fig.text(x/WIDTH_MM,1-top/HEIGHT_MM,s,**kw)
def title(x,top,letter,label):
    text_mm(x,top,letter,fontsize=10,fontweight='bold',va='top')
    text_mm(x+4.2,top,label,fontsize=9.3,va='top')

# Identical data-column bounds and central labels in both rows.
XLEFT,XRIGHT,COLWIDTH=18,109,60
CENTRE_X,CENTRE_WIDTH=79,29
TOP,PLOTHEIGHT=14.5,61.5
left=axes_mm(XLEFT,TOP,COLWIDTH,PLOTHEIGHT)
right=axes_mm(XRIGHT,TOP,COLWIDTH,PLOTHEIGHT)
middle=axes_mm(CENTRE_X,TOP,CENTRE_WIDTH,PLOTHEIGHT)
y=np.r_[np.arange(9),np.arange(9,12)+.55]
for ax in [left,right,middle]:ax.set_ylim(y[-1]+.65,-.65)
middle.set_axis_off()
left.set_xlim(.82,0);right.set_xlim(0,45)
for ax in [left,right]:
    ax.set_axisbelow(True);ax.grid(axis='x',color='#DFE4E5',lw=.45)
    ax.spines[['top','right','left']].set_visible(False)
    ax.tick_params(axis='y',left=False,labelleft=False)
    ax.axhspan(-.40,.40,color='#EAF5F4',zorder=0)
    ax.axhline(8.78,color='#9AA4A7',lw=.5,ls=(0,(3,3)))
middle.axhspan(-.40,.40,color='#EAF5F4',zorder=0)
middle.axhline(8.78,color='#9AA4A7',lw=.5,ls=(0,(3,3)))
mu=data[(data.price_id=='ALL')&(data.zone_or_farm=='Pooled')].set_index('method')
for k,m in enumerate(methods):
    yy=y[k];weight='bold' if m=='CLARA_online' else 'normal'
    left.barh(yy,mu.loc[m,'ERRF'],height=.56,color=colors[m],alpha=.73,lw=0,zorder=2)
    right.barh(yy,100*mu.loc[m,'TUWR'],height=.56,color=colors[m],alpha=.73,lw=0,zorder=2)
    for farm,mark,dy in [('Farm_A','o',-.12),('Farm_B','^',.12)]:
        r=data[(data.price_id=='ALL')&(data.zone_or_farm==farm)&(data.method==m)].iloc[0]
        for ax,v in [(left,r.ERRF),(right,100*r.TUWR)]:
            ax.scatter(v,yy+dy,s=14,marker=mark,facecolors='white',edgecolors='black',linewidths=.6,zorder=4)
    middle.text(.5,yy,names.get(m,m),ha='center',va='center',fontweight=weight,fontsize=8.0 if m in controls else 8.5)
    left.text(-14/COLWIDTH,yy,f'{mu.loc[m,"ERRF"]:.4f}',transform=left.get_yaxis_transform(),
              ha='left',va='center',fontsize=8.4,fontweight=weight,clip_on=False)
    right.text(1+12/COLWIDTH,yy,f'{100*mu.loc[m,"TUWR"]:.2f}',transform=right.get_yaxis_transform(),
               ha='right',va='center',fontsize=8.4,fontweight=weight,clip_on=False)
left.set_xticks([0,.2,.4,.6,.8]);right.set_xticks([0,10,20,30,40])
left.set_xlabel('Mean ERRF',labelpad=3);right.set_xlabel('TUWR (%)',labelpad=3)
title(XLEFT,2.5,'a','Economic risk');title(XRIGHT,2.5,'b','Temporal undercoverage')
legend=[Patch(facecolor='#AAB8BC',edgecolor='none',label='Two-farm mean'),
        Line2D([],[],ls='',marker='o',mfc='white',mec='black',ms=3.7,label='Farm A'),
        Line2D([],[],ls='',marker='^',mfc='white',mec='black',ms=3.7,label='Farm B')]
fig.legend(handles=legend,loc='center',bbox_to_anchor=(.51,1-9.7/HEIGHT_MM),ncol=3,
           columnspacing=1.5,handlelength=1.1,handletextpad=.5,frameon=False,borderpad=0)

pids=['R01','R02','R05P6179775281','R10','R20']
price_labels=['1','2','20/3.56','10','20']
rows=data[(data.zone_or_farm=='Pooled')&data.method.isin(primary)&data.price_id.isin(pids)]
assert len(rows)==45
cost=rows.pivot(index='method',columns='price_id',values='ERRF').reindex(index=primary,columns=pids)
tw=rows.pivot(index='method',columns='price_id',values='TUWR').reindex(index=primary,columns=pids)*100
assert np.isfinite(cost.to_numpy()).all() and np.all(cost.to_numpy()>0)
gap=100*(cost/cost.min(axis=0)-1)
maximum=float(np.ceil(gap.to_numpy().max()/5)*5)
costmap=LinearSegmentedColormap.from_list('economic_cost',['#FFFFFF','#F2D5BF','#D99B68'])
twmap=LinearSegmentedColormap.from_list('undercoverage',['#FFFFFF','#D7E8E9','#79ADB3'])
MTOP,MHEIGHT=96.5,37.8
axc=axes_mm(XLEFT,MTOP,COLWIDTH,MHEIGHT)
axt=axes_mm(XRIGHT,MTOP,COLWIDTH,MHEIGHT)
mlabel=axes_mm(CENTRE_X,MTOP,CENTRE_WIDTH,MHEIGHT)
mlabel.set_ylim(8.5,-.5);mlabel.set_axis_off()
ims=[]
for ax,values,cmap,vmax in [(axc,gap,costmap,maximum),(axt,tw,twmap,40)]:
    ims.append(ax.imshow(values.to_numpy(),aspect='auto',cmap=cmap,vmin=0,vmax=vmax,interpolation='nearest'))
    ax.set_xticks(range(5),price_labels);ax.set_yticks([])
    ax.tick_params(length=0,pad=3)
    ax.set_xlabel('Exceedance-to-capacity price ratio',fontsize=8.2,labelpad=3)
    ax.set_xticks(np.arange(-.5,5,1),minor=True);ax.set_yticks(np.arange(-.5,9,1),minor=True)
    ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='minor',bottom=False,left=False)
    for spine in ax.spines.values():spine.set_visible(False)
    for i in range(9):
        for j in range(5):
            ax.text(j,i,f'{values.iloc[i,j]:.1f}',ha='center',va='center',fontsize=8.3,
                    fontweight='bold' if i==0 else 'normal')
for i,m in enumerate(primary):
    mlabel.text(.5,i,names.get(m,m),ha='center',va='center',fontsize=8.5,
                fontweight='bold' if i==0 else 'normal')
title(XLEFT,89.7,'c','Cost across prices')
title(XRIGHT,89.7,'d','Undercoverage across prices')
for x,im,label in [(XLEFT,ims[0],'ERRF above the price-specific minimum (%)'),(XRIGHT,ims[1],'TUWR (%)')]:
    cb=fig.colorbar(im,cax=axes_mm(x,146.5,COLWIDTH,1.6),orientation='horizontal')
    cb.set_ticks([0,10,20,30,40]);cb.set_label(label,fontsize=8.0,labelpad=2.5)
    cb.ax.tick_params(labelsize=7.8,length=2,pad=2);cb.outline.set_visible(False)

# Reject numerical changes and geometry drift before exporting.
for frame,name in [(gap,'Figure8_cost_by_price.csv'),(tw,'Figure8_TUWR_by_price.csv')]:
    old=pd.read_csv(OLD/name,index_col=0)
    pd.testing.assert_frame_equal(frame,old,check_names=False,rtol=1e-13,atol=1e-13)
    frame.to_csv(FIG/name,encoding='utf-8-sig')
src=mu.loc[methods].reset_index()
pd.testing.assert_frame_equal(src,pd.read_csv(OLD/'Figure8_summary_data.csv'),check_names=False,rtol=1e-13,atol=1e-13)
src.to_csv(FIG/'Figure8_summary_data.csv',index=False,encoding='utf-8-sig')
for a,b in [(left,axc),(right,axt)]:
    assert np.allclose([a.get_position().x0,a.get_position().width],[b.get_position().x0,b.get_position().width])
assert np.allclose([left.get_position().y0,left.get_position().height],
                   [right.get_position().y0,right.get_position().height])
fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[]
for text in fig.findobj(mpl.text.Text):
    if text.get_visible() and text.get_text():
        box=text.get_window_extent(renderer)
        if box.x0<-.5 or box.y0<-.5 or box.x1>fig.bbox.x1+.5 or box.y1>fig.bbox.y1+.5:
            outside.append(text.get_text())
assert not outside,outside
fig.savefig(FIG/'Figure8_External_Adaptation.pdf')
fig.savefig(FIG/'Figure8_External_Adaptation.svg')
fig.savefig(FIG/'Figure8_External_Adaptation.png',dpi=600)
fig.savefig(FIG/'Figure8_External_Adaptation.tiff',dpi=600,pil_kwargs={'compression':'tiff_lzw'})
fig.savefig(FIG/'Figure8_preview.png',dpi=300)
(OUT/'figure_qa.json').write_text(json.dumps({
    'status':'PASS','source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'matches_published_summary':True,'primary_methods':9,'controls':3,'prices':5,'farms':2,
    'width_mm':WIDTH_MM,'height_mm':HEIGHT_MM,'previous_height_mm':190,
    'height_reduction_pct':100*(190-HEIGHT_MM)/190,'row_outlines':0,
    'common_column_geometry':True,'text_outside_canvas':outside,
    'font_preference':'Times New Roman, explicit author instruction; overrides generic font default'
},indent=2),encoding='utf-8')
print('Figure 8 exported; data unchanged; shared column alignment verified.',flush=True)
