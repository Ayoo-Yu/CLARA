"""Complete decision records in the approved compact grouped-bar layout."""
from pathlib import Path
import json
import hashlib
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import fitz

SCRIPT_DIR=Path(__file__).resolve().parent
REPO=SCRIPT_DIR.parents[1]
HERE=REPO/'results/current_figure_data/figure07'
OUTPUT=REPO/'outputs/current/figure07'
OUTPUT.mkdir(exist_ok=True,parents=True)
DATA = HERE / 'figure7_source_data.csv'
META = HERE / 'figure7_case_metadata.csv'
OUT = OUTPUT / 'Figure_07'
width_mm = 190
height_mm = 100
METHODS = ['Static', 'ACI', 'AgACI', 'EnbPI-RH', 'TSC', 'EEE']
COLORS = {'Static':'#A2A7AC', 'ACI':'#469A68', 'AgACI':'#B79328',
          'EnbPI-RH':'#68AEBB', 'TSC':'#8557A6', 'EEE':'#377EB8'}
ORANGE = '#D96F21'
RUST = '#C35C45'
mpl.rcParams.update({
    'font.family': 'serif', 'font.serif': ['Times New Roman'],
    'font.size': 9, 'mathtext.fontset': 'stix',
    'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.linewidth': .55, 'hatch.linewidth': .45,
    'axes.edgecolor': 'black', 'text.color': 'black',
    'xtick.direction': 'in', 'ytick.direction': 'in',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def ink(fill):
    c = np.array(mpl.colors.to_rgb(fill))
    linear = np.where(c <= .04045, c/12.92, ((c+.055)/1.055)**2.4)
    lum = linear @ np.array([.2126, .7152, .0722])
    return 'black' if (lum+.05)/.05 >= 1.05/(lum+.05) else 'white'


def main():
    data = pd.read_csv(DATA, float_precision='round_trip')
    meta = pd.read_csv(META).set_index('case_id')
    assert len(data) == 12 and not data.duplicated(['case_id','action']).any()
    assert (data.lower <= data.target).all() and (data.target <= data.upper).all()
    assert data.tuwr_point.le(data.tuwr_limit).all()
    fig = plt.figure(figsize=(width_mm/25.4, height_mm/25.4), dpi=300)
    page = fig.add_axes([0,0,1,1])
    page.set(xlim=(0,width_mm), ylim=(0,height_mm))
    page.axis('off')
    text_artists = []

    def text(x,y,s,**kwargs):
        options = dict(va='center', ha='left', fontsize=9, color='black')
        options.update(kwargs)
        t = page.text(x,y,s,**options)
        text_artists.append(t)
        return t

    def line(x0,x1,y0,y1=None,lw=.55):
        page.plot([x0,x1],[y0,y0 if y1 is None else y1], color='black', lw=lw,
                  solid_capstyle='butt')

    plot_y, plot_h = 18, 65
    left, split, right = 46, 97, 164
    cost = fig.add_axes([left/width_mm, plot_y/height_mm, (split-left)/width_mm, plot_h/height_mm])
    gap = 1.2
    interval = fig.add_axes([(split+gap)/width_mm, plot_y/height_mm, (right-split-gap)/width_mm, plot_h/height_mm])
    for ax in [cost, interval]:
        ax.set_ylim(11.5,-.5)
        ax.set_yticks([])
        ax.xaxis.set_ticks_position('top')
        ax.xaxis.set_label_position('top')
        ax.tick_params(axis='x', length=2.1, width=.55, pad=2, labelsize=8.7)
        ax.axhline(5.5, color='black', lw=.65, zorder=7)
        for spine in ax.spines.values():
            spine.set_color('black'); spine.set_linewidth(.55)
    # A narrow white gutter separates the two independent scales, as in the reference.
    cost.set_xlim(2.5,0)
    cost.set_xticks([2.5,2,1.5,1,.5,0])
    cost.set_xticklabels(['2.5','2','1.5','1','0.5','0'])
    interval.set_xlim(0,1)
    interval.set_xticks([.2,.4,.6,.8,1])
    interval.set_xticklabels(['0.2','0.4','0.6','0.8','1.0'])
    text((left+split)/2,91.8, r'Historical cost score $Q$', ha='center', fontsize=10)
    text((split+right)/2,91.8,'Prediction interval (p.u.)',ha='center',fontsize=10)
    text(176,94,'Historical',ha='center',fontsize=9.2)
    text(176,89.6,'coverage (%)',ha='center',fontsize=9.2)
    text(176,13.7,'Required ≥ 88%',ha='center',fontsize=8.6)

    # Nested group columns and thin separators reproduce the reference geometry.
    coverage_right = 188
    for x in [4,13,22,left,coverage_right]:
        line(x,x,plot_y,plot_y+plot_h)
    for yy in [plot_y,plot_y+plot_h/2,plot_y+plot_h]:
        line(4,coverage_right,yy,lw=.65 if yy==plot_y+plot_h/2 else .55)

    rows = []
    for case_i,(case,title) in enumerate([('ordinary','Ordinary'),('ramp','Predicted ramp')]):
        group = data.loc[data.case_id.eq(case)].set_index('action').loc[METHODS]
        selected = group.index[group.chosen][0]
        assert group.loc[group.pass_new,'risk_score'].idxmin() == selected
        assert group.chosen.sum() == 1
        center_y = plot_y+plot_h*(.75 if case_i==0 else .25)
        date = pd.to_datetime(meta.loc[case,'issue_timestamp'])
        text(8.5,center_y,date.strftime('%d %b %Y\n%H:%M UTC'),ha='center',rotation=90,fontsize=8)
        text(17.5,center_y,title,ha='center',rotation=90,fontsize=9.3)
        for i,action in enumerate(METHODS):
            r = group.loc[action]
            y = case_i*6+i
            py = plot_y+plot_h*(1-(y+.5)/12)
            chosen, failed = bool(r.chosen), not bool(r.pass_new)
            col = COLORS[action]
            edge = 'black' if chosen or failed else col
            linewidth = 1.1 if chosen else .25
            hatch = '///' if failed else None
            cost.barh(y,r.risk_score,height=.79, color=col, edgecolor=edge,
                      linewidth=linewidth, hatch=hatch, zorder=3)
            # Failure is stated in the coverage column; keep interval bounds unobscured.
            interval.barh(y,r.upper-r.lower,left=r.lower,height=.79,color=col,
                          edgecolor='black' if chosen else col,
                          linewidth=.75 if chosen else .25,zorder=3)
            # Exact numbers inside bars, as in the approved reference.
            ct = cost.text(.10,y,f'{r.risk_score:.3f}',ha='right',va='center',
                           fontsize=8.8,color=ink(col),fontweight='bold' if chosen else 'normal',zorder=6,
                           bbox={'facecolor':col,'edgecolor':'none','pad':.2} if failed else None)
            text_artists.append(ct)
            text(23.5,py,action,fontsize=9.1,fontweight='bold' if chosen else 'normal')
            if failed:
                page.add_patch(Rectangle((right,py-plot_h/24),coverage_right-right,plot_h/12,
                                         facecolor='#F8ECE7',edgecolor='none',zorder=.5))
            value = f'{100*r.coverage_point:.2f}'
            text(176,py,value,ha='center',fontsize=9.1,
                 fontweight='bold' if chosen or failed else 'normal',color=RUST if failed else 'black')
            rows.append({'case':case,'action':action,'Q':float(r.risk_score),
                         'bounds':[float(r.lower),float(r.upper)],'coverage':float(r.coverage_point),
                         'selected':chosen,'failed':failed})
        observed = float(group.target.iloc[0])
        interval.plot([observed,observed],[case_i*6-.44,case_i*6+5.44],color=ORANGE,lw=.8,zorder=9)

    handles = [Patch(facecolor='white',edgecolor='black',linewidth=1.1),
               Patch(facecolor=COLORS['Static'],edgecolor='black',hatch='///',linewidth=.25),
               Line2D([],[],color=ORANGE,linewidth=.8)]
    labels = ['Selected', 'Fails coverage', 'Later observation: 0.403 / 0.087 p.u.']
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.51,.038),ncol=3,
               frameon=False,fontsize=8.7,handlelength=1.6,handletextpad=.5,columnspacing=1.6)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for t in fig.findobj(plt.Text):
        if not t.get_visible() or not t.get_text():
            continue
        b = t.get_window_extent(renderer)
        assert b.x0 >= -1 and b.y0 >= -1 and b.x1 <= fig.bbox.width+1 and b.y1 <= fig.bbox.height+1, t.get_text()
    fig.savefig(str(OUT)+'.svg',facecolor='white')
    fig.savefig(str(OUT)+'.pdf',facecolor='white')
    fig.savefig(str(OUT)+'.png',dpi=600,facecolor='white')
    fig.savefig(str(OUT)+'.tiff',dpi=600,facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    fig.savefig(OUTPUT/'preview.png',dpi=300,facecolor='white')
    d = fitz.open(str(OUT)+'.pdf')
    p = d[0]
    pdf_text = p.get_text()
    assert all(f'{r["Q"]:.3f}' in pdf_text and f'{r["coverage"]*100:.2f}' in pdf_text for r in rows)
    p.get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(OUTPUT/'pdf_preview.png')
    qa = {'status':'PASS','records':12,'excluded_rows':0,'plotted':rows,
          'data_sha256':hashlib.sha256(DATA.read_bytes()).hexdigest(),
          'size_mm':[width_mm,height_mm], 'rows_order':METHODS,
          'font_override':'Author explicitly requires existing Times New Roman figure typography.',
          'reference_adaptation':'Layout and graphical conventions only; no invented additive decomposition.',
          'all_later_observations_inside_all_candidate_intervals':True,
          'all_candidates_pass_tuwr':True}
    (OUTPUT/'qa.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'status':'PASS','output':str(OUT)},ensure_ascii=False))
    plt.close(fig)


if __name__=='__main__':
    main()
