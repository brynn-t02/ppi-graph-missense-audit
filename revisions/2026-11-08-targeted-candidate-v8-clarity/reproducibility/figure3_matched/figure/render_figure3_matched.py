#!/usr/bin/env python3
"""Update only Figure 3B–F from the frozen matched-ensemble analysis.

Six-panel geometry, point sizes and colors follow the original Figure 3 v1.
Panel A is copied as its exact original vector operators, with a pixel audit.
The four displayed cells and all 53 histogram cells were frozen before analysis.
"""
from pathlib import Path
import argparse, hashlib, json, math, shutil, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, RectangleObject
from PIL import Image

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEFAULT_SOURCE=ROOT/'outputs/BIBM_workshop_targeted_candidate_v5_text_20261108/source/figure_drafts/Figure3_topology_context_v1_accepted_clarified.pdf'
DEFAULT_FONT=ROOT/'outputs/BIBM_workshop_font_clean_20261025/source/resources/fonts'
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean_axis(ax):
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
def render(pdf,png):
    subprocess.run(['pdftoppm','-singlefile','-r','300','-png',str(pdf),str(png.with_suffix(''))],check=True,capture_output=True)
def read(p): return pd.read_csv(p,float_precision='round_trip')

COL_AUC='#3182BD'; COL_AP='#9ECAE1'; COL_POS=COL_AUC; COL_NEG='#D95F0E'; COL_GRAY='#7A7A7A'; COL_TEXT='#252525'
ORDER=['mid_kcore__high_betweenness','mid_kcore__mid_clustering','high_neighbor_risk__mid_kcore','high_kcore__low_clustering']
LABELS=['mid k-core\nhigh betweenness','mid k-core\nmid clustering','high risk\nmid k-core','high k-core\nlow clustering']
AXES_PT=[(50.7,296.886155,224.316,161.862595),(387.174,296.886155,224.316,161.862595),(723.648,296.886155,224.316,161.862595),(50.7,34.66875,224.316,161.862595),(387.174,34.66875,224.316,161.862595),(723.648,34.66875,224.316,161.862595)]

def paired_delta_bars(ax,title,df,labels):
    ax.set_title(title,loc='left',fontweight='bold')
    x=np.arange(len(labels)); w=.34
    ax.bar(x-w/2,df.delta_auc,w,color=COL_AUC,edgecolor='black',linewidth=.4,label='Delta AUC')
    ax.bar(x+w/2,df.delta_auprc,w,color=COL_AP,edgecolor='black',linewidth=.4,label='Delta AUPRC')
    ax.axhline(0,color=COL_GRAY,lw=.8,ls='--')
    ax.set_xticks(x); ax.set_xticklabels(labels,rotation=20,ha='right')
    ax.set_ylabel('PPI-GCN - ESM raw'); ax.set_ylim(-.04,.115)
    ax.legend(frameon=False,loc='upper left'); clean_axis(ax)

def preserve_original_A(new_pdf,source,out_pdf):
    old=PdfReader(source).pages[0]; new=PdfReader(new_pdf).pages[0]
    w,h=float(old.mediabox.width),float(old.mediabox.height)
    assert list(old.mediabox)==list(new.mediabox)
    rect=[0,h-260,340,h-45]
    cs=ContentStream(old.get_contents(),old.pdf); hits=[]
    for i,(args,op) in enumerate(cs.operations):
        if op==b'TJ':
            text=''.join(v.decode('utf-16be') if isinstance(v,bytes) else v for v in args[0] if isinstance(v,(str,bytes)))
            if text=='A  ESM representations weakly align with PPI topology': hits.append(i)
    assert len(hits)==1
    end=hits[0]+3
    assert cs.operations[end-2][1]==b'ET' and cs.operations[end-1][1]==b'Q'
    kept=cs.operations[:end]
    depth=sum(1 if op==b'q' else -1 if op==b'Q' else 0 for args,op in kept)
    cs.operations=kept+[([],b'Q') for _ in range(depth)]
    old.replace_contents(cs)
    old.trimbox=RectangleObject(rect); old.cropbox=RectangleObject(rect)
    new.merge_page(old)
    wr=PdfWriter(); wr.add_page(new)
    wr.add_metadata({'/Title':'Exploratory topology-context analysis from matched model ensembles','/Subject':'Original panel A; frozen matched-ensemble predictions in B–F; paired protein-bootstrap intervals in F.'})
    with out_pdf.open('wb') as f: wr.write(f)
    return rect,len(cs.operations)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--analysis-dir',type=Path,default=HERE.parent/'analysis')
    ap.add_argument('--source-pdf',type=Path,default=DEFAULT_SOURCE);ap.add_argument('--font-dir',type=Path,default=DEFAULT_FONT)
    args=ap.parse_args(); data=args.analysis_dir/'data/processed'
    for name in ['DejaVuSans.ttf','DejaVuSans-Bold.ttf']:
        font_manager.fontManager.addfont(str(args.font_dir/name))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':7.8,'ytick.labelsize':7.8,'legend.fontsize':7.2,'figure.dpi':160,'savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.8})
    paths={
      'B':data/'topology_gain_analysis_exp26B/topology_stratified_metrics.csv',
      'C_betweenness':data/'betweenness_gain_analysis_exp30/betweenness_stratified_metrics.csv',
      'C_clustering':data/'clustering_gain_analysis_exp31/clustering_stratified_metrics.csv',
      'D':data/'neighborhood_risk_enrichment_exp27/neighbor_risk_stratified_metrics.csv',
      'E':data/'topology_profile_joint_map_exp33/joint_gain_maps.csv',
      'F':args.analysis_dir/'table5_selected_original_four_cells.csv'}
    tables={k:read(p) for k,p in paths.items()}
    page=PdfReader(args.source_pdf).pages[0]; w,h=float(page.mediabox.width),float(page.mediabox.height)
    fig=plt.figure(figsize=(w/72,h/72)); axes=[fig.add_axes([x/w,y/h,aw/w,ah/h]) for x,y,aw,ah in AXES_PT]
    axes[0].axis('off')
    # Baseline and text anchor exactly follow the clarified original title.
    fig.text(.5,505.564125/h,'Exploratory PPI-GCN gains across network topology contexts',ha='center',va='baseline',fontsize=14,fontweight='bold')
    b=pd.concat([tables['B'].query("group_col=='degree_bucket'").sort_values('group'),tables['B'].query("group_col=='kcore_bucket'").sort_values('group')],ignore_index=True)
    paired_delta_bars(axes[1],'B  Degree and k-core gains are non-monotonic',b,['degree low','degree mid','degree high','k-core low','k-core mid','k-core high'])
    c=pd.concat([tables['C_betweenness'].sort_values('group'),tables['C_clustering'].sort_values('group')],ignore_index=True)
    paired_delta_bars(axes[2],'C  Gains across betweenness and clustering strata',c,['between low','between mid','between high','cluster low','cluster mid','cluster high'])
    # The original clarified C title starts at x=688pt to fit its long text.
    axes[2]._left_title.set_position(((688-723.648)/224.316,1.0))
    for frame in [b,c]:
        vals=frame[['delta_auc','delta_auprc']].to_numpy(); assert vals.min()>=-.04 and vals.max()<=.115
    d=tables['D'].sort_values('group');ax=axes[3];x=np.arange(3); bw=.34
    ax.set_title('D  Neighbour-risk strata show local-context dependence',loc='left',fontweight='bold')
    ax.bar(x-bw/2,d.delta_auc,bw,color=COL_AUC,edgecolor='black',linewidth=.4,label='Delta AUC')
    ax.bar(x+bw/2,d.delta_auprc,bw,color=COL_AP,edgecolor='black',linewidth=.4,label='Delta AUPRC')
    ax.axhline(0,color=COL_GRAY,lw=.8,ls='--');ax.set_xticks(x);ax.set_xticklabels(['risk low','risk mid','risk high'])
    ax.set_ylabel('PPI-GCN - ESM raw');ax.set_ylim(-.015,.075)
    for xi,pos in zip(x,d.pos_rate): ax.text(xi,-.010,f'pos {pos*100:.1f}%',ha='center',va='top',fontsize=6.8,color=COL_GRAY)
    ax.legend(frameon=False,loc='upper right');clean_axis(ax)
    assert d[['delta_auc','delta_auprc']].to_numpy().min()>=-.015 and d[['delta_auc','delta_auprc']].to_numpy().max()<=.075
    joint=tables['E'];assert len(joint)==53 and (joint.n>=50).all()
    vals=joint.delta_auc.to_numpy();bins=np.linspace(-.075,.155,13);oldbins=bins.copy();step=bins[1]-bins[0]
    while vals.min()<bins[0]: bins=np.r_[bins[0]-step,bins]
    while vals.max()>bins[-1]: bins=np.r_[bins,bins[-1]+step]
    counts,edges=np.histogram(vals,bins=bins);assert counts.sum()==53
    cent=(edges[1:]+edges[:-1])/2;ax=axes[4]
    ax.set_title('E  Joint topology cells contain both positive and negative gains',loc='left',fontweight='bold')
    ax.bar(cent,counts,width=np.diff(edges)*.92,color=[COL_POS if v>=0 else COL_NEG for v in cent],edgecolor='black',linewidth=.35,alpha=.78)
    ax.axvline(0,color=COL_GRAY,lw=.8,ls='--');ax.set_ylabel('Number of joint cells');ax.set_xlabel('Delta AUC')
    ax.set_xlim(edges[0],edges[-1]);ax.set_ylim(0,max(counts)+2)
    ax.text(edges[-1]-.004,max(counts)+1.35,'53 tested cells',ha='right',va='top',fontsize=7,color=COL_TEXT);clean_axis(ax)
    f=tables['F'].set_index('cell_name').loc[ORDER].reset_index();assert list(f.display_id)==['J0','J1','J2','J3'];assert(f.bootstrap_unit=='protein').all() and(f.bootstrap_repeats==2000).all()
    ax=axes[5];ax.set_title('F  Selected exploratory joint-cell profiles',loc='left',fontweight='bold')
    yy=np.arange(4)[::-1]
    for yi,(_,r) in zip(yy,f.iterrows()):
        color=COL_POS if r.point_delta_auc>=0 else COL_NEG
        ax.errorbar(r.point_delta_auc,yi,xerr=[[r.point_delta_auc-r.boot_delta_auc_ci_lower],[r.boot_delta_auc_ci_upper-r.point_delta_auc]],fmt='o',color=color,ecolor=color,elinewidth=1.8,capsize=3,markersize=4.5)
    ax.axvline(0,color=COL_GRAY,lw=.8,ls='--');ax.set_yticks(yy);ax.set_yticklabels(LABELS)
    ax.set_xlabel('Delta AUC with 95% protein-bootstrap interval');ax.set_xlim(-.225,.225);ax.set_xticks([-.2,-.1,0,.1,.2]);ax.set_ylim(-.6,3.6);clean_axis(ax)
    assert f.boot_delta_auc_ci_lower.min()>-.225 and f.boot_delta_auc_ci_upper.max()<.225
    raw=HERE/'Figure3_matched_render_only.pdf';out=HERE/'Figure3_topology_context_matched_ensembles.pdf';png=out.with_suffix('.png')
    # Fixed physical canvas: no bbox_inches=tight and no scaling of fonts/axes.
    fig.savefig(raw);plt.close(fig)
    rect,ops=preserve_original_A(raw,args.source_pdf,out);render(out,png)
    original_png=HERE/'Figure3_original_pixel_audit.png';render(args.source_pdf,original_png)
    a=np.asarray(Image.open(original_png).convert('RGB'));z=np.asarray(Image.open(png).convert('RGB'));assert a.shape==z.shape
    x0,y0,x1,y1=rect; c0=math.ceil(x0*300/72)+2;c1=math.floor(x1*300/72)-2;r0=math.ceil((h-y1)*300/72)+2;r1=math.floor((h-y0)*300/72)-2
    ac=a[r0:r1,c0:c1];zc=z[r0:r1,c0:c1];changed=int(np.any(ac!=zc,axis=2).sum());assert changed==0,changed
    fontcheck=shutil.which('pdffonts') or '/Users/teyian/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdffonts'
    fonts=subprocess.run([fontcheck,str(out)],check=True,text=True,capture_output=True).stdout
    assert not any(' no ' in line for line in fonts.splitlines()[2:]),fonts
    (HERE/'font_embedding_audit.txt').write_text(fonts)
    audit={'source_pdf':str(args.source_pdf),'source_sha256':sha(args.source_pdf),'output_sha256':sha(out),'png_sha256':sha(png),'input_sha256':{k:sha(v) for k,v in paths.items()},'canvas_pt':[w,h],'axes_pt':AXES_PT,'fonts_and_colors_follow_original':True,'all_fonts_embedded':True,'original_A_vector_ops':ops,'original_A_clip_pt':rect,'original_A_pixel_bounds':[c0,r0,c1,r1],'original_A_changed_pixels_300dpi':changed,'original_A_pixel_sha256':hashlib.sha256(ac.tobytes()).hexdigest(),'new_A_pixel_sha256':hashlib.sha256(zc.tobytes()).hexdigest(),'original_four_selected_cells':ORDER,'all_53_cells_histogram_count':int(counts.sum()),'histogram_original_bins':oldbins.tolist(),'histogram_new_bins':edges.tolist(),'axis_changes':{'B':None,'C':None,'D':None,'E':'One original-width bin added left because minimum delta AUC=-0.0767215 is below old -0.075 boundary. Counts updated across all 53 frozen cells.','F':'Limits expanded from [-0.075,0.175] to [-0.225,0.225] to show full new protein-bootstrap intervals and leave room for endpoint tick labels.'},'panel_F_interval':'2000 paired within-cell protein-clustered bootstrap resamples; conditional on fitted ensemble; unadjusted for subgroup selection/multiplicity','no_hidden_old_B_to_F_text':True,'no_training_or_prediction_recomputation':True}
    paneldata={'B':b.to_dict('records'),'C':c.to_dict('records'),'D':d.to_dict('records'),'E_cells':joint.to_dict('records'),'E_histogram':{'edges':edges.tolist(),'counts':counts.tolist()},'F':f.to_dict('records')}
    (HERE/'panel_data.json').write_text(json.dumps(paneldata,indent=2)+'\n')
    (HERE/'figure3_render_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    raw.unlink();original_png.unlink()
    print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
