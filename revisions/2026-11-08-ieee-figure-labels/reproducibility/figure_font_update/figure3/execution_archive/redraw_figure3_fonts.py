from pathlib import Path
import importlib.util,json,hashlib,os,copy
os.environ['MPLBACKEND']='Agg'
os.environ['MPLCONFIGDIR']='/private/tmp/ppi_fig3_mpl_cache'
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np
WORK=Path(__file__).resolve().parent
BASE=WORK.parents[2]
OUT=BASE/'outputs/BIBM_workshop_IEEE_figure_labels_20261108/source/figure_drafts'
LEGACY=Path('/Users/teyian/Documents/paper1/make_figure3_topology_context_v1.py')
REG='/System/Library/Fonts/Supplemental/Times New Roman.ttf'
BOLD='/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'
font_manager.fontManager.addfont(REG);font_manager.fontManager.addfont(BOLD)
spec=importlib.util.spec_from_file_location('legacy_figure3',LEGACY);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':300})
fig,aa=plt.subplots(2,3,figsize=(503.79/72,345/72))
axs=aa.ravel()
for f,ax in zip([old.panel_a,old.panel_b,old.panel_c,old.panel_d,old.panel_e,old.panel_f],axs):f(ax)
def science_snapshot():
 out=[]
 for ax in axs:
  out.append({'xlim':list(ax.get_xlim()),'ylim':list(ax.get_ylim()),'bars':[{'xy':list(p.get_xy()),'width':p.get_width(),'height':p.get_height(),'rgba':list(p.get_facecolor())} for p in ax.patches], 'lines':[{'x':np.asarray(l.get_xdata()).tolist(),'y':np.asarray(l.get_ydata()).tolist(),'color':l.get_color()} for l in ax.lines], 'interval_segments':[np.asarray(c.get_segments()).tolist() for c in ax.collections if hasattr(c,'get_segments')]})
 return out
before=science_snapshot()
# Fixed placement in final-size PDF points. Data artists, limits and CI segments are untouched.
positions=[(37,192,122,105),(205,192,123,105),(374,192,122,105),(37,35,122,97),(205,35,110,97),(396,35,100,97)]
for ax,(x,y,w,h) in zip(axs,positions):
 ax.set_position([x/503.79,y/345,w/503.79,h/345])
 ax.tick_params(axis='both',labelsize=8,pad=2,length=2.5,width=.55)
 ax.xaxis.labelpad=3;ax.yaxis.labelpad=3
 for spine in ax.spines.values():spine.set_linewidth(.6)
# All numeric ticks reproduce the current figure's displayed values.
axs[0].set_yticks(np.arange(.855,.881,.005));axs[0].set_yticklabels([f'{v:.3f}' for v in np.arange(.855,.881,.005)])
for ax in axs[1:3]:ax.set_yticks(np.arange(-.04,.101,.02));ax.set_yticklabels([('−' if v<-.00001 else '')+f'{abs(v):.2f}' for v in np.arange(-.04,.101,.02)])
axs[3].set_yticks(np.arange(-.01,.071,.01));axs[3].set_yticklabels([('−' if v<-.00001 else '')+f'{abs(v):.2f}' for v in np.arange(-.01,.071,.01)])
axs[4].set_yticks(np.arange(0,15,2))
for ax in axs[4:]:ax.set_xticks([-.05,0,.05,.10,.15]);ax.set_xticklabels(['−0.05','0.00','0.05','0.10','0.15'])
# Split six categorical ticks into three levels plus centered group names.
for ax,names in [(axs[1],['degree','k-core']),(axs[2],['between','cluster'])]:
 ax.set_xticklabels(['low','mid','high']*2,rotation=0,ha='center')
 ax.text(1,-.255,names[0],transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
 ax.text(4,-.255,names[1],transform=ax.get_xaxis_transform(),ha='center',va='top',fontsize=8)
axs[0].set_xticklabels(['PPI\nneighbour','Random\npair','Within','Between'])
axs[0].text(2.5,-.19,'community',transform=axs[0].get_xaxis_transform(),ha='center',va='top',fontsize=8)
axs[0].texts[0].set_position((.5,.884));axs[0].texts[0].set_ha('center');axs[0].texts[0].set_text("Cliff's delta\n=0.124")
axs[0].texts[1].set_position((2.5,.884));axs[0].texts[1].set_ha('center');axs[0].texts[1].set_text('Delta mean\n=0.010')
axs[5].set_xlabel('Delta AUC with 95%\nbootstrap interval')
# Retain accepted exploratory title wording, allowing only line wraps.
titles=[
 'A  ESM representations weakly\nalign with PPI topology',
 'B  Degree and k-core gains\nare non-monotonic',
 'C  Performance differences across\nbetweenness and clustering strata',
 'D  Performance differences across\nneighbour-risk strata',
 'E  Joint topology cells contain\nboth positive and negative gains',
 'F  Selected exploratory\njoint-cell profiles']
for ax in axs:
 ax.set_title('',loc='left');ax.set_title('')
for i,txt in enumerate(titles):
 x=[37,205,347][i%3]
 if i==2:x=374
 y=319 if i<3 else 154
 fig.text(x/503.79,y/345,txt,fontfamily='Times New Roman',fontweight='bold',fontsize=8,ha='left',va='top',linespacing=1.05)
fig.text(.5,337/345,'Exploratory PPI-GCN performance across topology contexts',ha='center',va='top',fontsize=11,fontweight='bold',fontfamily='Times New Roman')
# Normalize all pre-existing annotations and legends to at least eight points.
from matplotlib.text import Text
for t in fig.findobj(Text):
 t.set_fontfamily('Times New Roman')
 if t.get_fontsize()<8:t.set_fontsize(8)
for ax in axs:
 leg=ax.get_legend()
 if leg:
  for t in leg.get_texts():t.set_fontsize(8)
# Avoid clipping the baseline-rate notes while preserving their strings.
for t in axs[3].texts:
 if t.get_text().startswith('pos '):t.set_y(-.008)
# Record the exact numeric data and interval segments, not a new statistical analysis.
after=science_snapshot()
assert before==after,'Scientific data or axes limits changed'
# Set font sizes only; preserve line and point colours. No system font files are copied.
fig.canvas.draw()
texts=[]
renderer=fig.canvas.get_renderer()
for t in fig.findobj(Text):
 if not t.get_visible() or not t.get_text():continue
 box=t.get_window_extent(renderer)
 texts.append({'text':t.get_text(),'fontsize':t.get_fontsize(),'family':t.get_fontfamily(),'bbox_pdf_pt':[float(x)*72/fig.dpi for x in box.extents]})
files={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [LEGACY]+[old.P/q for q in ['esm_ppi_alignment_analysis/summary.json','esm_ppi_community_alignment_analysis/summary.json','topology_gain_analysis_exp26B/topology_stratified_metrics.csv','betweenness_gain_analysis_exp30/betweenness_stratified_metrics.csv','clustering_gain_analysis_exp31/clustering_stratified_metrics.csv','neighborhood_risk_enrichment_exp27/neighbor_risk_stratified_metrics.csv','topology_profile_joint_map_exp33/joint_gain_maps.csv','topology_profile_joint_map_bootstrap_exp33B/key_joint_cells_bootstrap_summary.csv']]}
(WORK/'scientific_data_unchanged.json').write_text(json.dumps({'status':'PASS','before':before,'after':after,'source_sha256':files,'scope':'Original fixed-reference Figure 3; no matched ensemble or retraining. Data artists and error-bar segments unchanged before/after layout edits.'},indent=2,default=lambda o:o.item() if hasattr(o,'item') else o))
(WORK/'text_layout.json').write_text(json.dumps(texts,indent=2))
fig.savefig(OUT/'Figure3_topology_context_v1_accepted.pdf',bbox_inches=None)
fig.savefig(OUT/'Figure3_topology_context_v1_accepted.png',dpi=300,bbox_inches=None)
print(OUT/'Figure3_topology_context_v1_accepted.pdf')
