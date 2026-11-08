from pathlib import Path
import os, json, hashlib, re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch,FancyBboxPatch
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator,FormatStrFormatter

WORK=Path(__file__).resolve().parent
OUT=Path(os.environ.get('PAPER1_FIGURE_OUTPUT', 'outputs/BIBM_workshop_IEEE_figure_labels_20261108/source/figure_drafts')).resolve()
OUT.mkdir(parents=True, exist_ok=True)
os.environ['PAPER1_FIG_OUT']=str(WORK/'unused_original_script_outputs')
os.environ['PAPER1_PROJECT']='/Users/teyian/Documents/paper1/BIBM2026_submission'
W,H=503.79,345.
FONTS=[Path('/System/Library/Fonts/Supplemental/Times New Roman.ttf'),Path('/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf')]
for p in FONTS: font_manager.fontManager.addfont(str(p))

def load_original(n):
 import pandas as pd
 num='1' if 'figure1' in n else '2'
 p=WORK/f'original_plot_functions_figure{num}.py'
 g=json.loads((WORK/'compact_plot_inputs.json').read_text())[num]
 g.update({'np':np,'plt':plt,'FancyArrowPatch':FancyArrowPatch,'FancyBboxPatch':FancyBboxPatch})
 if num=='1':g['df44']=pd.DataFrame(g.pop('df44_records'))
 exec(compile(p.read_text(),str(p),'exec'),g)
 return g,p
f1,p1=load_original('make_figure1_connectivity_v2_local.py')
f2,p2=load_original('make_figure2_fusion_homology_v1.py')
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.titlesize':8,'axes.labelsize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.6,'xtick.major.size':2,'ytick.major.size':2,'xtick.major.pad':2,'ytick.major.pad':2,'figure.dpi':150,'savefig.dpi':300})

# Compare every data rectangle and data line / error-bar segment before and after
# layout. Axis decorations, text and the flow diagrams do not enter this audit.
def signature(ax):
 def arr(a):return np.asarray(a,dtype=float).round(14).tolist()
 return {'rectangles':[{'x':float(p.get_x()),'y':float(p.get_y()),'w':float(p.get_width()),'h':float(p.get_height()),'face':arr(p.get_facecolor())} for p in ax.patches if isinstance(p,matplotlib.patches.Rectangle)],
 'lines':[{'x':arr(l.get_xdata()),'y':arr(l.get_ydata()),'color':str(l.get_color()),'marker':str(l.get_marker())} for l in ax.lines],
 'collections':[{'segments':[arr(s) for s in c.get_segments()],'colors':arr(c.get_colors())} for c in ax.collections if hasattr(c,'get_segments')], 'xlim':arr(ax.get_xlim()),'ylim':arr(ax.get_ylim())}

def original_sigs(g):
 fig,axs=plt.subplots(2,3)
 for i,l in enumerate('abcdef'):g['panel_'+l](axs.flat[i])
 d={l:signature(axs.flat[i]) for i,l in enumerate('abcdef') if l!='a'}
 plt.close(fig);return d
old1,old2=original_sigs(f1),original_sigs(f2)

def axes(fig,r):
 x,y,w,h=r;return fig.add_axes([x/W,y/H,w/W,h/H])
def title(fig,x,y,t):return fig.text(x/W,y/H,t,fontsize=8,fontweight='bold',ha='left',va='top',linespacing=1.06)
def figtext(fig,x,y,s,**kw):return fig.text(x/W,y/H,s,fontsize=8,ha='center',va='center',**kw)
def legend(ax,fig,x,y,ncol=1,**kw):
 old=ax.get_legend()
 if old:old.remove()
 return ax.legend(loc='center',bbox_to_anchor=(x/W,y/H),bbox_transform=fig.transFigure,ncol=ncol,frameon=False,fontsize=8,handlelength=.85,handletextpad=.35,columnspacing=.7,labelspacing=.12,borderpad=0,**kw)
def normalize_fonts(fig):
 for t in fig.findobj(Text):
  t.set_fontfamily('Times New Roman')
  t.set_fontsize(max(8,t.get_fontsize()))
  t.set_linespacing(1.03)

def schematic1(fig):
 # Same ten vertices, twelve edges, colors and flow boxes as the accepted panel.
 ax=axes(fig,(3,204,159,97));ax.axis('off');ax.set_xlim(0,1);ax.set_ylim(-.02,1)
 train=np.array([[.12,.62],[.25,.75],[.33,.53],[.18,.37]])
 test=np.array([[.70,.62],[.84,.75],[.82,.46],[.62,.37]])
 val=np.array([[.42,.20],[.52,.28]])
 def edge(a,b,c='#BDBDBD',lw=.75):ax.plot([a[0],b[0]],[a[1],b[1]],color=c,lw=lw,zorder=1)
 for vs in [train,test]:
  for a,b in [(0,1),(1,2),(2,3),(0,3)]:edge(vs[a],vs[b])
 edge(val[0],train[3]);edge(val[1],test[3]);edge(train[1],test[0],f1['COL_HIGH'],1.3);edge(train[2],test[3],f1['COL_HIGH'],1.3)
 for vs,col in [(train,f1['COL_GCN']),(test,f1['COL_LOW']),(val,f1['COL_ORANGE'])]:ax.scatter(vs[:,0],vs[:,1],s=35,color=col,edgecolor='white',linewidth=.5,zorder=3)
 ax.text(.49,.81,'direct train-test\nPPI links',color=f1['COL_HIGH'],ha='center',va='center',fontsize=8)
 ax.text(.47,.12,'Validation',color='#B15928',fontweight='bold',ha='center',fontsize=8)
 for x,c,l in [(0.06,f1['COL_GCN'],'train'),(.34,f1['COL_ORANGE'],'validation'),(.76,f1['COL_LOW'],'test')]:
  ax.scatter(x,.94,s=19,color=c);ax.text(x+.035,.94,l,va='center',fontsize=8)
 for x,s in [(25,'ESM\nfeatures'),(80,'PPI\npropagation'),(139,'MLP\noutput')]:
  figtext(fig,x,201,s,bbox=dict(boxstyle='round,pad=.23',fc='white',ec='#4D4D4D',lw=.6))
 for x0,x1 in [(43,53),(107,120)]:
  fig.add_artist(FancyArrowPatch((x0/W,201/H),(x1/W,201/H),transform=fig.transFigure,arrowstyle='->',mutation_scale=7,color='#4D4D4D',lw=.6))
 return ax

def schematic2(fig):
 ax=axes(fig,(0,175,180,123));ax.axis('off');ax.set_xlim(0,180);ax.set_ylim(0,123)
 def box(cx,cy,w,h,s,c):
  ax.add_patch(FancyBboxPatch((cx-w/2,cy-h/2),w,h,boxstyle='round,pad=.1,rounding_size=1',fc=c,ec='#4D4D4D',lw=.6,zorder=3))
  ax.text(cx,cy,s,ha='center',va='center',fontsize=8,linespacing=1.03,zorder=4)
 box(89,105,90,29,'ClinVar-STRING\nmissense variants\nn=177,213',f2['COL_BASE'])
 box(44,63,69,29,'CADD\n177,205\n99.995%',f2['COL_CADDAM'])
 box(132,63,80,29,'AlphaMissense\n171,753\n96.92%','#A1D99B')
 box(44,23,70,24,'CADD+AM\nbaseline','#FDD0A2')
 box(132,23,84,43,'Representation controls:\nraw ESM\nedge-ablated\npermuted\nreal PPI-GCN',f2['COL_GRAPH'])
 for a,b in [((71,90),(46,78)),((107,90),(130,78)),((44,48),(44,35)),((79,23),(89,23))]:
  ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=7,color='#4D4D4D',lw=.65,zorder=2))
 from matplotlib.path import Path as MPath
 ax.add_patch(FancyArrowPatch(path=MPath([(132,48),(132,47),(84,47),(76,35)],[MPath.MOVETO,MPath.LINETO,MPath.LINETO,MPath.LINETO]),arrowstyle='-|>',mutation_scale=7,color='#4D4D4D',lw=.65,zorder=2))
 return ax

def f1draw():
 fig=plt.figure(figsize=(W/72,H/72))
 figtext(fig,W/2,337,'Fixed PPI propagation under connectivity and topology controls',fontweight='bold').set_fontsize(10)
 rects=[None,(204,207,117,86),(412,219,87,81),(36,38,126,87),(216,38,108,87),(383,38,116,87)]
 axs=[schematic1(fig)]+[axes(fig,r) for r in rects[1:]]
 for l,a in zip('bcdef',axs[1:]):f1['panel_'+l](a);a.set_title('',loc='left');a.set_title('');a.tick_params(labelsize=8)
 titles=[(3,322,'A  Protein-level split on\n    a shared PPI graph'),(176,322,'B  Full-graph PPI-GCN\n    point gains'),(345,322,'C  Variant vs protein-cluster\n    bootstrap'),(3,177,'D  Removing direct\n    train-test PPI edges'),(176,177,'E  Partial train-test\n    edge ablation'),(345,177,'F  Topology permutation\n    control')]
 for x,y,t in titles:title(fig,x,y,t)
 legend(axs[1],fig,256,301,ncol=2)
 legend(axs[3],fig,99,144)
 legend(axs[4],fig,258,144)
 legend(axs[5],fig,431,144)
 for idx,new in [(3,'AUC point estimate\nnear raw ESM'),(5,'Point estimates\nnear raw ESM')]:
  for t in list(axs[idx].texts):t.remove()
  axs[idx].text(1.0,.730,new,ha='center',va='center',fontsize=8,color='#4D4D4D')
 # Compact tick spacing, unchanged numerical limits and data.
 axs[2].set_xticks([-.02,0,.02,.04,.06,.08]);axs[2].tick_params(axis='x',labelrotation=45)
 axs[2].set_yticklabels(['AUC\nvariant','AUC\nprotein-clustered','AUPRC\nvariant','AUPRC\nprotein-clustered'])
 axs[4].set_xticks([0,20,40,60,80,100])
 axs[4].set_xlabel('Removed direct train-test\nPPI pairs (%)',labelpad=3)
 axs[4].set_yticks([-.010,-.005,0,.005,.010,.015,.020,.025]);axs[4].tick_params(axis='y',labelsize=8)
 for i in [1,3,5]:axs[i].set_yticks([.60,.65,.70,.75,.80] if i!=1 else [.55,.60,.65,.70,.75,.80])
 axs[4].yaxis.labelpad=1;axs[3].yaxis.labelpad=2;axs[1].yaxis.labelpad=2;axs[5].yaxis.labelpad=2
 normalize_fonts(fig)
 return fig,axs

def f2draw():
 fig=plt.figure(figsize=(W/72,H/72))
 figtext(fig,W/2,337,'Strong predictors bound the added value of PPI graph context',fontweight='bold').set_fontsize(10)
 rects=[None,(207,219,117,77),(413,219,80,77),(34,35,133,88),(241,35,83,88),(383,35,116,88)]
 axs=[schematic2(fig)]+[axes(fig,r) for r in rects[1:]]
 for l,a in zip('bcdef',axs[1:]):f2['panel_'+l](a);a.set_title('',loc='left');a.set_title('')
 titles=[(3,321,'A  Fusion-control setup'),(184,321,'B  Original split: matched\n    fusion controls'),(346,321,'C  Real PPI-GCN margin\n    over controls'),(3,162,'D  Homology-aware split:\n    ESM vs PPI-GCN'),(184,162,'E  Homology-aware:\n    strong-predictor fusion'),(346,162,'F  Test-set prevalence\n    differs by split')]
 for x,y,t in titles:title(fig,x,y,t)
 legend(axs[1],fig,265,303,ncol=2)
 legend(axs[2],fig,455,202,ncol=2)
 legend(axs[3],fig,100,136,ncol=2)
 legend(axs[4],fig,268,136,ncol=2)
 axs[1].set_xticklabels(['CADD+AM','+raw\nESM','+edge-\nablated','+topology-\npermuted','+real\nPPI-GCN'],rotation=48,ha='right',rotation_mode='anchor')
 axs[2].set_yticklabels(['vs CADD+AM\narch-matched','vs raw ESM\nfusion','vs edge-ablated\nfusion','vs topology-\npermuted fusion'])
 axs[2].set_xticks([0,.02,.04,.06]);axs[2].set_xlabel('Delta performance',labelpad=19)
 axs[4].set_xticks([.80,.85,.90,.95]);axs[4].set_xlabel('Test performance',labelpad=3)
 axs[3].set_yticks([.2,.3,.4,.5,.6,.7,.8]);axs[3].yaxis.labelpad=1
 axs[1].set_yticks([.90,.92,.94,.96,.98]);axs[1].yaxis.labelpad=2
 axs[5].set_yticks([0,.05,.10,.15,.20,.25,.30,.35]);axs[5].yaxis.labelpad=1
 normalize_fonts(fig)
 return fig,axs

audit={'canvas_pt':[W,H],'font':'Times New Roman','minimum_font_pt':8,'no_training_or_statistical_recalculation':True,'layout_only':True,'font_files_not_distributed':True,'sources':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [p1,p2,WORK/'compact_plot_inputs.json']},'figures':{}}
for n,g,old,draw in [(1,f1,old1,f1draw),(2,f2,old2,f2draw)]:
 fig,axs=draw();new={l:signature(axs[i]) for i,l in enumerate('abcdef') if l!='a'}
 # Limits are captured before manually setting ticks; original limits intentionally unchanged.
 assert old==new, f'Plotted numeric data changed in Figure {n}'
 basename=['','Figure1_connectivity_topology_controls_v2_accepted','Figure2_fusion_homology_v1_accepted'][n]
 fig.canvas.draw()
 texts=[{'s':t.get_text(),'pt':t.get_fontsize(),'bbox':list(t.get_window_extent(fig.canvas.get_renderer()).bounds)} for t in fig.findobj(Text) if t.get_visible() and t.get_text()]
 (WORK/f'figure{n}_plotted_data.json').write_text(json.dumps(new,indent=2))
 (WORK/f'figure{n}_text_extents.json').write_text(json.dumps(texts,indent=2))
 fig.savefig(OUT/(basename+'.pdf'))
 fig.savefig(OUT/(basename+'.png'),dpi=300)
 audit['figures'][str(n)]={'data_signature_equal':old==new,'panels_compared':['B','C','D','E','F'],'pdf_sha256':hashlib.sha256((OUT/(basename+'.pdf')).read_bytes()).hexdigest(),'png_sha256':hashlib.sha256((OUT/(basename+'.png')).read_bytes()).hexdigest(),'min_artist_font_size':min(t['pt'] for t in texts)}
 plt.close(fig)
(WORK/'plotting_audit.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit,indent=2))
