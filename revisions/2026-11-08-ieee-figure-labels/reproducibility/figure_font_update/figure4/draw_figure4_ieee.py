#!/usr/bin/env python3
"""Typography-only redraw from the original Figure 4 aggregate source tables.
No predictions, bootstrap samples, model files, or new statistical tests are read.
"""
from pathlib import Path
import os,json,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'matplotlib_cache'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties,fontManager
from matplotlib.text import Text

FONT=Path(os.environ.get('TIMES_NEW_ROMAN_REGULAR','/System/Library/Fonts/Supplemental/Times New Roman.ttf'))
BOLD=Path(os.environ.get('TIMES_NEW_ROMAN_BOLD','/System/Library/Fonts/Supplemental/Times New Roman Bold.ttf'))
for f in [FONT,BOLD]:fontManager.addfont(str(f))
REG=FontProperties(fname=str(FONT),size=8)
BOL=FontProperties(fname=str(BOLD),size=8)
plt.rcParams.update({'font.family':'Times New Roman','font.size':8,'axes.labelsize':8,
 'axes.titlesize':8,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
 'pdf.fonttype':42,'ps.fonttype':42,'axes.linewidth':.55,'xtick.major.width':.5,
 'ytick.major.width':.5,'xtick.major.size':2,'ytick.major.size':2,'savefig.dpi':300,
 'axes.unicode_minus':True})
WIDTH,HEIGHT=503.79,345.0
P=HERE/'source_data'
OUT=Path(os.environ.get('FIGURE4_OUTPUT',str(HERE/'Figure4_prediction_transitions_v2_accepted.pdf')))
files={
 'summary':'corrected_sample_identification_exp34/summary.json',
 'esm_profile':'corrected_sample_profile_exp35/esm_to_gcn_corrected_profile_summary.csv',
 'full_profile':'corrected_sample_profile_exp35/cadd_am_to_full_corrected_profile_summary.csv',
 'esm_stats':'fp_vs_fn_stats_exp36B/esm_to_gcn_fp_vs_fn_stats.csv',
 'full_stats':'fp_vs_fn_stats_exp36B/cadd_am_to_full_fp_vs_fn_stats.csv',
 'contrasts':'difficulty_matched_correction_contrast_exp40/difficulty_matched_correction_contrast_results.csv'}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
summary=json.loads((P/files['summary']).read_text())
data={k:pd.read_csv(P/v) for k,v in files.items() if k!='summary'}
FEATURES=['degree_bucket','kcore_bucket','neighbor_risk_bucket','neighbor_pathogenic_rate',
 'neighbor_train_variant_count','betweenness_bucket','clustering_coefficient',
 'boundary_role_bucket','within_comm_neighbor_fraction']
LABELS=['degree','k-core','neighbour risk','neighbour pathogenic rate',
 'neighbour train variants','betweenness','clustering coefficient','boundary role',
 'within-community fraction']
COL_CORRECT='#9ECAE1';COL_WRONG='#BDBDBD';COL_IMPROVED='#3182BD'
COL_DEGRADED='#D95F0E';COL_FN='#2171B5';COL_FP='#F16913';COL_GRAY='#7A7A7A'
fig=plt.figure(figsize=(WIDTH/72,HEIGHT/72),dpi=150)
def ax(x,y,w,h):return fig.add_axes([x/WIDTH,y/HEIGHT,w/WIDTH,h/HEIGHT])
def title(x,y,text):return fig.text(x/WIDTH,y/HEIGHT,text,fontproperties=BOL,ha='left',va='top',linespacing=1.05)
def clean(a):
 a.spines['top'].set_visible(False);a.spines['right'].set_visible(False)
def transition(a,key):
 d=summary[key];left=0
 for count,color in [(d['count_both_correct'],COL_CORRECT),(d['count_both_wrong'],COL_WRONG),(d['count_improved'],COL_IMPROVED),(d['count_degraded'],COL_DEGRADED)]:
  a.barh(.60,count/d['n'],left=left,height=.23,color=color,edgecolor='black',linewidth=.35)
  if count/d['n']>.1:a.text(left+count/d['n']/2,.60,f'{count:,}',ha='center',va='center',fontsize=8,color='#252525')
  left+=count/d['n']
 assert sum(d[k] for k in ['count_both_correct','count_both_wrong','count_improved','count_degraded'])==18220
 a.text(.015,.38,f"Corrected: FN rescued {d['count_FN_rescued']:,};\nFP suppressed {d['count_FP_suppressed']:,}",fontsize=8,ha='left',va='top',color=COL_FN,linespacing=1.1)
 a.text(.015,.14,f"Degraded: TN to FP {d['count_TN_to_FP']:,};\nTP to FN {d['count_TP_to_FN']:,}",fontsize=8,ha='left',va='top',color=COL_DEGRADED,linespacing=1.1)
 a.set(xlim=(0,1),ylim=(-.12,1),yticks=[])
 a.set_xticks([0,.2,.4,.6,.8,1]);a.set_xlabel('Fraction of test variants',labelpad=2);clean(a)
def matrix(df):
 rows=df[(df.group_type=='error_transition') & df.group_name.isin(['FP_suppressed','FN_rescued'])].set_index('group_name')
 vals=np.array([[rows.loc[g,'mean_'+f] for g in ['FP_suppressed','FN_rescued']] for f in FEATURES],dtype=float)
 # Identical display transform to the original script; these are already stored means.
 scaled=np.array([np.zeros_like(r) if np.isclose(r.max(),r.min()) else (r-r.mean())/(r.max()-r.min()) for r in vals])
 return vals,scaled
matrices={}
def heat(a,key):
 vals,m=matrix(data[key]);matrices[key]={'means':vals.tolist(),'display_matrix':m.tolist()}
 a.imshow(m,aspect='auto',cmap='RdBu_r',vmin=-1,vmax=1)
 a.set_xticks([0,1]);a.set_xticklabels(['FP\nsuppressed','FN\nrescued'])
 a.set_yticks(np.arange(9));a.set_yticklabels(LABELS)
 a.tick_params(length=0,pad=2)

# The same 2 x 3 scientific panels, with empty space reassigned to readable labels.
fig.text(.5,340/HEIGHT,'Graph augmentation changes false-negative and false-positive predictions\nin different network contexts',fontproperties=FontProperties(fname=str(BOLD),size=9),ha='center',va='top',linespacing=1.0)
title(7,316,'A  Raw ESM to PPI-GCN transitions')
title(175,316,'B  CADD+AM to graph-fusion transitions')
title(339,316,'C  Raw ESM correction profiles')
A=ax(8,185,153,117);B=ax(176,185,153,117);C=ax(430,185,72,117)
transition(A,'esm_vs_gcn');transition(B,'cadd_am_vs_full');heat(C,'esm_profile')
title(3,151,'D  Graph-fusion correction profiles')
title(171,151,'E  Directions of mean differences\nbetween corrected groups')
title(339,151,'F  Rescued versus persistent\nfalse-negative profiles')
D=ax(91,43,72,86);E=ax(258,43,71,86);F=ax(430,43,72,86)
heat(D,'full_profile')
directions=[]
for j,key in enumerate(['esm_stats','full_stats']):
 s=data[key].set_index('feature').loc[FEATURES]
 for k,(feature,row) in enumerate(s.iterrows()):
  fn_higher=bool(row.diff_fp_minus_fn<0);color=COL_FN if fn_higher else COL_FP
  E.scatter(j,8-k,s=25,color=color,alpha=.85,edgecolor='black',linewidth=.25)
  directions.append({'comparison':key,'feature':feature,'FN_higher':fn_higher,'color':color})
E.set_yticks(np.arange(9));E.set_yticklabels(LABELS[::-1]);E.set_xlim(-.45,1.45);E.set_ylim(-.5,8.5)
E.set_xticks([0,1]);E.set_xticklabels(['ESM to\nPPI-GCN','CADD+AM\nto graph\nfusion']);E.tick_params(axis='y',pad=2,length=2);clean(E)
keep=[FEATURES[i] for i in [0,1,2,3,5,7,8]]
df=data['contrasts'];s=df[(df.condition=='FN_rescued_vs_persistent_FN') & df.feature.isin(keep)].set_index('feature').loc[keep]
g=s.hedges_g.to_numpy(float);y=np.arange(7)[::-1]
bars=F.barh(y,g,color=[COL_FN if v>0 else COL_DEGRADED for v in g],edgecolor='black',linewidth=.35)
F.axvline(0,color=COL_GRAY,lw=.6,ls='--');F.set_yticks(y);F.set_yticklabels([LABELS[FEATURES.index(f)] for f in keep]);F.set_xlim(-1.05,1.12)
F.set_xticks([-1,-.5,0,.5,1]);F.set_xticklabels(['−1','−0.5','0','0.5','1'])
F.set_xlabel("Hedges' g\nrescued FN vs\npersistent FN",labelpad=2);F.tick_params(axis='y',pad=2,length=2);clean(F)
assert np.array_equal(np.array([b.get_width() for b in bars]),g)
for t in fig.findobj(Text):
 if t.get_text():
  t.set_fontproperties(FontProperties(fname=str(BOLD if t.get_fontweight()=='bold' else FONT),size=max(8,t.get_fontsize())))
# Explicit title FontProperties carry bold fname even without weight='bold'.
for t in fig.texts:t.set_fontproperties(FontProperties(fname=str(BOLD),size=9 if t==fig.texts[0] else 8))
fig.canvas.draw();renderer=fig.canvas.get_renderer()
text_boxes=[];outside=[]
for t in fig.findobj(Text):
 if not t.get_text() or not t.get_visible():continue
 box=t.get_window_extent(renderer);points=[v*72/fig.dpi for v in [box.x0,box.y0,box.x1,box.y1]]
 text_boxes.append({'text':t.get_text(),'font_size':t.get_fontsize(),'box_pt':points})
 if points[0]<-.5 or points[1]<-.5 or points[2]>WIDTH+.5 or points[3]>HEIGHT+.5:outside.append(text_boxes[-1])
fig.savefig(OUT,metadata={'Title':'Figure 4: original prediction transitions and descriptive topology profiles','Subject':'IEEE-size Times New Roman figure labels; source data unchanged.'})
fig.savefig(HERE/'Figure4_after.png',dpi=300)
report={'canvas_pt':[WIDTH,HEIGHT],'min_requested_font_pt':8,'aggregate_sources':{k:{'path':str(Path('source_data')/v),'sha256':sha(P/v)} for k,v in files.items()},'summary':summary,'heatmap_matrices':matrices,'dot_directions':directions,'hedges_g':dict(zip(keep,g.tolist())),'all_dots_same_size':25,'all_dots_same_alpha':.85,'significance_stars':0,'new_statistical_analysis':False,'text_boxes':text_boxes,'out_of_canvas_text':outside,'output_sha256':sha(OUT)}
(HERE/'figure4_drawing_audit.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'output':str(OUT),'canvas_pt':[WIDTH,HEIGHT],'outside_text':outside,'sha256':sha(OUT)}))
