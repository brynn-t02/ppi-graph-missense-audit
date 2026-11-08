#!/usr/bin/env python3
"""Freeze original Figure 3 group definitions before reading new predictions."""
from pathlib import Path
import ast, hashlib, json
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
SRC=Path('/Users/teyian/Documents/paper1/server_snapshot_full/root')
DATA=SRC/'data/processed'
OLD=ROOT/'outputs/BIBM_workshop_font_clean_20261025/source/figure_inputs/data/processed'
HASHES={}
def track(p):
 p=Path(p);HASHES[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p,**kw):return pd.read_csv(track(p),float_precision='round_trip',**kw)
def literal(p,name):
 tree=ast.parse(track(p).read_text())
 return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
def jdump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
test=read(DATA/'clinvar_test_main.csv',usecols=['VariationID','Name','Label','STRING_protein_id'])
cols=['Name','Label','STRING_protein_id','degree_bucket','kcore_bucket','neighbor_risk_bucket','betweenness_bucket','clustering_bucket','boundary_role_bucket']
topo=read(DATA/'topology_profile_joint_map_exp33/merged_topology_profile_features.csv',usecols=cols)
assert len(test)==len(topo)==18220 and test.VariationID.is_unique
for c in ['Name','Label','STRING_protein_id']:assert np.array_equal(test[c],topo[c]),c
topo['VariationID']=test.VariationID
specs=literal(SRC/'33_topology_profile_joint_map.py','joint_specs')
selected=literal(SRC/'33b_joint_topology_map_bootstrap.py','KEY_CELLS')
minimum=literal(SRC/'33_topology_profile_joint_map.py','MIN_GROUP_N');assert minimum==50
old=read(OLD/'topology_profile_joint_map_exp33/joint_gain_maps.csv')
definitions=[]
for rc,cc in specs:
 for (rv,cv),g in topo.groupby([rc,cc],sort=True):
  if len(g)<minimum:continue
  definitions.append(dict(row_col=rc,row_group=int(rv),col_col=cc,col_group=int(cv),map_name=f'{rc}__x__{cc}',n=len(g),positive=int(g.Label.sum()),proteins=int(g.STRING_protein_id.nunique()),membership_sha256=hashlib.sha256('\n'.join(map(str,sorted(g.VariationID))).encode()).hexdigest()))
assert len(definitions)==53
for i,(d,(_,o)) in enumerate(zip(definitions,old.iterrows())):
 d['cell_index']=i
 for k in ['row_col','row_group','col_col','col_group','map_name','n']:assert d[k]==o[k],(i,k)
 assert np.isclose(d['positive']/d['n'],o.pos_rate)
for i,s in enumerate(selected):
 matched=[d for d in definitions if (d['row_col'],d['row_group'],d['col_col'],d['col_group'])==(s['row_col'],s['row_val'],s['col_col'],s['col_val'])]
 assert len(matched)==1
 s['cell_index']=matched[0]['cell_index'];s['display_id']=f'J{i}'
groups=['degree_bucket','kcore_bucket','neighbor_risk_bucket','betweenness_bucket','clustering_bucket','boundary_role_bucket']
strata=[]
for col in groups:
 for g,sub in topo.groupby(col,sort=True):
  if len(sub)<minimum:continue
  assert g in [0,1,2]
  strata.append(dict(group_col=col,group=int(g),n=len(sub),positive=int(sub.Label.sum()),proteins=int(sub.STRING_protein_id.nunique()),membership_sha256=hashlib.sha256('\n'.join(map(str,sorted(sub.VariationID))).encode()).hexdigest()))
assert len(strata)==18
cuts={}
for folder in ['topology_gain_analysis_exp26B','neighborhood_risk_enrichment_exp27','betweenness_gain_analysis_exp30','clustering_gain_analysis_exp31','community_boundary_gain_analysis_exp32']:
 p=track(DATA/folder/'summary.json');j=json.loads(p.read_text());cuts.update({k:v for k,v in j.items() if k.endswith('_cuts')})
for p in ['26b_topology_stratified_gain_analysis.py','27_neighborhood_risk_enrichment_analysis.py','30_betweenness_bridge_role_gain_analysis.py','31_clustering_local_density_gain_analysis.py','32_community_boundary_gain_analysis.py']:
 if (SRC/p).exists():track(SRC/p)
protocol={
 'version':'Figure3 matched ensembles, frozen original groups, protein-clustered bootstrap v6',
 'created_client_date':'2026-11-08','classifier_seeds':[42,43,44],
 'prediction_aggregation':'float64 arithmetic mean of canonical per-protein probabilities mapped to original variants',
 'point_estimand':'Variant-level ROC AUC and non-interpolated average precision; full FPP minus raw ESM',
 'bootstrap':{'unit':'STRING_protein_id','repeats':2000,'within_each_frozen_group':True,'sampling':'Draw K of K distinct group proteins with replacement; include all variants of each sampled protein with its multiplicity, identically for both models','interval':'2.5 and 97.5 percentiles of finite paired differences','joint_seed':'20261108 + original zero-based cell_index','stratum_seed':'20261108 + 1000 + zero-based stratum_index','rng':'numpy.random.default_rng / PCG64','undefined':'Draws with fewer than two classes are recorded as NaN for both metrics, excluded from percentile intervals, and counted; no redraw until success','conditioning':'Fitted models, frozen graph and memberships; no retraining or graph sampling','selection_and_multiplicity':'No subgroup-selection or multiple-comparison adjustment; descriptive exploratory intervals'},
 'minimum_group_variants':minimum,'original_bucket_cutpoints':cuts,
 'joint_maps':specs,'joint_cells':definitions,'one_dimensional_strata':strata,'fixed_original_displayed_cells':selected,
 'freeze_scope':'Original cell definitions, membership and four display objects are fixed before loading matched predictions; no replacement by a better-performing group.',
 'source_input_sha256':HASHES,
 'stopped_draft_reuse':'No stopped-draft results or bootstrap values reused; its bootstrap resampled variants and is unsuitable here. General metric-kernel implementation reviewed and adapted.'}
jdump(OUT/'frozen_protocol.json',protocol)
print(json.dumps({'cells':len(definitions),'strata':len(strata),'selected':selected,'protocol_sha256':hashlib.sha256((OUT/'frozen_protocol.json').read_bytes()).hexdigest()},ensure_ascii=False,indent=2))
