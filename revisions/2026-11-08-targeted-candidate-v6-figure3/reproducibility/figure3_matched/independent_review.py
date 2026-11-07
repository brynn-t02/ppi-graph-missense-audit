#!/usr/bin/env python3
"""Independent read-only Figure3 check: explicit variant expansion and rank-based AUC.
Does not import or execute production calculation / verification functions.
"""
from pathlib import Path
import json, hashlib, time
import numpy as np
import pandas as pd
W=Path(__file__).resolve().parent; A=W/'analysis'
ROOT=W.parents[1]; CAN=ROOT/'outputs/BIBM_workshop_font_clean_20261025/source/canonical_analysis_20261025'
OLD=ROOT/'outputs/BIBM_workshop_font_clean_20261025/source/figure_inputs/data/processed'
LOCAL=Path('/Users/teyian/Documents/paper1/server_snapshot_full/root/data/processed')
def read(p):return pd.read_csv(p,float_precision='round_trip')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def same(x,y):assert np.array_equal(x,y,equal_nan=True)
def close(x,y):assert np.allclose(x,y,atol=1e-12,rtol=0,equal_nan=True),(x,y)
# Independently derive Mann-Whitney U from average ranks of tied predictions.
# AP is sum over descending thresholds of positive mass * precision.
def scores(labels,prob):
 y=np.asarray(labels,dtype=np.int64);p=np.asarray(prob,dtype=np.float64)
 pos=int(y.sum());neg=len(y)-pos
 if not pos or not neg:return np.array([np.nan,np.nan])
 unique,inverse,counts=np.unique(p,return_inverse=True,return_counts=True)
 end=np.cumsum(counts);average_rank=end-(counts-1)/2
 rank_sum=float(np.sum(average_rank[inverse][y==1]))
 auc=(rank_sum-pos*(pos+1)/2)/(pos*neg)
 positive=np.bincount(inverse,weights=y,minlength=len(unique))[::-1]
 total=counts[::-1];cp=np.cumsum(positive);ct=np.cumsum(total)
 ap=float(np.sum(positive/pos * cp/ct))
 return np.array([auc,ap])
start=time.time();P=json.loads((A/'frozen_protocol.json').read_text());D=read(A/'frozen_18220_figure3_predictions_memberships.csv.gz')
assert len(D)==18220 and D.VariationID.is_unique and D.STRING_protein_id.nunique()==1413 and D.label.sum()==5822
T=read(LOCAL/'clinvar_test_main.csv').set_index('VariationID').loc[D.VariationID]
same(D.label.to_numpy(),T.Label.to_numpy());assert D.STRING_protein_id.tolist()==T.STRING_protein_id.tolist()
F=read(CAN/'public/frozen_samples_177213.csv.gz');F=F[F.original_split=='test'].set_index('VariationID').loc[D.VariationID]
same(D.label.to_numpy(),F.label.to_numpy());assert D.STRING_protein_id.tolist()==F.STRING_protein_id.tolist()
topo=read(LOCAL/'topology_profile_joint_map_exp33/merged_topology_profile_features.csv')
assert topo.Name.tolist()==T.Name.tolist()
feature_cols=['degree_bucket','kcore_bucket','neighbor_risk_bucket','neighbor_pathogenic_rate','neighbor_train_variant_count','betweenness_bucket','betweenness','clustering_bucket','clustering_coefficient','boundary_role_bucket','within_comm_neighbor_fraction','community_id']
for col in feature_cols:
 same(D[col].to_numpy(),topo[col].to_numpy());assert D.groupby('STRING_protein_id')[col].nunique(dropna=False).max()==1
ensemble=[]
for arm in ['raw_ESM','full_FPP']:
 q=read(CAN/f'public/test_predictions__original__{arm}.csv.gz').set_index('VariationID').loc[D.VariationID]
 same(q.label.to_numpy(),D.label.to_numpy());assert q.STRING_protein_id.tolist()==D.STRING_protein_id.tolist()
 m=q[[f'p_seed_{s}'for s in [42,43,44]]].to_numpy(dtype=np.float64)
 for i,s in enumerate([42,43,44]):same(m[:,i],D[f'{arm}_seed_{s}'].to_numpy())
 same(m.mean(axis=1,dtype=np.float64),D[arm].to_numpy());same(q.ensemble_float64.to_numpy(),D[arm].to_numpy())
 assert D.groupby('STRING_protein_id')[arm].nunique().max()==1
 got=scores(D.label,D[arm]);ref=read(CAN/'ensemble_metrics.csv');ref=ref[(ref.population=='original')&(ref.arm==arm)].iloc[0];close(got,[ref.auc,ref.auprc])
 ensemble.append({'arm':arm,'auc':float(got[0]),'average_precision':float(got[1]),'exact_mean_and_protein_constant':True})
keys=['row_col','row_group','col_col','col_group']
old=read(OLD/'topology_profile_joint_map_exp33/joint_gain_maps.csv');new=read(A/'table5_complete_53_joint_cells.csv')
assert len(old)==len(new)==len(P['joint_cells'])==53
assert old[keys].to_dict('records')==new[keys].to_dict('records')==[{k:r[k]for k in keys}for r in P['joint_cells']]
oldsel=read(OLD/'topology_profile_joint_map_bootstrap_exp33B/key_joint_cells_bootstrap_summary.csv')
newsel=read(A/'table5_selected_original_four_cells.csv')
assert oldsel.cell_name.tolist()==newsel.cell_name.tolist()==[r['cell_name']for r in P['fixed_original_displayed_cells']]
assert newsel.cell_index.tolist()==[5,12,24,14]
bs=read(A/'all_53_joint_cell_protein_bootstrap_samples.csv.gz');ss=read(A/'all_18_strata_protein_bootstrap_samples.csv.gz')
assert len(bs)==106000 and len(ss)==36000
members=read(A/'all_53_joint_cell_memberships.csv.gz'); selected=set(newsel.cell_index); reports=[];maxerr=0.;pointchecks=0; drawschecked=0;samplechecks=[]
for kind,groups,table,samples,index_col in [('joint',P['joint_cells'],new,bs,'cell_index'),('stratum',P['one_dimensional_strata'],read(A/'table5_all_18_strata.csv'),ss,'stratum_index')]:
 for i,s in enumerate(groups):
  mask=((D[s['row_col']]==s['row_group'])&(D[s['col_col']]==s['col_group']))if kind=='joint' else (D[s['group_col']]==s['group'])
  sub=D[mask].reset_index(drop=True);assert len(sub)==s['n'] and int(sub.label.sum())==s['positive']
  ids=sorted(map(int,sub.VariationID));assert hashlib.sha256('\n'.join(map(str,ids)).encode()).hexdigest()==s['membership_sha256']
  if kind=='joint':
   stored=members[members.cell_index==i];assert sorted(stored.VariationID.tolist())==ids
  row=table[table[index_col]==i].iloc[0] if kind=='joint' else table[(table.group_col==s['group_col'])&(table.group==s['group'])].iloc[0]
  y=sub.label.to_numpy(dtype=int);r=sub.raw_ESM.to_numpy();f=sub.full_FPP.to_numpy()
  raw=scores(y,r);full=scores(y,f);close(raw,[row.raw_auc,row.raw_auprc]);close(full,[row.full_auc,row.full_auprc]);close(full-raw,[row.delta_auc,row.delta_auprc]);pointchecks+=1
  draws=samples[samples[index_col]==i].reset_index(drop=True);assert draws.repeat.tolist()==list(range(1,2001))
  values=draws[['delta_auc','delta_auprc']].to_numpy();valid=np.isfinite(values).all(axis=1)
  assert np.array_equal(np.isnan(values[:,0]),np.isnan(values[:,1]))
  ci=np.percentile(values[valid],[2.5,97.5],axis=0)
  close(ci,[[row.boot_delta_auc_ci_lower,row.boot_delta_auprc_ci_lower],[row.boot_delta_auc_ci_upper,row.boot_delta_auprc_ci_upper]])
  assert valid.sum()==row.finite_replicates and (~valid).sum()==row.discarded_one_class_replicates
  # Independent RNG reconstruction plus explicit membership lists, preserving the complete
  # set of variants of each drawn protein with multiplicity in BOTH models.
  proteins=sorted(sub.STRING_protein_id.unique());K=len(proteins)
  locations=[np.flatnonzero(sub.STRING_protein_id.to_numpy()==p)for p in proteins]
  nvec=np.array([len(ix)for ix in locations]);pvec=np.array([y[ix].sum()for ix in locations]);rng=np.random.default_rng(20261108+i+(1000 if kind=='stratum'else 0))
  bad=0;errs=[]
  for j in range(2000):
   chosen=rng.integers(K,size=K)
   # Every draw includes exactly K protein draws; all rows of each selected protein.
   n=int(nvec[chosen].sum());positive=int(pvec[chosen].sum());isbad=positive==0 or positive==n
   bad+=isbad;assert n==draws.resampled_variants.iloc[j];assert isbad==(not valid[j])
   dofull=(kind=='joint'and i in selected)or j in [0,1,19,999,1999]
   if dofull:
    ix=np.concatenate([locations[k]for k in chosen]);assert len(ix)==n and y[ix].sum()==positive
    got=scores(y[ix],f[ix])-scores(y[ix],r[ix]);close(got,values[j]);drawschecked+=1
    if not isbad:maxerr=max(maxerr,float(np.abs(got-values[j]).max()))
    if j==19 and kind=='joint' and i in selected:
     samplechecks.append({'cell_index':i,'repeat':j+1,'protein_draws':K,'distinct_drawn_proteins':len(set(chosen)),'expanded_variants':n,'positive_variants':positive,'paired_same_expansion':True,'delta_auc':float(got[0]),'delta_auprc':float(got[1])})
  assert bad==row.discarded_one_class_replicates
  reports.append({'kind':kind,'index':i,'n':len(sub),'proteins':K,'invalid_one_class_draws':bad,'finite_draws':2000-bad,'all2000_sample_counts_and_class_validity_reconstructed':True,'CI_from_saved_samples_verified':True,'all2000_metric_draws_independently_recomputed':kind=='joint'and i in selected})
  if kind=='joint' and i in selected:print('selected cell passed',i,flush=True)
result={'date':'2026-11-08','passed':True,'implementation':'Independent Mann-Whitney tied-average-rank AUC and threshold-group AP; independent explicit repeated variant-row expansion. Does not import or execute analysis code.','sklearn_used':False,'reason_no_sklearn':'Not installed in available local Python runtimes; independent formulas used instead.','canonical_rows_labels_proteins':{'variants':18220,'proteins':1413,'positives':5822,'six_seed_columns_exact':True,'two_float64_means_exact':True},'ensemble_metrics':ensemble,'frozen_original_53_group_keys_memberships_and_four_display_objects_verified':True,'fixed_display_cell_indices':[5,12,24,14],'fixed_topology_features_verified':feature_cols,'point_metrics_checked_groups':pointchecks,'percentile_intervals_checked_groups':71,'all142000_resample_variant_counts_and_class_validity_verified':True,'independent_metric_replicates_checked':drawschecked,'all2000draws_recomputed_for_original_four_display_cells':True,'max_absolute_metric_difference':maxerr,'joint_total_one_class_replicates':sum(r['invalid_one_class_draws']for r in reports if r['kind']=='joint'),'strata_total_one_class_replicates':sum(r['invalid_one_class_draws']for r in reports if r['kind']=='stratum'),'selected_group_example_expansions':samplechecks,'group_checks':reports,'bootstrap_assessment':'Paired draws sample K group-specific unique proteins with replacement, include all variants with protein multiplicities, and use same expanded rows for bothmodels. One-class draws are recorded missing and excluded without retry. Percentile intervals condition on fitted models/graph and do not remove original post-hoc selection or multiplicity.','limits':'No new training/inference/checkpoint loading/server access. No novel scientific analysis outside requested read-onlyverification.','read_only_analysis_assets':True,'seconds':time.time()-start,'verified_input_sha256':{str(p):sha(p)for p in [A/'frozen_protocol.json',A/'frozen_18220_figure3_predictions_memberships.csv.gz',A/'table5_complete_53_joint_cells.csv',A/'table5_selected_original_four_cells.csv',A/'all_53_joint_cell_protein_bootstrap_samples.csv.gz',A/'all_18_strata_protein_bootstrap_samples.csv.gz']}}
(W/'independent_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items()if k not in ['group_checks','verified_input_sha256','selected_group_example_expansions']},ensure_ascii=False,indent=2))
