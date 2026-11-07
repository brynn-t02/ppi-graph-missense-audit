from pathlib import Path
import json,math
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent;A=W/'analysis';C=R/'outputs/BIBM_workshop_font_clean_20261025/source/canonical_analysis_20261025/public';L=Path('/Users/teyian/Documents/paper1/server_snapshot_full/root/data/processed')
load=lambda p:pd.read_csv(p,float_precision='round_trip')
test=load(L/'clinvar_test_main.csv');topo=load(L/'topology_profile_joint_map_exp33/merged_topology_profile_features.csv');assert len(topo)==len(test)==18220
for k in ['Name','Label','STRING_protein_id']:assert np.array_equal(test[k],topo[k])
features=['degree_bucket','kcore_bucket','neighbor_risk_bucket','neighbor_pathogenic_rate','neighbor_train_variant_count','betweenness_bucket','betweenness','clustering_bucket','clustering_coefficient','boundary_role_bucket','within_comm_neighbor_fraction']
f=topo[features].copy();f['VariationID']=test.VariationID;f=f.set_index('VariationID').sort_index();labels=test.set_index('VariationID').sort_index().Label.to_numpy();pred={}
for arm,name in [('raw','test_predictions__original__raw_ESM.csv.gz'),('full','test_predictions__original__full_FPP.csv.gz'),('scores','test_predictions__original_fusion__scores_capacity_matched.csv.gz'),('fusion','test_predictions__original_fusion__scores_plus_full_FPP.csv.gz')]:
 p=load(C/name).set_index('VariationID').sort_index();assert p.index.equals(f.index);assert np.array_equal(p.label,labels);pred[arm]=p[['p_seed_42','p_seed_43','p_seed_44']].to_numpy(dtype=np.float64).mean(1)
profiles=load(A/'table6_all_transition_feature_means.csv');contrasts=load(A/'table6_all_44_error_stratified_contrasts.csv');means_checked=0;effects_checked=0
for pair,old,new in [('esm_vs_gcn','raw','full'),('cadd_am_vs_full','scores','fusion')]:
 b=pred[old]>=.5;n=pred[new]>=.5;y=labels.astype(bool)
 # Form groups from eight possible (label, baseline prediction, augmented prediction) states.
 code=y.astype(int)*4+b.astype(int)*2+n.astype(int)
 idx={'both_correct':[0,7],'both_wrong':[3,4],'improved':[2,5],'degraded':[1,6],'FN_rescued':[5],'FP_suppressed':[2],'TN_to_FP':[1],'TP_to_FN':[6],'persistent_FN':[4],'persistent_FP':[3]}
 masks={k:np.isin(code,v) for k,v in idx.items()}
 for row in profiles[profiles.model_pair==pair].itertuples():
  mask=masks[row.group_name];assert int(mask.sum())==row.n
  for feat in features:
   x=f.loc[mask,feat].to_numpy(dtype=float);x=x[np.isfinite(x)];value=math.fsum(x)/len(x)
   assert np.isclose(value,getattr(row,'mean_'+feat),rtol=1e-12,atol=1e-10),(pair,row.group_name,feat);means_checked+=1
 for row in contrasts[contrasts.model_pair==pair].itertuples():
  left,right=('FN_rescued','persistent_FN') if row.condition.startswith('FN') else ('FP_suppressed','persistent_FP')
  x=f.loc[masks[left],row.feature].to_numpy(dtype=float);x=x[np.isfinite(x)];z=f.loc[masks[right],row.feature].to_numpy(dtype=float);z=z[np.isfinite(z)]
  nx,nz=len(x),len(z);mx=math.fsum(x)/nx;mz=math.fsum(z)/nz;vx=math.fsum((v-mx)**2 for v in x)/(nx-1);vz=math.fsum((v-mz)**2 for v in z)/(nz-1)
  g=(1-3/(4*(nx+nz)-9))*(mx-mz)/math.sqrt(((nx-1)*vx+(nz-1)*vz)/(nx+nz-2))
  assert (nx,nz)==(row.n_left,row.n_right);assert np.isclose(g,row.hedges_g,atol=1e-12,rtol=1e-12),(pair,row.condition,row.feature);effects_checked+=1
ind=load(W/'independent_threshold_counts.csv');actual=load(A/'table10_all_threshold_transition_counts.csv');names=['both_correct','both_wrong','improved','degraded','FN_rescued','FP_suppressed','TN_to_FP','TP_to_FN','persistent_FN','persistent_FP','net'];checked=0
for row in ind.itertuples():
 q=actual[(actual.model_pair==row.model_pair)&np.isclose(actual.threshold,row.threshold)].iloc[0]
 for k in names:assert int(getattr(row,k))==int(q[k]);checked+=1
result={'status':'passed','independent_feature_means':means_checked,'independent_error_stratified_Hedges_g':effects_checked,'independent_threshold_count_values':checked,'method':'Source topology aligned by Name/label/protein then VariationID; independent 8-state truth table; math.fsum mean/variance','Figure3_unmodified':True}
(W/'independent_feature_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
