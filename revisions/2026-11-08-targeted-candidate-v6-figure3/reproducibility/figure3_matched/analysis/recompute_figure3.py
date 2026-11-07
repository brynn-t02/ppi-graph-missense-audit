#!/usr/bin/env python3
"""Figure 3 only: frozen matched predictions, original groups, protein bootstrap.
No training, inference, graph processing, feature extraction or model selection.
Run freeze_figure3_protocol.py first. Writes only beside this script.
"""
from pathlib import Path
import gzip, hashlib, io, json, platform, time
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[2]
BASE=ROOT/'outputs/BIBM_workshop_font_clean_20261025/source'
CAN=BASE/'canonical_analysis_20261025';PUB=CAN/'public'
MATCH=BASE/'matched_validation_20261024';OLD=BASE/'figure_inputs/data/processed'
LOCAL=Path('/Users/teyian/Documents/paper1/server_snapshot_full/root/data/processed')
DATA=OUT/'data/processed'
SEEDS=[42,43,44]
FEATURES=['degree_bucket','kcore_bucket','neighbor_risk_bucket','neighbor_pathogenic_rate','neighbor_train_variant_count','betweenness_bucket','betweenness','clustering_bucket','clustering_coefficient','boundary_role_bucket','within_comm_neighbor_fraction','community_id']
INPUTS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def register(p):
 p=Path(p);INPUTS[str(p)]={'sha256':sha(p),'bytes':p.stat().st_size};return p
def read(p,**kw):return pd.read_csv(register(p),float_precision='round_trip',**kw)
def read_json(p):return json.loads(register(p).read_text())
def dump(p,v):
 p=Path(p);assert p.is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True)
 p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def save(df,p):
 p=Path(p);assert p.is_relative_to(OUT);p.parent.mkdir(parents=True,exist_ok=True)
 df.to_csv(p,index=False,float_format='%.17g');assert p.stat().st_size<5_000_000
def compact(df,p):
 p=Path(p);assert p.is_relative_to(OUT)
 with p.open('wb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as z:
   with io.TextIOWrapper(z,encoding='utf-8',newline='') as t:df.to_csv(t,index=False,float_format='%.17g')
 assert p.stat().st_size<5_000_000,(p,p.stat().st_size)
class WeightedScores:
 """Tie-aware ROC AUC and non-interpolated AP with variant multiplicities."""
 def __init__(self,y,p):
  self.y=np.asarray(y,dtype=np.int8);self.p=np.asarray(p,dtype=np.float64)
  self.order=np.argsort(self.p,kind='stable');s=self.p[self.order]
  self.ends=np.flatnonzero(np.r_[s[1:]!=s[:-1],True]);self.sy=self.y[self.order]
 def __call__(self,w):
  w=np.asarray(w,dtype=np.float64)[self.order]
  tp=np.cumsum(w*self.sy)[self.ends];fp=np.cumsum(w*(1-self.sy))[self.ends]
  pos,neg=tp[-1],fp[-1]
  if pos==0 or neg==0:return np.array([np.nan,np.nan])
  auc=1-np.trapezoid(np.r_[0,tp/pos],np.r_[0,fp/neg])
  ps=np.diff(np.r_[0,tp])[::-1];ns=np.diff(np.r_[0,fp])[::-1]
  ct,cf=np.cumsum(ps),np.cumsum(ns)
  precision=np.divide(ct,ct+cf,out=np.zeros_like(ct),where=(ct+cf)>0)
  return np.array([auc,np.sum(ps/pos*precision)])
def metrics(y,p):return WeightedScores(y,p)(np.ones(len(y)))
def brute_metrics(y,p):
 y=np.asarray(y);p=np.asarray(p);pos=p[y==1];neg=p[y==0]
 if not len(pos) or not len(neg):return np.array([np.nan,np.nan])
 auc=np.mean((pos[:,None]>neg[None,:])+.5*(pos[:,None]==neg[None,:]))
 ap,last=0.,0.
 for v in np.unique(p)[::-1]:
  active=p>=v;tp=y[active].sum();rec=tp/y.sum()
  ap+=(rec-last)*tp/active.sum();last=rec
 return np.array([auc,ap])
def group_metrics(df):
 y=df.label.to_numpy();r=metrics(y,df.raw_ESM);f=metrics(y,df.full_FPP)
 return {'n':len(df),'proteins':int(df.STRING_protein_id.nunique()),'positive':int(y.sum()),'negative':int(len(y)-y.sum()),'positive_carrying_proteins':int(df.loc[df.label==1,'STRING_protein_id'].nunique()),'negative_carrying_proteins':int(df.loc[df.label==0,'STRING_protein_id'].nunique()),'pos_rate':float(y.mean()),'raw_auc':float(r[0]),'raw_auprc':float(r[1]),'full_auc':float(f[0]),'full_auprc':float(f[1]),'delta_auc':float(f[0]-r[0]),'delta_auprc':float(f[1]-r[1]),'delta_brier':float(np.mean((df.full_FPP-y)**2)-np.mean((df.raw_ESM-y)**2)),'mean_gcn_minus_esm_prob':float(np.mean(df.full_FPP-df.raw_ESM))}
def bootstrap(df,seed,repeats=2000):
 # Stable sorted protein order is part of the reproducibility convention.
 proteins,inv=np.unique(df.STRING_protein_id.to_numpy(dtype=str),return_inverse=True)
 k=len(proteins);y=df.label.to_numpy();r=WeightedScores(y,df.raw_ESM);f=WeightedScores(y,df.full_FPP)
 rng=np.random.default_rng(seed);vals=np.empty((repeats,2));size=np.empty(repeats,dtype=int)
 checks=[]
 for i in range(repeats):
  counts=np.bincount(rng.integers(0,k,size=k),minlength=k);w=counts[inv]
  vals[i]=f(w)-r(w);size[i]=w.sum()
  # Independent explicit-row expansion on initial draws in each group.
  if i<2:
   ix=np.repeat(np.arange(len(df)),w)
   expanded=metrics(y[ix],df.full_FPP.to_numpy()[ix])-metrics(y[ix],df.raw_ESM.to_numpy()[ix])
   assert np.allclose(vals[i],expanded,rtol=0,atol=1e-12,equal_nan=True)
   checks.append(True)
 ok=np.isfinite(vals).all(1)
 ci=np.quantile(vals[ok],[.025,.975],axis=0) if ok.any() else np.full((2,2),np.nan)
 return {'boot_delta_auc_ci_lower':float(ci[0,0]),'boot_delta_auc_ci_upper':float(ci[1,0]),'boot_delta_auprc_ci_lower':float(ci[0,1]),'boot_delta_auprc_ci_upper':float(ci[1,1]),'bootstrap_repeats':repeats,'bootstrap_seed':seed,'finite_replicates':int(ok.sum()),'discarded_one_class_replicates':int((~ok).sum()),'bootstrap_unit':'protein','resample_variant_n_min':int(size.min()),'resample_variant_n_max':int(size.max()),'explicit_expansion_checks':len(checks)},vals,size
def memberhash(sub):return hashlib.sha256('\n'.join(map(str,sorted(sub.VariationID))).encode()).hexdigest()
def main():
 start=time.time();protocol=read_json(OUT/'frozen_protocol.json')
 for p,h in protocol['source_input_sha256'].items():assert sha(p)==h,p
 topo=read(LOCAL/'topology_profile_joint_map_exp33/merged_topology_profile_features.csv')
 test=read(LOCAL/'clinvar_test_main.csv',usecols=['VariationID','Name','Label','STRING_protein_id'])
 assert len(topo)==len(test)==18220 and test.VariationID.is_unique
 for c in ['Name','Label','STRING_protein_id']:assert np.array_equal(test[c],topo[c])
 frozen=read(PUB/'frozen_samples_177213.csv.gz');ft=frozen[frozen.original_split=='test'].set_index('VariationID').loc[test.VariationID]
 assert len(frozen)==177213 and frozen.VariationID.is_unique and len(ft)==18220
 assert np.array_equal(ft.label,test.Label) and np.array_equal(ft.STRING_protein_id,test.STRING_protein_id)
 original_rows=read(MATCH/'original_test_rows.csv')
 for c in ['VariationID','Label','STRING_protein_id']:assert np.array_equal(original_rows[c],test[c])
 df=test.rename(columns={'Label':'label'}).copy()
 for c in FEATURES:
  df[c]=topo[c].to_numpy();assert df.groupby('STRING_protein_id')[c].nunique(dropna=False).max()==1
 public_manifest=read_json(CAN/'public_manifest.json');pubhashes={r['file']:r['sha256'] for r in public_manifest['files']}
 assert sha(PUB/'frozen_samples_177213.csv.gz')==pubhashes['frozen_samples_177213.csv.gz']
 source_manifest=read_json(CAN/'source_manifest.json');inference=read_json(CAN/'canonical_inference_audit.json')
 verification=read_json(MATCH/'completed_run_verification.json');assert verification['passed']
 targets=read(CAN/'ensemble_metrics.csv');seed_targets=read(CAN/'evaluation_per_seed.csv')
 runrows=[];em=[]
 for arm in ['raw_ESM','full_FPP']:
  filename=f'test_predictions__original__{arm}.csv.gz';p=read(PUB/filename);assert sha(PUB/filename)==pubhashes[filename]
  for c in ['VariationID','label','STRING_protein_id']:assert np.array_equal(p[c],df[c])
  matrix=p[[f'p_seed_{s}' for s in SEEDS]].to_numpy(dtype=np.float64)
  assert np.isfinite(matrix).all() and ((matrix>=0)&(matrix<=1)).all()
  df[arm]=matrix.mean(axis=1,dtype=np.float64)
  assert np.array_equal(df[arm].to_numpy(),p.ensemble_float64.to_numpy())
  assert df.groupby('STRING_protein_id')[arm].nunique().max()==1
  for j,seed in enumerate(SEEDS):
   df[f'{arm}_seed_{seed}']=matrix[:,j];assert df.groupby('STRING_protein_id')[f'{arm}_seed_{seed}'].nunique().max()==1
   actual=metrics(df.label,matrix[:,j]);t=seed_targets[(seed_targets.population=='original')&(seed_targets.arm==arm)&(seed_targets.seed==seed)].iloc[0]
   assert np.allclose(actual,[t.auc,t.auprc],rtol=0,atol=1e-12)
   rp=MATCH/'runs/original'/arm/f'seed_{seed}';record=read_json(rp/'metrics.json');history=read_json(rp/'history.json')
   assert record['best_epoch']==max(history,key=lambda r:r['val_auc'])['epoch']
   assert record['epochs_run']==len(history)
   cp=f'assets/outputs/runs/original/{arm}/seed_{seed}/model.pt';cph=source_manifest['selected_checkpoints'][cp]
   run=[r for r in inference['runs'] if r['arm']==arm and r['seed']==seed][0]
   assert run['checkpoint_sha256']==cph and run['parts']['test']['canonical_max_within_protein_spread']==0
   runrows.append({'arm':arm,'seed':seed,'selected_epoch':record['best_epoch'],'epochs_run':record['epochs_run'],'canonical_auc':actual[0],'canonical_auprc':actual[1],'recorded_checkpoint_sha256':cph,'history_selection_verified':True,'per_protein_probability_verified':True})
  actual=metrics(df.label,df[arm]);target=targets[(targets.population=='original')&(targets.arm==arm)].iloc[0]
  assert np.allclose(actual,[target.auc,target.auprc],rtol=0,atol=1e-12)
  em.append({'arm':arm,'auc':actual[0],'auprc':actual[1],'canonical_auc':target.auc,'canonical_auprc':target.auprc,'exact_float64_mean_match':True})
 for arm in ['raw_ESM','full_FPP']:
  for offset in [0,7,29]:
   ix=np.arange(offset,len(df),83)
   assert np.allclose(metrics(df.label.iloc[ix],df[arm].iloc[ix]),brute_metrics(df.label.iloc[ix],df[arm].iloc[ix]),rtol=0,atol=1e-12)
 for y,p in [([0,1,1,0],[.5,.5,.5,.5]),([0,1,0,1],[.1,.1,.8,.8])]:assert np.allclose(metrics(y,p),brute_metrics(y,p),rtol=0,atol=1e-12)
 compact(df,OUT/'frozen_18220_figure3_predictions_memberships.csv.gz')
 save(pd.DataFrame(runrows),OUT/'six_seed_provenance.csv');save(pd.DataFrame(em),OUT/'ensemble_metrics_alignment.csv')
 strata=[];sboot=[];joint=[];jboot=[];membership=[]
 for i,s in enumerate(protocol['one_dimensional_strata']):
  sub=df[df[s['group_col']]==s['group']].reset_index(drop=True)
  assert len(sub)==s['n'] and int(sub.label.sum())==s['positive'] and memberhash(sub)==s['membership_sha256']
  ci,v,n=bootstrap(sub,20261108+1000+i)
  strata.append({**s,**group_metrics(sub),**ci})
  sboot.append(pd.DataFrame({'stratum_index':i,'repeat':np.arange(1,2001),'delta_auc':v[:,0],'delta_auprc':v[:,1],'resampled_variants':n}))
  print(f'stratum {i+1}/18',flush=True)
 for i,s in enumerate(protocol['joint_cells']):
  sub=df[(df[s['row_col']]==s['row_group'])&(df[s['col_col']]==s['col_group'])].reset_index(drop=True)
  assert len(sub)==s['n'] and int(sub.label.sum())==s['positive'] and memberhash(sub)==s['membership_sha256']
  ci,v,n=bootstrap(sub,20261108+i)
  joint.append({**s,**group_metrics(sub),**ci})
  jboot.append(pd.DataFrame({'cell_index':i,'repeat':np.arange(1,2001),'delta_auc':v[:,0],'delta_auprc':v[:,1],'resampled_variants':n}))
  membership.append(sub[['VariationID','label','STRING_protein_id']].assign(cell_index=i))
  if i%5==0:print(f'joint {i+1}/53',flush=True)
 strata=pd.DataFrame(strata);joint=pd.DataFrame(joint)
 save(strata,OUT/'table5_all_18_strata.csv');save(joint,OUT/'table5_complete_53_joint_cells.csv')
 compact(pd.concat(sboot,ignore_index=True),OUT/'all_18_strata_protein_bootstrap_samples.csv.gz')
 compact(pd.concat(jboot,ignore_index=True),OUT/'all_53_joint_cell_protein_bootstrap_samples.csv.gz')
 compact(pd.concat(membership,ignore_index=True),OUT/'all_53_joint_cell_memberships.csv.gz')
 selected=[]
 for s in protocol['fixed_original_displayed_cells']:
  r=joint[joint.cell_index==s['cell_index']].iloc[0].to_dict();r.update(s);r['point_delta_auc']=r['delta_auc'];r['point_delta_auprc']=r['delta_auprc'];selected.append(r)
 selected=pd.DataFrame(selected);save(selected,OUT/'table5_selected_original_four_cells.csv')
 save(joint,DATA/'topology_profile_joint_map_exp33/joint_gain_maps.csv')
 save(selected,DATA/'topology_profile_joint_map_bootstrap_exp33B/key_joint_cells_bootstrap_summary.csv')
 save(strata[strata.group_col.isin(['degree_bucket','kcore_bucket'])],DATA/'topology_gain_analysis_exp26B/topology_stratified_metrics.csv')
 for col,rel in [('betweenness_bucket','betweenness_gain_analysis_exp30/betweenness_stratified_metrics.csv'),('clustering_bucket','clustering_gain_analysis_exp31/clustering_stratified_metrics.csv'),('neighbor_risk_bucket','neighborhood_risk_enrichment_exp27/neighbor_risk_stratified_metrics.csv')]:save(strata[strata.group_col==col],DATA/rel)
 oldjoint=read(OLD/'topology_profile_joint_map_exp33/joint_gain_maps.csv')
 keys=['row_col','row_group','col_col','col_group']
 before=oldjoint[keys+['delta_auc','delta_auprc']].merge(joint[keys+['delta_auc','delta_auprc']],on=keys,validate='one_to_one',suffixes=('_before','_after'))
 before['auc_sign_changed']=np.sign(before.delta_auc_before)!=np.sign(before.delta_auc_after)
 before['auprc_sign_changed']=np.sign(before.delta_auprc_before)!=np.sign(before.delta_auprc_after)
 save(before,OUT/'before_after_all_53_cells.csv')
 oldselected=read(OLD/'topology_profile_joint_map_bootstrap_exp33B/key_joint_cells_bootstrap_summary.csv')
 bsel=oldselected[['cell_name','point_delta_auc','point_delta_auprc','boot_delta_auc_ci_lower','boot_delta_auc_ci_upper']].merge(selected[['cell_name','point_delta_auc','point_delta_auprc','boot_delta_auc_ci_lower','boot_delta_auc_ci_upper']],on='cell_name',validate='one_to_one',suffixes=('_before','_after'))
 bsel['auc_sign_changed']=np.sign(bsel.point_delta_auc_before)!=np.sign(bsel.point_delta_auc_after);save(bsel,OUT/'before_after_selected_cells.csv')
 oldstrata=[]
 for rel in ['topology_gain_analysis_exp26B/topology_stratified_metrics.csv','neighborhood_risk_enrichment_exp27/neighbor_risk_stratified_metrics.csv','betweenness_gain_analysis_exp30/betweenness_stratified_metrics.csv','clustering_gain_analysis_exp31/clustering_stratified_metrics.csv','community_boundary_gain_analysis_exp32/community_boundary_stratified_metrics.csv']:
  path=OLD/rel
  if not path.exists():path=LOCAL/rel
  old=read(path);old['group']=old['group_id'] if 'group_id' in old else old['group']
  oldstrata.append(old)
 os=pd.concat(oldstrata,ignore_index=True);os=os[os.group_col.isin(strata.group_col)]
 bs=os[['group_col','group','delta_auc','delta_auprc']].merge(strata[['group_col','group','delta_auc','delta_auprc']],on=['group_col','group'],validate='one_to_one',suffixes=('_before','_after'))
 assert len(bs)==18;bs['auc_sign_changed']=np.sign(bs.delta_auc_before)!=np.sign(bs.delta_auc_after);bs['auprc_sign_changed']=np.sign(bs.delta_auprc_before)!=np.sign(bs.delta_auprc_after);save(bs,OUT/'before_after_all_18_strata.csv')
 excluded=[]
 for c in ['neighbor_risk_bucket','boundary_role_bucket']:
  sub=df[df[c]==-1];excluded.append({'group_col':c,'code':-1,'n':len(sub),'proteins':int(sub.STRING_protein_id.nunique()),'reason':'Original insufficient-signal/unassigned category below original 50-variant threshold; no category reassignment'})
 save(pd.DataFrame(excluded),OUT/'original_excluded_strata.csv')
 audit={'rows':len(df),'proteins':int(df.STRING_protein_id.nunique()),'positives':int(df.label.sum()),'stable_ID_label_protein_alignment':True,'canonical_three_seed_float64_mean_exact':True,'per_protein_scores_constant':True,'six_seed_metric_and_history_checks':6,'two_ensemble_metrics_match':True,'canonical_metric_tolerance':1e-12,'original_53_cells_18_strata_and_four_selected_frozen':True,'topology_features_reused_without_recomputation':True,'new_bootstrap_runs':71,'bootstrap_repeats_each':2000,'bootstrap_unit':'protein','paired_model_weights':True,'explicit_row_expansion_checks':142,'stopped_draft_bootstrap_reused':False,'all_joint_point_estimates_finite':bool(np.isfinite(joint[['delta_auc','delta_auprc']].to_numpy()).all()),'joint_auc_direction_flips':int(before.auc_sign_changed.sum()),'joint_auprc_direction_flips':int(before.auprc_sign_changed.sum()),'strata_auc_direction_flips':int(bs.auc_sign_changed.sum()),'strata_auprc_direction_flips':int(bs.auprc_sign_changed.sum()),'joint_one_class_replicates_total':int(joint.discarded_one_class_replicates.sum()),'strata_one_class_replicates_total':int(strata.discarded_one_class_replicates.sum()),'joint_min_proteins':int(joint.proteins.min()),'joint_min_positive_carrying_proteins':int(joint.positive_carrying_proteins.min()),'min_joint_finite_replicates':int(joint.finite_replicates.min()),'no_training_inference_features_graph_or_server':True,'provenance_limit':'Actual stored predictions, run histories and prior checkpoint hashes checked; weights were not reaccessed and inference was not repeated.','python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'seconds':time.time()-start}
 dump(OUT/'alignment_and_statistical_audit.json',audit);dump(OUT/'input_sha256.json',INPUTS)
 print(json.dumps(audit,indent=2));print(selected[['display_id','n','proteins','delta_auc','delta_auprc','boot_delta_auc_ci_lower','boot_delta_auc_ci_upper']].to_string(index=False))
if __name__=='__main__':main()
