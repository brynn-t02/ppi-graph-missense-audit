"""Paired protein-cluster bootstrap of real fitted prediction ensembles.
No refitting or independent-cohort inference. Permutation spread stays separated.
"""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import roc_auc_score,average_precision_score
B=Path('/Data/u40902/TYX/ppi_audit_20261024');O=B/'outputs'
assert json.loads((O/'progress.json').read_text())['completed']==48
runs=pd.read_csv(O/'per_run_metrics.csv');assert len(runs)==48
for split in ['original','original_fusion','homology']:
 for arm,g in runs[runs.split==split].groupby('arm'):assert sorted(g.seed.tolist())==[42,43,44]
def ensemble(split,arm):
 return np.mean([np.load(O/'runs'/split/arm/f'seed_{s}'/'test_probs.npy') for s in [42,43,44]],axis=0)
reps={a:ensemble('original',a) for a in runs[runs.split=='original'].arm.unique()}
fusion={a:ensemble('original_fusion',a) for a in runs[runs.split=='original_fusion'].arm.unique()}
fusion['scores_plus_permuted_FPP_graphmean']=np.mean([fusion[f'scores_plus_permuted_FPP_g{i}'] for i in range(3)],axis=0)
hom={a:ensemble('homology',a) for a in runs[runs.split=='homology'].arm.unique()}
# Average over the frozen 3 graph permutations and 3 classifier seeds, retaining graph/seed hierarchy in per-run outputs.
reps['permuted_FPP_graphmean']=np.mean([reps[f'permuted_FPP_g{i}'] for i in range(3)],axis=0)
comparisons={'original':[('full_FPP','raw_ESM'),('full_FPP','edge_ablated_FPP'),('full_FPP','permuted_FPP_graphmean')],
 'original_fusion':[('scores_plus_full_FPP','scores_capacity_matched'),('scores_plus_full_FPP','scores_plus_raw_ESM'),('scores_plus_full_FPP','scores_plus_edge_ablated_FPP'),('scores_plus_full_FPP','scores_plus_permuted_FPP_graphmean')],
 'homology':[('scores_plus_full_FPP','scores_capacity_matched'),('scores_plus_raw_ESM','scores_capacity_matched'),('scores_plus_full_FPP','scores_plus_raw_ESM')]}
result={'bootstrap_repeats':2000,'seed':20261024,'resampling_unit':'STRING protein, all variants retained per sampled protein with multiplicity','scope':'Intervals conditional on fitted ensembles and frozen graph permutations; omit retraining and new-graph uncertainty. Exploratory comparisons, no multiplicity adjustment.','comparisons':{}}
rng=np.random.default_rng(20261024)
for split,pred in [('original',reps),('original_fusion',fusion),('homology',hom)]:
 rows=pd.read_csv(O/f'{"original" if split=="original_fusion" else split}_test_rows.csv');y=rows.Label.to_numpy();ids=rows.STRING_protein_id.astype(str).to_numpy();uniq,inv=np.unique(ids,return_inverse=True);n=len(uniq)
 cache={}
 for name,p in pred.items():
  assert len(p)==len(y) and np.isfinite(p).all();order=np.argsort(p,kind='stable');end=np.r_[p[order][1:]!=p[order][:-1],True];ends=np.flatnonzero(end);cache[name]=(order,ends)
 def weighted_scores(p,w,name):
  order,ends=cache[name];yy=y[order];ww=w[order];tp=np.cumsum(ww*yy)[ends];fp=np.cumsum(ww*(1-yy))[ends];P=tp[-1];N=fp[-1]
  if P==0 or N==0:return np.array([np.nan,np.nan])
  auc=np.trapz(np.r_[0,tp/P],np.r_[0,fp/N]);# ascending-score cumulative is low-score selected, so complement area
  auc=1-auc
  # AP uses descending-score grouped cumulative sums.
  pos=np.diff(np.r_[0,tp]);neg=np.diff(np.r_[0,fp]);pos=pos[::-1];neg=neg[::-1];ctp=np.cumsum(pos);cfp=np.cumsum(neg);ap=np.sum((pos/P)*np.divide(ctp,ctp+cfp,out=np.zeros_like(ctp),where=ctp+cfp>0))
  return np.array([auc,ap])
 ones=np.ones(len(y))
 for name,p in pred.items():assert np.allclose(weighted_scores(p,ones,name),[roc_auc_score(y,p),average_precision_score(y,p)],atol=1e-12)
 # A spot check against sklearn sample_weight validates tied-score weighting.
 checkw=rng.integers(0,4,size=n)[inv].astype(float)
 for name,p in pred.items():assert np.allclose(weighted_scores(p,checkw,name),[roc_auc_score(y,p,sample_weight=checkw),average_precision_score(y,p,sample_weight=checkw)],atol=1e-10)
 samples={f'{a}_vs_{b}':[] for a,b in comparisons[split]}
 for t in range(2000):
  w=np.bincount(rng.integers(0,n,size=n),minlength=n)[inv].astype(float)
  values={name:weighted_scores(p,w,name) for name,p in pred.items()}
  for a,b in comparisons[split]:samples[f'{a}_vs_{b}'].append(values[a]-values[b])
  if (t+1)%500==0:print(split,'bootstrap',t+1,flush=True)
 if split=='original':
  vv=[]
  for t in range(2000):
   w=np.bincount(rng.integers(0,len(y),size=len(y)),minlength=len(y)).astype(float)
   vv.append(weighted_scores(pred['full_FPP'],w,'full_FPP')-weighted_scores(pred['raw_ESM'],w,'raw_ESM'))
  qv=np.quantile(np.asarray(vv),[.025,.975],axis=0)
  result['variant_bootstrap_original_full_vs_raw']={'auc_ci95':qv[:,0].tolist(),'auprc_ci95':qv[:,1].tolist(),'repeats':2000,'scope':'Variant resampling ignores within-protein dependence; descriptive contrast with protein-cluster resampling.'}
 result['comparisons'][split]={}
 for a,b in comparisons[split]:
  key=f'{a}_vs_{b}';vals=np.asarray(samples[key]);point=weighted_scores(pred[a],ones,a)-weighted_scores(pred[b],ones,b);q=np.nanquantile(vals,[.025,.975],axis=0)
  result['comparisons'][split][key]={'delta_auc':float(point[0]),'auc_ci95':[float(q[0,0]),float(q[1,0])],'delta_auprc':float(point[1]),'auprc_ci95':[float(q[0,1]),float(q[1,1])],'finite_replicates':int(np.isfinite(vals).all(1).sum()),'proteins':n}
# Paired seed deltas for primary three-arm comparisons; average permutation metrics within seed first.
seedrows=[]
for split,pairs in comparisons.items():
 for a,b in pairs:
  for seed in [42,43,44]:
   def row(arm):
    filt=runs[(runs.split==split)&(runs.seed==seed)]
    if arm.endswith('permuted_FPP_graphmean'):return filt[filt.arm.str.contains('permuted_FPP_g')][['test_auc','test_auprc']].mean()
    return filt[filt.arm==arm].iloc[0][['test_auc','test_auprc']].astype(float)
   v=row(a)-row(b);seedrows.append({'split':split,'comparison':f'{a}_vs_{b}','seed':seed,'delta_auc':float(v.test_auc),'delta_auprc':float(v.test_auprc)})
pd.DataFrame(seedrows).to_csv(O/'paired_seed_deltas.csv',index=False)
(O/'cluster_bootstrap.json').write_text(json.dumps(result,indent=2)+'\n')
# Re-evaluate current fitted fusion predictions on the existing complete-score test subset.
rows=pd.read_csv(O/'original_test_rows.csv')
cadd=pd.read_csv(B/'data/clinvar_with_cadd.csv',usecols=['VariationID','CADD_PHRED'])
am=pd.read_csv(B/'data/clinvar_with_alphamissense.csv',usecols=['VariationID','am_pathogenicity'])
cc=rows.merge(cadd,on='VariationID',how='left',sort=False,validate='one_to_one').merge(am,on='VariationID',how='left',sort=False,validate='one_to_one')
assert np.array_equal(cc.VariationID.to_numpy(),rows.VariationID.to_numpy())
mask=cc.CADD_PHRED.notna().to_numpy() & cc.am_pathogenicity.notna().to_numpy();assert int(mask.sum())==17581
y=rows.Label.to_numpy();ccout=[]
for _,r in runs[runs.split=='original_fusion'].iterrows():
 p=np.load(O/'runs/original_fusion'/r.arm/f'seed_{r.seed}'/'test_probs.npy')
 assert np.allclose([roc_auc_score(y,p),average_precision_score(y,p)],[r.test_auc,r.test_auprc],atol=1e-12)
 ccout.append({'arm':r.arm,'seed':int(r.seed),'complete_rows':int(mask.sum()),'all_auc':r.test_auc,'all_auprc':r.test_auprc,'complete_auc':float(roc_auc_score(y[mask],p[mask])),'complete_auprc':float(average_precision_score(y[mask],p[mask]))})
pd.DataFrame(ccout).to_csv(O/'current_complete_case_per_run.csv',index=False)
(O/'analysis_complete.json').write_text(json.dumps({'complete':True,'runs':48,'bootstrap_repeats':2000,'time':time.time()})+'\n')
print(json.dumps(result,indent=2),flush=True)
