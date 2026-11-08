#!/usr/bin/env python3
"""Independent compact audit: weighted Mann-Whitney AUC and score-bin AP.
Uses only the delivered frozen data, protocol and result tables; no source inputs,
training packages or sklearn are needed. The first four protein draws in every
group and all percentile intervals are checked against stored bootstrap samples.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent
def read(name):return pd.read_csv(P/name,float_precision='round_trip')
def score(y,p,w=None):
 y=np.asarray(y);p=np.asarray(p);w=np.ones(len(y)) if w is None else np.asarray(w)
 _,idx=np.unique(p,return_inverse=True)
 a=np.bincount(idx,weights=w*y);b=np.bincount(idx,weights=w*(1-y));tp=a.sum();tn=b.sum()
 if not tp or not tn:return np.array([np.nan,np.nan])
 auc=np.sum(a*(np.cumsum(b)-b+.5*b))/(tp*tn)
 ct=np.cumsum(a[::-1]);cf=np.cumsum(b[::-1]);precision=np.divide(ct,ct+cf,out=np.zeros_like(ct),where=(ct+cf)>0)
 ap=np.sum(a[::-1]*precision)/tp
 return np.array([auc,ap])
def main():
 data=read('frozen_18220_figure3_predictions_memberships.csv.gz');assert len(data)==18220 and data.VariationID.is_unique
 protocol=json.loads((P/'frozen_protocol.json').read_text())
 for arm in ['raw_ESM','full_FPP']:
  m=data[[f'{arm}_seed_{s}' for s in [42,43,44]]].to_numpy(dtype=np.float64)
  assert np.array_equal(m.mean(axis=1,dtype=np.float64),data[arm])
  assert data.groupby('STRING_protein_id')[arm].nunique().max()==1
 targets=read('ensemble_metrics_alignment.csv')
 for _,t in targets.iterrows():assert np.allclose(score(data.label,data[t.arm]),[t.auc,t.auprc],atol=1e-12,rtol=0)
 sample_checks=0;membership_checks=0;point_checks=0;ci_checks=0
 for category,defs,summary,samples,idcol in [
  ('joint',protocol['joint_cells'],read('table5_complete_53_joint_cells.csv'),read('all_53_joint_cell_protein_bootstrap_samples.csv.gz'),'cell_index'),
  ('stratum',protocol['one_dimensional_strata'],read('table5_all_18_strata.csv'),read('all_18_strata_protein_bootstrap_samples.csv.gz'),'stratum_index')]:
  for i,s in enumerate(defs):
   sub=data[(data[s['row_col']]==s['row_group'])&(data[s['col_col']]==s['col_group'])] if category=='joint' else data[data[s['group_col']]==s['group']]
   assert len(sub)==s['n'] and sub.label.sum()==s['positive'] and sub.STRING_protein_id.nunique()==s['proteins']
   h=hashlib.sha256('\n'.join(map(str,sorted(sub.VariationID))).encode()).hexdigest();assert h==s['membership_sha256'];membership_checks+=1
   row=summary.iloc[i];a=score(sub.label,sub.raw_ESM);b=score(sub.label,sub.full_FPP)
   assert np.allclose(np.r_[a,b,b-a],row[['raw_auc','raw_auprc','full_auc','full_auprc','delta_auc','delta_auprc']].to_numpy(dtype=float),rtol=0,atol=1e-12);point_checks+=6
   boot=samples[samples[idcol]==i].sort_values('repeat');assert len(boot)==2000
   values=boot[['delta_auc','delta_auprc']].to_numpy();valid=np.isfinite(values).all(axis=1)
   assert valid.sum()==row.finite_replicates and (~valid).sum()==row.discarded_one_class_replicates
   ci=np.quantile(values[valid],[.025,.975],axis=0)
   target=np.array([[row.boot_delta_auc_ci_lower,row.boot_delta_auprc_ci_lower],[row.boot_delta_auc_ci_upper,row.boot_delta_auprc_ci_upper]])
   assert np.allclose(ci,target,rtol=0,atol=1e-12);ci_checks+=4
   proteins,inv=np.unique(sub.STRING_protein_id.to_numpy(dtype=str),return_inverse=True);k=len(proteins)
   seed=20261108+i+(1000 if category=='stratum' else 0);assert seed==row.bootstrap_seed
   rng=np.random.default_rng(seed)
   for j in range(4):
    counts=np.bincount(rng.integers(0,k,size=k),minlength=k);w=counts[inv]
    d=score(sub.label,sub.full_FPP,w)-score(sub.label,sub.raw_ESM,w)
    assert w.sum()==boot.iloc[j].resampled_variants
    assert np.allclose(d,values[j],rtol=0,atol=1e-12,equal_nan=True);sample_checks+=1
 selected=read('table5_selected_original_four_cells.csv');joint=read('table5_complete_53_joint_cells.csv')
 for i,s in enumerate(protocol['fixed_original_displayed_cells']):
  sr=selected.iloc[i];assert sr.cell_index==s['cell_index'] and sr.display_id==s['display_id'] and sr.cell_name==s['cell_name']
  for c in ['n','proteins','delta_auc','delta_auprc','boot_delta_auc_ci_lower','boot_delta_auc_ci_upper']:
   assert sr[c]==joint[joint.cell_index==s['cell_index']].iloc[0][c]
 allmembers=read('all_53_joint_cell_memberships.csv.gz')
 for i,s in enumerate(protocol['joint_cells']):
  g=allmembers[allmembers.cell_index==i]
  assert hashlib.sha256('\n'.join(map(str,sorted(g.VariationID))).encode()).hexdigest()==s['membership_sha256']
 audit={'passed':True,'rows':len(data),'group_membership_checks':membership_checks,'point_metric_checks':point_checks,'percentile_endpoint_checks':ci_checks,'independent_weighted_bootstrap_draw_checks':sample_checks,'selected_objects_unchanged':True,'all53_explicit_memberships_verified':True,'metric_algorithm':'Weighted Mann-Whitney pair counts and distinct-score noninterpolated AP, independently of production trapezoidal AUC','full_bootstrap_rerun':False}
 (P/'independent_compact_verification.json').write_text(json.dumps(audit,indent=2)+'\n');print(json.dumps(audit,indent=2))
if __name__=='__main__':main()
