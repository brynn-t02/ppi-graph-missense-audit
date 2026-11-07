"""Verify the supplied frozen Figure4 data; no training, inference or bootstrap.
Run: python verify_compact.py  (requires numpy and pandas).
"""
from pathlib import Path
import json,math
import numpy as np
import pandas as pd
P=Path(__file__).resolve().parent/'analysis'
read=lambda f:pd.read_csv(P/f,float_precision='round_trip')
d=read('frozen_18220_predictions_topology.csv.gz');assert len(d)==18220 and d.VariationID.is_unique
assert d.label.isin([0,1]).all() and d.STRING_protein_id.notna().all()
y=d.label.to_numpy(dtype=int);arms=['raw_ESM','full_FPP','scores','fusion']
metrics=read('ensemble_metrics_alignment.csv').set_index('alias')
for a in arms:
 p=d[[a+'_seed_'+str(s) for s in [42,43,44]]].to_numpy(dtype=np.float64).mean(1,dtype=np.float64)
 assert np.array_equal(p,d[a].to_numpy())
 if a in arms[:2]:assert (d.groupby('STRING_protein_id')[[a]+[a+'_seed_'+str(s) for s in [42,43,44]]].nunique()==1).all().all()
 ranks=pd.Series(p).rank().to_numpy();pos=int(y.sum());neg=len(y)-pos;auc=(ranks[y==1].sum()-pos*(pos+1)/2)/(pos*neg)
 q=pd.DataFrame({'p':p,'y':y}).groupby('p').agg(n=('y','size'),positive=('y','sum')).iloc[::-1];ap=(q.positive/pos*q.positive.cumsum()/q.n.cumsum()).sum()
 assert abs(auc-metrics.loc[a,'canonical_auc'])<1e-12 and abs(ap-metrics.loc[a,'canonical_auprc'])<1e-12
profiles=read('table6_all_transition_feature_means.csv');effect=read('table6_all_44_error_stratified_contrasts.csv');thresholds=read('table10_all_threshold_transition_counts.csv')
features=[c[5:] for c in profiles.columns if c.startswith('mean_')]
checks={'metric_pairs':4,'seed_predictions':12,'rows':18220,'feature_means':0,'effect_sizes':0,'threshold_partitions':0}
for pair,a,b in [('esm_vs_gcn','raw_ESM','full_FPP'),('cadd_am_vs_full','scores','fusion')]:
 def masks(t):
  old=d[a].to_numpy()>=t;new=d[b].to_numpy()>=t;positive=y==1;bc=old==positive;nc=new==positive
  return {'both_correct':bc&nc,'both_wrong':~bc&~nc,'improved':~bc&nc,'degraded':bc&~nc,'FN_rescued':positive&~old&new,'FP_suppressed':~positive&old&~new,'TN_to_FP':~positive&~old&new,'TP_to_FN':positive&old&~new,'persistent_FN':positive&~old&~new,'persistent_FP':~positive&old&new}
 for row in thresholds[thresholds.model_pair==pair].itertuples():
  ms=masks(row.threshold);assert sum(ms[k].sum() for k in ['both_correct','both_wrong','improved','degraded'])==18220
  for k,m in ms.items():assert int(m.sum())==int(getattr(row,k))
  assert ms['improved'].sum()==ms['FN_rescued'].sum()+ms['FP_suppressed'].sum()
  assert ms['degraded'].sum()==ms['TN_to_FP'].sum()+ms['TP_to_FN'].sum();checks['threshold_partitions']+=1
 ms=masks(.5)
 for row in profiles[profiles.model_pair==pair].itertuples():
  m=ms[row.group_name];assert m.sum()==row.n and abs(y[m].mean()-row.pos_rate)<1e-12
  for f in features:
   v=d.loc[m,f].to_numpy(dtype=float);v=v[np.isfinite(v)];mean=math.fsum(v)/len(v)
   assert np.isclose(mean,getattr(row,'mean_'+f),atol=1e-10,rtol=1e-12);checks['feature_means']+=1
 for row in effect[effect.model_pair==pair].itertuples():
  a,b=('FN_rescued','persistent_FN') if row.condition.startswith('FN') else ('FP_suppressed','persistent_FP')
  x=d.loc[ms[a],row.feature].dropna().to_numpy();z=d.loc[ms[b],row.feature].dropna().to_numpy();n,m=len(x),len(z);mx=math.fsum(x)/n;mz=math.fsum(z)/m
  ss=math.fsum((v-mx)**2 for v in x)+math.fsum((v-mz)**2 for v in z);g=(1-3/(4*(n+m)-9))*(mx-mz)/math.sqrt(ss/(n+m-2))
  assert (n,m)==(row.n_left,row.n_right) and abs(g-row.hedges_g)<1e-12;checks['effect_sizes']+=1
print(json.dumps({'status':'passed',**checks},indent=2))
