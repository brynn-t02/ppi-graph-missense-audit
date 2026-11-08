from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent;C=R/'outputs/BIBM_workshop_font_clean_20261025/source/canonical_analysis_20261025'
paths={'raw_ESM':'test_predictions__original__raw_ESM.csv.gz','full_FPP':'test_predictions__original__full_FPP.csv.gz','scores':'test_predictions__original_fusion__scores_capacity_matched.csv.gz','fusion':'test_predictions__original_fusion__scores_plus_full_FPP.csv.gz'}
frozen=pd.read_csv(C/'public/frozen_samples_177213.csv.gz');frozen=frozen[frozen.original_split=='test'].set_index('VariationID').sort_index();assert len(frozen)==18220 and frozen.index.is_unique
frames={};results={};hashes={}
for arm,name in paths.items():
 path=C/'public'/name;d=pd.read_csv(path,float_precision='round_trip').set_index('VariationID').sort_index();assert d.index.equals(frozen.index)
 assert np.array_equal(d.label,frozen.label) and np.array_equal(d.STRING_protein_id,frozen.STRING_protein_id)
 ps=d[[f'p_seed_{s}' for s in [42,43,44]]].astype('float64');p=ps.mean(axis=1).to_numpy(dtype=np.float64);assert np.array_equal(p,d.ensemble_float64.to_numpy())
 if arm in ['raw_ESM','full_FPP']:assert (d.groupby('STRING_protein_id')[list(ps.columns)].nunique()==1).all().all()
 y=d.label.to_numpy();n1=int(y.sum());n0=len(y)-n1
 ranks=pd.Series(p).rank(method='average').to_numpy();auc=(ranks[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
 groups=pd.DataFrame({'prob':p,'positive':y}).groupby('prob',sort=True).agg(positives=('positive','sum'),total=('positive','size')).iloc[::-1]
 ap=float((groups.positives/n1*(groups.positives.cumsum()/groups.total.cumsum())).sum())
 results[arm]={'auc':float(auc),'auprc':ap};d['mean_probability']=p;frames[arm]=d;hashes[str(path.relative_to(R))]=hashlib.sha256(path.read_bytes()).hexdigest()
em=pd.read_csv(C/'ensemble_metrics.csv',float_precision='round_trip')
for a,pop,name in [('raw_ESM','original','raw_ESM'),('full_FPP','original','full_FPP'),('scores','original_fusion','scores_capacity_matched'),('fusion','original_fusion','scores_plus_full_FPP')]:
 q=em[(em.population==pop)&(em.arm==name)].iloc[0]
 assert abs(results[a]['auc']-q.auc)<1e-12 and abs(results[a]['auprc']-q.auprc)<1e-12
counts={};thresholds=[]
for pair,a,b in [('esm_vs_gcn','raw_ESM','full_FPP'),('cadd_am_vs_full','scores','fusion')]:
 for t in [i/10 for i in range(1,10)]:
  y=frozen.label.to_numpy();old=(frames[a].mean_probability.to_numpy()>=t).astype(int);new=(frames[b].mean_probability.to_numpy()>=t).astype(int)
  tab=pd.DataFrame({'label':y,'baseline':old,'augmented':new}).value_counts().to_dict()
  get=lambda y,b,n:int(tab.get((y,b,n),0))
  z={'both_correct':get(0,0,0)+get(1,1,1),'both_wrong':get(0,1,1)+get(1,0,0),'improved':get(0,1,0)+get(1,0,1),'degraded':get(0,0,1)+get(1,1,0),'FN_rescued':get(1,0,1),'FP_suppressed':get(0,1,0),'TN_to_FP':get(0,0,1),'TP_to_FN':get(1,1,0),'persistent_FN':get(1,0,0),'persistent_FP':get(0,1,1)}
  assert sum(z[x] for x in ['both_correct','both_wrong','improved','degraded'])==18220
  if t==.5:counts[pair]=z
  thresholds.append({'model_pair':pair,'threshold':t,**z,'net':z['improved']-z['degraded']})
out={'status':'independent_counts_metrics_passed','rows':len(frozen),'unique_proteins':int(frozen.STRING_protein_id.nunique()),'positive':int(frozen.label.sum()),'alignment':'ID sorted independently, label/protein matched frozen manifest','means':'float64;exact canonical equality','metrics_method':'rank-U AUC; unique-probability grouped noninterpolated AP','metrics':results,'counts':counts,'all_18_threshold_partitions_verified':True,'source_hashes':hashes}
(W/'independent_counts_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');pd.DataFrame(thresholds).to_csv(W/'independent_threshold_counts.csv',index=False)
print(json.dumps({k:v for k,v in out.items() if k!='source_hashes'},ensure_ascii=False,indent=2))
