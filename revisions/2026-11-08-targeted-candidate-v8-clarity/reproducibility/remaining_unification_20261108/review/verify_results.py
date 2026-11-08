"""Independent validation of compact completed results; does not fit or select models."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np,pandas as pd
from independent_metrics import metrics
parser=argparse.ArgumentParser();parser.add_argument('results',type=Path);args=parser.parse_args();o=args.results
r={'status':'in_progress','result_directory':str(o.resolve())};errs=[]
def read(p):return pd.read_csv(p,float_precision='round_trip')
def close(a,b,label,tol=2e-12):
 e=float(np.max(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))));assert e<=tol,(label,e);errs.append(e)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
complete=json.loads((o/'analysis_complete.json').read_text());assert complete['passed'] and complete['classifier_fits']==39 and complete['canonical_arms']==13
training=read(o/'training_per_run_metrics.csv');can=read(o/'canonical_per_run_metrics.csv');summ=read(o/'canonical_summary_metrics.csv');ens_tab=read(o/'ensemble_metrics.csv')
assert len(training)==len(can)==39 and len(summ)==len(ens_tab)==13
assert not can.duplicated(['split','arm','seed']).any()
for g in (training,can):assert all(sorted(z.seed)==[42,43,44] for _,z in g.groupby(['split','arm']))
expected={('homology','raw_ESM'),('homology','full_FPP'),('original','full_FPP'),('original','edge_ablated_FPP')}
strategies=['random_pair_removal','high_confidence_first','low_confidence_first']
expected|={('original',f'{s}_fraction_{f:.2f}') for s in strategies for f in [.25,.5,.75]}
assert set(map(tuple,can[['split','arm']].drop_duplicates().values))==expected
canonicalbase=Path('outputs/BIBM_workshop_font_clean_20261025/source/canonical_analysis_20261025/public')
frozen=read(canonicalbase/'frozen_samples_177213.csv.gz').set_index('VariationID')
rows={s:read(o/f'{s}_test_rows.csv') for s in ['original','homology']};seedmetrics={};ensembles={};allpred={};checks=[]
for split,d in rows.items():
 assert d.VariationID.is_unique;ref=frozen.loc[d.VariationID]
 assert np.array_equal(d.Label.values,ref.label.values) and np.array_equal(d.STRING_protein_id.values,ref.STRING_protein_id.values)
 assert ref[split+'_split'].eq('test').all() and len(d)==int(frozen[split+'_split'].eq('test').sum())
 if split=='homology':assert np.array_equal(d._cluster_id,ref.homology_cluster_id)
assert len(rows['original'])==18220 and rows['original'].STRING_protein_id.nunique()==1413 and rows['original'].Label.sum()==5822
for split,arm in sorted(expected):
 d=read(o/'public'/f'test_predictions__{split}__{arm}.csv.gz');base=rows[split]
 assert np.array_equal(d.VariationID,base.VariationID) and np.array_equal(d.label,base.Label) and np.array_equal(d.STRING_protein_id,base.STRING_protein_id)
 ps=d[['p_seed_42','p_seed_43','p_seed_44']].values.astype(np.float64);assert np.isfinite(ps).all() and ps.min()>=0 and ps.max()<=1
 assert d.groupby('STRING_protein_id')[['p_seed_42','p_seed_43','p_seed_44']].nunique().eq(1).all().all()
 ens=ps.mean(axis=1,dtype=np.float64);assert np.array_equal(ens,d.ensemble_float64.values)
 allpred[(split,arm)]=d;ensembles[(split,arm)]=ens
 vals=[]
 for j,seed in enumerate([42,43,44]):
  got=metrics(d.label,ps[:,j]);row=can[(can.split==split)&(can.arm==arm)&(can.seed==seed)].iloc[0]
  close(got,[row.test_auc,row.test_auprc],(split,arm,seed));seedmetrics[(split,arm,seed)]=got;vals.append(got)
 vals=np.asarray(vals);row=summ[(summ.split==split)&(summ.arm==arm)].iloc[0]
 close(vals.mean(0),[row.auc_mean,row.auprc_mean],(split,arm,'mean'));close(vals.std(0,ddof=1),[row.auc_sd,row.auprc_sd],(split,arm,'sd'))
 row=ens_tab[(ens_tab.split==split)&(ens_tab.arm==arm)].iloc[0];close(metrics(d.label,ens),[row.auc,row.auprc],(split,arm,'ensemble'))
 checks.append({'split':split,'arm':arm,'seeds':3,'variants':len(d),'proteins':d.STRING_protein_id.nunique(),'auc_mean':float(vals[:,0].mean()),'auprc_mean':float(vals[:,1].mean())})
curve=read(o/'figure1E_summary.csv');crun=read(o/'figure1E_per_seed.csv');assert len(curve)==15 and len(crun)==45
for _,q in curve.iterrows():
 arm='full_FPP' if q.fraction_removed==0 else 'edge_ablated_FPP' if q.fraction_removed==1 else f'{q.strategy}_fraction_{q.fraction_removed:.2f}'
 assert q.arm==arm and q.n==3
 v=np.array([seedmetrics[('original',arm,s)] for s in [42,43,44]])
 close([q.auc_mean,q.auprc_mean],v.mean(0),('curve',arm,'mean'));close([q.auc_sd,q.auprc_sd],v.std(0,ddof=1),('curve',arm,'sd'))
 for _,q2 in crun[(crun.strategy==q.strategy)&(crun.fraction_removed==q.fraction_removed)].iterrows():close([q2.test_auc,q2.test_auprc],seedmetrics[('original',arm,q2.seed)],('curve',arm,q2.seed))
# All selection and early-stopping histories, if passed as requested.
history_checks=[]
for _,q in training.iterrows():
 p=o/'runs'/q.split/q.arm/f'seed_{int(q.seed)}';hist=json.loads((p/'history.json').read_text());logged=json.loads((p/'metrics.json').read_text())
 assert len(hist)==q.epochs_run and len(hist)<=50 and [h['epoch'] for h in hist]==list(range(1,len(hist)+1))
 best=-np.inf;bestep=None;bad=0
 for i,h in enumerate(hist):
  assert np.isfinite([h['loss'],h['val_auc'],h['val_auprc']]).all()
  if h['val_auc']>best:best=h['val_auc'];bestep=h['epoch'];bad=0
  else:bad+=1
  if i<len(hist)-1:assert bad<5
 assert q.best_epoch==bestep and (bad==5 or len(hist)==50)
 close(best,q.val_auc,('best validation',q.split,q.arm,q.seed))
 for k in ['best_epoch','epochs_run','seed','split','arm']:assert logged[k]==q[k]
 history_checks.append({'split':q.split,'arm':q.arm,'seed':int(q.seed),'best_epoch':bestep,'epochs_run':len(hist)})
# Independently replay paired bootstrap via weighted concordance/rank groups.
d=rows['homology'];y=d.Label.values;uniq,inv=np.unique(d.STRING_protein_id.values,return_inverse=True)
p0=ensembles[('homology','raw_ESM')];p1=ensembles[('homology','full_FPP')]
bs=read(o/'homology_paired_protein_bootstrap_replicates.csv.gz');assert len(bs)==2000 and bs.replicate.tolist()==list(range(2000))
rng=np.random.default_rng(20261024);digest=hashlib.sha256();diffs=[];expanded=[]
for i in range(2000):
 counts=np.bincount(rng.integers(0,len(uniq),len(uniq)),minlength=len(uniq));digest.update(counts.astype('<i8').tobytes());w=counts[inv]
 v=metrics(y,p1,w)-metrics(y,p0,w);diffs.append(v)
 assert np.array_equal(np.isfinite(v),np.isfinite(bs.iloc[i][['delta_auc','delta_auprc']].values.astype(float)))
 if np.isfinite(v).all():close(v,bs.iloc[i][['delta_auc','delta_auprc']].values,('bootstrap',i))
 if i in [0,1,2,499,999,1499,1999]:
  ix=np.repeat(np.arange(len(y)),w);direct=metrics(y[ix],p1[ix])-metrics(y[ix],p0[ix]);close(v,direct,('explicit rows',i));expanded.append(i)
vv=np.asarray(diffs);valid=np.isfinite(vv).all(1);ci=np.quantile(vv[valid],[.025,.975],axis=0);point=metrics(y,p1)-metrics(y,p0)
bj=json.loads((o/'figure2D_protein_bootstrap.json').read_text());assert bj['seed']==20261024 and bj['resamples']==2000 and bj['valid_resamples']==valid.sum() and bj['weights_sha256']==digest.hexdigest()
close(point,[bj['delta_auc'],bj['delta_auprc']],'bootstrappoint');close(ci[:,0],bj['auc_ci95'],'AUCCI');close(ci[:,1],bj['auprc_ci95'],'APCI')
# Separate original full-minus-ablated comparison; same independent metric implementation.
od=rows['original'];oy=od.Label.values;ou,oi=np.unique(od.STRING_protein_id.values,return_inverse=True)
op0=ensembles[('original','edge_ablated_FPP')];op1=ensembles[('original','full_FPP')]
obs=read(o/'original_ablation_paired_protein_bootstrap_replicates.csv.gz');assert len(obs)==2000
orng=np.random.default_rng(20261024);odigest=hashlib.sha256();odiff=[]
for j in range(2000):
 oc=np.bincount(orng.integers(0,len(ou),len(ou)),minlength=len(ou));odigest.update(oc.astype('<i8').tobytes());ow=oc[oi]
 ov=metrics(oy,op1,ow)-metrics(oy,op0,ow);odiff.append(ov)
 assert np.array_equal(np.isfinite(ov),np.isfinite(obs.iloc[j][['delta_auc','delta_auprc']].values.astype(float)))
 if np.isfinite(ov).all():close(ov,obs.iloc[j][['delta_auc','delta_auprc']].values,('originalbootstrap',j))
 if j in [0,1,2,499,999,1499,1999]:
  ix=np.repeat(np.arange(len(oy)),ow);close(ov,metrics(oy[ix],op1[ix])-metrics(oy[ix],op0[ix]),('originalexpanded',j))
odiff=np.asarray(odiff);ovalid=np.isfinite(odiff).all(1);oci=np.quantile(odiff[ovalid],[.025,.975],axis=0);opoint=metrics(oy,op1)-metrics(oy,op0)
obj=json.loads((o/'figure1D_protein_bootstrap.json').read_text());assert obj['seed']==20261024 and obj['resamples']==2000 and obj['valid_resamples']==ovalid.sum() and obj['weights_sha256']==odigest.hexdigest()
close(opoint,[obj['delta_auc'],obj['delta_auprc']],'originalpoint');close(oci[:,0],obj['auc_ci95'],'originalAUCCI');close(oci[:,1],obj['auprc_ci95'],'originalAPCI')

# Actual edge selection summaries must preserve original graph intervention definitions.
old=pd.read_csv('/Users/teyian/Documents/paper1/server_snapshot_full/root/data/processed/esm_gcn_partial_edge_ablation_exp44/edge_stats_all_partial_runs.csv')
graph=json.loads((o/'partial_graph_audits.json').read_text());assert len(graph)==9
for v in graph:
 q=old[(old.strategy==v['strategy'])&(old.fraction_removed==v['fraction_removed'])].iloc[0]
 for now,before in [('pairs_removed','num_selected_pairs_removed'),('edge_entries_removed','num_removed_edge_entries'),('edge_entries_remaining','num_remaining_edges'),('score_min','removed_pair_score_min'),('score_median','removed_pair_score_median'),('score_max','removed_pair_score_max')]:assert v[now]==q[before],(v['arm'],now,v[now],q[before])
 assert v['classifier_seeds']==[42,43,44]
 if v['strategy']=='random_pair_removal':assert v['graph_seed']==42+int(v['fraction_removed']*1000)
for name in ['input_audit.json','endpoint_input_gate_v2.json','full_operator_gate.json']:assert json.loads((o/name).read_text())['passed']
gate=json.loads((o/'endpoint_input_gate_v2.json').read_text());assert gate['first_runs_locked'] and gate['all_six_historical_metrics_reproduced'] is False
assert gate['first_tensor_sha256']=='3915bffdcf3940bd4ade55f8a9c28edd872f3b47701f0c0e0cfa636ec4c81242'
assert all(x['first_run_hashes_match'] for x in gate['runs']) and all(x['historical_exact_replication_passed'] for x in gate['runs'] if x['arm']=='full_FPP')
assert (o/'failure.json').exists() and (o/'protocol_freeze.json').exists() and (o/'protocol_freeze_v2.json').exists()
r.update(status='passed',classifier_runs=39,canonical_arms=13,curve_display_conditions=15,curve_unique_conditions=11,seed_metric_checks=39,ensemble_checks=13,all_prediction_ID_label_protein_and_split_checks=True,same_protein_scores_constant=True,float64_ensemble_exact=True,sample_SD_ddof=1,history_checks=history_checks,bootstrap_replays=4000,original_ablation_delta=opoint.tolist(),original_ablation_ci=oci.tolist(),original_bootstrap_weights_sha256=odigest.hexdigest(),explicit_expanded_row_checks=expanded,bootstrap_weights_sha256=digest.hexdigest(),bootstrap_valid=int(valid.sum()),homology_delta=point.tolist(),homology_ci=ci.tolist(),max_absolute_metric_difference=max(errs),arms=checks,actual_graph_summary_same_as_original=True)
(Path(__file__).parent/'results_verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:r[k] for k in ['status','classifier_runs','canonical_arms','bootstrap_replays','max_absolute_metric_difference','homology_delta','homology_ci']},indent=2))
