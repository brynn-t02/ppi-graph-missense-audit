"""Canonical frozen-checkpoint postprocessing; no training, no old-file writes."""
from pathlib import Path
import os,json,time,hashlib,gzip,io,importlib.util,sys,collections
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score
B=Path(os.environ.get('PPI_AUDIT_ASSET_ROOT', 'assets'));E=B/'external_validation_20261024';R=B/'review_synthesis_20261025';O=Path(os.environ.get('PPI_CANONICAL_OUTPUT_ROOT', R/'canonical_analysis'));P=O/'public'
O.mkdir(parents=True,exist_ok=True);P.mkdir(exist_ok=True);torch.set_num_threads(4)
SEEDS=[42,43,44];PARTS=['train','val','test'];inputs={};start=time.time()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def record(p):inputs[str(p)]=sha(p);return p
def dump(name,x):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def status(phase,**kw):dump('progress.json',{'phase':phase,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw});print(phase,kw,flush=True)
def metric(y,p):return {'auc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p))}
def savecsv(df,name):df.to_csv(O/name,index=False,float_format='%.17g')
def publiccsv(df,name):
 p=P/(name+'.csv.gz')
 with p.open('wb') as f:
  with gzip.GzipFile(filename='',mode='wb',fileobj=f,mtime=0) as g:
   with io.TextIOWrapper(g,encoding='utf-8',newline='') as t:df.to_csv(t,index=False,float_format='%.17g')
 assert p.stat().st_size<5_000_000,(str(p),p.stat().st_size)
 return p
spec=importlib.util.spec_from_file_location('weighted',record(Path(__file__).with_name('analyze_external_validation.py')));w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
assert w.validate_weighted_metrics()['passed']
class MLP(nn.Module):
 def __init__(self):
  super().__init__();self.net=nn.Sequential(nn.Linear(1280,512),nn.ReLU(),nn.Dropout(.3),nn.Linear(512,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,1))
 def forward(self,x):return self.net(x).squeeze(-1)
status('reading_frozen_inputs')
a=torch.load(record(B/'data/model_inputs_main.pt'),map_location='cpu',weights_only=False);mapping={str(k):int(v) for k,v in a['protein_to_idx'].items()};nodes=[None]*len(mapping)
for k,v in mapping.items():nodes[v]=k
assert len(nodes)==19622 and len(set(nodes))==19622
rows={};allparts={}
for split in ['original','homology']:
 allparts[split]={}
 for part in PARTS:
  p=B/'data'/(f'clinvar_{part}_main.csv' if split=='original' else f'homology_aware_split/clinvar_{part}_homology.csv')
  df=pd.read_csv(record(p));assert not df.VariationID.duplicated().any();df['actual_node_index']=df.STRING_protein_id.map(mapping);assert df.actual_node_index.notna().all()
  if split=='original':
   assert np.array_equal(df.actual_node_index.to_numpy(),a[part+'_protein_idx'].numpy());assert np.array_equal(df.Label.to_numpy(),a[part+'_labels'].numpy())
  allparts[split][part]=df
 rows[split]=allparts[split]['test']
 rows['original_fusion']=rows['original']
# Full sample membership and labels: no clinical narratives or individual data.
orig=pd.concat([v.assign(original_split=k) for k,v in allparts['original'].items()],ignore_index=True)
hom=pd.concat([v.assign(homology_split=k) for k,v in allparts['homology'].items()],ignore_index=True)
assert len(orig)==177213 and len(hom)==177213
manifest=orig[['VariationID','Label','STRING_protein_id','original_split']].merge(hom[['VariationID','Label','STRING_protein_id','homology_split','_cluster_id']],on=['VariationID','Label','STRING_protein_id'],validate='one_to_one',how='inner')
assert len(manifest)==177213
manifest=manifest.rename(columns={'Label':'label','_cluster_id':'homology_cluster_id'}).sort_values('VariationID')
for x,y in [('train','val'),('train','test'),('val','test')]:assert not set(manifest[manifest.homology_split==x].homology_cluster_id)&set(manifest[manifest.homology_split==y].homology_cluster_id)
publiccsv(manifest,'frozen_samples_177213');publiccsv(pd.DataFrame({'node_index':range(len(nodes)),'STRING_protein_id':nodes}),'frozen_node_order')
oldruns=pd.read_csv(record(B/'outputs/per_run_metrics.csv'));newruns=pd.read_csv(record(E/'outputs/per_run_metrics.csv'));runs=pd.concat([oldruns,newruns],ignore_index=True);assert len(runs)==57
pred={s:{} for s in ['original','original_fusion','homology']};nodelookup={};audits=[];updated=[];checkpoints={}
for arm,grp in runs[runs.split=='original'].groupby('arm',sort=True):
 if arm=='raw_ESM':z=a['x'].float()
 else:
  ep=(E/'outputs/derived_embeddings/BioGRID_FPP.pt' if arm=='BioGRID_FPP' else B/'data/gcn_embeddings/gcn_node_embeddings.pt' if arm=='full_FPP' else B/'outputs/derived_embeddings'/f'{arm}.pt')
  q=torch.load(record(ep),map_location='cpu',weights_only=False);z=(q['embeddings'] if isinstance(q,dict) else q).float()
 assert z.shape==(19622,1280) and torch.isfinite(z).all()
 arrays=[]
 for _,rr in grp.sort_values('seed').iterrows():
  seed=int(rr.seed);base=E if arm=='BioGRID_FPP' else B;q=base/'outputs/runs/original'/arm/f'seed_{seed}';cp=record(q/'model.pt');h=sha(cp);checkpoints[str(cp)]=h
  model=MLP();model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True));model.eval()
  with torch.inference_mode():pp=torch.cat([torch.sigmoid(model(z[i:i+1024])) for i in range(0,len(z),1024)]).numpy()
  assert pp.dtype==np.float32 and np.isfinite(pp).all() and np.min(pp)>=0 and np.max(pp)<=1
  np.save(O/f'node_{arm}_seed_{seed}.npy',pp);nodelookup[(arm,seed)]=pp.astype(np.float64)
  audit={'arm':arm,'seed':seed,'checkpoint_sha256':h,'parts':{}}
  for part in ['val','test']:
   df=allparts['original'][part];canon=pp[df.actual_node_index.to_numpy()];saved=np.load(record(q/f'{part}_probs.npy'),allow_pickle=False)
   err=float(np.max(np.abs(canon-saved)));assert err<=2e-6,(arm,seed,part,err)
   spread=pd.DataFrame({'pid':df.STRING_protein_id,'p':saved}).groupby('pid').p.agg(['min','max']);spread=spread['max']-spread['min']
   audit['parts'][part]={'max_abs_canonical_vs_saved':err,'old_same_protein_nonzero_spread_count':int((spread>0).sum()),'old_max_within_protein_spread':float(spread.max()),'canonical_max_within_protein_spread':0.0,'old_metrics':metric(df.Label,saved),'canonical_metrics':metric(df.Label,canon)}
  if arm in ['raw_ESM','full_FPP','BioGRID_FPP']:
   oldnp=np.load(record(E/'outputs/node_predictions'/arm/f'seed_{seed}.npy'));audit['max_abs_vs_prior_external_node_scores']=float(np.max(np.abs(oldnp-pp)));assert audit['max_abs_vs_prior_external_node_scores']<=2e-6
  assert sha(cp)==h;audits.append(audit);arrays.append(pp[rows['original'].actual_node_index.to_numpy()].astype(np.float64))
  r=rr.to_dict();m=metric(rows['original'].Label,arrays[-1]);r.update(test_auc=m['auc'],test_auprc=m['auprc'],reporting='canonical_node_lookup');updated.append(r)
 pred['original'][arm]=np.stack(arrays);status('canonical_node_inference',arm=arm,completed_checkpoints=len(audits))
# Fusion/score probabilities are genuinely variant-specific; preserve their values.
for split in ['original_fusion','homology']:
 for arm,grp in runs[runs.split==split].groupby('arm',sort=True):
  arr=[]
  for _,rr in grp.sort_values('seed').iterrows():
   q=(E if 'BioGRID' in arm else B)/'outputs/runs'/split/arm/f'seed_{int(rr.seed)}';p=np.load(record(q/'test_probs.npy'),allow_pickle=False).astype(np.float64);arr.append(p)
   m=metric(rows[split].Label,p);assert np.allclose([m['auc'],m['auprc']],[rr.test_auc,rr.test_auprc],atol=1e-12,rtol=0)
   r=rr.to_dict();r['reporting']='unchanged_frozen_variant_prediction';updated.append(r);checkpoints[str(q/'model.pt')]=sha(record(q/'model.pt'))
  pred[split][arm]=np.stack(arr)
# Collect external membership and probabilities, now using the same canonical nodes.
external=pd.read_csv(record(E/'cohort_recon/external_cohort_all_variant_disjoint.csv'));external['actual_node_index']=external.STRING_protein_id.map(mapping);assert len(external)==40900 and external.actual_node_index.notna().all()
for name,flag,n in [('external_primary','primary_trainval_unseen',10785),('external_strict','unseen_all_original',6934),('external_all',None,40900)]:
 d=external.copy() if flag is None else external[external[flag]==1].copy();assert len(d)==n;rows[name]=d.rename(columns={'label':'Label'});idx=d.actual_node_index.to_numpy(dtype=int);pred[name]={arm:np.stack([nodelookup[(arm,s)][idx] for s in SEEDS]) for arm in ['raw_ESM','full_FPP','BioGRID_FPP']}
# All CSV scores use17significant digits, preserving exact probability values.
for split in ['original','original_fusion','homology']:
 iddf=rows[split][['VariationID','Label','STRING_protein_id']].rename(columns={'Label':'label'})
 for arm,pp in pred[split].items():
  d=iddf.copy()
  for j,s in enumerate(SEEDS):d[f'p_seed_{s}']=pp[j]
  d['ensemble_float64']=pp.mean(axis=0,dtype=np.float64);publiccsv(d,f'test_predictions__{split}__{arm}')
extpub=external[['external_id','uniprot_id','protein_variant','STRING_protein_id','label','primary_trainval_unseen','unseen_all_original']].copy();idx=external.actual_node_index.to_numpy(dtype=int)
for arm in ['raw_ESM','full_FPP','BioGRID_FPP']:
 for s in SEEDS:extpub[f'{arm}_seed_{s}']=nodelookup[(arm,s)][idx]
 extpub[f'{arm}_ensemble_float64']=np.mean([nodelookup[(arm,s)][idx] for s in SEEDS],axis=0,dtype=np.float64)
publiccsv(extpub,'external_membership_and_predictions')
current=pd.DataFrame(updated);assert len(current)==57
savecsv(current,'per_run_metrics.csv');summary=current.groupby(['split','arm']).agg(n=('seed','count'),auc_mean=('test_auc','mean'),auc_sd=('test_auc','std'),auprc_mean=('test_auprc','mean'),auprc_sd=('test_auprc','std')).reset_index();savecsv(summary,'summary_metrics.csv')
dump('canonical_inference_audit.json',{'inference':'CPUfloat32/4threads/1024nodes/fixedactualnodeorder/evalmode','checkpoint_count':len(audits),'max_abs_tolerance':2e-6,'runs':audits,'checkpoint_hashes':checkpoints,'new_fits':0})
# Graph permutation aggregate uses all nine canonical predictions at equal weight.
pred['original']['permuted_FPP_graphmean']=np.stack([pred['original'][f'permuted_FPP_g{i}'] for i in range(3)]).mean(axis=0,dtype=np.float64)
pred['original_fusion']['scores_plus_permuted_FPP_graphmean']=np.stack([pred['original_fusion'][f'scores_plus_permuted_FPP_g{i}'] for i in range(3)]).mean(axis=0,dtype=np.float64)
pairs={
'original':[('full_FPP','raw_ESM'),('full_FPP','edge_ablated_FPP'),('full_FPP','permuted_FPP_graphmean'),('BioGRID_FPP','raw_ESM'),('BioGRID_FPP','full_FPP')],
'original_fusion':[('scores_plus_full_FPP',a) for a in ['scores_capacity_matched','scores_plus_raw_ESM','scores_plus_edge_ablated_FPP','scores_plus_permuted_FPP_graphmean']]+[('scores_plus_BioGRID_FPP',a) for a in ['scores_capacity_matched','scores_plus_raw_ESM','scores_plus_full_FPP']],
'homology':[('scores_plus_full_FPP','scores_capacity_matched'),('scores_plus_raw_ESM','scores_capacity_matched'),('scores_plus_full_FPP','scores_plus_raw_ESM')]+[('scores_plus_BioGRID_FPP',a) for a in ['scores_capacity_matched','scores_plus_raw_ESM','scores_plus_full_FPP']]}
for k in ['external_primary','external_strict','external_all']:pairs[k]=[('full_FPP','raw_ESM'),('BioGRID_FPP','raw_ESM'),('BioGRID_FPP','full_FPP')]
bootstrap={'bootstrap_repeats':2000,'seed':20261024,'resampling_unit':'STRINGprotein(allvariants,multiplicity)','rng_policy':'Restartseedperpopulation;validationrngseparate;identicaldrawsforallcomparisons','precision':'allseedprobabilitiescastfloat64beforeensemble','metric':'ROC AUC and noninterpolatedsklearnaverageprecision','scope':'Conditionalonfrozenfits/networks;exploratorywithoutmultiplicityadjustment','comparisons':{}}
evals=[];ensrows=[];seeddeltas=[];ens={}
for split,arms in pred.items():
 y=rows[split].Label.to_numpy();ids=rows[split].STRING_protein_id.to_numpy();ens[split]={arm:pp.mean(axis=0,dtype=np.float64) for arm,pp in arms.items()}
 for arm,pp in arms.items():
  for j,s in enumerate(SEEDS):evals.append({'population':split,'arm':arm,'seed':s,'variants':len(y),'proteins':len(set(ids)),'positive':int(y.sum()),**metric(y,pp[j])})
  ensrows.append({'population':split,'arm':arm,'variants':len(y),'proteins':len(set(ids)),'positive':int(y.sum()),'prevalence':float(y.mean()),**metric(y,ens[split][arm])})
 bootstrap['comparisons'][split]=w.paired_bootstrap(y,ids,ens[split],pairs[split],repeats=2000,seed=20261024)
 for arm,ctrl in pairs[split]:
  for j,s in enumerate(SEEDS):
   mm,mc=metric(y,arms[arm][j]),metric(y,arms[ctrl][j]);seeddeltas.append({'split':split,'comparison':arm+'_vs_'+ctrl,'seed':s,'delta_auc':mm['auc']-mc['auc'],'delta_auprc':mm['auprc']-mc['auprc']})
 status('bootstrap_complete',population=split,comparisons=len(pairs[split]))
# Variant-resampling contrast is explicitly separate and ignores within-protein dependence.
y=rows['original'].Label.to_numpy();pp=ens['original'];f=w.WeightedScores(y,pp['full_FPP']);g=w.WeightedScores(y,pp['raw_ESM']);rng=np.random.default_rng(20261024);vv=[]
for i in range(2000):
 weights=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y));vv.append(f(weights)-g(weights))
q=np.quantile(vv,[.025,.975],axis=0);bootstrap['variant_bootstrap_original_full_vs_raw']={'auc_ci95':q[:,0].tolist(),'auprc_ci95':q[:,1].tolist(),'repeats':2000,'seed':20261024,'scope':'Variantresamplingignoreswithinproteindependence;descriptivecontrastonly'}
savecsv(pd.DataFrame(evals),'evaluation_per_seed.csv');savecsv(pd.DataFrame(ensrows),'ensemble_metrics.csv');savecsv(pd.DataFrame(seeddeltas),'paired_seed_deltas.csv');dump('cluster_bootstrap.json',bootstrap)
f=pd.DataFrame(evals);es=f[f.population.str.startswith('external_')].groupby(['population','arm']).agg(n_seeds=('seed','count'),auc_mean=('auc','mean'),auc_sd=('auc','std'),auprc_mean=('auprc','mean'),auprc_sd=('auprc','std')).reset_index().rename(columns={'population':'cohort'});es['cohort']=es.cohort.map({'external_primary':'primary','external_strict':'strict_unseen_all_original','external_all':'all_variant_disjoint'});savecsv(es,'external_summary_metrics.csv')
# Complete score subset: unchanged fusion predictions, current uniform metric code.
c=pd.read_csv(record(B/'data/clinvar_with_cadd.csv'),usecols=['VariationID','CADD_PHRED']);am=pd.read_csv(record(B/'data/clinvar_with_alphamissense.csv'),usecols=['VariationID','am_pathogenicity']);cc=rows['original'].merge(c,on='VariationID',validate='one_to_one',sort=False).merge(am,on='VariationID',validate='one_to_one',sort=False);assert np.array_equal(cc.VariationID,rows['original'].VariationID);mask=cc.CADD_PHRED.notna()&cc.am_pathogenicity.notna();assert int(mask.sum())==17581
ccout=[]
for arm,pp in pred['original_fusion'].items():
 if 'graphmean' in arm:continue
 for j,s in enumerate(SEEDS):
  old=metric(cc.Label,pp[j]);cm=metric(cc.Label[mask],pp[j][mask]);ccout.append({'arm':arm,'seed':s,'complete_rows':int(mask.sum()),'all_auc':old['auc'],'all_auprc':old['auprc'],'complete_auc':cm['auc'],'complete_auprc':cm['auprc']})
savecsv(pd.DataFrame(ccout),'current_complete_case_per_run.csv')
# Independently recalculate five complete2000-resampleintervals via sklearn.
independent=[]
for split,arm,ctrl in [('original','full_FPP','raw_ESM'),('original_fusion','scores_plus_full_FPP','scores_plus_raw_ESM'),('homology','scores_plus_full_FPP','scores_capacity_matched'),('external_primary','full_FPP','raw_ESM'),('external_strict','full_FPP','raw_ESM')]:
 y=rows[split].Label.to_numpy();ids=rows[split].STRING_protein_id.to_numpy();uniq,inv=np.unique(ids,return_inverse=True);rng=np.random.default_rng(20261024);p,q=ens[split][arm],ens[split][ctrl];values=[]
 for i in range(2000):
  weights=np.bincount(rng.integers(0,len(uniq),len(uniq)),minlength=len(uniq))[inv]
  values.append([roc_auc_score(y,p,sample_weight=weights)-roc_auc_score(y,q,sample_weight=weights),average_precision_score(y,p,sample_weight=weights)-average_precision_score(y,q,sample_weight=weights)])
 ci=np.quantile(values,[.025,.975],axis=0);prior=bootstrap['comparisons'][split][arm+'_vs_'+ctrl];err=float(np.max(np.abs(ci-np.array([prior['auc_ci95'],prior['auprc_ci95']]).T)));assert err<1e-10,(split,err);independent.append({'population':split,'comparison':arm+'_vs_'+ctrl,'repeats_independently_recomputed':2000,'max_abs_ci_error':err});status('independent_ci_verified',population=split)
# Confirm source/checkpoint files remain unchanged and publiccompressionroundtrips.
for path,h in inputs.items():assert sha(Path(path))==h,path
pub=[]
for p in sorted(P.glob('*.csv.gz')):
 df=pd.read_csv(p,float_precision='round_trip');assert len(df)>0
 pub.append({'file':p.name,'bytes':p.stat().st_size,'rows':len(df),'columns':list(df.columns),'sha256':sha(p)})
dump('public_manifest.json',{'files':pub,'row_precision':'17significantdigitsforbinary64;pdread_csvfloat_precision=round_trip','no_patient_identifiers':True,'benchmark_cohort_rows':177213,'external_rows':40900})
dump('source_manifest.json',{'inputs':[{'path':k,'sha256':v} for k,v in sorted(inputs.items())],'selected_checkpoints':checkpoints,'source_code_sha256':sha(Path(__file__)),'protocol_sha256':sha(Path(__file__).with_name('protocol.json'))})
dump('verification.json',{'passed':True,'new_fits':0,'frozen_checkpoints':len(checkpoints),'representation_checkpoints_canonically_inferred':21,'seed_metric_rows':len(evals),'ensemble_rows':len(ensrows),'protein_bootstrap_comparisons':sum(len(v) for v in pairs.values()),'independent_sklearn_complete_ci_checks':independent,'weighted_kernel_synthetic_tests':w.validate_weighted_metrics(),'all_original_inputs_unchanged':True,'full_manifest_labels_splits_clusters_verified':True,'public_files':len(pub),'seconds':time.time()-start})
# Bounded small archive of summaries and compact public manifests (no arrays/tensors).
import tarfile
small=[p for p in O.glob('*') if p.is_file() and p.suffix in ['.json','.csv']]+list(P.glob('*.csv.gz'))
with tarfile.open(O/'compact_results.tar.gz','w:gz') as t:
 for p in small:
  assert p.stat().st_size<5_000_000;t.add(p,arcname=str(p.relative_to(O)))
dump('transfer_manifest.json',{'files':[{'relative_path':str(p.relative_to(O)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in small]})
status('complete',new_fits=0,checkpoint_count=len(checkpoints),seconds=time.time()-start)
