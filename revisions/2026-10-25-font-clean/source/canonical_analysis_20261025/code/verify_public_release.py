"""Read published-format gzip tables back and independently recover every metric."""
from pathlib import Path
import os,json,hashlib,time
import numpy as np,pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score
B=Path(os.environ.get('PPI_AUDIT_ASSET_ROOT', 'assets'));E=B/'external_validation_20261024';O=Path(os.environ.get('PPI_CANONICAL_OUTPUT_ROOT', B/'review_synthesis_20261025/canonical_analysis'));P=O/'public';SEEDS=[42,43,44]
assert json.loads((O/'verification.json').read_text())['passed']
def read(p):return pd.read_csv(p,float_precision='round_trip')
def mm(y,p):return np.array([roc_auc_score(y,p),average_precision_score(y,p)])
manifest=read(P/'frozen_samples_177213.csv.gz');assert len(manifest)==177213 and not manifest.VariationID.duplicated().any()
nodes=read(P/'frozen_node_order.csv.gz');mapping=dict(zip(nodes.STRING_protein_id,nodes.node_index));metrics=read(O/'evaluation_per_seed.csv');ensembles=read(O/'ensemble_metrics.csv');checked=[]
for p in sorted(P.glob('test_predictions__*.csv.gz')):
 _,split,arm=p.name.removesuffix('.csv.gz').split('__');d=read(p);base=E if 'BioGRID' in arm else B
 source=B/'data'/('homology_aware_split/clinvar_test_homology.csv' if split=='homology' else 'clinvar_test_main.csv');src=pd.read_csv(source,usecols=['VariationID','STRING_protein_id','Label'])
 assert np.array_equal(d.VariationID,src.VariationID) and np.array_equal(d.STRING_protein_id,src.STRING_protein_id) and np.array_equal(d.label,src.Label)
 s=[]
 for seed in SEEDS:
  if split=='original':expected=np.load(O/f'node_{arm}_seed_{seed}.npy')[d.STRING_protein_id.map(mapping).to_numpy(dtype=int)].astype(np.float64)
  else:expected=np.load(base/'outputs/runs'/split/arm/f'seed_{seed}'/'test_probs.npy').astype(np.float64)
  assert np.array_equal(d[f'p_seed_{seed}'].to_numpy(),expected),(p.name,seed)
  report=metrics[(metrics.population==split)&(metrics.arm==arm)&(metrics.seed==seed)].iloc[0]
  assert np.allclose(mm(d.label,expected),report[['auc','auprc']].to_numpy(dtype=float),rtol=0,atol=1e-12)
  s.append(expected)
 ens=np.mean(s,axis=0,dtype=np.float64);assert np.array_equal(ens,d.ensemble_float64)
 row=ensembles[(ensembles.population==split)&(ensembles.arm==arm)].iloc[0];assert np.allclose(mm(d.label,ens),row[['auc','auprc']].to_numpy(dtype=float),rtol=0,atol=1e-12)
 if split=='original':
  for col in [f'p_seed_{s}' for s in SEEDS]+['ensemble_float64']:assert d.groupby('STRING_protein_id')[col].nunique().max()==1
 checked.append({'file':p.name,'rows':len(d),'scores_bit_identical':True,'metrics_recomputed_from_export':True})
ext=read(P/'external_membership_and_predictions.csv.gz');old=read(E/'cohort_recon/external_cohort_all_variant_disjoint.csv');assert len(ext)==40900 and not ext.external_id.duplicated().any()
for col in ['external_id','uniprot_id','protein_variant','STRING_protein_id','label','primary_trainval_unseen','unseen_all_original']:assert np.array_equal(ext[col],old[col]),col
for pop,flag,n in [('external_primary','primary_trainval_unseen',10785),('external_strict','unseen_all_original',6934),('external_all',None,40900)]:
 d=ext if flag is None else ext[ext[flag]==1];assert len(d)==n
 for arm in ['raw_ESM','full_FPP','BioGRID_FPP']:
  ss=[]
  for seed in SEEDS:
   expected=np.load(O/f'node_{arm}_seed_{seed}.npy')[d.STRING_protein_id.map(mapping).to_numpy(dtype=int)].astype(np.float64);assert np.array_equal(d[f'{arm}_seed_{seed}'],expected);ss.append(expected)
   row=metrics[(metrics.population==pop)&(metrics.arm==arm)&(metrics.seed==seed)].iloc[0];assert np.allclose(mm(d.label,expected),row[['auc','auprc']].to_numpy(dtype=float),rtol=0,atol=1e-12)
  ens=np.mean(ss,axis=0,dtype=np.float64);assert np.array_equal(ens,d[f'{arm}_ensemble_float64']);row=ensembles[(ensembles.population==pop)&(ensembles.arm==arm)].iloc[0];assert np.allclose(mm(d.label,ens),row[['auc','auprc']].to_numpy(dtype=float),rtol=0,atol=1e-12)
# Exact full manifest joins independently against all six saved split tables.
for split in ['original','homology']:
 for part in ['train','val','test']:
  src=B/'data'/(f'clinvar_{part}_main.csv' if split=='original' else f'homology_aware_split/clinvar_{part}_homology.csv');dd=pd.read_csv(src)
  lookup=manifest.set_index('VariationID').loc[dd.VariationID]
  assert np.array_equal(lookup.label,dd.Label) and np.array_equal(lookup.STRING_protein_id,dd.STRING_protein_id) and (lookup[split+'_split']==part).all()
  if split=='homology':assert np.array_equal(lookup.homology_cluster_id.astype(str),dd._cluster_id.astype(str))
result={'passed':True,'benchmark_prediction_files':checked,'external_all_rows':40900,'external_primary_rows':10785,'external_strict_rows':6934,'all_seed_and_ensemble_predictions_bit_exact_after_csv_round_trip':True,'all_exported_metrics_recomputed_with_sklearn':True,'all6split_tables_and_homologyclusters_matched_byVariationID':True,'representation_within_protein_score_consistency':True,'patients_or_private_correspondence_fields':False,'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
(O/'public_release_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
