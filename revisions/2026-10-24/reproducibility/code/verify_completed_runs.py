from pathlib import Path
import json,hashlib,os,time
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score,average_precision_score
B=Path('/Data/u40902/TYX/ppi_audit_20261024');O=B/'outputs';runs=pd.read_csv(O/'per_run_metrics.csv');assert len(runs)==48
assert runs.groupby('split').size().to_dict()=={'homology':9,'original':18,'original_fusion':21}
checks=[]
for _,r in runs.iterrows():
 q=O/'runs'/r.split/r.arm/f'seed_{r.seed}';hist=json.loads((q/'history.json').read_text());assert 1<=len(hist)<=50;v=np.asarray([h['val_auc'] for h in hist]);assert np.argmax(v)+1==int(r.best_epoch);assert int(r.epochs_run)==len(hist);assert abs(v.max()-r.val_auc)<1e-12
 for part in ['val','test']:
  fn=f'homology_aware_split/clinvar_{part}_homology.csv' if r.split=='homology' else f'clinvar_{part}_main.csv'
  y=pd.read_csv(B/'data'/fn,usecols=['Label']).Label.to_numpy();p=np.load(q/f'{part}_probs.npy');assert len(p)==len(y) and np.isfinite(p).all() and (p>=0).all() and (p<=1).all();m=[roc_auc_score(y,p),average_precision_score(y,p)];assert np.allclose(m,[r[part+'_auc'],r[part+'_auprc']],atol=1e-12)
 state=torch.load(q/'model.pt',map_location='cpu',weights_only=True);assert tuple(state['net.0.weight'].shape)==(512,int(r.dim));assert tuple(state['net.3.weight'].shape)==(128,512);assert tuple(state['net.6.weight'].shape)==(1,128);assert all(torch.isfinite(z).all() for z in state.values())
 checks.append({'split':r.split,'arm':r.arm,'seed':int(r.seed),'selected_epoch_matches_history_maximum':True,'val_and_test_metrics_reproduced':True,'checkpoint_finite_expected_dimensions':True})
code_hashes={}
for p in sorted((B/'code').rglob('*')):
 if p.is_file():code_hashes[str(p.relative_to(B/'code'))]=hashlib.sha256(p.read_bytes()).hexdigest()
r={'passed':True,'runs_verified':len(checks),'checks':checks,'code_sha256':code_hashes,'checked_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())};(O/'completed_run_verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'passed':True,'runs_verified':48},indent=2))
