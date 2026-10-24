"""Bounded matched validation using actual frozen ClinVar/STRING inputs.
Historical full FPP used reverse edges + two self loops; old controls used one.
New graph interventions use the historical FULL operator consistently.
"""
import os,time,json,hashlib,traceback,sys,platform,importlib.metadata as md
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score
B=Path('/Data/u40902/TYX/ppi_audit_20261024');D=B/'data';O=B/'outputs'
torch.set_num_threads(4)
DEVICE='cuda'; assert torch.cuda.is_available()
SEEDS=[42,43,44];PARTS=['train','val','test']
def dump(name,d):
 p=O/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def load(p):return torch.load(p,map_location='cpu',weights_only=False)
def emb(p):
 a=load(p);return a['embeddings'].float() if isinstance(a,dict) else a.float()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for z in iter(lambda:f.read(2**20),b''):h.update(z)
 return h.hexdigest()
def metrics(y,p):
 assert np.isfinite(p).all()
 return {'auc':float(roc_auc_score(y,p)),'auprc':float(average_precision_score(y,p))}
class MLP(nn.Module):
 def __init__(self,n):
  super().__init__();self.net=nn.Sequential(nn.Linear(n,512),nn.ReLU(),nn.Dropout(.3),nn.Linear(512,128),nn.ReLU(),nn.Dropout(.3),nn.Linear(128,1))
 def forward(self,x):return self.net(x).squeeze(-1)
def seed(n):np.random.seed(n);torch.manual_seed(n);torch.cuda.manual_seed_all(n)

def check_inputs():
 checks=[]
 for e in json.loads((B/'code/input_manifest.json').read_text()):
  p=D/e['path'];assert p.stat().st_size==e['bytes'];assert sha(p)==e['sha256'];checks.append(e['path'])
 a=load(D/'model_inputs_main.pt');assert tuple(a['x'].shape)==(19622,1280);assert tuple(a['edge_index'].shape)==(2,13715404)
 assert torch.isfinite(a['x']).all()
 mapping={str(k):int(v) for k,v in a['protein_to_idx'].items()};splits={};audits={}
 c=pd.read_csv(D/'clinvar_with_cadd.csv',usecols=['VariationID','CADD_PHRED']).drop_duplicates('VariationID')
 am=pd.read_csv(D/'clinvar_with_alphamissense.csv',usecols=['VariationID','am_pathogenicity']).drop_duplicates('VariationID')
 for split in ['original','homology']:
  dfs={};nodes={};labels={};scores={};protsets=[];clusters=[]
  for part in PARTS:
   rel=f'clinvar_{part}_main.csv' if split=='original' else f'homology_aware_split/clinvar_{part}_homology.csv'
   df=pd.read_csv(D/rel,low_memory=False);assert not df['VariationID'].duplicated().any()
   df=df.merge(c,on='VariationID',how='left',validate='one_to_one',sort=False).merge(am,on='VariationID',how='left',validate='one_to_one',sort=False)
   idx=df['STRING_protein_id'].astype(str).map(mapping);assert idx.notna().all();nodes[part]=torch.tensor(idx.to_numpy(),dtype=torch.long)
   labels[part]=torch.tensor(df['Label'].to_numpy(),dtype=torch.float32);assert set(labels[part].unique().tolist())=={0.,1.}
   if split=='original':
    assert torch.equal(nodes[part],a[f'{part}_protein_idx'].long());assert torch.equal(labels[part],a[f'{part}_labels'].float())
   else:clusters.append(set(df['_cluster_id'].astype(str)))
   protsets.append(set(df['STRING_protein_id'].astype(str)));dfs[part]=df
  for i in range(3):
   for j in range(i):
    assert not protsets[i]&protsets[j]
    if split=='homology':assert not clusters[i]&clusters[j]
  fills={n:float(pd.to_numeric(dfs['train'][n],errors='coerce').median()) for n in ['CADD_PHRED','am_pathogenicity']}
  for part,df in dfs.items():
   df['am_missing']=pd.to_numeric(df['am_pathogenicity'],errors='coerce').isna().astype(float)
   for n,v in fills.items():df[n]=pd.to_numeric(df[n],errors='coerce').fillna(v)
   scores[part]=torch.tensor(df[['CADD_PHRED','am_pathogenicity','am_missing']].to_numpy(),dtype=torch.float32);assert torch.isfinite(scores[part]).all()
   if part=='test':
    cols=['VariationID','STRING_protein_id','Label']+(['_cluster_id'] if split=='homology' else [])
    df[cols].to_csv(O/f'{split}_test_rows.csv',index=False)
  audits[split]={'rows':{p:len(dfs[p]) for p in PARTS},'positives':{p:int(labels[p].sum()) for p in PARTS},'train_imputation_medians':fills,'protein_disjoint':True,'cluster_disjoint':True if split=='homology' else None}
  splits[split]={'dfs':dfs,'nodes':nodes,'labels':labels,'scores':scores}
 assert set(pd.concat(list(splits['original']['dfs'].values()))['VariationID'])==set(pd.concat(list(splits['homology']['dfs'].values()))['VariationID'])
 dump('input_audit.json',{'verified_hashes':checks,'splits':audits})
 return a,splits

def propagate(a,edge,columns=None):
 # Exact historical full operator: symmetrize edge list AFTER adding loops.
 x=a['x'].float() if columns is None else a['x'].float()[:,columns]
 n=x.shape[0];r,c=edge.long();loop=torch.arange(n);r0=torch.cat([r,loop]);c0=torch.cat([c,loop]);r1=torch.cat([r0,c0]);c1=torch.cat([c0,r0])
 A=torch.sparse_coo_tensor(torch.stack([r1,c1]),torch.ones(len(r1)),(n,n)).coalesce()
 deg=torch.sparse.sum(A,1).to_dense();inv=deg.pow(-.5);row,col=A.indices();val=inv[row]*A.values()*inv[col]
 S=torch.sparse_coo_tensor(A.indices(),val,(n,n)).coalesce().to(DEVICE)
 with torch.no_grad():
  z=x.to(DEVICE);h=torch.sparse.mm(S,z);h2=torch.sparse.mm(S,h);out=((z+h+h2)/3).cpu()
 del S,A;torch.cuda.empty_cache();return out

def representations(a):
 out={'raw_ESM':a['x'].float(),'full_FPP':emb(D/'gcn_embeddings/gcn_node_embeddings.pt')}
 cols=torch.arange(0,1280,80);calculated=propagate(a,a['edge_index'],cols)
 err=float((calculated-out['full_FPP'][:,cols]).abs().max());assert torch.allclose(calculated,out['full_FPP'][:,cols],rtol=2e-4,atol=2e-5),err
 audits={'full_saved_vs_recomputed_16_columns_max_abs_error':err,'operator':'A=(edge+I)+(edge+I)^T with duplicate weights summed, D from weighted row sums; FPP=(X+SX+S^2X)/3','historical_control_difference':'Old controls used (edge+edge.T+I); rerun uses full operator for ALL graph arms.'}
 n=a['x'].shape[0];edge=a['edge_index'];tr=torch.zeros(n,dtype=torch.bool);te=tr.clone();tr[a['train_protein_idx'].unique()]=True;te[a['test_protein_idx'].unique()]=True
 r,c=edge;remove=(tr[r]&te[c])|(te[r]&tr[c]);assert int(remove.sum())==1242422
 changes=[('edge_ablated_FPP',edge[:,~remove])]
 for i in range(3):
  saved=load(D/f'esm_gcn_topology_permutation_exp45/graph_permutation_{i}/node_permutation.pt').long()
  expected=torch.tensor(np.random.default_rng(6042+i).permutation(n));assert torch.equal(saved,expected)
  changes.append((f'permuted_FPP_g{i}',saved[edge]))
 for name,e in changes:
  print('GENERATING matched operator',name,flush=True)
  path=O/'derived_embeddings'/f'{name}.pt';path.parent.mkdir(exist_ok=True)
  z=propagate(a,e);assert torch.isfinite(z).all();torch.save(z,path);out[name]=z
  audits[name]={'path':str(path),'sha256':sha(path),'shape':list(z.shape),'historical_embedding_reused':False}
 assert all(v.shape==a['x'].shape and torch.isfinite(v).all() for v in out.values())
 dump('propagation_audit.json',audits);return out

def features(s,z=None,external=False):
 # GPU-resident lookup avoids expanding identical protein features for every variant.
 nodes={p:s['nodes'][p].to(DEVICE) for p in PARTS};zs=None if z is None else z.to(DEVICE)
 ext={p:s['scores'][p].to(DEVICE) for p in PARTS} if external else None
 def get(part,idx):
  idx=idx.to(DEVICE);pieces=[]
  if ext is not None:pieces.append(ext[part][idx])
  if zs is not None:pieces.append(zs[nodes[part][idx]])
  return pieces[0] if len(pieces)==1 else torch.cat(pieces,dim=1)
 return get,(0 if z is None else z.shape[1])+(3 if external else 0)

def evaluate(model,get,part,y):
 model.eval();probs=[]
 with torch.no_grad():
  for (idx,) in torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(y))),batch_size=1024,shuffle=False,num_workers=0):
   probs.append(torch.sigmoid(model(get(part,idx))).cpu().numpy())
 p=np.concatenate(probs);return metrics(y.numpy(),p),p

def fit(split,arm,s,get,dim,nseed,smoke=False):
 seed(nseed);labels=s['labels'];train=labels['train'];model=MLP(dim).to(DEVICE);optimizer=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
 weight=(len(train)-train.sum())/train.sum();lossfn=nn.BCEWithLogitsLoss(pos_weight=weight.to(DEVICE))
 # DataLoader on row indices preserves the historical CPU shuffle mechanism.
 loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.arange(len(train)),train),batch_size=1024,shuffle=True,num_workers=0)
 best=-np.inf;beststate=None;bestepoch=0;bad=0;history=[];t0=time.time()
 for epoch in range(1,2 if smoke else 51):
  model.train();loss_total=0.;seen=0
  for batch,(idx,y) in enumerate(loader):
   optimizer.zero_grad();loss=lossfn(model(get('train',idx)),y.to(DEVICE));assert torch.isfinite(loss);loss.backward();optimizer.step();loss_total+=float(loss)*len(y);seen+=len(y)
   if smoke and batch==3:break
  vm,vp=evaluate(model,get,'val',labels['val']);history.append({'epoch':epoch,'loss':loss_total/seen,'val_auc':vm['auc'],'val_auprc':vm['auprc']})
  print(f'{split} {arm} seed={nseed} epoch={epoch} loss={loss_total/seen:.5f} val_auc={vm["auc"]:.6f}',flush=True)
  if vm['auc']>best:best=vm['auc'];beststate={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};bestepoch=epoch;bad=0
  else:bad+=1
  if bad>=5:break
 if smoke:return {'arm':arm,'split':split,'dim':dim,'training_batches':4,'validation':vm,'elapsed_seconds':time.time()-t0}
 model.load_state_dict(beststate);vm,vp=evaluate(model,get,'val',labels['val']);tm,tp=evaluate(model,get,'test',labels['test'])
 q=O/'runs'/split/arm/f'seed_{nseed}';q.mkdir(parents=True,exist_ok=False);np.save(q/'val_probs.npy',vp);np.save(q/'test_probs.npy',tp);torch.save(beststate,q/'model.pt')
 result={'split':split,'arm':arm,'seed':nseed,'best_epoch':bestepoch,'epochs_run':epoch,'val_auc':vm['auc'],'val_auprc':vm['auprc'],'test_auc':tm['auc'],'test_auprc':tm['auprc'],'seconds':time.time()-t0,'dim':dim}
 (q/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(q/'history.json').write_text(json.dumps(history,indent=2)+'\n');print('FINAL '+json.dumps(result),flush=True)
 return result

def main():
 env={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'visible_device':os.environ.get('CUDA_VISIBLE_DEVICES'),'packages':{n:md.version(n) for n in ['numpy','pandas','scikit-learn','scipy','torchvision','triton']},'torch_threads':torch.get_num_threads(),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32}
 dump('runtime_environment.json',env);a,splits=check_inputs();reps=representations(a)
 smoke=[]
 for split,arm,z,external in [('original','raw_ESM',reps['raw_ESM'],False),('original','edge_ablated_FPP',reps['edge_ablated_FPP'],False),('homology','scores',None,True),('homology','scores_plus_full_FPP',reps['full_FPP'],True)]:
  get,dim=features(splits[split],z,external);smoke.append(fit(split,arm,splits[split],get,dim,42,True));del get;torch.cuda.empty_cache()
 dump('runtime_smoke.json',{'passed':True,'test_set_used':False,'checks':smoke})
 jobs=[('original',name,z,False) for name,z in reps.items()]+[('original_fusion','scores_capacity_matched',None,True)]+[('original_fusion','scores_plus_'+name,z,True) for name,z in reps.items()]+[('homology','scores_capacity_matched',None,True),('homology','scores_plus_raw_ESM',reps['raw_ESM'],True),('homology','scores_plus_full_FPP',reps['full_FPP'],True)]
 results=[];dump('progress.json',{'state':'training','completed':0,'planned':48})
 for split,arm,z,external in jobs:
  s=splits['original' if split=='original_fusion' else split]
  get,dim=features(s,z,external)
  for nseed in SEEDS:
   result=fit(split,arm,s,get,dim,nseed);results.append(result);pd.DataFrame(results).to_csv(O/'per_run_metrics.csv',index=False);dump('progress.json',{'state':'training','completed':len(results),'planned':48,'last':result})
  del get;torch.cuda.empty_cache()
 assert len(results)==48
 df=pd.DataFrame(results);summary=df.groupby(['split','arm']).agg(n=('seed','count'),auc_mean=('test_auc','mean'),auc_sd=('test_auc','std'),auprc_mean=('test_auprc','mean'),auprc_sd=('test_auprc','std')).reset_index();summary.to_csv(O/'summary_metrics.csv',index=False)
 dump('progress.json',{'state':'training_complete_analysis_pending','completed':len(results),'planned':48});print(summary.to_string(index=False),flush=True)
if __name__=='__main__':
 try:main()
 except Exception:
  dump('failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
