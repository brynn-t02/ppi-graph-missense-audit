"""Unchanged fitted-classifier and propagation functions from the verified matched protocol.
Only filesystem roots are parameterized. No score/fusion inputs are required here.
"""
import os,time,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import roc_auc_score,average_precision_score
B=Path(os.environ.get('PPI_UNIFICATION_ROOT','/root/autodl-tmp/ppi_unification_20261108'))
D=B/'data/processed';O=B/'outputs'
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
