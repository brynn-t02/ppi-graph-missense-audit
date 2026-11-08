"""Canonical reporting and paired protein uncertainty for completed unified fits.
No fitting, threshold tuning, feature regeneration, or test-based selection.
"""
import json, os, time, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

B=Path(os.environ.get('PPI_UNIFICATION_ROOT','/root/autodl-tmp/ppi_unification_20261108'))
O=B/'outputs';P=O/'public';P.mkdir(exist_ok=True)
SEEDS=[42,43,44]
STRATEGIES=['random_pair_removal','high_confidence_first','low_confidence_first']
def dump(name,value):(O/name).write_text(json.dumps(value,indent=2)+'\n')
def metric(y,p):return np.array([roc_auc_score(y,p),average_precision_score(y,p)],dtype=np.float64)
def save(df,path):df.to_csv(path,index=False,float_format='%.17g')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()

assert json.loads((O/'training_complete.json').read_text())['classifier_fits']==39
assert json.loads((O/'endpoint_input_gate_v2.json').read_text())['passed']
runs=pd.read_csv(O/'training_per_run_metrics.csv');assert len(runs)==39
rows={s:pd.read_csv(O/f'{s}_test_rows.csv') for s in ['original','homology']}
allmetrics=[];ensemble_rows=[];ensembles={};checkpoints={}
for (split,arm),group in runs.groupby(['split','arm'],sort=True):
    assert sorted(group.seed)==SEEDS
    d=rows[split];y=d.Label.to_numpy();df=d.rename(columns={'Label':'label'}).copy();pred=[]
    for seed in SEEDS:
        q=O/'runs'/split/arm/f'seed_{seed}'
        p=np.load(q/'canonical_test_probs.npy').astype(np.float64)
        assert len(p)==len(d) and np.isfinite(p).all()
        assert pd.DataFrame({'protein':d.STRING_protein_id,'p':p}).groupby('protein').p.nunique().eq(1).all()
        r=group[group.seed==seed].iloc[0].to_dict();r.update(dict(zip(['test_auc','test_auprc'],metric(y,p))))
        r['reporting']='CPU float32 node inference, mapped to variants';allmetrics.append(r)
        df[f'p_seed_{seed}']=p;pred.append(p)
        checkpoints[f'{split}/{arm}/seed_{seed}']=sha(q/'model.pt')
    ens=np.mean(np.stack(pred),axis=0,dtype=np.float64);df['ensemble_float64']=ens
    ensembles[(split,arm)]=ens
    v=metric(y,ens);ensemble_rows.append({'split':split,'arm':arm,'auc':v[0],'auprc':v[1],'n':len(d),'proteins':d.STRING_protein_id.nunique(),'positive':int(y.sum())})
    save(df,P/f'test_predictions__{split}__{arm}.csv.gz')
canonical=pd.DataFrame(allmetrics)
save(canonical,O/'canonical_per_run_metrics.csv')
summary=canonical.groupby(['split','arm']).agg(n=('seed','count'),auc_mean=('test_auc','mean'),auc_sd=('test_auc','std'),auprc_mean=('test_auprc','mean'),auprc_sd=('test_auprc','std')).reset_index()
save(summary,O/'canonical_summary_metrics.csv');save(pd.DataFrame(ensemble_rows),O/'ensemble_metrics.csv')
curve=[];curve_runs=[]
for strategy in STRATEGIES:
    for fraction in [0.,.25,.5,.75,1.]:
        arm='full_FPP' if fraction==0 else 'edge_ablated_FPP' if fraction==1 else f'{strategy}_fraction_{fraction:.2f}'
        r=summary[(summary.split=='original')&(summary.arm==arm)].iloc[0].to_dict()
        r.update(strategy=strategy,fraction_removed=fraction,source='newly verified endpoint fit shared by strategies' if fraction in [0.,1.] else 'new matched interior fit')
        curve.append(r)
        for rr in canonical[(canonical.split=='original')&(canonical.arm==arm)].to_dict('records'):
            curve_runs.append({**rr,'strategy':strategy,'fraction_removed':fraction,'endpoint_shared':fraction in [0.,1.]})
save(pd.DataFrame(curve),O/'figure1E_summary.csv');save(pd.DataFrame(curve_runs),O/'figure1E_per_seed.csv')
save(summary[summary.split=='homology'],O/'figure2D_summary.csv')

# Protein-clustered paired bootstrap uses the same frozen sequence as canonical work.
d=rows['homology'];y=d.Label.to_numpy(dtype=np.int64);uniq,inv=np.unique(d.STRING_protein_id.astype(str).to_numpy(),return_inverse=True)
pred={a:ensembles[('homology',a)] for a in ['raw_ESM','full_FPP']};cache={}
for a,p in pred.items():
    order=np.argsort(p,kind='stable');v=p[order];ends=np.flatnonzero(np.r_[v[1:]!=v[:-1],True]);cache[a]=(order,ends)
def weighted(arm,w):
    order,ends=cache[arm];yy=y[order];ww=w[order]
    tp=np.cumsum(ww*yy)[ends];fp=np.cumsum(ww*(1-yy))[ends];pos=tp[-1];neg=fp[-1]
    if pos==0 or neg==0:return np.array([np.nan,np.nan])
    auc=1-np.trapz(np.r_[0,tp/pos],np.r_[0,fp/neg])
    pg=np.diff(np.r_[0,tp])[::-1];ng=np.diff(np.r_[0,fp])[::-1];ctp=np.cumsum(pg);cfp=np.cumsum(ng)
    ap=np.sum(pg/pos*np.divide(ctp,ctp+cfp,out=np.zeros_like(ctp),where=ctp+cfp>0))
    return np.array([auc,ap])
ones=np.ones(len(y));base={a:metric(y,p) for a,p in pred.items()}
for a in pred:assert np.max(np.abs(weighted(a,ones)-base[a]))<1e-12
# Validation RNG is separate and must not consume the bootstrap draw sequence.
validation_rng=np.random.default_rng(20261108);vw=validation_rng.integers(0,4,len(uniq))[inv].astype(float)
for a,p in pred.items():
    direct=np.array([roc_auc_score(y,p,sample_weight=vw),average_precision_score(y,p,sample_weight=vw)])
    assert np.max(np.abs(weighted(a,vw)-direct))<1e-12
rng=np.random.default_rng(20261024);repeats=[];weights_digest=hashlib.sha256();independent_checks=[]
for i in range(2000):
    counts=np.bincount(rng.integers(0,len(uniq),size=len(uniq)),minlength=len(uniq))
    weights_digest.update(counts.astype('<i8').tobytes());w=counts[inv].astype(float)
    m0=weighted('raw_ESM',w);m1=weighted('full_FPP',w);delta=m1-m0
    repeats.append({'replicate':i,'delta_auc':delta[0],'delta_auprc':delta[1]})
    if i in [0,1,2,499,999,1499,1999] and np.isfinite(delta).all():
        direct={a:np.array([roc_auc_score(y,p,sample_weight=w),average_precision_score(y,p,sample_weight=w)]) for a,p in pred.items()}
        err=float(np.max(np.abs(delta-(direct['full_FPP']-direct['raw_ESM']))));assert err<1e-12
        independent_checks.append({'replicate':i,'max_abs_error':err})
    if (i+1)%500==0:print('homology paired protein bootstrap',i+1,flush=True)
bs=pd.DataFrame(repeats);save(bs,O/'homology_paired_protein_bootstrap_replicates.csv.gz')
vv=bs[['delta_auc','delta_auprc']].to_numpy();valid=np.isfinite(vv).all(1);assert valid.sum()>1900
ci=np.quantile(vv[valid],[.025,.975],axis=0);delta=base['full_FPP']-base['raw_ESM']
result={'comparison':'full_FPP minus raw_ESM','point_estimates_from':'float64 means of three canonical node-mapped probability arrays','delta_auc':float(delta[0]),'delta_auprc':float(delta[1]),'auc_ci95':ci[:,0].tolist(),'auprc_ci95':ci[:,1].tolist(),'resamples':2000,'valid_resamples':int(valid.sum()),'seed':20261024,'resampling_unit':'STRING protein: all variants with sampled multiplicity, identical weights for both models','proteins':len(uniq),'variants':len(y),'positive_variants':int(y.sum()),'interval':'2.5/97.5 percentile of paired bootstrap differences','scope':'Conditional on fitted models and frozen graph; no retraining or graph-sampling uncertainty; no multiplicity adjustment','weights_sha256':weights_digest.hexdigest(),'independent_sklearn_weighted_checks':independent_checks}
dump('figure2D_protein_bootstrap.json',result)
pd.DataFrame([{'seed':s,'delta_auc':float(canonical[(canonical.split=='homology')&(canonical.arm=='full_FPP')&(canonical.seed==s)].test_auc.iloc[0]-canonical[(canonical.split=='homology')&(canonical.arm=='raw_ESM')&(canonical.seed==s)].test_auc.iloc[0]),'delta_auprc':float(canonical[(canonical.split=='homology')&(canonical.arm=='full_FPP')&(canonical.seed==s)].test_auprc.iloc[0]-canonical[(canonical.split=='homology')&(canonical.arm=='raw_ESM')&(canonical.seed==s)].test_auprc.iloc[0])} for s in SEEDS]).to_csv(O/'homology_paired_seed_deltas.csv',index=False)
# Protein-clustered paired bootstrap uses the same frozen sequence as canonical work.
d=rows['original'];y=d.Label.to_numpy(dtype=np.int64);uniq,inv=np.unique(d.STRING_protein_id.astype(str).to_numpy(),return_inverse=True)
pred={a:ensembles[('original',a)] for a in ['edge_ablated_FPP','full_FPP']};cache={}
for a,p in pred.items():
    order=np.argsort(p,kind='stable');v=p[order];ends=np.flatnonzero(np.r_[v[1:]!=v[:-1],True]);cache[a]=(order,ends)
def weighted(arm,w):
    order,ends=cache[arm];yy=y[order];ww=w[order]
    tp=np.cumsum(ww*yy)[ends];fp=np.cumsum(ww*(1-yy))[ends];pos=tp[-1];neg=fp[-1]
    if pos==0 or neg==0:return np.array([np.nan,np.nan])
    auc=1-np.trapz(np.r_[0,tp/pos],np.r_[0,fp/neg])
    pg=np.diff(np.r_[0,tp])[::-1];ng=np.diff(np.r_[0,fp])[::-1];ctp=np.cumsum(pg);cfp=np.cumsum(ng)
    ap=np.sum(pg/pos*np.divide(ctp,ctp+cfp,out=np.zeros_like(ctp),where=ctp+cfp>0))
    return np.array([auc,ap])
ones=np.ones(len(y));base={a:metric(y,p) for a,p in pred.items()}
for a in pred:assert np.max(np.abs(weighted(a,ones)-base[a]))<1e-12
# Validation RNG is separate and must not consume the bootstrap draw sequence.
validation_rng=np.random.default_rng(20261108);vw=validation_rng.integers(0,4,len(uniq))[inv].astype(float)
for a,p in pred.items():
    direct=np.array([roc_auc_score(y,p,sample_weight=vw),average_precision_score(y,p,sample_weight=vw)])
    assert np.max(np.abs(weighted(a,vw)-direct))<1e-12
rng=np.random.default_rng(20261024);repeats=[];weights_digest=hashlib.sha256();independent_checks=[]
for i in range(2000):
    counts=np.bincount(rng.integers(0,len(uniq),size=len(uniq)),minlength=len(uniq))
    weights_digest.update(counts.astype('<i8').tobytes());w=counts[inv].astype(float)
    m0=weighted('edge_ablated_FPP',w);m1=weighted('full_FPP',w);delta=m1-m0
    repeats.append({'replicate':i,'delta_auc':delta[0],'delta_auprc':delta[1]})
    if i in [0,1,2,499,999,1499,1999] and np.isfinite(delta).all():
        direct={a:np.array([roc_auc_score(y,p,sample_weight=w),average_precision_score(y,p,sample_weight=w)]) for a,p in pred.items()}
        err=float(np.max(np.abs(delta-(direct['full_FPP']-direct['edge_ablated_FPP']))));assert err<1e-12
        independent_checks.append({'replicate':i,'max_abs_error':err})
    if (i+1)%500==0:print('original ablation paired protein bootstrap',i+1,flush=True)
bs=pd.DataFrame(repeats);save(bs,O/'original_ablation_paired_protein_bootstrap_replicates.csv.gz')
vv=bs[['delta_auc','delta_auprc']].to_numpy();valid=np.isfinite(vv).all(1);assert valid.sum()>1900
ci=np.quantile(vv[valid],[.025,.975],axis=0);delta=base['full_FPP']-base['edge_ablated_FPP']
result={'comparison':'full_FPP minus edge_ablated_FPP','point_estimates_from':'float64 means of three canonical node-mapped probability arrays','delta_auc':float(delta[0]),'delta_auprc':float(delta[1]),'auc_ci95':ci[:,0].tolist(),'auprc_ci95':ci[:,1].tolist(),'resamples':2000,'valid_resamples':int(valid.sum()),'seed':20261024,'resampling_unit':'STRING protein: all variants with sampled multiplicity, identical weights for both models','proteins':len(uniq),'variants':len(y),'positive_variants':int(y.sum()),'interval':'2.5/97.5 percentile of paired bootstrap differences','scope':'Conditional on fitted models and frozen graph; no retraining or graph-sampling uncertainty; no multiplicity adjustment','weights_sha256':weights_digest.hexdigest(),'independent_sklearn_weighted_checks':independent_checks}
dump('figure1D_protein_bootstrap.json',result)
dump('analysis_complete.json',{'passed':True,'classifier_fits':39,'canonical_arms':13,'curve_conditions':15,'unique_curve_conditions':11,'curve_endpoints_reused_across_strategies':True,'homology_models':2,'sample_SD_ddof':1,'bootstrap_repeats':2000,'bootstrap_comparisons':2,'new_training_or_selection_by_analysis':False,'checkpoint_hashes':checkpoints,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
print(summary.to_string(index=False),flush=True);print(json.dumps(result,indent=2),flush=True)
