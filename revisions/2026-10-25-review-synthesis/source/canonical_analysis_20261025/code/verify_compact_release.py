"""Verify public gzip hashes and recompute metrics without models or tensors.

Default requires only numpy/pandas. --ci additionally requires scikit-learn and
recomputes all 27 protein-bootstrap comparisons plus the variant-bootstrap
contrast. This program reads artifacts and prints a JSON report; it writes none.
"""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
import pandas as pd

def metrics(y, p):
    """Tie-aware binary ROC AUC and non-interpolated average precision."""
    y=np.asarray(y,dtype=np.int8); p=np.asarray(p,dtype=np.float64)
    assert set(np.unique(y))=={0,1} and np.isfinite(p).all()
    order=np.argsort(-p,kind='stable'); yy=y[order]; pp=p[order]
    ends=np.flatnonzero(np.r_[pp[1:]!=pp[:-1],True])
    tp=np.cumsum(yy,dtype=np.float64)[ends]
    fp=(ends+1)-tp; pos=np.diff(np.r_[0.,tp]); neg=np.diff(np.r_[0.,fp])
    auc=np.sum(pos*(fp[-1]-fp+.5*neg))/(tp[-1]*fp[-1])
    ap=np.sum(pos*tp/(tp+fp))/tp[-1]
    return np.array([auc,ap])

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--ci',action='store_true')
    args=parser.parse_args(); root=args.root; public=root/'public'
    read=lambda p:pd.read_csv(p,float_precision='round_trip')
    manifest=json.loads((root/'public_manifest.json').read_text())
    for row in manifest['files']:
        p=public/row['file']; assert p.stat().st_size==row['bytes']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],p.name
        d=read(p); assert len(d)==row['rows'] and list(d.columns)==row['columns']
    full=read(public/'frozen_samples_177213.csv.gz')
    assert len(full)==177213 and not full.VariationID.duplicated().any()
    assert set(full.label)=={0,1}
    for split in ['original','homology']: assert set(full[split+'_split'])=={'train','val','test'}
    clusters={part:set(full.loc[full.homology_split==part,'homology_cluster_id']) for part in ['train','val','test']}
    assert not (clusters['train']&clusters['val'] or clusters['train']&clusters['test'] or clusters['val']&clusters['test'])
    seeds=[42,43,44]; predictions={}; rows={}
    for p in sorted(public.glob('test_predictions__*.csv.gz')):
        _,pop,arm=p.name.removesuffix('.csv.gz').split('__');d=read(p)
        if pop in rows:
            for column in ['VariationID','label','STRING_protein_id']: assert np.array_equal(d[column],rows[pop][column])
        else: rows[pop]=d
        actual_split='original' if pop=='original_fusion' else pop
        member=full.set_index('VariationID').loc[d.VariationID]
        assert (member[actual_split+'_split']=='test').all()
        assert np.array_equal(member.label,d.label) and np.array_equal(member.STRING_protein_id,d.STRING_protein_id)
        pp=np.stack([d[f'p_seed_{s}'].to_numpy(dtype=np.float64) for s in seeds])
        assert np.array_equal(pp.mean(axis=0,dtype=np.float64),d.ensemble_float64)
        if pop=='original':
            for col in [f'p_seed_{s}' for s in seeds]+['ensemble_float64']:
                assert d.groupby('STRING_protein_id')[col].nunique().max()==1
        predictions.setdefault(pop,{})[arm]=pp
    for pop,prefix in [('original',''),('original_fusion','scores_plus_')]:
        predictions[pop][prefix+'permuted_FPP_graphmean']=np.stack([predictions[pop][prefix+f'permuted_FPP_g{i}'] for i in range(3)]).mean(axis=0,dtype=np.float64)
    ext=read(public/'external_membership_and_predictions.csv.gz')
    assert len(ext)==40900 and not ext.external_id.duplicated().any()
    assert (ext.unseen_all_original<=ext.primary_trainval_unseen).all()
    for pop,flag,n in [('external_all',None,40900),('external_primary','primary_trainval_unseen',10785),('external_strict','unseen_all_original',6934)]:
        d=ext if flag is None else ext[ext[flag]==1];assert len(d)==n
        rows[pop]=d;predictions[pop]={}
        for arm in ['raw_ESM','full_FPP','BioGRID_FPP']:
            pp=np.stack([d[f'{arm}_seed_{s}'].to_numpy(dtype=np.float64) for s in seeds])
            assert np.array_equal(pp.mean(axis=0,dtype=np.float64),d[f'{arm}_ensemble_float64'])
            predictions[pop][arm]=pp
    per_seed=read(root/'evaluation_per_seed.csv');ens_report=read(root/'ensemble_metrics.csv')
    errors=[];ensembles={}; count=0
    for pop,arms in predictions.items():
        y=rows[pop].label.to_numpy();ensembles[pop]={}
        for arm,pp in arms.items():
            for i,seed in enumerate(seeds):
                report=per_seed[(per_seed.population==pop)&(per_seed.arm==arm)&(per_seed.seed==seed)]
                assert len(report)==1
                errors.append(float(np.max(np.abs(metrics(y,pp[i])-report[['auc','auprc']].to_numpy(dtype=float)[0]))));count+=1
            ens=pp.mean(axis=0,dtype=np.float64);ensembles[pop][arm]=ens
            report=ens_report[(ens_report.population==pop)&(ens_report.arm==arm)];assert len(report)==1
            errors.append(float(np.max(np.abs(metrics(y,ens)-report[['auc','auprc']].to_numpy(dtype=float)[0]))))
    assert count==len(per_seed)==90 and sum(map(len,ensembles.values()))==len(ens_report)==30
    assert max(errors)<1e-12
    result={'passed':True,'public_file_hashes':len(manifest['files']),'frozen_sample_rows':len(full),'seed_metric_rows':count,'ensemble_metric_rows':len(ens_report),'max_abs_metric_error':max(errors),'metrics':'tie-aware binary ROC AUC and non-interpolated AP','bootstrap_recomputed':False,'limitations':'This compact-only check does not recreate training, input features, or the clinical-score completeness mask used by the complete-case table.'}
    if args.ci:
        from analyze_external_validation import paired_bootstrap,WeightedScores,validate_weighted_metrics
        assert validate_weighted_metrics()['passed']
        expected=json.loads((root/'cluster_bootstrap.json').read_text());ci_errors=[];comparisons=0
        for pop,pairs in expected['comparisons'].items():
            pair_list=[tuple(k.split('_vs_')) for k in pairs]
            out=paired_bootstrap(rows[pop].label.to_numpy(),rows[pop].STRING_protein_id.to_numpy(),ensembles[pop],pair_list,repeats=2000,seed=20261024)
            for name,row in out.items():
                for key in ['delta_auc','delta_auprc','auc_ci95','auprc_ci95']:
                    ci_errors.append(float(np.max(np.abs(np.asarray(row[key])-np.asarray(pairs[name][key])))))
                comparisons+=1
        y=rows['original'].label.to_numpy(); f=WeightedScores(y,ensembles['original']['full_FPP']);g=WeightedScores(y,ensembles['original']['raw_ESM']);rng=np.random.default_rng(20261024);vv=[]
        for _ in range(2000):
            weight=np.bincount(rng.integers(0,len(y),len(y)),minlength=len(y));vv.append(f(weight)-g(weight))
        q=np.quantile(vv,[.025,.975],axis=0);v=expected['variant_bootstrap_original_full_vs_raw']
        ci_errors.append(float(np.max(np.abs(q-np.array([v['auc_ci95'],v['auprc_ci95']]).T))))
        assert comparisons==27 and max(ci_errors)<1e-10
        result.update(bootstrap_recomputed=True,protein_comparisons=27,bootstrap_repeats=2000,variant_bootstrap_contrast=True,max_abs_interval_error=max(ci_errors))
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
