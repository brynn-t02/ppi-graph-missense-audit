"""Analyze frozen BioGRID fits and external-source predictions, without fitting.

Run after the experiment runner, before verify_external_validation.py. Inputs
remain read-only; only compact summaries are written. Intervals condition on fits.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

DEFAULT_OLD_ROOT = Path('/Data/u40902/TYX/ppi_audit_20261024')
DEFAULT_ROOT = DEFAULT_OLD_ROOT / 'external_validation_20261024'
SEEDS = (42, 43, 44)
NODE_ARMS = ('raw_ESM', 'full_FPP', 'BioGRID_FPP')
NEW_ARMS = {
    'original': 'BioGRID_FPP',
    'original_fusion': 'scores_plus_BioGRID_FPP',
    'homology': 'scores_plus_BioGRID_FPP',
}
OLD_ARMS = {
    'original': ('raw_ESM', 'full_FPP'),
    'original_fusion': ('scores_capacity_matched', 'scores_plus_raw_ESM', 'scores_plus_full_FPP'),
    'homology': ('scores_capacity_matched', 'scores_plus_raw_ESM', 'scores_plus_full_FPP'),
}
COHORTS = {
    'external_primary': ('external_cohort_primary.csv', 10785, 'primary'),
    'external_strict': ('external_cohort_unseen_all_original.csv', 6934, 'strict sensitivity'),
    'external_all': ('external_cohort_all_variant_disjoint.csv', 40900, 'descriptive; includes original training proteins'),
}
EXTERNAL_ROWS = {
    'external_primary': 'primary.csv',
    'external_strict': 'strict_unseen_all_original.csv',
    'external_all': 'all_variant_disjoint.csv',
}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def probabilities(path, expected_rows):
    p = np.load(path, allow_pickle=False)
    assert p.shape == (expected_rows,), (str(path), p.shape, expected_rows)
    assert np.isfinite(p).all() and ((0 <= p) & (p <= 1)).all(), str(path)
    return p.astype(np.float64, copy=False)


def binary_labels(values):
    y = np.asarray(values)
    assert y.ndim == 1 and set(np.unique(y)) == {0, 1}
    return y.astype(np.int8)


def metrics(y, p):
    return {'auc': float(roc_auc_score(y, p)), 'auprc': float(average_precision_score(y, p))}


class WeightedScores:
    """Tie-grouped weighted ROC AUC and non-interpolated average precision."""
    def __init__(self, y, p):
        self.y = binary_labels(y)
        self.p = np.asarray(p, dtype=np.float64)
        assert self.p.shape == self.y.shape and np.isfinite(self.p).all()
        self.order = np.argsort(self.p, kind='stable')
        ordered = self.p[self.order]
        self.ends = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
        self.ordered_y = self.y[self.order]

    def __call__(self, weights):
        w = np.asarray(weights, dtype=np.float64)
        assert w.shape == self.y.shape and np.isfinite(w).all() and (w >= 0).all()
        w = w[self.order]
        tp = np.cumsum(w * self.ordered_y)[self.ends]
        fp = np.cumsum(w * (1 - self.ordered_y))[self.ends]
        positive, negative = tp[-1], fp[-1]
        if positive == 0 or negative == 0:
            return np.array([np.nan, np.nan])
        auc = 1.0 - np.trapz(np.r_[0, tp / positive], np.r_[0, fp / negative])
        pos = np.diff(np.r_[0, tp])[::-1]
        neg = np.diff(np.r_[0, fp])[::-1]
        ctp, cfp = np.cumsum(pos), np.cumsum(neg)
        precision = np.divide(ctp, ctp + cfp, out=np.zeros_like(ctp), where=(ctp + cfp) > 0)
        ap = np.sum(pos / positive * precision)
        return np.array([auc, ap])


def validate_weighted_metrics():
    """Meaningful numerical regression: ties, zero weights, class imbalance."""
    rng = np.random.default_rng(20261024)
    cases = [
        (np.array([0, 1, 0, 1]), np.array([.5, .5, .5, .5]), np.array([1, 2, 0, 3])),
        (np.array([1, 0, 1, 0, 0, 1]), np.array([.2, .2, .8, .8, .4, .4]), np.array([0, 3, 2, 1, 4, 2])),
    ]
    for _ in range(12):
        y = rng.integers(0, 2, 200)
        p = rng.integers(0, 8, 200) / 7
        w = rng.integers(0, 5, 200)
        cases.append((y, p, w))
    for y, p, w in cases:
        fast = WeightedScores(y, p)(w)
        ref = np.array([roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w)])
        assert np.allclose(fast, ref, atol=1e-12, rtol=0), (fast, ref)
    return {'passed': True, 'synthetic_tied_weight_cases': len(cases), 'tolerance': 1e-12}


def paired_bootstrap(y, ids, predictions, pairs, repeats=2000, seed=20261024):
    ids = np.asarray(ids, dtype=str)
    unique, inverse = np.unique(ids, return_inverse=True)
    assert len(ids) == len(y) and len(unique) > 1
    funcs = {name: WeightedScores(y, p) for name, p in predictions.items()}
    ones = np.ones(len(y))
    points = {name: fun(ones) for name, fun in funcs.items()}
    # Separate validation RNG leaves the documented bootstrap sequence untouched.
    validation_rng = np.random.default_rng(seed + 1)
    for _ in range(3):
        w = validation_rng.integers(0, 4, len(unique))[inverse]
        assert np.sum(w * y) > 0 and np.sum(w * (1 - y)) > 0
        for name, p in predictions.items():
            ref = [roc_auc_score(y, p, sample_weight=w), average_precision_score(y, p, sample_weight=w)]
            assert np.allclose(funcs[name](w), ref, atol=1e-10, rtol=0), name
    samples = {f'{a}_vs_{b}': np.empty((repeats, 2)) for a, b in pairs}
    rng = np.random.default_rng(seed)
    for i in range(repeats):
        weights = np.bincount(rng.integers(0, len(unique), len(unique)), minlength=len(unique))[inverse]
        values = {name: f(weights) for name, f in funcs.items()}
        for a, b in pairs:
            samples[f'{a}_vs_{b}'][i] = values[a] - values[b]
    out = {}
    for a, b in pairs:
        key = f'{a}_vs_{b}'
        valid = np.isfinite(samples[key]).all(axis=1)
        assert valid.sum() >= int(.99 * repeats), (key, int(valid.sum()), repeats)
        ci = np.quantile(samples[key][valid], [.025, .975], axis=0)
        delta = points[a] - points[b]
        out[key] = {
            'delta_auc': float(delta[0]), 'auc_ci95': ci[:, 0].tolist(),
            'delta_auprc': float(delta[1]), 'auprc_ci95': ci[:, 1].tolist(),
            'finite_replicates': int(valid.sum()), 'discarded_one_class_replicates': int((~valid).sum()),
            'proteins': len(unique), 'variants': len(y),
        }
    return out


def original_test_rows(old_root, split):
    split = 'original' if split == 'original_fusion' else split
    name = 'clinvar_test_main.csv' if split == 'original' else 'homology_aware_split/clinvar_test_homology.csv'
    rows = pd.read_csv(old_root / 'data' / name, usecols=['VariationID', 'STRING_protein_id', 'Label'])
    frozen = pd.read_csv(old_root / 'outputs' / f'{split}_test_rows.csv')
    for column in ['VariationID', 'STRING_protein_id', 'Label']:
        assert np.array_equal(rows[column].astype(str), frozen[column].astype(str)), (split, column)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--old-root', type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    numerical_check = validate_weighted_metrics()
    if args.self_test:
        print(json.dumps(numerical_check, indent=2)); return
    root, old = args.root, args.old_root
    out = root / 'outputs'
    progress = json.loads((out / 'progress.json').read_text())
    assert progress['completed'] == 9
    new_runs = pd.read_csv(out / 'per_run_metrics.csv')
    assert len(new_runs) == 9
    per_run, ensemble_rows, seed_deltas = [], [], []
    blocks, input_paths = [], set()
    for split, new_arm in NEW_ARMS.items():
        rows = original_test_rows(old, split)
        predictions = {}
        for arm in (*OLD_ARMS[split], new_arm):
            source = root if arm == new_arm else old
            seed_predictions = []
            for seed in SEEDS:
                q = source / 'outputs/runs' / split / arm / f'seed_{seed}'
                p = probabilities(q / 'test_probs.npy', len(rows))
                input_paths.update([q / 'test_probs.npy', q / 'metrics.json'])
                seed_predictions.append(p)
                m = metrics(binary_labels(rows.Label.to_numpy()), p)
                recorded = json.loads((q / 'metrics.json').read_text())
                assert np.allclose([m['auc'], m['auprc']], [recorded['test_auc'], recorded['test_auprc']], atol=1e-12, rtol=0)
                per_run.append({'population': split, 'arm': arm, 'seed': seed, 'source': 'new_BioGRID_fit' if source == root else 'frozen_original_fit', 'variants': len(rows), 'proteins': rows.STRING_protein_id.nunique(), 'positive': int(rows.Label.sum()), 'auc': m['auc'], 'auprc': m['auprc']})
            predictions[arm] = np.stack(seed_predictions)
        pairs = [(new_arm, arm) for arm in OLD_ARMS[split]]
        blocks.append((split, 'matched second-network comparison', rows.Label.to_numpy(), rows.STRING_protein_id.to_numpy(), predictions, pairs))
    node_predictions = {}
    for arm in NODE_ARMS:
        pp = []
        for seed in SEEDS:
            path = out / 'node_predictions' / arm / f'seed_{seed}.npy'
            pp.append(probabilities(path, 19622)); input_paths.add(path)
        node_predictions[arm] = np.stack(pp)
    for population, (filename, n, role) in COHORTS.items():
        path = root / 'cohort_recon' / filename
        original_rows = pd.read_csv(path)
        mapped_path = out / 'external_rows' / EXTERNAL_ROWS[population]
        rows = pd.read_csv(mapped_path)
        assert len(rows) == n and not rows.external_id.duplicated().any()
        for column in ['external_id', 'label', 'STRING_protein_id']:
            assert np.array_equal(rows[column].astype(str), original_rows[column].astype(str)), (population, column)
        assert rows.actual_node_index.notna().all() and np.equal(rows.actual_node_index, rows.actual_node_index.astype(int)).all()
        idx = rows.actual_node_index.to_numpy(dtype=np.int64)
        assert (idx >= 0).all() and (idx < 19622).all()
        y = binary_labels(rows.label.to_numpy()); input_paths.update([path, mapped_path])
        predictions = {arm: p[:, idx] for arm, p in node_predictions.items()}
        for arm, pp in predictions.items():
            for seed, p in zip(SEEDS, pp):
                m = metrics(y, p)
                per_run.append({'population': population, 'arm': arm, 'seed': seed, 'source': 'frozen_checkpoint_external_inference', 'variants': len(rows), 'proteins': rows.STRING_protein_id.nunique(), 'positive': int(y.sum()), 'auc': m['auc'], 'auprc': m['auprc']})
        pairs = [('full_FPP', 'raw_ESM'), ('BioGRID_FPP', 'raw_ESM'), ('BioGRID_FPP', 'full_FPP')]
        blocks.append((population, role, y, rows.STRING_protein_id.to_numpy(), predictions, pairs))
    bootstrap = {
        'bootstrap_repeats': 2000, 'seed': 20261024,
        'rng_policy': 'Restart documented seed per population; identical protein resamples paired across all models within a population.',
        'resampling_unit': 'STRING protein with replacement, retaining all variant rows with cluster multiplicity',
        'scope': 'Conditional on fitted ensembles and frozen networks; no retraining, newly sampled network, independent-patient, or homology-disjoint guarantee. Comparisons exploratory without multiplicity adjustment. All-cohort external results descriptive because training proteins occur.',
        'numerical_validation': numerical_check, 'comparisons': {},
    }
    for population, role, yy, ids, pp, pairs in blocks:
        y = binary_labels(yy)
        ens = {arm: pred.mean(axis=0) for arm, pred in pp.items()}
        for arm, p in ens.items():
            m = metrics(y, p)
            ensemble_rows.append({'population': population, 'role': role, 'arm': arm, 'variants': len(y), 'proteins': len(np.unique(ids)), 'positive': int(y.sum()), 'prevalence': float(y.mean()), **m})
        bootstrap['comparisons'][population] = paired_bootstrap(y, ids, ens, pairs)
        for a, b in pairs:
            for j, seed in enumerate(SEEDS):
                ma, mb = metrics(y, pp[a][j]), metrics(y, pp[b][j])
                seed_deltas.append({'population': population, 'comparison': f'{a}_vs_{b}', 'seed': seed, 'delta_auc': ma['auc'] - mb['auc'], 'delta_auprc': ma['auprc'] - mb['auprc']})
        print('ANALYZED', population, len(y), 'variants', len(np.unique(ids)), 'proteins', flush=True)
    frame = pd.DataFrame(per_run)
    assert len(frame) == 60 and frame.groupby(['population', 'arm']).size().eq(3).all()
    frame.to_csv(out / 'evaluation_per_seed.csv', index=False)
    summary = frame.groupby(['population', 'arm']).agg(n=('seed', 'count'), variants=('variants', 'first'), proteins=('proteins', 'first'), positive=('positive', 'first'), auc_mean=('auc', 'mean'), auc_sample_sd=('auc', 'std'), auprc_mean=('auprc', 'mean'), auprc_sample_sd=('auprc', 'std')).reset_index()
    summary.to_csv(out / 'evaluation_summary.csv', index=False)
    pd.DataFrame(ensemble_rows).to_csv(out / 'ensemble_metrics.csv', index=False)
    pd.DataFrame(seed_deltas).to_csv(out / 'paired_seed_deltas.csv', index=False)
    dump(out / 'cluster_bootstrap.json', bootstrap)
    input_paths.update([root / 'cohort_recon/cohort_protocol.json', root / 'cohort_recon/cohort_manifest.json', root / 'code/protocol.json'])
    dump(out / 'analysis_input_hashes.json', [{'path': str(p), 'sha256': sha256(p)} for p in sorted(input_paths)])
    dump(out / 'analysis_complete.json', {
        'complete': True, 'new_fits': 9, 'evaluated_population_arm_seed_rows': len(frame),
        'external_fusion_performed': False, 'bootstrap_repeats': 2000, 'seed': 20261024,
        'sd_convention': 'sample SD, ddof=1 across three classifier seeds',
        'date_limitation': 'Original retained ClinVar file contains 2026 evaluations; a 31March2024 content-release assertion is unsupported. Identify frozen input by checksum and archive provenance. No manuscript modified by this program.',
        'finished_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    })


if __name__ == '__main__':
    main()
