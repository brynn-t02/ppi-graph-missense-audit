"""CPU-only independent gates for nine BioGRID fits and external evaluation.

Run after runner --run and analyze_external_validation.py. No fitting, CUDA use,
network access or manuscript editing occurs. Only a compact JSON audit is saved.
"""
from pathlib import Path
import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from analyze_external_validation import (
    DEFAULT_ROOT, DEFAULT_OLD_ROOT, SEEDS, NEW_ARMS, OLD_ARMS, NODE_ARMS,
    COHORTS, EXTERNAL_ROWS, sha256, dump, probabilities, binary_labels,
    validate_weighted_metrics,
)


def score(y, p):
    return np.array([roc_auc_score(y, p), average_precision_score(y, p)], dtype=float)


def same(a, b, tolerance=1e-12):
    assert np.allclose(a, b, atol=tolerance, rtol=0), (a, b)


def check_state(path, dimension):
    state = torch.load(path, map_location='cpu', weights_only=True)
    expected = {'net.0.weight': (512, dimension), 'net.0.bias': (512,),
                'net.3.weight': (128, 512), 'net.3.bias': (128,),
                'net.6.weight': (1, 128), 'net.6.bias': (1,)}
    assert set(state) == set(expected)
    for key, shape in expected.items():
        assert tuple(state[key].shape) == shape and torch.isfinite(state[key]).all(), (path, key)
    return sha256(path)


def check_run(path, source_rows, expected_identity, dimension):
    recorded = json.loads((path / 'metrics.json').read_text())
    for key, value in expected_identity.items():
        assert recorded[key] == value, (path, key)
    assert recorded['dim'] == dimension
    history = json.loads((path / 'history.json').read_text())
    assert 1 <= len(history) <= 50 and recorded['epochs_run'] == len(history)
    assert [h['epoch'] for h in history] == list(range(1, len(history) + 1))
    values = np.asarray([h['val_auc'] for h in history], dtype=float)
    assert np.isfinite(values).all()
    assert np.isfinite(np.asarray([h['loss'] for h in history], dtype=float)).all()
    assert recorded['best_epoch'] == int(np.argmax(values)) + 1
    same(values.max(), recorded['val_auc'])
    best, bad = -np.inf, 0
    for j, value in enumerate(values):
        if value > best: best, bad = value, 0
        else: bad += 1
        assert bad < 5 or j == len(values) - 1, (path, 'continued beyond patience')
    assert len(values) == 50 or bad == 5, (path, 'stopped before patience or budget')
    for part in ['val', 'test']:
        y = binary_labels(source_rows[part].Label.to_numpy())
        p = probabilities(path / f'{part}_probs.npy', len(y))
        same(score(y, p), [recorded[f'{part}_auc'], recorded[f'{part}_auprc']])
    checkpoint_hash = check_state(path / 'model.pt', dimension)
    return recorded, checkpoint_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=DEFAULT_ROOT)
    parser.add_argument('--old-root', type=Path, default=DEFAULT_OLD_ROOT)
    args = parser.parse_args()
    root, old = args.root, args.old_root
    out = root / 'outputs'
    torch.set_num_threads(4)
    numerical_check = validate_weighted_metrics()
    protocol = json.loads((root / 'code/protocol.json').read_text())
    assert protocol['planned_new_fits'] == 9 and protocol['seeds'] == list(SEEDS)
    freeze = json.loads((out / 'pre_inference_manifest.json').read_text())
    for name, record in freeze['files'].items():
        path = Path(name)
        assert path.stat().st_size == record['bytes'] and sha256(path) == record['sha256'], name
    runs = pd.read_csv(out / 'per_run_metrics.csv')
    expected = {(split, arm, seed) for split, arm in NEW_ARMS.items() for seed in SEEDS}
    assert len(runs) == 9 and set(zip(runs.split, runs.arm, runs.seed)) == expected
    source = {}
    for split in ['original', 'homology']:
        source[split] = {}
        for part in ['val', 'test']:
            name = f'clinvar_{part}_main.csv' if split == 'original' else f'homology_aware_split/clinvar_{part}_homology.csv'
            source[split][part] = pd.read_csv(old / 'data' / name, usecols=['VariationID', 'STRING_protein_id', 'Label'])
    source['original_fusion'] = source['original']
    fitted_checks, input_metrics, seed_predictions = [], {}, {}
    for row in runs.itertuples(index=False):
        key = (row.split, row.arm, int(row.seed))
        path = out / 'runs' / row.split / row.arm / f'seed_{row.seed}'
        dimension = 1280 if row.split == 'original' else 1283
        record, digest = check_run(path, source[row.split], dict(zip(['split', 'arm', 'seed'], key)), dimension)
        for field in ['best_epoch', 'epochs_run', 'dim', 'val_auc', 'val_auprc', 'test_auc', 'test_auprc']:
            same(record[field], getattr(row, field))
        input_metrics[key] = np.array([record['test_auc'], record['test_auprc']])
        seed_predictions[key] = probabilities(path / 'test_probs.npy', len(source[row.split]['test']))
        fitted_checks.append({'split': row.split, 'arm': row.arm, 'seed': int(row.seed), 'selected_epoch': record['best_epoch'], 'epochs_run': record['epochs_run'], 'val_and_test_metrics_reproduced': True, 'strict_validation_selection_and_patience': True, 'checkpoint_finite_expected_dimensions': True, 'checkpoint_sha256': digest})
    for split, arms in OLD_ARMS.items():
        y = binary_labels(source[split]['test'].Label.to_numpy())
        for arm in arms:
            for seed in SEEDS:
                path = old / 'outputs/runs' / split / arm / f'seed_{seed}'
                p = probabilities(path / 'test_probs.npy', len(y))
                record = json.loads((path / 'metrics.json').read_text())
                calculated = score(y, p)
                same(calculated, [record['test_auc'], record['test_auprc']])
                input_metrics[(split, arm, seed)] = calculated
                seed_predictions[(split, arm, seed)] = p
    # Actual serialized tensor mapping is authoritative, never sorted candidate IDs.
    tensor_input = torch.load(old / 'data/model_inputs_main.pt', map_location='cpu', weights_only=False)
    mapping = {str(k): int(v) for k, v in tensor_input['protein_to_idx'].items()}
    assert len(mapping) == 19622 and set(mapping.values()) == set(range(19622))
    for part in ['val', 'test']:
        rows = source['original'][part]
        idx = rows.STRING_protein_id.map(mapping).to_numpy(dtype=np.int64)
        assert np.array_equal(idx, tensor_input[f'{part}_protein_idx'].numpy())
        assert np.array_equal(rows.Label.to_numpy(), tensor_input[f'{part}_labels'].numpy())
    del tensor_input
    reproduction = json.loads((out / 'old_checkpoint_reproduction.json').read_text())
    assert reproduction['passed'] and reproduction['checkpoints'] == 6 and not reproduction['frozen_checkpoints_refit']
    assert reproduction['prediction_abs_tolerance'] <= 2e-6 and reproduction['metric_abs_tolerance'] <= 1e-8
    old_keys = {(arm, seed) for arm in ['raw_ESM', 'full_FPP'] for seed in SEEDS}
    assert {(x['arm'], x['seed']) for x in reproduction['checks']} == old_keys and len(reproduction['checks']) == 6
    for x in reproduction['checks']:
        path = old / 'outputs/runs/original' / x['arm'] / f'seed_{x["seed"]}'
        record, digest = check_run(path, source['original'], {'split': 'original', 'arm': x['arm'], 'seed': x['seed']}, 1280)
        assert x['passed'] and x['checkpoint_sha256'] == digest and x['selected_epoch'] == record['best_epoch']
        for part in ['val', 'test']:
            q = x['parts'][part]
            assert 0 <= q['max_probability_abs_difference'] <= 2e-6
            for metric in ['auc', 'auprc']:
                delta = q['metrics'][metric] - record[f'{part}_{metric}']
                same(delta, q['metric_difference'][metric])
                assert abs(delta) <= 1e-8
    inference = json.loads((out / 'external_inference_audit.json').read_text())
    assert inference['passed'] and inference['new_external_fits'] == 0 and not inference['external_clinical_score_fusion_performed']
    expected_node = {(arm, seed) for arm in NODE_ARMS for seed in SEEDS}
    assert len(inference['models']) == 9 and {(x['arm'], x['seed']) for x in inference['models']} == expected_node
    node_arrays = {}
    node_hashes = {}
    for x in inference['models']:
        arm, seed = x['arm'], int(x['seed'])
        provider = root if arm == 'BioGRID_FPP' else old
        checkpoint = provider / 'outputs/runs/original' / arm / f'seed_{seed}/model.pt'
        assert Path(x['checkpoint_path']).resolve() == checkpoint.resolve()
        assert sha256(checkpoint) == x['checkpoint_sha256']
        node_path = out / 'node_predictions' / arm / f'seed_{seed}.npy'
        assert sha256(node_path) == x['node_predictions_sha256']
        node_arrays[(arm, seed)] = probabilities(node_path, 19622)
        node_hashes[(arm, seed)] = x['checkpoint_sha256']
    assert len(inference['node_lookup_vs_saved_checks']) == 9
    assert {(x['arm'], x['seed']) for x in inference['node_lookup_vs_saved_checks']} == expected_node
    lookup_metric_deltas = []
    for x in inference['node_lookup_vs_saved_checks']:
        arm, seed = x['arm'], int(x['seed'])
        provider = root if arm == 'BioGRID_FPP' else old
        for part in ['val', 'test']:
            rows = source['original'][part]
            lookup = node_arrays[(arm, seed)][rows.STRING_protein_id.map(mapping).to_numpy(dtype=np.int64)]
            saved = probabilities(provider / 'outputs/runs/original' / arm / f'seed_{seed}' / f'{part}_probs.npy', len(rows))
            difference = float(np.max(np.abs(lookup - saved)))
            assert difference <= 2e-6
            same(difference, x['parts'][part]['max_probability_abs_difference'])
            values = score(binary_labels(rows.Label.to_numpy()), lookup)
            same(values, [x['parts'][part]['lookup_metrics']['auc'], x['parts'][part]['lookup_metrics']['auprc']])
            lookup_metric_deltas.append({'arm': arm, 'seed': seed, 'part': part, 'max_probability_abs_difference': difference, 'delta_auc_versus_variant_batching': float(values[0] - score(rows.Label.to_numpy(), saved)[0]), 'delta_auprc_versus_variant_batching': float(values[1] - score(rows.Label.to_numpy(), saved)[1])})
    external_metrics = pd.read_csv(out / 'external_per_run_metrics.csv')
    assert len(external_metrics) == 27
    cohorts, populations = {}, {}
    for population, (filename, count, role) in COHORTS.items():
        original_rows = pd.read_csv(root / 'cohort_recon' / filename)
        rows = pd.read_csv(out / 'external_rows' / EXTERNAL_ROWS[population])
        assert len(rows) == count and len(original_rows) == count and not rows.external_id.duplicated().any()
        for column in original_rows.columns:
            assert np.array_equal(rows[column].fillna('').astype(str), original_rows[column].fillna('').astype(str)), (population, column)
        expected_idx = rows.STRING_protein_id.map(mapping).to_numpy(dtype=np.int64)
        assert np.array_equal(expected_idx, rows.actual_node_index.to_numpy())
        y = binary_labels(rows.label.to_numpy())
        cohort_name = EXTERNAL_ROWS[population][:-4]
        for arm in NODE_ARMS:
            for seed in SEEDS:
                p = node_arrays[(arm, seed)][expected_idx]
                saved = probabilities(out / 'external_predictions' / cohort_name / arm / f'seed_{seed}_probs.npy', count)
                assert np.array_equal(p, saved)
                m = score(y, p)
                rr = external_metrics[(external_metrics.cohort == cohort_name) & (external_metrics.arm == arm) & (external_metrics.seed == seed)]
                assert len(rr) == 1
                row = rr.iloc[0]
                same(m, [row.auc, row.auprc])
                assert row.n == count and row.positives == int(y.sum()) and row.proteins == rows.STRING_protein_id.nunique()
                assert row.checkpoint_sha256 == node_hashes[(arm, seed)]
                input_metrics[(population, arm, seed)] = m
                seed_predictions[(population, arm, seed)] = p
        cohorts[population] = {'variants': count, 'proteins': rows.STRING_protein_id.nunique(), 'positive': int(y.sum()), 'role': role, 'actual_tensor_indices_verified': True}
        populations[population] = rows
    for split in NEW_ARMS:
        populations[split] = source[split]['test']
    # Independent recalculation of all analysis summaries and ensemble point estimates.
    analyzed = pd.read_csv(out / 'evaluation_per_seed.csv')
    assert len(analyzed) == 60 and len(input_metrics) == 60
    assert set(zip(analyzed.population, analyzed.arm, analyzed.seed)) == set(input_metrics)
    for row in analyzed.itertuples(index=False):
        same([row.auc, row.auprc], input_metrics[(row.population, row.arm, row.seed)])
    summary = pd.read_csv(out / 'evaluation_summary.csv')
    assert len(summary) == 20
    ensembles = {}
    ensemble_table = pd.read_csv(out / 'ensemble_metrics.csv')
    assert len(ensemble_table) == 20
    for row in summary.itertuples(index=False):
        vv = np.array([input_metrics[(row.population, row.arm, seed)] for seed in SEEDS])
        assert row.n == 3
        same([row.auc_mean, row.auprc_mean], vv.mean(axis=0))
        same([row.auc_sample_sd, row.auprc_sample_sd], vv.std(axis=0, ddof=1))
        pp = np.mean([seed_predictions[(row.population, row.arm, seed)] for seed in SEEDS], axis=0)
        rows = populations[row.population]
        label_column = 'label' if row.population.startswith('external_') else 'Label'
        m = score(binary_labels(rows[label_column].to_numpy()), pp)
        ensembles[(row.population, row.arm)] = m
        table_row = ensemble_table[(ensemble_table.population == row.population) & (ensemble_table.arm == row.arm)]
        assert len(table_row) == 1
        same(m, [table_row.iloc[0].auc, table_row.iloc[0].auprc])
    bootstrap = json.loads((out / 'cluster_bootstrap.json').read_text())
    assert bootstrap['bootstrap_repeats'] == 2000 and bootstrap['seed'] == 20261024
    assert set(bootstrap['comparisons']) == set(populations)
    comparison_count = 0
    for population, comparisons in bootstrap['comparisons'].items():
        pairs = [(NEW_ARMS[population], b) for b in OLD_ARMS[population]] if population in NEW_ARMS else [('full_FPP', 'raw_ESM'), ('BioGRID_FPP', 'raw_ESM'), ('BioGRID_FPP', 'full_FPP')]
        assert set(comparisons) == {f'{a}_vs_{b}' for a, b in pairs}
        for a, b in pairs:
            q = comparisons[f'{a}_vs_{b}']
            same([q['delta_auc'], q['delta_auprc']], ensembles[(population, a)] - ensembles[(population, b)])
            assert q['finite_replicates'] + q['discarded_one_class_replicates'] == 2000 and q['finite_replicates'] >= 1980
            for key in ['auc_ci95', 'auprc_ci95']:
                ci = np.asarray(q[key]); assert ci.shape == (2,) and np.isfinite(ci).all() and -1 <= ci[0] <= ci[1] <= 1
            comparison_count += 1
    paired = pd.read_csv(out / 'paired_seed_deltas.csv')
    assert len(paired) == comparison_count * 3
    for row in paired.itertuples(index=False):
        a, b = row.comparison.split('_vs_')
        same([row.delta_auc, row.delta_auprc], input_metrics[(row.population, a, row.seed)] - input_metrics[(row.population, b, row.seed)])
    for record in json.loads((out / 'analysis_input_hashes.json').read_text()):
        assert sha256(Path(record['path'])) == record['sha256']
    complete = json.loads((out / 'analysis_complete.json').read_text())
    assert complete['complete'] and complete['new_fits'] == 9 and not complete['external_fusion_performed']
    code_hashes = {str(p.relative_to(root / 'code')): sha256(p) for p in sorted((root / 'code').rglob('*')) if p.is_file() and '__pycache__' not in p.parts}
    report = {
        'passed': True, 'runs_verified': 9, 'checks': fitted_checks,
        'old_checkpoints_reproduction_verified': 6,
        'old_checkpoint_direct_metric_tolerance': 1e-8,
        'node_lookup_probability_tolerance': 2e-6,
        'node_lookup_vs_variant_batching': lookup_metric_deltas,
        'external_node_prediction_arrays_verified': 9,
        'external_population_arm_seed_predictions_verified': 27,
        'external_cohorts': cohorts, 'all_analyzed_seed_metric_rows_verified': 60,
        'ensemble_metrics_verified': 20, 'bootstrap_comparison_point_estimates_verified': comparison_count,
        'bootstrap_ci_checks': 'Bounds/finite count/point estimates checked; no duplicate 2000-replicate recomputation. Weighted tie-aware metric kernel checked against sklearn.',
        'weighted_metric_validation': numerical_check,
        'pre_inference_input_hashes_verified': len(freeze['files']),
        'code_sha256': code_hashes,
        'manuscript_modified': False,
        'checked_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
    }
    dump(out / 'completed_run_verification.json', report)
    print(json.dumps({'passed': True, 'runs_verified': 9, 'old_checkpoints_verified': 6, 'external_predictions_verified': 27}, indent=2))


if __name__ == '__main__':
    main()
