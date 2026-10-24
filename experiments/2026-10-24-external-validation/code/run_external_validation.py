"""Bounded second-network fitting and frozen external-source evaluation.

Run only in the restored server runtime, with CUDA_VISIBLE_DEVICES chosen by
the operator. --smoke performs two four-batch smoke cases and input checks;
--run repeats those smoke gates, verifies six old checkpoints, fits/resumes
exactly nine new classifiers, and evaluates three representations externally.
No old fit is restarted, and the old output directory is read-only here.
"""
import argparse
import collections
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time
import traceback

OLD_ROOT = Path('/Data/u40902/TYX/ppi_audit_20261024')
ROOT = OLD_ROOT / 'external_validation_20261024'
OUT = ROOT / 'outputs'
SEEDS = [42, 43, 44]
COHORT_FILES = {
    'all_variant_disjoint': 'external_cohort_all_variant_disjoint.csv',
    'primary': 'external_cohort_primary.csv',
    'strict_unseen_all_original': 'external_cohort_unseen_all_original.csv',
}
EXPECTED_ROWS = {'all_variant_disjoint': 40900, 'primary': 10785,
                 'strict_unseen_all_original': 6934}
JOBS = [('original', 'BioGRID_FPP', False),
        ('original_fusion', 'scores_plus_BioGRID_FPP', True),
        ('homology', 'scores_plus_BioGRID_FPP', True)]


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(name, value):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def progress(state, **kwargs):
    dump('progress.json', {'state': state, 'planned': 9, 'time_utc': utc(), **kwargs})


def checkpoint_path(arm, seed):
    base = OUT if arm == 'BioGRID_FPP' else OLD_ROOT / 'outputs'
    return base / 'runs' / 'original' / arm / f'seed_{seed}' / 'model.pt'


def load_legacy(torch):
    fraction = float(os.environ.get('MAX_GPU_MEMORY_FRACTION', '0.15'))
    assert 0 < fraction <= 0.5, 'Memory fraction outside bounded protocol.'
    assert torch.cuda.is_available()
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    torch.set_num_threads(4)
    module_path = OLD_ROOT / 'code' / 'run_matched_experiments.py'
    spec = importlib.util.spec_from_file_location('frozen_matched_validation', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.B == OLD_ROOT and module.D == OLD_ROOT / 'data'
    module.O = OUT
    assert module.O != OLD_ROOT / 'outputs'
    assert module.SEEDS == SEEDS
    return module, fraction


def freeze_inputs():
    """Save hashes before smoke, new fitting, or external model scoring."""
    selected = [ROOT / 'code' / 'protocol.json', Path(__file__).resolve(),
                ROOT / 'network_recon' / 'biogrid_edge_index.npy',
                ROOT / 'network_recon' / 'frozen_node_order.json',
                ROOT / 'network_recon' / 'network_build_audit.json',
                ROOT / 'cohort_recon' / 'cohort_final_freeze.json',
                ROOT / 'cohort_recon' / 'cohort_protocol.json',
                OLD_ROOT / 'code' / 'run_matched_experiments.py',
                OLD_ROOT / 'code' / 'input_manifest.json']
    selected += [ROOT / 'cohort_recon' / name for name in COHORT_FILES.values()]
    selected += [checkpoint_path(arm, seed) for arm in ['raw_ESM', 'full_FPP'] for seed in SEEDS]
    freeze = json.loads((ROOT / 'cohort_recon' / 'cohort_final_freeze.json').read_text())
    frozen_hashes = {record['name']: record for record in freeze['files']}
    for name in COHORT_FILES.values():
        path = ROOT / 'cohort_recon' / name
        assert path.stat().st_size == frozen_hashes[name]['bytes']
        assert digest(path) == frozen_hashes[name]['sha256'], f'Cohort freeze mismatch: {name}'
    graph_audit = json.loads((ROOT / 'network_recon' / 'network_build_audit.json').read_text())
    edge = ROOT / 'network_recon' / 'biogrid_edge_index.npy'
    assert digest(edge) == graph_audit['files'][edge.name]['sha256']
    assert digest(ROOT / 'network_recon' / 'frozen_node_order.json') == graph_audit['node_order_sha256']
    records = {str(p): {'bytes': p.stat().st_size, 'sha256': digest(p)} for p in selected}
    signature = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    previous_path = OUT / 'pre_inference_manifest.json'
    if previous_path.exists():
        previous = json.loads(previous_path.read_text())
        assert previous['input_signature'] == signature, 'Frozen inputs/code changed. Use a separately versioned output directory.'
        return previous
    receipt = {'created_utc': utc(), 'input_signature': signature, 'files': records,
               'old_checkpoint_count': 6, 'planned_new_fits': 9,
               'selection': 'Protocol and cohort frozen before any new classifier scoring; no test-selected source/filter/seed.'}
    dump('pre_inference_manifest.json', receipt)
    return receipt


def edge_audit(edge, n, np, expected_count):
    assert edge.ndim == 2 and edge.shape == (2, expected_count)
    assert np.issubdtype(edge.dtype, np.integer)
    assert int(edge.min()) >= 0 and int(edge.max()) < n
    assert not np.any(edge[0] == edge[1])
    codes = np.sort(edge[0].astype(np.int64) * n + edge[1])
    assert np.all(codes[1:] != codes[:-1]), 'Duplicate directed source edges.'
    reverse_codes = np.sort(edge[1].astype(np.int64) * n + edge[0])
    assert np.array_equal(codes, reverse_codes), 'Source graph is not bidirectional.'
    return {'nodes': n, 'directed_edges': expected_count,
            'undirected_edges': expected_count // 2, 'bidirectional': True,
            'unique_directed_edges': True, 'self_edges': 0, 'indices_in_bounds': True}


def prepare_network(legacy, a, torch, np):
    actual = {str(key): int(value) for key, value in a['protein_to_idx'].items()}
    n = len(actual)
    assert n == 19622 and sorted(actual.values()) == list(range(n))
    declared_nodes = json.loads((ROOT / 'network_recon' / 'frozen_node_order.json').read_text())
    assert len(declared_nodes) == len(set(declared_nodes)) == n
    assert set(declared_nodes) == set(actual), 'Network/source node universes differ.'
    remapping = np.asarray([actual[node] for node in declared_nodes], dtype=np.int64)
    original_edge = a['edge_index'].long().numpy()
    original_check = edge_audit(original_edge, n, np, 13715404)
    raw_edge = np.load(ROOT / 'network_recon' / 'biogrid_edge_index.npy', allow_pickle=False)
    source_check = edge_audit(raw_edge, n, np, 1862916)
    # Always remap by STRING identifier; identity is an observed special case.
    mapped_edge = remapping[raw_edge]
    mapped_check = edge_audit(mapped_edge, n, np, 1862916)
    edge = torch.from_numpy(mapped_edge.copy()).long()
    degree = np.bincount(mapped_edge[0], minlength=n)
    isolated = np.flatnonzero(degree == 0)
    assert len(isolated) == 1402
    actual_nodes = [None] * n
    for node, index in actual.items():
        actual_nodes[index] = node
    dump('actual_node_order.json', actual_nodes)
    np.save(OUT / 'biogrid_node_degree.npy', degree)
    path = OUT / 'derived_embeddings' / 'BioGRID_FPP.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        z = legacy.emb(path)
    else:
        z = legacy.propagate(a, edge)
        torch.save(z, path)
    assert tuple(z.shape) == (n, 1280) and torch.isfinite(z).all()
    isolate_max = float((z[isolated] - a['x'].float()[isolated]).abs().max())
    assert torch.allclose(z[isolated], a['x'].float()[isolated], rtol=1e-6, atol=1e-7), isolate_max
    # On an identity (isolated) operator the existing FPP averages X three
    # times, so tiny rounding differences are allowed, not altered features.
    sample = torch.arange(0, 1280, 80)
    rederived = legacy.propagate(a, edge, sample)
    max_error = float((rederived - z[:, sample]).abs().max())
    assert torch.allclose(rederived, z[:, sample], rtol=2e-4, atol=2e-5), max_error
    saved_full = legacy.emb(legacy.D / 'gcn_embeddings/gcn_node_embeddings.pt')
    assert saved_full.shape == a['x'].shape and torch.isfinite(saved_full).all()
    audit = {'actual_mapping_validated': True,
             'declared_node_order_equals_actual_model': bool(np.array_equal(remapping, np.arange(n))),
             'nodes_remapped_by_STRING_id': int(np.sum(remapping != np.arange(n))),
             'original_source_edge_check': original_check, 'biogrid_source_edge_check': source_check,
             'biogrid_actual_index_edge_check': mapped_check, 'isolates': len(isolated),
             'isolates_vs_raw_max_abs_error': isolate_max, 'recomputed_16_columns_max_abs_error': max_error,
             'common_operator': 'A=(E+I)+(E+I)^T; bidirectional E; coalesced sums; S=D^-1/2 A D^-1/2; (X+SX+S^2X)/3',
             'embedding_sha256': digest(path), 'embedding_shape': list(z.shape),
             'graph_labels_used': False}
    dump('network_execution_audit.json', audit)
    return actual, {'raw_ESM': a['x'].float(), 'full_FPP': saved_full, 'BioGRID_FPP': z}


def prepare_cohorts(actual, pd, np):
    cohorts = {}
    audits = {}
    for name, filename in COHORT_FILES.items():
        df = pd.read_csv(ROOT / 'cohort_recon' / filename, keep_default_na=False)
        assert len(df) == EXPECTED_ROWS[name]
        assert not df['external_id'].duplicated().any()
        assert not df[['STRING_protein_id', 'protein_variant']].duplicated().any()
        label = pd.to_numeric(df['label'], errors='raise').astype(int)
        assert set(label.unique()) == {0, 1}
        assert ((df['category'] == 'LP/P') == (label == 1)).all()
        assert set(df['category']) == {'LP/P', 'LB/B'}
        assert (df['exclusion_reason'] == '').all()
        for col in ['overlap_uniprot_aa', 'overlap_gene_aa', 'overlap_node_aa', 'overlap_dbSNP']:
            assert (pd.to_numeric(df[col], errors='raise') == 0).all()
        indices = df['STRING_protein_id'].astype(str).map(actual)
        assert indices.notna().all()
        indices = indices.astype(np.int64)
        df['actual_node_index'] = indices
        df['label'] = label
        changed = int(np.sum(indices.to_numpy() != pd.to_numeric(df['node_index']).to_numpy()))
        if name == 'primary':
            assert (pd.to_numeric(df['primary_trainval_unseen']) == 1).all()
            assert set(df['original_protein_stratum']) <= {'test', 'unlabelled'}
        if name == 'strict_unseen_all_original':
            assert (pd.to_numeric(df['unseen_all_original']) == 1).all()
            assert set(df['original_protein_stratum']) == {'unlabelled'}
        output = OUT / 'external_rows' / f'{name}.csv'
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        cohorts[name] = df
        audits[name] = {'rows': len(df), 'positives': int(label.sum()),
                        'proteins': int(df['STRING_protein_id'].nunique()),
                        'declared_index_values_replaced': changed,
                        'mapping_policy': 'All indices derived from actual model protein_to_idx via STRING ID.'}
    all_rows = cohorts['all_variant_disjoint'].set_index('external_id')
    for name in ['primary', 'strict_unseen_all_original']:
        sub = cohorts[name].set_index('external_id')
        assert set(sub.index) <= set(all_rows.index)
        for col in ['STRING_protein_id', 'protein_variant', 'label', 'actual_node_index']:
            assert all_rows.loc[sub.index, col].equals(sub[col])
    assert set(cohorts['strict_unseen_all_original']['external_id']) <= set(cohorts['primary']['external_id'])
    dump('cohort_execution_audit.json', {'passed': True, 'cohorts': audits,
         'subset_metadata_consistent': True, 'existing_source_overlap_checks_retained': True,
         'not_independent_patient_cohort': True})
    return cohorts


def load_model(path, dim, legacy, torch):
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert tuple(state['net.0.weight'].shape) == (512, dim)
    assert tuple(state['net.3.weight'].shape) == (128, 512)
    assert tuple(state['net.6.weight'].shape) == (1, 128)
    assert all(torch.isfinite(value).all() for value in state.values())
    model = legacy.MLP(dim).to(legacy.DEVICE)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def validate_saved_run(q, s, legacy, np):
    result = json.loads((q / 'metrics.json').read_text())
    history = json.loads((q / 'history.json').read_text())
    assert 1 <= len(history) <= 50
    values = np.asarray([entry['val_auc'] for entry in history])
    assert int(result['best_epoch']) == int(np.argmax(values)) + 1
    assert len(history) == int(result['epochs_run'])
    assert abs(values.max() - result['val_auc']) < 1e-12
    for part in ['val', 'test']:
        probs = np.load(q / f'{part}_probs.npy', allow_pickle=False)
        assert len(probs) == len(s['labels'][part])
        assert np.isfinite(probs).all() and np.all((probs >= 0) & (probs <= 1))
        observed = legacy.metrics(s['labels'][part].numpy(), probs)
        for key, value in observed.items():
            assert abs(value - result[f'{part}_{key}']) <= 1e-12
    return result


def verify_old_checkpoints(legacy, s, reps, torch, np):
    checks = []
    for arm in ['raw_ESM', 'full_FPP']:
        get, dim = legacy.features(s, reps[arm], False)
        for seed in SEEDS:
            p = checkpoint_path(arm, seed)
            before_hash = digest(p)
            saved = validate_saved_run(p.parent, s, legacy, np)
            model = load_model(p, dim, legacy, torch)
            parts = {}
            for part in ['val', 'test']:
                observed, probs = legacy.evaluate(model, get, part, s['labels'][part])
                old_probs = np.load(p.parent / f'{part}_probs.npy', allow_pickle=False)
                max_error = float(np.max(np.abs(probs - old_probs)))
                assert max_error <= 2e-6, (arm, seed, part, max_error)
                delta = {key: observed[key] - saved[f'{part}_{key}'] for key in observed}
                assert max(abs(value) for value in delta.values()) <= 1e-8, (arm, seed, part, delta)
                parts[part] = {'metrics': observed, 'metric_difference': delta, 'max_probability_abs_difference': max_error}
            assert digest(p) == before_hash
            checks.append({'arm': arm, 'seed': seed, 'checkpoint_sha256': before_hash,
                           'selected_epoch': saved['best_epoch'], 'parts': parts, 'passed': True})
            del model
        del get
        torch.cuda.empty_cache()
    dump('old_checkpoint_reproduction.json', {'passed': True, 'checkpoints': 6, 'checks': checks,
         'prediction_abs_tolerance': 2e-6, 'metric_abs_tolerance': 1e-8,
         'frozen_checkpoints_refit': False})


def smoke(legacy, splits, representation, torch):
    checks = []
    for split, arm, external in [('original', 'BioGRID_FPP', False),
                                  ('homology', 'scores_plus_BioGRID_FPP', True)]:
        get, dim = legacy.features(splits[split], representation, external)
        checks.append(legacy.fit(split, arm, splits[split], get, dim, 42, smoke=True))
        del get
        torch.cuda.empty_cache()
    dump('runtime_smoke.json', {'passed': True, 'test_set_used': False,
                              'new_fits_completed': 0, 'cases': checks})


def train_nine(legacy, splits, representation, torch, np, pd):
    results = []
    for split, arm, external in JOBS:
        s = splits['original' if split == 'original_fusion' else split]
        get, dim = legacy.features(s, representation, external)
        for seed in SEEDS:
            q = OUT / 'runs' / split / arm / f'seed_{seed}'
            progress('training', completed=len(results), current={'split': split, 'arm': arm, 'seed': seed})
            if q.exists():
                result = validate_saved_run(q, s, legacy, np)
                assert result['split'] == split and result['arm'] == arm and int(result['seed']) == seed
                model = load_model(q / 'model.pt', dim, legacy, torch)
                del model
            else:
                result = legacy.fit(split, arm, s, get, dim, seed, smoke=False)
                validate_saved_run(q, s, legacy, np)
            result['checkpoint_sha256'] = digest(q / 'model.pt')
            (q / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
            results.append(result)
            pd.DataFrame(results).to_csv(OUT / 'per_run_metrics.csv', index=False)
            progress('training', completed=len(results), last=result)
        del get
        torch.cuda.empty_cache()
    assert len(results) == 9
    summary = pd.DataFrame(results).groupby(['split', 'arm']).agg(
        n=('seed', 'count'), auc_mean=('test_auc', 'mean'), auc_sd=('test_auc', 'std'),
        auprc_mean=('test_auprc', 'mean'), auprc_sd=('test_auprc', 'std')).reset_index()
    summary.to_csv(OUT / 'summary_metrics.csv', index=False)
    return results


def external_inference(legacy, reps, cohorts, splits, torch, np, pd):
    results, checkpoint_records, node_lookup_checks = [], [], []
    for arm in ['raw_ESM', 'full_FPP', 'BioGRID_FPP']:
        z = reps[arm].to(legacy.DEVICE)
        for seed in SEEDS:
            p = checkpoint_path(arm, seed)
            checksum = digest(p)
            model = load_model(p, z.shape[1], legacy, torch)
            batches = []
            with torch.no_grad():
                for start in range(0, len(z), 1024):
                    batches.append(torch.sigmoid(model(z[start:start+1024])).cpu().numpy())
            node_probs = np.concatenate(batches)
            assert node_probs.shape == (19622,) and np.isfinite(node_probs).all()
            assert np.all((node_probs >= 0) & (node_probs <= 1))
            target = OUT / 'node_predictions' / arm / f'seed_{seed}.npy'
            target.parent.mkdir(parents=True, exist_ok=True)
            np.save(target, node_probs)
            # The node-wise batching may differ slightly in floating-point
            # arithmetic from repeated-variant batching; bound and report it.
            part_checks = {}
            for part in ['val', 'test']:
                lookup = node_probs[splits['original']['nodes'][part].numpy()]
                saved = np.load(p.parent / f'{part}_probs.npy', allow_pickle=False)
                max_error = float(np.max(np.abs(lookup - saved)))
                assert max_error <= 2e-6, (arm, seed, part, max_error)
                part_checks[part] = {'max_probability_abs_difference': max_error,
                                    'lookup_metrics': legacy.metrics(splits['original']['labels'][part].numpy(), lookup)}
            node_lookup_checks.append({'arm': arm, 'seed': seed, 'parts': part_checks})
            for name, df in cohorts.items():
                probs = node_probs[df['actual_node_index'].to_numpy(dtype=np.int64)]
                values = legacy.metrics(df['label'].to_numpy(dtype=np.int64), probs)
                directory = OUT / 'external_predictions' / name / arm
                directory.mkdir(parents=True, exist_ok=True)
                np.save(directory / f'seed_{seed}_probs.npy', probs)
                results.append({'cohort': name, 'arm': arm, 'seed': seed,
                                'n': len(df), 'positives': int(df['label'].sum()),
                                'proteins': int(df['STRING_protein_id'].nunique()),
                                'checkpoint_sha256': checksum, **values})
            assert digest(p) == checksum
            checkpoint_records.append({'arm': arm, 'seed': seed, 'checkpoint_path': str(p),
                                       'checkpoint_sha256': checksum, 'node_predictions_sha256': digest(target)})
            pd.DataFrame(results).to_csv(OUT / 'external_per_run_metrics.csv', index=False)
            progress('external_inference', completed=9, inferred_models=len(checkpoint_records), planned_external_models=9)
            del model
        del z
        torch.cuda.empty_cache()
    assert len(results) == 27 and len(checkpoint_records) == 9
    summary = pd.DataFrame(results).groupby(['cohort', 'arm']).agg(
        n_seeds=('seed', 'count'), auc_mean=('auc', 'mean'), auc_sd=('auc', 'std'),
        auprc_mean=('auprc', 'mean'), auprc_sd=('auprc', 'std')).reset_index()
    summary.to_csv(OUT / 'external_summary_metrics.csv', index=False)
    dump('external_inference_audit.json', {'passed': True, 'models': checkpoint_records,
         'node_lookup_vs_saved_checks': node_lookup_checks, 'new_external_fits': 0,
         'external_clinical_score_fusion_performed': False,
         'interpretation': 'External-source protein-context transfer; no independent-patient or variant-specific discrimination claim.'})
    return checkpoint_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--smoke', action='store_true', help='Validate inputs/operator and two smoke cases, no test inference or fitted models saved.')
    mode.add_argument('--run', action='store_true', help='Smoke, verify old six checkpoints, fit/resume nine new models, score external frozen cohorts.')
    args = parser.parse_args()
    assert ROOT.resolve() != OLD_ROOT.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / '.runner.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    import numpy as np
    import pandas as pd
    import torch
    legacy, fraction = load_legacy(torch)
    frozen = freeze_inputs()
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    old_runtime = json.loads((OLD_ROOT / 'outputs' / 'runtime_environment.json').read_text())
    assert str(torch.__version__) == old_runtime['torch']
    assert torch.version.cuda == old_runtime['cuda_build']
    assert torch.backends.cuda.matmul.allow_tf32 == old_runtime['matmul_tf32']
    assert torch.backends.cudnn.allow_tf32 == old_runtime['cudnn_tf32']
    dump('runtime_environment.json', {'time_utc': utc(), 'python': sys.version,
         'torch': str(torch.__version__), 'cuda_build': torch.version.cuda,
         'gpu': torch.cuda.get_device_name(0), 'visible_device': os.environ.get('CUDA_VISIBLE_DEVICES'),
         'max_gpu_memory_fraction': fraction, 'memory_free_bytes_at_start': free_bytes,
         'memory_total_bytes': total_bytes, 'torch_threads': torch.get_num_threads(),
         'matmul_tf32': torch.backends.cuda.matmul.allow_tf32, 'cudnn_tf32': torch.backends.cudnn.allow_tf32,
         'legacy_B_unchanged': str(legacy.B), 'legacy_D_unchanged': str(legacy.D), 'new_O': str(legacy.O),
         'input_signature': frozen['input_signature']})
    progress('checking_inputs', completed=0, mode='smoke' if args.smoke else 'run')
    a, splits = legacy.check_inputs()
    actual, reps = prepare_network(legacy, a, torch, np)
    cohorts = prepare_cohorts(actual, pd, np)
    progress('smoke', completed=0)
    smoke(legacy, splits, reps['BioGRID_FPP'], torch)
    if args.smoke:
        progress('smoke_passed', completed=0, test_set_used=False)
        return
    progress('verifying_frozen_checkpoints', completed=0)
    verify_old_checkpoints(legacy, splits['original'], reps, torch, np)
    runs = train_nine(legacy, splits, reps['BioGRID_FPP'], torch, np, pd)
    progress('external_inference', completed=9, inferred_models=0, planned_external_models=9)
    records = external_inference(legacy, reps, cohorts, splits, torch, np, pd)
    # Recheck each original checkpoint after all subsequent operations.
    for path, entry in frozen['files'].items():
        if path.endswith('/model.pt'):
            assert digest(Path(path)) == entry['sha256']
    progress('complete_analysis_pending', completed=9, old_checkpoint_verification_passed=True,
             external_models_inferred=len(records), external_metric_rows=27,
             elapsed_model_seconds=sum(row['seconds'] for row in runs),
             max_gpu_memory_allocated_bytes=torch.cuda.max_memory_allocated(0),
             max_gpu_memory_reserved_bytes=torch.cuda.max_memory_reserved(0))
    print('EXTERNAL_VALIDATION_TRAINING_AND_INFERENCE_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        if OUT.exists():
            dump('failure.json', {'time_utc': utc(), 'traceback': traceback.format_exc()})
        raise
