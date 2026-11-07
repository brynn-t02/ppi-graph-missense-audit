#!/usr/bin/env python3
"""Figure 4 only: deterministic postprocessing of archived seeds 42--44.

No training, inference, feature extraction, graph propagation, bootstrap,
significance testing, Figure 3 strata, or 53-cell analysis is performed.
All writes are restricted to this script's directory.
"""
from pathlib import Path
import gzip, hashlib, io, json, platform
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[2]
BASE = ROOT / 'outputs/BIBM_workshop_font_clean_20261025/source'
CAN = BASE / 'canonical_analysis_20261025'
PUB = CAN / 'public'
MATCH = BASE / 'matched_validation_20261024'
OLD = BASE / 'figure_inputs/data/processed'
LOCAL = Path('/Users/teyian/Documents/paper1/server_snapshot_full/root/data/processed')
DATA = OUT / 'data/processed'
SEEDS = [42, 43, 44]
FEATURES = ['degree_bucket', 'kcore_bucket', 'neighbor_risk_bucket',
            'neighbor_pathogenic_rate', 'neighbor_train_variant_count',
            'betweenness_bucket', 'betweenness', 'clustering_bucket',
            'clustering_coefficient', 'boundary_role_bucket',
            'within_comm_neighbor_fraction']
ARMS = {'raw_ESM': ('original', 'raw_ESM'),
        'full_FPP': ('original', 'full_FPP'),
        'scores': ('original_fusion', 'scores_capacity_matched'),
        'fusion': ('original_fusion', 'scores_plus_full_FPP')}
PAIRS = [('esm_vs_gcn', 'raw_ESM', 'full_FPP', 'esm_to_gcn'),
         ('cadd_am_vs_full', 'scores', 'fusion', 'cadd_am_to_full')]
INPUTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def register(path):
    path = Path(path)
    INPUTS[str(path)] = {'sha256': sha(path), 'bytes': path.stat().st_size}
    return path


def read(path, **kwargs):
    return pd.read_csv(register(path), float_precision='round_trip', **kwargs)


def read_json(path):
    return json.loads(register(path).read_text())


def dump(path, value):
    path = Path(path)
    assert path.is_relative_to(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def save(frame, path):
    path = Path(path)
    assert path.is_relative_to(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format='%.17g')
    assert path.stat().st_size < 5_000_000


def compact(frame, path):
    path = Path(path)
    assert path.is_relative_to(OUT)
    with path.open('wb') as handle:
        with gzip.GzipFile(fileobj=handle, filename='', mode='wb', mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
                frame.to_csv(text, index=False, float_format='%.17g')
    assert path.stat().st_size < 5_000_000


def metrics(labels, probabilities):
    """Tie-aware rank ROC AUC; noninterpolated AP at distinct scores."""
    y, p = np.asarray(labels), np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(p, kind='stable')
    y, p = y[order], p[order]
    ends = np.flatnonzero(np.r_[p[1:] != p[:-1], True])
    tp = np.cumsum(y)[ends].astype(np.float64)
    fp = (ends + 1) - tp
    pos, neg = tp[-1], fp[-1]
    auc = 1 - np.trapezoid(np.r_[0, tp / pos], np.r_[0, fp / neg])
    positive_by_score = np.diff(np.r_[0, tp])[::-1]
    negative_by_score = np.diff(np.r_[0, fp])[::-1]
    ct, cf = np.cumsum(positive_by_score), np.cumsum(negative_by_score)
    ap = np.sum(positive_by_score / pos * ct / (ct + cf))
    return float(auc), float(ap)


def brute_metrics(labels, probabilities):
    y, p = np.asarray(labels), np.asarray(probabilities)
    pos, neg = p[y == 1], p[y == 0]
    auc = np.mean((pos[:, None] > neg[None, :]) + .5 * (pos[:, None] == neg[None, :]))
    ap, previous_recall = 0., 0.
    for score in np.unique(p)[::-1]:
        active = p >= score
        tp = y[active].sum()
        recall = tp / y.sum()
        ap += (recall - previous_recall) * tp / active.sum()
        previous_recall = recall
    return np.array([auc, ap])


def masks(labels, baseline, proposed, threshold):
    y = np.asarray(labels)
    a, b = baseline >= threshold, proposed >= threshold
    ac, bc = a == y, b == y
    groups = {'both_correct': ac & bc, 'both_wrong': ~ac & ~bc,
              'improved': ~ac & bc, 'degraded': ac & ~bc,
              'FN_rescued': (y == 1) & ~a & b,
              'FP_suppressed': (y == 0) & a & ~b,
              'TN_to_FP': (y == 0) & ~a & b,
              'TP_to_FN': (y == 1) & a & ~b,
              'persistent_FN': (y == 1) & ~a & ~b,
              'persistent_FP': (y == 0) & a & b}
    partition = np.stack([groups[k] for k in ['both_correct', 'both_wrong', 'improved', 'degraded']])
    assert np.all(partition.sum(axis=0) == 1)
    assert np.array_equal(groups['improved'], groups['FN_rescued'] | groups['FP_suppressed'])
    assert np.array_equal(groups['degraded'], groups['TN_to_FP'] | groups['TP_to_FN'])
    assert np.array_equal(~ac, groups['FN_rescued'] | groups['FP_suppressed'] | groups['persistent_FN'] | groups['persistent_FP'])
    assert np.array_equal(~bc, groups['TN_to_FP'] | groups['TP_to_FN'] | groups['persistent_FN'] | groups['persistent_FP'])
    return groups


def contrast(frame, left, right, feature):
    # -1 is a retained historical encoding, not treated as missing. Exclude only
    # the feature's nonfinite observations; do not complete-case across features.
    all_x = frame.loc[left, feature].to_numpy(dtype=np.float64)
    all_y = frame.loc[right, feature].to_numpy(dtype=np.float64)
    x, y = all_x[np.isfinite(all_x)], all_y[np.isfinite(all_y)]
    n, m = len(x), len(y)
    assert n > 1 and m > 1
    pooled = np.sqrt(((n - 1) * np.var(x, ddof=1) + (m - 1) * np.var(y, ddof=1)) / (n + m - 2))
    assert pooled > 0
    difference = float(x.mean() - y.mean())
    correction = 1 - 3 / (4 * (n + m) - 9)
    return {'n_left_group': len(all_x), 'n_right_group': len(all_y),
            'n_left': n, 'n_right': m,
            'missing_left': int(len(all_x) - n), 'missing_right': int(len(all_y) - m),
            'negative_one_left': int((x == -1).sum()), 'negative_one_right': int((y == -1).sum()),
            'mean_left': float(x.mean()), 'mean_right': float(y.mean()),
            'diff_left_minus_right': difference, 'pooled_sample_sd': float(pooled),
            'small_sample_correction': float(correction), 'hedges_g': float(correction * difference / pooled)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Read the original row-order table and topology; attach ID only after exact
    # Name/Label/STRING equality. Frozen manifest then checks ID/label/protein.
    topology = read(LOCAL / 'topology_profile_joint_map_exp33/merged_topology_profile_features.csv')
    test = read(LOCAL / 'clinvar_test_main.csv', usecols=['VariationID', 'Name', 'Label', 'STRING_protein_id'])
    assert len(topology) == len(test) == 18220
    assert test.VariationID.is_unique
    for col in ['Name', 'Label', 'STRING_protein_id']:
        assert np.array_equal(topology[col], test[col]), col
    frozen = read(PUB / 'frozen_samples_177213.csv.gz')
    assert len(frozen) == 177213 and frozen.VariationID.is_unique
    subset = frozen[frozen.original_split == 'test']
    assert len(subset) == len(test) and set(subset.VariationID) == set(test.VariationID)
    frozen_test_order_equal = np.array_equal(subset.VariationID, test.VariationID)
    subset = subset.set_index('VariationID').loc[test.VariationID].reset_index()
    assert np.array_equal(subset.VariationID, test.VariationID)
    assert np.array_equal(subset.label, test.Label)
    assert np.array_equal(subset.STRING_protein_id, test.STRING_protein_id)
    original_rows = read(MATCH / 'original_test_rows.csv')
    for col in ['VariationID', 'Label', 'STRING_protein_id']:
        assert np.array_equal(original_rows[col], test[col])
    df = test.rename(columns={'Label': 'label'}).copy()
    for col in FEATURES + ['community_id']:
        df[col] = topology[col].to_numpy()
        assert df.groupby('STRING_protein_id')[col].nunique(dropna=False).max() == 1
    public_manifest = read_json(CAN / 'public_manifest.json')
    pubhashes = {r['file']: r['sha256'] for r in public_manifest['files']}
    assert sha(PUB / 'frozen_samples_177213.csv.gz') == pubhashes['frozen_samples_177213.csv.gz']
    source_manifest = read_json(CAN / 'source_manifest.json')
    canonical_protocol = read_json(CAN / 'code/protocol.json')
    old_protocol = read_json(ROOT / 'work/experiment_20261024/code/protocol.json')
    assert old_protocol['training']['seeds'] == SEEDS
    verification = read_json(MATCH / 'completed_run_verification.json')
    assert verification['passed']
    inference_audit = read_json(CAN / 'canonical_inference_audit.json')
    ensemble_targets = read(CAN / 'ensemble_metrics.csv')
    seed_targets = read(CAN / 'evaluation_per_seed.csv')
    source_hashes = {r['path']: r['sha256'] for r in source_manifest['inputs']}
    assert sha(LOCAL / 'clinvar_test_main.csv') == source_hashes['assets/data/clinvar_test_main.csv']
    audit, metrics_rows, run_rows = {}, [], []
    for alias, (population, arm) in ARMS.items():
        filename = f'test_predictions__{population}__{arm}.csv.gz'
        predictions = read(PUB / filename)
        assert sha(PUB / filename) == pubhashes[filename]
        for col in ['VariationID', 'label', 'STRING_protein_id']:
            assert np.array_equal(predictions[col], df[col]), (alias, col)
        probability_columns = [f'p_seed_{seed}' for seed in SEEDS]
        matrix = predictions[probability_columns].to_numpy(dtype=np.float64)
        assert np.isfinite(matrix).all() and ((matrix >= 0) & (matrix <= 1)).all()
        ensemble = matrix.mean(axis=1, dtype=np.float64)
        assert np.array_equal(ensemble, predictions.ensemble_float64.to_numpy(dtype=np.float64))
        df[alias] = ensemble
        for j, seed in enumerate(SEEDS):
            df[f'{alias}_seed_{seed}'] = matrix[:, j]
            actual = metrics(df.label, matrix[:, j])
            target = seed_targets[(seed_targets.population == population) & (seed_targets.arm == arm) & (seed_targets.seed == seed)].iloc[0]
            assert np.allclose(actual, [target.auc, target.auprc], atol=1e-12, rtol=0)
            path = MATCH / 'runs' / population / arm / f'seed_{seed}'
            recorded = read_json(path / 'metrics.json')
            history = read_json(path / 'history.json')
            assert recorded['split'] == population and recorded['arm'] == arm and recorded['seed'] == seed
            assert recorded['best_epoch'] == max(history, key=lambda r: r['val_auc'])['epoch']
            assert recorded['epochs_run'] == len(history)
            match = [r for r in verification['checks'] if r['split'] == population and r['arm'] == arm and r['seed'] == seed]
            assert len(match) == 1 and all(v is True for k, v in match[0].items() if k not in ['split', 'arm', 'seed'])
            checkpoint_path = f'assets/outputs/runs/{population}/{arm}/seed_{seed}/model.pt'
            checkpoint_sha = source_manifest['selected_checkpoints'][checkpoint_path]
            run = {'alias': alias, 'population': population, 'arm': arm, 'seed': seed,
                   'selected_epoch': recorded['best_epoch'], 'epochs_run': recorded['epochs_run'],
                   'recorded_checkpoint_path': checkpoint_path, 'recorded_checkpoint_sha256': checkpoint_sha,
                   'historical_test_auc': recorded['test_auc'], 'historical_test_auprc': recorded['test_auprc'],
                   'canonical_test_auc': actual[0], 'canonical_test_auprc': actual[1],
                   'canonical_minus_historical_auc': actual[0] - recorded['test_auc'],
                   'canonical_minus_historical_auprc': actual[1] - recorded['test_auprc'],
                   'selected_epoch_history_verified': True}
            if population == 'original':
                assert predictions.groupby('STRING_protein_id')[f'p_seed_{seed}'].nunique().max() == 1
                node_audit = [r for r in inference_audit['runs'] if r['arm'] == arm and r['seed'] == seed][0]
                assert node_audit['checkpoint_sha256'] == checkpoint_sha
                assert node_audit['parts']['test']['canonical_max_within_protein_spread'] == 0
                run['canonical_max_abs_vs_old_variant_probability'] = node_audit['parts']['test']['max_abs_canonical_vs_saved']
            else:
                assert np.allclose(actual, [recorded['test_auc'], recorded['test_auprc']], atol=1e-12, rtol=0)
            run_rows.append(run)
        actual = metrics(df.label, ensemble)
        target = ensemble_targets[(ensemble_targets.population == population) & (ensemble_targets.arm == arm)].iloc[0]
        assert np.allclose(actual, [target.auc, target.auprc], atol=1e-12, rtol=0)
        if population == 'original':
            assert df.groupby('STRING_protein_id')[alias].nunique().max() == 1
        metrics_rows.append({'alias': alias, 'population': population, 'arm': arm,
                             'auc': actual[0], 'auprc': actual[1], 'canonical_auc': target.auc,
                             'canonical_auprc': target.auprc, 'ensemble_exact_equal': True})
    # Independent algorithms check ties and real score subsets, no resampling CI.
    for alias in ARMS:
        for offset in [0, 7, 29]:
            idx = np.arange(offset, len(df), 83)
            assert np.allclose(metrics(df.label.iloc[idx], df[alias].iloc[idx]),
                               brute_metrics(df.label.iloc[idx], df[alias].iloc[idx]), atol=1e-12, rtol=0)
    for y, p in [([0, 1, 1, 0], [.5, .5, .5, .5]), ([0, 1, 0, 1], [.1, .1, .8, .8])]:
        assert np.allclose(metrics(y, p), brute_metrics(y, p), atol=1e-12, rtol=0)
    compact(df, OUT / 'frozen_18220_predictions_topology.csv.gz')
    save(pd.DataFrame(metrics_rows), OUT / 'ensemble_metrics_alignment.csv')
    save(pd.DataFrame(run_rows), OUT / 'twelve_seed_provenance.csv')
    summaries, profile_rows, counts_rows, fpfn_rows, error_rows, threshold_rows = {}, [], [], [], [], []
    member_frames, missing_rows = [], []
    wide = df[['VariationID', 'label', 'STRING_protein_id']].copy()
    for pair, baseline, proposed, stem in PAIRS:
        groups = masks(df.label, df[baseline].to_numpy(), df[proposed].to_numpy(), .5)
        summary = {'pair_name': pair, 'baseline_arm': ARMS[baseline][1], 'proposed_arm': ARMS[proposed][1],
                   'n': len(df), 'prediction_threshold': .5,
                   **{f'count_{name}': int(mask.sum()) for name, mask in groups.items()}}
        summary['count_corrected'] = summary['count_improved']
        summary['net_corrections'] = summary['count_improved'] - summary['count_degraded']
        summaries[pair] = summary
        for name, mask in groups.items():
            sub = df.loc[mask]
            wide[f'{pair}__{name}'] = mask.astype(np.int8)
            row = {'model_pair': pair, 'group_type': 'pair_status' if name in ['both_correct', 'both_wrong', 'improved', 'degraded'] else 'error_transition',
                   'group_name': name, 'n': len(sub), 'proteins': int(sub.STRING_protein_id.nunique()),
                   'positive': int(sub.label.sum()), 'negative': int(len(sub) - sub.label.sum()), 'pos_rate': float(sub.label.mean())}
            counts_rows.append(row.copy())
            for feature in FEATURES:
                finite = np.isfinite(sub[feature].to_numpy(dtype=np.float64))
                row[f'mean_{feature}'] = float(sub.loc[finite, feature].mean())
                missing_rows.append({'model_pair': pair, 'group_name': name, 'feature': feature,
                                     'group_n': len(sub), 'finite_n': int(finite.sum()),
                                     'missing_n': int((~finite).sum()), 'negative_one_n': int((sub[feature] == -1).sum())})
            profile_rows.append(row)
            members = sub[['VariationID', 'label', 'STRING_protein_id']].copy()
            members.insert(0, 'group_name', name)
            members.insert(0, 'model_pair', pair)
            member_frames.append(members)
        pair_profiles = pd.DataFrame([r for r in profile_rows if r['model_pair'] == pair])
        save(pair_profiles, DATA / f'corrected_sample_profile_exp35/{stem}_corrected_profile_summary.csv')
        for feature in FEATURES:
            r = contrast(df, groups['FP_suppressed'], groups['FN_rescued'], feature)
            fpfn_rows.append({'model_pair': pair, 'feature': feature, **r,
                             'n_fp_suppressed': r['n_left'], 'n_fn_rescued': r['n_right'],
                             'mean_fp_suppressed': r['mean_left'], 'mean_fn_rescued': r['mean_right'],
                             'diff_fp_minus_fn': r['diff_left_minus_right'], 'hedges_g_fp_minus_fn': r['hedges_g']})
        save(pd.DataFrame([r for r in fpfn_rows if r['model_pair'] == pair]), DATA / f'fp_vs_fn_stats_exp36B/{stem}_fp_vs_fn_stats.csv')
        for left, right in [('FN_rescued', 'persistent_FN'), ('FP_suppressed', 'persistent_FP')]:
            for feature in FEATURES:
                r = contrast(df, groups[left], groups[right], feature)
                error_rows.append({'model_pair': pair, 'condition': f'{left}_vs_{right}', 'feature': feature,
                                   **r, 'n_corrected_subset': r['n_left'], 'n_persistent_error': r['n_right'],
                                   'mean_corrected_subset': r['mean_left'], 'mean_persistent_error': r['mean_right'],
                                   'diff_corrected_minus_persistent': r['diff_left_minus_right']})
        for numerator in range(1, 10):
            threshold = numerator / 10
            tg = masks(df.label, df[baseline].to_numpy(), df[proposed].to_numpy(), threshold)
            row = {'model_pair': pair, 'threshold': threshold, 'n': len(df), **{k: int(v.sum()) for k, v in tg.items()}}
            row.update(corrected=row['improved'], net=row['improved'] - row['degraded'],
                       partition_total=row['both_correct'] + row['both_wrong'] + row['improved'] + row['degraded'])
            assert row['partition_total'] == 18220
            threshold_rows.append(row)
    dump(DATA / 'corrected_sample_identification_exp34/summary.json', summaries)
    dump(OUT / 'transition_summary.json', summaries)
    save(pd.DataFrame(counts_rows), OUT / 'table6_all_transition_counts.csv')
    save(pd.DataFrame(profile_rows), OUT / 'table6_all_transition_feature_means.csv')
    save(pd.DataFrame(missing_rows), OUT / 'feature_specific_missing_and_encoding_counts.csv')
    save(pd.DataFrame(fpfn_rows), OUT / 'table6_all_22_FP_FN_contrasts.csv')
    save(pd.DataFrame(error_rows), OUT / 'table6_all_44_error_stratified_contrasts.csv')
    save(pd.DataFrame(error_rows), DATA / 'error_stratified_correction_contrast_exp40/error_stratified_correction_contrast_results.csv')
    save(pd.DataFrame(threshold_rows), OUT / 'table10_all_threshold_transition_counts.csv')
    compact(wide, OUT / 'all_ten_transition_masks_by_variant.csv.gz')
    compact(pd.concat(member_frames, ignore_index=True), OUT / 'all_transition_group_members.csv.gz')
    old_summary = read_json(OLD / 'corrected_sample_identification_exp34/summary.json')
    count_changes, direction_changes = [], []
    for pair, baseline, proposed, stem in PAIRS:
        for key, value in summaries[pair].items():
            if key.startswith('count_') and key in old_summary[pair]:
                count_changes.append({'model_pair': pair, 'count': key, 'before': old_summary[pair][key], 'after': value, 'difference': value - old_summary[pair][key]})
        old_stats = read(OLD / f'fp_vs_fn_stats_exp36B/{stem}_fp_vs_fn_stats.csv')
        for new in [r for r in fpfn_rows if r['model_pair'] == pair]:
            old = old_stats[old_stats.feature == new['feature']].iloc[0]
            direction_changes.append({'model_pair': pair, 'feature': new['feature'],
                                      'before_mean_fp': old.mean_fp_suppressed, 'before_mean_fn': old.mean_fn_rescued,
                                      'before_diff_fp_minus_fn': old.diff_fp_minus_fn,
                                      'after_mean_fp': new['mean_fp_suppressed'], 'after_mean_fn': new['mean_fn_rescued'],
                                      'after_diff_fp_minus_fn': new['diff_fp_minus_fn'],
                                      'sign_changed': bool(np.sign(old.diff_fp_minus_fn) != np.sign(new['diff_fp_minus_fn']))})
    save(pd.DataFrame(count_changes), OUT / 'before_after_transition_counts.csv')
    save(pd.DataFrame(direction_changes), OUT / 'before_after_FP_FN_direction.csv')
    old_error = read(OLD / 'difficulty_matched_correction_contrast_exp40/difficulty_matched_correction_contrast_results.csv')
    current_error = pd.DataFrame(error_rows)
    old_error = old_error.merge(current_error[current_error.model_pair == 'esm_vs_gcn'][['condition', 'feature', 'hedges_g']], on=['condition', 'feature'], suffixes=('_before', '_after'), validate='one_to_one')
    old_error['hedges_g_sign_changed'] = np.sign(old_error.hedges_g_before) != np.sign(old_error.hedges_g_after)
    # Only the before/after descriptive g and means; do not carry old invalid p/FDR.
    save(old_error[['condition', 'feature', 'hedges_g_before', 'hedges_g_after', 'hedges_g_sign_changed']], OUT / 'before_after_error_stratified_hedges_g.csv')
    audit.update(passed=True, n=18220, proteins=int(df.STRING_protein_id.nunique()), positive=int(df.label.sum()),
                 test_ID_Name_label_STRING_alignment=True, frozen_original_test_membership_exact=True,
                 frozen_manifest_test_row_order_already_equal=bool(frozen_test_order_equal),
                 frozen_manifest_aligned_by='unique VariationID lookup in original test order',
                 all_four_public_prediction_manifest_hashes_match=True, all_four_seed_means_exactly_equal_canonical=True,
                 all_twelve_seed_metrics_and_four_ensemble_metrics_match=True, tolerance=1e-12,
                 original_representation_same_protein_identical_for_each_seed_and_ensemble=True,
                 original_twelve_selected_epochs_preserved_and_history_verified=True,
                 metric_kernel_independent_real_subset_checks=12, metric_kernel_tied_checks=2,
                 threshold_rule='positive iff mean probability >= threshold', thresholds=[i / 10 for i in range(1, 10)],
                 transition_groups=10, all_twenty_group_memberships_saved=True,
                 all_eighteen_threshold_partitions_and_error_equalities_verified=True,
                 feature_definition='11 archived topology features unchanged; -1 retained; nonfinite excluded per feature',
                 hedges_g_definition='J*(mean(corrected)-mean(persistent))/pooled sample SD; J=1-3/(4*(n+m)-9); ddof=1',
                 statistical_scope='Descriptive error-stratified contrasts; no independent-group p/FDR or causal/difficulty-matched claim',
                 provenance_limit='Weights and large tensors not reaccessed; recorded checkpoint hashes and existing local run/history/manifests verified; canonical representation probabilities were generated in the prior 2026-10-25 CPU node lookup audit, not inferred in this run',
                 operations={'training': False, 'inference': False, 'ESM_extraction': False, 'graph_propagation': False,
                             'bootstrap': False, 'server_access': False, 'Figure3_recomputation': False},
                 python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                 source_code_sha256=sha(Path(__file__)))
    dump(OUT / 'metrics_alignment_audit.json', audit)
    dump(OUT / 'input_sha256.json', {'inputs': INPUTS, 'script_sha256': sha(Path(__file__))})
    displayed_f = ['degree_bucket', 'kcore_bucket', 'neighbor_risk_bucket',
                   'neighbor_pathogenic_rate', 'betweenness_bucket',
                   'boundary_role_bucket', 'within_comm_neighbor_fraction']
    direction_frame = pd.DataFrame(direction_changes)
    dump(OUT / 'figure4_direction_and_interpretation.json', {
        'actual_fresh_recomputation': True,
        'FP_vs_FN_direction_changes': direction_frame[direction_frame.sign_changed].to_dict('records'),
        'fusion_FP_vs_FN_direction_changes': int(direction_frame[direction_frame.model_pair == 'cadd_am_vs_full'].sign_changed.sum()),
        'F_original_seven_features_before_after': old_error[(old_error.condition == 'FN_rescued_vs_persistent_FN') & old_error.feature.isin(displayed_f)][['condition', 'feature', 'hedges_g_before', 'hedges_g_after', 'hedges_g_sign_changed']].to_dict('records'),
        'F_original_seven_features_current': current_error[(current_error.model_pair == 'esm_vs_gcn') & (current_error.condition == 'FN_rescued_vs_persistent_FN') & current_error.feature.isin(displayed_f)].to_dict('records'),
        'interpretation': [
            'The two current comparisons use three-seed probability ensembles at the common threshold >=0.5; counts do not describe the historical single-fit comparison.',
            'Raw ESM to Full FPP corrects 960 errors and introduces 993 at 0.5 (net -33); Full FPP fusion corrects 679 and introduces 224 relative to architecture-matched clinical scores (net +455). Ranking improvements are distinct from threshold-specific net corrections.',
            'Within raw-to-FPP corrected subsets, rescued FN have larger boundary-role and within-community-fraction means than suppressed FP; both directions reverse the old single-fit panel. The other nine of eleven directions remain unchanged.',
            'The F contrast conditions on baseline FN: rescued versus persistent. It is not the C/D comparison between rescued FN and suppressed FP, and does not match difficulty or establish a topology mechanism.',
            'For the original seven F display features, degree remains positive but smaller; k-core flips from negative to near-zero positive; neighbor risk flips to near-zero negative; boundary role and within-community fraction flip to positive. Neighbor pathogenic rate and betweenness bucket remain positive but much closer to zero.',
            'The FN-rescued group has 366 variants versus 1089 persistent FN; within-community fraction excludes 2 missing persistent values featurewise (1087), retaining all historical -1 codes.',
            'These descriptive variant-level effects reuse protein-assigned topology and dependent variants; no p-values, FDR, or new uncertainty estimates were computed.'
        ],
        'ensemble_metric_comparison': metrics_rows
    })
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print('direction changes:', pd.DataFrame(direction_changes).query('sign_changed')[['model_pair', 'feature']].to_dict('records'))
    files = [p for p in OUT.rglob('*') if p.is_file()]
    assert all(p.stat().st_size < 5_000_000 for p in files)
    dump(OUT / 'output_manifest.json', {'files': [{'path': str(p.relative_to(OUT)), 'bytes': p.stat().st_size, 'sha256': sha(p)} for p in sorted(files) if p.name != 'output_manifest.json']})


if __name__ == '__main__':
    main()
