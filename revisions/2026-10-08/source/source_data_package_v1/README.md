# Revised Source Data Package - 8 October 2026

This revision preserves the historical numerical source files and key names. The original submission archive is unchanged outside this revision directory. Manuscript terminology now uses FPP (fixed protein-feature propagation) for historical PPI-GCN/ppi_gcn entries. Historical GraphSAGE-style and GAT-style entries mean fixed mean and fixed cosine-weighted operators, not trained GNN baselines. Historical difficulty_matched filenames refer to baseline-error stratification, without continuous-score matching.

The four current figure PDFs and their portable generators are in revision_20261008/. The original numerical tables are retained rather than relabelled internally. Existing representation-level negative controls have different seeds/epoch budgets from the main full-graph comparison; see Supplementary Table 8. Joint topology cells are exploratory, with confidence intervals unadjusted for selection. Zero-crossing intervals do not prove equivalence.

New files in revision_20261008/:
- threshold_sweep_reference_predictions.csv: nine fixed thresholds applied to the Figure 4 retained reference pairs.
- complete_case_matched_per_run.csv and complete_case_matched_summary.csv: fixed-prediction evaluation on the original full test set and its complete-case subset. No retraining.
- frozen_reference_predictions.npz and frozen_matched_fusion_predictions.npz: numerical arrays, public ClinVar VariationIDs, labels and coverage mask for reproduction; no raw provider score tables.
- recompute_checks.py: reproduces the two new numerical checks with NumPy and pandas.
- figure_scripts/ and figure_inputs/: portable figure generation inputs. Python, numpy, pandas and matplotlib are required. Run each figure*_revised.py script; generated artwork goes into regenerated_figures/.
- historical_protocol_code/: retained code used to verify descriptions. These historical programs were not executed during this revision; they expect the original data environment and can run training if deliberately invoked.
- figures/: current publication artwork.

New analyses are post-hoc checks of already inspected data. They are not independent validation, calibration, complete-case retraining, matched-control reruns or external-cohort confirmation. The outstanding experimental requests remain outstanding. Raw ClinVar, STRING, CADD and AlphaMissense resources should be obtained from their providers subject to applicable terms.

SHA256SUMS.txt verifies all archive files except the checksum index itself. Historical data keys remain unchanged to preserve traceability.

---

Historical submission inventory (retained for provenance; newer terminology above takes precedence):

# Source Data Package v1

This compact anonymized source-data archive was prepared for the BIBM 2026 review submission.

The archive contains the CSV/JSON source files needed to verify plotted values, supplementary summary tables and key uncertainty analyses. It does not include the main manuscript PDF, supplementary-information PDF, accepted figure artwork, raw third-party resources, model checkpoints or full embedding matrices.

- Package version: `v1`, synchronized on 2026-06-19 after final anonymization and source-data cleanup.
Large per-variant prediction tables and raw external resources are not duplicated here; they should be deposited separately only if required by the target venue and licensing terms.

## Figure 1

- `figure1/fig1b_full_graph_ppi_gcn_vs_esm_summary.json`: raw ESM, full PPI-GCN, GraphSAGE-style and GAT-style performance summary.
- `figure1/fig1c_variant_level_bootstrap_summary.csv`: variant-level bootstrap summaries for key original-split comparisons.
- `figure1/fig1c_protein_clustered_bootstrap_summary.json`: protein-clustered bootstrap for PPI-GCN versus raw ESM.
- `figure1/fig1d_direct_train_test_edge_ablation_summary.json`: complete direct train-test PPI edge ablation summary.
- `figure1/fig1e_partial_train_test_edge_ablation_curve.csv`: partial train-test edge ablation curve.
- `figure1/fig1f_topology_permutation_summary.json`: topology-permutation control summary.

Figure 1 uses the retained original-split reference run for paired graph-ablation and bootstrap analyses. Three repeated classifier seeds showed the same direction but a smaller mean AUC difference; these repeated-run values are reported in the manuscript and supplementary tables.

## Figure 2

- `figure2/fig2a_cadd_match_report.json`: CADD matching report.
- `figure2/fig2a_alphamissense_match_report.json`: AlphaMissense matching report.
- `figure2/fig2b_original_split_fusion_performance.csv`: original-split matched repeated fusion-control performance for architecture-matched CADD+AlphaMissense, raw ESM fusion, edge-ablated PPI-GCN fusion, topology-permuted PPI-GCN fusion and real PPI-GCN fusion.
- `figure2/fig2c_real_ppi_margin_over_controls.csv`: marginal test-set gain of real PPI-GCN fusion over architecture-matched CADD+AlphaMissense and protein-representation controls, with protein-clustered paired-bootstrap intervals.
- `figure2/fig2c_original_split_bootstrap_gain_summary.csv`: earlier original-split bootstrap gain summary retained as supplementary uncertainty source data; it is not the architecture-matched paired-bootstrap source for the updated Figure 2C panel.
- `figure2/fig2d_homology_aware_esm_vs_ppigcn_summary.json`: homology-aware ESM versus PPI-GCN summary.
- `figure2/fig2e_homology_aware_strong_predictor_fusion.csv`: homology-aware architecture-matched CADD+AlphaMissense, raw ESM fusion and graph-fusion performance.
- `figure2/fig2e_homology_aware_bootstrap_gain_summary.csv`: earlier homology-aware reference-run bootstrap gain summary retained as supplementary uncertainty source data; it is not a fully architecture-matched repeated-ensemble bootstrap.
- `figure2/fig2f_test_set_prevalence_by_split.csv`: original and homology-aware test-set positive rates.

## Figure 3

- `figure3/fig3a_esm_ppi_alignment_summary.json`: ESM similarity for PPI-neighbour versus random protein pairs.
- `figure3/fig3a_esm_ppi_community_alignment_summary.json`: within-community versus between-community ESM similarity.
- `figure3/fig3b_degree_kcore_topology_stratified_metrics.csv`: degree and k-core stratified graph gains.
- `figure3/fig3c_betweenness_stratified_metrics.csv`: betweenness-stratified graph gains.
- `figure3/fig3c_clustering_stratified_metrics.csv`: clustering-coefficient-stratified graph gains.
- `figure3/fig3d_neighbour_risk_stratified_metrics.csv`: neighbour-risk-stratified graph gains.
- `figure3/fig3e_joint_topology_gain_maps.csv`: complete joint topology cell map.
- `figure3/fig3f_selected_joint_cells_bootstrap_summary.csv`: selected joint cells with bootstrap intervals.

## Figure 4

- `figure4/fig4ab_prediction_transition_counts_summary.json`: prediction-transition summary for raw ESM to PPI-GCN and CADD+AlphaMissense to graph fusion.
- `figure4/fig4c_raw_esm_correction_profile_summary.csv`: raw ESM correction-profile summary.
- `figure4/fig4d_graph_fusion_correction_profile_summary.csv`: graph-fusion correction-profile summary.
- `figure4/fig4e_raw_esm_fp_vs_fn_topology_stats.csv`: raw ESM false-positive versus false-negative topology statistics.
- `figure4/fig4e_graph_fusion_fp_vs_fn_topology_stats.csv`: graph-fusion false-positive versus false-negative topology statistics.
- `figure4/fig4f_difficulty_matched_correction_contrasts.csv`: difficulty-matched topology contrasts.

Figure 4 is an exploratory retained-reference transition analysis. The CADD+AlphaMissense to graph-fusion transition source was not recomputed from the architecture-matched repeated-run ensembles used in the final Figure 2 fusion-control audit.

## Supplementary Tables

- `supplementary_tables/supp_table_architecture_and_controls_stage_summary.csv`: architecture and negative-control summary table, including status labels that distinguish central evidence from provenance-only control summaries.
- `supplementary_tables/supp_table_core_repeated_benchmarks.csv`: repeated benchmark summary.
- `supplementary_tables/supp_table_homology_split_summary.json`: homology-aware split summary.
- `supplementary_tables/supp_table_feature_shuffled_gcn_rerun_metrics.json`: rerun feature-shuffled PPI-GCN negative-control metrics.

## Follow-up Baseline Audit

- `raw_esm_fusion_baseline/summary_metrics.csv`: three-seed summary for CADD+AlphaMissense and CADD+AlphaMissense+raw ESM on the original and homology-aware splits.
- `raw_esm_fusion_baseline/delta_summary.csv`: matched deltas for CADD+AlphaMissense+raw ESM versus CADD+AlphaMissense.
- `raw_esm_fusion_baseline/per_repeat_metrics.csv`: per-seed metrics for the same follow-up baseline.
- `raw_esm_fusion_baseline/summary.json`: machine-readable audit summary, including split sizes, missingness and feature dimensions.
- `graph_control_fusions/summary_metrics.csv`: matched original-split fusion-control summary for CADD+AlphaMissense, CADD+AlphaMissense+real PPI-GCN, CADD+AlphaMissense+edge-ablated PPI-GCN and CADD+AlphaMissense+topology-permuted PPI-GCN.
- `graph_control_fusions/delta_summary.csv`: matched deltas for the three graph-control fusion models versus CADD+AlphaMissense.
- `graph_control_fusions/per_repeat_metrics.csv`: per-seed and per-permutation metrics for the graph-control fusion audit.
- `graph_control_fusions/summary.json`: machine-readable audit summary for the graph-control fusion experiment.
- `architecture_matched_cadd_am/summary_metrics.csv`: three-seed architecture-matched CADD+AlphaMissense baseline using the same 512/128 MLP capacity as fusion models on the original and homology-aware splits.
- `architecture_matched_cadd_am/per_repeat_metrics.csv`: per-seed metrics for the architecture-matched CADD+AlphaMissense baseline.
- `architecture_matched_cadd_am/summary.json`: machine-readable audit summary for the architecture-matched baseline experiment.
- `fusion_control_margin_bootstrap/fusion_control_margin_bootstrap_summary.csv`: paired variant-level and protein-clustered bootstrap intervals for real PPI-GCN fusion versus raw ESM fusion, edge-ablated fusion, topology-permuted fusion and architecture-matched CADD+AlphaMissense.
- `fusion_control_margin_bootstrap/fusion_control_margin_bootstrap_summary.json`: machine-readable bootstrap summary for the same comparisons.

- `SHA256SUMS.txt`: SHA-256 checksums for package files, excluding `SHA256SUMS.txt` itself.

To verify checksums from the parent directory of this folder, run:

```bash
shasum -a 256 -c source_data_package_v1/SHA256SUMS.txt
```

## Important Cautions

- This package does not contain full raw ClinVar, CADD, AlphaMissense or STRING files.
- This package does not contain model checkpoints or full embedding matrices.
- Some CSV/JSON files are summary-level source data. If a target venue requires full per-sample predictions, those should be exported from the retained analysis environment rather than inferred from this package.
- The homology-aware split should be interpreted as a stricter stress test, not as a prevalence-matched replacement for the original split.
- AUPRC values should be compared within the same split unless the text explicitly discusses different test-set prevalence.
