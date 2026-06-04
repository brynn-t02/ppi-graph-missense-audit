# Source Data Package v1

This folder contains a compact source-data package for the current manuscript draft:

- Manuscript: `manuscript_english_working_draft_v18_submission_reading.md`
- Figures: accepted Figure 1-4 assets listed in `../figure_drafts/`
- Supplement: `../Supplementary_Information_v1.md`

The package is intentionally compact. It includes the CSV/JSON files needed to verify the plotted values, supplementary summary tables and key uncertainty analyses. Large per-variant prediction tables, model checkpoints, embedding matrices and raw external resources are not duplicated here; they remain in the local/server snapshot or should be deposited separately if required by the target venue.

## Figure 1

- `figure1/fig1b_full_graph_ppi_gcn_vs_esm_summary.json`: raw ESM, full PPI-GCN, GraphSAGE-style and GAT-style performance summary.
- `figure1/fig1c_variant_level_bootstrap_summary.csv`: variant-level bootstrap summaries for key original-split comparisons.
- `figure1/fig1c_protein_clustered_bootstrap_summary.json`: protein-clustered bootstrap for PPI-GCN versus raw ESM.
- `figure1/fig1d_direct_train_test_edge_ablation_summary.json`: complete direct train-test PPI edge ablation summary.
- `figure1/fig1e_partial_train_test_edge_ablation_curve.csv`: partial train-test edge ablation curve.
- `figure1/fig1f_topology_permutation_summary.json`: topology-permutation control summary.

## Figure 2

- `figure2/fig2a_cadd_match_report.json`: CADD matching report.
- `figure2/fig2a_alphamissense_match_report.json`: AlphaMissense matching report.
- `figure2/fig2b_original_split_fusion_performance.csv`: original-split CADD, AlphaMissense and graph-fusion performance.
- `figure2/fig2c_original_split_bootstrap_gain_summary.csv`: original-split bootstrap gain summary.
- `figure2/fig2d_homology_aware_esm_vs_ppigcn_summary.json`: homology-aware ESM versus PPI-GCN summary.
- `figure2/fig2e_homology_aware_bootstrap_gain_summary.csv`: homology-aware bootstrap gain summary.
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

## Supplementary Tables

- `supplementary_tables/supp_table_architecture_and_controls_stage_summary.csv`: architecture and negative-control summary table.
- `supplementary_tables/supp_table_core_repeated_benchmarks.csv`: repeated benchmark summary.
- `supplementary_tables/supp_table_homology_split_summary.json`: homology-aware split summary.
- `supplementary_tables/supp_table_feature_shuffled_gcn_rerun_metrics.json`: rerun feature-shuffled PPI-GCN negative-control metrics.

## Provenance

- `provenance/submission_housekeeping_sections_snapshot.md`: current data/code availability and submission-housekeeping draft.
- `provenance/figure_asset_manifest_snapshot.md`: current accepted figure asset manifest.
- `SHA256SUMS.txt`: SHA-256 checksums for package files, excluding `SHA256SUMS.txt` itself.

## Important Cautions

- This package does not contain full raw ClinVar, CADD, AlphaMissense or STRING files.
- This package does not contain model checkpoints or full embedding matrices.
- Some CSV/JSON files are summary-level source data. If a target venue requires full per-sample predictions, those should be exported from the retained local/server snapshot rather than inferred from this package.
- The homology-aware split should be interpreted as a stricter stress test, not as a prevalence-matched replacement for the original split.
- AUPRC values should be compared within the same split unless the text explicitly discusses different test-set prevalence.
