# Release candidate for PPI graph augmentation manuscript

This folder is a local release candidate for code and compact source data associated with the current manuscript draft.

## What is included

- `scripts/local_early_scripts/`: early data reconstruction, model-input construction, baseline model and negative-control scripts.
- `scripts/server_snapshot_root/`: server-side experiment scripts from CADD/AlphaMissense fusion through homology-aware audits, topology analyses, edge-ablation controls and topology-permutation controls.
- `scripts/figure_generation/`: local figure-generation scripts for accepted figure drafts.
- `scripts/document_build/`: local manuscript Word-building helper.
- `source_data/source_data_package_v1.tar.gz`: compact CSV/JSON source data supporting current main figures and supplementary tables.
- `figures/`: accepted figure PNG/PDF candidates.
- `manuscript_notes/`: working provenance notes, figure manifest and submission housekeeping notes.

## Figure-to-file map

The current manuscript figures are supported by the compact source-data archive and the accepted figure assets.

| Manuscript figure | Accepted asset | Figure script | Source-data folder |
| --- | --- | --- | --- |
| Figure 1 | `figures/Figure1_connectivity_topology_controls_v2_accepted.png/pdf` | `scripts/figure_generation/make_figure1_connectivity_v2_local.py` | `source_data_package_v1/figure1/` |
| Figure 2 | `figures/Figure2_fusion_homology_v1_accepted.png/pdf` | `scripts/figure_generation/make_figure2_fusion_homology_v1.py` | `source_data_package_v1/figure2/` |
| Figure 3 | `figures/Figure3_topology_context_v1_accepted.png/pdf` | `scripts/figure_generation/make_figure3_topology_context_v1.py` | `source_data_package_v1/figure3/` |
| Figure 4 | `figures/Figure4_prediction_transitions_v2_accepted.png/pdf` | `scripts/figure_generation/make_figure4_prediction_transitions_v2.py` | `source_data_package_v1/figure4/` |

The file `source_data/source_data_package_v1.tar.gz` contains the compact CSV/JSON files listed above. It can be unpacked with:

```bash
tar -xzf source_data/source_data_package_v1.tar.gz
```

## Analysis-script map

The scripts are grouped by their role in the project rather than by manuscript figure:

| Role | Main scripts |
| --- | --- |
| ClinVar filtering and split construction | `scripts/local_early_scripts/00*.py`, `01_build_string_mapping.py`, `02_build_gene_level_split.py` |
| ESM/PPI representation construction | `03a_prepare_model_inputs.py`, `03c_encode_gcn_embeddings.py`, `03d_encode_graphsage_embeddings.py`, `03e_encode_gat_embeddings.py` |
| Baseline and representation classifiers | `03b_run_esm_mlp_baseline.py`, `03f_run_variant_mlp_on_embeddings.py`, `04a_run_degree_baseline.py` |
| Negative controls | `04b_encode_gcn_embeddings_feature_shuffled.py`, `04c_run_variant_mlp_on_shuffled_embeddings.py`, `04d_run_label_shuffle_control.py`, `04e_run_label_shuffle_repeats.py`, `39a_random_feature_gcn_label_shuffle.py`, `39b_random_feature_gcn_edge_ablation.py` |
| External predictor matching and fusion | `06c_match_cadd_with_tabix.py`, `07c_match_alphamissense_hg38.py`, `06d_run_cadd_stacking.py`, `07d_run_alphamissense_fusion.py`, `08b_run_gcn_cadd_am_threeway_fusion.py` |
| Homology-aware stress test | `14bB_run_mmseqs_homology_audit.py`, `14c_build_homology_aware_split.py`, `14d_rerun_core_models_on_homology_split.py`, `19_run_single_tool_fusions_on_homology_split.py`, `21_run_homology_bootstrap_significance.py` |
| Topology and correction analyses | `23_esm_ppi_alignment_analysis.py` through `40_difficulty_matched_correction_contrast.py` |
| Connectivity and topology controls | `41_esm_feature_gcn_edge_ablation.py`, `42_clustered_bootstrap_key_claims.py`, `44_esm_gcn_partial_edge_ablation_curve.py`, `45_esm_gcn_topology_permutation_control.py` |

The exact script order reflects the historical experiment log, not a polished software pipeline.

## Reproducibility boundary

The original experiment scripts were AI-assisted. They are archived for transparency, but should not be described as automatically bug-free.

Current audit position:

- Core scripts inspected so far load retained real input files, labels, predictions, graph objects or compact source-data summaries.
- Random-feature, label-shuffle, graph-rewiring, edge-ablation and topology-permutation scripts are controlled perturbation experiments. They should be described as negative/control analyses, not as biological data.
- Historical table/figure assembly scripts with manually consolidated values are included for provenance only and should not be treated as primary computational evidence.
- The manuscript should rely on compact source data, accepted figure scripts/assets and retained summary outputs for reported claims.

This release candidate is therefore best interpreted as a transparent analysis archive and figure/source-data package. It is not yet a one-command end-to-end pipeline from raw third-party downloads to final manuscript figures.

## Minimal local checks

After cloning the repository, a reader can first check the package contents and checksums:

```bash
find . -type f | wc -l
shasum -a 256 -c SHA256SUMS.txt
```

The compact source-data package also contains its own README and checksum file after unpacking.

## What is not included

Large third-party or generated files are intentionally not included here:

- CADD whole-genome SNV files.
- AlphaMissense full genome-wide tables.
- Full ESM embedding matrices.
- Large PyTorch model/checkpoint files.
- Large per-variant prediction tables unless specifically needed later.

These files should be cited by official source and/or described in Methods, not uploaded casually to GitHub.

## Suggested archive route

1. Create a private GitHub repository first.
2. Upload this folder after checking the contents.
3. Make the repository public only after the manuscript text, code boundary and data redistribution boundary are final.
4. Create a stable DOI using Zenodo, OSF or an institutional repository when the submission version is ready.
