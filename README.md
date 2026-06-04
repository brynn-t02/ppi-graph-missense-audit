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

## Reproducibility boundary

The original experiment scripts were AI-assisted. They are archived for transparency, but should not be described as automatically bug-free.

Current audit position:

- Core scripts inspected so far load retained real input files, labels, predictions, graph objects or compact source-data summaries.
- Random-feature, label-shuffle, graph-rewiring, edge-ablation and topology-permutation scripts are controlled perturbation experiments. They should be described as negative/control analyses, not as biological data.
- Historical table/figure assembly scripts with manually consolidated values are included for provenance only and should not be treated as primary computational evidence.
- The manuscript should rely on compact source data, accepted figure scripts/assets and retained summary outputs for reported claims.

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

