# Fixed STRING protein-feature propagation audit

Code and compact evidence archive for an evaluation study of fixed protein-feature propagation in missense variant pathogenicity prediction. The fixed graph operator has no learned propagation parameters; downstream MLP classifiers are trained. Historical PPI-GCN names are retained in source files for traceability; revised manuscripts use FPP (fixed protein-feature propagation).

## Manuscript revisions

- [8 October 2026 main-review revision](revisions/2026-10-08/) — [paper](revisions/2026-10-08/paper.pdf), [supplement](revisions/2026-10-08/supplement.pdf), [editable source](revisions/2026-10-08/source/BIBM2026_main_paper_revised.tex).

Each dated directory contains the delivered anonymous PDFs, editable source, figures, generators, compact numerical inputs and checksums. Dated snapshots preserve their original text, including availability statements written before this public synchronization. The owner made this repository public, and the October revisions were synchronized on **24 October 2026**. The repository does not yet have a separate DOI deposit.

## What was changed

The October 8 revision clarifies fixed-operator scope, transductive feature access, uneven seed/epoch protocols, external-predictor development overlap and exploratory subgroup selection. It supplies complete-case and threshold-sensitivity checks recomputed from frozen predictions.

## Reproduce the frozen-output checks

The checks require Python, NumPy and pandas. They verify existing threshold counts and complete-case AUC/AP values; they do not train models.

```sh
python revisions/2026-10-08/source/source_data_package_v1/revision_20261008/recompute_checks.py
```

For manuscript and figure generation, see the dated directory README. Source-data checksums inside each `source_data_package_v1` use paths relative to its parent `source/` directory; snapshot checksums use paths relative to the dated directory. Root checksums use repository-relative paths.

## Remaining experimental validation

New experiments are waiting for the full code, data, split objects and environment from the original workstation. No new training or compute-server access was performed during the October revisions or this synchronization.

Priority work is matched classifier seeds and training budgets for representation-level graph controls, followed by repeated homology-aware graph fusion matched to the baselines. Independent or temporal cohorts, alternative/physical-only networks, genuinely trained graph models and independent subgroup confirmation are additional validation options, rather than completed results.

The reported evidence concerns one ClinVar cohort and one STRING association network. Fixed propagation results do not establish a performance ceiling for trained graph models. Post-hoc selected subgroup intervals do not account for selection; zero-crossing intervals do not establish equivalence. The homology graph-fusion result remains a single fitted run.

## Historical code and data

The original archive is retained in `scripts/`, `figures/`, `source_data/` and `manuscript_notes/`. The July-era draft under `manuscript_notes/` is historical; use the dated revision links above for the current text. Historical GraphSAGE-style/GAT-style filenames refer to fixed mean/cosine-weighted operators, not trained GraphSAGE/GAT models. Historical difficulty-matched filenames describe baseline-error stratification, without continuous-score matching.

This is a transparent analysis archive, not a one-command raw-data-to-paper pipeline. Large ClinVar, STRING, CADD and AlphaMissense provider resources, full ESM matrices and model checkpoints are not distributed here. Obtain provider data under the respective terms. The historical scripts may require the original environment and paths; do not run training scripts when only reproducing frozen-output checks.

No additional license or DOI is introduced by this synchronization. Included font license notices are retained. Research use should cite the corresponding data providers and methods as documented in the manuscripts.
