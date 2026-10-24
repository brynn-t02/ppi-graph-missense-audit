# Fixed STRING protein-feature propagation audit

Code and compact evidence archive for an evaluation study of fixed protein-feature propagation in missense variant pathogenicity prediction. The fixed graph operator has no learned propagation parameters; downstream MLP classifiers are trained. Historical PPI-GCN names are retained in source files for traceability; revised manuscripts use FPP (fixed protein-feature propagation).

## Manuscript revisions

- **Latest: [24 October 2026 workshop revision and matched experiments](revisions/2026-10-24/)** — [named paper](revisions/2026-10-24/BIBM2026_workshop_named_revised.pdf), [anonymous paper](revisions/2026-10-24/BIBM2026_workshop_anonymous_revised.pdf), [supplement](revisions/2026-10-24/BIBM2026_supplementary_named_revised.pdf), [revision conclusions](revisions/2026-10-24/REVISION_NOTES.md).
- [23 October 2026 workshop revision](revisions/2026-10-23/) — [paper](revisions/2026-10-23/paper.pdf), [supplement](revisions/2026-10-23/supplement.pdf), [editable source](revisions/2026-10-23/source/main.tex).
- [8 October 2026 main-review revision](revisions/2026-10-08/) — [paper](revisions/2026-10-08/paper.pdf), [supplement](revisions/2026-10-08/supplement.pdf), [editable source](revisions/2026-10-08/source/BIBM2026_main_paper_revised.tex).

Each dated directory contains delivered PDFs, editable source, figures, generators, compact numerical inputs and checksums. The October 24 directory includes both named and anonymous copies and current matched-run code, predictions and verification records. Dated snapshots preserve their original text, including availability statements written before this public synchronization. The owner made this repository public, and the October revisions were synchronized on **24 October 2026**. The repository does not yet have a separate DOI deposit.

## What was changed

The October 8 revision clarifies fixed-operator scope, transductive feature access, uneven seed/epoch protocols, external-predictor development overlap and exploratory subgroup selection. It supplies complete-case and threshold-sensitivity checks recomputed from frozen predictions.

The October 23 revision adds a shorter abstract, a main-text run-provenance table, a simplified overview, cautious subgroup interpretation, possible explanations for lower homology-fusion estimates, practical recommendations and a narrow conclusion.

The October 24 revision restores the accepted original title and confirmed authors (anonymous copies retained), completes 48 matched classifier fits, corrects a historical self-loop convention mismatch in new graph controls, repeats homology fusion three times per arm, and reports current paired protein bootstrap and complete-score subset checks. [Detailed conclusions and unresolved validation](revisions/2026-10-24/REVISION_NOTES.md) distinguish completed experiments from remaining suggestions.

## Reproduce the frozen-output checks

The checks require Python, NumPy and pandas. They verify existing threshold counts and complete-case AUC/AP values; they do not train models.

```sh
python revisions/2026-10-23/source/source_data_package_v1/revision_20261008/recompute_checks.py
```

For manuscript and figure generation, see the dated directory README. Source-data checksums inside each `source_data_package_v1` use paths relative to its parent `source/` directory; snapshot checksums use paths relative to the dated directory. Root checksums use repository-relative paths.

## Experimental status and remaining validation

Priority matched-control and repeated homology-fusion runs are complete, with an isolated restored runtime and 48 independently verified classifier fits. Six positive controls exactly reproduce historical recorded metrics. The current [protocol, code and environment specifications](revisions/2026-10-24/reproducibility/code/) and [predictions and verification records](revisions/2026-10-24/reproducibility/outputs/) are published.

An independent test cohort or a second network is an outstanding workshop-suggested strengthening experiment; neither has been performed. The current text acknowledges these limits, which is not the same as completing the proposed validation. More specific temporal/predictor-development-disjoint evaluation, physical-only or alternative networks, trained GNNs and independent subgroup confirmation remain outside the evidence base.

The reported evidence concerns one ClinVar cohort and one STRING association network. Fixed propagation results do not establish a performance ceiling for trained graph models. Post-hoc selected subgroup intervals do not account for selection; zero-crossing intervals do not establish equivalence. Repeated homology graph fusion establishes neither a stable benefit nor consistent harm versus the score-only baseline.

## Historical code and data

The original archive is retained in `scripts/`, `figures/`, `source_data/` and `manuscript_notes/`. The July-era draft under `manuscript_notes/` is historical; use the dated revision links above for the current text. Historical GraphSAGE-style/GAT-style filenames refer to fixed mean/cosine-weighted operators, not trained GraphSAGE/GAT models. Historical difficulty-matched filenames describe baseline-error stratification, without continuous-score matching.

This is a transparent analysis archive, not a one-command raw-data-to-paper pipeline. Large ClinVar, STRING, CADD and AlphaMissense provider resources, full ESM matrices and model checkpoints are not distributed here. Obtain provider data under the respective terms. The historical scripts may require the original environment and paths; do not run training scripts when only reproducing frozen-output checks.

No additional license or DOI is introduced by this synchronization. Included font license notices are retained. Research use should cite the corresponding data providers and methods as documented in the manuscripts.
