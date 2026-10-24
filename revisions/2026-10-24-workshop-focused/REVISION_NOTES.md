# Revision conclusions — 24 October 2026

Workshop comments are the current revision target; main-conference-only requests are set aside. This record distinguishes completed changes from unresolved validation; no new experiments were started during publication.

## Completed changes

1. Restored the accepted original title. The abstract and conclusion explicitly limit the study to fixed, non-trainable propagation and downstream trained classifiers; no claim about the ceiling of trained GNNs is made.
2. Kept the one-ClinVar-cohort / one-STRING-association-network scope explicit. STRING is not described as a purely physical-interaction network.
3. Described topology profiles as exploratory, with many inspected cells and no selection-adjusted confidence guarantee. Historical subgroups were not reselected on the new fits.
4. Added run-provenance mapping and completed matched classifier seeds, training budgets, selection rules and graph-operator controls. Corrected the historical self-loop mismatch for new controls, while preserving old evidence.
5. Replaced the asymmetric homology fusion comparison with three runs for each score/raw/FPP arm and protein-clustered uncertainty. Discussed possible overfitting/redundancy as hypotheses rather than established causes.
6. Further shortened the abstract, methods, numerical listings and discussion; retained the simple overview and practical recommendations. The former main Figure 4 and historical transition detail moved to Supplementary Figure S1. Current and historical panels/tables remain distinguished. Both main copies are 7 pages including references, with no font-size reduction.
7. Published the focused manuscript together with the unchanged environment specifications, experiment code and numerical evidence. Private review correspondence is not published.

## What the new results support

- On the original split, full FPP versus raw ESM has an ensemble AUC difference of +0.0107, with a protein-clustered 95% interval [-0.0218, +0.0426]. This does not establish a reliable protein-level gain.
- Original-split FPP fusion versus the architecture-matched CADD+AlphaMissense baseline has AUC difference +0.0147 [0.0066, 0.0231]. Versus raw-ESM fusion the difference is +0.0032 [0.0008, 0.0061].
- FPP fusion versus edge-ablated fusion has AUC difference +0.0013 [-0.0004, 0.0031]; that residual is uncertain. The sampled topology-permutation comparison has a small positive conditional interval and does not establish broad biological causality.
- In the homology split, FPP fusion versus scores has AUC difference +0.0001 [-0.0019, 0.0021] and AUPRC difference -0.0012 [-0.0199, 0.0154]. Neither stable benefit nor consistent harm is established. Mean run metrics and ensemble metrics are distinct aggregations.

Intervals condition on the fitted models and sampled graph permutations. They do not incorporate new-cohort uncertainty, repeated retraining uncertainty or exploratory-comparison selection. All newly reported values come from actual fitted predictions.

## Unresolved validation

An independent test cohort or a second network is an outstanding workshop-suggested strengthening experiment. Neither has been performed. Stating the limitation is a textual response, not completion of the proposed validation. The workshop wording suggests either type of additional evidence; it does not specify both as compulsory. More stringent temporal/predictor-development-disjoint evaluation and physical-only/alternative-network controls are more specific earlier-review directions.

A trained GNN, independently confirmed topology subgroups, calibration, review-status retraining and graded functional outcomes also remain outside the current evidence base. The current study's claims are restricted accordingly.
