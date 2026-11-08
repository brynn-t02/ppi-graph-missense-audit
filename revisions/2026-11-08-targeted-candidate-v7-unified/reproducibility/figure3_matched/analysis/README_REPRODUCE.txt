Figure 3 matched ensemble and protein-clustered bootstrap source data

recompute_figure3.py reads the original local inputs documented in input_sha256.json; those local paths are not standalone portable paths. freeze_figure3_protocol.py reads only original definitions/features and must precede recompute_figure3.py. No training/inference is performed.

Portable verification: python3 verify_compact_figure3.py (requires numpy and pandas). It uses only supplied compact data, protocol and tables. It checks all points and intervals, plus the first four independently computed protein draws for each group.

frozen_18220_figure3_predictions_memberships.csv.gz contains stable IDs, true labels, proteins, all six actual seed probabilities, two ensembles, and original topology features. all_53_joint_cell_memberships.csv.gz explicitly lists every cell member; fixed cell definitions and membership hashes are in frozen_protocol.json. Bootstrap sample files retain all 2000 draws per cell/stratum, with NaN in undefined single-class draws and resampled variant counts.

Figure 3A is outside this statistical recomputation and remains unchanged. Figure 3B–F display ensemble point values; F displays new protein-clustered intervals. Old variant-resampling intervals and stopped-draft results were not reused.

Metrics are variant-weighted within group. Resampling cluster = STRING protein; proteins and their full variant sets are repeated jointly for both models. Non-interpolated AP and tie-aware ROC AUC use float64 probabilities. Groups may overlap between maps; intervals are not independent tests, are unadjusted, and do not account for original post-hoc subgroup selection.
