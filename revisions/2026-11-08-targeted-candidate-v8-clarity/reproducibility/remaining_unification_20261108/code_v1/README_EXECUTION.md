# Remaining matched experiments

All computation runs on the new dedicated server. Existing inputs are uploaded
according to `input_manifest.json`; its `path` is relative to the experiment root.
The `source` field is for the local upload tool only. No existing experiment is
overwritten.

Set `PPI_UNIFICATION_ROOT` to the dedicated root (default
`/root/autodl-tmp/ppi_unification_20261108`), select the one assigned GPU with
`CUDA_VISIBLE_DEVICES=0`, and run `code/execute_pipeline.py` with the isolated
Python interpreter specified by `protocol.json`.

The pipeline performs 39 fits: six endpoint replications, six homology
representation fits, and 27 interior partial-ablation fits. The endpoint fits
must reproduce the existing matched results before the remaining fits run.
The same endpoints are shared across the three plotted strategies; these are
not nine independent endpoint fits. No choice is made based on test performance.

Training probabilities are retained unchanged. Reporting uses float32 CPU
node-wise inference in batches of 1,024 nodes, then maps one probability per
protein to variants. Each seed array is converted to float64 before averaging.
Individual-seed metrics supply means and sample SD (ddof=1); ensemble metrics
and paired protein-bootstrap differences are reported separately.

`pipeline_status.json`, `progress.json`, and any `failure.json` are written to
`outputs/`. A failed provenance or replication gate stops the pipeline. Do not
relax a gate without investigating and recording the cause. Complete runs can
resume with unchanged code; a partial run directory is never overwritten.

Download compact JSON/CSV summaries and `outputs/public/*.csv.gz` only.
`outputs/runs/`, `derived_embeddings/`, and `graph_masks/` contain server-only
checkpoints, tensors, and exact graph interventions. Preserve them on the server.
