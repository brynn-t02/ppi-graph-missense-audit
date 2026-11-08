# Remaining matched experiments

All computation runs on the new dedicated server. Existing inputs are uploaded
according to `input_manifest.json`; its `path` is relative to the experiment root.
The `source` field is for the local upload tool only. No existing experiment is
overwritten.

Set `PPI_UNIFICATION_ROOT` to the dedicated root (default
`/root/autodl-tmp/ppi_unification_20261108`), select the one assigned GPU with
`CUDA_VISIBLE_DEVICES=0`, and run `code/execute_pipeline.py` with the isolated
Python interpreter specified by `protocol.json`.

The protocol comprises 39 fits: six completed endpoint executions, six homology
representation fits, and 27 interior partial-ablation fits. V2 resumes the first
six fits and performs only the remaining 33. The three frozen-full fits reproduced
their reference metrics. Strict reproduction of the old regenerated ablated
features failed; v1 and the failure remain archived. V2 locks the FIRST new ablated
tensor and FIRST three fits by hash, after independent verification of the common
propagation operator and masks. Diagnostic regenerations are not selected or used
for training. The same endpoints are shared across the three plotted strategies;
these are not nine independent endpoint fits. No choice is made based on test
performance. `amendment_manifest.json` records this protocol amendment.

Training probabilities are retained unchanged. Reporting uses float32 CPU
node-wise inference in batches of 1,024 nodes, then maps one probability per
protein to variants. Each seed array is converted to float64 before averaging.
Individual-seed metrics supply means and sample SD (ddof=1); ensemble metrics
and paired protein-bootstrap differences are reported separately.

`pipeline_status_v2.json`, `progress.json`, and any `failure_v2.json` are written to
`outputs/`. A failed provenance or amended numerical gate stops the pipeline. Do not
relax a gate without investigating and recording the cause. Complete runs can
resume with unchanged code; a partial run directory is never overwritten.

Download compact JSON/CSV summaries and `outputs/public/*.csv.gz` only.
`outputs/runs/`, `derived_embeddings/`, and `graph_masks/` contain server-only
checkpoints, tensors, and exact graph interventions. Preserve them on the server.
