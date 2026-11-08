# Completed remaining unified experiments — 8 November 2026

39 actual classifier fits: 27 interior partial-ablation fits, 6 original full/fully-ablated endpoint executions, and 6 homology representation fits. Same fixed full-graph propagation operator and classifier seeds 42–44, maximum 50 epochs, patience 5, Adam 1e-3/weight decay 1e-5, batch 1024; hidden layers 512/128 and dropout 0.3. Existing ESM and homology assignments were reused. No clinical-score fusion or external validation was retrained.

`results/outputs/public/` contains 13 aligned per-seed test prediction tables. `canonical_per_run_metrics.csv`, `canonical_summary_metrics.csv` and `ensemble_metrics.csv` distinguish means/sample SD from metrics computed after float64 probability averaging. Both test sets contain 18,220 variants; original has 1,413 proteins/5,822 positives and homology has 3,438 proteins/1,924 positives. Two paired protein-bootstrap comparisons use 2,000 draws each. Exact draw differences and sample memberships are included. Curve error bars are sample SD of model AUC, not paired differences; the raw baseline is subtracted as a constant. The three curves share endpoint fits and use one fixed graph for each interior strategy/fraction, not graph-replicated uncertainty.

## Protocol amendment and limitations

`code_v1/`, its protocol freeze and `results/audit_before_amendment_v2/` preserve the original six-endpoint gate and failure. All three frozen-full positive controls exactly reproduced historical training records. The newly regenerated fully ablated tensor did not reproduce the old three ablated fits. Same-GPU repeat propagation differed at floating-point precision, while edge masks, the exact propagation/training implementations and independent CPU float64 operator checks passed. This does not prove the entire historical metric discrepancy was caused by one identified source.

`code_v2/amendment_manifest.json` froze the FIRST generated tensor and FIRST six fitting executions before further runs. No outcome-based tensor, seed or checkpoint selection occurred. V2 resumed those six and fitted the remaining 33; diagnostic tensors were never used to train. The new ablated endpoint and its bootstrap also replace Figure 1D, so Figure 1D and Figure 1E use the same endpoint. Historical files remain archived. Existing edge-ablated fusion results are a separate unchanged fit family.

## Independent verification

From any working directory, with NumPy and pandas:

    python /path/to/this-folder/review/verify_compact_portable.py /path/to/this-folder/results/outputs

The compact verifier checks all39 seed metrics,13 ensembles,15 curve points,39 training histories and4,000 paired bootstrap draws against two frozen reference files. It performs no training. Its output defaults to `review/portable_verification.json`. The independent review and CPU checkpoint/hash closure are in `review/`. All39 checkpoints loaded successfully and all parameter tensors were finite. The executed original verification and manuscript patch scripts retain their original workspace paths for provenance.

## Re-execution and server-only assets

Executed code and protocol are in `code_v2/`; use `PPI_UNIFICATION_ROOT` to select the experiment root, populated according to `input_manifest.json`. Run `code_v2/execute_pipeline.py` with the recorded isolated interpreter and GPU selection. V2 is the actual continuation of v1 and expects its first six frozen run assets and diagnostic gate; it is intentionally not a substitute clean-start benchmark recipe. For a full reconstruction, preserve/run v1 first and investigate any provenance gate failure rather than bypassing it. Exact original input, run and derived-feature hashes are archived.

Large tensors, graph interventions and39 model checkpoints remain on the dedicated experiment server under `/root/autodl-tmp/ppi_unification_20261108/`; they are not included in this public compact archive. The code does not establish public availability of all third-party source data. Download source resources from their original providers or restore the verified research backup. No SSH credentials or connection wrappers are included. The full 141-pin restored environment and runtime checks are recorded in the compact outputs.
