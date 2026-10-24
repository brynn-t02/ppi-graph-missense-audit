# Workshop reporting revision — 25 October 2026

- Named and anonymous main papers: **7 pages each, including references**.
- Named and anonymous Supplementary Information: **21 pages each**.
- Main Figures 1–3 use one canonical reporting pipeline; historical descriptive Figures S1–S3 remain clearly separated.
- `REVISION_NOTES.md` describes statistical, figure and wording changes relative to `2026-10-24-layout-refined`.
- `source/canonical_analysis_20261025/` contains current compact metrics, code, complete memberships and per-seed predictions, with checksums, provider attribution and verification records.
- `source/` contains editable manuscripts, figure scripts and compact historical inputs retained for traceability.
- `reproducibility/` contains the supplement renderer, build logs and exact manuscript differences.

No new fitting was performed. All 57 selected checkpoints remain unchanged. The current representation scores come from node-wise CPU evaluation of 21 frozen models; floating-point tie handling and float64 ensemble reporting are documented. Use canonical results for this revision, and treat `matched_validation_20261024/` and `external_validation_20261024/` summaries as dated records.

## Recompute compact metrics

Install Python, NumPy, pandas and scikit-learn, then run:

```sh
python source/canonical_analysis_20261025/code/verify_compact_release.py
```

See `source/canonical_analysis_20261025/PUBLIC_DATA_README.txt` for the exact invocation, file schemas, round-trip precision, asset mapping, source attribution and full-bootstrap reproduction. Compact exported predictions suffice for current full-population test metrics and protein bootstrap. Original score-presence flags are still required for complete-case subset reconstruction. Full inference requires separately retained models/tensors; provider resources are not bundled.

## Manuscript build

Compile `main.tex` or `main_anonymous.tex` inside `source/` using Tectonic or an IEEEtran-compatible engine. The T1 encoding declaration precedes the unchanged IEEEtran class. Font sizes, margins and column widths are unchanged. The named version permits URL breaks at hyphens; the anonymous version withholds the identifying URL.

Figure scripts read current `source/canonical_analysis_20261025` inputs directly. Supplement generation uses Python/ReportLab:

```sh
python reproducibility/render_supplement.py --root /path/to/this/revision
python reproducibility/render_supplement.py --root /path/to/this/revision --anonymous
```

Historical protocols and original outputs remain in their dated snapshots, including the [48-fit matched audit](../2026-10-24/) and [external-source/BioGRID experiment record](../../experiments/2026-10-24-external-validation/). The exact historical ClinVar content release remains unverified; frozen sample IDs, assignments and hashes identify the analyzed population.

Anonymity refers to the document content; this public repository also contains authorized named manuscripts. Large provider resources, feature tensors and model checkpoints remain separately stored. This is not a conference-system submission or PDF eXpress certification. All prior versions are preserved. SHA256SUMS.txt paths are relative to this directory.
