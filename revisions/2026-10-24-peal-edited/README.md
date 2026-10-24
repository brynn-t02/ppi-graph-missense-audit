# PEAL editorial revision — 24 October 2026

This version applies 12 targeted prose edits to the IEEE-checked workshop manuscript, using evidence-preserving principles from [academic-humanizer v0.3.3](https://github.com/AIScientists-Dev/academic-humanizer/blob/94b88b23703bed7df507acae7d6d5876209a0cdf/SKILL.md) and the author's Position–Evidence–Analysis–Link guidance.

The edit removes repeated study inventories and explanatory lists, gives Results and Discussion distinct roles, and replaces generic evaluation with specific interpretations. Necessary statistical, design and scope qualifications remain. It does not add experiments or alter the scientific results.

- Named and anonymous main papers: **7 pages each, including references**.
- Named and anonymous supplements: **19 pages each**, byte-identical to the IEEE-checked version.
- [Chinese before/after comparison](PEAL修改前后逐句对照.pdf): **7 pages**, showing all 12 edits with exact prose, reasons and preserved boundaries; a plain-text version is also included.
- `editorial_changes.json`: exact LaTeX before/after fragments and Chinese reasons.
- `source_verification.json`: numeric, citation, equation, caption and asset invariance checks.
- `editing_provenance.json`: fixed skill commit, content hash and version-retention policy.
- `final_quality_checks.json`: pages, embedded fonts, anonymity checks and PDF hashes.
- `source/`: editable manuscripts, unchanged supplement sources, figures and compact numerical inputs.
- `reproducibility/`: build logs, rendering environment and exact main-source differences.

Approximate comparable main-prose count is 2,862 to 2,783 words in the named copy (net 79 fewer); this includes abstract/captions/tables and excludes bibliography, citation keys and inline mathematics. This describes editing scale, not an authorship or detector score.

## Build and evidence

Compile `main.tex` or `main_anonymous.tex` inside `source/` with Tectonic or an IEEEtran-compatible engine. The T1 encoding declaration before IEEEtran preserves real Times-compatible bold and italic fonts. The conference class, margins and type sizes are unchanged. This is not a PDF eXpress certification.

Figure generators and compact inputs remain under `source/figure_scripts/` and `source/figure_inputs/`. Supplements are unchanged and can be generated using the retained `reproducibility/render_supplement.py` with `--root /path/to/this/revision` and, for the anonymous copy, `--anonymous`.

Experimental evidence remains in the [48-fit matched audit](../2026-10-24/) and [external-source and BioGRID record](../../experiments/2026-10-24-external-validation/). The [preceding IEEE-checked version](../2026-10-24-ieee-checked/) is preserved unchanged. No new training, inference or data retrieval was performed for this edit.

## Version retention

All earlier dated manuscript versions and experimental records are retained. No older manuscript is overwritten or deleted. Cleanup will wait for the author's final-version decision and explicit cleanup instruction.

The comparison in this directory contains manuscript prose and editorial rationale only. Private reviewer/author correspondence and machine-access records are excluded. Anonymity applies to the anonymous document content; the public repository also contains authorized named manuscripts. Large experimental tensors, checkpoints and provider data remain outside this compact archive. Checksums use paths relative to this directory.
