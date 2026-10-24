# Layout refinement and concise methods clarification — 24 October 2026

This separate revision fixes excessive word spacing in the opening of Methods D and restores three short method details already documented in the supplement: MMseqs clustering thresholds, the common loss/optimizer settings, and external-source label rules. No experiment was added or rerun.

Both named and anonymous main papers remain **7 pages including references**; both supplements are **19 pages**, byte-identical to the preceding PEAL revision. Figures, results, discussion, conclusion, abstract, citations and bibliography content are unchanged. The font sizes, margins, column width and IEEEtran class are unchanged. The anonymous last-page reference break was adjusted after text reflow.

- `词距修复与内容取舍说明.txt`: Chinese explanation of spacing, additions and content selection.
- `editorial_changes.json`: exact before/after text for the four edits.
- `source_verification.json` and `final_quality_checks.json`: source invariants, page counts, fonts, anonymity and PDF hashes.
- `Methods_D_词距修复预览.png`: cropped preview of the corrected paragraph.
- `source/`: editable manuscripts, unchanged supplements, figures and compact inputs.
- `reproducibility/`: compilation logs, exact source differences and retained rendering tools.

The three affected line gaps decreased from about 21.65/14.34/13.36 pt to 3.47/2.32/4.92 pt. Long resource names now have an optional line-break point. Normal justified text spacing is retained elsewhere.

## Content and version policy

Seven pages are retained because the main text contains the core methods, evidence and interpretation. Historical subgroup detail, full filtering counts and run logs remain supplementary. The page allowance is not treated as a target for padding.

The [previous PEAL version and its 12-edit comparison](../2026-10-24-peal-edited/) remain unchanged, as do all older versions and experimental records. This version adds only a dated directory and root index/checksums. No historical revision is deleted or overwritten.

## Build

Compile `main.tex` or `main_anonymous.tex` inside `source/` using Tectonic or an IEEEtran-compatible engine. Keep the T1 encoding declaration before IEEEtran. The figure and supplement generators are retained from the preceding revision; render supplements with `reproducibility/render_supplement.py --root /path/to/this/revision` and optionally `--anonymous`.

Scientific evidence remains in the [48-fit audit](../2026-10-24/) and [external-source/BioGRID experiment record](../../experiments/2026-10-24-external-validation/). Large data/checkpoints remain outside this compact package. Anonymity refers to document content; authorized named versions are also public. Private correspondence is not included. This formatting check is not PDF eXpress certification or conference submission.
