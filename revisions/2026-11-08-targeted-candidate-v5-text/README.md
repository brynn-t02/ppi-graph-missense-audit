# BIBM 2026 cumulative candidate v5 — requested text clarifications

Continues [v4 matched-ensemble Figure 4](../2026-11-08-targeted-candidate-v4-figure4/). Seven requested prose changes: one sentence in Results III.D, two Figure 4 caption clauses, the first three Discussion paragraphs, and Acknowledgment. All figures and numerical assets, Results III.E, Figure 5, Table I and supplementary content remain unchanged.

- `BIBM2026_workshop_targeted_candidate_v5_text.pdf`: 8-page main, including references.
- `BIBM2026_supplementary_targeted_candidate_v4_figure4.pdf`: unchanged 19-page supplement, retaining its v4 filename to identify the exact preserved bytes.
- `source/main.tex` and included figure PDFs: current editable manuscript inputs.
- `修改说明.txt`: itemized Chinese explanation, including the one necessary layout-only edit.
- `reproducibility/changes.json`, `main.diff`: exact seven content edits and removal of the obsolete manual reference-column break.
- `reproducibility/supplement_pdf_audit.json`: actual PDF Tables 6/10 checked against existing statistics, including 366/1089 FN groups and the feature-specific 1087 valid observations.
- `reproducibility/history/v4/`: preceding documentation retained as historical records.

Removing `IEEEtriggeratref{16}` allows references to flow naturally after the authorized text expansion and keeps the manuscript within eight pages. Fonts, sizes, margins, reference content/order and figure sizes are unchanged. All eight main pages were visually reviewed; fonts are embedded. The supplement PDF and source are byte-identical to v4, with relevant pages 8–10 and 12 independently inspected.

No models, probabilities, thresholds, metrics or uncertainty estimates were recomputed. At threshold 0.5, the existing matched ensembles yield 33 more errors for Full FPP versus raw ESM and 455 fewer for full-FPP fusion versus matched scores. These are observed net error counts, not significance claims or isolated estimates of a topology mechanism.

Build from `source/` using `tectonic --only-cached --keep-logs main.tex` with the existing standard cached TeX resources. Compact Figure 4 evidence remains byte-identical to v4. The lightweight v5 paper/source ZIP excludes `reproducibility/figure4_matched/`; the unchanged `BIBM_Figure4_matched_data_verification_20261108.zip` supplies that directory. Extract both in the same directory for the complete manifest. The GitHub snapshot contains both parts. No individual delivered file exceeds 5 MB.

Earlier snapshots and the previously certified PDF are preserved. This candidate has not been newly PDF eXpress certified or conference-submitted.
