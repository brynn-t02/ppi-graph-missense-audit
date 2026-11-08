# BIBM 2026 cumulative candidate v8 — statistical display clarifications

Continues [v7](../2026-11-08-targeted-candidate-v7-unified/). Only Figure1D's annotation and the requested Figure1/2 caption clauses change. The annotation now reads “Ensemble ΔAUC +0.0264”; the actual ensemble difference is unchanged and is not replaced by a difference of displayed means. Figure1C states95% intervals. Figure1 B/D/F bars are identified as means with sample SD, including pooled graph-permutation/classifier-seed variation for the permutation control. Figure1E's existing fixed-graph/three-seed/shared-endpoint explanation is retained. Figure2E explicitly states means/sample SD.

The optional Discussion sentence was omitted: trial compilation with it gave9pages; the requested caption changes alone retain8pages. Existing Results already report the homology representation comparison. Discussion, all numerical results, Tables and Supplementary Information are unchanged. No new training, inference, bootstrap or server access occurred.

- `BIBM2026_workshop_targeted_candidate_v8_clarity.pdf`:8-page main including references.
- `BIBM2026_supplementary_targeted_candidate_v7_unified.pdf`:unchanged21-page supplement, filename retained to make the reuse explicit.
- `source/`: current LaTeX and referenced assets.
- `修改说明.txt`: exact Chinese change details.
- `reproducibility/clarifications_v8/`: exact text replacements, vector annotation patch and independent scope checks.
- Existing Figure3/4 and remaining-unification experiment directories preserve their full v7 contents and results. Earlier main audits are under `reproducibility/history/v7/`.

The Figure1 annotation reuses its embedded7.4pt font; outside the text box, all pixels match at300dpi. All8 main pages were rendered and reviewed, and fonts are embedded. Figures2–5 and SI are byte-identical to v7. Build main with `tectonic --only-cached --keep-logs main.tex` in `source/`; SI still uses the inherited renderer and unchanged output filename.

The v8 paper/source ZIP excludes unchanged compact experiment directories. Extract it together with the three v7 Fig1E/Fig2D evidence parts and unchanged Figure3 evidence/bootstrap and Figure4 verification ZIPs to reconstruct the complete manifest. Every file/archive is below5MB. GitHub contains the complete snapshot. All previous versions remain preserved. This candidate is not newly PDF eXpress certified or submitted to the conference.
