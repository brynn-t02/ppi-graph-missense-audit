# BIBM 2026 targeted candidate v2 — cumulative continuation

This v2 continues the immediately preceding [8 November targeted candidate](../2026-11-08-targeted-candidate/). It does not restart from the original or certified manuscript. All previous Introduction, Methods, Table I, Section III.A and Figure 1 matched-result revisions remain intact.

- `BIBM2026_workshop_targeted_candidate_v2.pdf`: 8-page named candidate main.
- `BIBM2026_supplementary_targeted_candidate_v2.pdf`: 19-page candidate Supplementary Information.
- `source/main.tex`: editable LaTeX entry point with 30 inline references.
- `source/figure_drafts/`: five complete figure PDF inputs; Figure 3–5 are byte-identical to v1.
- `source/Supplementary_Information_revised.md`: updated direct Figure 2/3 explanatory mappings; all numerical table rows preserved.
- `修改说明.txt`: detailed Chinese cumulative change report.
- `reproducibility/`: before/after edits, exact diffs, per-fit numeric inputs, existing bootstrap intervals and preservation/export checks.

This round changes only Figure 1D/F annotation text; Results III.B/III.C and Figure 2/3 captions; Figure 2B/C/E numerical displays; and directly corresponding supplementary explanatory text. Figure 2A/D/F are preserved. Figure 3 remains a fixed-reference-pair analysis. Mean per-fit metrics and ensemble differences remain distinct. Original records are not overwritten.

Permutation sample SD was computed from the nine existing classifier fits (ddof=1). There was no new training, inference or bootstrap run, and no server or bulk-data access. The only mechanical pagination adjustments are Figure 2/3 top-placement permission and the final reference-column balance hook; figure dimensions, fonts and margins are unchanged.

Compile from `source/` with `tectonic --only-cached --keep-logs main.tex` using existing TeX resources (tested with 0.16.9). On a fresh setup, ordinary TeX resource retrieval may be required. Render Supplementary Information with `python reproducibility/render_supplement.py --root .`, with ReportLab and the bundled fonts. Its default intermediate output name retains the old naming convention; use the explicitly v2-named delivery PDF.

Both final PDFs have all font resources embedded and were visually checked. This candidate is unsubmitted and has not received a new IEEE PDF eXpress certification. All prior snapshots and the original certified PDF are preserved.
