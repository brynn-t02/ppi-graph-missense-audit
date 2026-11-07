# BIBM 2026 targeted candidate v3 — paragraph-spacing repair

Continues the immediately preceding [cumulative v2](../2026-11-08-targeted-candidate-v2/) with all earlier edits preserved. The only manuscript-source change adds `\raggedbottom` after `\maketitle`, preventing flush-bottom paragraph stretching. The page-4 gap between the two topology paragraphs decreases from 45.92075 pt to the normal 1.99253 pt.

- `BIBM2026_workshop_targeted_candidate_v3_spacing.pdf`: corrected 8-page main.
- `BIBM2026_supplementary_targeted_candidate_v2.pdf`: unchanged 19-page supplement from v2.
- `source/main.tex`: current editable entry point; all five figure dependencies are included.
- `修改说明.txt`: concise Chinese explanation.
- `reproducibility/main.diff`: the complete one-line source change relative to v2.
- `reproducibility/verification.json`: page, text/font preservation, embedding and spacing checks. Figure/data audits are inherited unchanged from v2 because figure assets and data are unchanged.

All page text, fonts, font sizes, horizontal positions, figures, statistics, reference contents, margins and image sizes are preserved. Only vertical spacing changes. All eight main pages were rendered and visually checked; all font resources remain embedded. No experiment, inference or statistical calculation was run.

Compile from `source/` using `tectonic --only-cached --keep-logs main.tex` with existing TeX resources (tested0.16.9). This remains an unsubmitted candidate without a new PDF eXpress certification. All earlier candidates and certified files are preserved.
