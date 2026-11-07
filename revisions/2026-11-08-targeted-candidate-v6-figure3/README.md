# BIBM 2026 cumulative candidate v6 — matched-ensemble topology profiles

Continues [v5](../2026-11-08-targeted-candidate-v5-text/). Figure 3B–F, Results III.C, its caption, Table I and directly related supplementary descriptions/Table 5 now use the canonical original-split raw-ESM and full-FPP ensembles (classifier seeds 42–44, float64 probability means). Figure 3A preserves the original vector content and is pixel-identical at 300 dpi. Figures 1, 2, 4 and 5 and all other scientific content are preserved.

- `BIBM2026_workshop_targeted_candidate_v6_figure3.pdf`: 8-page main.
- `BIBM2026_supplementary_targeted_candidate_v6_figure3.pdf`: 19-page Supplementary Information.
- `source/`: LaTeX, supplementary Markdown and referenced figures/fonts.
- `修改说明.txt`: detailed Chinese changes, results and remaining-asset audit.
- `reproducibility/figure3_matched/analysis/`: actual input hashes, frozen memberships, six seed probabilities, 18 strata/53 cells, 2000 paired protein-bootstrap draws per group, selected original four cells, before/after comparisons and verification programs.
- `reproducibility/figure3_matched/figure/`: executed plotting code, exact panel data and rendering audit.
- `reproducibility/figure3_matched/independent_review.json`: independent rank-AUC/AP and expanded-row bootstrap verification.

The original groups and four displayed cells were frozen before evaluating the new ensembles. The four displayed AUC differences remain three positive and one negative, but three of their protein-level intervals include zero. Protein resampling addresses repeated variants within proteins; intervals remain conditional on fitted models and graph and are unadjusted for the original same-test-set subgroup selection and multiple comparisons. Four non-displayed sparse cells have undefined single-class draws, recorded and excluded without replacement draws. Complete counts and finite-replicate counts are archived. This is not independent cohort confirmation.

No new model fitting, inference, feature extraction or graph propagation occurred. All 71 groups received newly computed protein-level bootstrap intervals; old variant-level intervals were not reused. Figure 3E adds one histogram bin for the new minimum; Figure 3F expands its axis to display full intervals. Canvas, panel positions, colors and font sizes follow the preceding figure. Old files and snapshots remain archived.

Build main from `source/` with `tectonic --only-cached --keep-logs main.tex`; run `python reproducibility/render_supplement.py` for SI with the bundled fonts and ReportLab. Portable numeric verification: `python reproducibility/figure3_matched/analysis/verify_compact_figure3.py` (NumPy/pandas, no training). Other executed reconstruction scripts retain their documented original workspace paths. All 27 pages were visually checked and fonts embedded.

The lightweight paper/source ZIP excludes Figure 3/4 numeric archives. Two Figure 3 ZIPs (compact evidence and bootstrap draws) and the unchanged v4 Figure 4 data ZIP supply those directories; extract all in one folder for the complete SHA256SUMS manifest. The GitHub snapshot contains every file. Every delivered file/ZIP is below 5 MB. Figure 1E and Figure 2D remain under their disclosed separate protocols; this round only audited their existing assets.

This candidate has not been newly PDF eXpress certified or conference-submitted. The earlier certified file and all revision snapshots remain unchanged.
