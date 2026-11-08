# Original manuscript + BioGRID: local figure and text corrections

Independent candidate, 8 November 2026. Based on the archived [original + BioGRID candidate](https://github.com/brynn-t02/ppi-graph-missense-audit/tree/81b2d92f978cca1590d006bfc334e422b5150e64/revisions/2026-11-08-original-second-network-only), not the cumulative v8 manuscript.

- `BIBM2026_workshop_original_plus_BioGRID_local_fixes.pdf`: 8-page named main paper, including references.
- `BIBM2026_supplementary_original_plus_BioGRID_local_fixes.pdf`: 12-page supplementary information.
- `source/`: main LaTeX, IEEE class, figures, fonts and supplementary Markdown.
- `original_submission/`: unchanged original source, manuscripts and compact source-data archive.
- `reproducibility/`: exact text differences, local figure-edit scripts and verification records, plus unchanged compact BioGRID evidence and pinned links to existing predictions and executed code.
- `逐条修改说明_中文.txt`: detailed Chinese explanation of this round's changes.

Figures 1 and 3 have only specified titles/annotations corrected. Figure 4 has two titles corrected, 18 direction dots made uniform in size/opacity, and seven significance stars removed. Figure 2 is byte-identical to the preceding candidate. No scientific values, bar heights, effect sizes, confidence intervals, subgroup definitions, dot centres, semantic colours or panel layouts were changed. Figure 3C's new title wraps within its original title band at the original font size. Corresponding captions, interpretation and supplementary notes were synchronized in 11 targeted text edits.

The empirical scope remains the original study plus existing BioGRID second-network validation. No HUMSAVAR or November unified-retraining results enter this manuscript. No training, inference or statistical recalculation was performed. The original control-protocol and dependence limitations remain explicit. All prior versions and experimental records are preserved.

Compile from `source/` with `tectonic main.tex` or a compatible IEEEtran LaTeX installation. Render supplementary Markdown with `python reproducibility/render_supplement.py` (ReportLab required). Figure-patch scripts document the executed local edit workflow; their recorded input locations refer to the preceding immutable candidate. Ordinary manuscript compilation uses the supplied final figure PDFs and requires no model data.

This edited candidate has not been newly certified by PDF eXpress or submitted to the conference.
