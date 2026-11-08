# Original manuscript + BioGRID second-network revision

Independent candidate, 8 November 2026. This version restarts from the preserved original submission source, not the cumulative v8 candidate.

- `BIBM2026_workshop_original_plus_BioGRID.pdf`: 8-page named main paper.
- `BIBM2026_supplementary_original_plus_BioGRID.pdf`: accompanying supplementary information.
- `source/main.tex`, `source/IEEEtran.cls`, `source/figure_drafts/`: buildable main source. All four figure PDFs are byte-identical to the original accepted figure files.
- `source/Supplementary_Information_revised.md`: supplementary source.
- `reproducibility/render_supplement.py`: renders the supplement with bundled DejaVu fonts.
- `original_submission/`: unchanged original main/supplement/source and compact source-data archive.
- `reproducibility/BioGRID/`: verified existing network-validation summaries and immutable links to executed code, complete memberships and per-seed predictions.
- `逐条修改说明_中文.txt`: detailed Chinese changes, including scientific limitations that cannot be silently removed.

Only BioGRID second-network validation is added to the original empirical evidence. No external-cohort results or November unified-retraining results enter this manuscript. Figures 1–4 and Results III.A–D retain their original numerical values. Configuration and interpretation clarifications do not imply that the original controls were fully matched. Original Figure 4 significance/difficulty-matching encodings remain visually present by request; their inferential interpretation is explicitly withdrawn in the caption and supplement. Invalid FDR columns are excluded from the revised supplementary table, with the original unchanged document preserved.

Build the main PDF from `source/` using `tectonic main.tex` (or a compatible IEEEtran LaTeX installation). The source is ready to compile and needs no experiment data. Render the supplement using Python with reportlab: `python reproducibility/render_supplement.py`. The local-original reconstruction helper is provenance, not required for ordinary compilation.

The original 28 bibliography entries are retained; only a BioGRID resource citation is added. No experiments, inference or bootstrap were run for this editorial revision. Previous candidates and experimental records remain archived. This modified candidate is not newly PDF eXpress certified or conference-submitted.
