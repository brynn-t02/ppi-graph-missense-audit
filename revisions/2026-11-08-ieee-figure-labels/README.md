# Final manuscript — IEEE figure typography update (8 November 2026)

This is the author-selected **original manuscript + BioGRID** content version, with the author-confirmed ClinVar date **31 March 2024**. This revision changes figure typography and necessary page balancing only.

- [Main manuscript, 8 pages](BIBM2026_workshop_final_IEEE_figures.pdf)
- [Supplementary Information, 12 pages, unchanged](BIBM2026_supplementary_original_plus_BioGRID_date_restored.pdf)
- [Detailed change statement in Chinese](图中文字调整说明_中文.md)
- [LaTeX source and vector figures](source/)
- [Typography scripts, compact plotting data and verification](reproducibility/figure_font_update/)

All four figures retain the original six-panel design, scientific data, uncertainty intervals and colors. Their labels now use embedded Times New Roman with a minimum effective placed size of 8.00024 PDF points. Long labels are wrapped, axes/legends repositioned where needed, and selected tick labels are made less dense without changing axis limits. Existing Figure 1E error-bar clipping at the original axis limits is retained, not introduced by this revision.

The main LaTeX content is unchanged except that the IEEE reference-column balancing trigger moves from reference 25 to reference 16 to maintain eight pages after figure heights increase. The supplement is byte-identical to the preceding date-restored version. This revision does not replace original results with the later unified-experiment results.

Per the author's latest instruction, the original PDF settings, 29 clickable links and 15 bookmarks are retained. The file is PDF 1.5. The author reports that the preceding PDF passed PDF eXpress; **this new file still requires its own online check**. No current-file certification or conference submission is claimed.

## Build

From `source/`, compile `main.tex` with a LaTeX installation supporting IEEEtran, or use `tectonic --only-cached --keep-logs main.tex` when the required bundle is already cached. No scientific datasets, checkpoints or GPU are required. The included figure PDFs already contain embedded fonts. Redrawing figures additionally requires NumPy, pandas, matplotlib and a locally installed Times New Roman font; font binaries are not distributed here.

The current main SHA256 is `baf39be676460541d1a1022d2237e1a62203ea8d3a13872860fd0ce90ccb3052`.

Prior manuscripts, scientific analyses and the original submission are preserved. The `reproducibility/prior_date_revision/` records refer to the prior date-restoration revision, not to the new PDF hash.
