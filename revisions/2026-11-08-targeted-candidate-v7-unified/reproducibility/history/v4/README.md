# BIBM 2026 cumulative candidate v4 — matched-ensemble Figure 4

Continues the [v3 spacing-repaired candidate](../2026-11-08-targeted-candidate-v3-spacing/) with earlier revisions preserved. Only Table I, Results III.D, Figure 4/caption and directly dependent supplementary descriptions/Tables 6 and 10 change. Figure 3 remains a fixed-reference-pair analysis; Figures 1, 2, 3 and 5 are byte-identical to v3.

- `BIBM2026_workshop_targeted_candidate_v4_figure4.pdf`: 8-page main.
- `BIBM2026_supplementary_targeted_candidate_v4_figure4.pdf`: 19-page supplement.
- `source/main.tex`: current LaTeX source; all referenced figure PDFs included. The older, unreferenced Figure 4 asset is retained for history.
- `修改说明.txt`: detailed Chinese changes, findings and limits.
- `reproducibility/main.diff` and `supplement.diff`: exact textual changes.
- `reproducibility/figure4_matched/analysis/`: fresh deterministic recomputation from canonical seeds42–44, alignment/lineage/metric checks, transitions, all11 features, all44 error-stratified contrasts and nine-threshold sensitivity per pair.
- `reproducibility/figure4_matched/figure/`: actual plotting program, panel data, layout/font/input audit.

At threshold0.5, raw ESM→Full FPP corrects960 and degrades993 (net−33); matched CADD+AlphaMissense→Full FPP fusion corrects679 and degrades224 (net+455). Both partitions close to18,220. All four averaged-probability metrics match the canonical records; representation probabilities are constant within each protein. Features retain their original definitions and encodings. Heatmaps, direction dots and Hedges'g are descriptive and account for neither protein-level inferential dependence nor difficulty matching. No p/FDR annotations are used.

Recompute numeric verification from the supplied compact inputs: `python reproducibility/figure4_matched/verify_compact.py` (NumPy/pandas; tested). It verifies12seed means, four ensemble AUC/AP pairs,220 feature means,44 effect sizes and18 threshold partitions without training/inference/bootstrap. The archived executed full recomputation/plot scripts retain original workspace paths and input hash records; they require those original assets if rerun. Canonical source predictions and run metadata remain in the preserved `2026-10-25-font-clean/source/` snapshot.

Build main from `source/` with `tectonic --only-cached --keep-logs main.tex` and cached standard TeX resources (tested0.16.9). The SI renderer and fonts are included. Main and supplement were fully rendered and checked; all fonts embedded. Prior experimental records/snapshots are preserved. No training, inference, feature generation, propagation, bootstrap, server/Baidu access occurred.

Paper/source and compact reanalysis data are packaged separately to keep every delivered file below5MB. Extract both into the same directory for the complete archive. This remains an unsubmitted candidate, without a new PDF eXpress certification.
