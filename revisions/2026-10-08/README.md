# Anonymous manuscript snapshot — 2026-10-08

The main-review revision limits the claims to fixed STRING propagation, clarifies historical training protocols and exploratory subgroup selection, and adds complete-case and threshold checks from frozen predictions.

- [Main paper](paper.pdf): 8 pages, anonymous.
- [Supplement](supplement.pdf): 11 pages. The October 23 snapshot retains the October 8 supplement unchanged.
- [Editable manuscript](source/BIBM2026_main_paper_revised.tex) and its figure dependencies.
- [Supplement source](source/Supplementary_Information_revised.md).
- [Figure generators](source/figure_scripts/) and [figure inputs](source/figure_inputs/).
- [Compact source data and frozen checks](source/source_data_package_v1/).

Build the manuscript from `source/` with an IEEEtran-compatible LaTeX installation or `tectonic BIBM2026_main_paper_revised.tex`. Regenerate figures with Python, NumPy, pandas and matplotlib by running the scripts under `source/figure_scripts/`. To reproduce the complete-case and threshold checks, run:

```sh
python source/source_data_package_v1/revision_20261008/recompute_checks.py
```

No new training or independent cohort validation was performed in this revision. The checks use already-inspected frozen predictions and are post-hoc, rather than independent confirmation. Historical training scripts are provenance copies and depend on the original data environment. Provider-controlled source resources and the full training environment are not part of this compact archive.

These dated PDFs and manuscript sources are preserved as delivered. Statements that public deposition remained outstanding describe the delivery date: the repository was made public by its owner and this snapshot was synchronized on 24 October 2026. That publication does not complete the outstanding experimental validation. See the repository root README for the current availability status and work remaining. Review emails, private reviewer-response drafts and local machine audit notes are not included here.

`SHA256SUMS.txt` verifies files relative to this snapshot directory.
