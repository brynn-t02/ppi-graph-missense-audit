# IEEE-checked workshop revision — 24 October 2026

This revision simplifies Table I and repairs actual font selection in the compiled PDF. The scientific results and figure data from the external-validation integrated revision are unchanged.

- Named and anonymous main papers: **7 pages each, including references**.
- Named and anonymous supplements: **19 pages each**, byte-identical to the preceding integrated revision.
- Table I: five experimental settings, with repeat counts and references to detailed protocols.
- Abstract and index terms: real Times-compatible bold fonts; labels use bold italic.
- References: 30 cited entries numbered by first citation, with balanced final-page columns.
- `source/`: editable manuscripts, supplement Markdown, figure generators and compact inputs.
- `reproducibility/`: rendering environment, supplement generator, compilation logs and differences from the preceding revision.
- `final_quality_checks.json`: page counts, actual embedded fonts, unchanged scientific inputs and PDF hashes.
- `REVISION_NOTES.md`: editorial changes and retained scientific conclusions.

The IEEEtran conference class is byte-identical to the user's original IEEE-conference-template-062824 class. Its margins and text sizes were not reduced. This is a manuscript formatting check, not a conference-system or PDF eXpress certification.

## Build

Compile `main.tex` or `main_anonymous.tex` from `source/` with Tectonic or an IEEEtran-compatible engine. The T1 font encoding declaration precedes IEEEtran so Times-compatible fonts are selected correctly under Tectonic/XeTeX. Tectonic may need its normal small font resources on first use. The IEEEtran class itself is unchanged.

The five scripts under `source/figure_scripts/` regenerate figures using the included compact inputs and Python, NumPy, pandas and Matplotlib. Render supplements with Python and ReportLab:

```sh
python reproducibility/render_supplement.py --root /path/to/this/revision
python reproducibility/render_supplement.py --root /path/to/this/revision --anonymous
```

## Experimental evidence

- [48-fit matched audit](https://github.com/brynn-t02/ppi-graph-missense-audit/tree/main/revisions/2026-10-24)
- [External-source and BioGRID validation](https://github.com/brynn-t02/ppi-graph-missense-audit/tree/main/experiments/2026-10-24-external-validation)
- [Preceding integrated manuscript](https://github.com/brynn-t02/ppi-graph-missense-audit/tree/main/revisions/2026-10-24-external-integrated)

Large provider data, full tensors, new checkpoints and external predictions are not redistributed in this compact package. Anonymity applies to document content; the public repository also contains authorized named manuscripts. Historical snapshots and experiment records are unchanged. Private reviewer/author correspondence and author-facing comparisons are excluded. SHA256SUMS.txt paths are relative to this directory.
