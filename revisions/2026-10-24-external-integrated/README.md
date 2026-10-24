# Workshop revision incorporating external validation — 24 October 2026

- `BIBM2026_workshop_named_revised.pdf` and `BIBM2026_workshop_anonymous_revised.pdf`:7 pages each, including references.
- `BIBM2026_supplementary_named_revised.pdf` and `BIBM2026_supplementary_anonymous_revised.pdf`:19 pages each.
- `source/`: editable manuscripts, supplement Markdown, all five figure generators and compact scientific inputs.
- `REVISION_NOTES.md`: scientific conclusions, content choices and remaining limits.
- `final_quality_checks.json`: page counts, author/anonymity checks, figure review, fonts and PDF hashes.

Compile `source/main.tex` or `source/main_anonymous.tex` from the source directory with Tectonic or an IEEEtran-compatible engine. Run the five scripts under `source/figure_scripts/` with Python, NumPy, pandas and Matplotlib to regenerate figures. The local executable versions used for rendering are recorded in `reproducibility/render_environment.json`.

Render either supplement with Python and ReportLab (included font files are under source/resources/fonts):

```sh
python reproducibility/render_supplement.py --root /path/to/this/revision
python reproducibility/render_supplement.py --root /path/to/this/revision --anonymous
```

## Experiment archive

This compact manuscript package does not duplicate models or full predictions. Exact executed code, protocols, source manifests and compact outputs are already published at:

- [External-source and BioGRID validation](https://github.com/brynn-t02/ppi-graph-missense-audit/tree/main/experiments/2026-10-24-external-validation)
- [48-fit matched audit, environment and historical source data](https://github.com/brynn-t02/ppi-graph-missense-audit/tree/main/revisions/2026-10-24)

Local figure inputs are sufficient to redraw all included figures; they retain historical/current provenance. Full provider resources, graph arrays, model tensors, new checkpoints and prediction arrays remain in the experiment archive on the compute server. Private review/author correspondence and machine connection records are excluded from the public package. The repository also contains authorized named versions, so anonymity refers to the document contents, not the repository owner.

Historical revision snapshots are unchanged. No new experimental outcomes were created by this editorial pass. SHA256SUMS.txt uses paths relative to this directory.
