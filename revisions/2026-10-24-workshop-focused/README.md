# Workshop-focused revision — 24 October 2026

The current manuscript prioritizes the workshop comments; main-conference-only requests are set aside. Content is selected for the central evaluation argument and the eight-page main-paper limit.

- [Named main paper](BIBM2026_workshop_named_revised.pdf) / [anonymous main paper](BIBM2026_workshop_anonymous_revised.pdf): **7 pages each, including references**.
- [Named supplement](BIBM2026_supplementary_named_revised.pdf) / [anonymous supplement](BIBM2026_supplementary_anonymous_revised.pdf): **14 pages each**.
- [Named manuscript source](source/main.tex), [anonymous source](source/main_anonymous.tex), [supplement source](source/Supplementary_Information_revised.md).
- [Content decisions and conclusions](REVISION_NOTES.md), [main-text changes](changes_from_20261024_matched.diff).
- [Experiment code, specifications and protocol](reproducibility/code/), [saved predictions and verification](reproducibility/outputs/).

## Content selection

The main paper retains three figures covering connectivity/topology controls, strong-predictor fusion, homology evaluation and exploratory topology context. Table I identifies current and historical results. The former main Figure 4 and detailed historical prediction transitions are now Supplementary Figure S1, with Tables 6 and 10. Methods, repeated numerical listings and extended validation discussion are shortened; complete records remain in the supplement and reproducibility archive. The accepted title and confirmed authors are unchanged.

No new training was performed for this editorial revision. The 48 fitted classifiers, original inputs, scripts and numerical results are unchanged from the [preceding matched-experiment revision](../2026-10-24/). Main-only requests from the earlier conference reviews are not new experiment commitments. Independent-cohort or second-network validation remains an unresolved workshop suggestion; stating this limitation does not complete that experiment.

## Build and provenance

Compile source/main.tex or source/main_anonymous.tex from source/ using an IEEEtran-compatible engine (e.g. Tectonic). Figure scripts use Python, NumPy, pandas and Matplotlib. Render the supplement with Python/ReportLab and the included fonts:

```sh
python reproducibility/render_supplement.py --root /path/to/2026-10-24-workshop-focused
python reproducibility/render_supplement.py --root /path/to/2026-10-24-workshop-focused --anonymous
```

Executed experiment code keeps its original study paths and verification hashes; adapt base paths before use on another machine. Original raw environment exports include local-build URLs; use the derived portable specifications. Executed older renderers and historical evidence are retained for provenance, while render_supplement.py is the current supplement builder.

Large provider resources, tensors, embeddings and checkpoints are not redistributed. Private reviewer/author correspondence is excluded. The anonymous PDFs omit author identifiers, but this public repository also includes authorized named versions. SHA256SUMS.txt covers files relative to this directory. No conference-system submission was performed.
