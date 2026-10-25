# BIBM workshop: reporting clarity (25 October 2026)

This revision clarifies figure wording, historical uncertainty summaries and access to Supplementary Information. All four original figures remain in the main paper; completed external-source and second-network validation remains Figure 5. Main PDFs are 8 pages including references; supplements are 18 pages.

## Read the paper and Supplementary Information

- [Named main paper — 8 pages](BIBM2026_workshop_named_revised.pdf)
- [Supplementary Information — named, 18 pages](BIBM2026_supplementary_named_revised.pdf)
- [Anonymous main paper — 8 pages](BIBM2026_workshop_anonymous_revised.pdf)
- [Supplementary Information — anonymous, 18 pages](BIBM2026_supplementary_anonymous_revised.pdf)
- [Detailed Chinese change statement](本轮修改逐条说明.txt)
- [Revision notes](REVISION_NOTES.md)
- [Quality checks](final_quality_checks.json)

The supplement is part of this public snapshot. The main paper explicitly includes Supplementary Information in its availability statement. These links do not depend on a separate supplementary-material slot in a conference submission system.

## Sources and reconstruction

The source directory contains LaTeX, the unchanged IEEEtran class, supplementary Markdown, figure assets, generators and compact numerical evidence. Compile source/main.tex or source/main_anonymous.tex with Tectonic from the source directory. Rebuild the supplement with python reproducibility/render_supplement.py (add --anonymous for the anonymous copy). Rendering package versions are in reproducibility/render_environment.json.

Only six figure text blocks changed. The scripts source/figure_scripts/figures1_3_text_clarity_patch.py and source/figure_scripts/figure4_reporting_clarity.py reproduce the local vector edits; bundled fonts and their license are in source/resources/fonts. Source and output hashes, allowed edit regions and pixel-difference checks are archived alongside the figures and in reproducibility/. Every earlier figure asset is retained unchanged.

Exact main/supplement source diffs, before/after records and independent content/visual checks are in reproducibility/. Table I, all 267 supplementary table lines and all 49 canonical-analysis files are unchanged. This revision does not retrain models, recompute bootstrap results or change scientific values.

The preceding revision is [2026-10-25-original-minimal](../2026-10-25-original-minimal/). All older versions remain available. Large datasets, feature tensors and model checkpoints remain stored separately. Conference submission and PDF eXpress certification are not part of this editorial revision.
