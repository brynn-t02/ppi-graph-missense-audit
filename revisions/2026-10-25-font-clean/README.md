# BIBM workshop: embedded-font cleanup (25 October 2026)

This export-only revision removes unused, unembedded Helvetica resources through reproducible generator settings. All four PDFs have fully embedded and subset font resources. All 52 pages are pixel-identical to the previous version at 300 dpi; manuscript content, references, plots and layout are unchanged.

- [Named main paper — 8 pages](BIBM2026_workshop_named_revised.pdf)
- [Anonymous main paper — 8 pages](BIBM2026_workshop_anonymous_revised.pdf)
- [Supplementary Information — named, 18 pages](BIBM2026_supplementary_named_revised.pdf)
- [Supplementary Information — anonymous, 18 pages](BIBM2026_supplementary_anonymous_revised.pdf)
- [Chinese change and reference audit](字体清理与参考文献核对说明.txt)
- [Quality checks](quality_checks.json)

The figure-4 overlay explicitly initializes its registered embedded font. The supplementary renderer explicitly selects DejaVu for document and table initialization. These settings prevent unused default Helvetica resources from returning; no font dictionary was removed in isolation.

Compile main.tex or main_anonymous.tex from source/ with Tectonic. Rebuild the supplement with python reproducibility/render_supplement.py (add --anonymous for the anonymous edition). Figure scripts and bundled font licenses are under source/. Exact generator diffs, font audits, pixel comparisons and reference mapping are in reproducibility/.

The original submission had 28 references; the current manuscript retains them and adds only UniProt and BioGRID for the completed validation. This export revision changes no citation or bibliography entry. All previous snapshots, including [heading-style](../2026-10-25-heading-style/), remain available.

Local checks are complete. PDF eXpress certification and conference submission have not been performed; they remain separate from this export and public archive.
