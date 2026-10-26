# Camera-ready source package — 26 October 2026

The manuscript source and five figure assets are unchanged from the [font-clean revision](../../revisions/2026-10-25-font-clean/). The matching [PDF eXpress-certified main](../../revisions/2026-10-25-pdf-express-certified/2026400920.pdf) remains the final PDF to upload separately. This is packaging, not a new manuscript revision or a completed conference submission.

- [Source archive in tar/gzip format](BIBM2026_manuscript_source.tar.gz)
- [Identical source archive in ZIP format](BIBM2026_manuscript_source.zip)
- [Clean-build verification](BUILD_VERIFICATION.json)
- [Supplementary Information](../../revisions/2026-10-25-font-clean/BIBM2026_supplementary_named_revised.pdf)

Each archive contains one entry-point `main.tex`, the unchanged `IEEEtran.cls`, the five PDF figure inputs, build instructions, a font-license notice, and checksums. All 30 references are inline in `main.tex`; no external bibliography file is missing. Experiment datasets, checkpoints, previous drafts, correspondence, and administrative documents are outside this source package.

A clean Tectonic 0.16.9 build from the seven required input files produced eight US-Letter pages. Every page is pixel-identical to the certified PDF at 300 dpi; character content and geometry match, and all font resources are embedded. The verification build does not itself carry PDF eXpress certification. The original returned certified PDF is preserved byte-for-byte.

The [public workshop final-submission entrance](https://wi-lab.com/cyberchair/2026/bibm26/scripts/final.php?subarea=S03), read on 26 October 2026, explicitly requests a PDF and compressed source files, lists tar/gzip/uuencode, and requires the copyright form before camera-ready upload plus a registration receipt. Use the `.tar.gz` archive for the listed compression convention. ZIP is an alternative if the authenticated form accepts it. The current source references PDF figures, not EPS; authenticated upload extension/size restrictions were not inspected.

Copyright signing, registration confirmation, and final submission are separate author actions; none was performed during this packaging step. Existing manuscript revisions and experiment archives are preserved.
