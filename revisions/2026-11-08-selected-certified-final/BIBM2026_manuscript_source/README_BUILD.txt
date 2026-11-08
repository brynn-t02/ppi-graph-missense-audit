BIBM 2026 Net-AI-BHI final manuscript source package
Prepared 8 November 2026. Main entry point: main.tex.

Selected content: original manuscript + BioGRID, author-confirmed ClinVar date
31 March 2024. This is the version BEFORE the figure-font-size revision.
All six required manuscript inputs are copied unchanged from that version:
  main.tex                    Paper, authors, macros, and 29 inline references
  IEEEtran.cls                IEEE conference class
  figure_drafts/*.pdf          Four exact vector figure files used by main.tex

The bibliography is inline in main.tex. No .bib or .bst file is missing.
No external TeX subfiles, experimental data, checkpoints, or GPU are needed.
Figure fonts are embedded in the PDFs. The source references PDF vector figures;
there is no required EPS input. The included DejaVu license covers those figure
fonts. IEEEtran's license notice remains in the class file.

Clean-directory compilation was tested using Tectonic 0.16.9:
  tectonic --only-cached --keep-logs main.tex
For a fresh Tectonic installation, use `tectonic main.tex` to obtain its standard
TeX resources. A regular pdfLaTeX installation is an intended alternative
(run `pdflatex main.tex` twice), but was not separately tested on this host.
Standard packages: fontenc, cite, graphicx, amsmath, amssymb, amsfonts,
textcomp, url, array, hyperref, xcolor.

Verified result: 8 US-Letter pages. Exact extracted text matches the selected
certified PDF on every page. All eight 300-dpi page renders are pixel-identical.
Existing Underfull hbox warnings remain; no Overfull box, missing-glyph,
undefined-command or unresolved-reference error was found.

The locally rebuilt PDF is only a verification artifact. Submit the unchanged
PDF eXpress-returned 2026424437.pdf separately as the final manuscript PDF.
Selected certified PDF SHA256:
4bb4c6d409a3c9ffa04610fd0cd7b1d68d0434f812fdf1557caaeff6a24b2831
PDF Info and XMP contain:
Certified by IEEE PDFExpress at November 8, 2026 13:16:32

Use the .tar.gz archive for the source upload because the official final-paper
page lists tar/gzip/uuencode. The .zip has identical contents for convenience
and is an alternative only when the upload form accepts it.
Neither source archive includes the final PDF, supplement, historical drafts,
experiment datasets, private correspondence, copyright form or registration
receipt. This archive contains the manuscript's required editable source.

No scientific text, data, figures, authors, references, or PDF settings were
changed while packaging. The rejected font-size revision is excluded.
