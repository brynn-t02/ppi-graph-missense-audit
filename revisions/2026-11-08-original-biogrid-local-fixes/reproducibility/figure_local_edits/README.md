# Local vector-edit provenance

The scripts here are records of the executed local workflow. Ordinary LaTeX compilation does not execute them: it uses the final PDFs under `source/figure_drafts/`.

Their immutable inputs are the original figure PDFs in the [preceding candidate](https://github.com/brynn-t02/ppi-graph-missense-audit/tree/81b2d92f978cca1590d006bfc334e422b5150e64/revisions/2026-11-08-original-second-network-only/source/figure_drafts). Each script checks its input hash. To rerun independently, set its input, output and font paths to your checkout; the Figure 4 helper's `Figure4_before.pdf` is a byte-identical copy of that preceding Figure 4. Runtime dependencies include Python, pypdf, NumPy, Pillow and Matplotlib; Poppler is used for the 300 dpi pixel audit.

No prediction arrays or numerical research data are loaded by these patches. Geometry and colour preservation are checked against the original vector streams and by raster differences outside explicitly authorized text/dot/star regions. Figure 4 E uses common radius 4 points and alpha 0.85. These constants encode no statistical significance.
