# Figure 4 typography-only redraw

The selected original-plus-BioGRID manuscript is the scientific baseline. This directory redraws its original Figure 4 using the six archived aggregate JSON/CSV files in `source_data/`; it does not load predictions or fit models. All feature definitions, prediction counts, displayed effect sizes, bar proportions and color directions remain unchanged. Previously removed inferential encodings remain absent: E has uniform direction-only dots, and F has no significance stars.

The output canvas is exactly 503.79 × 345 PDF points. Text uses embedded Times New Roman at 8 points; the overall heading uses 9 points. Only line wrapping, panel spacing and F-axis tick density change. All feature names remain unabridged.

Run `draw_figure4_ieee.py` with Python, matplotlib, NumPy and pandas. The `FIGURE4_OUTPUT` environment variable optionally selects the output PDF path; otherwise output is written beside the script. A PNG and drawing audit are written beside the script. `TIMES_NEW_ROMAN_REGULAR` and `TIMES_NEW_ROMAN_BOLD` can specify installed font paths; macOS system font paths are defaults. Font files are not redistributed.

Run `verify_figure4_ieee.py` with the same `FIGURE4_OUTPUT` value and pypdf, pdfplumber, Pillow and NumPy. It compares against the immutable `Figure4_before.pdf`, checks embedded fonts and actual PDF text sizes, recovers all seven effect sizes from original/new bar geometry, checks transition proportions and colors, compares every heatmap cell color and the 18 direction-dot colors, and verifies original count labels and feature names. The geometry tolerance of 5e-8 accommodates decimal PDF path serialization, not scientific changes.

The stored source tables contain the original archived test-statistic columns, but no inferential p-values/FDR values are used in this redraw. `source_data_manifest.json` records their exact hashes. `figure4_verification.json` records the completed numerical and PDF checks. No system font binaries, datasets, model files or credentials are included.
