Figure 3 font/layout reproduction

Run: python3 redraw_figure3_fonts.py
Requires Python, NumPy, pandas, and Matplotlib. The original plotting functions and all eight compact source summaries are included using relative paths. No model training, inference, bootstrap, or new subgroup selection is performed.

Default output: reproduced/Figure3_topology_context_v1_accepted.pdf and .png. Set PAPER1_FIG_OUT to change it. The delivery figure is never overwritten by default.

Times New Roman regular and bold must be available under the system default macOS paths, or set TNR_REGULAR and TNR_BOLD to licensed installed font files. Font files are not redistributed.

The original local execution wrapper is preserved under execution_archive/. The portable script changes input/output path resolution only; scientific plot arrays and the final typography/layout are unchanged.
