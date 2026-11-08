# Figure typography verification

The four figure PDFs and PNGs in `../../source/figure_drafts/` are the final outputs. Rendering scripts and compact aggregate inputs are under `figure12/`, `figure3/`, and `figure4/`. They preserve the original manuscript experiments; no new fitting or statistical resampling occurred.

`delivery_verification.json` records the final main PDF hash, page count, font embedding, effective figure font sizes, and preserved links/bookmarks. Independent reviews check original-vs-new scientific geometry and all eight final manuscript pages. The supplement is byte-identical to the preceding date-restored revision.

`verify_delivery.py` is the executed local-workspace audit, retained as an execution record; its baseline, logs, and runtime paths refer to the original task workspace. The figure-rendering scripts have their own path and font requirements. No proprietary font binaries are redistributed. Existing licensed DejaVu fonts elsewhere in the source package are for reproducing the unchanged supplement.
