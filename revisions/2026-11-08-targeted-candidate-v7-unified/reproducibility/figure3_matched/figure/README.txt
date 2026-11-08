Figure 3 matched-ensemble rendering

Only panels B–F use freshly recalculated statistics. A is copied from the v5 baseline as original vector operations and has 0 changed pixels at 300 dpi.
Original canvas: 955.164 x 523.40475 pt; axes boxes, colors, point sizes and six-panel layout retained.
Necessary axis updates: E extends one equal-width bin to -0.0941667 so all 53 fixed cells are represented; F extends to [-0.225,0.225] to include every protein-bootstrap interval and avoid clipping tick labels.
F uses the four original fixed cells in the original order, with new 2000-resample paired protein bootstrap intervals. No re-selection, model fitting or inference occurred in this renderer.

Execution used existing work/experiment_20261024/render_env/bin/python.
The renderer accepts --analysis-dir, --source-pdf and --font-dir for relocated inputs. Default paths describe the verified project workspace.
Fresh analysis inputs live in the sibling analysis folder. The unchanged source Figure 3A and DejaVu fonts must be supplied when running outside the original workspace.

figure3_render_audit.json records every input hash, original-A pixel equality, exact canvas/axes and necessary axis changes. panel_data.json contains all plotted B–F data.
