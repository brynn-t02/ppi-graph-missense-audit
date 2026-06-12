# Figure Asset Manifest v1

Date: 2026-06-04

Synchronization note:
- Figure 2 was updated after the 2026-06-11 follow-up fusion-control experiments to show CADD+AlphaMissense+raw ESM, edge-ablated PPI-GCN, topology-permuted PPI-GCN and real PPI-GCN controls. The revised Figure 2 PNG/PDF was visually locked on 2026-06-12.

## Accepted Main Figure Assets

### Figure 1

Status:
- Accepted PNG/PDF candidate after user visual inspection on 2026-06-04.
- Scientific content matches the current Figure 1 legend in `manuscript_bibm_v20_ieee_draft.tex`.

Files:
- PNG: `/Users/teyian/Documents/paper1/figure_drafts/Figure1_connectivity_topology_controls_v2_accepted.png`
- PDF: `/Users/teyian/Documents/paper1/figure_drafts/Figure1_connectivity_topology_controls_v2_accepted.pdf`

Source script:
- `/Users/teyian/Documents/paper1/make_figure1_connectivity_v2_local.py`

Notes:
- The accepted figure was generated on the AutoDL server from real result files.
- Earlier v2 attempts had layout crowding in panel A and B/C spacing; those should not be used.
- The accepted version uses an in-panel color legend for the schematic and non-overlapping legends for panels D and F.

### Figure 2

Status:
- Locked PNG/PDF candidate after user visual inspection and final self-check on 2026-06-12.
- Scientific content matches the current Figure 2 legend in `manuscript_bibm_v20_ieee_draft.tex`.

Files:
- PNG: `/Users/teyian/Documents/paper1/figure_drafts/Figure2_fusion_homology_v1_accepted.png`
- PDF: `/Users/teyian/Documents/paper1/figure_drafts/Figure2_fusion_homology_v1_accepted.pdf`

Source script:
- `/Users/teyian/Documents/paper1/make_figure2_fusion_homology_v1.py`

Notes:
- The locked Figure 2 script uses matched follow-up fusion-control source data from exp46 and exp47.
- Earlier Figure 2 attempts had panel A connector crowding, clipped box text, hidden arrowheads and several text overlaps; those should not be used.
- The locked BIBM v20 Figure 2 shows matched original-split fusion controls in panel B, real-PPI marginal gains over controls in panel C, homology-aware raw ESM versus PPI-GCN in panel D, homology-aware strong-predictor fusion in panel E, and split-prevalence context in panel F.

### Figure 3

Status:
- Accepted PNG/PDF candidate after user visual inspection on 2026-06-04.
- Scientific content matches the current Figure 3 legend in `manuscript_bibm_v20_ieee_draft.tex`.

Files:
- PNG: `/Users/teyian/Documents/paper1/figure_drafts/Figure3_topology_context_v1_accepted.png`
- PDF: `/Users/teyian/Documents/paper1/figure_drafts/Figure3_topology_context_v1_accepted.pdf`

Source script:
- `/Users/teyian/Documents/paper1/make_figure3_topology_context_v1.py`

Notes:
- The accepted figure was generated on the AutoDL server from real result files and copied from Downloads into the project.
- A v2 dot-plot alternative was generated for panels B/C/D but was not selected because it looked too sparse for main-text submission.
- The accepted v1 uses grouped bars for topology-stratified Delta AUC/Delta AUPRC, a joint-cell Delta AUC distribution, and selected joint-cell bootstrap intervals.

### Figure 4

Status:
- Accepted PNG/PDF candidate after user visual inspection on 2026-06-04.
- Scientific content matches the current Figure 4 legend in `manuscript_bibm_v20_ieee_draft.tex`.

Files:
- PNG: `/Users/teyian/Documents/paper1/figure_drafts/Figure4_prediction_transitions_v2_accepted.png`
- PDF: `/Users/teyian/Documents/paper1/figure_drafts/Figure4_prediction_transitions_v2_accepted.pdf`

Source script:
- `/Users/teyian/Documents/paper1/make_figure4_prediction_transitions_v2.py`

Notes:
- The accepted figure was generated on the AutoDL server from real result files and copied from Downloads into the project.
- Earlier Figure 4 attempts had overlapping legends/annotations and overly dark heatmap cells; those should not be used.
- The accepted version uses transition summary bars for panels A/B, row-normalized heatmaps for panels C/D, an FDR-scaled topology-feature dot plot for panel E, and a difficulty-matched effect-size plot for panel F.
- The final figure legend should explicitly state that, in panels C/D, red indicates the higher corrected group within each row and blue indicates the lower group; in panel E, blue indicates features higher in rescued false negatives, orange indicates features higher in suppressed false positives, and point size reflects FDR strength.

## Pending Main Figure Assets

- None.
