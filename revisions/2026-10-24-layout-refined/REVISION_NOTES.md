# Changes relative to the PEAL revision

Baseline commit: `85c3f41a39a7da72ac9b65b1f0bb7a079aff0661`.

1. Rephrase the Methods D opening and add an optional break after the resource-name slash. This resolves the three extreme spacing lines without changing scientific scope.
2. State the already-used MMseqs settings: minimum sequence identity 0.40, coverage 0.50, coverage mode 0.
3. State the already-executed training loss and settings: class-weighted binary cross-entropy; Adam learning rate 10^-3 and weight decay 10^-5; batches of 1,024; training negative/positive ratio as positive weight.
4. State the existing external label rule: pathogenic/likely pathogenic positive, benign/likely benign negative, uncertain significance excluded.
5. Rebalance the anonymous final reference page using the IEEEtran reference-break hook.

The additions clarify existing methods; no scientific outcome, experiment, figure, table, reference or conclusion changed. Both main papers remain seven pages, including references. All previous dated snapshots are retained.
