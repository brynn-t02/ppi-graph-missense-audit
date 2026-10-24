# Workshop revision and matched experiments — 24 October 2026

The accepted original title is restored: **Auditing PPI Graph Augmentation for Missense Variant Pathogenicity Prediction under Topology, Connectivity and Homology Constraints**.

- [Named main paper](BIBM2026_workshop_named_revised.pdf) and [anonymous main paper](BIBM2026_workshop_anonymous_revised.pdf): 8 pages each.
- [Named supplement](BIBM2026_supplementary_named_revised.pdf) and [anonymous supplement](BIBM2026_supplementary_anonymous_revised.pdf): 13 pages each.
- [Named manuscript source](source/main.tex), [anonymous source](source/main_anonymous.tex), [supplement source](source/Supplementary_Information_revised.md), [figure generators](source/figure_scripts/).
- [Experiment code and protocol](reproducibility/code/), [per-run predictions, metrics and audits](reproducibility/outputs/), [setup/training/analysis logs](reproducibility/logs/).
- [Revision conclusions and limitations](REVISION_NOTES.md).

## Publication and snapshot provenance

This dated directory publishes the 24 October manuscript and compact matched-experiment materials, in addition to the already public 8 and 23 October snapshots. The PDFs, editable manuscript and numerical evidence are preserved byte-for-byte from the completed delivery, which preceded this GitHub synchronization. Their statements that the new additions were not yet uploaded, and the publication flags in the delivery quality report, describe that earlier moment. **They are superseded for availability by this public directory.** Publication adds no new experimental result. No conference-system submission or reviewer communication was performed.

`DELIVERY_PROVENANCE.json` records the original SHA256 of every copied delivery file. `SHA256SUMS.txt` covers the entire public snapshot relative to this directory. Private review emails, reviewer-response drafts, author correspondence and local task records are not published. Anonymous PDFs omit author identifiers; the public repository also contains the authorized named versions and is therefore not an anonymous hosting service.

## What was actually run

An isolated Python 3.10.8 / torch 2.1.2+cu118 environment was restored from the supplied historical exports; 138 specified versions and runtime were checked. Four smoke cases passed. Six raw/full positive-control fits exactly reproduced historical recorded validation/test metrics and selected epochs.

48 classifiers were fitted: 18 original-split representations, 21 original-split fusion controls, and 9 homology-split fusion runs. Each arm uses classifier seeds 42–44, maximum 50 epochs, patience 5 and validation-AUC selection. Permutation controls use three graph permutations by three classifier seeds. New controls share the full encoder's self-loop convention, correcting a discovered difference in the historical controls. The old results remain separately labeled.

All 48 selected epochs, validation/test metrics and finite checkpoints were independently checked. Protein-clustered paired bootstrap uses 2,000 replicates of ensemble predictions and conditions on fitted models. Current complete-score test-subset checks do not constitute complete-case training or an independent cohort.

## Reproduction

The portable derived requirements and the raw environment exports are in `reproducibility/code/`. The raw freeze includes local build URLs and should not be installed verbatim. Executed scripts preserve original study paths to keep execution hashes verifiable; adapt their base paths on a different machine. Large provider resources, full model-input tensors, embeddings and model checkpoints are not distributed in this compact archive. Obtain ClinVar, STRING, CADD and AlphaMissense resources from their providers under the respective terms.

Compile `source/main.tex` or `source/main_anonymous.tex` from `source/` with an IEEEtran-compatible engine (e.g. Tectonic). Figure programs require Python, NumPy, pandas and Matplotlib. Supplement rendering requires ReportLab and included fonts:

```sh
python reproducibility/render_supplement.py --root /path/to/2026-10-24
python reproducibility/render_supplement.py --root /path/to/2026-10-24 --anonymous
```

This is a reproducibility archive with validated saved predictions, not a one-command raw-data-to-paper pipeline. No independent cohort, second network, trained GNN, independent subgroup confirmation or calibration experiment is claimed.
