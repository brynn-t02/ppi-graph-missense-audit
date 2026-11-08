# Existing BioGRID experiment audit and minimal inclusion recommendation

## Recommendation

Use one three-row main-text table, with the rows Original representation, Original fusion and Homology fusion. Report BioGRID mean AUC/AUPRC ± sample SD, the ensemble AUC difference from the corresponding raw-ESM model, and (for fusion) the difference from the architecture-matched score-only model. Put all 11 comparator means and all eight BioGRID paired contrasts in the supplementary section below. This is sufficient for a second-network check without replacing the four original figures or introducing the November partial-ablation/homology-representation experiments. A new figure is optional; a table is more compact.

The additional comparison is on the original ClinVar benchmark, not a new cohort. The BioGRID results use nine already-completed fits. Their comparator rows use the existing October 24 matched raw/STRING/score fits, reported with the October 25 canonical inference convention. No November 8 fit or new statistic is used. A fair comparison cannot substitute the original manuscript’s single-fit points for those comparator ensembles.

## Main-text three-row proposal

| Setting | BioGRID AUC ± SD | BioGRID AUPRC ± SD | ΔAUC vs corresponding raw model (95% CI) | ΔAUC vs CADD+AM (95% CI) |
|---|---:|---:|---:|---:|
| Original representation | 0.7592 ± 0.0133 | 0.6043 ± 0.0083 | -0.02012 [-0.04803, +0.01005] | — |
| Original fusion | 0.9693 ± 0.0013 | 0.9344 ± 0.0038 | +0.00092 [-0.00115, +0.00364] | +0.01242 [+0.00341, +0.02100] |
| Homology fusion | 0.9730 ± 0.0002 | 0.8265 ± 0.0013 | +0.00051 [-0.00077, +0.00169] | -0.00204 [-0.00421, -0.00004] |

Table note: All models in each comparison use three classifier seeds (42–44), with the same training protocol and preprocessing. The raw comparator is raw ESM for representations and CADD+AlphaMissense plus raw ESM for fusion. The score comparator uses the same 512/128 hidden-layer architecture as the fusion models, but fewer input-layer parameters. Means and sample standard deviations summarize the individual fits; differences and paired protein-bootstrap 95% intervals use averaged probabilities. Complete comparator results are provided in Supplementary Table X. Intervals condition on the fitted models and fixed networks.

Use five decimals for intervals: the homology BioGRID-minus-score AUC upper limit is −0.00004427608878427994, which would appear as −0.0000 at four decimals. It is −0.00004 at five decimals. Its AUPRC interval crosses zero; do not claim a broadly stable harmful graph effect.

## Minimal main Methods wording

To assess dependence on network source, we replaced STRING with the mapped human physical-interaction graph from BioGRID 5.0.262, retaining all 19,622 nodes and the benchmark splits. STRING and BioGRID graph arms used the same fixed propagation convention; all comparator models within each setting used seeds 42–44, the same hidden-layer architecture and the same training settings. Per-seed summaries and ensemble differences with paired protein-bootstrap intervals are reported separately (Supplementary Table X).

## Minimal main Results wording

BioGRID propagation achieved mean AUC 0.7592 on the original split, compared with 0.7835 for raw ESM; the protein-bootstrap interval for the ensemble difference included zero. BioGRID fusion improved over the architecture-matched CADD+AlphaMissense baseline on the original split (ΔAUC +0.01242, 95% CI [+0.00341, +0.02100]), but its increments over raw-ESM fusion remained uncertain under both splits. These results indicate that the incremental contribution of graph context depends on the interaction network; they do not establish a benefit shared by both network sources.

## Interpretive guardrails

- The STRING network has 6,857,702 undirected pairs; BioGRID has 931,458. Of BioGRID pairs, 526,635 (56.54%) also occur in STRING. They differ in coverage, density and evidence composition, so this is not an independent-evidence replication or an isolated biological mechanism test.
- Original-split representation: BioGRID−raw AUC and AP intervals both cross zero. The point estimates are lower; do not claim equivalence or proven harm.
- Original-split fusion: BioGRID−score intervals are positive; BioGRID−raw-fusion intervals both cross zero; BioGRID−STRING-fusion intervals are negative for both metrics. Do not attribute all gain over scores to graph topology.
- Homology fusion: BioGRID−raw-fusion intervals both cross zero; BioGRID−STRING-fusion intervals are negative for both metrics; BioGRID−score AUC interval lies narrowly below zero while AP includes zero.
- Score-only and fusion controls match hidden architecture, training and preprocessing, not input-layer parameter counts. No graph attention/aggregation parameters were trained.
- Protein-only models assign the same score to all variants in the same protein. All test variants remain evaluated, including nodes without BioGRID edges.
- The experiment did not fit a BioGRID representation-only model on the homology split. Do not pair the November homology raw/STRING representation fits with nonexistent BioGRID counterparts.

## Attribution and source precision

CSV values were copied as strings from existing canonical summaries, preserving their stored precision. JSON point estimates and interval endpoints were extracted from existing `cluster_bootstrap.json`; no bootstrap was rerun. `source_index.json` records hashes and exact local source paths. The canonical extraction uses float32 CPU node-wise inference for protein-only models, then float64 seed averaging; this corrects batching/tie differences without refitting or changing selected checkpoints.

Original manuscript figures can remain their original fitted comparisons. Clarify their actual run counts in their captions and introduce the added table as a three-seed network comparison. Different fitted prediction sets and mean-vs-ensemble summaries need not have identical numbers. Avoid labels such as Historical/Current in the research narrative; use concrete terms such as single-fit comparison, three-seed network comparison and ensemble difference.
