# BioGRID-only supplementary addition — ready-to-use content

## Supplementary Methods X. Alternative-network comparison

We constructed an alternative graph from BioGRID release 5.0.262 (compiled 25 September 2026; October 2026 release). Interactions were retained when both organisms were human and the experimental-system type was physical; Affinity Capture-RNA and Protein-RNA records were excluded. Entrez identifiers were mapped through unique aliases to the frozen STRING node universe. Mapped self-interactions were removed and duplicate unordered pairs collapsed. The resulting graph contained 931,458 undirected pairs on the same 19,622 nodes, including 1,402 isolates. It covered 18,006 of 18,220 original test variants and 17,748 of 18,220 homology-split test variants by at least one edge; all test variants were retained in evaluation. Of the BioGRID pairs, 526,635 (56.54%) occurred in STRING; pair-set Jaccard similarity was 0.07251.

Each unordered pair was represented in both directions. STRING and BioGRID graph features used the same self-loop convention and propagation: A=(E+I)+(E+I)^T, S=D^(-1/2) A D^(-1/2), and (X+SX+S^2X)/3. Edges were not weighted by publication count or clinical labels. BioGRID node identities were checked against the actual frozen model-input mapping. Isolated nodes retained their raw-feature values within numerical tolerance. STRING contains evidence also represented in BioGRID, and the networks differ in density, coverage, release and evidence composition; this comparison is not evidence-independent replication. The physical category can include co-complex and proximity evidence rather than direct binding alone.

Nine BioGRID classifiers were fitted: original-split protein representations, original-split score fusion and homology-split score fusion, each with classifier seeds 42, 43 and 44. Comparator raw-ESM, STRING and score-only models used the same seeds and training configuration. The MLP used 512/128 hidden units, ReLU, dropout 0.3, class-weighted binary cross-entropy, Adam with learning rate 0.001 and weight decay 0.00001, batch size 1,024, a maximum of 50 epochs and patience five. Checkpoints were selected only by validation AUC. Representation input dimension was 1,280; fusion concatenated CADD, AlphaMissense and the AlphaMissense missingness indicator before the 1,280-dimensional protein representation. Score-only and fusion classifiers used the same hidden-layer architecture, while input-layer parameter counts differed. Missing scores were imputed using training medians, with no added z-score scaling. Dataset memberships, ESM features and homology assignments were unchanged, and graph-source selection did not use test outcomes.

Protein-only prediction was computed once per node in fixed-order CPU batches and mapped to its variant rows; fusion predictions remained variant-specific. Per-seed probabilities were converted to float64 before averaging. AUPRC denotes non-interpolated average precision. Means and sample standard deviations (ddof=1) describe individual classifiers; paired differences describe metrics after averaging the probabilities. Confidence intervals use 2,000 paired protein-cluster bootstrap resamples, retaining all variants and multiplicities of sampled proteins. They condition on the fitted models and fixed networks, exclude retraining and graph-sampling uncertainty, and are not adjusted for multiple comparisons.

## Supplementary Table X-A. Full three-seed network-comparison summaries

| Setting | Model | Fits | AUC (mean ± sample SD) | AUPRC (mean ± sample SD) |
|---|---|---:|---:|---:|
| Original representation | BioGRID FPP | 3 | 0.7592 ± 0.0133 | 0.6043 ± 0.0083 |
| Original representation | STRING FPP | 3 | 0.7950 ± 0.0016 | 0.6485 ± 0.0117 |
| Original representation | Raw ESM | 3 | 0.7835 ± 0.0063 | 0.6117 ± 0.0066 |
| Original fusion | CADD+AM | 3 | 0.9572 ± 0.0001 | 0.9072 ± 0.0002 |
| Original fusion | CADD+AM + BioGRID FPP | 3 | 0.9693 ± 0.0013 | 0.9344 ± 0.0038 |
| Original fusion | CADD+AM + STRING FPP | 3 | 0.9716 ± 0.0007 | 0.9409 ± 0.0017 |
| Original fusion | CADD+AM + raw ESM | 3 | 0.9682 ± 0.0011 | 0.9326 ± 0.0037 |
| Homology fusion | CADD+AM | 3 | 0.9755 ± 0.0001 | 0.8403 ± 0.0006 |
| Homology fusion | CADD+AM + BioGRID FPP | 3 | 0.9730 ± 0.0002 | 0.8265 ± 0.0013 |
| Homology fusion | CADD+AM + STRING FPP | 3 | 0.9752 ± 0.0007 | 0.8369 ± 0.0060 |
| Homology fusion | CADD+AM + raw ESM | 3 | 0.9727 ± 0.0003 | 0.8253 ± 0.0016 |

These rows describe the three-seed network comparison. They should not be substituted for separate single-fit results or treated as the metrics of ensemble-averaged predictions. CADD+AM denotes the architecture-matched score baseline.

## Supplementary Table X-B. BioGRID ensemble differences

| Setting | Comparison | ΔAUC (95% CI) | ΔAUPRC (95% CI) |
|---|---|---:|---:|
| Original representation | BioGRID FPP − Raw ESM | -0.02012 [-0.04803, +0.01005] | -0.00249 [-0.05396, +0.03630] |
| Original representation | BioGRID FPP − STRING FPP | -0.03079 [-0.06916, +0.00407] | -0.04371 [-0.08938, -0.00108] |
| Original fusion | CADD+AM + BioGRID FPP − CADD+AM | +0.01242 [+0.00341, +0.02100] | +0.02788 [+0.00701, +0.05019] |
| Original fusion | CADD+AM + BioGRID FPP − CADD+AM + raw ESM | +0.00092 [-0.00115, +0.00364] | +0.00105 [-0.00371, +0.00691] |
| Original fusion | CADD+AM + BioGRID FPP − CADD+AM + STRING FPP | -0.00228 [-0.00420, -0.00058] | -0.00650 [-0.01164, -0.00233] |
| Homology fusion | CADD+AM + BioGRID FPP − CADD+AM | -0.00204 [-0.00421, -0.00004] | -0.01176 [-0.03147, +0.00596] |
| Homology fusion | CADD+AM + BioGRID FPP − CADD+AM + raw ESM | +0.00051 [-0.00077, +0.00169] | +0.00177 [-0.00629, +0.01049] |
| Homology fusion | CADD+AM + BioGRID FPP − CADD+AM + STRING FPP | -0.00216 [-0.00357, -0.00090] | -0.01052 [-0.01884, -0.00182] |

All intervals are percentile 95% intervals from 2,000 finite paired protein resamples. The original test split contains 18,220 variants on 1,413 proteins (5,822 positives); the homology test split contains 18,220 variants on 3,438 proteins (1,924 positives). Positive prevalence therefore differs, limiting direct cross-split AUPRC comparison. Near-zero intervals do not establish equivalence.
