# Supplementary Information

Title: Protein interaction graph augmentation for missense variant pathogenicity prediction is context-dependent and bounded by evaluation design

Document version: Supplementary Information v1, synchronized 2026-06-12

Linked manuscript: `manuscript_bibm_v20_ieee_draft.tex` / `manuscript_bibm_v20_ieee_draft.pdf`

## Notes on scope

This Supplementary Information supports the main manuscript with additional reconstruction details, split statistics, model comparisons, uncertainty estimates, topology-stratified summaries and negative-control audit information. It is not intended to reproduce every intermediate experiment chronologically. Full figure source data, topology-stratified metrics, corrected-sample statistics and fusion-control summaries are included in the accompanying compact source-data archive.

## Supplementary Methods

### Supplementary Methods 1. Dataset reconstruction and source files

We reconstructed a missense variant pathogenicity dataset from the ClinVar `variant_summary.txt.gz` table obtained from the official NCBI ClinVar FTP `tab_delimited` directory. We used the ClinVar variant-summary release dated 31 March 2024. The exact local/server analysis copy used in our rebuild pipeline was archived on 3 April 2026, and package checksums are provided for reproducibility. We retained germline GRCh38 single-nucleotide variant records and extracted protein-level HGVS annotations. Missense substitutions were identified using a strict three-letter amino-acid substitution pattern.

Pathogenic, Likely pathogenic and Pathogenic/Likely pathogenic records were grouped as pathogenic. Benign, Likely benign and Benign/Likely benign records were grouped as benign. Supported review status was defined as the exact ClinVar `ReviewStatus` strings `criteria provided, single submitter`, `criteria provided, multiple submitters, no conflicts`, `reviewed by expert panel`, and `practice guideline`; records outside this whitelist were excluded. Conflicting VariationIDs were removed, and duplicate VariationIDs were collapsed after conflict removal. The final mapped dataset contained 177,213 filtered, non-conflicting ClinVar missense variants with criteria-supported review status, including 127,953 benign and 49,260 pathogenic variants.

ClinVar gene symbols were mapped to STRING human protein nodes using upper-cased gene symbols and the STRING protein alias table. Only genes that mapped uniquely to a STRING protein present in the STRING links graph were retained. The reconstructed ClinVar-STRING dataset covered 14,123 unique variant-labelled STRING proteins embedded in a larger STRING node universe containing 19,622 proteins. The final variant tables retained genomic coordinates, reference and alternate VCF alleles, VariationID, binary label, gene symbol and mapped STRING protein identifier.

CADD and AlphaMissense scores were matched to ClinVar variants by GRCh38 genomic coordinates and reference/alternate alleles. CADD GRCh38 v1.7 whole-genome SNV scores were matched for 177,205 of 177,213 variants, leaving eight missing CADD rows. AlphaMissense hg38 scores were matched for 171,753 variants, leaving 5,460 missing AlphaMissense rows. CADD-only and AlphaMissense-only baselines used their corresponding matched scalar scores. CADD+AlphaMissense models used CADD PHRED, AlphaMissense pathogenicity and an AlphaMissense-missing indicator. Raw-ESM fusion concatenated these scalar features with the 1,280-dimensional raw ESM protein representation, whereas graph-fusion models concatenated the same scalar features with the 1,280-dimensional PPI-GCN representation. Follow-up graph-control fusions used the same scalar CADD+AlphaMissense features concatenated with real PPI-GCN, direct train-test edge-ablated PPI-GCN, or topology-permuted PPI-GCN representations.

### Supplementary Methods 2. Protein representations and graph propagation

Protein node features were generated with the ESM-2 checkpoint `facebook/esm2_t33_650M_UR50D`. Residue-level embeddings were mean-pooled to obtain one 1,280-dimensional vector per protein. Proteins longer than 1,022 amino acids were split into chunks and combined by length-weighted averaging. The final feature table covered all 19,622 STRING graph proteins, with no missing graph proteins.

The primary representation-propagation graph was built from STRING v12 human protein links and contained 19,622 nodes and 13,715,404 directed edge-index entries in the retained model input object before adding self-loops. PPI-GCN representations were generated using fixed two-hop GCN-style propagation rather than an end-to-end trainable graph neural network. Propagation used a binary adjacency derived from the retained edge list; STRING combined scores were not used as propagation weights. After adding reverse edges and self-loops, duplicate edges were coalesced. Let `A` denote this binary adjacency after symmetrization and self-loop addition, `D` its degree matrix and `X` the raw ESM feature matrix. We used symmetric normalization,

`A_hat = D^{-1/2} A D^{-1/2}`,

and formed the propagated representation as

`H = (X + A_hat X + A_hat^2 X) / 3`.

This operation produced one propagated representation per STRING node and did not use variant labels during propagation.

GraphSAGE-style and GAT-style propagated representations were generated as supplementary architecture controls. These controls were included to test whether the representation-level comparison was specific to the fixed PPI-GCN propagation variant. They are treated as architecture controls rather than as an architecture search, because the main contribution of the study is an audit of graph-augmentation gains rather than an optimized predictor design.

For topology profiling, we used a separate thresholded STRING graph based on combined score. The primary topology analyses used a combined-score threshold of 400, yielding 19,622 nodes and 929,472 graph edges. Degree, k-core, community membership, betweenness, clustering coefficient and community-boundary role were computed on this thresholded graph. Thus, the representation-propagation graph and the topology-description graph were related STRING-derived objects but not identical analysis objects.

For topology-permutation controls, the graph edges were relabelled by a random node permutation (`edge_index_perm = perm[edge_index]`), while ESM features, labels and train/validation/test variant assignments remained tied to the original protein identities. Variant features were then extracted from the propagated representation of each variant's original mapped STRING protein after propagation on the permuted graph. The graph edge count and unlabeled topology were preserved, but the original protein-to-neighbourhood correspondence was broken. Three graph permutations were combined with three downstream classifier seeds, and the downstream MLP was retrained for each permutation/seed pair.

### Supplementary Methods 3. Evaluation splits, homology-aware stress test and statistical analysis

The original evaluation split was constructed at the STRING protein level using seed 42 and an approximate 8:1:1 train, validation and test ratio. Variants assigned to the same STRING protein were kept in the same partition. This produced 141,864 training variants, 17,129 validation variants and 18,220 test variants, with no shared proteins across partitions. The corresponding numbers of unique proteins were 11,298, 1,412 and 1,413 in the train, validation and test sets. All proteins nevertheless remained embedded in the same full PPI graph, so train and test proteins could still be connected through PPI edges.

For homology-aware evaluation, STRING v12 human protein sequences were clustered with MMseqs using minimum sequence identity 0.40, coverage threshold 0.50 and cov-mode 0. Homologous clusters rather than individual proteins were assigned to train, validation and test partitions, with no protein or homology-cluster overlap across partitions. This split was used as a stricter stress test for reduced sequence-related sharing across partitions, not as a prevalence-matched replacement for the original protein-level split. It preserved the original row counts: 141,864 training variants, 17,129 validation variants and 18,220 test variants. The homology-aware split contained 7,370, 3,315 and 3,438 unique proteins in the train, validation and test sets, respectively, and was built from 9,731 protein clusters with mean cluster size 1.45 and maximum cluster size 251. Because test prevalence changed from 31.95% in the original split to 10.56% in the homology-aware split, AUPRC values should be interpreted within each split rather than directly compared across splits.

Variant-level pathogenicity was predicted with validation-selected multilayer perceptron classifiers. For representation-based and fusion models, each variant inherited the representation of its mapped STRING protein, and the classifier used two hidden layers with 512 and 128 units, ReLU activations, dropout 0.3 and a single-logit output. Low-dimensional external-score-only baselines used a smaller 32/16 classifier with dropout 0.2 unless explicitly marked as architecture-matched. Models were trained with BCEWithLogitsLoss and positive-class weighting, using Adam with learning rate 1e-3, weight decay 1e-5, batch size 1024, a maximum of 50 epochs and early stopping after five epochs without validation AUC improvement. The positive-class weight was computed from the training partition as `N_negative / N_positive`. The best epoch was selected by validation AUC.

In the retained MLP pipeline, no additional z-score standardization was applied to CADD, AlphaMissense or embedding-derived inputs. Missing CADD and AlphaMissense values were imputed using training-set medians, and binary missingness indicators were retained as binary indicators. The same feature construction was used across raw ESM, edge-ablated PPI-GCN, topology-permuted PPI-GCN and real PPI-GCN fusion controls. Repeated classifier comparisons used seeds 42, 43 and 44 unless otherwise stated; topology-permuted fusion combined three sampled graph permutations with these classifier seeds. These details mean that the architecture-matched fusion controls share downstream classifier capacity and feature construction, while alternative feature-scaling choices were not separately optimized in this audit.

Neighbour-risk, neighbour pathogenic-rate and neighbour training-variant-count features were computed from training-set labels only. Validation and test labels were not used to construct these topology-context features. When a test protein had multiple labelled training-neighbour variants, the neighbour pathogenic rate was variant-weighted rather than protein-weighted, so proteins with many labelled neighbouring variants contributed proportionally to the local training-label context.

Uncertainty was evaluated using 1,000 variant-level bootstrap resamples and protein-clustered bootstrap resampling. Variant-level bootstrap resampled individual test variants. Protein-clustered bootstrap resampled test proteins as clusters and retained all variants belonging to each sampled protein, thereby accounting for repeated variants from the same protein. Confidence intervals were reported as empirical 95% bootstrap intervals. Feature-level corrected-sample comparisons used Benjamini-Hochberg false-discovery-rate correction when multiple topology features were tested.

### Supplementary Methods 4. Negative controls and provenance caveats

Exploratory internal controls included random raw features, random-feature graph propagation, label-shuffling controls and feature-shuffled GCN propagation. Because complete provenance was not retained for all early controls, only the re-materialized feature-shuffled PPI-GCN control is formally reported here. This negative-control analysis assessed whether graph propagation could produce apparent predictive signal independently of meaningful protein-topology correspondence. The feature-shuffled GCN control was rerun on 4 June 2026 after an earlier audit found that one intermediate file was not retained. In this rerun, node features were randomly permuted across STRING proteins before GCN-style propagation, the propagated embeddings were saved, and the same downstream classifier was trained on the shuffled-feature representations. The rerun achieved validation AUC 0.7080, validation AUPRC 0.4241, test AUC 0.5868 and test AUPRC 0.3707, reproducing the earlier negative-control pattern and supporting its use as supplementary evidence.

Older random-feature and label-shuffling summaries are useful as robustness context, but they should be interpreted cautiously unless their intermediate files and exact provenance are included in the final supplementary package. We therefore do not use these controls as central main-text evidence. Their role is to clarify that graph-based learning can capture non-trivial structure even under control settings, reinforcing the need to distinguish protein-topology correspondence, dataset label structure and graph-propagation effects.

## Supplementary Tables

### Supplementary Table 1. Dataset, split and graph summary

| Object | Train | Validation | Test | Total or value | Notes |
|---|---:|---:|---:|---:|---|
| Original split variants | 141,864 | 17,129 | 18,220 | 177,213 | Protein-level split |
| Original split benign variants | 102,818 | 12,737 | 12,398 | 127,953 | Label 0 |
| Original split pathogenic variants | 39,046 | 4,392 | 5,822 | 49,260 | Label 1 |
| Original split unique proteins | 11,298 | 1,412 | 1,413 | 14,123 labelled proteins | No exact protein overlap |
| Homology-aware split variants | 141,864 | 17,129 | 18,220 | 177,213 | Homology-cluster split |
| Homology-aware split benign variants | 96,212 | 15,445 | 16,296 | 127,953 | Label 0 |
| Homology-aware split pathogenic variants | 45,652 | 1,684 | 1,924 | 49,260 | Label 1 |
| Homology-aware split unique proteins | 7,370 | 3,315 | 3,438 | 14,123 labelled proteins | No exact protein or cluster overlap |
| Homology-aware clusters | NA | NA | NA | 9,731 | Mean size 1.45; maximum size 251 |
| Representation graph | NA | NA | NA | 19,622 nodes; 13,715,404 directed edge-index entries | Used for representation propagation |
| Topology-analysis graph | NA | NA | NA | 19,622 nodes; 929,472 edges | STRING combined-score threshold 400 |
| CADD matching | NA | NA | NA | 177,205 matched; 8 missing | GRCh38 v1.7 whole-genome SNV scores |
| AlphaMissense matching | NA | NA | NA | 171,753 matched; 5,460 missing | hg38 AlphaMissense table |

### Supplementary Table 2. Representation and architecture-control performance

The original-split raw ESM and PPI-GCN rows report the retained reference run used for paired graph-ablation and bootstrap analyses in Figure 1. Repeated classifier seeds showed the same direction with a smaller mean AUC difference and are reported in the main text.

| Evaluation setting | Model | Validation AUC | Validation AUPRC | Test AUC | Test AUPRC | Status |
|---|---|---:|---:|---:|---:|---|
| Original split | Raw ESM | 0.7814 | 0.5569 | 0.7790 | 0.6074 | Retained reference run |
| Original split | PPI-GCN | 0.8237 | 0.5821 | 0.7997 | 0.6496 | Retained reference run |
| Original split | Raw ESM repeated classifiers | NA | NA | 0.7835 +/- 0.0051 | 0.6116 +/- 0.0054 | Three classifier seeds |
| Original split | PPI-GCN repeated classifiers | NA | NA | 0.7950 +/- 0.0013 | 0.6484 +/- 0.0096 | Three classifier seeds |
| Original split | GraphSAGE-style propagation | 0.8008 | 0.6136 | 0.7950 | 0.6044 | Architecture control |
| Original split | GAT-style propagation | 0.8005 | 0.5876 | 0.7949 | 0.6157 | Architecture control |
| Original split | Degree-only baseline | 0.7298 | 0.4228 | 0.6377 | 0.3901 | Control |
| Original split | Feature-shuffled PPI-GCN rerun | 0.7080 | 0.4241 | 0.5868 | 0.3707 | Rerun verified on 2026-06-04 |
| Homology-aware split | Raw ESM | 0.7315 | 0.2171 | 0.7248 | 0.2418 | Stress-test split |
| Homology-aware split | PPI-GCN | 0.7645 | 0.2450 | 0.7606 | 0.2768 | Stress-test split |

### Supplementary Table 3. External predictor and fusion-control performance

The main fusion-control comparison uses repeated classifier runs where available. Older single-tool rows are retained only as reference context and are not mixed into the matched Figure 2 control panel.

#### Supplementary Table 3A. Original split

| Model | n | Val AUC | Val AUPRC | Test AUC | Test AUPRC |
|---|---:|---:|---:|---:|---:|
| CADD | 1 | 0.9237 | 0.7855 | 0.9207 | 0.8049 |
| AM | 1 | 0.9597 | 0.9038 | 0.9470 | 0.8899 |
| CADD+AM (arch.) | 3 | 0.9667 | 0.9126 | 0.9572 +/- 0.00005 | 0.9068 +/- 0.0003 |
| + raw ESM | 3 | 0.9686 | 0.9160 | 0.9680 +/- 0.0007 | 0.9317 +/- 0.0029 |
| + edge-ablated GCN | 3 | 0.9728 | 0.9257 | 0.9708 +/- 0.0008 | 0.9395 +/- 0.0007 |
| + permuted GCN | 9 | 0.9718 | 0.9232 | 0.9694 +/- 0.0008 | 0.9355 +/- 0.0015 |
| + real PPI-GCN | 3 | 0.9732 | 0.9253 | 0.9718 +/- 0.0007 | 0.9415 +/- 0.0011 |

Notes for Table 3A: AM denotes AlphaMissense. "Arch." denotes the architecture-matched CADD+AlphaMissense baseline using the same 512/128 MLP capacity as the fusion controls. The rows beginning with "+" add the listed protein representation to the CADD+AM feature set. Compared with the architecture-matched CADD+AM baseline, raw ESM fusion increased test AUC/AUPRC by +0.0108/+0.0249, edge-ablated PPI-GCN fusion by +0.0136/+0.0327, topology-permuted PPI-GCN fusion by +0.0122/+0.0286 and real PPI-GCN fusion by +0.0146/+0.0347. The topology-permuted row combines three graph permutations and three classifier seeds.

#### Supplementary Table 3B. Homology-aware split

| Model | n | Val AUC | Val AUPRC | Test AUC | Test AUPRC |
|---|---:|---:|---:|---:|---:|
| CADD | 1 | 0.9506 | 0.6713 | 0.9504 | 0.6895 |
| AM | 1 | 0.9614 | 0.7945 | 0.9632 | 0.8029 |
| CADD+AM (arch.) | 3 | 0.9739 | 0.8362 | 0.9754 +/- 0.00005 | 0.8401 +/- 0.0002 |
| + raw ESM | 3 | 0.9715 | 0.8228 | 0.9728 +/- 0.0003 | 0.8249 +/- 0.0016 |
| + PPI-GCN | 1 | 0.9732 | 0.8273 | 0.9741 | 0.8333 |

Notes for Table 3B: the homology-aware graph-fusion row is a single retained point estimate. It is reported as a stress-test result, not as a fully symmetric repeated-ensemble comparison. Relative to the repeated-run architecture-matched CADD+AlphaMissense mean, the retained graph-fusion point estimate was -0.0013 AUC and -0.0068 AUPRC.

### Supplementary Table 4. Uncertainty and repeated-run summary

#### Supplementary Table 4A. Raw ESM versus PPI-GCN uncertainty

| Bootstrap setting | Metric | Delta | 95% CI low | 95% CI high | n |
|---|---|---:|---:|---:|---:|
| Original variant bootstrap | AUC | +0.0208 | +0.0174 | +0.0240 | 1,000 |
| Original variant bootstrap | AUPRC | +0.0420 | +0.0353 | +0.0484 | 1,000 |
| Original protein-clustered bootstrap | AUC | +0.0207 | -0.0080 | +0.0527 | 1,000 |
| Original protein-clustered bootstrap | AUPRC | +0.0421 | -0.0021 | +0.0798 | 1,000 |

#### Supplementary Table 4B. Original-split strong-predictor fusion-control uncertainty

Comparison codes: C0, real PPI-GCN fusion versus architecture-matched CADD+AlphaMissense; C1, real PPI-GCN fusion versus raw ESM fusion; C2, real PPI-GCN fusion versus edge-ablated PPI-GCN fusion; C3, real PPI-GCN fusion versus topology-permuted PPI-GCN fusion.

| ID | Metric | Delta | 95% CI low | 95% CI high | n |
|---|---|---:|---:|---:|---:|
| C0 | AUC | +0.0143 | +0.0062 | +0.0224 | 1,000 |
| C0 | AUPRC | +0.0341 | +0.0167 | +0.0566 | 1,000 |
| C1 | AUC | +0.0038 | +0.0005 | +0.0075 | 1,000 |
| C1 | AUPRC | +0.0093 | +0.0032 | +0.0167 | 1,000 |
| C2 | AUC | +0.0009 | -0.0009 | +0.0027 | 1,000 |
| C2 | AUPRC | +0.0017 | -0.0018 | +0.0052 | 1,000 |
| C3 | AUC | +0.0019 | -0.0005 | +0.0052 | 1,000 |
| C3 | AUPRC | +0.0050 | -0.0002 | +0.0112 | 1,000 |

#### Supplementary Table 4C. Earlier retained-reference fusion bootstrap retained as context

| Comparison | Metric | Delta | 95% CI low | 95% CI high | n |
|---|---|---:|---:|---:|---:|
| Retained graph fusion vs retained CADD+AM | AUC | +0.0133 | +0.0119 | +0.0146 | 1,000 |
| Retained graph fusion vs retained CADD+AM | AUPRC | +0.0310 | +0.0268 | +0.0350 | 1,000 |

Table 4A shows that variant-level bootstrap supports positive retained-reference PPI-GCN gains, whereas protein-clustered bootstrap widens uncertainty. Table 4B shows that the real PPI-GCN fusion margin over raw ESM fusion remained positive under protein-clustered paired bootstrap, but the smaller margins over edge-ablated and topology-permuted graph controls crossed zero. Table 4C is retained as an earlier retained-reference context analysis and is not architecture-matched.

The homology-aware strong-predictor bootstrap row retained in the source-data package comes from an earlier retained reference-run comparison. It should not be interpreted as a fully architecture-matched exp48 ensemble bootstrap. In the main manuscript, the homology-aware strong-predictor result is therefore described as a stress-test pattern with an explicitly retained graph-fusion point estimate rather than a fully symmetric repeated-run bootstrap comparison.

### Supplementary Table 5. Topology-stratified PPI-GCN gain over raw ESM

| Stratum | n | Positive rate | Delta AUC | Delta AUPRC |
|---|---:|---:|---:|---:|
| Degree-low | 2,795 | 0.0426 | +0.0314 | +0.0109 |
| Degree-mid | 4,762 | 0.2089 | +0.0339 | +0.0280 |
| Degree-high | 10,663 | 0.4415 | -0.0178 | -0.0021 |
| K-core-low | 2,847 | 0.0425 | +0.0297 | +0.0116 |
| K-core-mid | 5,685 | 0.3078 | +0.0775 | +0.1049 |
| K-core-high | 9,688 | 0.4078 | -0.0276 | +0.0015 |
| Risk-low | 6,323 | 0.1056 | +0.0641 | +0.0631 |
| Risk-mid | 5,829 | 0.2882 | +0.0265 | +0.0515 |
| Risk-high | 6,019 | 0.5772 | +0.0284 | +0.0270 |
| Betweenness-low | 3,050 | 0.1393 | +0.0614 | +0.0769 |
| Betweenness-mid | 4,486 | 0.1899 | +0.0522 | +0.0501 |
| Betweenness-high | 10,684 | 0.4254 | -0.0145 | +0.0027 |
| Clustering-low | 8,234 | 0.2888 | -0.0036 | -0.0099 |
| Clustering-mid | 5,950 | 0.4198 | +0.0544 | +0.0444 |
| Clustering-high | 4,036 | 0.2344 | +0.0348 | +0.0941 |
| Boundary-like | 6,198 | 0.3527 | +0.0182 | +0.0200 |
| Boundary-mixed | 6,598 | 0.2963 | +0.0140 | +0.0503 |
| Boundary-internal | 5,397 | 0.3111 | +0.0383 | +0.0151 |

Selected exploratory joint topology cells:

Cell codes: J0, mid k-core / high betweenness; J1, mid k-core / mid clustering; J2, high neighbour risk / mid k-core; J3, high k-core / low clustering.

| ID | n | Pos. rate | Delta AUC | Delta AUPRC | AUC CI |
|---|---:|---:|---:|---:|---|
| J0 | 1,950 | 0.4297 | +0.1429 | +0.1533 | +0.1246 to +0.1617 |
| J1 | 2,089 | 0.4045 | +0.1408 | +0.1428 | +0.1234 to +0.1581 |
| J2 | 1,904 | 0.5767 | +0.1274 | +0.0983 | +0.1045 to +0.1499 |
| J3 | 4,791 | 0.3870 | -0.0532 | -0.0480 | -0.0611 to -0.0457 |

Full topology maps are provided in the accompanying source-data archive rather than printed in full in the PDF supplement.

These joint cells were selected from the same test-set topology map used for estimation. Their bootstrap intervals describe within-cell sampling uncertainty but do not correct for the selection step; therefore, they should be interpreted as exploratory gain profiles rather than independently validated topology-defined subgroups.

### Supplementary Table 6. Corrected-sample and difficulty-matched statistics

Corrected-sample group counts for raw ESM to PPI-GCN:

All strong-predictor transition counts in this table correspond to the exploratory retained-reference model pair used in Figure 4, not to the architecture-matched repeated ensembles used in Figure 2.

| Group | n | Positive rate |
|---|---:|---:|
| both correct | 11,875 | 0.3270 |
| both wrong | 3,965 | 0.2678 |
| improved/corrected | 1,436 | 0.5829 |
| degraded | 944 | 0.0424 |
| false negatives rescued | 837 | 1.0000 |
| false positives suppressed | 599 | 0.0000 |

Corrected-sample group counts for retained-reference CADD+AlphaMissense to retained-reference graph fusion:

| Group | n | Positive rate |
|---|---:|---:|
| both correct | 15,981 | 0.3173 |
| both wrong | 1,298 | 0.3274 |
| improved/corrected | 666 | 0.3333 |
| degraded | 275 | 0.3818 |
| false negatives rescued | 222 | 1.0000 |
| false positives suppressed | 444 | 0.0000 |

Selected false-positive-suppressed versus false-negative-rescued topology contrasts for raw ESM to PPI-GCN:

| Feature | FP mean | FN mean | Diff | FDR |
|---|---:|---:|---:|---:|
| degree bucket | 0.8698 | 1.9833 | -1.1135 | 4.48e-188 |
| k-core bucket | 0.8381 | 1.1254 | -0.2874 | 3.12e-22 |
| neighbour risk bucket | 0.5726 | 1.6941 | -1.1215 | 2.74e-129 |
| neighbour pathogenic rate | 0.2717 | 0.4329 | -0.1612 | 5.95e-90 |
| betweenness bucket | 0.9482 | 1.8530 | -0.9048 | 1.82e-120 |
| clustering coefficient | 0.2534 | 0.2022 | +0.0512 | 1.16e-07 |
| boundary role bucket | 1.1519 | 0.3321 | +0.8198 | 2.12e-87 |

Selected false-positive-suppressed versus false-negative-rescued topology contrasts for retained-reference CADD+AlphaMissense to retained-reference graph fusion:

| Feature | FP mean | FN mean | Diff | FDR |
|---|---:|---:|---:|---:|
| degree bucket | 1.5541 | 1.8198 | -0.2658 | 2.60e-06 |
| k-core bucket | 1.5518 | 1.6036 | -0.0518 | 0.8373 |
| neighbour risk bucket | 0.7793 | 1.7523 | -0.9730 | 4.34e-55 |
| neighbour pathogenic rate | 0.3276 | 0.4955 | -0.1679 | 1.57e-56 |
| betweenness bucket | 1.5541 | 1.7252 | -0.1712 | 2.96e-04 |
| clustering coefficient | 0.1807 | 0.2176 | -0.0368 | 1.64e-14 |
| boundary role bucket | 0.9685 | 0.7207 | +0.2477 | 3.81e-05 |

Difficulty-matched rescued false-negative contrasts:

| Feature | Corrected subset mean | Persistent error mean | Hedges' g | FDR |
|---|---:|---:|---:|---:|
| Degree bucket | 1.9833 | 1.5546 | +0.9675 | 0.0011 |
| K-core bucket | 1.1254 | 1.5358 | -0.8441 | 0.0011 |
| Neighbour risk bucket | 1.6941 | 1.0904 | +0.8163 | 0.0011 |
| Neighbour pathogenic rate | 0.4329 | 0.3793 | +0.5259 | 0.0011 |
| Betweenness bucket | 1.8530 | 1.5348 | +0.5923 | 0.0011 |
| Clustering coefficient | 0.2022 | 0.2322 | -0.4011 | 0.0011 |
| Boundary role bucket | 0.3321 | 0.9953 | -0.8000 | 0.0011 |
| Within-community neighbour fraction | 0.5655 | 0.7093 | -0.7950 | 0.0011 |

These are association and stratification results, not causal mechanism proof. The final interpretation should use wording such as "differ" or "are associated with" rather than "drive" or "cause".

### Supplementary Table 7. Negative-control and provenance audit

The formal supplementary table retains the negative-control result that was rerun and re-materialized during the final audit. Older random-feature and label-shuffling summaries remain internal robustness context unless their full provenance is rebuilt and included in a later archive.

| Control | Setting | n repeats | Validation AUC | Validation AUPRC | Test AUC | Test AUPRC | Provenance and use |
|---|---|---:|---:|---:|---:|---:|---|
| Feature-shuffled PPI-GCN rerun | Original split | 1 | 0.7080 | 0.4241 | 0.5868 | 0.3707 | Refreshed and retained; use as supplementary negative-control evidence |

The omitted older random-feature and label-shuffling summaries are scientifically useful as cautionary context because they suggest that graph structure and dataset label organization can produce non-trivial prediction patterns under some controls. However, without the full intermediate files in the final source-data package, they should not be presented as central evidence or as a standalone biological interpretation.

## Source-data archive status

The compact source-data archive accompanying this supplement contains figure source data for Figures 1-4, topology-stratified metrics, joint topology maps, corrected-sample statistics, fusion-control summaries, bootstrap outputs and checksum files. Large third-party resources, including provider-controlled CADD and AlphaMissense score files, are not redistributed and should be obtained from their original providers.
