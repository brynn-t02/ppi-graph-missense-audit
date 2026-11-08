# Supplementary Information

Title: Auditing PPI Graph Augmentation for Missense Variant Pathogenicity Prediction under Topology, Connectivity and Homology Constraints

## Notes on scope

This Supplementary Information provides reconstruction details, split statistics, model comparisons, uncertainty estimates, topology-stratified summaries and negative-control results. Tables 1–7 document the analyses shown in Figures 1–4, and Table 8 identifies their run counts and settings. BioGRID second-network validation and its matched baselines are described separately in Methods 5 and Table 9. Figure source data, topology-stratified metrics, corrected-sample statistics and fusion-control summaries are available in the public code and data archive.

## Supplementary Methods

### Supplementary Methods 1. Dataset reconstruction and source files

We reconstructed a missense variant pathogenicity dataset from the ClinVar `variant_summary.txt.gz` table obtained from the official NCBI ClinVar FTP `tab_delimited` directory. The precise ClinVar content-release date could not be verified from the saved records. Checksums and sample memberships identify the frozen processed benchmark; these records do not establish temporal independence. We retained germline GRCh38 single-nucleotide variant records and extracted protein-level HGVS annotations. Missense substitutions were identified using a strict three-letter amino-acid substitution pattern.

Pathogenic, Likely pathogenic and Pathogenic/Likely pathogenic records were grouped as pathogenic. Benign, Likely benign and Benign/Likely benign records were grouped as benign. Supported review status was defined as the exact ClinVar `ReviewStatus` strings `criteria provided, single submitter`, `criteria provided, multiple submitters, no conflicts`, `reviewed by expert panel`, and `practice guideline`; records outside this whitelist were excluded. Conflicting VariationIDs were removed, and duplicate VariationIDs were collapsed after conflict removal. The final mapped dataset contained 177,213 filtered, non-conflicting ClinVar missense variants with criteria-supported review status, including 127,953 benign and 49,260 pathogenic variants.

ClinVar gene symbols were mapped to STRING human protein nodes using upper-cased gene symbols and the STRING protein alias table. Only genes that mapped uniquely to a STRING protein present in the STRING links graph were retained. Mapping was performed at the protein-identifier level; isoform-specific effects and transcript-level ambiguity were not modeled separately. The reconstructed ClinVar-STRING dataset covered 14,123 unique variant-labelled STRING proteins embedded in a larger STRING node universe containing 19,622 proteins. The final variant tables retained genomic coordinates, reference and alternate VCF alleles, VariationID, binary label, gene symbol and mapped STRING protein identifier.

CADD and AlphaMissense scores were matched to ClinVar variants by GRCh38 genomic coordinates and reference/alternate alleles. CADD GRCh38 v1.7 whole-genome SNV scores were matched for 177,205 of 177,213 variants, leaving eight missing CADD rows. AlphaMissense hg38 scores were matched for 171,753 variants, leaving 5,460 missing AlphaMissense rows. CADD-only and AlphaMissense-only baselines used their corresponding matched scalar scores. CADD+AlphaMissense models used CADD PHRED, AlphaMissense pathogenicity and an AlphaMissense-missing indicator. Raw-ESM fusion concatenated these scalar features with the 1,280-dimensional raw ESM protein representation, whereas graph-fusion models concatenated the same scalar features with the 1,280-dimensional PPI-GCN representation. Follow-up graph-control fusions used the same scalar CADD+AlphaMissense features concatenated with real PPI-GCN, direct train-test edge-ablated PPI-GCN, or topology-permuted PPI-GCN representations.

### Supplementary Methods 2. Protein representations and graph propagation

Protein node features were generated with the Facebook ESM-2 t33 650M UR50D checkpoint. Residue-level embeddings were mean-pooled to obtain one 1,280-dimensional vector per protein. Proteins longer than 1,022 amino acids were split into chunks and combined by length-weighted averaging. The final feature table covered all 19,622 STRING graph proteins, with no missing graph proteins.

The primary representation-propagation graph was built from STRING v12 human protein links and contained 19,622 nodes and 13,715,404 directed edge-index entries in the retained model input object before adding self-loops. PPI-GCN representations were generated using fixed two-hop GCN-style propagation rather than an end-to-end trainable graph neural network. This design evaluates fixed graph-based smoothing while training only the downstream classifier. The full-graph operator adds self-loops before symmetrization and sums duplicate contributions: $A=(E+I)+(E+I)^T$, where $E$ is the edge-index count matrix. STRING combined scores are not propagation weights. Direct-edge-ablation and topology-permutation controls instead add one self-loop after symmetrization, using $A=E+E^T+I$. Thus these controls differ in self-loop weighting as well as graph connectivity. Let $D$ denote the weighted degree matrix of the applicable adjacency and $X$ the raw ESM feature matrix. We used symmetric normalization,

$$
\hat{A}=D^{-\frac{1}{2}} A D^{-\frac{1}{2}},
$$

and formed the propagated representation as

$$
H=\frac{1}{3}\left(X+\hat{A}X+\hat{A}^{2}X\right).
$$

This operation produced one propagated representation per STRING node and did not use variant labels during propagation. Held-out nodes and features remained in the graph, giving transductive feature access. Each embedding-only model uses identical inputs for variants mapped to the same protein and therefore does not distinguish substitutions within that protein. These fixed-propagation results do not establish the performance of trained graph neural networks.

Two supplementary fixed operators were used. Fixed mean propagation applies row-normalized two-hop mean aggregation with residual averaging. Fixed cosine-weighted propagation applies two rounds of cosine-softmax-weighted neighbour aggregation with residual averaging, including self-neighbours. Both operate without learned aggregation or attention parameters; they are fixed operator controls, not trained GraphSAGE or GAT models.

For topology profiling, we used a separate thresholded STRING graph based on combined score. The primary topology analyses used a combined-score threshold of 400, yielding 19,622 nodes and 929,472 graph edges. Degree, k-core, community membership, betweenness, clustering coefficient and community-boundary role were computed on this thresholded graph. The representation-propagation graph and the topology-description graph were related STRING-derived objects but not identical analysis objects.

For topology-permutation controls, the graph edges were relabelled by a random node permutation (`edge_index_perm = perm[edge_index]`), while ESM features, labels and train/validation/test variant assignments remained tied to the original protein identities. Variant features were then extracted from the propagated representation of each variant's original mapped STRING protein after propagation on the permuted graph. The graph edge count and unlabeled topology were preserved, but the original protein-to-neighbourhood correspondence was broken. Three graph permutations were combined with three downstream classifier seeds, and the downstream MLP was retrained for each permutation/seed pair.

### Supplementary Methods 3. Evaluation splits, homology-aware stress test and statistical analysis

The original evaluation split was constructed at the STRING protein level using seed 42 and an approximate 8:1:1 train, validation and test ratio. Variants assigned to the same STRING protein were kept in the same partition. This produced 141,864 training variants, 17,129 validation variants and 18,220 test variants, with no shared proteins across partitions. The corresponding numbers of unique proteins were 11,298, 1,412 and 1,413 in the train, validation and test sets. All proteins remained embedded in the same full PPI graph, so train and test proteins could still be connected through PPI edges.

For homology-aware evaluation, STRING v12 human protein sequences were clustered with MMseqs using minimum sequence identity 0.40, coverage threshold 0.50 and cov-mode 0. Homologous clusters rather than individual proteins were assigned to train, validation and test partitions, with no protein or homology-cluster overlap across partitions. This split was used as a stricter stress test for reduced sequence-related sharing across partitions, not as a prevalence-matched replacement for the original protein-level split. It preserved the original row counts: 141,864 training variants, 17,129 validation variants and 18,220 test variants. The homology-aware split contained 7,370, 3,315 and 3,438 unique proteins in the train, validation and test sets, respectively, and was built from 9,731 protein clusters with mean cluster size 1.45 and maximum cluster size 251. Because test prevalence changed from 31.95% in the original split to 10.56% in the homology-aware split, AUPRC values should be interpreted within each split rather than directly compared across splits.

Variant-level pathogenicity was predicted with validation-selected multilayer perceptron classifiers. For representation-based and fusion models, each variant inherited the representation of its mapped STRING protein, and the classifier used two hidden layers with 512 and 128 units, ReLU activations, dropout 0.3 and a single-logit output. Low-dimensional external-score-only baselines used a smaller 32/16 classifier with dropout 0.2, while architecture-matched CADD+AlphaMissense and fusion-control comparisons used the same 512/128 hidden-layer architecture. Their input-layer parameter counts differed because their input dimensions differed. The main-model protocol used BCEWithLogitsLoss and positive-class weighting, Adam with learning rate 1e-3, weight decay 1e-5, batch size 1024, a maximum of 50 epochs and early stopping after five consecutive validation checks without validation AUC improvement. Representation-only direct-edge-ablation, partial-ablation and permutation controls used a 30-epoch maximum and different classifier seeds, as detailed in Table 8. The positive-class weight was computed from the training partition as `N_negative / N_positive`. The best epoch was selected by validation AUC.

In the evaluated MLP configurations, no additional z-score standardization was applied to CADD, AlphaMissense or embedding-derived inputs. Missing CADD and AlphaMissense values were imputed using training-set medians, and binary missingness indicators were retained as binary indicators. The same preprocessing and feature construction were used across raw ESM, edge-ablated PPI-GCN, topology-permuted PPI-GCN and real PPI-GCN fusion controls. The repeated positive and fusion comparisons used seeds 42, 43 and 44; Table 8 lists the representation-control seed exceptions. Topology-permuted fusion combined three sampled graph permutations with these classifier seeds. Fusion comparisons used the same hidden-layer architecture and preprocessing, not identical input-layer parameter counts or a common graph-control self-loop convention. Absolute fusion performance may depend on feature-scaling choices that were not separately optimized.

Neighbour-risk, neighbour pathogenic-rate and neighbour training-variant-count features were computed from training-set labels only. Validation and test labels were not used to construct these topology-context features. When a test protein had multiple labelled training-neighbour variants, the neighbour pathogenic rate was variant-weighted rather than protein-weighted, so proteins with many labelled neighbouring variants contributed proportionally to the local training-label context.

Uncertainty was evaluated using 1,000 variant-level bootstrap resamples and protein-clustered bootstrap resampling. Variant-level bootstrap resampled individual test variants. Protein-clustered bootstrap resampled test proteins as clusters and retained all variants belonging to each sampled protein, thereby accounting for repeated variants from the same protein. Confidence intervals were reported as empirical 95% bootstrap intervals conditional on the fitted predictions and sampled graphs; they do not include retraining or new graph-sampling uncertainty. AUPRC denotes non-interpolated average precision. The corrected-sample feature tests treated variants as independent observations despite repeated variants sharing protein-level features. Their p values, false-discovery-rate values and significance encodings do not support valid inferential claims. Group means and Hedges' g are reported descriptively in Table 6. Edge removal also changes normalized propagation weights and retains indirect cross-partition paths, so control differences do not isolate a causal effect of direct train-test edges.

### Supplementary Methods 4. Negative-control analysis

Exploratory controls included random raw features, random-feature graph propagation, label shuffling and feature-shuffled GCN propagation. The feature-shuffled PPI-GCN control has complete run provenance and is reported here. Under the downstream evaluation protocol, it achieved validation AUC 0.7080, validation AUPRC 0.4241, test AUC 0.5868 and test AUPRC 0.3707. The result is used only as supplementary negative-control evidence and is not part of the main comparative claims.

### Supplementary Methods 5. BioGRID second-network validation

We constructed an alternative graph from BioGRID release 5.0.262 (compiled 25 September 2026; October 2026 release). Interactions were retained when both organisms were human and the experimental-system type was physical; Affinity Capture-RNA and Protein-RNA records were excluded. Entrez identifiers were mapped through unique aliases to the frozen STRING node universe. Mapped self-interactions were removed and duplicate unordered pairs collapsed. The resulting graph contained 931,458 undirected pairs on the same 19,622 nodes, including 1,402 isolates. It covered 18,006 of 18,220 original test variants and 17,748 of 18,220 homology-split test variants by at least one edge; all test variants were retained in evaluation. Of the BioGRID pairs, 526,635 (56.54%) occurred in STRING; pair-set Jaccard similarity was 0.07251.

Each unordered pair was represented in both directions. STRING and BioGRID graph features used the same self-loop convention and propagation: $A=(E+I)+(E+I)^T$, $S=D^{-1/2}AD^{-1/2}$, and $(X+SX+S^2X)/3$. Edges were not weighted by publication count or clinical labels. BioGRID node identities were checked against the actual frozen model-input mapping. Isolated nodes retained their raw-feature values within numerical tolerance. STRING contains evidence also represented in BioGRID, and the networks differ in density, coverage, release and evidence composition; this comparison is not evidence-independent replication. The physical category can include co-complex and proximity evidence rather than direct binding alone.

Nine BioGRID classifiers were fitted: original-split protein representations, original-split score fusion and homology-split score fusion, each with classifier seeds 42, 43 and 44. Comparator raw-ESM, STRING and score-only models used the same seeds and training configuration. The MLP used 512/128 hidden units, ReLU, dropout 0.3, class-weighted binary cross-entropy, Adam with learning rate 0.001 and weight decay 0.00001, batch size 1,024, a maximum of 50 epochs and patience five. Checkpoints were selected only by validation AUC. Representation input dimension was 1,280; fusion concatenated CADD, AlphaMissense and the AlphaMissense missingness indicator before the 1,280-dimensional protein representation. Score-only and fusion classifiers used the same hidden-layer architecture, while input-layer parameter counts differed. Missing scores were imputed using training medians, with no added z-score scaling. Dataset memberships, ESM features and homology assignments were unchanged, and graph-source selection did not use test outcomes.

Protein-only prediction was computed once per node in fixed-order CPU batches and mapped to its variant rows; fusion predictions remained variant-specific. Per-seed probabilities were converted to float64 before averaging. AUPRC denotes non-interpolated average precision. Means and sample standard deviations (ddof=1) describe individual classifiers; paired differences describe metrics after averaging the probabilities. Confidence intervals use 2,000 paired protein-cluster bootstrap resamples, retaining all variants and multiplicities of sampled proteins. They condition on the fitted models and fixed networks, exclude retraining and graph-sampling uncertainty, and are not adjusted for multiple comparisons.

## Supplementary Tables

### Supplementary Table 1. Dataset, split and graph summary

| Item | Train | Validation | Test | Total or value | Notes |
|---|---:|---:|---:|---:|---|
| Orig. variants | 141,864 | 17,129 | 18,220 | 177,213 | Protein split |
| Orig. benign | 102,818 | 12,737 | 12,398 | 127,953 | Label 0 |
| Orig. pathogenic | 39,046 | 4,392 | 5,822 | 49,260 | Label 1 |
| Orig. proteins | 11,298 | 1,412 | 1,413 | 14,123 labelled proteins | No protein overlap |
| Homol. variants | 141,864 | 17,129 | 18,220 | 177,213 | Cluster split |
| Homol. benign | 96,212 | 15,445 | 16,296 | 127,953 | Label 0 |
| Homol. pathogenic | 45,652 | 1,684 | 1,924 | 49,260 | Label 1 |
| Homol. proteins | 7,370 | 3,315 | 3,438 | 14,123 labelled proteins | No protein or cluster overlap |
| Homol. clusters | NA | NA | NA | 9,731 | Mean 1.45; max 251 |
| Repr. graph | NA | NA | NA | 19,622 nodes; 13,715,404 directed entries | Propagation graph |
| Topology graph | NA | NA | NA | 19,622 nodes; 929,472 edges | STRING score >=400 |
| CADD matching | NA | NA | NA | 177,205 matched; 8 missing | GRCh38 v1.7 whole-genome SNV scores |
| AlphaMissense matching | NA | NA | NA | 171,753 matched; 5,460 missing | hg38 AlphaMissense table |

Orig. denotes the original protein-level split. Homol. denotes the homology-aware cluster split. Repr. denotes the representation-propagation graph.

### Supplementary Table 2. Representation and architecture-control performance

The original-split raw ESM and PPI-GCN point estimates are from one fixed reference fit per model, as shown in Figure 1B–C. The separate three-seed rows summarize seeds 42, 43 and 44 using population SD (ddof=0). Homology-aware representation rows are one fit per model, without an empirical SD (Figure 2D). Table 8 summarizes these sources.

| Split | Model | Val AUC | Val AUPRC | Test AUC | Test AUPRC | Status |
|---|---|---:|---:|---:|---:|---|
| Orig. | Raw ESM | 0.7814 | 0.5569 | 0.7790 | 0.6074 | Reference |
| Orig. | PPI-GCN | 0.8237 | 0.5821 | 0.7997 | 0.6496 | Reference |
| Orig. | Raw ESM (3 seeds) | NA | NA | 0.7835 +/- 0.0051 | 0.6116 +/- 0.0054 | 3 seeds |
| Orig. | PPI-GCN (3 seeds) | NA | NA | 0.7950 +/- 0.0013 | 0.6484 +/- 0.0096 | 3 seeds |
| Orig. | Fixed mean propagation | 0.8008 | 0.6136 | 0.7950 | 0.6044 | Fixed operator |
| Orig. | Fixed cosine-weighted propagation | 0.8005 | 0.5876 | 0.7949 | 0.6157 | Fixed operator |
| Orig. | Degree-only | 0.7298 | 0.4228 | 0.6377 | 0.3901 | Control |
| Orig. | Feature-shuffled GCN | 0.7080 | 0.4241 | 0.5868 | 0.3707 | Verified control |
| Homol. | Raw ESM | 0.7315 | 0.2171 | 0.7248 | 0.2418 | Stress test |
| Homol. | PPI-GCN | 0.7645 | 0.2450 | 0.7606 | 0.2768 | Stress test |

Orig. denotes the original split. Homol. denotes the homology-aware split.

### Supplementary Table 3. External predictor and fusion-control performance

Figure 2B–C uses repeated fusion-control fits. Single-score CADD and AlphaMissense rows provide separate baseline context. For rows with repeated fits, the displayed SD is sample SD (ddof=1). The nine-fit permutation row pools three graphs and three classifier seeds per graph; its SD reflects both graph and classifier variation.

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

Notes for Table 3A: AM denotes AlphaMissense. "Arch." denotes the architecture-matched CADD+AlphaMissense baseline using the same 512/128 hidden-layer architecture and preprocessing as the fusion controls, with fewer input-layer parameters. The rows beginning with "+" add the listed protein representation to the CADD+AM feature set. Compared with the architecture-matched CADD+AM baseline, raw ESM fusion increased test AUC/AUPRC by +0.0108/+0.0249, edge-ablated PPI-GCN fusion by +0.0136/+0.0327, topology-permuted PPI-GCN fusion by +0.0122/+0.0286 and real PPI-GCN fusion by +0.0146/+0.0347. The edge-ablated row uses one deterministic direct train-test edge-ablation graph evaluated across three classifier seeds, whereas the topology-permuted row combines three random graph permutations with three classifier seeds.

#### Supplementary Table 3B. Homology-aware split

| Model | n | Val AUC | Val AUPRC | Test AUC | Test AUPRC |
|---|---:|---:|---:|---:|---:|
| CADD | 1 | 0.9506 | 0.6713 | 0.9504 | 0.6895 |
| AM | 1 | 0.9614 | 0.7945 | 0.9632 | 0.8029 |
| CADD+AM (arch.) | 3 | 0.9739 | 0.8362 | 0.9754 +/- 0.00005 | 0.8401 +/- 0.0002 |
| + raw ESM | 3 | 0.9715 | 0.8228 | 0.9728 +/- 0.0003 | 0.8249 +/- 0.0016 |
| + PPI-GCN | 1 | 0.9732 | 0.8273 | 0.9741 | 0.8333 |

Notes for Table 3B: the homology-aware graph-fusion row is one fitted model, whereas the score and raw-ESM fusion rows use three seeds. Relative to the architecture-matched CADD+AlphaMissense mean, the graph-fusion point estimate was -0.0013 AUC and -0.0068 AUPRC. This asymmetric comparison does not establish a reproducible loss from graph features. High-dimensional feature redundancy, additional input-layer parameters, overfitting and feature scaling are possible explanations for limited incremental performance; they were not tested individually.

### Supplementary Table 4. Uncertainty and repeated-run summary

#### Supplementary Table 4A. Raw ESM versus PPI-GCN uncertainty

| Bootstrap | Metric | Delta | CI low | CI high | n |
|---|---|---:|---:|---:|---:|
| Variant | AUC | +0.0208 | +0.0174 | +0.0240 | 1,000 |
| Variant | AUPRC | +0.0420 | +0.0353 | +0.0484 | 1,000 |
| Protein-clustered | AUC | +0.0207 | -0.0080 | +0.0527 | 1,000 |
| Protein-clustered | AUPRC | +0.0421 | -0.0021 | +0.0798 | 1,000 |

For Table 4A, Delta is PPI-GCN minus raw ESM. Variant-resampling rows report the mean difference across 1,000 paired bootstrap resamples; protein-resampling rows report the observed test-set metric difference. Both 95% intervals use the 2.5th and 97.5th percentiles of their respective bootstrap distributions.

#### Supplementary Table 4B. Original-split strong-predictor fusion-control uncertainty

Comparison codes: C0, real PPI-GCN fusion versus architecture-matched CADD+AlphaMissense; C1, real PPI-GCN fusion versus raw ESM fusion; C2, real PPI-GCN fusion versus edge-ablated PPI-GCN fusion; C3, real PPI-GCN fusion versus topology-permuted PPI-GCN fusion.

| ID | Metric | Delta | CI low | CI high | n |
|---|---|---:|---:|---:|---:|
| C0 | AUC | +0.0143 | +0.0062 | +0.0224 | 1,000 |
| C0 | AUPRC | +0.0341 | +0.0167 | +0.0566 | 1,000 |
| C1 | AUC | +0.0038 | +0.0005 | +0.0075 | 1,000 |
| C1 | AUPRC | +0.0093 | +0.0032 | +0.0167 | 1,000 |
| C2 | AUC | +0.0009 | -0.0009 | +0.0027 | 1,000 |
| C2 | AUPRC | +0.0017 | -0.0018 | +0.0052 | 1,000 |
| C3 | AUC | +0.0019 | -0.0005 | +0.0052 | 1,000 |
| C3 | AUPRC | +0.0050 | -0.0002 | +0.0112 | 1,000 |

#### Supplementary Table 4C. Fixed-reference fusion bootstrap

Comparison code: R0, fixed-reference graph fusion versus fixed-reference CADD+AlphaMissense; not architecture-matched.

| ID | Metric | Delta | CI low | CI high | n |
|---|---|---:|---:|---:|---:|
| R0 | AUC | +0.0133 | +0.0119 | +0.0146 | 1,000 |
| R0 | AUPRC | +0.0310 | +0.0268 | +0.0350 | 1,000 |

In Table 4A, the protein-clustered intervals include zero, whereas variant resampling ignores repeated variants within proteins and gives narrower positive intervals. Table 4B uses metrics of averaged predictions from the fitted ensembles, rather than differences between the per-run metric means in Table 3A. Its PPI-GCN-versus-raw-fusion interval is positive, while the margins over edge-ablated and topology-permuted controls include zero. Table 4C uses a separate fixed reference pair without architecture matching.

The homology-aware graph-fusion comparison in Table 3B remains a single-fit stress-test estimate; Table 4B concerns the original split.

### Supplementary Table 5. Topology-stratified PPI-GCN gain over raw ESM

These results correspond to the fixed reference model pair used in Figure 3B–F. Delta denotes PPI-GCN minus raw ESM. The topology-description graph uses the score-400 threshold and differs from the propagation graph (Methods 2).

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

The risk strata contain 18,171 variants: 49 variants from 10 proteins had fewer than five labelled training-neighbour variants and were assigned code -1. The boundary strata contain 18,193 variants: 27 variants from five proteins had community ID -1 after communities smaller than 50 nodes were excluded, leaving their within-community neighbour fraction undefined. These omitted groups fell below the minimum reporting size of 50 variants. Other strata cover all 18,220 test variants.

Selected exploratory joint topology cells:

Cell codes: J0, mid k-core / high betweenness; J1, mid k-core / mid clustering; J2, high neighbour risk / mid k-core; J3, high k-core / low clustering.

| ID | n | Pos. rate | Delta AUC | Delta AUPRC | AUC CI |
|---|---:|---:|---:|---:|---|
| J0 | 1,950 | 0.4297 | +0.1429 | +0.1533 | +0.1246 to +0.1617 |
| J1 | 2,089 | 0.4045 | +0.1408 | +0.1428 | +0.1234 to +0.1581 |
| J2 | 1,904 | 0.5767 | +0.1274 | +0.0983 | +0.1045 to +0.1499 |
| J3 | 4,791 | 0.3870 | -0.0532 | -0.0480 | -0.0611 to -0.0457 |

Full topology maps are provided in the prepared source-data archive rather than printed in full in the PDF supplement.

The four highlighted cells were selected from 53 joint cells examined on the same test set. Their intervals resampled variants within each cell, without accounting for within-protein dependence, subgroup selection or multiple comparisons. They describe exploratory associations and require independent validation.

### Supplementary Table 6. Descriptive prediction transitions and baseline-error-stratified profiles

Corrected-sample group counts for raw ESM to PPI-GCN:

Both comparisons use the fixed prediction pairs shown in Figure 4 at threshold 0.5. The score/fusion pair differs from the architecture-matched repeated ensembles in Figure 2. Counts, group means and effect sizes are descriptive.

| Group | n | Positive rate |
|---|---:|---:|
| both correct | 11,875 | 0.3270 |
| both wrong | 3,965 | 0.2678 |
| improved/corrected | 1,436 | 0.5829 |
| degraded | 944 | 0.0424 |
| false negatives rescued | 837 | 1.0000 |
| false positives suppressed | 599 | 0.0000 |

Corrected-sample group counts for fixed-reference CADD+AlphaMissense to fixed-reference graph fusion:

| Group | n | Positive rate |
|---|---:|---:|
| both correct | 15,981 | 0.3173 |
| both wrong | 1,298 | 0.3274 |
| improved/corrected | 666 | 0.3333 |
| degraded | 275 | 0.3818 |
| false negatives rescued | 222 | 1.0000 |
| false positives suppressed | 444 | 0.0000 |

Selected false-positive-suppressed versus false-negative-rescued topology contrasts for raw ESM to PPI-GCN:

| Feature | FP mean | FN mean | Diff |
| --- | ---: | ---: | ---: |
| degree bucket | 0.8698 | 1.9833 | -1.1135 |
| k-core bucket | 0.8381 | 1.1254 | -0.2874 |
| neighbour risk bucket | 0.5726 | 1.6941 | -1.1215 |
| neighbour pathogenic rate | 0.2717 | 0.4329 | -0.1612 |
| betweenness bucket | 0.9482 | 1.8530 | -0.9048 |
| clustering coefficient | 0.2534 | 0.2022 | +0.0512 |
| boundary role bucket | 1.1519 | 0.3321 | +0.8198 |

Selected false-positive-suppressed versus false-negative-rescued topology contrasts for fixed-reference CADD+AlphaMissense to fixed-reference graph fusion:

| Feature | FP mean | FN mean | Diff |
| --- | ---: | ---: | ---: |
| degree bucket | 1.5541 | 1.8198 | -0.2658 |
| k-core bucket | 1.5518 | 1.6036 | -0.0518 |
| neighbour risk bucket | 0.7793 | 1.7523 | -0.9730 |
| neighbour pathogenic rate | 0.3276 | 0.4955 | -0.1679 |
| betweenness bucket | 1.5541 | 1.7252 | -0.1712 |
| clustering coefficient | 0.1807 | 0.2176 | -0.0368 |
| boundary role bucket | 0.9685 | 0.7207 | +0.2477 |

Rescued versus persistent false negatives within the raw-ESM baseline false-negative group:

| Feature | Corrected subset mean | Persistent error mean | Hedges' g |
| --- | ---: | ---: | ---: |
| Degree bucket | 1.9833 | 1.5546 | +0.9675 |
| K-core bucket | 1.1254 | 1.5358 | -0.8441 |
| Neighbour risk bucket | 1.6941 | 1.0904 | +0.8163 |
| Neighbour pathogenic rate | 0.4329 | 0.3793 | +0.5259 |
| Betweenness bucket | 1.8530 | 1.5348 | +0.5923 |
| Clustering coefficient | 0.2022 | 0.2322 | -0.4011 |
| Boundary role bucket | 0.3321 | 0.9953 | -0.8000 |
| Within-community neighbour fraction | 0.5655 | 0.7093 | -0.7950 |

The raw-ESM baseline false-negative contrast contains 837 rescued and 1,062 persistent false negatives. Within-community neighbour fraction has two missing values in the persistent group, giving 1,060 available observations for that feature; missingness does not change the transition counts. Hedges' g uses the pooled within-group standard deviation with a small-sample correction. These are variant-weighted summaries, with missing values handled per feature. Bucket codes, including -1 for insufficient or unassigned context, retain their numerical encoding; bucket means are not biological effect magnitudes.

The comparison stratifies by the baseline binary error type; it does not match continuous prediction scores or eliminate residual prediction-difficulty differences. The associated feature tests treated variants as independent despite shared protein-level features, so their FDR/significance claims are not used. Figure 4E uses uniform dots to indicate the direction of group-mean differences, and Figure 4F displays descriptive effect sizes without significance markers. In Figure 4C–D, warm and blue colors indicate means above and below the within-row mean after range scaling, not comparable effect magnitudes across features. In E, blue denotes a higher mean among rescued false negatives and warm colors a higher mean among suppressed false positives; in F, blue and orange denote positive and negative Hedges' g. Directions and descriptive effect sizes do not establish correction mechanisms.

### Supplementary Table 7. Negative-control audit

Feature-shuffled propagation supplies one verified supplementary negative control. Other exploratory random-feature and label-shuffling summaries are not part of the reported evidence.

| Control | Split | n | Val A | Val AP | Test A | Test AP |
|---|---|---:|---:|---:|---:|---:|
| Feat.-shuffled GCN | Orig. | 1 | 0.7080 | 0.4241 | 0.5868 | 0.3707 |

Val A and Test A denote validation and test AUC, respectively. Val AP and Test AP denote validation and test AUPRC. This control is used only as supplementary negative-control evidence and is not part of the main comparative claims.

### Supplementary Table 8. Experimental configurations and run sources

| Analysis or figure | Fitted models | Seeds, budget and displayed uncertainty |
|---|---|---|
| Original raw/full reference pair; Figure 1B–C | 1 per arm | Maximum 50 epochs; point estimates; 1,000 paired resamples in C |
| Original raw/full repeated summary; Table 2 | 3 per arm | Classifier seeds 42–44; maximum 50 epochs; population SD, ddof=0 |
| Direct-edge ablation; Figure 1D | 5 ablated fits; raw/full reference points | Seeds 3042–3046; maximum 30 epochs; sample SD for ablated arm |
| Partial edge ablation; Figure 1E interior fractions | 3 per strategy and fraction | Maximum 30 epochs; strategy/fraction-specific seeds; sample SD |
| Partial edge ablation; Figure 1E endpoints | Shared full reference and five ablated fits | 0% and 100% endpoints reused across strategies, not independent repeats |
| Topology permutation; Figure 1F | 3 graphs × 3 classifiers | Graph seeds 6042–6044; classifier seeds 7042–7044, 7142–7144, 7242–7244; maximum 30 epochs; pooled sample SD |
| Original-split fusion; Figure 2B–C | 3 per arm; permutation 9 total | Seeds 42–44; maximum 50 epochs; sample SD in B; 1,000 protein-bootstrap resamples of averaged predictions in C |
| Homology-aware representations; Figure 2D | 1 per arm | Maximum 50 epochs; point estimates without an empirical SD |
| Homology-aware fusion; Figure 2E | Scores 3; raw fusion 3; graph fusion 1 | Maximum 50 epochs; seeds 42–44 and sample SD for repeated arms; graph fusion is one point |
| Topology profiles; Figure 3B–F | 1 fixed raw/full pair | Test-set subgroup analyses; selected-cell intervals resample variants |
| Prediction transitions; Figure 4 | 2 fixed prediction pairs | Threshold 0.5; descriptive profiles; no valid feature-test significance inference |

For partial-ablation interior fractions $f=0.25,0.50,0.75$, classifier seeds are $5042+1000f+r$, with repeat index $r=0,1,2$; high-confidence-first and low-confidence-first add offsets 100000 and 200000, respectively. Sample SD uses ddof=1. The pooled permutation SD combines graph and classifier variation. The full-graph operator adds loops before symmetrization; the ablation/permutation controls add one loop after symmetrization (Methods 2). Thus Figure 1D–F comparisons differ in operator convention, seeds and training budget; their dispersion and point estimates do not isolate a causal topology effect. Matching hidden layers in Figure 2 does not equalize input-layer parameter counts or the propagation conventions of its graph controls. BioGRID and its own matched comparator settings are given in Methods 5 and Table 9.

### Supplementary Table 9. BioGRID and its matched comparison models

#### Supplementary Table 9A. Full three-seed network-comparison summaries

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

Table 9 reports the BioGRID network comparison together with its matched comparator models. Entries in Table 9A are means and sample standard deviations across three fits; Table 9B reports paired differences computed from averaged prediction probabilities. All within-table contrasts use the models and evaluation protocol described in Supplementary Methods 5.

#### Supplementary Table 9B. BioGRID ensemble differences

| Setting | Comparison | ΔAUC (95% CI) | ΔAUPRC (95% CI) |
|---|---|---:|---:|
| Original representation | BioGRID FPP − Raw ESM | -0.02012 [-0.04803, +0.01005] | -0.00249 [-0.05396, +0.03630] |
| Original representation | BioGRID FPP − STRING FPP | -0.03079 [-0.06916, +0.00407] | -0.04371 [-0.08938, -0.00108] |
| Original fusion | BioGRID fusion − score baseline | +0.01242 [+0.00341, +0.02100] | +0.02788 [+0.00701, +0.05019] |
| Original fusion | BioGRID fusion − raw-ESM fusion | +0.00092 [-0.00115, +0.00364] | +0.00105 [-0.00371, +0.00691] |
| Original fusion | BioGRID fusion − STRING fusion | -0.00228 [-0.00420, -0.00058] | -0.00650 [-0.01164, -0.00233] |
| Homology fusion | BioGRID fusion − score baseline | -0.00204 [-0.00421, -0.00004] | -0.01176 [-0.03147, +0.00596] |
| Homology fusion | BioGRID fusion − raw-ESM fusion | +0.00051 [-0.00077, +0.00169] | +0.00177 [-0.00629, +0.01049] |
| Homology fusion | BioGRID fusion − STRING fusion | -0.00216 [-0.00357, -0.00090] | -0.01052 [-0.01884, -0.00182] |

All intervals are percentile 95% intervals from 2,000 finite paired protein resamples. The original test split contains 18,220 variants on 1,413 proteins (5,822 positives); the homology test split contains 18,220 variants on 3,438 proteins (1,924 positives). Positive prevalence therefore differs, limiting direct cross-split AUPRC comparison. Near-zero intervals do not establish equivalence.

## Data and code availability

Supplementary Information, code and compact source data are publicly available at https://github.com/brynn-t02/ppi-graph-missense-audit. The archive includes source data for Figures 1–4, topology-stratified metrics, joint topology maps, descriptive corrected-sample statistics, fusion-control summaries, bootstrap outputs, sample memberships and checksums. BioGRID validation includes its own matched comparator predictions and configurations. The original inputs and run records remain archived separately. Large third-party resources, including provider-controlled CADD and AlphaMissense score files, should be obtained from their original providers.
