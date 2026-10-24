# External-source and second-network experiments — 24 October 2026

This experimental addendum addresses the workshop suggestion to strengthen validation with an external test source or another network. Both experiments were completed before deciding whether to incorporate them into the paper. **The manuscript is unchanged:** the current [named paper](../../revisions/2026-10-24-workshop-focused/BIBM2026_workshop_named_revised.pdf) and [anonymous paper](../../revisions/2026-10-24-workshop-focused/BIBM2026_workshop_anonymous_revised.pdf) remain seven pages, including references. This directory contains no revised manuscript or conference submission.

## Principal findings

On the primary external-source cohort, fixed propagation over STRING improved the frozen protein-representation models' performance relative to raw ESM. The mean AUC across three classifier seeds increased from **0.6932 to 0.7304**. For predictions averaged across those seeds, the paired protein-bootstrap AUC difference was **+0.03748, 95% interval [0.01733, 0.05638]**; the average-precision difference was **+0.04632 [0.01514, 0.07446]**.

| External primary cohort | AUC, mean ± sample SD | Average precision, mean ± sample SD |
|---|---:|---:|
| Raw ESM | 0.6932 ± 0.0067 | 0.5719 ± 0.0088 |
| STRING fixed propagation | 0.7304 ± 0.0042 | 0.6146 ± 0.0074 |
| BioGRID fixed propagation | 0.7025 ± 0.0116 | 0.5816 ± 0.0064 |

The stricter **nested** subset gave the same direction for STRING versus raw ESM: ensemble AUC difference **+0.03518 [0.00339, 0.06667]**, and average-precision difference **+0.05496 [0.01109, 0.09620]**. It is a sensitivity analysis of the same source, not a second independent replication.

BioGRID versus raw ESM remained uncertain on the external primary cohort: AUC difference **+0.01248 [−0.01319, 0.03504]**. Its interval also crossed zero in the strict subset. These findings support transfer of the particular STRING representation under the specified exclusions; they do not establish a general benefit from adding any interaction network.

## Second-network results on the frozen original benchmark

Nine new classifiers were fitted: BioGRID representation alone, BioGRID plus clinical-score fusion on the original split, and BioGRID plus clinical-score fusion on the homology split, each with seeds 42–44. Existing raw-ESM, STRING and score baselines were reused without retuning.

| Original benchmark setting | BioGRID result | Paired ensemble AUC interpretation |
|---|---|---|
| Representation alone | Mean AUC 0.7592, versus raw ESM 0.7835 and STRING 0.7950 | BioGRID minus raw: −0.02014 [−0.04808, 0.01000]; no established gain |
| Original-split fusion | Mean AUC 0.9693 | Above scores alone: +0.01242 [0.00341, 0.02100]; incremental margin over raw-ESM fusion uncertain: +0.00092 [−0.00115, 0.00364] |
| Homology-split fusion | Mean AUC 0.9730 | Below scores alone in this conditional AUC comparison: −0.00204 [−0.00421, −0.00004]; the average-precision interval crosses zero |

BioGRID fusion was below STRING fusion in both splits under the fitted-ensemble intervals. These are exploratory comparisons without multiplicity adjustment; in particular, the homology AUC interval against scores alone ends close to zero. Network coverage, density, release and included evidence differ, so the results do not isolate the causal effect of network identity or demonstrate that physical interactions are unhelpful.

Complete AUC/average-precision values, all seeds and all comparisons are in [evaluation_summary.csv](results/evaluation_summary.csv), [evaluation_per_seed.csv](results/evaluation_per_seed.csv), [ensemble_metrics.csv](results/ensemble_metrics.csv) and [cluster_bootstrap.json](results/cluster_bootstrap.json). Seed means and metrics of averaged predictions are different summaries; the reported intervals refer to the latter.

## Cohort, network and frozen design

The external source is **UniProtKB/Swiss-Prot HUMSAVAR 2026_03**, released 2 September 2026, with matching canonical human reviewed sequences. LP/P and LB/B labels define the binary comparison; uncertain labels are excluded. Canonical accessions must map to a unique frozen graph node with exact full-sequence identity and a matching reference residue. The union of UniProt-plus-substitution, STRING-node-plus-substitution, gene-plus-substitution and dbSNP identifiers excludes overlap with all 177,213 original benchmark variants. Seven ambiguous CALM accession/node records were excluded before inference; primary and strict membership did not change.

| Population | Variants | Proteins | Positive / negative | Use |
|---|---:|---:|---:|---|
| Primary | 10,785 | 2,498 | 4,301 / 6,484 | Excludes original training/validation labelled proteins and their exact sequences |
| Strict, nested within primary | 6,934 | 1,713 | 2,555 / 4,379 | Also excludes original test labelled proteins and their exact sequences |
| All variant-disjoint | 40,900 | 9,840 | 15,950 / 24,950 | Descriptive only; includes original training proteins |

The alternative graph is **BioGRID 5.0.262**, released 1 October 2026. Human–human physical-category records were retained after excluding RNA-capture/protein–RNA methods, ambiguous Entrez mappings, self edges and duplicate pairs. It has **931,458 undirected edges on the same 19,622 frozen nodes**, including 1,402 isolates. All original test rows and isolates remain in the analysis. Physical-category experimental records can include complex association or proximity and do not all establish direct binary binding. Of the BioGRID pairs, 526,635 (56.54%) also occur in the retained STRING network, so this is an alternative-network check rather than evidence-independent replication.

The fixed operator is the same across networks: `A=(E+I)+(E+I)^T`, summed duplicate contributions, symmetric weighted-degree normalization, and `(X+SX+S²X)/3`. The downstream MLP uses hidden layers 512/128, dropout 0.3, a maximum of 50 epochs and validation-AUC early stopping with patience five. The [protocol](code/protocol.json) and [freeze receipt](code/protocol_freeze.json) precede new fitting and external scoring. The primary and strict cohorts were not selected by model performance. No external classifier was fitted or tuned, and no external CADD/AlphaMissense fusion was evaluated.

## Verification and interpretation limits

[Independent verification](results/completed_run_verification.json) checked all nine selected epochs, validation/test metrics, finite checkpoint dimensions, actual tensor indices, 27 external population/model/seed predictions, 60 per-seed metric rows and 20 ensemble summaries. Six old checkpoints reproduced their archived validation/test metrics within the recorded direct-inference tolerance. Full prediction arrays and checkpoints remain in the retained experiment archive; they are not redistributed in this compact addendum.

Intervals use 2,000 paired protein-cluster bootstrap replicates, seed 20261024, and retain all variants with sampled protein multiplicity. The weighted metric implementation was checked against scikit-learn on tied-score and zero-weight examples and real predictions. Intervals condition on the fitted ensembles and frozen graphs; they omit retraining, network-source and new-cohort uncertainty. Within-protein scores are identical in external evaluation. Node batching can differ slightly from historical variant batching in floating-point rounding and tied ranks; these differences are recorded separately from direct checkpoint reproduction in the verification report.

HUMSAVAR is a separately curated variant source, **not an independent patient cohort or a prospective trial**. UniProt and ClinVar may share literature or annotations. The stated exclusions do not prove evidence independence, full homology separation or exclusion from every external predictor's development data. The protein-only models cannot distinguish substitutions within the same protein, and these protein-defined substitutions need not all be genomic single-nucleotide variants. Even proteins absent from supervised training remain nodes in the frozen transductive feature graph. External score fusion and calibrated clinical decision rules are outside this experiment.

**ClinVar date provenance:** the raw ClinVar file currently preserved in the source archive contains records with 2026 `LastEvaluated` dates. That observation does not establish when the processed training tables were generated, nor prove that those tables were built from that particular raw-file state. The actual historical content release has not been independently verified. Earlier references to 31 March 2024 should therefore not be treated as a verified content-release date. Input checksums and row identities are retained; this finding alone does not change the frozen model inputs or measured results. The paper has not been edited in this experiment turn.

## Contents and reproduction

- `code/`: exact executed runner, analyzer, verifier, protocol and freeze receipt; the exact legacy training module is included separately. These preserve execution paths and hashes for provenance. Adapt paths and create a new output/freeze directory before running elsewhere.
- `results/`: compact scientific results, nine metric/history pairs and verification records. `environment_summary.json` extracts scientific runtime fields; device identifiers and machine inventories are omitted.
- `sources/`: compact source counts, mapping, source checksums and release metadata. Preparation-time “pending” flags describe their original stage; the completed execution audits establish the later tensor-index and runtime checks.
- `reconstruction/`: source reconstruction programs with local source-data paths replaced by `PPI_SOURCE_DATA_ROOT`; see its [instructions](reconstruction/README.md). These are explicitly path-adapted copies, not byte-identical executed scripts.
- `licenses/`: retained UniProt and BioGRID notices. Provider downloads, complete external variant tables, graph arrays, embeddings and checkpoints are omitted.
- [PUBLICATION_PROVENANCE.json](PUBLICATION_PROVENANCE.json) maps copied/adapted files to original and published SHA256 values. [SHA256SUMS.txt](SHA256SUMS.txt) covers this public addendum.

This is a compact experimental record, not a one-command raw-data-to-results distribution. Full rerunning requires the pinned provider resources, original model inputs and retained checkpoints. The earlier matched-training materials and environment specifications are available in the [24 October archive](../../revisions/2026-10-24/reproducibility/). No new project license, author correspondence, credentials or server connection configuration is introduced here.
