### Supplementary Table 5. Topology-stratified FPP gain over raw ESM

These results use matched raw-ESM and full-FPP ensembles, with probabilities averaged in float64 across classifier seeds 42–44. The original test memberships, topology buckets, 53 joint cells and four highlighted cells are fixed. The topology-description graph is distinct from the propagation graph (Methods 2). The same test set was used to examine the original cells and select the highlighted profiles; these profiles have not been independently confirmed in the external source.

| Stratum | n | Positive rate | Delta AUC | Delta AUPRC |
| --- | --- | --- | --- | --- |
| Degree-low | 2,795 | 0.0426 | +0.0185 | +0.0050 |
| Degree-mid | 4,762 | 0.2089 | +0.0135 | +0.0195 |
| Degree-high | 10,663 | 0.4415 | -0.0216 | +0.0061 |
| K-core-low | 2,847 | 0.0425 | +0.0175 | +0.0044 |
| K-core-mid | 5,685 | 0.3078 | +0.0603 | +0.0960 |
| K-core-high | 9,688 | 0.4078 | -0.0283 | +0.0125 |
| Risk-low | 6,323 | 0.1056 | +0.0231 | +0.0474 |
| Risk-mid | 5,829 | 0.2882 | +0.0334 | +0.0617 |
| Risk-high | 6,019 | 0.5772 | +0.0133 | +0.0291 |
| Betweenness-low | 3,050 | 0.1393 | +0.0545 | +0.0781 |
| Betweenness-mid | 4,486 | 0.1899 | +0.0368 | +0.0417 |
| Betweenness-high | 10,684 | 0.4254 | -0.0210 | +0.0086 |
| Clustering-low | 8,234 | 0.2888 | -0.0022 | -0.0122 |
| Clustering-mid | 5,950 | 0.4198 | +0.0343 | +0.0471 |
| Clustering-high | 4,036 | 0.2344 | +0.0309 | +0.0902 |
| Boundary-like | 6,198 | 0.3527 | +0.0026 | +0.0076 |
| Boundary-mixed | 6,598 | 0.2963 | +0.0060 | +0.0587 |
| Boundary-internal | 5,397 | 0.3111 | +0.0135 | +0.0184 |

The risk strata contain 18,171 variants: 49 variants from 10 proteins had fewer than five labelled training-neighbour variants and were assigned the insufficient-signal code -1. The boundary strata contain 18,193 variants: 27 variants from five proteins had community ID -1 after communities with fewer than 50 nodes were excluded by the topology script. Of these, four variants mapped to two isolated proteins; the other 23 mapped to three non-isolated proteins. Their within-community fraction was undefined. Both omitted groups were smaller than the minimum group size of 50. Other strata cover all 18,220 test variants.

Selected exploratory joint topology cells:

Cell codes: J0, mid k-core / high betweenness; J1, mid k-core / mid clustering; J2, high neighbour risk / mid k-core; J3, high k-core / low clustering. The displayed cells are unchanged from the original selection.

| ID | n | Pos. rate | Delta AUC | Delta AUPRC | AUC CI |
| --- | --- | --- | --- | --- | --- |
| J0 | 1,950 | 0.4297 | +0.1115 | +0.1202 | -0.0012 to +0.1782 |
| J1 | 2,089 | 0.4045 | +0.0861 | +0.0836 | +0.0065 to +0.1445 |
| J2 | 1,904 | 0.5767 | +0.0832 | +0.0686 | -0.0177 to +0.1514 |
| J3 | 4,791 | 0.3870 | -0.0486 | -0.0424 | -0.2020 to +0.0312 |

J0–J3 contain 126, 163, 134 and 95 distinct proteins, respectively. Paired 95% intervals use 2,000 protein-clustered resamples within each fixed cell: sampled proteins contribute all their variants, with the same multiplicities for both models. Percentile intervals condition on the fitted ensembles and graph; they do not include retraining, graph-sampling or subgroup-selection uncertainty and are not adjusted for multiple comparisons. All 2,000 resamples were evaluable in the four displayed cells. Of the other cells, four had 2, 227, 95 and 724 single-class resamples, respectively; their undefined AUC/AP differences were excluded from the corresponding percentile intervals, without replacement draws. Full point estimates, protein counts, finite-replicate counts, intervals and memberships for all 53 cells, together with the 18 one-dimensional strata, are supplied in the compact source-data archive. These exploratory profiles do not independently confirm topology-defined benefit subgroups.
