# Integrated external-source and second-network revision

The manuscript now incorporates the completed external-source and BioGRID experiments. The accepted title and confirmed author information are retained, with anonymous copies. Both main papers are **7 pages including references**; supplements are **19 pages**. No new model fitting or provider-data downloads were performed during this editorial integration.

## Selected content

- Main Methods explain external filtering, nested primary/strict sets, frozen transductive node access and the alternative network. Results report the supported STRING transfer and mixed BioGRID evidence, including uncertain or unfavorable comparisons.
- Main Figure 3 is a new paired-effect/95%interval plot. Historical exploratory topology panels move to Supplementary Figure S2; historical transitions remain Figure S1. The main paper keeps three figures.
- Figures 1/2 have repaired legend placement, annotation spacing, diagram wrapping and error-bar visibility. Historical supplement figures have clearer labels and coding keys. Numerical inputs are unchanged.
- Table I separates the original 48 matched audit fits,9 additional BioGRID fits and frozen external inference. Supplementary Methods 6 and Tables 14–17 retain filtering, mapping, all20 seed summaries and17 paired comparisons.
- The unsupported exact ClinVar source-release date is removed. Frozen processed-input hashes identify the evaluated benchmark; the date of the retained raw backup does not establish when processed tables were generated.

## Supported conclusions and limits

On the primary external source, STRING-minus-raw ensemble ΔAUC is +0.03748 [0.01733,0.05638], with the same direction on the nested strict subset (+0.03518 [0.00339,0.06667]). BioGRID-minus-raw intervals cross zero externally. Original-split BioGRID fusion improves over scores alone, but its incremental advantage over raw-embedding fusion is uncertain; network choice matters. BioGRID-versus-score homology AUC has an upper interval just below zero, while its AP interval crosseszero. These conditional, exploratory comparisons do not establish stable harm.

HUMSAVAR is an external curated source, not independent patients or prospective clinical validation. Shared annotation evidence, frozen-graph transduction and incomplete homology/development-data separation remain. No external CADD/AlphaMissense fusion, trained GNN, independently confirmed subgroup or calibrated clinical rule is claimed. The new analyzer averages probabilities in float64; earlier matched ensemble results retain their float32 provenance. Primary and strict are nested and not independent replications.

The workshop remains the revision criterion; additional main-conference-only studies are not new commitments. Historical snapshots and executed experiment results are preserved. No conference-system submission was performed.
