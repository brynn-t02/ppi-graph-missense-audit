# Reconstructing the source inputs

These programs are supplied for reproducibility. Packaging them did not perform a download, reconstruct a large file or rerun an experiment. Before use, set `PPI_SOURCE_DATA_ROOT` to the original source-data directory containing `raw/` and `processed/`. It must contain the checksum-identified STRING v12 inputs, the frozen ClinVar partitions, the matched AlphaMissense identifier table and, when deriving the node order again, `esm_node_features_full.csv`.

The reconstruction copies change only source-data path configuration and its required `os` import. Their original and published hashes are listed in `../PUBLICATION_PROVENANCE.json`. Input-file hashes refer to scientific data, not to the adapted script bytes. Metadata under `../sources/` normalizes original local path prefixes; historical hashes and scientific values are retained.

## External cohort

Obtain the exact HUMSAVAR and canonical-sequence files identified in `../sources/cohort_recon/download_manifest.json`. The supplied downloader uses mutable “current release” URLs: a fresh download must not be assumed to be release 2026_03 or to reproduce the frozen cohort. Compare file hashes and release headers before continuing.

Place those two files in `cohort_recon/`, alongside the included protocol. Then run `build_cohort.py` with `--node-map ../network_recon/frozen_node_order.json`, followed by `verify_cohort.py`, from the cohort directory with `PPI_SOURCE_DATA_ROOT` set. The builder creates new derived tables and manifests; it does not read model predictions. Expected final counts are 40,900 total variant-disjoint records, 10,785 primary records and 6,934 strict records. Inspect the final pre-inference CALM collision exclusion recorded in the included protocol amendment.

The rsID exclusion cache is recomputed when absent. If reusing a cache while changing the original cohort, delete the old cache first; its historical invalidation key records the raw-file checksum rather than the complete processed-cohort membership. Exact reconstruction requires the frozen original cohort, not a fresh ClinVar release. Before inference, independently map each STRING identifier through the actual model tensor's `protein_to_idx`; sorted candidate order is not sufficient proof.

## BioGRID network

The included `network_recon/` mapping tables and node order are the frozen files. Their hashes are preserved. To regenerate them, use `prepare_entrez_mapping.py` with the frozen source inputs. Place the exact BioGRID 5.0.262 ZIP, identified by the source metadata, into the same directory, then run `build_biogrid_network.py`. `audit_string_overlap.py` compares its edge pairs with the retained STRING source. Expected output is 931,458 undirected nonself pairs across 19,622 nodes, with 1,402 isolates.

Reconstructed graph indices must be checked against the actual model tensor before propagation. The completed execution audit in `../results/network_execution_audit.json` records that this gate passed in the reported experiment, with no remapping required.

## Model execution

The unmodified scripts in `../code/` document the executed experiment. They expect the retained parent experiment's data, six original raw/STRING checkpoints and legacy module in the original layout. The included legacy module has the same hash as that dependency. They also expect the new cohort and network files in the experiment layout. The compact public addendum omits large arrays and weights, so it cannot run from these small files alone.

For a new execution, adapt the saved base paths in copies, verify the pinned inputs and existing checkpoint hashes, and use a new output directory. The scientific sequence is runner `--run`, analysis, then verification; the runner's `--smoke` mode checks execution without test evaluation or saved fitted models. Preserve every selected checkpoint and full prediction array. Do not reuse a pre-inference manifest after changing inputs or code, and do not present an adapted rerun as byte-identical to the archived execution.
