# Editorial decisions

Baseline: `ce3c0209ddd04d842de5e876b95bcc50669c3a87`, `revisions/2026-10-24-ieee-checked/`.

## Changes

1. Introduction: remove a generic relevance statement and repeated study checklists; make the control requirements follow from split connectivity, homology and strong-score comparators.
2. Methods: consolidate the transductive explanation and remove empty self-evaluation. Graph structure and ESM features are available across partitions; held-out labels are excluded from propagation. Variant-to-protein mapping and inability of embedding-only inputs to resolve residue-specific effects remain explicit.
3. Results: start with the matched comparison instead of describing Table I. Keep all results and uncertainty; move the repeated four-factor fusion explanation to Discussion. The within-split AUPRC comparison rule remains in Methods.
4. Discussion: relate protein-level uncertainty to the shared representation within a protein, consolidate the homology-fusion explanation, and state the unresolved BioGRID comparisons directly. Mechanism candidates remain untested explanations, not findings.
5. Conclusion: state that graph-benefit claims depend on comparator, split, network and protein-level uncertainty. Limits concerning trained graph models and clinical decision validity remain.

## Preserved

All numerical results and their order, citation sequence, 30 bibliography entries, equations, inline mathematics, abstract, keywords, three figures/captions and Table I are unchanged. The supplements are byte-identical to the preceding revision. Named and anonymous main papers remain seven pages including references, without reducing type or margins. Only final reference-column break positions were readjusted after prose reflow.

The main scientific interpretation remains evaluation-dependent: STRING transfer is supported on the specified external-source sets; BioGRID external gains and incremental gains over raw-feature fusion remain uncertain. The homology-aware STRING fusion comparison establishes neither benefit nor consistent harm. Necessary distinctions between historical/current, mean/ensemble and variant/protein-level analyses remain.

The separate Chinese comparison documents all 12 changes. All earlier versions remain intact; this commit adds a dated snapshot and updates only the root index/checksums outside that snapshot.
