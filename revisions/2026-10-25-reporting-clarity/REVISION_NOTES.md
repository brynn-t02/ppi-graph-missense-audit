# Reporting-clarity revision

Baseline: d158580fd3d88c6eece5589d15f11a5c4a774694, revisions/2026-10-25-original-minimal.

## Changes

1. Replace six Figure 1/3/4 text blocks to distinguish historical point estimates and exploratory subgroup observations from established causal or universal effects. Original font sizes, data, chart coordinates and colors are preserved. At 300 dpi, no pixels changed outside the bounded text regions.
2. Clarify Figure 4 C/D row-centered, range-scaled colors; E's fixed directional colors; and F's effect-sign colors. Remove process-oriented caption wording.
3. Explain that Supplementary Table 4A variant rows report mean bootstrap differences whereas protein-clustered rows report observed test-set differences. Preserve all four reported differences and their percentile intervals. Link Figure 1C to this convention.
4. Identify the two abstract confidence intervals as intervals for ensemble AUC gains. Describe homology-aware graph-versus-score uncertainty neutrally and remove a repeated defensive sentence. Change “confirmed” to “yielded” for the conditional fusion comparison.
5. Describe frozen ClinVar inputs and fixed propagation operators directly in the supplement, while retaining unverified historical release-date and no-learned-propagation limitations.
6. Correct “assisted with”; replace the isolated numbered Practical recommendations paragraph with an unnumbered bold lead-in.
7. Explicitly list Supplementary Information in the public availability statement and expose the supplement through direct README links.

## Checked and retained

The current MMseqs command already displays and extracts the correct ASCII spaces and hyphens: --min-seq-id 0.40 -c 0.50 --cov-mode 0 --threads 8. Both named and anonymous PDFs were checked visually and by extracted characters; no command edit was needed.

Original small figure labels remain; this revision does not redesign or rescale the figures. All five figures remain in the main paper. Figure 2 and Figure 5 used assets are byte-identical to the preceding revision. Historical and current estimates retain separate provenance.

## Verification

Both main PDFs are 8 pages including references; both supplements are 18 pages. All pages were visually inspected. Used fonts are embedded; abstract text uses an actual bold face; anonymous copies omit author identifiers. There are no overfull, undefined-reference or missing-font/glyph warnings.

Scientific values, equations, citations and bibliography are unchanged. All 49 canonical-analysis files and all 267 supplementary table lines match the preceding version. No new experiments, server access or experimental-resource retrieval occurred. All older snapshots are preserved.
