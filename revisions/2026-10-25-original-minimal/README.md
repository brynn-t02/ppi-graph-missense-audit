# BIBM workshop: original-based revision (25 October 2026)

The main manuscript retains all four original figures and adds the already validated external-source/second-network results as Figure 5. Named and anonymous main PDFs are 8 pages including references; supplements are 18 pages.

## Files

- BIBM2026_workshop_named_revised.pdf / BIBM2026_workshop_anonymous_revised.pdf
- BIBM2026_supplementary_named_revised.pdf / BIBM2026_supplementary_anonymous_revised.pdf
- 原稿最小修订逐条说明.txt: detailed Chinese explanation
- REVISION_NOTES.md: concise changes and limitations
- source/: LaTeX, unchanged IEEEtran class, supplementary Markdown, original and revised figures, compact data and scripts
- reproducibility/: original baseline, exact source differences, figure audits, numerical/visual checks and supplement renderer
- final_quality_checks.json and SHA256SUMS.txt

## Rebuild

From source/, compile main.tex or main_anonymous.tex with Tectonic. The PDF uses actual T1 Times-compatible fonts. Rebuild the supplement with python reproducibility/render_supplement.py or add --anonymous; the recorded rendering environment is in reproducibility/render_environment.json. Figure 1/4 local-vector patch scripts and Figure 5's script are under source/figure_scripts/. Original accepted assets remain immutable.

The two original propagation equations and all original figure numerical inputs are retained. Current matched estimates are reported separately; the original figures do not silently become current-protocol results. The current canonical-analysis directory contains the same 49 verified files as the previous snapshot.

All older dated revisions are preserved. Large third-party source resources, tensors and model checkpoints are not included here. No conference submission or PDF eXpress certification is implied.
