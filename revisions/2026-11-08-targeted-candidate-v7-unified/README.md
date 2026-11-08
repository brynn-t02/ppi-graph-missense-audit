# BIBM 2026 cumulative candidate v7 — remaining unified experiments

Continues [v6](../2026-11-08-targeted-candidate-v6-figure3/) without overwriting earlier versions. Figure1E partial ablation and Figure2D homology representations now use matched operator/training settings and three classifier seeds. Figure1D shares the newly verified fully ablated endpoint with1E; its bar/SD and paired interval are synchronized. Figure3 receives only the requested B title and two caption clarifications. Figures4/5 and all other panels are preserved.

- `BIBM2026_workshop_targeted_candidate_v7_unified.pdf`:8-page main including references.
- `BIBM2026_supplementary_targeted_candidate_v7_unified.pdf`:21-page Supplementary Information.
- `source/`: current LaTeX, supplementary Markdown, figures and fonts.
- `修改说明.txt`: detailed Chinese changes, results, protocol amendment and preserved scope.
- `reproducibility/remaining_unification_20261108/`: executed v1/v2 protocols/code, all39 fitting histories and metrics,13 compact per-seed prediction tables,4,000 protein-bootstrap draws, input/feature/checkpoint hashes, diagnostic records and independent verification.

Homology mean AUC is0.7288 for raw ESM and0.7571 for full FPP; ensemble ΔAUC is+0.0288 (protein95%CI[+0.0185,+0.0403]). Partial-ablation curves decrease toward the completely ablated endpoint but are not uniformly monotonic. No strategy superiority is inferred. The original full endpoint exactly reproduced; the regenerated ablated endpoint did not exactly reproduce the historical run. Input/operator gates and first-tensor/first-fit locking were independently checked; the original failure and amendment are retained. No favorable rerun was selected.

Supplementary Tables8,11,12 and corresponding main-text statements are synchronized. Tables5/6/10 and external-validation results remain unchanged. Means/sample SD and metrics of averaged probabilities remain distinct. Protein intervals are conditional on fitted models and sampled graphs. There were39 actual new fitting executions; no new ESM extraction, homology clustering, clinical-score fusion or external-validation training.

Build main in `source/` with `tectonic --only-cached --keep-logs main.tex`; render SI with `python reproducibility/render_supplement.py` and ReportLab. All29 pages were visually reviewed and all fonts embedded. The compact independent verifier is documented in the new experiment subfolder. Model weights and large feature tensors remain server-only, not included in GitHub. Public prediction tables do not constitute full weight backups.

The paper/source ZIP and new experiment ZIP parts plus the unchanged Figure3 evidence/bootstrap and Figure4 verification ZIPs reconstruct this directory. Extract into the same directory and check `SHA256SUMS.txt`; each archive is below5MB. The complete GitHub snapshot contains all files. Historical data/audits are labeled in `reproducibility/history/` and the experiment v1 records; `verification.json` concerns this current candidate.

Not newly PDF eXpress certified or conference-submitted. The previous certified PDF and all older revision snapshots remain unchanged.
