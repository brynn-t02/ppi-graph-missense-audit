# Original manuscript + BioGRID: author-confirmed date restored

Independent candidate, 8 November 2026, based on [the local figure/text correction revision](https://github.com/brynn-t02/ppi-graph-missense-audit/tree/903a11fe5c89af43b1eb40cc3d8ced8f937d3adc/revisions/2026-11-08-original-biogrid-local-fixes).

- `BIBM2026_workshop_original_plus_BioGRID_date_restored.pdf`: 8-page main paper.
- `BIBM2026_supplementary_original_plus_BioGRID_date_restored.pdf`: 12-page supplement.
- `source/`: buildable LaTeX, unchanged figures and supplementary Markdown.
- `日期恢复说明_中文.txt`: exact scope and provenance of this edit.
- `reproducibility/date_change_record.json`: two before/after sentence replacements. The 31 March 2024 date is based on the author's explicit confirmation; this edit did not independently verify original download logs.
- `original_submission/` and `reproducibility/BioGRID/`: unchanged original archive and compact network-validation evidence.

Only the main Methods II.A date sentence was restored verbatim from the original, with its corresponding supplementary sentence synchronized. All other manuscript content, four figure PDFs, scientific data and bibliography are unchanged. All earlier snapshots and provenance audits remain archived. No experiments, inference, statistical recomputation or server work were performed.

Compile with `tectonic main.tex` from `source/`; render supplementary Markdown with `python reproducibility/render_supplement.py` (ReportLab required). This modified candidate is not newly PDF eXpress certified or conference-submitted.
