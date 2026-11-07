# BIBM 2026 targeted candidate — 8 November 2026

This is a separate, unsubmitted candidate based on the unchanged font-clean source corresponding to the certified camera-ready manuscript. It is not a continuation of the broader unified draft. The certified and submitted files are preserved.

- `BIBM2026_workshop_targeted_candidate.pdf`: 8-page named candidate main.
- `BIBM2026_supplementary_targeted_candidate.pdf`: 18-page candidate Supplementary Information.
- `source/main.tex`: editable main entry point; 30 references are inline.
- `source/figure_drafts/`: five exact figure inputs. Only Figure 1B/C/D/F changed; 1A/E and Figures 2–5 are preserved.
- `source/Supplementary_Information_revised.md`: Figure 1-related supplementary synchronization only.
- `修改说明.txt`: concise Chinese completion report.
- `reproducibility/`: exact diffs, before/after edits, scope checks, Figure 1 saved mean/SD and bootstrap inputs, PDF and pixel checks, and the unchanged supplementary renderer.

Requested main edits: Introduction last paragraph; four Methods locations; Table I caption, eight rows and note; Section III.A and Figure 1 caption. Abstract, other Results sections, Discussion, title/authors/equations/references and Figures 2–5 are unchanged. The dataset/date paragraph follows the certified baseline in this limited candidate.

No training, inference, statistical recomputation, server access or paid API was performed for this targeted edit. Existing saved numerical results and a previously verified Figure 1 vector asset were used. Both current bootstrap schemes were present in the preserved canonical output. Historical source records are not overwritten or relabelled.

Build main: from `source/`, run `tectonic --only-cached --keep-logs main.tex` with standard TeX resources already cached (tested with Tectonic 0.16.9); on a fresh setup, standard resource retrieval may be needed. The supplementary renderer can be run with `python reproducibility/render_supplement.py --root .`, with reportlab and bundled DejaVu fonts. Its default generated filename retains the earlier naming convention; the targeted candidate PDF is the separately named delivery copy.

All final pages were visually reviewed and all font resources are embedded. This candidate has not received a new IEEE PDF eXpress certification and was not uploaded to the conference system.
