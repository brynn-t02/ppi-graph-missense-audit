"""Read-only content checks for the v8 statistical-label clarification.

No training, inference, bootstrap, server access, or source edits.
"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import re

B = Path(__file__).resolve().parent
ROOT = B.parents[1]
OLD = ROOT / 'outputs/BIBM_workshop_targeted_candidate_v7_unified_20261108'
NEW = ROOT / 'outputs/BIBM_workshop_targeted_candidate_v8_clarity_20261108'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def block(s, begin, end):
    i = s.index(begin)
    j = s.index(end, i + len(begin))
    return s[i:j]


before = (OLD / 'source/main.tex').read_text()
after = (NEW / 'source/main.tex').read_text()
changes = json.loads((B / 'main_changes.json').read_text())
assert len(changes) == 4
expected = before
for change in changes:
    assert expected.count(change['before']) == 1
    expected = expected.replace(change['before'], change['after'])
assert expected == after

# The only newly introduced numeric token is the CI level 95% in Figure 1C.
old_numbers = re.findall(r'\d+(?:\.\d+)?', before)
new_numbers = re.findall(r'\d+(?:\.\d+)?', after)
assert Counter(new_numbers) - Counter(old_numbers) == Counter({'95': 1})
assert not (Counter(old_numbers) - Counter(new_numbers))
canonical_after = after.replace(
    'C, Paired ensemble differences with 95\\% intervals from variant-level',
    'C, Paired ensemble differences with intervals from variant-level', 1)
assert re.findall(r'\d+(?:\.\d+)?', canonical_after) == old_numbers
assert block(before, 'E, Partial train-test edge ablation', 'F, Topology-permutation') == block(
    after, 'E, Partial train-test edge ablation', 'F, Topology-permutation')
discussion_marker = next(x for x in ['\\section{Discussion}', '\\section{DISCUSSION}'] if x in before)
i = before.index(discussion_marker)
j = after.index(discussion_marker)
assert before[i:] == after[j:]  # Discussion and all following sections/citations unchanged.

# Check all inherited tabular and JSON artifacts, not just a selected subset.
assets = []
for p in sorted(OLD.rglob('*')):
    if p.is_file() and p.suffix in {'.csv', '.json', '.gz'}:
        rel = p.relative_to(OLD)
        q = NEW / rel
        assert q.is_file(), rel
        assert p.read_bytes() == q.read_bytes(), rel
        assets.append({'path': str(rel), 'sha256': sha(q)})

unchanged_figures = []
for p in sorted((OLD / 'source/figure_drafts').iterdir()):
    if p.is_file() and re.match(r'Figure[2345]', p.name):
        q = NEW / 'source/figure_drafts' / p.name
        assert q.is_file() and p.read_bytes() == q.read_bytes(), p.name
        unchanged_figures.append({'file': p.name, 'sha256': sha(q)})

si_md = Path('source/Supplementary_Information_revised.md')
si_pdf = Path('BIBM2026_supplementary_targeted_candidate_v7_unified.pdf')
for rel in [si_md, si_pdf]:
    assert (OLD / rel).read_bytes() == (NEW / rel).read_bytes(), str(rel)

figure_audit = json.loads((B / 'figure/audit.json').read_text())
assert sha(Path(figure_audit['source'])) == figure_audit['source_sha256']
assert sha(Path(figure_audit['output'])) == figure_audit['output_sha256']
assert figure_audit['change'] == {'before': 'ΔAUC +0.0264', 'after': 'Ensemble ΔAUC +0.0264'}
assert figure_audit['only_one_text_block_changed']
assert figure_audit['outside_annotation_changed_pixels'] == 0

report = {
    'status': 'passed',
    'main_exact_replacements': 4,
    'main_only_changes_in_figure1_and_figure2_captions': True,
    'new_numeric_tokens': ['95'],
    'all_other_numeric_tokens_and_order_unchanged': True,
    'figure1E_caption_byte_identical': True,
    'discussion_and_every_following_section_byte_identical': True,
    'supplement_markdown_and_pdf_byte_identical': True,
    'inherited_csv_json_gzip_assets_all_byte_identical': len(assets),
    'figures_2_to_5_unchanged': unchanged_figures,
    'figure1D_change': 'Only ΔAUC +0.0264 -> Ensemble ΔAUC +0.0264',
    'figure1D_scope_confirmed_by_text_and_pixel_audit': True,
    'statistical_wording_review': {
        'Figure1_B_D_F_nonpermutation': 'Means and sample SD (ddof=1) across three classifier seeds.',
        'Figure1_F_permutation': 'Means and sample SD (ddof=1) across nine fits: three graphs by three classifier seeds. Dispersion combines graph and classifier variation.',
        'Figure2_E': 'All three model arms use three actual classifier seeds and sample SD (ddof=1).',
        'Figure1_D_annotation': 'The +0.0264 label is the ensemble full-minus-edge-ablated AUC difference, not the difference of displayed means.',
        'substantive_issues': []},
    'no_training_inference_bootstrap_or_server_access': True,
    'main_source_sha256': sha(NEW / 'source/main.tex'),
    'supplement_markdown_sha256': sha(NEW / si_md),
    'supplement_pdf_sha256': sha(NEW / si_pdf),
    'inherited_asset_hashes': assets}
(B / 'independent_scope_review.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({k: v for k, v in report.items() if k not in
                  ['inherited_asset_hashes', 'figures_2_to_5_unchanged']}, indent=2))
