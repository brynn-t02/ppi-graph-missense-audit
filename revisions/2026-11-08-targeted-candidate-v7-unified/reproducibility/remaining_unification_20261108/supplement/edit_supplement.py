"""Minimal SI synchronization; requires completed, verified new server results."""
import argparse,csv,difflib,hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
WORK=ROOT/'work/remaining_unification_20261108'
HERE=WORK/'supplement'
TARGET=ROOT/'outputs/BIBM_workshop_targeted_candidate_v7_unified_20261108/source/Supplementary_Information_revised.md'
ap=argparse.ArgumentParser();ap.add_argument('--results',type=Path,default=WORK/'results/outputs');ap.add_argument('--apply',action='store_true');args=ap.parse_args()
R=args.results
before=(HERE/'Supplementary_Information_before.md').read_text()
assert TARGET.read_text()==before,'Candidate SI changed outside this patch; reconcile before applying.'
complete=json.loads((R/'analysis_complete.json').read_text());assert complete['passed'] and complete['classifier_fits']==39
gate=json.loads((R/'endpoint_input_gate_v2.json').read_text());assert gate['passed'] and len(gate['runs'])==6
assert gate['first_runs_locked'] and not gate['all_six_historical_metrics_reproduced']
runtime=json.loads((R/'runtime_environment.json').read_text())
bootstrap=json.loads((R/'figure2D_protein_bootstrap.json').read_text());assert bootstrap['resamples']==2000
ablation_bootstrap=json.loads((R/'figure1D_protein_bootstrap.json').read_text());assert ablation_bootstrap['resamples']==2000
def readcsv(name):
    with (R/name).open(newline='') as f:return list(csv.DictReader(f))
hom=readcsv('figure2D_summary.csv');curve=readcsv('figure1E_summary.csv')
assert len(hom)==2 and len(curve)==15
assert {x['arm'] for x in hom}=={'raw_ESM','full_FPP'}
assert all(int(x['n'])==3 for x in hom+curve)
def pm(row,k):return f"{float(row[k+'_mean']):.4f} ± {float(row[k+'_sd']):.4f}"
def signed(x):return f'{float(x):+.4f}'
def ci(k):return f"{signed(bootstrap['delta_'+k])} ({signed(bootstrap[k+'_ci95'][0])}, {signed(bootstrap[k+'_ci95'][1])})"
def ablation_ci(k):return f"{signed(ablation_bootstrap['delta_'+k])} ({signed(ablation_bootstrap[k+'_ci95'][0])}, {signed(ablation_bootstrap[k+'_ci95'][1])})"
text=before;changes=[]
def replace(old,new,label):
    global text
    assert text.count(old)==1,(label,text.count(old))
    text=text.replace(old,new,1);changes.append({'label':label,'before':old,'after':new})
def paragraph(starts):
    found=[p for p in text.split('\n\n') if p.startswith(starts)];assert len(found)==1,starts
    return found[0]

replace(paragraph('Figure guide:'),
    'Figure guide: Main Figure 1B–F uses matched representation experiments; Tables 11–12 supply the performance summaries and ensemble intervals, and Table 4A supplies both resampling intervals for Figure 1C. The partial-ablation curve in Figure 1E is detailed in Table 11B. Main Figure 2B/D/E uses matched fusion or representation results from Table 11, and Figure 2C uses the paired ensemble intervals in Table 12. Figure 3A shows ESM similarity, while Figure 3B–F uses matched original-split ensembles for the topology analysis and protein-clustered intervals in Table 5; Figure 4 uses matched original-split ensembles for the transition and feature profiles in Tables 6 and 10. Tables 2–3, 4B–C and 8B preserve separate reference analyses and their protocols. Matched configurations and reporting conventions are specified in Table 8A and Methods 5. External-source and BioGRID validation are documented in Methods 6, Tables 14–17 and main Figure 5. Different fitted prediction sets should not be interchanged.',
    'Figure guide')
replace('The main repeated comparisons used seeds 42, 43 and 44 (exceptions are detailed in Table 8); topology-permuted fusion combined three sampled graph permutations with these classifier seeds.',
    'The matched comparisons used seeds 42, 43 and 44 (Table 8A); topology-permuted fusion combined three sampled graph permutations with these classifier seeds. Archived reference protocols are specified in Table 8B.',
    'Methods 3 classifier-seed scope')
replace(paragraph('Table 2 preserves separate raw-ESM'),
    'Table 2 preserves separate raw-ESM and FPP reference estimates, historical repeated-run summaries and operator controls. The matched representation estimates in main Figure 1B/D/F and Figure 2D are supplied by Table 11A, and the partial-ablation series in Figure 1E by Table 11B. Their ensemble intervals are supplied by Table 12 and Table 4A. The reference rows in Table 2 should not be substituted for these matched comparisons.',
    'Table 2 source scope')
replace(paragraph('Table 4A supplies the matched ensemble intervals'),
    'Table 4A supplies the matched ensemble intervals in main Figure 1C. Main Figure 2C uses the matched fusion intervals in Table 12; Tables 4B–C retain their separate fitted comparisons. Main Figure 2D and Figure 2E use three matched fits per arm for representations and fusion, respectively (Tables 11–12); the estimates in Tables 2 and 3B remain separate reference results.',
    'Table 4 source scope')

old8=text.split('### Supplementary Table 8. ',1)[1].split('### Supplementary Table 9.',1)[0]
old8='### Supplementary Table 8. '+old8
old8table='\n'.join(line for line in old8.splitlines() if line.startswith('|'))
new8='''### Supplementary Table 8. Experimental configurations and run provenance

#### Supplementary Table 8A. Matched comparisons used in the main figures

| Analysis | Fits and classifier seeds | Maximum epochs | Configuration |
|---|---|---:|---|
| Original-split representations | 3 per arm; 42–44; 9 for permutations | 50 | Common propagation and training protocol |
| Partial ablation, .25/.50/.75 | 3 per strategy/fraction; 42–44 | 50 | Nine fixed intervention graphs; 27 fits |
| Partial ablation, 0 and 1 | 3 per endpoint; 42–44 | 50 | Six endpoint fits; shared across strategies |
| Original-split fusion | 3 per arm; 42–44; 9 for permutations | 50 | Same hidden-layer architecture and preprocessing |
| Homology-aware representations | 3 per arm; 42–44 | 50 | Raw ESM and full FPP |
| Homology-aware fusion | 3 per arm; 42–44 | 50 | Scores, raw-ESM fusion and full-FPP fusion |
| Topology and error profiles | Frozen matched ensembles | NA | No additional fitting for Figures 3B–F and 4 |

All graph arms in Table 8A use the full propagation convention, with self-loops added before symmetrization. Partial ablation removes unique undirected train-test protein pairs and all corresponding edge entries. Each interior strategy/fraction uses one fixed graph and three classifier seeds; dispersion across fits does not measure variation across graph masks. Random removals use seeds 292, 542 and 792 at fractions 0.25, 0.50 and 0.75, respectively, so the random masks are not nested. Confidence-based removals use real STRING combined scores and retain the original pair ordering and score-tie convention. The full-graph and fully ablated endpoints use the same three fits across all strategies, rather than independent endpoint graphs.

#### Supplementary Table 8B. Archived reference configurations

'''+old8table+'''

Table 8B documents the archived reference analyses in Tables 2–4 and 9. Their 30-epoch partial-ablation series and single-fit homology representation estimates are not the source of main Figure 1E or Figure 2D, which use the matched configurations in Table 8A and the results in Table 11. Figure 3B–F uses the matched original-split raw/full ensembles with fixed topology groups and protein-clustered intervals (Table 5). Figure 4 uses the matched original-split raw/full and score/full-fusion ensembles for its error profiles.

'''
replace(old8,new8,'Table 8 matched protocol and unchanged archival table')

replace(paragraph('The compact source-data archive contains'),
    'The compact source-data archive contains matched source data for main Figure 1B–F and Figure 2B–E, matched-ensemble topology metrics and protein-clustered intervals for Figure 3B–F, ESM-similarity source data for Figure 3A, matched-ensemble transition and feature data for Figure 4, and external-source and network-comparison data for Figure 5. The complete 177,213-row sample/split/cluster manifest, frozen node order, and core per-seed and external predictions are supplied in `canonical_analysis_20261025/public/`. The partial-ablation and homology-representation runs, endpoint-replication checks, canonical per-seed predictions, summary metrics and protein-bootstrap results are supplied in `remaining_unification_20261108/`. Tables 5, 6 and 10 use the original matched ensembles; their prediction sets are unchanged by the additional representation comparisons. Archived reference analyses retain their original source files and conventions. Figure-generation scripts, executed protocols, source manifests and checksums accompany the compact outputs. Large tensors and checkpoints are retained separately; third-party resources, including provider-controlled CADD and AlphaMissense score files, should be obtained from their original providers.',
    'Source-data archive')
replace('### Supplementary Methods 5. Matched validation and restored runtime (24 October 2026)',
    '### Supplementary Methods 5. Matched validation and restored runtime',
    'Methods 5 heading')
replace('All 48 downstream classifiers use seeds 42–44,',
    'Matched downstream classifiers use seeds 42–44,',
    'Methods 5 shared training protocol')
old=paragraph('The original-split representation comparison uses')
new=old+'\n\n'+(
    'The matched STRING experiments comprise 81 distinct condition–seed combinations: the 48 core comparisons above, 27 partial-ablation interior fits and six homology-aware representation fits. Six additional full-graph/fully ablated endpoint fits bring the total to 87 fitting executions. The three frozen-full positive controls reproduced their saved metrics and selected epochs. Sparse GPU propagation was not bitwise reproducible across repeated evaluations. We retained the first regenerated fully ablated tensor and its first three fits, froze their checksums and checked the propagation independently in float64 on the CPU in 16 columns. These fully ablated fits differ numerically from the archived fits and supply both Figure 1D and the shared fraction-1 endpoint in Figure 1E. Diagnostic regenerations were excluded from training. The full-graph endpoint is likewise shared across strategies. The additional fits used the same Python and numerical-library versions in a separate runtime: 122 exported pip-package pins and 16 portable conda-library pins were verified, together with pip, setuptools and wheel, for 141 verified package versions. Input checksums, frozen node order, all sample labels and split memberships, homology-cluster disjointness, and the full-graph propagation calculation were verified before fitting. No ESM extraction or homology clustering was repeated. Partial ablation retained 621,211 eligible undirected train-test pairs (1,242,422 edge entries) with complete STRING score coverage; each selected pair was removed in both directions, followed by recomputation of normalized propagation. The 25%, 50% and 75% conditions removed 155,303, 310,606 and 465,908 pairs, respectively. Graph masks were fixed independently of test labels, and every classifier was selected only by validation AUC.'
)
replace(old,new,'Methods 5 additional matched runs and endpoint checks')
old=paragraph('**Unified current reporting.**')
new=('**Unified reporting.** Tables 11–17 and main Figures 1B–F, 2B–E, 3B–F, 4 and 5 use the same inference and aggregation convention. For protein-only representation models, CPU evaluation in float32 with fixed batches of 1,024 nodes in the frozen node order produces one probability per protein, which is mapped to its variant rows. Fusion inputs remain variant-specific. Seed probabilities are converted to float64 before averaging, and AUPRC is calculated as non-interpolated average precision. Per-run means and sample standard deviations summarize individual classifier metrics; ensemble differences are calculated after averaging probabilities. The original reporting audit covered 21 representation checkpoints: on the original validation/test rows, the largest change from variant-batched probabilities was 3.58e-7 and the maximum within-protein spread decreased from 2.38e-7 to exactly zero. The 39 additional fitting executions use the same node-wise inference convention, with every mapped validation/test probability checked against the saved variant-batched output within 2e-6. The homology representation interval uses 2,000 paired protein-clustered resamples of the full-FPP and raw-ESM ensembles (Table 12), conditional on the fitted models and fixed graph. These intervals omit retraining and graph-sampling uncertainty and are not adjusted for multiple comparisons.')
replace(old,new,'Methods 5 reporting scope')

replace(paragraph('Current values use canonical per-run metrics'),
    'Values use canonical per-run metrics and means ± sample SD (ddof=1) across classifier fits. Permutation rows are shown separately by graph; their three seeds are classifier repeats conditional on that graph. Tables 2–3, 8B and 9 retain separate reference sources. Figures 3B–F and 4 and Tables 5, 6 and 10 use the matched original-split ensembles described in Methods 5.\n\n#### Supplementary Table 11A. Representation and fusion comparisons',
    'Table 11 introduction')
anchor='| Split | Arm | Fits | Test AUC | Test AUPRC |\n|---|---|---:|---:|---:|'
added=[]
for arm,label in [('raw_ESM','Raw ESM'),('full_FPP','Full FPP')]:
    x=next(x for x in hom if x['arm']==arm)
    added.append(f'| Homol. repr. | {label} | 3 | {pm(x,"auc")} | {pm(x,"auprc")} |')
replace(anchor,anchor+'\n'+'\n'.join(added),'Table 11A homology representation rows')
ablation_row=next(x for x in curve if x['strategy']=='random_pair_removal' and float(x['fraction_removed'])==1.)
old_ablation_table11='| Original | Ablated FPP | 3 | 0.7702 ± 0.0077 | 0.6077 ± 0.0140 |'
new_ablation_table11=f'| Original | Ablated FPP | 3 | {pm(ablation_row,"auc")} | {pm(ablation_row,"auprc")} |'
replace(old_ablation_table11,new_ablation_table11,'Table 11A common complete-ablation endpoint')
replace(paragraph('Current per-run metrics, seed, selected epoch'),
    'Core per-run metrics, seeds and selected epochs are recorded in `canonical_analysis_20261025/per_run_metrics.csv`, with training histories in `matched_validation_20261024/`. Per-run metrics for the homology representations, partial-ablation curve and updated fully ablated representation endpoint are supplied in `remaining_unification_20261108/`. Pooled permutation SD mixes graph and classifier dispersion and is descriptive. Paired seed differences are retained with their corresponding prediction sets.',
    'Table 11 run provenance')
curvelines=['#### Supplementary Table 11B. Matched partial train-test pair ablation','',
    '| Removal strategy | Fraction | Fits | Test AUC | Test AUPRC |',
    '|---|---:|---:|---:|---:|']
labels={'random_pair_removal':'Random','high_confidence_first':'High-score first','low_confidence_first':'Low-score first'}
for strategy in labels:
    for x in sorted([x for x in curve if x['strategy']==strategy],key=lambda x:float(x['fraction_removed'])):
        curvelines.append(f'| {labels[strategy]} | {float(x["fraction_removed"]):.2f} | 3 | {pm(x,"auc")} | {pm(x,"auprc")} |')
curvelines+=['','Entries are means ± sample SD across classifier seeds 42–44. Each interior condition uses one fixed graph; SD therefore describes classifier variation conditional on that graph. The fraction-0 and fraction-1 rows share the same full-graph and fully ablated fits across strategies. The 15 displayed points correspond to 11 distinct graph conditions and 33 classifier fits, including six endpoint fits.','']
replace('### Supplementary Table 12. Current protein-clustered paired ensemble margins',
    '\n'.join(curvelines)+'\n### Supplementary Table 12. Current protein-clustered paired ensemble margins',
    'Table 11B partial-ablation results')
for old in [line for line in text.splitlines() if line.startswith('| Homology |') and ('−' in line)]:
    replace(old,old.replace('| Homology |','| Homol. fusion |',1),'Table 12 fusion label')
anchor='| Split | Comparison | ΔAUC (95% CI) | ΔAUPRC (95% CI) |\n|---|---|---:|---:|'
newrow=f'| Homol. repr. | full FPP − raw ESM | {ci("auc")} | {ci("auprc")} |'
replace(anchor,anchor+'\n'+newrow,'Table 12 homology representation interval')
old_ablation_table12='| Original | full FPP − ablated FPP | +0.0248 (+0.0066, +0.0494) | +0.0429 (+0.0127, +0.0623) |'
new_ablation_table12=f'| Original | full FPP − ablated FPP | {ablation_ci("auc")} | {ablation_ci("auprc")} |'
replace(old_ablation_table12,new_ablation_table12,'Table 12 common complete-ablation paired interval')
replace('The unified reporting audit covers 57 frozen checkpoints,',
    'The external-validation reporting audit covers 57 frozen checkpoints,',
    'Methods 6 audit scope')

# Preserve unrelated numerical tables and the complete external-validation section.
def section(s,start,end):return s.split(start,1)[1].split(end,1)[0]
for n in [1,2,3,4,5,6,7,9,10]:
    st=f'### Supplementary Table {n}.'
    if n==7:en='### Supplementary Table 8.'
    elif n==10:en='## Remaining validation boundaries'
    else:en=f'### Supplementary Table {n+1}.'
    rows0=[l for l in section(before,st,en).splitlines() if l.startswith('|')]
    rows1=[l for l in section(text,st,en).splitlines() if l.startswith('|')]
    assert rows0==rows1,('protected table',n)
assert old8table in text
for l in section(before,'### Supplementary Table 11.','### Supplementary Table 12.').splitlines():
    if l.startswith('|') and l!=old_ablation_table11:assert l in text,('Table11existingrow',l)
for l in section(before,'### Supplementary Table 12.','### Supplementary Table 13.').splitlines():
    if l.startswith('|') and l!=old_ablation_table12:assert l.replace('| Homology |','| Homol. fusion |',1) in text,('Table12existingrow',l)
assert before.split('### Supplementary Table 13.',1)[1].replace('The unified reporting audit covers 57 frozen checkpoints,','The external-validation reporting audit covers 57 frozen checkpoints,')==text.split('### Supplementary Table 13.',1)[1]
assert 'Figure 2D retains the single-fit' not in text
assert 'Figure 1E use 30 epochs' not in text
assert 'Figure 1E and Figure 2D retain their separate' not in text
(HERE/'Supplementary_Information_proposed.md').write_text(text)
(HERE/'changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
(HERE/'supplement.diff').write_text(''.join(difflib.unified_diff(before.splitlines(True),text.splitlines(True),fromfile='v6/Supplementary_Information_revised.md',tofile='v7/Supplementary_Information_revised.md')))
(HERE/'verification.json').write_text(json.dumps({'passed':True,'blocks':len(changes),'homology_rows_added':2,'partial_curve_rows_added':15,'homology_interval_rows_added':1,'archived_table8_rows_unchanged':True,'tables1_to7_9_10_numeric_rows_unchanged':True,'table11_original15rows_unchanged':True,'table12_original9numericrows_unchanged':True,'ablation_summary_and_interval_updated_consistently':True,'table13_and_external_numbers_unchanged':True,'methods6_only_audit_scope_phrase_changed':True,'actual_result_files_used':['analysis_complete.json','endpoint_input_gate_v2.json','figure2D_summary.csv','figure1E_summary.csv','figure2D_protein_bootstrap.json','figure1D_protein_bootstrap.json'],'runtime_gpu':runtime['gpu']},indent=2)+'\n')
if args.apply:TARGET.write_text(text)
print(json.dumps({'prepared':True,'applied':args.apply,'blocks':len(changes),'target':str(TARGET)},ensure_ascii=False))
