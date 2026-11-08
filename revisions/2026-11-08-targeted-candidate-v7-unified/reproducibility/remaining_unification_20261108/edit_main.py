from pathlib import Path
import json,csv,difflib
W=Path(__file__).resolve().parent;R=W/'results/outputs';O=Path('outputs/BIBM_workshop_targeted_candidate_v6_figure3_20261108');N=Path('outputs/BIBM_workshop_targeted_candidate_v7_unified_20261108')
assert json.loads((R/'analysis_complete.json').read_text())['passed']
rows=list(csv.DictReader((R/'canonical_summary_metrics.csv').open()))
def row(split,arm):return next(x for x in rows if x['split']==split and x['arm']==arm)
def n(x):return f'{float(x):.4f}'
def signed(x):return ('+' if x>=0 else '$-$')+f'{abs(x):.4f}'
def ci(v):return '['+', '.join(signed(x) for x in v)+']'
f=row('homology','full_FPP');r=row('homology','raw_ESM');a=row('original','edge_ablated_FPP');h=json.loads((R/'figure2D_protein_bootstrap.json').read_text());b=json.loads((R/'figure1D_protein_bootstrap.json').read_text())
s=(O/'source/main.tex').read_text();changes=[]
def replace(old,new):
 global s
 assert s.count(old)==1,(old[:120],s.count(old));s=s.replace(old,new);changes.append({'before':old,'after':new})
replace(r'Training settings for additional sensitivity analyses are specified in Table~\ref{tab:run_sources} and Supplementary Table 8.',r'Detailed experimental configurations are specified in Table~\ref{tab:run_sources} and Supplementary Table 8.')
replace('We fitted 48 models under matched STRING protocols across representation-only comparisons, original-split fusion and homology-aware fusion.','Matched STRING protocols covered representation-only comparisons, partial edge ablation, original-split fusion and homology-aware representations and fusion.')
replace('Partial edge-ablation sensitivity & 3 per strategy and interior fraction & Separate sensitivity protocol; endpoints reused','Partial edge-ablation sensitivity & 3 per graph condition & Matched settings; shared endpoint fits')
replace('Homology-aware representations & 1 per arm & Fixed reference comparison','Homology-aware representations & 3 per arm & Matched training settings')
replace('Matched STRING comparisons used three classifier seeds, a common propagation operator and a maximum of 50 training epochs. Partial edge-ablation sensitivity analyses used a separate 30-epoch protocol, with reused endpoints. Homology-aware representation profiles used single reference fits. Detailed configurations are provided in Supplementary Table 8.','Matched STRING comparisons used classifier seeds 42--44, a common propagation operator and a maximum of 50 training epochs. Partial-ablation strategies share the full-graph and fully ablated endpoint fits. Detailed configurations are provided in Supplementary Table 8.')
old=next(p for p in s.split('\n\n') if p.startswith('The STRING graph contained'))
# Preserve full/raw summaries and describe the unified curve without overclaiming a strict trend.
new=(r'The STRING graph contained 1,242,422 direct train-test edge-index entries (9.06\% of all entries), corresponding to 621,211 unique undirected protein pairs. To assess connectivity sensitivity, we removed these direct edges and evaluated the resulting representations using the matched training protocol. Full FPP achieved mean AUC 0.7950 and AUPRC 0.6485, compared with '+n(a['auc_mean'])+' and '+n(a['auprc_mean'])+r' for edge-ablated FPP (Fig.~\ref{fig:connectivity}D; Supplementary Table 11). The paired ensemble AUC difference was '+signed(b['delta_auc'])+r' (protein-clustered 95\% CI '+ci(b['auc_ci95'])+r'; Supplementary Table 12). Partial edge-ablation comparisons used the same propagation operator and classifier settings at each removal fraction, with common endpoint fits across strategies (Fig.~\ref{fig:connectivity}E; Supplementary Tables 8 and 11B). Intermediate estimates varied across strategies and were not uniformly monotonic. Edge removal also changes graph normalization and retains indirect cross-partition paths, so the observed differences cannot be attributed exclusively to direct connections.')
replace(old,new)
replace('E, Partial train-test edge-ablation sensitivity analysis using a separate training protocol with reused endpoint estimates; implementation details are provided in Supplementary Table 8.','E, Partial train-test edge ablation under matched settings. Curves show mean AUC differences from raw ESM; error bars denote sample standard deviation of model AUC across three classifier seeds, with one fixed graph per strategy/fraction and shared endpoint fits.')
old='Under the homology-aware split, a separate single-fit representation comparison yielded AUC 0.7248 for raw ESM and 0.7606 for full FPP (Fig. 2D).'
new='Under the homology-aware split, three-seed mean AUC/AUPRC was '+n(r['auc_mean'])+'/'+n(r['auprc_mean'])+' for raw ESM and '+n(f['auc_mean'])+'/'+n(f['auprc_mean'])+' for full FPP (Fig. 2D; Supplementary Table 11). The ensemble AUC difference was '+signed(h['delta_auc'])+r' (protein-clustered 95\% CI '+ci(h['auc_ci95'])+'; Supplementary Table 12).'
replace(old,new)
replace('D, Single-fit comparison of raw ESM and full FPP representations under the homology-aware split.','D, Three-seed matched raw-ESM and full-FPP representation comparisons under the homology-aware split, with sample standard deviation.')
replace('B--D, Performance differences between full FPP and raw ESM across degree, k-core, betweenness, clustering and neighbour-risk strata.','B--D, Ensemble performance differences (full FPP minus raw ESM) across degree, k-core, betweenness, clustering and neighbour-risk strata.')
replace(r'F, Four preselected joint-cell profiles with paired protein-clustered 95\% bootstrap intervals (2,000 resamples).',r'F, Four highlighted joint-cell profiles selected on the same test set, with paired protein-clustered 95\% bootstrap intervals (2,000 resamples).')
(N/'source/main.tex').write_text(s)
(W/'main_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
(W/'main.diff').write_text(''.join(difflib.unified_diff((O/'source/main.tex').read_text().splitlines(True),s.splitlines(True),fromfile='v6/main.tex',tofile='v7/main.tex')))
print({'blocks':len(changes),'homology':h,'ablation':b})
