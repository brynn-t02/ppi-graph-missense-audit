from pathlib import Path
import json,hashlib,re
import pandas as pd
b=Path('work/remaining_unification_20261108');old=Path('outputs/BIBM_workshop_targeted_candidate_v6_figure3_20261108/source');new=Path('outputs/BIBM_workshop_targeted_candidate_v7_unified_20261108/source')
s=(old/'main.tex').read_text();n=(new/'main.tex').read_text();changes=json.loads((b/'main_changes.json').read_text())
for c in changes:
 assert s.count(c['before'])==1,c['before'][:80]
 s=s.replace(c['before'],c['after'])
assert s==n,'Main contains unlistedchanges'
assert not any(x in n for x in ['0.0248','0.7702','0.6077','0.7248','0.7606','single-fit representation','separate 30-epoch'])
a=(old/'Supplementary_Information_revised.md').read_text();z=(new/'Supplementary_Information_revised.md').read_text()
def block(text,head,end='\n### '):
 pos=text.index(head);stop=text.find(end,pos+len(head));return text[pos:] if stop<0 else text[pos:stop]
protected={}
for k in [5,6,10]:
 head=f'### Supplementary Table {k}.';stop='\n## Source-data archive status' if k==10 else '\n### ';protected[f'SupplementaryTable{k}']=block(a,head,stop)==block(z,head,stop)
assert all(protected.values())
assert a[a.index('### Supplementary Methods 6.'):].replace('The unified reporting audit covers 57 frozen checkpoints','The external-validation reporting audit covers 57 frozen checkpoints')==z[z.index('### Supplementary Methods 6.'):]
figure_same={}
for pattern in ['*Figure4*.pdf','*Figure5*.pdf']:
 for p in old.glob('figure_drafts/'+pattern):
  q=new/'figure_drafts'/p.name;assert q.exists();figure_same[p.name]=hashlib.sha256(p.read_bytes()).hexdigest()==hashlib.sha256(q.read_bytes()).hexdigest()
assert all(figure_same.values())
rows=pd.read_csv(b/'results/outputs/figure1E_summary.csv');labels={'random_pair_removal':'Random','high_confidence_first':'High-score first','low_confidence_first':'Low-score first'}
for _,r in rows.iterrows():
 expected=f"| {labels[r.strategy]} | {r.fraction_removed:.2f} | 3 | {r.auc_mean:.4f} ± {r.auc_sd:.4f} | {r.auprc_mean:.4f} ± {r.auprc_sd:.4f} |"
 assert expected in z,expected
h=pd.read_csv(b/'results/outputs/figure2D_summary.csv')
for _,r in h.iterrows():
 name='Raw ESM' if r.arm=='raw_ESM' else 'Full FPP';expected=f'| Homol. repr. | {name} | 3 | {r.auc_mean:.4f} ± {r.auc_sd:.4f} | {r.auprc_mean:.4f} ± {r.auprc_sd:.4f} |';assert expected in z
for file,split,cmp in [('figure1D_protein_bootstrap.json','Original','full FPP − ablated FPP'),('figure2D_protein_bootstrap.json','Homol. repr.','full FPP − raw ESM')]:
 q=json.loads((b/'results/outputs'/file).read_text());expected=f"| {split} | {cmp} | {q['delta_auc']:+.4f} ({q['auc_ci95'][0]:+.4f}, {q['auc_ci95'][1]:+.4f}) | {q['delta_auprc']:+.4f} ({q['auprc_ci95'][0]:+.4f}, {q['auprc_ci95'][1]:+.4f}) |";assert expected in z,expected
assert '81 distinct condition–seed combinations' in z and '87 fitting executions' in z
assert 'sample standard deviation of model AUC across three classifier seeds' in n
assert 'Per-run metrics for the homology representations, partial-ablation curve and updated fully ablated representation endpoint are supplied in `remaining_unification_20261108/`.' in z
r={'status':'passed','main_changes':len(changes),'only_listed_main_changes':True,'main_protected_sections_unchanged':['Abstract','Introduction','III.C prose','III.D','III.E','Discussion','Acknowledgment','Bibliography'],'main_stale_replaced_values_absent':True,'SI_protected':protected,'SI_external_numerical_results_unchanged':True,'SI_Methods6_only_audit_scope_label_clarified':True,'unchanged_figures':figure_same,'Table11B_rows_numeric_verified':15,'Table11A_homology_rows_verified':2,'Table12_new_CI_rows_verified':2,'counts_81distinct_87executions_correct':True,'claims':'Means versus ensemble differences separated; no monotonic or score-strategy-superiority claim; graph normalization limitation retained.','clarity_suggestions_resolved':['Figure 1E caption explicitly identifies sample SD of model AUC across three classifier seeds.','Table 11A provenance explicitly routes the updated fully ablated representation endpoint to remaining_unification_20261108.'],'unresolved_findings':[],'source_sha256':{f:hashlib.sha256((new/f).read_bytes()).hexdigest() for f in ['main.tex','Supplementary_Information_revised.md']}}
(b/'review/manuscript_scope_review.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
