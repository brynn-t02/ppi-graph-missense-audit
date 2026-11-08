from pathlib import Path
import json,difflib
R=Path(__file__).resolve().parents[2];W=Path(__file__).resolve().parent
B=R/'outputs/BIBM_workshop_targeted_candidate_v7_unified_20261108'
O=R/'outputs/BIBM_workshop_targeted_candidate_v8_clarity_20261108'
old=(B/'source/main.tex').read_text();s=old;changes=[]
def replace(a,b,why):
 global s
 assert s.count(a)==1,(why,s.count(a));s=s.replace(a,b);changes.append({'before':a,'after':b,'reason':why})
replace('B, Mean AUC and AUPRC for raw ESM and full FPP under matched classifier seeds; error bars denote sample standard deviation.', 'B, AUC and AUPRC for raw ESM and full FPP under matched classifier seeds.', 'Avoid duplicate B error-bar explanation; unified B/D/F explanation below.')
replace('C, Paired ensemble differences under variant-level and protein-clustered bootstrap resampling.',r'C, Paired ensemble differences with 95\% intervals from variant-level and protein-clustered bootstrap resampling.','User exact Figure1C interval clarification.')
replace('F, Topology-permutation control using three graph permutations and three classifier seeds per graph. Bootstrap intervals are conditional on fitted models and the evaluated graphs.', 'F, Topology-permutation control using three graph permutations and three classifier seeds per graph. Bars in B, D and F show mean metrics with sample standard deviations; for the permutation control, the pooled standard deviation reflects variation across graph permutations and classifier seeds. Bootstrap intervals are conditional on fitted models and the evaluated graphs.', 'User exact B/D/F and pooled permutation-SD explanation; Figure1E unchanged.')
replace('E, Three-seed matched fusion comparisons under the homology-aware split.', 'E, Three-seed matched fusion comparisons under the homology-aware split; bars show mean metrics and error bars denote sample standard deviations.', 'User exact Figure2E mean/SD clarification.')
# The optional Discussion addition was trial-compiled and produced9pages.
# Omitted under the user's explicit8page/layout condition; existing Results retained.
(O/'source/main.tex').write_text(s)
(W/'main_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
(W/'main.diff').write_text(''.join(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='v7/main.tex',tofile='v8/main.tex')))
print('exact edits',len(changes))
