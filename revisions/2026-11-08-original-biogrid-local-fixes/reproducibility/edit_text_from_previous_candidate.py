from pathlib import Path
import json,re,difflib,hashlib
R=Path(__file__).resolve().parents[3]
W=R/'work/original_BioGRID_local_fixes_20261108'
OLD=R/'outputs/BIBM_workshop_original_second_network_only_20261108'
NEW=R/'outputs/BIBM_workshop_original_BioGRID_local_fixes_20261108'
changes=[]
main=(OLD/'source/main.tex').read_text()
si=(OLD/'source/Supplementary_Information_revised.md').read_text()
def change(target,old,new,reason):
 global main,si
 s=main if target=='main' else si
 assert s.count(old)==1,(target,reason,s.count(old))
 s=s.replace(old,new)
 if target=='main':main=s
 else:si=s
 changes.append({'document':target,'reason':reason,'before':old,'after':new})

change('main','Topology groups and error transitions showed exploratory associations rather than established correction mechanisms.','Exploratory analyses described variation across topology groups and prediction-error transitions.','F1: neutral abstract sentence; no wholesale rewrite')
change('main','Third, it relates graph-associated gains and prediction changes to exploratory topology contexts and error transitions.','Third, it relates graph-associated gains and prediction changes to topology contexts and error transitions through exploratory analyses.','F2: exploratory modifies analyses')
change('main',r'Similar profile shifts appeared in the retained \CADDAM{} to graph-fusion comparison (Fig. 4D).','For graph fusion, the degree and neighbour-risk contrasts had the same direction, whereas the clustering-coefficient contrast reversed (Fig. 4D).','F5: accurately describe clustering direction reversal')
change('main','These are descriptive associations; repeated variants within proteins preclude interpreting the feature-wise significance encodings as valid evidence.','We interpret these feature contrasts descriptively because variants mapped to the same protein share network features.','A3: body synchronized with significance-encoding removal')
change('main',r'F, Hedges\textquotesingle{} $g$ comparing rescued and persistent false negatives within baseline false-negative cases, not difficulty matching; blue/orange denotes positive/negative effects. Significance encodings in E--F do not account for protein dependence and are not used for inference.',r'F, Descriptive Hedges\textquotesingle{} $g$ comparing rescued and persistent false negatives within the raw-ESM baseline false-negative group; blue and orange denote positive and negative effects. Feature comparisons are descriptive and make no inferential significance claims.','A3: caption describes actual F contrast and no significance markers')
change('main',r'B--F use a fixed reference pair; these descriptive intervals exclude protein dependence, subgroup selection and multiplicity. Panel titles describe associations, not established mechanisms.',r'B--F use a fixed reference pair. The within-cell variant-bootstrap intervals do not account for within-protein dependence or post-hoc subgroup selection and are not adjusted for multiple comparisons.','C: precise uncertainty description; remove commentary about titles')
change('main',r'E, Homology-aware fusion: scores/raw fusion use three fits with sample SD; graph fusion is a single estimate.',r'E, Homology-aware fusion: the score-only and raw-ESM fusion bars show means across three fits, whereas graph fusion shows one fitted-model estimate. Standard deviations for the repeated fits are reported in Supplementary Table 3B.','D1: Figure2E has no displayed error bars; SD remains in SI')
change('main',r'C, Full-graph fusion differences from ensemble-averaged probabilities with paired protein-bootstrap 95\% intervals; these need not equal differences between B\textquotesingle s mean metrics and condition on fitted models and sampled graphs.',r'C, Full-graph fusion differences computed from ensemble-averaged probabilities, with paired protein-bootstrap 95\% intervals conditional on the fitted models and sampled graphs. The point estimates need not equal differences between the mean metrics in B.','D2: distinguish ensemble differences and mean metrics clearly')
change('si','Figure 4E/F retains dot-size, opacity and significance labels from those tests, but these encodings must not be interpreted as valid inferential evidence.','Figure 4E uses uniform dots to indicate the direction of group-mean differences, and Figure 4F displays descriptive effect sizes without significance markers.','A3: SI synchronized with actual E/F encoding changes')
change('si','They describe exploratory associations and require independent validation; the stronger wording of Figure 3 panel titles should not be interpreted as an established mechanism.','They describe exploratory associations and require independent validation.','C: remove commentary about superseded figure titles')
change('si','These rows describe the three-seed network comparison. They should not be substituted for separate single-fit results or treated as the metrics of ensemble-averaged predictions. CADD+AM denotes the architecture-matched score baseline; every fusion model includes CADD, AlphaMissense and its missingness indicator. These comparisons use their own matched fits and canonical protein-level prediction mapping. Consequently, their means, SDs and ensemble differences should not be combined with the fixed reference points or separately fitted controls in Tables 2–4.','Table 9 reports the BioGRID network comparison together with its matched comparator models. Entries in Table 9A are means and sample standard deviations across three fits; Table 9B reports paired differences computed from averaged prediction probabilities. All within-table contrasts use the models and evaluation protocol described in Supplementary Methods 5.','E: direct explanation of within-table comparison protocol')

for rel,new in [('main.tex',main),('Supplementary_Information_revised.md',si)]:
 old=(OLD/'source'/rel).read_text()
 (NEW/'source'/rel).write_text(new)
 (NEW/'reproducibility'/(rel+'.diff')).write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='previous/source/'+rel,tofile='source/'+rel)))
(NEW/'reproducibility/text_before_after.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
(W/'text_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n')
# All literal Markdown tables and all substantive metric values stay unchanged.
old_si=(OLD/'source/Supplementary_Information_revised.md').read_text()
assert [x for x in si.splitlines() if x.startswith('|')]==[x for x in old_si.splitlines() if x.startswith('|')]
for original,revised in [((OLD/'source/main.tex').read_text(),main),(old_si,si)]:
 assert re.findall(r'(?<![A-Za-z])[-+]?\d+\.\d+',original)==re.findall(r'(?<![A-Za-z])[-+]?\d+\.\d+',revised)
print('Applied',len(changes),'targeted text edits; all table rows and decimal values unchanged.')
