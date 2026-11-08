from pathlib import Path
import re, json, shutil, hashlib, difflib

W=Path(__file__).resolve().parents[1]
ROOT=W.parents[1]
B=ROOT/'outputs/BIBM_workshop_original_second_network_only_20261108'
S=B/'source'
O=Path('/Users/teyian/Documents/paper1/release_candidate_20260605')
original=(O/'manuscript_notes/manuscript_bibm_v20_ieee_draft.tex').read_text()
text=original
changes=[]
def replace(old,new,reason):
    global text
    assert text.count(old)==1, (reason,text.count(old),old[:90])
    text=text.replace(old,new)
    changes.append({'reason':reason,'before':old,'after':new})
def paragraph(start,new,reason):
    old=next(p for p in text.split('\n\n') if p.startswith(start))
    replace(old,new,reason)
def caption(label,new):
    match=re.search(r'\\caption\{([^\n]*)\}\n\\label\{'+re.escape(label)+r'\}',text)
    assert match,label
    replace(match.group(1),new,'Caption '+label)

replace(r'\documentclass[conference]{IEEEtran}',r'\RequirePackage[T1]{fontenc}'+'\n'+r'\documentclass[conference]{IEEEtran}','Actual embedded bold/italic fonts')
replace(r'\usepackage{url}',r'\usepackage[hyphens]{url}'+'\n'+r'\usepackage{array}','Compact tables and breakable repository URL')
text=re.sub(r'\\special\{pdf:literal direct [^\n]*\}\n','',text)
replace(r'\author{}',r'''\author{\IEEEauthorblockN{Yuxin Tian, Xianguo Zhang\textsuperscript{*}, Weibo Meng, Jieqi Yin, and Yongqin Jin}
\IEEEauthorblockA{College of Computer Science (College of Software), College of Artificial Intelligence\\
Inner Mongolia University, Hohhot 010021, China\\
Email: yuxin\_t02@163.com; 2595083628@qq.com\\
\textsuperscript{*}Corresponding author: Xianguo Zhang}}
\hypersetup{pdfauthor={Yuxin Tian, Xianguo Zhang, Weibo Meng, Jieqi Yin, Yongqin Jin}}''','Previously confirmed authors')
replace(r'\maketitle',r'\maketitle'+'\n'+r'\raggedbottom','Avoid stretched paragraph gaps')
start=text.index('Protein-protein interaction (PPI) graphs provide')
end=text.index('\n\n\\end{abstract}',start)
replace(text[start:end],r'''Protein-protein interaction (PPI) graphs can add molecular context to missense variant prediction, but their contribution depends on evaluation design and baseline strength. We audit fixed two-hop propagation of ESM protein embeddings using 177,213 ClinVar variants and a STRING association graph. Only downstream classifiers are trained. Across three seeds, propagation increased mean AUC from 0.7835 to 0.7950; protein-clustered intervals for the reference-model gains included zero. Connectivity and topology controls showed performance sensitivity, although their propagation and training settings were not fully matched. With CADD and AlphaMissense included, protein-representation controls retained much of the fusion gain. Under homology-aware splitting, the single graph-fusion estimate was below the repeated strong-predictor baseline. Topology groups and error transitions showed exploratory associations rather than established correction mechanisms. BioGRID second-network validation did not establish a consistent added benefit over raw-embedding controls. These findings support context-dependent protein-representation complementarity and careful evaluation of network-specific increments. They concern fixed propagation and should not be generalized to trained graph neural networks.''','Short abstract and scope; only BioGRID added')
replace('a controlled evaluation framework that separates graph-topology effects from representation complementarity, train-test graph connectivity and downstream classifier capacity','an evaluation framework that examines graph topology, representation complementarity, train-test graph connectivity and downstream classifier configuration','Introduction avoid isolated causal effects')
replace('Third, it links graph-associated gains and prediction changes to topology contexts and error transitions, offering practical guidance on when PPI context adds useful information in missense variant prediction.','Third, it relates graph-associated gains and prediction changes to exploratory topology contexts and error transitions. A second-network evaluation using BioGRID tests the dependence of the findings on network choice.','Introduction exploratory scope and second network')
replace('We used the ClinVar variant-summary release dated 31 March 2024 and retained checksums for the exact input file used in the reconstruction.','Checksums and sample identifiers freeze the processed benchmark; its exact ClinVar content-release date could not be verified.','Do not restore unverified content-release date')
replace('The implementation symmetrized the retained edge list, added self-loops and coalesced duplicate edges. After self-loop addition, let $\\tilde{A}$ denote the binary adjacency matrix and $\\tilde{D}$ its degree matrix.',r'The full-graph implementation added self-loops before symmetrization and summed duplicate entries, giving two self-loop contributions. Let $\tilde{A}$ denote this weighted adjacency matrix and $\tilde{D}$ its degree matrix.','Correct actual full propagation operator')
replace("The full graph contained train, validation and test nodes with their ESM features, making the evaluation transductive.","The full graph contained train, validation and test nodes with their ESM features, making the evaluation transductive. Protein-only inputs cannot distinguish variants within the same protein.",'Clarify protein-level prediction boundary')
replace('used the 512/128 classifier to keep downstream capacity matched.','used the same 512/128 hidden-layer architecture and preprocessing, with different input-layer parameter counts.','Architecture matching is not total parameter matching')
replace('a maximum of 50 epochs and early stopping','a maximum of 50 epochs (30 for the representation ablation/permutation controls) and early stopping','Disclose actual training budgets')
replace('Repeated classifier comparisons used seeds 42, 43 and 44.','Seeds and repeat counts varied by analysis (Table~\\ref{tab:runs}; Supplementary Table 8).','Do not imply same seeds everywhere')
replace('Fusion controls concatenated \\CADDAM{} features with raw ESM, edge-ablated PPI-GCN and topology-permuted PPI-GCN representations, separating protein-representation complementarity from topology-specific effects.','Fusion controls concatenated \\CADDAM{} features with raw ESM, edge-ablated PPI-GCN and topology-permuted PPI-GCN representations. The ablated/permuted encoders added one self-loop after symmetrization, unlike the full encoder. These comparisons do not isolate topology from normalization and training differences.','Disclose operator mismatch without rerunning original figures')
replace('Multiple topology-feature comparisons used Benjamini-Hochberg false-discovery-rate correction \\cite{Benjamini1995FDR}.','Topology-feature profiles are descriptive: multiple variants share each protein\'s features, and false-discovery-rate correction \\cite{Benjamini1995FDR} does not address this dependence. No inferential significance is claimed for these profiles.','Withdraw invalid feature-wise inference')

insert=r'''\begin{table}[t]
\caption{Summary of fitted models and repeated runs.}
\label{tab:runs}
\centering
\footnotesize
\begin{tabular}{@{}p{0.43\columnwidth}p{0.51\columnwidth}@{}}
\hline
Analysis & Fits and configuration \\
\hline
Fig. 1B,C; Fig. 2D & One reference fit per model \\
Fig. 1D: edge ablation & Five ablated fits; 30 epochs \\
Fig. 1E: partial ablation & Three per interior setting; 30 epochs; endpoints reused \\
Fig. 1F: permutation & Three graphs $\times$ three seeds; 30 epochs \\
Fig. 2B,C: original fusion & Three per arm; nine for permutation \\
Fig. 2E: homology fusion & Scores/raw fusion: three; graph fusion: one \\
Figs. 3B--F and 4 & Fixed reference prediction pairs \\
BioGRID comparison & Three per setting; 50 epochs \\
\hline
\end{tabular}
\par\smallskip\raggedright\footnotesize
The text also reports three-seed raw/full representation means. Fig. 1D--F reuse single-fit raw/full references. Bootstrap resamples are not classifier fits. Complete seeds, budgets and SD definitions are in Supplementary Table 8.
\end{table}

\subsection{Second-network validation}

BioGRID 5.0.262 human physical interactions \cite{Stark2006BioGRID} were mapped to the same 19,622 nodes, yielding 931,458 undirected edges after excluding RNA-related methods and ambiguous mappings. Nine classifiers covered representations under the original split and score fusion under both splits. Within this comparison, BioGRID and STRING used the same full-graph propagation convention; all models shared seeds 42--44, validation selection and a 50-epoch budget with patience five. Predictions were aligned by variant and protein identifiers and averaged in float64. Paired protein-bootstrap 95\% intervals used 2,000 resamples conditional on fitted models, without retraining or multiplicity adjustment (Supplementary Methods 5 and Table 9).

'''
replace(r'\section{Results}',insert+r'\section{Results}','Short run table and self-contained BioGRID protocol')
replace('Full-graph PPI-GCN gains depend on graph connectivity','Full-graph PPI-GCN and connectivity sensitivity','Avoid causal title for unmatched original controls')
replace('Across three repeated classifier seeds, the comparison remained positive with a smaller mean AUC difference:','Across three classifier seeds, mean metrics (population SD) showed a smaller positive AUC difference:','Label original ddof=0 statistics accurately')
replace('Relative to raw ESM, the AUC gain was nearly eliminated ($\\Delta$ AUC +0.0001); a smaller AUPRC gain remained ($\\Delta$ AUPRC +0.0181).','Relative to the raw-ESM reference, these point differences were $\\Delta$ AUC +0.0001 and $\\Delta$ AUPRC +0.0181; they do not establish equivalence.','Keep original numbers without effect-elimination claim')
replace('The retained full-graph gain depended on protein-specific topology alignment, and its AUC component remained sensitive to cross-split connectivity.','These sensitivity comparisons are descriptive because operator, seed and training-budget differences prevent attribution solely to topology or direct connectivity.','Keep controls with clear limitations')
caption('fig:connectivity',r'Workflow and primary audit of fixed PPI propagation. A, Audit design. B, Single-fit raw/full reference comparison; three-seed means are reported in the text. C, Paired differences with variant-level and protein-clustered bootstrap 95\% intervals; definitions are in Supplementary Table 4A. D, Direct train-test edge ablation. E, Partial ablation with three fits per interior setting, a single-fit full endpoint and a five-fit ablated endpoint. F, Topology permutation. D and F compare five ablated and nine permuted fits with single-fit raw/full references. Error bars show sample SD where repeated fits exist; permutation SD pools graph and classifier variation. Operator and training differences limit causal interpretation of these controls.')
replace('with a small topology-specific margin for real PPI-GCN (Fig. 2B,C).','with a small additional margin for real PPI-GCN (Fig. 2B,C).','Avoid isolating topology in fusion claim')
replace('Under the homology-aware split, PPI-GCN remained higher than raw ESM,','In the single-fit homology-aware representation comparison, PPI-GCN was higher than raw ESM,','Explicit one fit')
replace('A retained-reference bootstrap based on a different comparison protocol yielded the same direction. ','','Remove confusing comparison to different protocol')
caption('fig:fusion_homology',r'Strong-predictor fusion and homology-aware evaluation. A, Score matching and fusion-control setup; architecture matching refers to hidden layers, not equal input-parameter counts. B, Original-split fusion means and sample SD across three fits per arm, except permutation (three graphs $\times$ three seeds). C, Full-graph fusion differences from ensemble-averaged probabilities with paired protein-bootstrap 95\% intervals; these need not equal differences between B\textquotesingle s mean metrics and condition on fitted models and sampled graphs. D, Single-fit homology-aware representation comparison. E, Homology-aware fusion: scores/raw fusion use three fits with sample SD; graph fusion is a single estimate. F, Test-set positive prevalence. The ablated/permuted and full encoders differ in self-loop convention.')
replace('Graph gains concentrate in specific topology contexts','Exploratory graph gains across topology contexts','Cautious topology heading')
replace('ESM representation space showed weak but detectable correspondence with PPI topology.','ESM representation space showed modest correspondence with PPI topology.','Remove implicit significance')
replace('Betweenness and clustering also shaped gain profiles:','Performance differences also varied across betweenness and clustering strata:','Association rather than mechanism')
replace('Neighbour-risk stratification and joint topology maps localized the graph signal to specific contexts.','Neighbour-risk stratification and joint topology maps showed heterogeneous performance differences.','Exploratory wording')
replace('The bootstrap intervals characterize within-cell sampling uncertainty for these dataset-selected profiles.','These profiles were selected after examining 53 joint cells on the same test set. Their variant-level intervals do not account for within-protein dependence, subgroup selection or multiple comparisons, and the findings require independent validation.','Explicit exploratory selection and uncertainty')
caption('fig:topology_context',r'Exploratory topology-context analysis. A, ESM similarity across network relationships. B--D, Performance differences across network strata. E, AUC differences across 53 joint cells. F, Selected profiles with variant-level bootstrap intervals. B--F use a fixed reference pair; these descriptive intervals exclude protein dependence, subgroup selection and multiplicity. Panel titles describe associations, not established mechanisms.')
replace('Multiple topology features remained significant after FDR correction (Fig. 4E), and difficulty-matched contrasts retained differences across degree, k-core, neighbour-risk, neighbour pathogenic rate, betweenness and boundary role (Fig. 4F). These descriptive associations connect graph-related prediction changes with network context.','Figure 4E shows directions of group-mean differences. Figure 4F compares rescued and persistent false negatives within baseline false-negative cases, without continuous prediction-difficulty matching. These are descriptive associations; repeated variants within proteins preclude interpreting the feature-wise significance encodings as valid evidence.','Correct original Figure4 inference and matching claims without changing picture')
caption('fig:prediction_transitions',r'Prediction transitions and descriptive topology profiles. A--B, Raw ESM to PPI-GCN and score-only to graph-fusion transitions at threshold 0.5, using fixed reference pairs. C--D, Row-centered, range-scaled feature means: warm/blue indicates above/below the row mean. E, Blue/orange indicates higher means in rescued false negatives/suppressed false positives. F, Hedges\textquotesingle{} $g$ comparing rescued and persistent false negatives within baseline false-negative cases, not difficulty matching; blue/orange denotes positive/negative effects. Significance encodings in E--F do not account for protein dependence and are not used for inference.')

network=r'''\subsection{Second-network validation with BioGRID}

BioGRID did not show a consistent additional benefit over raw-embedding controls (Table~\ref{tab:biogrid}). Under the original split, its fusion improved on scores alone (ensemble $\Delta$AUC +0.0124, protein-bootstrap 95\% CI [+0.0034, +0.0210]), but not clearly on raw-ESM fusion (+0.0009, [-0.0011, +0.0036]). The homology-aware BioGRID fusion estimate was below the score baseline ($\Delta$AUC -0.00204, [-0.00421, -0.00004]). These results limit claims of network-independent gains; complete paired controls and AUPRC intervals are in Supplementary Table 9.

\begin{table}[t]
\caption{BioGRID results under shared training settings.}
\label{tab:biogrid}
\centering\footnotesize\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}>{\raggedright\arraybackslash}p{0.26\columnwidth}>{\raggedright\arraybackslash}p{0.24\columnwidth}>{\raggedright\arraybackslash}p{0.42\columnwidth}@{}}
\hline
Setting & Mean AUC / AUPRC & Ensemble $\Delta$AUC vs raw [95\% CI] \\
\hline
Original representations & 0.7592 / 0.6043 & -0.0201 [-0.0480, +0.0101] \\
Original fusion & 0.9693 / 0.9344 & +0.0009 [-0.0011, +0.0036] \\
Homology fusion & 0.9730 / 0.8265 & +0.0005 [-0.0008, +0.0017] \\
\hline
\end{tabular}
\par\smallskip\raggedright\footnotesize
Each setting uses three fits per model. Raw denotes raw ESM for representations and raw-ESM fusion for fusion rows. Differences use mean probabilities and paired protein bootstrap within this comparison; SD and all comparator metrics are in Supplementary Table 9.
\end{table}

'''
replace(r'\section{Discussion}',network+r'\section{Discussion}','Only added empirical validation: BioGRID network')
replace('and the gain was sensitive to protein-topology correspondence and cross-split connectivity.','while the graph controls suggested sensitivity to connectivity and topology under differing operator and training settings.','Discussion preserve confounding boundary')
replace('with a smaller real-topology margin.','with a smaller full-graph margin.','Avoid isolated topology mechanism')
replace('The prevalence shift defines this analysis as a stress test. Graph complementarity depends on the baseline predictor, split design and graph-control baseline.','Redundant high-dimensional features, extra input-layer parameters, overfitting and feature scaling may explain the limited increment, but were not tested individually. The single graph-fusion fit also limits conclusions about a stable disadvantage. The prevalence shift limits cross-split AUPRC comparison.','Reviewer homology explanation with uncertainty')
paragraph('Graph propagation changed predictions in structured topology contexts.',r'Graph propagation changed predictions in heterogeneous topology contexts. These post-hoc associations do not establish correction mechanisms. BioGRID validation further shows that fusion complementarity need not imply a reliable graph-specific increment. STRING and BioGRID differ in density, coverage and evidence types and overlap in interactions, so their comparison does not isolate biological evidence quality. Independent cohorts would be needed to assess transfer beyond the ClinVar benchmark.','Exploratory discussion and limits of second network')
replace('Within these constraints, PPI graph augmentation provides context-dependent network information alongside strong sequence and variant-effect predictors.','The conclusions concern fixed propagation, with only downstream classifiers trained, and should not be generalized to trained graph neural networks.\n\n\\noindent\\textit{Practical recommendations.} Compare graph features with raw embeddings and strong score baselines; report run configurations and protein-level uncertainty. Treat selected topology groups as hypotheses and check decision thresholds and calibration before clinical use.','Fixed GNN scope and concise practical recommendations')
paragraph('A compact anonymized source-data archive',r'Supplementary Information, code, compact source data, predictions, protocols and checksums are publicly available at \url{https://github.com/brynn-t02/ppi-graph-missense-audit}. Large third-party resources should be obtained from their original providers.','Verified public materials including SI')
replace(r'\section*{AI-Tool Disclosure}',r'\section*{Acknowledgment}','IEEE conventional heading')
paragraph('OpenAI ChatGPT, OpenAI Codex and Anthropic Claude were used',r'OpenAI ChatGPT, OpenAI Codex and Anthropic Claude assisted with literature organization, code development and language editing. The authors take responsibility for study design, experimental execution and verification, code review, interpretation, references and the final manuscript.','Accurate concise tool disclosure and author responsibility')

# Preserve the original 28 reference texts; append only the BioGRID resource citation.
bio=r'\bibitem{Stark2006BioGRID} C. Stark, B.-J. Breitkreutz, T. Reguly, L. Boucher, A. Breitkreutz, and M. Tyers, ``BioGRID: a general repository for interaction datasets,\'\' \emph{Nucleic Acids Research}, vol. 34, pp. D535--D539, 2006, doi: 10.1093/nar/gkj109.'
bio=bio.replace("\\'\\'", "''")
replace(r'\end{thebibliography}',bio+'\n'+r'\end{thebibliography}','Only new reference: BioGRID')
pre,bib=text.split(r'\begin{thebibliography}{99}')
entries=dict(re.findall(r'\\bibitem\{([^}]+)\}([^\n]+)',bib))
order=[]
for group in re.findall(r'\\cite\{([^}]+)\}',pre):
    for key in group.split(','):
        if key not in order:order.append(key)
assert set(order)==set(entries),(set(entries)-set(order))
text=pre+r'\begin{thebibliography}{99}'+'\n'+'\n'.join(r'\bibitem{'+k+'}'+entries[k] for k in order)+'\n'+r'\end{thebibliography}'+'\n'+r'\end{document}'+'\n'

replace(r'\begin{thebibliography}{99}',r'\IEEEtriggeratref{25}'+'\n'+r'\begin{thebibliography}{99}','Balance final reference columns using IEEEtran hook')

S.mkdir(parents=True,exist_ok=True)
(S/'figure_drafts').mkdir(exist_ok=True)
(S/'main.tex').write_text(text)
shutil.copy2('/Users/teyian/Documents/paper1/IEEEtran.cls',S/'IEEEtran.cls')
for file in (O/'figures').glob('*accepted.pdf'):shutil.copy2(file,S/'figure_drafts'/file.name)
fonts=ROOT/'outputs/BIBM_workshop_targeted_candidate_v8_clarity_20261108/source/resources/fonts'
shutil.copytree(fonts,S/'resources/fonts',dirs_exist_ok=True)
(B/'original_submission').mkdir(exist_ok=True)
for old,new in [('manuscript_bibm_v20_ieee_draft.tex','main_original.tex'),('BIBM2026_main_paper_anonymous.pdf','Original_submission_main.pdf'),('Supplementary_Information_v1.md','Supplementary_Information_original.md'),('BIBM2026_supplementary_anonymous.pdf','Original_submission_supplement.pdf')]:
    shutil.copy2(O/'manuscript_notes'/old,B/'original_submission'/new)
(B/'reproducibility').mkdir(exist_ok=True)
(W/'main_changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
(B/'reproducibility/main_before_after.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
(B/'reproducibility/main.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='original_submission/main_original.tex',tofile='source/main.tex')))
print(B)
print('Changed blocks',len(changes),'Abstract words',len(re.search(r'\\boldmath\n(.*?)\n\n\\end\{abstract\}',text,re.S).group(1).split()))
