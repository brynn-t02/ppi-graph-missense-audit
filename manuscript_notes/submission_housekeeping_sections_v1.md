# Submission Housekeeping Sections v1

Date: 2026-06-04

Current manuscript basis:
- `/Users/teyian/Documents/paper1/manuscript_english_working_draft_v18_submission_reading.md`

Purpose:
- Draft submission-facing statements that are usually required by journals.
- Use conservative placeholders where information is not yet known.
- Avoid inventing author, funding, repository, license or redistribution details.

## Data Availability

Draft:

The analyses used publicly available resources and locally generated processed files. ClinVar variant annotations were obtained from the official NCBI ClinVar FTP `tab_delimited` directory as `variant_summary.txt.gz`; the FTP listing recorded the file release timestamp as 31 March 2024, and the retained copy used in the present rebuild pipeline was archived locally on 3 April 2026. STRING v12 human protein-protein association data and protein sequences were used to construct the PPI graph and homology-aware split. CADD GRCh38 v1.7 whole-genome SNV scores and AlphaMissense hg38 scores were used as external variant-effect predictors. Processed tables supporting the figures, trained-test prediction outputs, summary metric files and figure source data will be made available at `[repository or archive URL to be added]`, subject to the redistribution terms of the original data providers. Large third-party source files, including whole-genome CADD score files, should be obtained from their original providers.

Notes to verify before submission:
- Confirm whether processed ClinVar-derived tables can be redistributed directly or should be shared as scripts plus derived non-identifying summary tables.
- Confirm STRING license/redistribution conditions for graph edge lists and protein identifiers.
- Confirm CADD and AlphaMissense redistribution terms. CADD whole-genome score files are large third-party resources and should probably be referenced rather than mirrored.
- Add final archive/repository URL only after the repository is created.

## Code Availability

Draft:

Code used for data reconstruction, ESM feature generation, graph propagation, model training, statistical analyses and figure generation will be made available at `[repository or archive URL to be added]`. The repository will include scripts for reproducing the main figures and summary tables from processed inputs, together with environment information and instructions for obtaining third-party source data.

Notes to verify before submission:
- Decide whether to release the current script collection as a cleaned GitHub repository, a Zenodo archive, supplementary code, or a combination.
- Separate reproducible scripts from exploratory notebooks or obsolete intermediate scripts.
- Include a minimal README that maps scripts to manuscript figures and supplementary tables.
- Do not promise one-click full regeneration unless the repository actually supports it.

## Supplementary Information

Draft:

Supplementary information should include additional methodological details, architecture-control results, negative-control analyses, repeated-run summaries, calibration and PR-curve summaries, and additional subgroup analyses. The supplement should be organized around reviewer questions rather than chronological experiment order.

Proposed supplementary structure:

1. Supplementary Methods: dataset filtering, identifier mapping, ESM feature generation, graph construction, classifier training, homology-aware split construction and statistical testing.
2. Supplementary Table 1: split sizes, label counts, protein counts and graph statistics.
3. Supplementary Table 2: raw ESM, PPI-GCN, GraphSAGE-style and GAT-style representation-control results.
4. Supplementary Table 3: CADD, AlphaMissense and fusion model performance, including repeated-run summaries.
5. Supplementary Table 4: negative-control and random-control results, including the rerun feature-shuffled GCN control.
6. Supplementary Table 5: homology-aware split metrics and prevalence summary.
7. Supplementary Table 6: topology-stratified and corrected-sample statistical tests.
8. Supplementary Figure 1: PR curves and split-prevalence context.
9. Supplementary Figure 2: calibration/Brier/sensitivity summaries, if retained.
10. Supplementary Figure 3: quartile-stratified CADD and AlphaMissense analyses, if retained.

Notes to verify before submission:
- Avoid a "supplementary black hole"; each item should answer a concrete reviewer concern.
- Keep older random-control results cautious unless provenance is fully documented or rerun.
- Decide whether calibration/clinical-threshold analyses are useful enough to include without implying clinical deployment.

## Author Contributions

Draft template:

`[Author initials]` conceived the study. `[Author initials]` designed the computational experiments. `[Author initials]` performed data reconstruction, model training and statistical analyses. `[Author initials]` generated figures. `[Author initials]` interpreted the results. `[Author initials]` wrote the manuscript with input from all authors. All authors reviewed and approved the final manuscript.

Notes to verify before submission:
- Replace placeholders with real author initials.
- If AI tools contributed to writing or analysis, follow the target journal's AI-tool disclosure policy. Do not list AI as an author.

## Acknowledgements

Draft template:

We thank `[names or groups to be added]` for discussions and feedback. This work used publicly available data from ClinVar, STRING, CADD and AlphaMissense.

Notes to verify before submission:
- Add institutional, advisor, lab or collaborator acknowledgements only if appropriate.
- Add compute-resource acknowledgement if required by AutoDL, institution or funding source.

## Funding

Draft template:

This work was supported by `[funding source and grant number to be added]`.

Alternative if no external funding:

The authors declare that no specific funding was received for this work.

Notes to verify before submission:
- Do not invent funding source or grant number.
- Use the exact funding wording required by the funder, if any.

## Competing Interests

Draft template:

The authors declare no competing interests.

Notes to verify before submission:
- Confirm with all authors before final submission.
- If any author has relevant employment, consulting, equity, patent or paid advisory relationships, this must be disclosed.

## Ethics Statement

Draft:

This study used publicly available aggregate variant annotations and did not involve direct recruitment of human participants, intervention, or access to identifiable individual-level clinical records. Any institutional review requirement should be assessed according to the policies of the authors' institution before submission.

Notes to verify before submission:
- Confirm whether the target journal requires a formal ethics statement even for public-data computational studies.
- Confirm institutional policy before using "not required" language.

## AI Tool Disclosure

Draft template:

During manuscript preparation, AI-assisted tools were used for language editing, organization and drafting support. All scientific claims, numerical results, citations and interpretations were checked by the authors against local analysis outputs or cited sources. AI tools were not used as authors.

Notes to verify before submission:
- Use only if the target journal requires or allows AI-tool disclosure.
- Adjust wording to match the exact journal policy.
- This statement should not imply that AI generated unverified data or results.

## Current Recommendation

Do not merge these sections into the v18 manuscript until the placeholders are resolved. The safest next move is to create a clean repository/expanded archive plan, then fill Data Availability and Code Availability with concrete links and redistribution boundaries.

## Decision Checklist v1

This checklist is kept in the same housekeeping file to avoid creating another small planning file. It records which details must be supplied by the authors before these sections can be moved into the submission manuscript.

### Required before formal submission

1. Repository or archive plan
   - Decision needed: GitHub, Zenodo, OSF, institutional repository, journal supplementary code, or a combination.
   - Current safest path: GitHub for readable code plus Zenodo/OSF archive for a citable frozen version, if the authors are comfortable creating a public repository.
   - Do not write a final URL until the repository/archive exists.

2. Data redistribution boundary
   - Decision needed: whether processed ClinVar-derived variant tables can be shared directly, or whether the repository should provide scripts plus summary/source-data tables.
   - Current conservative wording: third-party source files such as whole-genome CADD scores should be obtained from original providers rather than mirrored.
   - Keep this conservative unless the original licences are checked.

3. Author initials and contribution roles
   - Decision needed: author list and initials.
   - Need to know who did: study conception, experiment design, data reconstruction, model training, statistical analysis, figure generation, manuscript writing and manuscript review.
   - Do not list any AI tool as an author.

4. Funding
   - Decision needed: exact funding source and grant number, or confirmation that no specific funding was received.
   - If no funding: use "The authors declare that no specific funding was received for this work."

5. Competing interests
   - Decision needed: confirmation from all authors.
   - If none: use "The authors declare no competing interests."

6. Ethics statement
   - Decision needed: whether the authors' institution requires a formal review or exemption statement for public-data computational analysis.
   - Current conservative wording avoids saying "ethics approval was not required" until institutional policy is confirmed.

7. AI-tool disclosure
   - Decision needed: target journal policy and author preference.
   - Current truthful core: AI-assisted tools were used for language editing, organization and drafting support; all claims, numbers, citations and interpretations were checked by the authors; AI tools were not authors.

8. Acknowledgements
   - Decision needed: advisor, lab, collaborator, compute platform or institutional acknowledgement.
   - Do not add names without permission.

### Items already reasonably supported

- ClinVar source: official NCBI ClinVar FTP `tab_delimited` directory, file `variant_summary.txt.gz`.
- ClinVar official FTP release timestamp visible in the confirmed listing: 31 March 2024.
- Retained analysis copy archived locally/server-side on 3 April 2026.
- STRING v12, CADD GRCh38 v1.7 and AlphaMissense hg38 are the third-party resources to cite and route users back to.
- Data availability should distinguish figure/source-data tables from large third-party files.

### Recommended wording stance

- Use "will be made available" only after the repository/archive path is practically decided.
- Use "subject to original-provider redistribution terms" for processed data until licences are reviewed.
- Avoid "all data are included" because large CADD/STRING/AlphaMissense files should not be mirrored casually.
- Avoid "fully reproducible from scratch" unless the final code package really contains all scripts, environment information and source-data acquisition instructions.

## Housekeeping Update v2

Date: 2026-06-05

Newly resolved item:

- A compact source-data package has been prepared at `/Users/teyian/Documents/paper1/source_data_package_v1` and archived as `/Users/teyian/Documents/paper1/source_data_package_v1.tar.gz`.
- The package contains figure-level and supplementary-table-level CSV/JSON source data, a README and SHA-256 checksums.
- The package intentionally does not duplicate large third-party raw files, model checkpoints, full embedding matrices or all per-variant prediction tables.

### Current Safest Submission-Facing Wording

These drafts can be used once the authors decide where the source-data package and code will be deposited. Do not insert final URLs until they exist.

#### Data availability, conservative version

The analyses used publicly available resources and locally generated processed files. ClinVar variant annotations were obtained from the official NCBI ClinVar FTP `tab_delimited` directory as `variant_summary.txt.gz`; the FTP listing recorded the file release timestamp as 31 March 2024, and the retained copy used in the present rebuild pipeline was archived locally on 3 April 2026. STRING v12 human protein-protein association data and protein sequences were used to construct the PPI graph and homology-aware split. CADD GRCh38 v1.7 whole-genome SNV scores and AlphaMissense hg38 scores were used as external variant-effect predictors. Source data supporting the main figures and supplementary summary tables are provided in the accompanying source-data package and will be deposited at `[repository or archive URL to be added]`. Large third-party source files, including whole-genome CADD score files and other provider-controlled resources, should be obtained from their original providers.

#### Code availability, conservative version

Analysis scripts used for data reconstruction, feature generation, graph propagation, model training, statistical analyses and figure generation will be deposited at `[repository or archive URL to be added]`. The code release will include instructions for obtaining third-party source data and for reproducing the main figures and summary tables from processed inputs. The current manuscript does not claim one-click regeneration of all raw third-party resources or all model checkpoints.

#### AI-tool disclosure, truthful conservative version

During manuscript preparation, AI-assisted tools were used for language editing, organization, drafting support and internal consistency checking. All scientific claims, numerical results, citations and interpretations were checked by the authors against local analysis outputs, source-data files or cited literature. AI tools were not used as authors and were not used to generate or fabricate experimental data.

#### Ethics statement, conservative version

This study used publicly available aggregate variant annotations and computationally derived representations. It did not involve direct recruitment of human participants, intervention, or access to identifiable individual-level clinical records. Whether formal institutional review or exemption documentation is required should be confirmed according to the authors' institutional policies before submission.

### Author Input Needed Before Final Insertion

1. Repository/archive choice:
   - Options: GitHub plus Zenodo, OSF, institutional repository, journal supplementary files, or temporary "available upon reasonable request" if the target venue allows it.
   - Current recommendation: use a public archive for the source-data package and a cleaned code repository/archive when the code is ready.

2. Author list and initials:
   - Need full author names and initials.
   - Need roles for conception, experiment design, data reconstruction, model training, statistical analysis, figure generation, interpretation, writing and review.

3. Funding:
   - Provide exact funding source and grant number, or confirm no specific funding.

4. Competing interests:
   - Confirm whether all authors have no competing interests.

5. Acknowledgements:
   - Decide whether to acknowledge advisor, lab, collaborators, compute platform, institution or data providers beyond formal citations.

6. Ethics wording:
   - Confirm whether the institution requires a formal statement for public-data computational studies.

7. AI disclosure policy:
   - Confirm the target journal/conference policy before final wording.
   - Because AI tools were heavily used in organization and writing support, the safest route is to disclose truthfully and emphasize author verification of all claims, numbers and citations.

### Do Not Write Unless Confirmed

- Do not write a funding grant number without the exact source.
- Do not write "ethics approval was not required" until institutional policy is confirmed.
- Do not write "all data and code are publicly available" until links exist.
- Do not claim full raw-data redistribution because CADD, STRING, AlphaMissense and ClinVar have separate provider terms and large file constraints.
- Do not list any AI system as an author.

## Author-Confirmed Housekeeping Notes v3

Date: 2026-06-05

Current author information supplied by the author:

- Current author list: Tian Yuxin / 田雨欣 only, unless additional authors are later confirmed.
- Standard initials for contribution statements: `Y.T.`.
- Current affiliation supplied by the author: Inner Mongolia University / 内蒙古大学.
- Funding: no specific funding currently reported.
- Competing interests: no competing interests currently reported.
- Acknowledgement: the author expects to acknowledge the supervisor/advisor, Xianguo Zhang / 张献国.
- Ethics: no known special institutional ethics requirement beyond academic integrity; keep wording conservative for public-data computational analysis until the final target venue is selected.
- Target venue status: BIBM is the current working target, but final venue-specific formatting and disclosure language should be adjusted after the final venue is confirmed.

### Provisional Final Statements Based on Current Author Input

#### Author contributions, single-author provisional version

Y.T. conceived the study, designed the computational experiments, performed data reconstruction, model training and statistical analyses, generated figures, interpreted the results and wrote the manuscript.

Note: if an advisor or collaborator is later included as an author, this statement must be revised to reflect real contributions.

#### Funding, provisional version

The author declares that no specific funding was received for this work.

#### Competing interests, provisional version

The author declares no competing interests.

#### Acknowledgements, provisional version

The author thanks Xianguo Zhang for supervision, guidance and feedback. This work used publicly available data from ClinVar, STRING, CADD and AlphaMissense.

Note: confirm the official English spelling of the advisor's name before final submission if the institution uses a preferred transliteration.

#### Ethics statement, current provisional version

This study used publicly available aggregate variant annotations and computationally derived representations. It did not involve direct recruitment of human participants, intervention, or access to identifiable individual-level clinical records.

Note: if BIBM or another final target venue requires a stronger formal statement, this wording should be adjusted to match that venue's policy.
