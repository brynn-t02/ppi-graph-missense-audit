import os
"""Independent release-gate verification of source-cohort correspondence.
No model files or scores are accessed. Uses direct source records and sequences.
"""
import csv,gzip,json,re,hashlib,collections
from pathlib import Path
R=Path(__file__).resolve().parent
D=Path(os.environ['PPI_SOURCE_DATA_ROOT'])
def readseq(p,uniprot):
 out={};k='';s=''
 for l in gzip.open(p,'rt'):
  if l.startswith('>'):
   if k:out[k]=s
   k=l[1:].split()[0];k=k.split('|')[1] if uniprot else k;s=''
  else:s+=l.strip()
 out[k]=s;return out
u=readseq(R/'uniprot_human_reviewed_canonical_2026_03.fasta.gz',True);g=readseq(D/'raw/9606.protein.sequences.v12.0.fa.gz',False)
orig={s:list(csv.DictReader((D/f'processed/clinvar_{s}_main.csv').open())) for s in ['train','val','test']}
tv={x['STRING_protein_id'] for s in ['train','val'] for x in orig[s]};allp=tv|{x['STRING_protein_id'] for x in orig['test']};tvseq={g[p] for p in tv};allseq={g[p] for p in allp}
a=list(csv.DictReader((R/'external_cohort_all_variant_disjoint.csv').open()));p=list(csv.DictReader((R/'external_cohort_primary.csv').open()));z=list(csv.DictReader((R/'external_cohort_unseen_all_original.csv').open()));rs=set(json.loads((R/'original_cohort_rsid_cache.json').read_text())['rsids'])
assert len(a)==len({x['external_id'] for x in a})
node_accessions=collections.defaultdict(set)
for x in a:node_accessions[x['STRING_protein_id']].add(x['uniprot_id'])
assert all(len(v)==1 for v in node_accessions.values())
repair=json.loads((R/'cohort_protocol_deviation_before_inference.json').read_text())
assert not ({x['external_id'] for x in repair['affected']} & {x['external_id'] for x in a})
for name in ['external_cohort_primary.csv','external_cohort_unseen_all_original.csv']:
 assert hashlib.sha256((R/name).read_bytes()).hexdigest()==repair['prior_candidate_hashes'][name]
for x in a:
 s=u[x['uniprot_id']];assert s==g[x['STRING_protein_id']]
 assert s[int(x['protein_variant'][1:-1])-1]==x['protein_variant'][0]
 assert x['label']=={'LP/P':'1','LB/B':'0'}[x['category']]
 assert x['rsid'] not in rs
 assert all(x[k]=='0' for k in ['overlap_uniprot_aa','overlap_gene_aa','overlap_node_aa','overlap_dbSNP'])
 assert not re.search(r'\bsomatic\b|\bcancer sample\b|\btumou?r sample\b',x['disease'],re.I)
for x in p:assert x['STRING_protein_id'] not in tv and g[x['STRING_protein_id']] not in tvseq
for x in z:assert x['STRING_protein_id'] not in allp and g[x['STRING_protein_id']] not in allseq
assert {x['external_id'] for x in z}<={x['external_id'] for x in p}<={x['external_id'] for x in a}
# This verifies all output manifest hashes, including the build program.
m=json.loads((R/'cohort_manifest.json').read_text())
for x in m['outputs']:assert hashlib.sha256(Path(x['path']).read_bytes()).hexdigest()==x['sha256']
report={'status':'passed','all_variant_disjoint_rows':len(a),'primary_rows':len(p),'unseen_all_original_rows':len(z),'checks':['unique external protein-substitution identities','one retained UniProt accession per STRING node; all7 flagged collision rows excluded','primary/strict CSV hashes unchanged by repair','whole canonical sequence matches frozen STRING sequence','reference amino acid and class match','dbSNP site overlap absent','primary train/validation protein and exact-sequence exclusion','strict sensitivity excludes all original labelled proteins and their exact sequences','nested populations','all frozen output hashes match'],'node_index_caveat':'Inference must separately assert every row node_index equals actual model_inputs_main.pt protein_to_idx; candidate table mapping currently derives sorted frozen feature identifiers.'}
(R/'cohort_quality_checks.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
