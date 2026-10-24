import os
"""Build a frozen external-source HUMSAVAR holdout; no scores/fitting are read.
Standard-library only. Provider inputs remain read-only. See cohort_protocol.json.
"""
from pathlib import Path
import csv,gzip,hashlib,json,re,collections,datetime,argparse
P=argparse.ArgumentParser();P.add_argument('--node-map',type=Path,required=True);P.add_argument('--source-root',type=Path,default=Path(os.environ['PPI_SOURCE_DATA_ROOT']));args=P.parse_args()
R=Path(__file__).resolve().parent;D=args.source_root/'processed';RAW=args.source_root/'raw'
AA=dict(zip(['Ala','Arg','Asn','Asp','Cys','Gln','Glu','Gly','His','Ile','Leu','Lys','Met','Phe','Pro','Ser','Thr','Trp','Tyr','Val'],'ARNDCQEGHILKMFPSTWYV'))
pat=re.compile(r'^p\.([A-Z][a-z]{2})([1-9][0-9]*)([A-Z][a-z]{2})$')
def aa(x):
 m=pat.fullmatch(x)
 if not m or m[1] not in AA or m[3] not in AA:return None
 return AA[m[1]]+m[2]+AA[m[3]]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while b:=f.read(1048576):h.update(b)
 return h.hexdigest()
def fasta(p,uniprot=False):
 d={};key=None;seq=[]
 with gzip.open(p,'rt') as f:
  for l in f:
   if l.startswith('>'):
    if key:d[key]=''.join(seq)
    key=l[1:].split()[0];key=key.split('|')[1] if uniprot else key;seq=[]
   else:seq.append(l.strip())
  if key:d[key]=''.join(seq)
 return d
node_map=json.loads(args.node_map.read_text());node_map={k:i for i,k in enumerate(node_map)} if isinstance(node_map,list) else node_map.get('protein_to_idx',node_map);node_map={str(k):int(v) for k,v in node_map.items()}
assert len(node_map)==19622 and len(set(node_map.values()))==19622
stringseq=fasta(RAW/'9606.protein.sequences.v12.0.fa.gz');uniseq=fasta(R/'uniprot_human_reviewed_canonical_2026_03.fasta.gz',True)
original=[];splitprots={}
for split in ['train','val','test']:
 rows=list(csv.DictReader((D/f'clinvar_{split}_main.csv').open()));original.extend(rows);splitprots[split]={x['STRING_protein_id'] for x in rows}
oldids={x['VariationID'] for x in original};assert len(oldids)==177213 and len(original)==177213
oldgeneaa={(x['GeneSymbol'].upper(),aa(x['HGVS_p'])) for x in original};oldnodeaa={(x['STRING_protein_id'],aa(x['HGVS_p'])) for x in original};olduniprot=set()
for x in csv.DictReader((D/'clinvar_with_alphamissense.csv').open()):
 if x['VariationID'] in oldids and x['uniprot_id'] and x['protein_variant']:olduniprot.add((x['uniprot_id'],x['protein_variant']))
# Supplemental rsID exclusion is deliberately conservative: reject every allele
# sharing a dbSNP site with an original cohort variant, even if different ALT.
rawpath=RAW/'variant_summary.txt.gz';cache=R/'original_cohort_rsid_cache.json';rawhash=sha(rawpath)
if cache.exists() and json.loads(cache.read_text())['raw_sha256']==rawhash:
 info=json.loads(cache.read_text());oldrs=set(info['rsids'])
else:
 oldrs=set();seen=set();dates=collections.Counter();examples=[]
 with gzip.open(rawpath,'rt') as f:
  header=f.readline().rstrip('\n').split('\t');idx={k:i for i,k in enumerate(header)}
  for l in f:
   z=l.rstrip('\n').split('\t');vid=z[idx['VariationID']]
   if vid not in oldids or z[idx['Assembly']]!='GRCh38':continue
   seen.add(vid);rs=z[idx['RS# (dbSNP)']]
   for v in re.split(r'[;|,]',rs):
    if v.isdigit() and int(v)>0:oldrs.add('rs'+v)
   date=z[idx['LastEvaluated']];dates[date[-4:]]+=1
   if date.endswith('2026') and len(examples)<10:examples.append({k:z[idx[k]] for k in ['VariationID','Name','LastEvaluated','Assembly']})
 info={'raw_sha256':rawhash,'original_variants_found':len(seen),'rsids':sorted(oldrs),'evaluation_year_counts':dict(dates),'2026_examples':examples}
 cache.write_text(json.dumps(info,indent=2))
assert info['original_variants_found']==177213
print('Original cohort overlap keys extracted',len(oldrs),flush=True)
# Exact aliases only; all alias sources retained in per-accession audit.
aliases=collections.defaultdict(set);aliassource=collections.defaultdict(set)
with gzip.open(RAW/'9606.protein.aliases.v12.0.txt.gz','rt') as f:
 next(f)
 for l in f:
  pid,alias,source=l.rstrip('\n').split('\t')
  if alias in uniseq and pid in node_map:
   aliases[alias].add(pid);aliassource[alias].add(source)
rawrows=[]
for l in (R/'humsavar_2026_03.txt').read_text().splitlines():
 z=l.split(None,6)
 if len(z)>=6 and z[2].startswith('VAR_'):
  rawrows.append({'gene':z[0],'uniprot_id':z[1],'humsavar_id':z[2],'HGVS_p':z[3],'category':z[4],'rsid':z[5],'disease':z[6] if len(z)>6 else ''})
assert len(rawrows)==85015
trainval=splitprots['train']|splitprots['val'];allprots=trainval|splitprots['test']
trainvalseqs={stringseq[x] for x in trainval};allseqs={stringseq[x] for x in allprots}
conflict=collections.defaultdict(set)
for x in rawrows:
 conflict[(x['uniprot_id'],aa(x['HGVS_p']))].add(x['category'])
rs_overlap_keys={(x['uniprot_id'],aa(x['HGVS_p'])) for x in rawrows if x['rsid'] in oldrs}
somatic_keys={(x['uniprot_id'],aa(x['HGVS_p'])) for x in rawrows if re.search(r'\bsomatic\b|\bcancer sample\b|\btumou?r sample\b',x['disease'],re.I)}
labelmap={'LP/P':1,'LB/B':0};audit=[];kept=[];seenkeys=set();reason_counts=collections.Counter();mapping_report={}
for x in rawrows:
 x=dict(x);acc=x['uniprot_id'];change=aa(x['HGVS_p']);reason='';pids=aliases[acc];exact={pid for pid in pids if uniseq.get(acc)==stringseq[pid]};pid=next(iter(exact)) if len(exact)==1 else ''
 x.update({'protein_variant':change or '', 'STRING_protein_id':pid, 'node_index':node_map.get(pid,''), 'label':labelmap.get(x['category'],''), 'overlap_uniprot_aa':int((acc,change) in olduniprot),'overlap_gene_aa':int((x['gene'].upper(),change) in oldgeneaa),'overlap_node_aa':int((pid,change) in oldnodeaa),'overlap_dbSNP':int((acc,change) in rs_overlap_keys)})
 mapping_report[acc]={'alias_graph_nodes':sorted(pids),'full_sequence_identical_nodes':sorted(exact),'alias_sources':sorted(aliassource[acc])}
 if x['category'] not in labelmap:reason='uncertain_significance'
 elif change is None or change[0]==change[-1]:reason='not_standard_single_amino_acid_substitution'
 elif (acc,change) in somatic_keys:reason='explicit_somatic_or_cancer_sample'
 elif len(conflict[(acc,change)])>1:reason='conflicting_source_duplicate'
 elif acc not in uniseq:reason='canonical_sequence_missing'
 elif not pids:reason='no_accession_alias_to_graph'
 elif not exact:reason='no_full_sequence_identity'
 elif len(exact)>1:reason='ambiguous_identical_graph_nodes'
 elif int(change[1:-1])>len(uniseq[acc]) or uniseq[acc][int(change[1:-1])-1]!=change[0]:reason='reference_residue_mismatch'
 elif any(x[k] for k in ['overlap_uniprot_aa','overlap_gene_aa','overlap_node_aa','overlap_dbSNP']):reason='overlaps_original_177213'
 elif (acc,change) in seenkeys:reason='redundant_source_duplicate'
 if not reason:
  seenkeys.add((acc,change));s=stringseq[pid]
  x['original_protein_stratum']='train' if pid in splitprots['train'] else 'val' if pid in splitprots['val'] else 'test' if pid in splitprots['test'] else 'unlabelled'
  x['primary_trainval_unseen']=int(pid not in trainval and s not in trainvalseqs)
  x['unseen_all_original']=int(pid not in allprots and s not in allseqs)
  x['external_id']=acc+':'+change
  kept.append(x)
 x['exclusion_reason']=reason;reason_counts[reason or 'retained']+=1;audit.append(x)
# Pre-inference repair: identical paralogs can share an accession alias/node.
# Do not assign protein-network context arbitrarily. Apply to otherwise-retained
# candidates only, after all previous exclusions and canonical-variant dedup.
node_accessions=collections.defaultdict(set)
for x in kept:node_accessions[x['STRING_protein_id']].add(x['uniprot_id'])
collision_nodes={pid:sorted(accs) for pid,accs in node_accessions.items() if len(accs)>1}
collision_rows=[x for x in kept if x['STRING_protein_id'] in collision_nodes]
for x in collision_rows:
 x['exclusion_reason']='cross_accession_node_collision'
reason_counts['retained']-=len(collision_rows)
reason_counts['cross_accession_node_collision']+=len(collision_rows)
kept=[x for x in kept if x['STRING_protein_id'] not in collision_nodes]
def writecsv(path,rows):
 fields=list(dict.fromkeys(k for x in rows for k in x))
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
def stats(rows):
 return {'variants':len(rows),'proteins':len({x['STRING_protein_id'] for x in rows}),'positive':sum(x['label'] for x in rows),'negative':sum(1-x['label'] for x in rows),'positive_proteins':len({x['STRING_protein_id'] for x in rows if x['label']==1}),'negative_proteins':len({x['STRING_protein_id'] for x in rows if x['label']==0})}
writecsv(R/'all_source_rows_with_exclusions.csv',audit);writecsv(R/'external_cohort_all_variant_disjoint.csv',kept)
primary=[x for x in kept if x['primary_trainval_unseen']];writecsv(R/'external_cohort_primary.csv',primary)
strong=[x for x in kept if x['unseen_all_original']];writecsv(R/'external_cohort_unseen_all_original.csv',strong)
counts={'source_rows':len(rawrows),'unique_source_canonical_variants':len(conflict),'source_labels':dict(collections.Counter(x['category'] for x in rawrows)),'sequential_attrition':dict(reason_counts),'all_variant_disjoint':stats(kept),'primary_trainval_unseen':stats(primary),'unseen_all_original':stats(strong),'by_original_protein_stratum':{s:stats([x for x in kept if x['original_protein_stratum']==s]) for s in ['train','val','test','unlabelled']},'sequence_mapping_accessions':len(mapping_report),'cross_accession_collision_nodes':collision_nodes,'cross_accession_collision_variants':[x['external_id'] for x in collision_rows],'original_variant_count':len(original),'old_rsid_count':len(oldrs),'clinvar_snapshot_correction':{'actual_input_has_2026_evaluations':bool(info['2026_examples']),'examples':info['2026_examples'],'wording':'Retained checksum-identified ClinVar snapshot, archived03April2026; FTP Released field31March2024 does not establish the content release.'},'no_model_scores_read':True,'completed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
q=counts['primary_trainval_unseen'];counts['primary_viable_for_exploratory_interval']=q['variants']>=100 and min(q['positive'],q['negative'])>=20 and min(q['positive_proteins'],q['negative_proteins'])>=20
(R/'cohort_counts.json').write_text(json.dumps(counts,indent=2));(R/'accession_mapping_audit.json').write_text(json.dumps(mapping_report,indent=2))
inputs=[args.node_map,R/'humsavar_2026_03.txt',R/'uniprot_human_reviewed_canonical_2026_03.fasta.gz',R/'cohort_protocol.json',R/'cohort_protocol_deviation_before_inference.json',RAW/'9606.protein.aliases.v12.0.txt.gz',RAW/'9606.protein.sequences.v12.0.fa.gz',rawpath,D/'clinvar_with_alphamissense.csv']+[D/f'clinvar_{s}_main.csv' for s in ['train','val','test']]
outputs=[R/'external_cohort_all_variant_disjoint.csv',R/'external_cohort_primary.csv',R/'external_cohort_unseen_all_original.csv',R/'all_source_rows_with_exclusions.csv',R/'cohort_counts.json',R/'accession_mapping_audit.json',Path(__file__).resolve()]
(R/'cohort_manifest.json').write_text(json.dumps({'inputs':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in inputs],'outputs':[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)} for p in outputs]},indent=2))
print(json.dumps(counts,indent=2),flush=True)
