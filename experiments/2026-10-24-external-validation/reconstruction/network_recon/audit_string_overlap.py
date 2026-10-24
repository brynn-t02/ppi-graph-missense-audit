import os
"""Stream frozen STRING source once; exact overlap, bounded-memory 2-bit map."""
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STRING = Path(os.environ['PPI_SOURCE_DATA_ROOT']) / 'raw/9606.protein.links.v12.0.txt.gz'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    nodes = json.loads((ROOT/'frozen_node_order.json').read_text())
    mapping = {p:i for i,p in enumerate(nodes)}
    n = len(nodes)
    bio = set()
    with gzip.open(ROOT/'biogrid_undirected_edges.tsv.gz', 'rt') as f:
        for row in csv.DictReader(f, delimiter='\t'):
            u,v = int(row['node_idx_a']), int(row['node_idx_b'])
            assert u < v and nodes[u] == row['STRING_protein_id_a'] and nodes[v] == row['STRING_protein_id_b']
            bio.add(u*n+v)
    seen = bytearray((n*n+3)//4)
    rows = self_rows = unique = duplicates = both_directions = overlap = 0
    with gzip.open(STRING, 'rt') as f:
        header = next(f).split()
        assert header == ['protein1', 'protein2', 'combined_score']
        for line in f:
            a,b,_ = line.split()
            u,v = mapping[a],mapping[b]
            rows += 1
            if u == v:
                self_rows += 1
                continue
            direction = 1 if u < v else 2
            lo,hi = min(u,v),max(u,v)
            code = lo*n+hi
            offset,shift = code//4,(code%4)*2
            old = (seen[offset] >> shift) & 3
            if old & direction:
                duplicates += 1
            else:
                seen[offset] |= direction << shift
                if old == 0:
                    unique += 1
                    overlap += code in bio
                else:
                    both_directions += 1
    assert rows-self_rows-duplicates == unique+both_directions
    result = {'string_source':str(STRING),'string_source_bytes':STRING.stat().st_size,'string_source_sha256':sha256(STRING),
              'string_directed_rows':rows,'string_self_rows':self_rows,'string_duplicate_directed_rows':duplicates,
              'string_undirected_nonself_pairs':unique,'string_pairs_observed_both_directions':both_directions,
              'biogrid_undirected_nonself_pairs':len(bio),'intersection_pairs':overlap,'union_pairs':unique+len(bio)-overlap,
              'biogrid_only_pairs':len(bio)-overlap,'string_only_pairs':unique-overlap,
              'jaccard':overlap/(unique+len(bio)-overlap),'fraction_biogrid_pairs_in_string':overlap/len(bio),
              'fraction_string_pairs_in_biogrid':overlap/unique,'node_order_requires_tensor_check':True,
              'interpretation':'Alternative network robustness: graph source, coverage, density and release dates differ. Shared primary interaction evidence prevents evidence-independent interpretation.'}
    assert both_directions == unique and duplicates == 0 and self_rows == 0
    (ROOT/'string_network_overlap.json').write_text(json.dumps(result,indent=2)+'\n')
    p=ROOT/'network_build_audit.json'
    audit=json.loads(p.read_text())
    audit['string_overlap_status']='complete_exact_local_source_scan'
    audit['string_overlap']=result
    audit['files']['string_network_overlap.json']={'bytes':(ROOT/'string_network_overlap.json').stat().st_size,'sha256':sha256(ROOT/'string_network_overlap.json')}
    audit['files']['audit_string_overlap.py']={'bytes':(ROOT/'audit_string_overlap.py').stat().st_size,'sha256':sha256(ROOT/'audit_string_overlap.py')}
    p.write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
