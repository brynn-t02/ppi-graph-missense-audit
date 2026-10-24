import os
"""Build prespecified alternative network. Standard library only, no training.

Reads pinned BioGRID ZIP plus previously derived strict Entrez mapping and
frozen node order. Keeps STRING IDs in output so the executor can verify the
reconstructed index against the actual model_inputs_main.pt mapping.
"""
import collections
import csv
import datetime
import gzip
import hashlib
import io
import json
import math
import struct
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ['PPI_SOURCE_DATA_ROOT'])
ZIP = ROOT / 'BIOGRID-ALL-5.0.262.tab3.zip'
URL = 'https://downloads.thebiogrid.org/Download/BioGRID/Release-Archive/BIOGRID-5.0.262/BIOGRID-ALL-5.0.262.tab3.zip'
EXCLUDED_METHODS = {'affinity capture-rna', 'protein-rna'}


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_tsv_gzip(path, header, rows):
    # Zero timestamp gives deterministic gzip output for a fixed source.
    with path.open('wb') as raw:
        with gzip.GzipFile(filename='', fileobj=raw, mode='wb', mtime=0) as packed:
            with io.TextIOWrapper(packed, encoding='utf8', newline='') as text:
                writer = csv.writer(text, delimiter='\t', lineterminator='\n')
                writer.writerow(header)
                writer.writerows(rows)


def write_npy_edges(path, directed):
    # NPY 1.0, C-contiguous int64 [2,E]; little endian, no pickle payload.
    header = repr({'descr': '<i8', 'fortran_order': False, 'shape': (2, len(directed))})
    padding = (64 - ((10 + len(header) + 1) % 64)) % 64
    encoded = (header + ' ' * padding + '\n').encode('ascii')
    with path.open('wb') as f:
        f.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(encoded)) + encoded)
        for component in (0, 1):
            buffer = bytearray()
            for pair in directed:
                buffer.extend(struct.pack('<q', pair[component]))
                if len(buffer) >= 1024 * 1024:
                    f.write(buffer)
                    buffer.clear()
            f.write(buffer)


def quantile(values, q):
    pos = (len(values) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def main():
    nodes = json.loads((ROOT / 'frozen_node_order.json').read_text())
    assert nodes == sorted(set(nodes)) and len(nodes) == 19622
    n = len(nodes)
    node_idx = {p: i for i, p in enumerate(nodes)}
    with (ROOT / 'entrez_to_string_unique.tsv').open() as f:
        mapping = {row['entrez_id']: row['STRING_protein_id'] for row in csv.DictReader(f, delimiter='\t')}
    with (ROOT / 'entrez_ambiguous.tsv').open() as f:
        ambiguous = {row['entrez_id'] for row in csv.DictReader(f, delimiter='\t')}
    counts = collections.Counter()
    raw_taxonomy_pairs = collections.Counter()
    human_method_counts = collections.Counter()
    kept_method_counts = collections.Counter()
    excluded_ids = collections.Counter()
    source_counts = collections.Counter()
    pair_codes = set()
    with zipfile.ZipFile(ZIP) as z:
        infos = z.infolist()
        assert len(infos) == 1 and infos[0].filename == 'BIOGRID-ALL-5.0.262.tab3.txt'
        info = infos[0]
        with z.open(info) as member, io.TextIOWrapper(member, encoding='utf8', newline='') as handle:
            reader = csv.reader(handle, delimiter='\t')
            header = next(reader)
            required = ['Entrez Gene Interactor A', 'Entrez Gene Interactor B',
                        'Organism ID Interactor A', 'Organism ID Interactor B',
                        'Experimental System', 'Experimental System Type', 'Source Database']
            fields = {name: header.index(name) for name in required}
            for row in reader:
                assert len(row) == len(header), f'Unexpected column count at record {counts["input_rows"]+1}'
                counts['input_rows'] += 1
                tax_a = row[fields['Organism ID Interactor A']]
                tax_b = row[fields['Organism ID Interactor B']]
                raw_taxonomy_pairs[(tax_a, tax_b)] += 1
                if tax_a != '9606' or tax_b != '9606':
                    counts['excluded_non_human_human'] += 1
                    continue
                counts['human_human_rows'] += 1
                if row[fields['Experimental System Type']].lower() != 'physical':
                    counts['excluded_nonphysical_human_human'] += 1
                    continue
                counts['human_human_physical_rows'] += 1
                method = row[fields['Experimental System']]
                human_method_counts[method] += 1
                if method.casefold() in EXCLUDED_METHODS:
                    counts['excluded_rna_method_rows'] += 1
                    continue
                counts['eligible_protein_physical_rows'] += 1
                a, b = row[fields['Entrez Gene Interactor A']], row[fields['Entrez Gene Interactor B']]
                if a in ambiguous or b in ambiguous:
                    counts['excluded_ambiguous_endpoint_rows'] += 1
                    for x in (a, b):
                        if x in ambiguous:
                            excluded_ids[(x, 'ambiguous')] += 1
                    continue
                if a not in mapping or b not in mapping:
                    counts['excluded_unmapped_endpoint_rows'] += 1
                    for x in (a, b):
                        if x not in mapping:
                            excluded_ids[(x, 'unmapped_or_outside_universe')] += 1
                    continue
                counts['both_endpoints_uniquely_mapped_rows'] += 1
                u, v = node_idx[mapping[a]], node_idx[mapping[b]]
                if u == v:
                    counts['excluded_mapped_selfinteraction_rows'] += 1
                    continue
                counts['retained_nonself_evidence_rows'] += 1
                kept_method_counts[method] += 1
                source_counts[row[fields['Source Database']]] += 1
                u, v = min(u, v), max(u, v)
                code = u*n+v
                if code in pair_codes:
                    counts['duplicate_pair_evidence_rows'] += 1
                pair_codes.add(code)
        # Reading the entire member verifies its CRC. No 1.56GB extraction.
    assert counts['input_rows'] == counts['excluded_non_human_human'] + counts['excluded_nonphysical_human_human'] + counts['excluded_rna_method_rows'] + counts['excluded_ambiguous_endpoint_rows'] + counts['excluded_unmapped_endpoint_rows'] + counts['excluded_mapped_selfinteraction_rows'] + counts['retained_nonself_evidence_rows']
    assert len(pair_codes) + counts['duplicate_pair_evidence_rows'] == counts['retained_nonself_evidence_rows']
    pairs = [(key//n, key % n) for key in sorted(pair_codes)]
    directed = sorted(pairs + [(v, u) for u, v in pairs])
    assert len(directed) == 2*len(pairs) == len(set(directed))
    assert all(u != v and 0 <= u < n and 0 <= v < n for u, v in directed)
    write_tsv_gzip(ROOT / 'biogrid_undirected_edges.tsv.gz',
                   ['node_idx_a', 'node_idx_b', 'STRING_protein_id_a', 'STRING_protein_id_b'],
                   ((u, v, nodes[u], nodes[v]) for u, v in pairs))
    write_tsv_gzip(ROOT / 'biogrid_bidirectional_edges.tsv.gz',
                   ['source_idx', 'target_idx', 'source_STRING_id', 'target_STRING_id'],
                   ((u, v, nodes[u], nodes[v]) for u, v in directed))
    write_npy_edges(ROOT / 'biogrid_edge_index.npy', directed)
    write_tsv_gzip(ROOT / 'excluded_endpoint_ids.tsv.gz',
                   ['entrez_id', 'reason', 'endpoint_occurrences'],
                   ((k[0], k[1], v) for k, v in sorted(excluded_ids.items())))
    degree = [0] * n
    parent = list(range(n))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for u, v in pairs:
        degree[u] += 1
        degree[v] += 1
        ru, rv = find(u), find(v)
        if ru != rv:
            parent[rv] = ru
    components = collections.Counter(find(i) for i in range(n))
    degree_sorted = sorted(degree)
    write_tsv_gzip(ROOT / 'biogrid_node_degree.tsv.gz', ['node_idx', 'STRING_protein_id', 'degree'],
                   ((i, nodes[i], degree[i]) for i in range(n)))
    split_stats = {}
    for scheme in ['original', 'homology']:
        group_sets = {}
        coverage = {}
        for split in ['train', 'val', 'test']:
            path = DATA / 'processed' / (f'clinvar_{split}_main.csv' if scheme == 'original' else f'homology_aware_split/clinvar_{split}_homology.csv')
            seen = set()
            rows = covered_rows = 0
            with path.open() as f:
                for row in csv.DictReader(f):
                    idx = node_idx[row['STRING_protein_id']]
                    rows += 1
                    covered_rows += degree[idx] > 0
                    seen.add(idx)
            group_sets[split] = seen
            coverage[split] = {'variants': rows, 'variants_with_edge': covered_rows, 'proteins': len(seen), 'proteins_with_edge': sum(degree[i] > 0 for i in seen)}
        assert not group_sets['train'] & group_sets['val']
        assert not group_sets['train'] & group_sets['test']
        assert not group_sets['test'] & group_sets['val']
        tr, te = group_sets['train'], group_sets['test']
        coverage['direct_train_test_undirected_edges'] = sum((u in tr and v in te) or (v in tr and u in te) for u, v in pairs)
        split_stats[scheme] = coverage
    graph = {'total_nodes_including_isolates': n, 'nonisolated_nodes': sum(d > 0 for d in degree), 'isolates': sum(d == 0 for d in degree),
             'undirected_nonself_edges': len(pairs), 'directed_nonself_edges': len(directed),
             'connected_components_including_isolates': len(components), 'largest_component_nodes': max(components.values()),
             'density_undirected': 2*len(pairs)/(n*(n-1)), 'degree_mean': sum(degree)/n,
             'degree_quantiles': {str(q): quantile(degree_sorted, q) for q in [0, .25, .5, .75, .9, .95, .99, 1]}}
    audit = {'status': 'graph_built_no_fitting', 'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
             'source': {'release': '5.0.262', 'compiled_date': '2026-09-25', 'release_date': '2026-10-01', 'url': URL,
                        'download_http_status': 200, 'bytes': ZIP.stat().st_size, 'sha256': sha256(ZIP),
                        'zip_member': info.filename, 'zip_member_bytes': info.file_size, 'zip_member_crc32': f'{info.CRC:08x}',
                        'zip_member_crc_verified_by_full_read': True, 'license': 'MIT'},
             'filter_counts': dict(counts), 'human_physical_system_counts': dict(human_method_counts),
             'retained_nonself_evidence_system_counts': dict(kept_method_counts), 'retained_source_database_counts': dict(source_counts),
             'excluded_rna_methods_casefold': sorted(EXCLUDED_METHODS), 'graph': graph, 'split_edge_coverage': split_stats,
             'mapping_sha256': sha256(ROOT / 'entrez_to_string_unique.tsv'), 'node_order_sha256': sha256(ROOT / 'frozen_node_order.json'),
             'node_order_requires_tensor_check': True,
             'operator_requirement': 'E contains both directions; add I then symmetrize and normalize identically to frozen full STRING operator. No pre-existing self edges.',
             'source_nonindependence': 'STRING integrates BioGRID evidence; this is an alternative network, not evidence-independent replication.',
             'string_overlap_status': 'pending_local_source_scan', 'files': {}}
    for name in ['biogrid_undirected_edges.tsv.gz', 'biogrid_bidirectional_edges.tsv.gz', 'biogrid_edge_index.npy', 'biogrid_node_degree.tsv.gz', 'excluded_endpoint_ids.tsv.gz', 'build_biogrid_network.py']:
        path = ROOT/name
        audit['files'][name] = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
    (ROOT/'network_build_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(json.dumps({'graph': graph, 'filter_counts': dict(counts), 'split_edge_coverage': split_stats}, indent=2), flush=True)


if __name__ == '__main__':
    main()
