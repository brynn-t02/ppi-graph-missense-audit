import os
"""Read-only source audit; derive strict BioGRID Entrez-to-frozen-STRING map.

No network, torch, labels, fitting, or GPU access. Node order must additionally
be checked against model_inputs_main.pt by the execution pipeline.
"""
import collections
import csv
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ['PPI_SOURCE_DATA_ROOT'])
ALIASES = DATA / 'raw/9606.protein.aliases.v12.0.txt.gz'
FEATURES = DATA / 'processed/esm_node_features_full.csv'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    # Read only the frozen protein IDs. Historical prepare_model_inputs uses
    # sorted graph nodes; these features contain exactly the same 19,622 IDs.
    with FEATURES.open() as handle:
        next(handle)
        nodes = sorted({line.split(',', 1)[0] for line in handle})
    assert len(nodes) == 19622
    node_idx = {node: i for i, node in enumerate(nodes)}
    candidates = collections.defaultdict(set)
    sources = collections.Counter()
    with gzip.open(ALIASES, 'rt') as handle:
        next(handle)
        for line in handle:
            protein, alias, source = line.rstrip('\n').split('\t')
            if 'entrez' in source.lower():
                sources[source] += 1
            if 'Ensembl_HGNC_entrez_id' in source.split():
                assert alias.isdigit()
                candidates[alias].add(protein)
    unique = {
        key: next(iter(values)) for key, values in candidates.items()
        if len(values) == 1 and next(iter(values)) in node_idx
    }
    with (ROOT / 'entrez_to_string_unique.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(['entrez_id', 'STRING_protein_id', 'node_idx'])
        writer.writerows((key, unique[key], node_idx[unique[key]])
                         for key in sorted(unique, key=int))
    with (ROOT / 'entrez_ambiguous.tsv').open('w') as handle:
        writer = csv.writer(handle, delimiter='\t', lineterminator='\n')
        writer.writerow(['entrez_id', 'STRING_candidates'])
        writer.writerows((key, '|'.join(sorted(values)))
                         for key, values in sorted(candidates.items(), key=lambda item: int(item[0]))
                         if len(values) > 1)
    (ROOT / 'frozen_node_order.json').write_text(json.dumps(nodes) + '\n')
    mapped_nodes = set(unique.values())
    coverage = {}
    for split in ['train', 'val', 'test']:
        path = DATA / 'processed' / f'clinvar_{split}_main.csv'
        total = matched = 0
        proteins = set()
        with path.open() as handle:
            for row in csv.DictReader(handle):
                protein = row['STRING_protein_id']
                total += 1
                matched += protein in mapped_nodes
                proteins.add(protein)
        coverage[split] = {
            'variants': total, 'variants_with_strict_entrez_mapping': matched,
            'proteins': len(proteins),
            'proteins_with_strict_entrez_mapping': len(proteins & mapped_nodes),
            'note': 'Mapping availability only; presence of actual BioGRID edges not yet measured.'
        }
    audit = {
        'date': '2026-10-24', 'status': 'mapping_only_no_network_download_or_training',
        'aliases_path': str(ALIASES), 'aliases_bytes': ALIASES.stat().st_size,
        'aliases_sha256': sha256(ALIASES), 'frozen_feature_id_source': str(FEATURES),
        'node_count': len(nodes), 'strict_mapping_source': 'Ensembl_HGNC_entrez_id',
        'entrez_source_row_counts': dict(sources), 'candidate_entrez_ids': len(candidates),
        'ambiguous_entrez_ids_excluded': sum(len(v) > 1 for v in candidates.values()),
        'unique_entrez_ids_in_universe': len(unique), 'mapped_nodes': len(mapped_nodes),
        'mapping_node_coverage': len(mapped_nodes) / len(nodes),
        'original_split_mapping_coverage': coverage,
        'mapping_sha256': sha256(ROOT / 'entrez_to_string_unique.tsv'),
        'node_order_sha256': sha256(ROOT / 'frozen_node_order.json'),
        'node_order_requires_tensor_check': True,
        'selection': 'No labels, classifier outputs, or test performance used to derive mapping.'
    }
    (ROOT / 'mapping_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
