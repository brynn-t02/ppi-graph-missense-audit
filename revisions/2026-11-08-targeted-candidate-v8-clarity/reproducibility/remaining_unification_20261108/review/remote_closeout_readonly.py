"""Read-only CPU verification sent over an existing SSH control socket.

No training/inference, CUDA allocation, server file writes, downloads, or cleanup.
Only compact JSON is returned to the local review folder.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time

import numpy as np
import torch
from torch import nn

torch.set_num_threads(4)
B = Path('/root/autodl-tmp/ppi_unification_20261108')
O = B / 'outputs'


def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1280, 512), nn.ReLU(), nn.Dropout(.3),
                                 nn.Linear(512, 128), nn.ReLU(), nn.Dropout(.3),
                                 nn.Linear(128, 1))


expected = {'net.0.weight': [512, 1280], 'net.0.bias': [512],
            'net.3.weight': [128, 512], 'net.3.bias': [128],
            'net.6.weight': [1, 128], 'net.6.bias': [1]}
complete = json.loads((O / 'analysis_complete.json').read_text())
pipeline = json.loads((O / 'pipeline_status_v2.json').read_text())
assert complete['passed'] and complete['classifier_fits'] == 39
assert pipeline['phase'] == 'complete'
paths = sorted((O / 'runs').glob('*/*/seed_*/model.pt'))
assert len(paths) == 39
assert len(complete['checkpoint_hashes']) == 39
canonical = {f"{x['split']}/{x['arm']}/seed_{x['seed']}": x
             for x in json.loads((O / 'canonical_inference_audit.json').read_text())}
assert len(canonical) == 39
records = []
for p in paths:
    key = str(p.parent.relative_to(O / 'runs'))
    digest = sha(p)
    assert digest == complete['checkpoint_hashes'][key]
    assert digest == canonical[key]['checkpoint_sha256']
    state = torch.load(p, map_location='cpu', weights_only=True)
    assert set(state) == set(expected)
    for name, shape in expected.items():
        t = state[name]
        assert list(t.shape) == shape and t.device.type == 'cpu'
        assert t.dtype == torch.float32 and torch.isfinite(t).all().item()
    model = MLP()
    model.load_state_dict(state, strict=True)
    assert sum(t.numel() for t in state.values()) == 721665
    files = {q.name: {'sha256': sha(q), 'bytes': q.stat().st_size}
             for q in sorted(p.parent.iterdir()) if q.is_file()}
    assert files['canonical_node_probs.npy']['sha256'] == canonical[key]['node_prediction_sha256']
    arr = np.load(p.parent / 'canonical_node_probs.npy', mmap_mode='r', allow_pickle=False)
    assert arr.shape == (19622,) and arr.dtype == np.float32
    assert np.isfinite(arr).all() and arr.min() >= 0 and arr.max() <= 1
    records.append({'run': key, 'checkpoint_sha256': digest,
                    'load_state_dict_strict': True, 'all_parameters_finite': True,
                    'parameter_count': 721665, 'files': files})
    del model, state, arr

amend = json.loads((B / 'code_v2/amendment_manifest.json').read_text())
assert sha(O / 'derived_embeddings/edge_ablated_FPP.pt') == amend['first_tensor_sha256']
for row in amend['first_six_runs']:
    q = O / 'runs/original' / row['arm'] / f"seed_{row['seed']}"
    for name, field in [('model.pt', 'checkpoint_sha256'),
                        ('canonical_node_probs.npy', 'canonical_node_prediction_sha256'),
                        ('metrics.json', 'metrics_sha256'), ('history.json', 'history_sha256')]:
        assert sha(q / name) == row[field]

input_rows = json.loads((B / 'code_v2/input_manifest.json').read_text())
for row in input_rows:
    p = B / row['path']
    assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256']

embeddings = []
for meta in sorted((O / 'derived_embeddings').glob('*.json')):
    info = json.loads(meta.read_text())
    p = meta.with_suffix('.pt')
    assert sha(p) == info['sha256']
    embeddings.append({'path': str(p), 'bytes': p.stat().st_size,
                       'sha256': info['sha256'], 'recorded_shape': info['shape']})

active = []
scripts = ['execute_pipeline.py', 'run_unified.py', 'analyze_unified.py',
           'finish_new_environment.py', 'restore_new_server.py', 'diagnose_propagation.py']
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name) == os.getpid():
        continue
    try:
        args = (p / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()
        cwd = (p / 'cwd').resolve()
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        continue
    if any(s in args for s in scripts) and (str(B) in args or str(cwd).startswith(str(B))):
        active.append({'pid': int(p.name), 'cmdline': args})

disk = shutil.disk_usage(B)
sizes = {}
for name in ['data', 'reference', 'environment', 'outputs', 'logs', 'diagnostics', 'audit_before_amendment_v2']:
    p = B / name
    if p.exists():
        r = subprocess.run(['du', '-sb', str(p)], check=True, capture_output=True, text=True)
        sizes[name] = int(r.stdout.split()[0])
gpu = subprocess.run(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                      '--format=csv,noheader,nounits'], capture_output=True, text=True)
assert not torch.cuda.is_initialized()
report = {'status': 'passed', 'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'server_root': str(B), 'read_only': True, 'cuda_initialized': False,
          'server_writes_or_deletions': False, 'large_file_downloads': False,
          'checkpoints_loaded_and_verified': len(records), 'expected_shapes': expected,
          'all_checkpoint_hashes_match_analysis_and_canonical_audits': True,
          'first_six_models_predictions_metrics_and_histories_locked': True,
          'first_ablated_tensor_locked': True, 'input_hashes_verified': len(input_rows),
          'derived_embedding_hashes_verified': embeddings,
          'pipeline_status': pipeline, 'active_own_pipeline_processes': active,
          'own_pipeline_finished': not active,
          'nvidia_smi_compute_apps': gpu.stdout.strip() if gpu.returncode == 0 else 'unavailable',
          'directory_apparent_bytes': sizes,
          'filesystem_bytes': {'total': disk.total, 'used': disk.used, 'free': disk.free},
          'checkpoint_records': records}
assert not active, active
print(json.dumps(report, indent=2))
