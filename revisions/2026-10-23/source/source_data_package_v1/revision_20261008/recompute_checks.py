"""Reproduce revision tables from frozen predictions. Requires Python, numpy, pandas."""
from pathlib import Path
import numpy as np
import pandas as pd

P=Path(__file__).resolve().parent

def metrics(y,s):
    order=np.argsort(-s,kind='stable');ys=y[order];ss=s[order]
    ends=np.r_[np.where(np.diff(ss)!=0)[0],len(s)-1]
    tp=np.cumsum(ys)[ends].astype(float);fp=ends+1-tp
    # np.trapezoid in modern NumPy, with compatibility for older versions.
    integrate=getattr(np,'trapezoid',np.trapz)
    auc=integrate(np.r_[0,tp/y.sum()],np.r_[0,fp/(len(y)-y.sum())])
    ap=np.sum(np.diff(np.r_[0,tp/y.sum()])*tp/(ends+1))
    return float(auc),float(ap)

f=np.load(P/'frozen_reference_predictions.npz',allow_pickle=False);y=f['labels']
rows=[]
for name,a,b in [('Raw ESM to FPP','raw_esm','fpp'),('Reference CADD+AM to graph fusion','cadd_am_reference','graph_fusion_reference')]:
    for th in np.arange(1,10)/10:
        pa=f[a]>=th;pb=f[b]>=th;ca=pa==y;cb=pb==y
        rows.append({'comparison':name,'threshold':float(th),'jointly_correct':int((ca&cb).sum()),'jointly_wrong':int((~ca&~cb).sum()),
            'corrected':int((~ca&cb).sum()),'degraded':int((ca&~cb).sum()),'fn_rescued':int(((y==1)&~pa&pb).sum()),
            'fp_suppressed':int(((y==0)&pa&~pb).sum()),'tn_to_fp':int(((y==0)&~pa&pb).sum()),'tp_to_fn':int(((y==1)&pa&~pb).sum()),
            'net_corrections':int((~ca&cb).sum()-(ca&~cb).sum())})
got=pd.DataFrame(rows);expected=pd.read_csv(P/'threshold_sweep_reference_predictions.csv')
pd.testing.assert_frame_equal(got,expected,check_dtype=False)

f=np.load(P/'frozen_matched_fusion_predictions.npz',allow_pickle=False);y=f['labels'];mask=f['complete_case_mask']
names={'cadd_plus_am_archmatched_512_128':'CADD+AM matched','cadd_am_plus_raw_esm':'+ raw ESM',
    'cadd_am_plus_real_ppi_gcn':'+ real FPP','cadd_am_plus_edge_ablated_ppi_gcn':'+ edge-ablated FPP',
    'cadd_am_plus_topology_permuted_ppi_gcn':'+ permuted FPP'}
rows=[]
for key in f.files:
    if '__' not in key:continue
    model,run=key.split('__',1)
    for subset,m in [('All test rows',np.ones(len(y),dtype=bool)),('Complete case',mask)]:
        auc,ap=metrics(y[m],f[key][m]);rows.append({'model':names[model],'run':run,'subset':subset,'n':int(m.sum()),'positive_rate':float(y[m].mean()),'auc':auc,'average_precision':ap})
got=pd.DataFrame(rows).sort_values(['model','run','subset']).reset_index(drop=True)
expected=pd.read_csv(P/'complete_case_matched_per_run.csv').sort_values(['model','run','subset']).reset_index(drop=True)
pd.testing.assert_frame_equal(got,expected,check_dtype=False,rtol=1e-10,atol=1e-10)
print('PASS: all threshold counts and all matched per-run complete-case AUC/AP values reproduced.')
