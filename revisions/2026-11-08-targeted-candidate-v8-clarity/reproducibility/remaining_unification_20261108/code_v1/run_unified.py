"""Server-only remaining matched representation experiments. No test selection.

Run from the isolated restored Python environment with PPI_UNIFICATION_ROOT set.
Old assets are read-only; all new artifacts live in this task's outputs directory.
"""
import argparse, hashlib, importlib.metadata as metadata, json, os, platform, sys, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matched_training_core as m
import pair_selection_original as pairs

B=m.B; D=m.D; O=m.O; CODE=B/'code'; REF=B/'reference'; SEEDS=m.SEEDS
STRATEGIES=['random_pair_removal','high_confidence_first','low_confidence_first']
FRACTIONS=[.25,.5,.75]

def status(phase, **kw):
    v={'phase':phase,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),**kw}
    m.dump('progress.json',v); print(json.dumps(v),flush=True)

def savecsv(df,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(path,index=False,float_format='%.17g')

def hash_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()

def check_inputs():
    O.mkdir(exist_ok=True)
    verified=[]
    for x in json.loads((CODE/'input_manifest.json').read_text()):
        p=B/x['path']; assert p.stat().st_size==x['bytes'],str(p)
        assert m.sha(p)==x['sha256'],str(p); verified.append(x['path'])
    # Freeze exact executable sources/protocol before any data-dependent work.
    hashes={p.name:m.sha(p) for p in sorted(CODE.iterdir()) if p.is_file()}
    freeze=O/'protocol_freeze.json'
    if freeze.exists(): assert json.loads(freeze.read_text())['code_hashes']==hashes,'Code changed after freeze'
    else:m.dump('protocol_freeze.json',{'code_hashes':hashes,'time_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
    assert np.__version__=='1.26.4',np.__version__
    assert torch.__version__=='2.1.2+cu118',torch.__version__
    assert metadata.version('scikit-learn')=='1.7.2'
    assert platform.python_version()=='3.10.8',platform.python_version()
    assert pd.__version__=='2.3.3' and metadata.version('scipy')=='1.15.3'
    assert torch.get_num_threads()==4
    assert torch.backends.cuda.matmul.allow_tf32 is False and torch.backends.cudnn.allow_tf32 is True
    m.dump('runtime_environment.json',{'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'cuda_build':torch.version.cuda,'gpu':torch.cuda.get_device_name(0),'cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES'),'packages':{k:metadata.version(k) for k in ['numpy','pandas','scipy','scikit-learn']},'threads':torch.get_num_threads(),'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32})
    a=m.load(D/'model_inputs_main.pt')
    assert a['x'].shape==(19622,1280) and a['edge_index'].shape==(2,13715404)
    assert torch.isfinite(a['x']).all()
    mapping={str(k):int(v) for k,v in a['protein_to_idx'].items()}
    assert sorted(mapping.values())==list(range(19622))
    actual=[None]*len(mapping)
    for k,v in mapping.items():actual[v]=k
    nd=pd.read_csv(REF/'canonical/frozen_node_order.csv.gz')
    assert nd.node_index.tolist()==list(range(19622)) and nd.STRING_protein_id.tolist()==actual
    assert {int(k):str(v) for k,v in a['idx_to_protein'].items()}==dict(enumerate(actual))
    out={}; manifest_parts={};audit={}
    assignments=pd.read_csv(D/'homology_aware_split/homology_cluster_assignments.csv').set_index('cluster_id')
    assert assignments.index.is_unique
    for split in ['original','homology']:
        s={'dfs':{},'nodes':{},'labels':{}};allrows=[]
        for part in m.PARTS:
            rel=f'clinvar_{part}_main.csv' if split=='original' else f'homology_aware_split/clinvar_{part}_homology.csv'
            d=pd.read_csv(D/rel,low_memory=False)
            assert d.VariationID.is_unique and d.STRING_protein_id.notna().all()
            idx=d.STRING_protein_id.astype(str).map(mapping);assert idx.notna().all()
            s['nodes'][part]=torch.tensor(idx.to_numpy(),dtype=torch.long)
            s['labels'][part]=torch.tensor(d.Label.to_numpy(),dtype=torch.float32)
            assert set(d.Label.unique())=={0,1}
            if split=='original':
                assert torch.equal(s['nodes'][part],a[part+'_protein_idx'].long())
                assert torch.equal(s['labels'][part],a[part+'_labels'].float())
            else:
                assert d._cluster_id.notna().all()
                assert d._cluster_id.map(assignments.assigned_split).eq(part).all()
            s['dfs'][part]=d
            allrows.append(d.assign(**{split+'_split':part}))
        for i,p1 in enumerate(m.PARTS):
            for p2 in m.PARTS[:i]:
                assert not set(s['dfs'][p1].STRING_protein_id)&set(s['dfs'][p2].STRING_protein_id)
                if split=='homology':assert not set(s['dfs'][p1]._cluster_id)&set(s['dfs'][p2]._cluster_id)
        manifest_parts[split]=pd.concat(allrows,ignore_index=True)
        assert len(manifest_parts[split])==177213 and manifest_parts[split].VariationID.is_unique
        audit[split]={'rows':{p:len(s['dfs'][p]) for p in m.PARTS},'positives':{p:int(s['labels'][p].sum()) for p in m.PARTS},'test_proteins':s['dfs']['test'].STRING_protein_id.nunique(),'protein_disjoint':True,'homology_cluster_disjoint':split=='homology'}
        savecsv(s['dfs']['test'][['VariationID','STRING_protein_id','Label']+(['_cluster_id'] if split=='homology' else [])],O/f'{split}_test_rows.csv')
        out[split]=s
    manifest=manifest_parts['original'][['VariationID','Label','STRING_protein_id','original_split']].merge(manifest_parts['homology'][['VariationID','Label','STRING_protein_id','homology_split','_cluster_id']],on=['VariationID','Label','STRING_protein_id'],validate='one_to_one',how='inner').rename(columns={'Label':'label','_cluster_id':'homology_cluster_id'}).sort_values('VariationID').reset_index(drop=True)
    frozen=pd.read_csv(REF/'canonical/frozen_samples_177213.csv.gz').sort_values('VariationID').reset_index(drop=True)
    pd.testing.assert_frame_equal(manifest,frozen,check_dtype=False)
    assert len(out['original']['dfs']['test'])==18220
    assert out['original']['dfs']['test'].STRING_protein_id.nunique()==1413
    assert int(out['original']['labels']['test'].sum())==5822
    m.dump('input_audit.json',{'passed':True,'verified_hashes':verified,'frozen_manifest_equal':True,'node_order_equal':True,'splits':audit})
    return a,out

def canonical_prediction(split,arm,seed,z,s):
    q=O/'runs'/split/arm/f'seed_{seed}'
    cp=q/'model.pt'; digest=m.sha(cp)
    model=m.MLP(1280);model.load_state_dict(m.load(cp));model.eval()
    with torch.inference_mode():
        p=torch.cat([torch.sigmoid(model(z[i:i+1024])) for i in range(0,len(z),1024)]).numpy()
    assert p.dtype==np.float32 and np.isfinite(p).all() and p.min()>=0 and p.max()<=1
    np.save(q/'canonical_node_probs.npy',p)
    checks={}
    for part in ['val','test']:
        mapped=p[s['nodes'][part].numpy()];old=np.load(q/f'{part}_probs.npy',allow_pickle=False)
        err=float(np.max(np.abs(mapped-old)));assert err<=2e-6,(split,arm,seed,part,err)
        np.save(q/f'canonical_{part}_probs.npy',mapped)
        checks[part]={'max_abs_change_from_gpu_variant_batch':err,**m.metrics(s['labels'][part].numpy(),mapped)}
    assert m.sha(cp)==digest
    return {'split':split,'arm':arm,'seed':seed,'checkpoint_sha256':digest,'node_prediction_sha256':m.sha(q/'canonical_node_probs.npy'),'parts':checks}

def fit_three(split,arm,z,s,results,audits):
    get,dim=m.features(s,z,False)
    for seed in SEEDS:
        q=O/'runs'/split/arm/f'seed_{seed}'
        if q.exists():
            # A complete run may be resumed; no partially written fit is silently replaced.
            for f in ['metrics.json','history.json','test_probs.npy','val_probs.npy','model.pt']:assert (q/f).is_file(),f'Incomplete run: {q}'
            r=json.loads((q/'metrics.json').read_text());assert r['split']==split and r['arm']==arm and r['seed']==seed
        else:r=m.fit(split,arm,s,get,dim,seed)
        audits.append(canonical_prediction(split,arm,seed,z,s));results.append(r)
        savecsv(pd.DataFrame(results),O/'training_per_run_metrics.csv')
        m.dump('canonical_inference_audit.json',audits)
        status('training',completed=len(results),planned=39,last={'split':split,'arm':arm,'seed':seed})
    del get;torch.cuda.empty_cache()

def verify_endpoints(s):
    audit=[]
    for arm in ['full_FPP','edge_ablated_FPP']:
        ref=pd.read_csv(REF/f'canonical/test_predictions__original__{arm}.csv.gz',float_precision='round_trip')
        d=s['dfs']['test'];assert np.array_equal(ref.VariationID,d.VariationID)
        assert np.array_equal(ref.label,d.Label) and np.array_equal(ref.STRING_protein_id,d.STRING_protein_id)
        for seed in SEEDS:
            q=O/'runs/original'/arm/f'seed_{seed}';old=REF/'endpoint_runs'/arm/f'seed_{seed}'
            r=json.loads((q/'metrics.json').read_text());o=json.loads((old/'metrics.json').read_text())
            assert r['best_epoch']==o['best_epoch'] and r['epochs_run']==o['epochs_run'],(arm,seed,'epochs')
            errors={k:abs(r[k]-o[k]) for k in ['val_auc','val_auprc','test_auc','test_auprc']}
            assert max(errors.values())<1e-10,(arm,seed,errors)
            changes={part:float(np.max(np.abs(np.load(q/f'{part}_probs.npy')-np.load(old/f'{part}_probs.npy')))) for part in ['val','test']}
            assert max(changes.values())<=2e-6,(arm,seed,changes)
            canonical=np.load(q/'canonical_test_probs.npy').astype(np.float64)
            err=float(np.max(np.abs(canonical-ref[f'p_seed_{seed}'].to_numpy())))
            assert err<=2e-6,(arm,seed,'canonical',err)
            now=m.metrics(d.Label,canonical);prior=m.metrics(d.Label,ref[f'p_seed_{seed}'].to_numpy())
            canonical_metric_errors={k:abs(now[k]-prior[k]) for k in now}
            assert max(canonical_metric_errors.values())<1e-10,(arm,seed,'canonical metrics',canonical_metric_errors)
            audit.append({'arm':arm,'seed':seed,'selected_epoch_matches':True,'metric_abs_errors':errors,'probability_max_abs_errors':changes,'canonical_probability_max_abs_error':err,'canonical_metric_abs_errors':canonical_metric_errors})
    m.dump('endpoint_reproduction_gate.json',{'passed':True,'runs':audit,'rule':'All six fits reproduce selected epochs and recorded metrics; new canonical predictions within2e-6. All current curve endpoints use newly fitted runs from this invocation.'})

def representation(a,edge,name):
    p=O/'derived_embeddings'/f'{name}.pt';meta=p.with_suffix('.json')
    source={'edge_array_sha256':hash_array(edge.numpy()),'operator':'A=(edge+I)+(edge+I)^T; weighted row degree; (X+SX+S^2X)/3'}
    if p.exists():
        v=json.loads(meta.read_text());assert v['source']==source and v['sha256']==m.sha(p)
        z=m.emb(p)
    else:
        status('propagation',arm=name);z=m.propagate(a,edge)
        p.parent.mkdir(exist_ok=True);torch.save(z,p)
        meta.write_text(json.dumps({'source':source,'sha256':m.sha(p),'shape':list(z.shape)},indent=2)+'\n')
    assert z.shape==(19622,1280) and torch.isfinite(z).all()
    return z

def main():
    status('input_verification');a,splits=check_inputs()
    full=m.emb(D/'gcn_embeddings/gcn_node_embeddings.pt');cols=torch.arange(0,1280,80)
    check=m.propagate(a,a['edge_index'],cols)
    err=float((check-full[:,cols]).abs().max());assert torch.allclose(check,full[:,cols],rtol=2e-4,atol=2e-5),err
    m.dump('full_operator_gate.json',{'passed':True,'max_abs_error_16_columns':err,'saved_full_used':True})
    edge,ordered_pairs,pair_entries,pair_stats=pairs.identify_train_test_pairs_and_entries(a)
    assert pair_stats['num_train_test_edge_entries']==1242422 and len(ordered_pairs)==621211
    assert pair_stats['pair_entry_count_min']==pair_stats['pair_entry_count_max']==2
    ablated,removed=pairs.make_pair_level_ablated_edge_index(edge,pair_entries,ordered_pairs)
    assert len(removed)==1242422
    # Independently verify all-pair deletion equals the current Boolean edge control.
    n=len(a['x']);tr=torch.zeros(n,dtype=torch.bool);te=tr.clone()
    tr[a['train_protein_idx'].unique()]=True;te[a['test_protein_idx'].unique()]=True
    r,c=edge;mask=(tr[r]&te[c])|(te[r]&tr[c]);assert torch.equal(ablated,edge[:,~mask])
    m.dump('pair_identity_audit.json',{**pair_stats,'ordered_pairs_sha256':hash_array(np.asarray(ordered_pairs,dtype=np.int64)),'full_ablation_matches_matched_boolean_definition':True,'removed_entry_index_sha256':hash_array(np.sort(np.asarray(removed,dtype=np.int64)))})
    zabl=representation(a,ablated,'edge_ablated_FPP')
    results=[];audits=[]
    # Endpoints are both independent replication gates and shared curve endpoints.
    fit_three('original','full_FPP',full,splits['original'],results,audits)
    fit_three('original','edge_ablated_FPP',zabl,splits['original'],results,audits)
    verify_endpoints(splits['original'])
    del zabl,ablated
    for arm,z in [('raw_ESM',a['x'].float()),('full_FPP',full)]:
        fit_three('homology',arm,z,splits['homology'],results,audits)
    status('STRING_score_mapping')
    scores,score_stats=pairs.load_string_scores_for_pairs(ordered_pairs,{i:p for p,i in a['protein_to_idx'].items()})
    assert score_stats['num_pairs_missing_string_score']==0 and score_stats['num_pairs_with_string_score']==621211
    m.dump('score_coverage.json',score_stats)
    edge_rows=[]
    for strategy in STRATEGIES:
        for fraction in FRACTIONS:
            arm=f'{strategy}_fraction_{fraction:.2f}'
            graph_seed=42+int(fraction*1000)
            selected=pairs.select_pairs_for_removal(ordered_pairs,scores,fraction,strategy,graph_seed)
            ee,ii=pairs.make_pair_level_ablated_edge_index(edge,pair_entries,selected)
            assert len(selected)==round(fraction*621211) and len(ii)==2*len(selected)
            values=np.asarray([scores[p] for p in selected],dtype=float);assert np.isfinite(values).all()
            v={'arm':arm,'strategy':strategy,'fraction_removed':fraction,'graph_seed':graph_seed if strategy=='random_pair_removal' else None,'classifier_seeds':SEEDS,'pairs_removed':len(selected),'edge_entries_removed':len(ii),'edge_entries_remaining':int(ee.shape[1]),'selected_pair_sha256':hash_array(np.asarray(selected,dtype=np.int64)),'removed_entry_sha256':hash_array(np.sort(np.asarray(ii,dtype=np.int64))),'score_min':float(values.min()),'score_median':float(np.median(values)),'score_max':float(values.max())}
            edge_rows.append(v);m.dump('partial_graph_audits.json',edge_rows)
            # Store true compact pair membership for reproducible graph interventions.
            q=O/'graph_masks';q.mkdir(exist_ok=True)
            np.savez_compressed(q/f'{arm}.npz',selected_pairs=np.asarray(selected,dtype=np.int32),removed_entry_indices=np.asarray(ii,dtype=np.int32))
            z=representation(a,ee,arm);fit_three('original',arm,z,splits['original'],results,audits)
            del z,ee,ii,selected;torch.cuda.empty_cache()
    assert len(results)==39 and len(audits)==39
    m.dump('training_complete.json',{'complete':True,'classifier_fits':39,'endpoint_reproduction_fits':6,'homology_fits':6,'partial_interior_fits':27,'test_labels_used_for_model_selection':False,'selected_by':'validation AUC only','original_source_files_unchanged':True})
    status('training_complete',completed=39,planned=39)

if __name__=='__main__':
    try:main()
    except Exception:
        m.dump('failure.json',{'traceback':traceback.format_exc(),'time':time.time()});raise
