from pathlib import Path
import ast,json,hashlib
p=Path(__file__).with_name('reflow_figures12.py')
g={'__file__':str(p.resolve())}
exec(p.read_text().split("audit={'canvas_pt'")[0],g)
w=p.parent
pack={}
for n in [1,2]:
 d=g[f'f{n}'];src=g[f'p{n}'];tree=ast.parse(src.read_text())
 defs=[ast.get_source_segment(src.read_text(),x) for x in tree.body if isinstance(x,ast.FunctionDef)]
 (w/f'original_plot_functions_figure{n}.py').write_text('\n\n'.join(defs)+'\n')
 keys=[k for k,v in d.items() if k.startswith('COL_')]
 if n==1:keys += ['ESM_AUC','ESM_AUPRC','GCN_AUC','GCN_AUPRC','VAR_AUC','VAR_AUPRC','PROT_AUC','PROT_AUPRC','EDGE_AUC_MEAN','EDGE_AUC_SD','EDGE_AUPRC_MEAN','EDGE_AUPRC_SD','PERM_AUC_MEAN','PERM_AUC_SD','PERM_AUPRC_MEAN','PERM_AUPRC_SD']
 else:keys += ['cadd_report','am_report','homo','orig_caddam','orig_raw','orig_edge','orig_perm','orig_real','homo_caddam','homo_raw','margin_rows','orig_prev','homo_prev']
 pack[str(n)]={k:d[k] for k in keys}
 if n==1:pack['1']['df44_records']=d['df44'].to_dict('records')
 paths=[]
 if n==1:paths=[d[k] for k in ['P_03F','P_12','P_42A','P_41','P_44','P_45']]
 else:paths=[d['P']/'cadd_match_report.json',d['P']/'alphamissense_match_report.json',d['P']/'homology_rerun_core/homology_rerun_summary.json']+[d['SOURCE']/q for q in ['exp46_cadd_am_raw_esm_baseline/summary_metrics.csv','exp47_cadd_am_graph_control_fusions/summary_metrics.csv','cadd_am_architecture_matched_exp48/summary_metrics.csv','fusion_control_margin_bootstrap_exp49/fusion_control_margin_bootstrap_summary.csv']]+[d['P']/'clinvar_test_main.csv',d['P']/'homology_aware_split/clinvar_test_homology.csv']
 pack[str(n)]['source_provenance']=[{'path':str(q),'bytes':q.stat().st_size,'sha256':hashlib.sha256(q.read_bytes()).hexdigest()} for q in [src]+paths]
(w/'compact_plot_inputs.json').write_text(json.dumps(pack,indent=2))
print('Frozen compact plotting inputs only:',(w/'compact_plot_inputs.json').stat().st_size)
