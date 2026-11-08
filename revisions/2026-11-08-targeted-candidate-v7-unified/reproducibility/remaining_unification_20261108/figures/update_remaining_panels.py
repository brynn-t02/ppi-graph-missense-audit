"""Replace Figure 1E / Figure 2D using completed, verified statistics.

Figure 1D's edge-ablated arm and ensemble annotation are also updated to the
same regenerated endpoint as E; its raw/full results remain unchanged.

Input schemas (full precision; native analysis aliases also accepted):
  curve: strategy,fraction_removed,test_auc_mean,test_auc_sd,raw_auc_mean,n_fits
  homology: arm,auc_mean,auc_sd,auprc_mean,auprc_sd,n_fits
All SD values must be sample SD (ddof=1), with seeds 42,43,44.
The curve subtracts a constant common raw-ESM mean from each model mean;
its error bars retain the original plot definition: SD of fitted-model AUC.
This script does not train, infer, choose checkpoints, or compute bootstrap.
"""
from pathlib import Path
from io import BytesIO
import argparse, hashlib, json, math, re, shutil, subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, NameObject, RectangleObject

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'outputs/BIBM_workshop_targeted_candidate_v6_figure3_20261108/source/figure_drafts'
FONTS=ROOT/'outputs/BIBM_workshop_font_clean_20261025/source/resources/fonts'
for name in ['DejaVuSans.ttf','DejaVuSans-Bold.ttf']:
    font_manager.fontManager.addfont(str(FONTS/name))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.titlesize':9.,'axes.labelsize':8.6,'xtick.labelsize':8.1,'ytick.labelsize':8.1,'legend.fontsize':7.4,'axes.linewidth':.8,'pdf.fonttype':42,'ps.fonttype':42})
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()

def make_form(fig, writer, width, height):
    buf=BytesIO();fig.savefig(buf,format='pdf',transparent=True);plt.close(fig);buf.seek(0)
    overlay=PdfReader(buf).pages[0]
    form=DecodedStreamObject();form.set_data(overlay.get_contents().get_data())
    form.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Form'),NameObject('/BBox'):RectangleObject([0,0,width,height]),NameObject('/Resources'):overlay['/Resources'].clone(writer)})
    return writer._add_object(form)

def clean(ax):
    ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)

def finish(writer, source, output, region, data_path, details, preserved_regions=None):
    writer.write(output)
    page=PdfReader(output).pages[0];h=float(page.mediabox.height);dpi=300
    for tag,p in [('before',source),('after',output)]:
        subprocess.run(['pdftoppm','-singlefile','-r',str(dpi),'-png',str(p),str(HERE/f'{output.stem}_{tag}')],check=True,capture_output=True)
    a=np.array(Image.open(HERE/f'{output.stem}_before.png').convert('RGB'))
    b=np.array(Image.open(HERE/f'{output.stem}_after.png').convert('RGB'))
    assert a.shape==b.shape
    changed=np.any(a!=b,axis=2);mask=np.zeros(changed.shape,bool)
    x0,y0,x1,y1=region
    mask[max(0,math.floor((h-y1)*dpi/72)):min(len(mask),math.ceil((h-y0)*dpi/72)), max(0,math.floor(x0*dpi/72)):min(mask.shape[1],math.ceil(x1*dpi/72))]=True
    outside=int((changed&~mask).sum());assert outside==0,outside
    preservation={}
    for name,box in (preserved_regions or {}).items():
        x0,y0,x1,y1=box
        r0,r1=max(0,math.ceil((h-y1)*dpi/72)),min(len(mask),math.floor((h-y0)*dpi/72))
        c0,c1=max(0,math.ceil(x0*dpi/72)),min(mask.shape[1],math.floor(x1*dpi/72))
        preservation[name]=int(changed[r0:r1,c0:c1].sum())
    assert all(v==0 for v in preservation.values()),preservation
    Image.open(HERE/f'{output.stem}_after.png').save(output.with_suffix('.png'))
    fonttool=shutil.which('pdffonts') or '/Users/teyian/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdffonts'
    fonts=subprocess.run([fonttool,str(output)],check=True,capture_output=True,text=True).stdout
    assert all(' no ' not in row for row in fonts.splitlines()[2:]),fonts
    output.with_suffix('.fonts.txt').write_text(fonts)
    audit={'source':str(source),'source_sha256':sha(source),'output':str(output),'output_sha256':sha(output),'input':str(data_path),'input_sha256':sha(data_path),'canvas_pt':[float(page.mediabox.width),h],'pixel_comparison_dpi':dpi,'changed_pixels':int(changed.sum()),'allowed_panel_region_pt':region,'outside_changed_pixels':outside,'preserved_region_changed_pixels':preservation,'all_other_panels_pixel_identical':True,'all_fonts_embedded':True,**details}
    output.with_suffix('.audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(audit,indent=2))

def figure1(path):
    df=pd.read_csv(path,float_precision='round_trip')
    # Native server-analysis output names are accepted without rounding.
    df=df.rename(columns={'auc_mean':'test_auc_mean','auc_sd':'test_auc_sd','n':'n_fits'})
    reference=ROOT/'outputs/BIBM_workshop_font_clean_20261025/source/canonical_analysis_20261025/per_run_metrics.csv'
    original=pd.read_csv(reference,float_precision='round_trip')
    raw=original[(original.split=='original')&(original.arm=='raw_ESM')]
    assert len(raw)==3 and sorted(raw.seed)==[42,43,44]
    raw_mean=float(raw.test_auc.mean())
    if 'raw_auc_mean' in df:assert np.all(df.raw_auc_mean.to_numpy()==raw_mean)
    else:df['raw_auc_mean']=raw_mean
    required=['strategy','fraction_removed','test_auc_mean','test_auc_sd','raw_auc_mean','n_fits']
    assert set(required)<=set(df)
    assert len(df)==15 and (df.n_fits==3).all()
    strategies=['random_pair_removal','high_confidence_first','low_confidence_first']
    assert set(df.strategy)==set(strategies)
    for s in strategies:assert sorted(df.loc[df.strategy==s,'fraction_removed'].tolist())==[0,.25,.5,.75,1]
    assert df.raw_auc_mean.nunique()==1
    assert (df.test_auc_sd>=0).all() and (df.test_auc_sd<.1).all()
    df['delta_auc']=df.test_auc_mean-df.raw_auc_mean
    # Endpoints are the same fitted models reused across the three strategies.
    for f in [0,1]:
        endpoint=df[df.fraction_removed==f]
        assert endpoint[['test_auc_mean','test_auc_sd']].nunique().max()==1
    source=BASE/'Figure1_connectivity_topology_controls_v2_accepted_minimal_clarified.pdf'
    writer=PdfWriter(clone_from=source);page=writer.pages[0]
    width,height=map(float,[page.mediabox.width,page.mediabox.height])
    fig=plt.figure(figsize=(width/72,height/72));axis=[374.490922,34.66875,211.523156,180.502326]
    x,y,w,h=axis;ax=fig.add_axes([x/width,y/height,w/width,h/height])
    ax.set_title('E  Partial train-test edge ablation',loc='left',fontweight='bold')
    styles={'random_pair_removal':('Random removal','#636363','o'),'high_confidence_first':('High-confidence first','#DE2D26','s'),'low_confidence_first':('Low-confidence first','#3182BD','^')}
    for strategy,(label,color,marker) in styles.items():
        d=df[df.strategy==strategy].sort_values('fraction_removed')
        ax.errorbar(d.fraction_removed*100,d.delta_auc,yerr=d.test_auc_sd,color=color,ecolor=color,marker=marker,markersize=4.1,lw=1.6,capsize=2.4,label=label)
    lo=(df.delta_auc-df.test_auc_sd).min();hi=(df.delta_auc+df.test_auc_sd).max()
    low=min(-.015,math.floor((lo-.001)/.005)*.005);high=max(.03,math.ceil((hi+.001)/.005)*.005)
    # Retain the upper-right legend; add only vertical headroom if a new SD
    # bar would intrude into its three rows. The data/SD are never clipped.
    while high-(high-low)*.22 < hi+.001:
        high=round(high+.005,8)
    ax.axhline(0,color='#4D4D4D',lw=.85,ls='--')
    ax.set_xlabel('Removed direct train-test PPI pairs (%)');ax.set_ylabel('Delta AUC vs ESM raw')
    ax.set_xlim(-3,103);ax.set_ylim(low,high);ax.set_yticks(np.arange(low,high+.000001,.005));ax.legend(frameon=False,loc='upper right');clean(ax)
    # Only replace this pre-existing Form. All page commands and other Forms
    # remain the same, including the two annotation wording edits in v2.
    raw=page.get_contents().get_data();assert b'/PanelEReplacement Do' in raw
    page['/Resources']['/XObject'][NameObject('/PanelEReplacement')]=make_form(fig,writer,width,height)
    assert page.get_contents().get_data()==raw
    # The first regenerated, common-operator edge endpoint is shared with D.
    # Preserve the raw/full D means and SD; update only its edge arm and the
    # paired ensemble annotation, avoiding two inconsistent 100% endpoints.
    bootstrap_path=path.with_name('figure1D_protein_bootstrap.json')
    boot=json.loads(bootstrap_path.read_text())
    assert boot['comparison']=='full_FPP minus edge_ablated_FPP'
    assert boot['resamples']==2000 and boot['variants']==18220
    d_summary={}
    for arm in ['raw_ESM','full_FPP']:
        d=original[(original.split=='original')&(original.arm==arm)]
        assert len(d)==3
        d_summary[arm]={'auc_mean':float(d.test_auc.mean()),'auc_sd':float(d.test_auc.std(ddof=1)), 'auprc_mean':float(d.test_auprc.mean()),'auprc_sd':float(d.test_auprc.std(ddof=1))}
    endpoint=df[df.fraction_removed==1].iloc[0]
    d_summary['edge_ablated_FPP']={'auc_mean':float(endpoint.test_auc_mean),'auc_sd':float(endpoint.test_auc_sd),'auprc_mean':float(endpoint.auprc_mean),'auprc_sd':float(endpoint.auprc_sd)}
    numeric=page['/Resources']['/XObject']['/UnifiedNumericPanels'].get_object()
    numeric_raw=numeric.get_data()
    start=numeric_raw.index(b'45.0125 34.66875 m');start=numeric_raw.rfind(b'/A1 gs',0,start)
    end=numeric_raw.index(b'691.526805 34.66875 m');end=numeric_raw.rfind(b'/A1 gs',0,end)
    assert start>0 and end>start
    numeric.set_data(numeric_raw[:start]+numeric_raw[end:])
    annotations=page['/Resources']['/XObject']['/TwoAnnotationTextV2'].get_object()
    ar=annotations.get_data()
    old_number_block=b'q\n1 0 -0 1 194.1858719054 145.2056446667 cm\nBT\n/F1 7.4 Tf\n0 0 Td\n[ ('+'ΔAUC +0.0248'.encode('utf-16be')+b') ] TJ\nET\nQ'
    assert ar.count(old_number_block)==1
    annotations.set_data(ar.replace(old_number_block,b''))
    dfig=plt.figure(figsize=(width/72,height/72));d_axis=[45.0125,34.66875,223.965695,180.502326]
    x,y,w,h=d_axis;dax=dfig.add_axes([x/width,y/height,w/width,h/height]);xx=np.arange(2);bw=.24
    dax.set_title('D  Removing direct train-test PPI edges',loc='left',fontweight='bold')
    for off,arm,col,label in zip(np.linspace(-bw,bw,3),['raw_ESM','full_FPP','edge_ablated_FPP'],['#9ECAE1','#3182BD','#FDD0A2'],['ESM raw','Full FPP','Edge-ablated FPP']):
        s=d_summary[arm]
        dax.bar(xx+off,[s['auc_mean'],s['auprc_mean']],width=bw,yerr=[s['auc_sd'],s['auprc_sd']],capsize=2.2,color=col,edgecolor='black',linewidth=.45,label=label)
    dax.set_xticks(xx);dax.set_xticklabels(['AUC','AUPRC']);dax.set_ylabel('Test performance');dax.set_ylim(.56,.83);dax.legend(frameon=False,loc='upper right',fontsize=7.);clean(dax)
    number_text=f"ΔAUC {boot['delta_auc']:+.4f}"
    dfig.text(221.6702469054/width,145.2056446667/height,number_text,ha='center',va='baseline',fontsize=7.4,color='#4D4D4D')
    page['/Resources']['/XObject'][NameObject('/MatchedEdgeEndpointD')]=make_form(dfig,writer,width,height)
    stream=DecodedStreamObject();stream.set_data(raw+b'\nq\n/MatchedEdgeEndpointD Do\nQ\n');page.replace_contents(stream)
    writer.add_metadata({'/Title':'Figure 1: matched partial edge-ablation sensitivity','/Subject':'Panels D/E share the first regenerated matched edge endpoint; E uses the common propagation operator and three classifier seeds.'})
    output=HERE/source.name
    finish(writer,source,output,[0,0,625,237],path,{'panels':['D','E'],'axes_E_pt':axis,'axes_D_pt':d_axis,'raw_reference_file':str(reference),'raw_reference_sha256':sha(reference),'raw_reference_mean_auc':raw_mean,'mean_and_SD_source':'three classifiers per configuration; sample SD ddof=1','delta_definition':'mean model AUC minus common mean raw-ESM AUC','errorbar_definition':'sample SD of model AUC; reference mean treated as a constant shift','y_limits_previous':[-.015,.030],'y_limits_new':[low,high],'y_limit_change_reason':'Show all new sample SD bars and retain clear space below the original upper-right legend.','points_E':df.to_dict('records'),'points_D':d_summary,'D_ensemble_annotation':number_text,'D_bootstrap_source':str(bootstrap_path),'D_bootstrap_sha256':sha(bootstrap_path),'D_full_minus_edge_bootstrap':boot,'D_raw_full_metrics_unchanged':True},preserved_regions={'D raw AUC bar':[54,35,83,192],'D full AUC bar':[83,35,112,205],'D raw AP bar':[173,35,201,90],'D full AP bar':[201,35,229,108]})

def figure2(path):
    df=pd.read_csv(path,float_precision='round_trip')
    df=df.rename(columns={'n':'n_fits'})
    required=['arm','auc_mean','auc_sd','auprc_mean','auprc_sd','n_fits']
    assert set(required)<=set(df);assert len(df)==2 and (df.n_fits==3).all()
    assert set(df.arm)=={'raw_ESM','full_FPP'}
    df=df.set_index('arm').loc[['raw_ESM','full_FPP']].reset_index()
    assert (df[['auc_sd','auprc_sd']]>=0).all().all()
    source=BASE/'Figure2_fusion_homology_v1_accepted.pdf';writer=PdfWriter(clone_from=source);page=writer.pages[0]
    width,height=map(float,[page.mediabox.width,page.mediabox.height])
    raw=page.get_contents().get_data();start=raw.index(b'39.653125 34.66875 m');start=raw.rfind(b'/A2 gs',0,start)
    end=raw.index(b'810.620434 34.66875 m');end=raw.rfind(b'/A2 gs',0,end)
    assert start>0 and end>start
    retained=raw[:start]+raw[end:]
    fig=plt.figure(figsize=(width/72,height/72));axis=[39.653125,34.66875,288.2500206697,174.0895522388]
    x,y,w,h=axis;ax=fig.add_axes([x/width,y/height,w/width,h/height])
    ax.set_title('D  Homology-aware split: ESM vs PPI-GCN',loc='left',fontweight='bold')
    xx=np.arange(2);bar_width=.34
    for off,(_,r),color,label in zip([-bar_width/2,bar_width/2],df.iterrows(),['#9ECAE1','#3182BD'],['ESM','PPI-GCN']):
        ax.bar(xx+off,[r.auc_mean,r.auprc_mean],bar_width,yerr=[r.auc_sd,r.auprc_sd],capsize=2.2,color=color,edgecolor='black',linewidth=.4,label=label)
    ax.set_xticks(xx);ax.set_xticklabels(['AUC','AUPRC']);ax.set_ylabel('Test performance')
    low=min(.18,math.floor(float((df[['auc_mean','auprc_mean']].to_numpy()-df[['auc_sd','auprc_sd']].to_numpy()).min())*20)/20)
    high=max(.80,math.ceil(float((df[['auc_mean','auprc_mean']].to_numpy()+df[['auc_sd','auprc_sd']].to_numpy()).max())*20)/20)
    ax.set_ylim(low,high);ax.legend(frameon=False,loc='upper right');clean(ax)
    page['/Resources']['/XObject'][NameObject('/MatchedHomologyRepresentations')]=make_form(fig,writer,width,height)
    stream=DecodedStreamObject();stream.set_data(retained+b'\nq\n/MatchedHomologyRepresentations Do\nQ\n');page.replace_contents(stream)
    writer.add_metadata({'/Title':'Figure 2: matched homology representation comparison','/Subject':'Panel D uses mean and sample SD across classifier seeds 42,43,44; other panels unchanged.'})
    output=HERE/source.name
    finish(writer,source,output,[0,0,345,237],path,{'panel':'D','axes_pt':axis,'original_D_byte_range_removed':[start,end],'mean_and_SD_source':'three classifiers per arm; sample SD ddof=1','y_limits_previous':[.18,.80],'y_limits_new':[low,high],'points':df.to_dict('records')})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--curve',type=Path);p.add_argument('--homology',type=Path);a=p.parse_args()
    assert a.curve or a.homology
    if a.curve:figure1(a.curve)
    if a.homology:figure2(a.homology)
