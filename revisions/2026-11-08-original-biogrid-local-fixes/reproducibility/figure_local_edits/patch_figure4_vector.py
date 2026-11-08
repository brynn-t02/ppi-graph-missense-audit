#!/usr/bin/env python3
"""Exact local Figure 4 patch: descriptive titles, uniform E dots, no F stars.
No data are loaded or recalculated. The original vector figure is immutable.
"""
from pathlib import Path
import hashlib,json,math,subprocess,copy
import numpy as np
from PIL import Image
from pypdf import PdfReader,PdfWriter
from pypdf.generic import ArrayObject,ByteStringObject,ContentStream,FloatObject,NameObject
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
SOURCE=HERE/'Figure4_before.pdf'
OUTPUT=ROOT/'outputs/BIBM_workshop_original_BioGRID_local_fixes_20261108/source/figure_drafts/Figure4_prediction_transitions_v2_accepted.pdf'
EXPECTED_SHA='835c6840ba23c8f95c0275e4df110c5ee6af8b369c48f19c996e5d93ff17ed64'
TITLES={
 'E  Topology features separate corrected groups':'E  Directions of mean differences between corrected groups',
 'F  Difficulty-matched contrasts retain topology differences':'F  Rescued versus persistent false-negative profiles',
}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(SOURCE)==EXPECTED_SHA
reader=PdfReader(SOURCE);writer=PdfWriter(clone_from=SOURCE);page=writer.pages[0]
# The original bold subset lacks lowercase b. Use the same DejaVu Sans Bold
# file, newly subset for the two titles only; all original fonts stay untouched.
title_font_reader=PdfReader(HERE/'title_font_resource.pdf')
title_font=title_font_reader.pages[0]['/Resources']['/Font']['/F1'].get_object().clone(writer)
page['/Resources']['/Font'][NameObject('/FTitle')]=writer._add_object(title_font)
stream=ContentStream(page.get_contents(),writer);ops=stream.operations;before=copy.deepcopy(ops)
edited=set();circles=[];removals=set();stars=[];title_changes=[]
assert float(page['/Resources']['/ExtGState']['/A3']['/ca'])==0.85
for i,(args,operator) in enumerate(ops):
 if operator!=b'gs' or str(args[0]) not in ('/A3','/A4'):continue
 start=i+6
 assert ops[start][1]==b'm'
 assert all(ops[j][1]==b'c' for j in range(start+1,start+9))
 assert ops[start+9][1]==b'h' and ops[start+10][1]==b'B'
 coords=np.array([float(v) for j in range(start,start+9) for v in ops[j][0]]).reshape(-1,2)
 lo,hi=coords.min(axis=0),coords.max(axis=0);center=(lo+hi)/2;radius=(hi-lo)/2
 assert np.allclose(radius[0],radius[1],atol=2e-6)
 assert 490<center[0]<630 and 45<center[1]<205
 for j in range(start,start+9):
  values=np.array(list(map(float,ops[j][0]))).reshape(-1,2)
  values=center+(values-center)*4.0/radius
  ops[j]=([FloatObject(float(v)) for v in values.ravel()],ops[j][1]);edited.add(j)
 ops[i]=([NameObject('/A3')],operator);edited.add(i)
 circles.append({'center_pdf_points':center.tolist(),'original_radius':radius.tolist(),'new_radius':4.0,'old_alpha_resource':str(args[0]),'new_alpha':0.85})
assert len(circles)==18
for i,(args,operator) in enumerate(ops):
 if operator!=b'TJ':continue
 text=''.join(str(v) for v in args[0] if isinstance(v,str))
 if text=='*':
  assert [ops[j][1] for j in range(i-5,i+3)]==[b'q',b'cm',b'BT',b'Tf',b'Td',b'TJ',b'ET',b'Q']
  trans=list(map(float,ops[i-4][0]));assert 990<trans[4]<1005 and 50<trans[5]<200
  removals.update(range(i-5,i+3));stars.append(trans[4:])
 elif text in TITLES:
  old_font=str(ops[i-2][0][0])
  ops[i-2]=([NameObject('/FTitle'),ops[i-2][0][1]],b'Tf');edited.add(i-2)
  ops[i]=([ArrayObject([ByteStringObject(TITLES[text].encode('utf-16be'))])],operator);edited.add(i)
  title_changes.append({'before':text,'after':TITLES[text],'transform':list(map(float,ops[i-4][0])),'old_font_resource':old_font,'font':str(ops[i-2][0][0]),'font_size':float(ops[i-2][0][1]),'same_family':'DejaVu Sans Bold'})
assert len(stars)==7 and len(title_changes)==2
unchanged=[i for i in range(len(ops)) if i not in edited and i not in removals]
assert all(ops[i]==before[i] for i in unchanged)
# All color-setting operators, rectangular bars, clipping paths and text positions remain unchanged.
for op in [b'rg',b'RG',b'g',b'G',b're',b'cm',b'Td']:
 assert [v for i,(v,o) in enumerate(ops) if o==op and i not in removals]==[v for i,(v,o) in enumerate(before) if o==op and i not in removals]
stream.operations=[x for i,x in enumerate(ops) if i not in removals];page.replace_contents(stream)
writer.add_metadata({'/Title':'Figure 4: prediction transitions and descriptive topology profiles','/Subject':'E uniform direction dots; F descriptive effect sizes without significance markers.'})
with OUTPUT.open('wb') as f:writer.write(f)
r=PdfReader(OUTPUT);text=r.pages[0].extract_text()
assert '*' not in text and all(x not in text for x in TITLES) and all(x in text for x in TITLES.values())
assert np.allclose(list(map(float,reader.pages[0].mediabox)),list(map(float,r.pages[0].mediabox)),atol=1e-5,rtol=0)
# Inspect every font descriptor, including Type0 descendant fonts.
def fonts_embedded(pdf):
 fonts=[]
 for key,ref in pdf.pages[0]['/Resources']['/Font'].items():
  f=ref.get_object();descendants=f.get('/DescendantFonts',[f]);embedded=[]
  for d in descendants:
   d=d.get_object();fd=d.get('/FontDescriptor');fd=fd.get_object() if fd else {}
   embedded.append(any(k in fd and len(fd[k].get_object().get_data())>0 for k in ['/FontFile','/FontFile2','/FontFile3']))
  fonts.append({'resource':key,'name':str(f.get('/BaseFont')),'embedded':all(embedded)})
 return fonts
font_report=fonts_embedded(r);assert all(x['embedded'] for x in font_report)
dpi=300
for label,p in [('before_300',SOURCE),('after_300',OUTPUT)]:
 run=subprocess.run(['pdftoppm','-singlefile','-r',str(dpi),'-png',str(p),str(HERE/label)],capture_output=True,text=True,check=True)
 assert not run.stderr.strip(),run.stderr
arr0=np.array(Image.open(HERE/'before_300.png').convert('RGB'));arr1=np.array(Image.open(HERE/'after_300.png').convert('RGB'))
assert arr0.shape==arr1.shape
changed=np.any(arr0!=arr1,axis=2);allowed=np.zeros(changed.shape,dtype=bool)
width,height=map(float,[page.mediabox.width,page.mediabox.height])
regions={'panel_E_dots':[495,47,625,206],'panel_F_stars':[994,51,1008,202],'panel_E_title':[448,212,780,228],'panel_F_title':[786,212,width,228]}
def mask_rect(rect):
 x0,y0,x1,y1=rect;c0,c1=math.floor(x0*dpi/72)-2,math.ceil(x1*dpi/72)+2;r0,r1=math.floor((height-y1)*dpi/72)-2,math.ceil((height-y0)*dpi/72)+2
 return slice(max(0,r0),min(allowed.shape[0],r1)),slice(max(0,c0),min(allowed.shape[1],c1))
for rect in regions.values():allowed[mask_rect(rect)]=True
outside=int((changed & ~allowed).sum());assert outside==0,outside
# F plotted bars span x≈788..1009, y≈40..210; only the right-hand star strip can differ.
f_region=np.zeros_like(allowed);f_region[mask_rect([786,40,1010,209])]=True
f_stars=np.zeros_like(allowed);f_stars[mask_rect(regions['panel_F_stars'])]=True
f_changed=int((changed & f_region & ~f_stars).sum());assert f_changed==0
report={'passed':True,'source_sha256':EXPECTED_SHA,'output_sha256':sha(OUTPUT),'source_unchanged':sha(SOURCE)==EXPECTED_SHA,'original_operations':len(before),'edited_operations':len(edited),'removed_operations':len(removals),'all_unlisted_operations_unchanged':True,'titles':title_changes,'uniform_dots':circles,'removed_F_stars':stars,'fonts':font_report,'all_fonts_embedded':True,'same_canvas':True,'pixel_check':{'dpi':dpi,'dimensions':[arr1.shape[1],arr1.shape[0]],'total_changed_pixels':int(changed.sum()),'outside_allowed_regions_changed_pixels':outside,'allowed_regions_pdf_points':regions,'A_B_C_D_all_pixels_unchanged':True,'F_bars_colors_axes_pixel_changes':f_changed},'data_or_statistics_recomputed':False,'title_positions_family_and_size_unchanged':True,'title_font_subset_augmented':True,'dot_centers_and_color_operators_unchanged':True}
(HERE/'figure4_vector_patch_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'output':str(OUTPUT),'audit':str(HERE/'figure4_vector_patch_audit.json'),'outside_allowed_pixels':outside,'F_bar_area_changes':f_changed,'all_fonts_embedded':True}))
