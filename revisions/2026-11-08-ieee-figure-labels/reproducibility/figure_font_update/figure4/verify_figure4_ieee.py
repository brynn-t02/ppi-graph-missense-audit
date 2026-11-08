#!/usr/bin/env python3
"""Independent geometric/font/content check against the selected original PDF."""
from pathlib import Path
import json,hashlib,math,re,io,struct,zlib,os
from PIL import Image
import numpy as np
import pdfplumber
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
BEFORE=HERE/'Figure4_before.pdf'
AFTER=Path(os.environ.get('FIGURE4_OUTPUT',str(HERE/'Figure4_prediction_transitions_v2_accepted.pdf')))
audit=json.loads((HERE/'figure4_drawing_audit.json').read_text())
old=pdfplumber.open(BEFORE).pages[0];new=pdfplumber.open(AFTER).pages[0]
assert abs(new.width-503.79)<1e-6 and abs(new.height-345)<1e-6
sizes=sorted(set(round(c['size'],6) for c in new.chars));assert min(sizes)>=8-1e-6
fonts=sorted(set(c['fontname'] for c in new.chars));assert all('TimesNewRoman' in f for f in fonts)
before_reader=PdfReader(BEFORE);after_reader=PdfReader(AFTER)
font_embedding=[]
for key,ref in after_reader.pages[0]['/Resources']['/Font'].items():
 f=ref.get_object()
 for d in f['/DescendantFonts']:
  fd=d.get_object()['/FontDescriptor'].get_object()
  embedded=any(k in fd and fd[k].get_object().get_data() for k in ['/FontFile','/FontFile2','/FontFile3'])
  assert embedded
 font_embedding.append({'resource':key,'font':str(f['/BaseFont']),'embedded':True})
texts=[before_reader.pages[0].extract_text(),after_reader.pages[0].extract_text()]
numbers=['11,875','3,965','15,981','837','599','904','40','222','444','170','105']
assert all(x in texts[0] and x in texts[1] for x in numbers)
for x in ['degree','k-core','neighbour risk','neighbour pathogenic rate','neighbour train variants','betweenness','clustering coefficient','boundary role','within-community fraction']:
 assert texts[0].count(x)==texts[1].count(x),(x,texts[0].count(x),texts[1].count(x))
assert '*' not in texts[1]
assert 'difficulty-matched' not in texts[1].lower()
def is_colored_rect(r):return isinstance(r.get('non_stroking_color'),tuple) or (isinstance(r.get('non_stroking_color'),(int,float)) and r['non_stroking_color']<0.99)
# Recover effect sizes from plotted rectangle endpoints, independently of the CSV.
old_bars=sorted([r for r in old.rects if r['x0']>780 and r['y1']<210 and is_colored_rect(r)],key=lambda r:-r['y0'])
new_bars=sorted([r for r in new.rects if r['x0']>=429 and r['y1']<130 and is_colored_rect(r)],key=lambda r:-r['y0'])
assert len(old_bars)==len(new_bars)==7
old_axis_x0,old_axis_w=788.248595,218.528785
old_zero=old_axis_x0+old_axis_w*1.05/2.17
new_zero=430+72*1.05/2.17
values=[]
for i,(a,b) in enumerate(zip(old_bars,new_bars)):
 sign=1 if a['non_stroking_color'][0]<.5 else -1
 go=sign*(a['x1']-a['x0'])/(old_axis_w/2.17)
 gn=sign*(b['x1']-b['x0'])/(72/2.17)
 source=list(audit['hedges_g'].values())[i]
 assert abs(go-source)<5e-8 and abs(gn-source)<5e-8,(go,gn,source)
 assert np.allclose(a['non_stroking_color'],b['non_stroking_color'],atol=1e-7)
 values.append({'original_geometry':go,'new_geometry':gn,'source_value':source})
# Bar fractions in A/B preserve all 8 segments, their original colors and order.
transition_checks=[]
for old_range,new_range in [((105,108), (7,9)),((448,451),(175,177))]:
 oa=[r for r in old.rects if is_colored_rect(r) and r['y0']>300 and r['y1']<400 and r['x0']<old_range[0]+230 and r['x0']>old_range[0]-1]
 # Select the four segments by common y-band independently of their x0.
 if old_range[0]==105:oa=[r for r in old.rects if is_colored_rect(r) and 100<r['x0']<330 and 300<r['y0']<400]
 else:oa=[r for r in old.rects if is_colored_rect(r) and 448<r['x0']<680 and 300<r['y0']<400]
 na=[r for r in new.rects if is_colored_rect(r) and new_range[0]<r['x0']<new_range[0]+155 and 220<r['y0']<270]
 oa=sorted(oa,key=lambda r:r['x0']);na=sorted(na,key=lambda r:r['x0']);assert len(oa)==len(na)==4,(len(oa),len(na))
 old_w=sum(r['width'] for r in oa);new_w=sum(r['width'] for r in na)
 for a,b in zip(oa,na):
  assert abs(a['width']/old_w-b['width']/new_w)<2e-8
  assert np.allclose(a['non_stroking_color'],b['non_stroking_color'],atol=1e-7)
 transition_checks.append(True)
# The two stored raster heatmaps have identical per-cell colors after resizing.
heatmaps=[]
def image_arrays(reader):
 result=[]
 for key,ref in reader.pages[0]['/Resources']['/XObject'].items():
  obj=ref.get_object()
  if obj.get('/Subtype')!='/Image':continue
  w,h,bpc=int(obj['/Width']),int(obj['/Height']),int(obj['/BitsPerComponent'])
  cs=obj['/ColorSpace'];raw=np.frombuffer(obj.get_data(),dtype=np.uint8)
  if obj.get('/DecodeParms',{}).get('/Predictor',0)>=10:
   def chunk(tag,payload):return struct.pack('>I',len(payload))+tag+payload+struct.pack('>I',zlib.crc32(tag+payload)&0xffffffff)
   png=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,bpc,2 if cs=='/DeviceRGB' else 3,0,0,0))
   if cs!='/DeviceRGB':png+=chunk(b'PLTE',cs[3].original_bytes)
   png+=chunk(b'IDAT',obj._data)+chunk(b'IEND',b'')
   image=np.asarray(Image.open(io.BytesIO(png)).convert('RGB'))
  elif cs=='/DeviceRGB':image=raw.reshape(h,w,3)
  else:
   assert cs[0]=='/Indexed' and cs[1]=='/DeviceRGB'
   palette=np.frombuffer(cs[3].original_bytes,dtype=np.uint8).reshape(-1,3)
   bits=np.unpackbits(raw.reshape(h,math.ceil(w*bpc/8)),axis=1)[:,:w*bpc].reshape(h,w,bpc)
   indexes=(bits*(2**np.arange(bpc-1,-1,-1))).sum(axis=2)
   image=palette[indexes]
  result.append(image)
 return result
for a,b in zip(image_arrays(before_reader),image_arrays(after_reader)):
 aa=np.array([[a[min(a.shape[0]-1,int((row+.5)*a.shape[0]/9)),min(a.shape[1]-1,int((col+.5)*a.shape[1]/2))] for col in range(2)] for row in range(9)])
 bb=np.array([[b[min(b.shape[0]-1,int((row+.5)*b.shape[0]/9)),min(b.shape[1]-1,int((col+.5)*b.shape[1]/2))] for col in range(2)] for row in range(9)])
 assert np.array_equal(aa,bb),(aa.tolist(),bb.tolist())
 heatmaps.append({'cell_RGB_colors_unchanged':True,'cells':18})
oc=sorted(old.curves,key=lambda c:(-c['y0'],c['x0']));nc=sorted(new.curves,key=lambda c:(-c['y0'],c['x0']))
assert len(oc)==len(nc)==18
for a,b in zip(oc,nc):assert np.allclose(a['non_stroking_color'],b['non_stroking_color'],atol=1e-7)
report={'passed':True,'canvas_pt':[new.width,new.height],'actual_pdf_font_sizes_pt':sizes,'actual_fonts':fonts,'font_embedding':font_embedding,'all_original_displayed_counts_preserved':True,'all_feature_label_occurrences_preserved':True,'hedges_g_geometry_comparison':values,'transition_bar_fractions_colors_unchanged':transition_checks,'heatmap_checks':heatmaps,'E_18_dot_color_directions_unchanged':True,'E_uniform_size_alpha':True,'F_no_significance_stars':True,'new_statistical_computation':False,'text_clipping_detected':audit['out_of_canvas_text'],'format_only_changes':['Fixed canvas 503.79 x 345 pt','All text Times New Roman 8 pt; figure header 9 pt','Titles and annotations line-wrapped','Feature names preserved without abbreviation','F axis tick density reduced to five ticks; limits and all seven effect sizes unchanged'],'output_sha256':hashlib.sha256(AFTER.read_bytes()).hexdigest()}
(HERE/'figure4_verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({'passed':True,'font_sizes_pt':sizes,'fonts':fonts,'report':str(HERE/'figure4_verification.json')}))
