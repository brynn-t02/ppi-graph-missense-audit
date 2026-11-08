"""Clarify one Figure1D annotation; preserve every scientific graphic byte."""
from pathlib import Path
import hashlib, json, math, re, shutil, subprocess
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
NAME='Figure1_connectivity_topology_controls_v2_accepted_minimal_clarified.pdf'
SRC=ROOT/'outputs/BIBM_workshop_targeted_candidate_v7_unified_20261108/source/figure_drafts'/NAME
OUT=ROOT/'outputs/BIBM_workshop_targeted_candidate_v8_clarity_20261108/source/figure_drafts'/NAME
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
source_sha=sha(SRC)
old='ΔAUC +0.0264';new='Ensemble ΔAUC +0.0264'
writer=PdfWriter(clone_from=SRC);page=writer.pages[0]
form=page['/Resources']['/XObject']['/MatchedEdgeEndpointD'].get_object()
raw=form.get_data()
font=form['/Resources']['/Font']['/F1'].get_object()
assert font['/BaseFont'].endswith('+DejaVuSans')
cid=font['/DescendantFonts'][0].get_object();widths={};w=cid['/W'];i=0
while i<len(w):
    start=int(w[i]);next_=w[i+1]
    if isinstance(next_,list):
        for j,value in enumerate(next_):widths[start+j]=float(value)
        i+=2
    else:
        for j in range(start,int(next_)+1):widths[j]=float(w[i+2])
        i+=3
assert all(ord(c) in widths for c in new)
size=7.4
old_width=sum(widths[ord(c)] for c in old)*size/1000
new_width=sum(widths[ord(c)] for c in new)*size/1000
old_left=194.1858719054
new_left=old_left-(new_width-old_width)/2
old_block=b'1 0 -0 1 194.1858719054 145.2056446667 cm\nBT\n/F1 7.4 Tf\n0 0 Td\n[ ('+old.encode('utf-16be')+b') ] TJ'
new_block=f'1 0 -0 1 {new_left:.10f} 145.2056446667 cm\nBT\n/F1 7.4 Tf\n0 0 Td\n[ ('.encode()+new.encode('utf-16be')+b') ] TJ'
assert raw.count(old_block)==1
replacement=raw.replace(old_block,new_block)
form.set_data(replacement)
writer.write(OUT)

# All page instructions and all other Forms remain byte-for-byte intact.
source=PdfReader(SRC).pages[0];updated=PdfReader(OUT).pages[0]
assert source.get_contents().get_data()==updated.get_contents().get_data()
for name,value in source['/Resources']['/XObject'].items():
    old_data=value.get_object().get_data()
    new_data=updated['/Resources']['/XObject'][name].get_object().get_data()
    assert new_data==(replacement if name=='/MatchedEdgeEndpointD' else old_data),name
old_text=source.extract_text();new_text=updated.extract_text()
assert old_text.count(old)==1 and new_text.count(new)==1
assert new_text.replace(new,old)==old_text
assert sha(SRC)==source_sha

dpi=300;height=float(source.mediabox.height)
for tag,path in [('before',SRC),('after',OUT)]:
    subprocess.run(['pdftoppm','-singlefile','-r',str(dpi),'-png',str(path),str(HERE/tag)],check=True,capture_output=True)
a=np.array(Image.open(HERE/'before.png').convert('RGB'));b=np.array(Image.open(HERE/'after.png').convert('RGB'))
assert a.shape==b.shape
changed=np.any(a!=b,axis=2);allowed=np.zeros(changed.shape,bool)
box=[173,143,270,152]
x0,y0,x1,y1=box
allowed[math.floor((height-y1)*dpi/72):math.ceil((height-y0)*dpi/72),math.floor(x0*dpi/72):math.ceil(x1*dpi/72)]=True
outside=int((changed&~allowed).sum());assert outside==0,outside
Image.open(HERE/'after.png').save(OUT.with_suffix('.png'))
fonts=shutil.which('pdffonts') or '/Users/teyian/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdffonts'
font_report=subprocess.run([fonts,str(OUT)],check=True,text=True,capture_output=True).stdout
assert all(' no ' not in line for line in font_report.splitlines()[2:]),font_report
(HERE/'font_embedding.txt').write_text(font_report)
report={'source':str(SRC),'source_sha256':source_sha,'output':str(OUT),'output_sha256':sha(OUT),'png_sha256':sha(OUT.with_suffix('.png')),'change':{'before':old,'after':new},'existing_embedded_font_reused':str(font['/BaseFont']),'font_size_pt':size,'font_resources_added':False,'old_left_pt':old_left,'new_left_pt':new_left,'rendered_center_x_preserved':old_left+old_width/2,'baseline_y_preserved':145.2056446667,'annotation_value_unchanged':'+0.0264','only_one_text_block_changed':True,'page_commands_and_other_forms_byte_identical':True,'all_other_extracted_text_identical':True,'all_fonts_embedded':True,'pixel_comparison_dpi':dpi,'changed_pixels':int(changed.sum()),'allowed_annotation_box_pt':box,'outside_annotation_changed_pixels':outside,'scientific_data_and_all_bars_error_lines_colors_unchanged':True,'source_unchanged':sha(SRC)==source_sha,'visual_review':'pending'}
(HERE/'audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps(report,indent=2,ensure_ascii=False))
