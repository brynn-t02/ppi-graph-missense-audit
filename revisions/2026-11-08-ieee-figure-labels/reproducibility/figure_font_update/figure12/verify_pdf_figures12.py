from pathlib import Path
import json,hashlib,subprocess,math
import pdfplumber
from pypdf import PdfReader
from pypdf.generic import ContentStream
W=Path(__file__).resolve().parent
O=Path('outputs/BIBM_workshop_IEEE_figure_labels_20261108/source/figure_drafts')
fonttool='/Users/teyian/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdffonts'
result={'visual_review':'Both final figures rendered with Poppler and reviewed; no title/legend/axis-label overlap, no missing text or new graphic clipping. Figure 1E retains its original y-axis limits and original error-bar clipping at those limits.','layout_changes':['Times New Roman regular/bold; all actual PDF text font sizes >=8 pt at fixed paper width 503.79 pt.','Canvas fixed 503.79 x 345 pt, without tight cropping.','Long titles/labels wrapped without changing wording; legends moved into clear header bands.','Figure 1A same 10 vertices, 12 edges, colors, split semantics and three-step flow, with readable label placement.','Figure 2A same five boxes, coverage numbers and directed relationships, reflowed vertically to fit 8 pt text.','Figure 1C x ticks rotated; Figure 2B model labels rotated for separation.','Figure 2C x ticks sparsified from 0.01 to 0.02 step; Figure 2E x ticks sparsified from 0.025 to 0.05 step. Axis quantities and ranges unchanged.'],'no_model_display_names_shortened':True,'no_font_files_copied':True,'figures':{}}
for i in [1,2]:
 p=next(O.glob(f'Figure{i}*.pdf'));r=PdfReader(p);pg=r.pages[0]
 sizes=[float(a[1]) for a,op in ContentStream(pg.get_contents(),r).operations if op==b'Tf']
 used=[]
 def visitor(s,cm,tm,f,size):
  if s.strip():used.append({'text':s,'font_size':size,'effective_cm_scale':math.hypot(cm[0],cm[1])})
 pg.extract_text(visitor_text=visitor)
 assert min(sizes)>=8
 assert all(abs(q['effective_cm_scale']-1)<1e-5 for q in used)
 with pdfplumber.open(p) as d:
  page=d.pages[0]
  outside=[c['text'] for c in page.chars if c['text'].strip() and (c['x0']<0 or c['x1']>page.width or c['top']<0 or c['bottom']>page.height)]
  assert not outside
  fontnames=sorted(set(c['fontname'] for c in page.chars));assert all('TimesNewRoman' in f for f in fontnames)
 ft=subprocess.check_output([fonttool,str(p)],text=True)
 (W/f'figure{i}_pdffonts.txt').write_text(ft)
 assert ' yes yes yes ' in ft and ' no ' not in ft
 result['figures'][str(i)]={'pdf':str(p.resolve()),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'page_points':[float(pg.mediabox.width),float(pg.mediabox.height)],'pdf_Tf_sizes':sorted(set(sizes)),'all_text_CTM_unit_scale':True,'text_outside_page':outside,'font_names':fontnames,'all_fonts_embedded':True,'visual_png':str((W/f'Figure{i}_after.png').resolve()),'plotted_data_audit':str((W/f'figure{i}_plotted_data.json').resolve())}
(W/'final_qa.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps(result,indent=2,ensure_ascii=False))
