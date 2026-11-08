from pathlib import Path
from collections import Counter
import hashlib, json, math, re, subprocess
from pypdf import PdfReader
from pypdf.generic import ContentStream

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'outputs/BIBM_workshop_IEEE_figure_labels_20261108'
BASE = ROOT / 'outputs/BIBM_workshop_original_BioGRID_date_restored_20261108'
WORK = ROOT / 'work/ieee_figure_labels_20261108'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
main = OUT / 'BIBM2026_workshop_final_IEEE_figures.pdf'
supp = OUT / 'BIBM2026_supplementary_original_plus_BioGRID_date_restored.pdf'
r = PdfReader(main)
assert len(r.pages) == 8
assert r.pdf_header == '%PDF-1.5'
assert not r.is_encrypted
assert all(tuple(float(x) for x in p.mediabox) == (0, 0, 612, 792) for p in r.pages)
assert sha(supp) == sha(BASE / supp.name)
oldtex = (BASE / 'source/main.tex').read_text()
newtex = (OUT / 'source/main.tex').read_text()
assert oldtex.replace(r'\IEEEtriggeratref{25}', r'\IEEEtriggeratref{16}') == newtex
assert (BASE / 'source/IEEEtran.cls').read_bytes() == (OUT / 'source/IEEEtran.cls').read_bytes()
assert (BASE / 'source/Supplementary_Information_revised.md').read_bytes() == (OUT / 'source/Supplementary_Information_revised.md').read_bytes()
assert '31 March 2024' in newtex
assert not re.search(r'Overfull|Missing character|Undefined control|undefined references', (WORK / 'final_main.log').read_text(), re.I)

I = (1., 0., 0., 1., 0., 0.)
def mul(m, n):
    a,b,c,d,e,f = m; A,B,C,D,E,F = n
    return (a*A+c*B, b*A+d*B, a*C+c*D, b*C+d*D, a*E+c*F+e, b*E+d*F+f)

records = []
def walk(stream, resources, ctm=I, depth=0, page_num=None):
    assert depth < 20
    stack=[]; tm=I; fs=0.; font=None
    for args, op in ContentStream(stream, r).operations:
        if op == b'q': stack.append((ctm, tm, fs, font))
        elif op == b'Q': ctm,tm,fs,font=stack.pop()
        elif op == b'cm': ctm=mul(ctm, tuple(float(v) for v in args))
        elif op == b'BT': tm=I
        elif op == b'Tm': tm=tuple(float(v) for v in args)
        elif op == b'Tf':
            fs=float(args[1]); font=resources['/Font'][args[0]].get_object()
        elif op in (b'Tj', b'TJ', b"'", b'"') and font is not None:
            combined=mul(ctm,tm)
            effective=fs*math.hypot(combined[2],combined[3])
            name=str(font.get('/BaseFont',''))
            if 'TimesNewRoman' in name:
                records.append({'page':page_num,'font':name,'font_instruction_pt':fs,'effective_vertical_pt':effective})
        elif op == b'Do':
            x=resources['/XObject'][args[0]].get_object()
            if x.get('/Subtype') == '/Form':
                walk(x, x.get('/Resources', resources).get_object(), mul(ctm, tuple(float(v) for v in x.get('/Matrix',I))), depth+1, page_num)

for n,p in enumerate(r.pages,1): walk(p.get_contents(),p['/Resources'].get_object(),page_num=n)
assert records
minimum=min(x['effective_vertical_pt'] for x in records)
assert minimum >= 7.999, minimum
def outline_count(seq): return sum(outline_count(x) if isinstance(x,list) else 1 for x in seq)
ann=[]
for p in r.pages:
    aa=p.get('/Annots',[])
    if hasattr(aa,'get_object'): aa=aa.get_object()
    ann += [str(x.get_object().get('/Subtype')) for x in aa]
assert Counter(ann)==Counter({'/Link':29})
assert outline_count(r.outline)==15
fonts_bin='/Users/teyian/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdffonts'
fontlines=subprocess.check_output([fonts_bin,str(main)],text=True)
(WORK/'final_fonts.txt').write_text(fontlines)
assert all(line.split()[-5:-3] == ['yes','yes'] for line in fontlines.splitlines()[2:] if line.strip())
figures={p.name:{'sha256':sha(p),'width_pt':float(PdfReader(p).pages[0].mediabox.width),'height_pt':float(PdfReader(p).pages[0].mediabox.height)} for p in (OUT/'source/figure_drafts').glob('*.pdf')}
v={'status':'PASS','main_sha256':sha(main),'supplement_sha256':sha(supp),'main_pages':8,'supplement_pages':12,'paper_size_pt':[612,792],
   'all_manuscript_content_unchanged':True,'only_main_tex_change':'IEEEtriggeratref 25 -> 16; reference-column balancing only',
   'supplement_byte_identical':True,'figure_count':len(figures),'figures':figures,
   'effective_placed_figure_text_min_pt':minimum,'figure_text_font':'Embedded Times New Roman regular/bold','all_fonts_embedded_and_subset':True,
   'links_preserved':29,'bookmarks_preserved':15,'pdf_version_preserved':'1.5','latest_user_instruction':'Retain links/bookmarks and original PDF settings; only figure typography and necessary pagination changes.',
   'new_pdf_express_certification_completed':False,'prior_pdf_pass_reported_by_user':True,'no_training_or_data_reanalysis':True}
(WORK/'delivery_verification.json').write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(v,ensure_ascii=False,indent=2))
