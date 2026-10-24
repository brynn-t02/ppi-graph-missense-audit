from pathlib import Path
import re
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Preformatted, KeepTogether
from reportlab.lib.pagesizes import A4

import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root',default=str(Path(__file__).resolve().parents[1]));parser.add_argument('--anonymous',action='store_true');args=parser.parse_args()
B=Path(args.root).resolve()
fonts=B/'source/resources/fonts'
for name,file in [('DejaVu','DejaVuSerif.ttf'),('DejaVu-Bold','DejaVuSerif-Bold.ttf'),('DejaVu-Italic','DejaVuSerif-Italic.ttf'),('DejaVuMono','DejaVuSansMono.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
pdfmetrics.registerFontFamily('DejaVu',normal='DejaVu',bold='DejaVu-Bold',italic='DejaVu-Italic',boldItalic='DejaVu-Bold')
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyRevision',fontName='DejaVu',fontSize=9,leading=13,spaceAfter=7,splitLongWords=True))
styles.add(ParagraphStyle(name='TitleRevision',fontName='DejaVu-Bold',fontSize=18,leading=23,spaceAfter=12))
styles.add(ParagraphStyle(name='H2Revision',fontName='DejaVu-Bold',fontSize=12,leading=17,spaceBefore=10,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='H3Revision',fontName='DejaVu-Bold',fontSize=10,leading=14,spaceBefore=9,spaceAfter=5,keepWithNext=True))
styles.add(ParagraphStyle(name='H4Revision',fontName='DejaVu-Bold',fontSize=9,leading=13,spaceBefore=7,spaceAfter=4,keepWithNext=True))
styles.add(ParagraphStyle(name='TableRevision',fontName='DejaVu',fontSize=7.3,leading=9.6,splitLongWords=True))
styles.add(ParagraphStyle(name='EquationRevision',fontName='DejaVu',fontSize=10,leading=16,alignment=TA_CENTER,spaceAfter=8))
styles.add(ParagraphStyle(name='CodeRevision',fontName='DejaVuMono',fontSize=7.4,leading=10,spaceAfter=7))

def fmt(s):
    s=escape(s)
    s=re.sub(r'`([^`]+)`',r'<font name="DejaVuMono">\1</font>',s)
    s=re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',s)
    s=s.replace('$','').replace('\\Delta','Δ').replace('\\times','×').replace('\\pm','±')
    return s

def footer(canvas,doc):
    canvas.saveState();canvas.setFont('DejaVu',7.2)
    canvas.setFillColor(colors.HexColor('#555555'))
    canvas.drawString(42,26,'Supplementary Information | Matched revision 24 October 2026')
    canvas.drawRightString(A4[0]-42,26,str(doc.page));canvas.restoreState()

lines=(B/'source'/('Supplementary_Information_anonymous.md' if args.anonymous else 'Supplementary_Information_revised.md')).read_text().splitlines()
story=[];i=0;width=A4[0]-84
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            parts=[x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\-\s]+',x) for x in parts): rows.append(parts)
            i+=1
        nc=len(rows[0]);assert all(len(row)==nc for row in rows)
        if rows[0][0]=='Split' and nc==5: weights=[.8,2.4,.4,1.25,1.25]
        elif rows[0][0]=='Split' and nc==4: weights=[.8,2.6,1.5,1.5]
        elif rows[0][0]=='Analysis': weights=[1.3,2.4,.65,2.0]
        elif nc==7 and rows[0][0]=='Threshold':weights=[.7,1,1,1,1,1,1]
        elif nc==7 and rows[0][0]=='Split':weights=[.55,1.85,.8,.8,1.2,1.2,.9]
        elif nc==6 and rows[0][0]=='Item':weights=[1.35,.9,.9,.9,1.7,1.6]
        elif rows[0][0]=='Model' and nc==6:weights=[2,.5,1.05,1.05,1.05,1.05]
        elif rows[0][0]=='Model' and nc==7:weights=[1.7,.4,.8,.8,1.25,1.25,.8]
        else:weights=[max(1,min(2.2,max(len(r[j]) for r in rows)/12)) for j in range(nc)]
        cols=[width*w/sum(weights) for w in weights]
        data=[[Paragraph(('<b>'+fmt(v)+'</b>') if ri==0 else fmt(v),styles['TableRevision']) for v in row] for ri,row in enumerate(rows)]
        table=Table(data,colWidths=cols,repeatRows=1,hAlign='LEFT')
        table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e6edf3')),
            ('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#64748b')),
            ('LINEBELOW',(0,1),(-1,-1),.25,colors.HexColor('#d8dfe5')),
            ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),
            ('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        if (rows[0][0]=='Split' and nc in (4,5)) or (rows[0][0]=='Arm' and nc==6):
            table.setStyle(TableStyle([('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        story.append(table);story.append(Spacer(1,8));continue
    if line.startswith('```'):
        code=[];i+=1
        while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
        story.append(Preformatted('\n'.join(code),styles['CodeRevision'],maxLineLength=87));i+=1;continue
    if line=='$$':
        eq=[];i+=1
        while i<len(lines) and lines[i].strip()!='$$':eq.append(lines[i]);i+=1
        raw=''.join(eq)
        if raw.startswith('\\hat{A}'):
            text='A&#770; = D<super>-1/2</super> A D<super>-1/2</super>'
        else:text='H = (X + A&#770;X + A&#770;<super>2</super>X) / 3'
        story.append(Paragraph(text,styles['EquationRevision']));i+=1;continue
    if line.startswith('#### '):style='H4Revision';text=line[5:]
    elif line.startswith('### '):style='H3Revision';text=line[4:]
    elif line.startswith('## '):style='H2Revision';text=line[3:]
    elif line.startswith('# '):style='TitleRevision';text=line[2:]
    else:
        style='BodyRevision';para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not lines[i].startswith(('#','|','```','$$')):para.append(lines[i].strip());i+=1
        story.append(Paragraph(fmt(' '.join(para)),styles[style]));continue
    story.append(Paragraph(fmt(text),styles[style]));i+=1

path=B/('BIBM2026_supplementary_anonymous_revised.pdf' if args.anonymous else 'BIBM2026_supplementary_named_revised.pdf')
doc=SimpleDocTemplate(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=42,bottomMargin=43,
    title='Supplementary Information - PPI Graph Augmentation Audit',author=('Anonymous' if args.anonymous else 'Yuxin Tian, Xianguo Zhang, Weibo Meng, Jieqi Yin, Yongqin Jin'),allowSplitting=True)
doc.build(story,onFirstPage=footer,onLaterPages=footer)
print(path)
