#!/usr/bin/env python3
"""Replace only Figure 4E's heading in the existing vector figure.

The old text drawing block is deleted. The replacement uses the same font
family/style, 9 pt size, black color and baseline; no white cover is drawn.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import math
import subprocess
import tempfile

import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream
from reportlab.pdfgen.canvas import Canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'figure_drafts/Figure4_prediction_transitions_v2_accepted_minimal.pdf'
OUTPUT = SOURCE.with_name(SOURCE.stem + '_clarified.pdf')
FONT = BASE / 'resources/fonts/DejaVuSans-Bold.ttf'
OLD = 'E  Topology features separate corrected groups'
NEW = 'E  Mean topology differences between corrected groups'
EXPECTED_SHA = 'd8059c76d34c509ec143170796559fc26c781da141f96e860e193abaa4d79b25'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def text(args):
    return ''.join(v.decode('utf-16be') if isinstance(v, bytes) else v
                   for v in args[0] if isinstance(v, (str, bytes)))


def main():
    assert sha(SOURCE) == EXPECTED_SHA
    writer = PdfWriter(clone_from=SOURCE)
    page = writer.pages[0]
    cs = ContentStream(page.get_contents(), writer)
    hits = [i for i, (a, op) in enumerate(cs.operations) if op == b'TJ' and text(a) == OLD]
    assert len(hits) == 1
    i = hits[0]
    assert [op for a, op in cs.operations[i-5:i+3]] == [b'q', b'cm', b'BT', b'Tf', b'Td', b'TJ', b'ET', b'Q']
    x, y = map(float, cs.operations[i-4][0][4:])
    assert float(cs.operations[i-3+1][0][1]) == 9
    cs.operations = cs.operations[:i-5] + cs.operations[i+3:]
    page.replace_contents(cs)
    # The original embedded font subset lacks lowercase b. Re-embed the same
    # DejaVu Sans Bold font for the new heading rather than substituting glyphs.
    pdfmetrics.registerFont(TTFont('OriginalFigureDejaVuSansBold', str(FONT)))
    overlay = BytesIO()
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    canvas = Canvas(overlay, pagesize=(width, height), pageCompression=1)
    canvas.setFont('OriginalFigureDejaVuSansBold', 9)
    canvas.setFillColorRGB(0, 0, 0)
    canvas.drawString(x, y, NEW)
    canvas.save()
    overlay.seek(0)
    page.merge_page(PdfReader(overlay).pages[0])
    with OUTPUT.open('wb') as stream:
        writer.write(stream)
    out_text = PdfReader(OUTPUT).pages[0].extract_text()
    assert OLD not in out_text and NEW in out_text
    dpi = 300
    bounds = [447, 212, 740, 229]
    with tempfile.TemporaryDirectory(prefix='figure4-title-audit-') as td:
        td = Path(td)
        for name, path in [('before', SOURCE), ('after', OUTPUT)]:
            subprocess.run(['pdftoppm', '-singlefile', '-r', str(dpi), '-png', str(path), str(td/name)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        before = np.asarray(Image.open(td/'before.png').convert('RGB'))
        after = np.asarray(Image.open(td/'after.png').convert('RGB'))
        assert before.shape == after.shape
        changed = np.any(before != after, axis=2)
        allowed = np.zeros(changed.shape, dtype=bool)
        x0, y0, x1, y1 = bounds
        c0, c1 = math.floor(x0*dpi/72)-2, math.ceil(x1*dpi/72)+2
        r0, r1 = math.floor((height-y1)*dpi/72)-2, math.ceil((height-y0)*dpi/72)+2
        allowed[r0:r1,c0:c1] = True
        outside = int((changed & ~allowed).sum())
        assert outside == 0, f'Unexpected changes outside E heading: {outside}'
        OUTPUT.with_suffix('.png').write_bytes((td/'after.png').read_bytes())
    audit = {'source_sha256': EXPECTED_SHA, 'output_sha256': sha(OUTPUT),
             'output_png_sha256': sha(OUTPUT.with_suffix('.png')), 'font_sha256': sha(FONT),
             'old_heading': OLD, 'new_heading': NEW, 'font': 'DejaVu Sans Bold', 'font_size_pt': 9,
             'baseline': [x,y], 'title_right_edge_pt': x+pdfmetrics.stringWidth(NEW,'OriginalFigureDejaVuSansBold',9),
             'original_text_block_removed': True, 'white_cover_used': False,
             'dpi': dpi, 'allowed_change_region_pdf_points': bounds,
             'changed_pixels': int(changed.sum()), 'pixels_changed_outside_heading': outside,
             'all_other_panels_axes_points_colors_and_numbers_unchanged': True,
             'source_unchanged': sha(SOURCE) == EXPECTED_SHA}
    OUTPUT.with_suffix('.audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit,ensure_ascii=False))


if __name__ == '__main__':
    main()
