#!/usr/bin/env python3
"""Replace only specified Figure 1/3 text blocks, retaining all other vector bytes.

Accepted/minimal PDFs remain immutable. Old text operators are removed, not
covered. Replacement text uses the original DejaVu Sans face and point sizes
in a separate transparent vector Form; this also embeds newly needed glyphs.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import math
import re
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, DecodedStreamObject, NameObject, RectangleObject, DictionaryObject

BASE = Path(__file__).resolve().parents[1]
FONTS = {
    'normal': BASE / 'resources/fonts/DejaVuSans.ttf',
    'bold': BASE / 'resources/fonts/DejaVuSans-Bold.ttf',
}
FIGURES = [
    {
        'source': 'Figure1_connectivity_topology_controls_v2_accepted_minimal.pdf',
        'source_sha256': 'd66058d280efc1e3704bea1b02faaa8eff1140957e866f67c66e7bb35f897c5d',
        'remove': [
            'PPI graph augmentation depends on topology alignment and train-test connectivity',
            'AUC gain', 'removed', 'returns to', 'ESM level',
        ],
        'insert': [
            {'text': 'Fixed-propagation audit under connectivity and topology controls',
             'x': 'center', 'y': 553.228125, 'size': 13.5, 'weight': 'bold', 'ha': 'center', 'color': 'black'},
            {'text': 'AUC near the', 'x': 221.5, 'y': 153.0337693798, 'size': 7.4, 'ha': 'center', 'color': '#4D4D4D'},
            {'text': 'raw-ESM', 'x': 221.5, 'y': 145.2056443798, 'size': 7.4, 'ha': 'center', 'color': '#4D4D4D'},
            {'text': 'reference', 'x': 221.5, 'y': 137.3775193798, 'size': 7.4, 'ha': 'center', 'color': '#4D4D4D'},
            {'text': 'Point estimates', 'x': 867.5, 'y': 153.0337693798, 'size': 7.4, 'ha': 'center', 'color': '#4D4D4D'},
            {'text': 'near raw ESM', 'x': 867.5, 'y': 145.2056443798, 'size': 7.4, 'ha': 'center', 'color': '#4D4D4D'},
        ],
        'allowed_regions_pdf_points': {
            'overall_title': [143, 547, 790, 568],
            'panel_D_annotation': [196, 134, 248, 162],
            'panel_F_annotation': [836, 142, 900, 162],
        },
        'scope': 'Overall title and annotations in D/F only; panel E repair preserved.',
    },
    {
        'source': 'Figure3_topology_context_v1_accepted.pdf',
        'source_sha256': '71a3aab296cc4bc6f09f4f39421915cad84892b2cf73c82f22083a61abe6f8fe',
        'remove': [
            'PPI-GCN gains concentrate in specific network topology contexts',
            'C  Bridge role and local density shape gain',
        ],
        'insert': [
            {'text': 'Exploratory PPI-GCN gains across network topology contexts',
             'x': 'center', 'y': 505.564125, 'size': 14, 'weight': 'bold', 'ha': 'center', 'color': 'black'},
            {'text': 'C  Gains across betweenness and clustering strata',
             'x': 688, 'y': 464.74875, 'size': 9, 'weight': 'bold', 'ha': 'left', 'color': 'black'},
        ],
        'allowed_regions_pdf_points': {
            'overall_title': [219, 499, 741, 521],
            'panel_C_title': [685, 461, 950, 476],
        },
        'scope': 'Overall title and C title only; all topology values and plots retained.',
    },
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def replace_figure(spec):
    source = BASE / 'figure_drafts' / spec['source']
    assert sha(source) == spec['source_sha256'], 'Unexpected source figure'
    output = source.with_name(source.stem + '_clarified.pdf')
    writer = PdfWriter(clone_from=source)
    page = writer.pages[0]
    width, height = map(float, (page.mediabox.width, page.mediabox.height))
    raw = page.get_contents().get_data()
    block_re = re.compile(rb'(?m)^q\n[^\n]+ cm\nBT\n.*?\nET\nQ\b', re.S)
    selected = []
    for match in block_re.finditer(raw):
        block = DecodedStreamObject()
        block.set_data(match.group())
        ops = ContentStream(block, writer).operations
        text = ''.join(str(item) for args, op in ops if op == b'TJ' for item in args[0] if isinstance(item, str))
        if text in spec['remove']:
            selected.append({'start': match.start(), 'end': match.end(), 'text': text})
    assert sorted(x['text'] for x in selected) == sorted(spec['remove']), selected
    # Preserve byte-exact content outside the removed, self-contained text blocks.
    retained = raw
    for block in reversed(selected):
        retained = retained[:block['start']] + retained[block['end']:]

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig = plt.figure(figsize=(width/72, height/72))
    for item in spec['insert']:
        x = width/2 if item['x'] == 'center' else item['x']
        fig.text(x/width, item['y']/height, item['text'],
                 fontproperties=FontProperties(fname=FONTS[item.get('weight', 'normal')], size=item['size']),
                 ha=item['ha'], va='baseline', color=item['color'])
    buffer = BytesIO()
    fig.savefig(buffer, format='pdf', transparent=True)
    plt.close(fig)
    buffer.seek(0)
    overlay = PdfReader(buffer).pages[0]
    form = DecodedStreamObject()
    form.set_data(overlay.get_contents().get_data())
    form.update({NameObject('/Type'): NameObject('/XObject'),
                 NameObject('/Subtype'): NameObject('/Form'),
                 NameObject('/BBox'): RectangleObject([0, 0, width, height]),
                 NameObject('/Resources'): overlay['/Resources'].clone(writer)})
    resources = page['/Resources']
    if '/XObject' not in resources:
        resources[NameObject('/XObject')] = DictionaryObject()
    resources['/XObject'][NameObject('/TextClarityReplacement')] = writer._add_object(form)
    patched = DecodedStreamObject()
    patched.set_data(retained + b'\nq\n/TextClarityReplacement Do\nQ\n')
    page.replace_contents(patched)
    writer.add_metadata({'/Title': spec['insert'][0]['text'], '/Subject': spec['scope']})
    with output.open('wb') as out:
        writer.write(out)

    readback = PdfReader(output)
    out_text = readback.pages[0].extract_text()
    for old in spec['remove']:
        assert old not in out_text, f'Old text remains: {old}'
    for new in spec['insert']:
        assert new['text'] in out_text, f'New text not extractable: {new["text"]}'
    assert list(readback.pages[0].mediabox) == list(page.mediabox)
    # Old blocks are physically absent, not covered by an opaque patch.
    for block in selected:
        assert raw[block['start']:block['end']] not in readback.pages[0].get_contents().get_data()

    dpi = 300
    with tempfile.TemporaryDirectory(prefix='figure-text-clarity-') as tmp:
        tmp = Path(tmp)
        for key, pdf in [('before', source), ('after', output)]:
            subprocess.run(['pdftoppm', '-singlefile', '-r', str(dpi), '-png', str(pdf), str(tmp/key)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        before = np.asarray(Image.open(tmp/'before.png').convert('RGB'))
        after = np.asarray(Image.open(tmp/'after.png').convert('RGB'))
        assert before.shape == after.shape
        changed = np.any(before != after, axis=2)
        allowed = np.zeros(changed.shape, dtype=bool)
        for x0, y0, x1, y1 in spec['allowed_regions_pdf_points'].values():
            c0, c1 = math.floor(x0*dpi/72), math.ceil(x1*dpi/72)
            r0, r1 = math.floor((height-y1)*dpi/72), math.ceil((height-y0)*dpi/72)
            allowed[max(0,r0):min(allowed.shape[0],r1), max(0,c0):min(allowed.shape[1],c1)] = True
        outside = int((changed & ~allowed).sum())
        assert outside == 0, f'{outside} changed pixels outside allowed text regions'
        output.with_suffix('.png').write_bytes((tmp/'after.png').read_bytes())
    audit = {
        'source': source.name, 'source_sha256': spec['source_sha256'],
        'output': output.name, 'output_sha256': sha(output),
        'output_png_sha256': sha(output.with_suffix('.png')),
        'method': 'Remove original text operators and insert new transparent vector text Form; retain all other original content bytes.',
        'removed_text_blocks': selected, 'inserted_text': spec['insert'],
        'old_text_physically_removed_and_not_extractable': True,
        'new_text_extractable': True, 'new_glyphs_embedded': True,
        'portable_font_files': {weight: {'path': str(path.relative_to(BASE)), 'sha256': sha(path)} for weight, path in FONTS.items()},
        'all_font_point_sizes_preserved': True,
        'same_canvas': True, 'canvas_pt': [width, height],
        'data_axes_colors_and_non_text_geometry_unchanged': True,
        'pixel_audit': {'dpi': dpi, 'dimensions': [after.shape[1], after.shape[0]],
                        'changed_pixels': int(changed.sum()),
                        'outside_allowed_text_regions_changed_pixels': outside,
                        'allowed_text_regions_pdf_points': spec['allowed_regions_pdf_points']},
        'source_unchanged': sha(source) == spec['source_sha256'],
        'abbreviations_needed': False,
        'position_changes': 'Main titles recentered; Figure1 D/F annotations wrap at original point size; Figure3 C title moved left within existing title-row whitespace.'
    }
    output.with_suffix('.audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(output), 'pixel_audit': audit['pixel_audit']}, ensure_ascii=False))


if __name__ == '__main__':
    for specification in FIGURES:
        replace_figure(specification)
