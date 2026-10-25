#!/usr/bin/env python3
"""Remove unsupported significance encodings from original Figure 4 only.

The source PDF is immutable. Existing paths, fonts, colors, axes, heatmaps,
bar values, and panel placements are retained. This script modifies the
original vector content stream rather than redrawing the figure.
Requires pypdf, Pillow, numpy, and pdftoppm for the pixel audit.
"""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import tempfile

import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ArrayObject, ByteStringObject, ContentStream, FloatObject, NameObject

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'figure_drafts/Figure4_prediction_transitions_v2_accepted.pdf'
OUTPUT = SOURCE.with_name(SOURCE.stem + '_minimal.pdf')
PNG = OUTPUT.with_suffix('.png')
AUDIT = OUTPUT.with_suffix('.audit.json')
SOURCE_SHA = '835c6840ba23c8f95c0275e4df110c5ee6af8b369c48f19c996e5d93ff17ed64'
OLD_TITLE = 'F  Difficulty-matched contrasts retain topology differences'
NEW_TITLE = 'F  Error-stratified contrasts retain topology differences'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(SOURCE) == SOURCE_SHA, 'Unexpected source figure; stop rather than patch a different version.'
    writer = PdfWriter(clone_from=SOURCE)
    page = writer.pages[0]
    contents = ContentStream(page.get_contents(), writer)
    ops = contents.operations
    original_op_count = len(ops)
    edited = set()
    circles = []
    # The 18 individual PathCollection circles in E use /A3 or /A4 only.
    for i, (args, operator) in enumerate(ops):
        if operator != b'gs' or str(args[0]) not in ('/A3', '/A4'):
            continue
        start = i + 6
        assert ops[start][1] == b'm'
        assert all(ops[j][1] == b'c' for j in range(start + 1, start + 9))
        assert ops[start + 9][1] == b'h' and ops[start + 10][1] == b'B'
        coordinates = np.array([float(v) for j in range(start, start + 9) for v in ops[j][0]]).reshape(-1, 2)
        lo, hi = coordinates.min(axis=0), coordinates.max(axis=0)
        center = (lo + hi) / 2
        radius = (hi - lo) / 2
        assert np.allclose(radius[0], radius[1], atol=2e-6)
        assert 490 < center[0] < 630 and 45 < center[1] < 205
        target_radius = 4.0  # identical symbol area for all rows: 64 pt^2
        for j in range(start, start + 9):
            values = np.array(list(map(float, ops[j][0]))).reshape(-1, 2)
            values = center + (values - center) * target_radius / radius
            ops[j] = ([FloatObject(float(v)) for v in values.ravel()], ops[j][1])
            edited.add(j)
        ops[i] = ([NameObject('/A3')], operator)  # fixed alpha 0.85 for every dot
        edited.add(i)
        circles.append({'center_pdf_points': center.tolist(), 'old_radius_pdf_points': radius.tolist(),
                        'new_radius_pdf_points': target_radius, 'old_alpha_resource': str(args[0]),
                        'new_alpha': 0.85})
    assert len(circles) == 18
    removals = set()
    stars = []
    titles = 0
    for i, (args, operator) in enumerate(ops):
        if operator != b'TJ':
            continue
        text = ''.join(str(v) for v in args[0] if isinstance(v, str))
        if text == '*':
            assert [ops[j][1] for j in range(i - 5, i + 3)] == [b'q', b'cm', b'BT', b'Tf', b'Td', b'TJ', b'ET', b'Q']
            transform = list(map(float, ops[i - 4][0]))
            assert 990 < transform[4] < 1005 and 50 < transform[5] < 200
            removals.update(range(i - 5, i + 3))
            stars.append(transform[4:])
        elif text == OLD_TITLE:
            ops[i] = ([ArrayObject([ByteStringObject(NEW_TITLE.encode('utf-16be'))])], operator)
            edited.add(i)
            titles += 1
    assert len(stars) == 7 and titles == 1
    contents.operations = [op for i, op in enumerate(ops) if i not in removals]
    page.replace_contents(contents)
    writer.add_metadata({'/Title': 'Figure 4: original layout, descriptive statistical encodings',
                         '/Subject': 'Uniform panel E dots; panel F stars removed; Error-stratified title.'})
    with OUTPUT.open('wb') as stream:
        writer.write(stream)
    modified_reader = PdfReader(OUTPUT)
    output_text = modified_reader.pages[0].extract_text()
    assert '*' not in output_text and OLD_TITLE not in output_text and NEW_TITLE in output_text
    assert np.allclose(list(map(float, page.mediabox)), list(map(float, modified_reader.pages[0].mediabox)), atol=1e-5, rtol=0)
    # Render both versions identically. Any changed pixel must be inside the
    # panel-E plotting area, panel-F star strip, or panel-F title line.
    dpi = 300
    with tempfile.TemporaryDirectory(prefix='figure4-vector-audit-') as tmp:
        tmp = Path(tmp)
        for name, pdf in [('before', SOURCE), ('after', OUTPUT)]:
            subprocess.run(['pdftoppm', '-singlefile', '-r', str(dpi), '-png', str(pdf), str(tmp/name)], check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        before = np.asarray(Image.open(tmp/'before.png').convert('RGB'))
        after = np.asarray(Image.open(tmp/'after.png').convert('RGB'))
        assert before.shape == after.shape
        changed = np.any(before != after, axis=2)
        allowed = np.zeros(changed.shape, dtype=bool)
        width, height = map(float, [page.mediabox.width, page.mediabox.height])
        regions = {'panel_E_dots': [495, 47, 625, 206],
                   'panel_F_removed_stars': [994, 51, 1008, 202],
                   'panel_F_title': [786, 212, width, 228]}
        for x0, y0, x1, y1 in regions.values():
            c0, c1 = math.floor(x0*dpi/72)-2, math.ceil(x1*dpi/72)+2
            r0, r1 = math.floor((height-y1)*dpi/72)-2, math.ceil((height-y0)*dpi/72)+2
            allowed[max(0,r0):min(allowed.shape[0],r1),max(0,c0):min(allowed.shape[1],c1)] = True
        outside = int((changed & ~allowed).sum())
        assert outside == 0, f'{outside} pixels changed outside explicitly edited regions'
        PNG.write_bytes((tmp/'after.png').read_bytes())
        pixel_audit = {'dpi': dpi, 'image_dimensions': [after.shape[1], after.shape[0]],
                       'total_changed_pixels': int(changed.sum()), 'outside_allowed_regions_changed_pixels': outside,
                       'allowed_regions_pdf_points': regions, 'panel_A_B_C_D_all_pixels_unchanged': True,
                       'panel_F_bar_values_colors_axes_unchanged': True}
    report = {'source': SOURCE.name, 'source_sha256': SOURCE_SHA, 'output': OUTPUT.name,
              'output_sha256': sha(OUTPUT), 'output_png_sha256': sha(PNG),
              'method': 'Surgical vector-operation patch; no raster overlay and no figure redraw.',
              'original_operation_count': original_op_count,
              'edited_operation_count': len(edited), 'removed_operation_count': len(removals),
              'panel_E_uniform_symbols': circles, 'panel_F_removed_stars': stars,
              'title_before': OLD_TITLE, 'title_after': NEW_TITLE,
              'hidden_star_or_old_title_text_remaining': False,
              'same_canvas': True, 'pixel_audit': pixel_audit,
              'source_unchanged': sha(SOURCE) == SOURCE_SHA}
    AUDIT.write_text(json.dumps(report, indent=2, ensure_ascii=False)+'\n')
    print(json.dumps({'output': str(OUTPUT), 'audit': str(AUDIT), 'pixel_audit': pixel_audit}, ensure_ascii=False))


if __name__ == '__main__':
    main()
