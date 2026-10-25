#!/usr/bin/env python3
"""Replace only original Figure 1E with the same historical curve and wider y limits.

The accepted PDF is the visual authority. Other panels are retained as their
original vector operations; no later revision's layout or data are used.
"""
from pathlib import Path
from io import BytesIO
import hashlib
import json
import math
import subprocess
import tempfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, DecodedStreamObject, NameObject, RectangleObject, DictionaryObject

BASE = Path(__file__).resolve().parents[1]
SOURCE = BASE / 'figure_drafts/Figure1_connectivity_topology_controls_v2_accepted.pdf'
OUTPUT = SOURCE.with_name(SOURCE.stem + '_minimal.pdf')
PNG = OUTPUT.with_suffix('.png')
AUDIT = OUTPUT.with_suffix('.audit.json')
DATA = BASE / 'figure_inputs/data/processed/esm_gcn_partial_edge_ablation_exp44/edge_ablation_curve_summary.csv'
SOURCE_SHA = 'a156c40ca2d0cab500121cb1e96d44758c63ab02f037b9423f4cf4fdf780bcb4'
AXIS = [374.490922, 34.66875, 211.523156, 180.502326]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    assert sha(SOURCE) == SOURCE_SHA
    writer = PdfWriter(clone_from=SOURCE)
    page = writer.pages[0]
    width, height = map(float, [page.mediabox.width, page.mediabox.height])
    contents = ContentStream(page.get_contents(), writer)
    ops = contents.operations
    # Accepted Matplotlib vector output draws each whole axes contiguously.
    # Locate the E/F white axes background paths, retaining all surrounding ops.
    def axes_start(x):
        for i, (args, op) in enumerate(ops):
            if op == b'm' and len(args) == 2 and abs(float(args[0])-x) < 1e-5 and abs(float(args[1])-AXIS[1]) < 1e-5:
                if [ops[j][1] for j in range(i-6, i)] == [b'gs', b'g', b'j', b'w', b'G', b'g'] and ops[i+5][1] == b'f':
                    return i-6
        raise AssertionError('Accepted axes boundary not found')
    start, end = axes_start(AXIS[0]), axes_start(691.526805)
    assert (start, end) == (1686, 2393)
    removed = ops[start:end]
    assert sum(o == b'q' for _, o in removed) == sum(o == b'Q' for _, o in removed)
    # Preserve the untouched raw bytes, not a re-serialization of all float
    # operands (which can move an unrelated glyph by one antialiasing pixel).
    original_bytes = page.get_contents().get_data()
    marker = b'/A1 gs 1 g 0 j 0 w 0 G 1 g\n\n'
    byte_start = original_bytes.index(marker + b'374.490922 34.66875 m')
    byte_end = original_bytes.index(marker + b'691.526805 34.66875 m')
    retained_bytes = original_bytes[:byte_start] + original_bytes[byte_end:]

    # Recreate this one axes at its exact accepted PDF coordinates and point
    # sizes. Only ylim changes; the same CSV, colors, line styles and markers
    # are retained. The enclosing PDF canvas is unchanged.
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.2,
        'axes.titlesize': 9.0, 'axes.labelsize': 8.6,
        'xtick.labelsize': 8.1, 'ytick.labelsize': 8.1,
        'legend.fontsize': 7.4, 'axes.linewidth': .8,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })
    df = pd.read_csv(DATA)
    fig = plt.figure(figsize=(width/72, height/72))
    x, y, w, h = AXIS
    ax = fig.add_axes([x/width, y/height, w/width, h/height])
    ax.set_title('E  Partial train-test edge ablation', loc='left', fontweight='bold')
    styles = {
        'random_pair_removal': ('Random removal', '#636363', 'o'),
        'high_confidence_first': ('High-confidence first', '#DE2D26', 's'),
        'low_confidence_first': ('Low-confidence first', '#3182BD', '^'),
    }
    for strategy, (label, color, marker) in styles.items():
        sub = df[df.strategy == strategy].sort_values('fraction_removed')
        ax.errorbar(sub.fraction_removed*100, sub.delta_test_auc_vs_raw_esm,
                    yerr=sub.test_auc_sd, color=color, ecolor=color,
                    marker=marker, markersize=4.1, lw=1.6, capsize=2.4, label=label)
    ax.axhline(0, color='#4D4D4D', lw=.85, ls='--')
    ax.set_xlabel('Removed direct train-test PPI pairs (%)')
    ax.set_ylabel('Delta AUC vs ESM raw')
    ax.set_xlim(-3, 103)
    ax.set_ylim(-.015, .030)
    ax.set_yticks(np.arange(-.015, .030001, .005))
    ax.legend(frameon=False, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    buffer = BytesIO()
    fig.savefig(buffer, format='pdf', transparent=True)
    plt.close(fig)
    buffer.seek(0)
    # Add the replacement as a separate Form with its own font resources,
    # retaining the original page content and font dictionaries verbatim.
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
    resources['/XObject'][NameObject('/PanelEReplacement')] = writer._add_object(form)
    replaced = DecodedStreamObject()
    replaced.set_data(retained_bytes + b'\nq\n/PanelEReplacement Do\nQ\n')
    page.replace_contents(replaced)
    writer.add_metadata({'/Title': 'Figure 1: accepted layout with unclipped panel E intervals',
                         '/Subject': 'Panel E y limits expanded to -0.015 and 0.030; historical values unchanged.'})
    with OUTPUT.open('wb') as f:
        writer.write(f)
    values = df.delta_test_auc_vs_raw_esm
    lows, highs = values-df.test_auc_sd, values+df.test_auc_sd
    assert lows.min() > -.015 and highs.max() < .030
    dpi = 300
    with tempfile.TemporaryDirectory(prefix='figure1-local-axis-') as tmp:
        tmp = Path(tmp)
        for name, p in [('before', SOURCE), ('after', OUTPUT)]:
            subprocess.run(['pdftoppm', '-singlefile', '-r', str(dpi), '-png', str(p), str(tmp/name)],
                           check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        before = np.asarray(Image.open(tmp/'before.png').convert('RGB'))
        after = np.asarray(Image.open(tmp/'after.png').convert('RGB'))
        assert before.shape == after.shape
        changed = np.any(before != after, axis=2)
        allowed = np.zeros(changed.shape, dtype=bool)
        region = [315, 0, 625, 237]  # E only, with its title/tick/axis labels
        x0, y0, x1, y1 = region
        c0, c1 = math.floor(x0*dpi/72), math.ceil(x1*dpi/72)
        r0, r1 = math.floor((height-y1)*dpi/72), math.ceil((height-y0)*dpi/72)
        allowed[max(0,r0):min(allowed.shape[0],r1), max(0,c0):min(allowed.shape[1],c1)] = True
        outside = int((changed & ~allowed).sum())
        assert outside == 0, f'{outside} changed pixels outside E'
        PNG.write_bytes((tmp/'after.png').read_bytes())
    report = {
        'source_sha256': SOURCE_SHA, 'output_sha256': sha(OUTPUT), 'png_sha256': sha(PNG),
        'data_file': str(DATA.relative_to(BASE)), 'data_sha256': sha(DATA),
        'method': 'Remove only original axes-E vector operations; add an exact-coordinate vector axes-E with expanded ylim.',
        'original_pdf_canvas_pt': [width, height], 'axes_E_bbox_pt': AXIS,
        'ylim_before': [-.010, .026], 'ylim_after': [-.015, .030],
        'historical_curve_values_and_SD_unchanged': True,
        'full_errorbar_extent': [float(lows.min()), float(highs.max())],
        'originally_clipped_points': df.loc[(lows < -.010) | (highs > .026),
            ['strategy', 'fraction_removed', 'delta_test_auc_vs_raw_esm', 'test_auc_sd']].to_dict('records'),
        'removed_axes_operation_range': [start, end],
        'removed_axes_raw_byte_range': [byte_start, byte_end],
        'all_untouched_original_vector_bytes_preserved': True,
        'pixel_audit': {'dpi': dpi, 'changed_pixels': int(changed.sum()),
                        'allowed_region_pdf_points': region, 'outside_panel_E_changed_pixels': outside,
                        'panels_A_B_C_D_F_and_overall_title_pixel_identical': True},
        'unavoidable_local_differences': 'Expanded y-axis range changes the E tick labels and vertical mapping of the same curve and SD bars; no other content changes.',
        'source_unchanged': sha(SOURCE) == SOURCE_SHA,
    }
    AUDIT.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(OUTPUT), 'audit': str(AUDIT), 'outside_changed_pixels': outside}))


if __name__ == '__main__':
    main()
