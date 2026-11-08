"""Replace only the Figure 3B heading in the existing vector PDF."""
from pathlib import Path
import hashlib, json, math, subprocess, shutil
import numpy as np
from PIL import Image
from pypdf import PdfReader, PdfWriter
from pypdf.generic import ContentStream, DecodedStreamObject

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SRC = ROOT/'outputs/BIBM_workshop_targeted_candidate_v6_figure3_20261108/source/figure_drafts/Figure3_topology_context_matched_ensembles.pdf'
OUT = HERE/SRC.name
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
before_sha = sha(SRC)
old_title = 'B  Degree and k-core gains are non-monotonic'
new_title = 'B  Performance differences across degree and k-core strata'
writer = PdfWriter(clone_from=SRC)
page = writer.pages[0]
stream = ContentStream(page.get_contents(), writer)
hits = []
for i, (args, op) in enumerate(stream.operations):
    if op == b'TJ':
        text = ''.join(v.decode('utf-16be') if isinstance(v, bytes) else v for v in args[0] if isinstance(v, (str, bytes)))
        if text == old_title:
            hits.append(i)
assert len(hits) == 1
i = hits[0]
assert stream.operations[i-2] == (['/F2', 9], b'Tf')
assert stream.operations[i-4][1] == b'cm'
assert abs(float(stream.operations[i-4][0][4])-387.174) < 1e-8
# Same embedded bold font and 9pt size. A 12.174pt leftward title adjustment
# leaves a 9.26pt gap before C's title; axes and all other content are untouched.
raw = page.get_contents().get_data()
old_block = b'387.174 464.74875 cm\nBT\n/F2 9 Tf\n0 0 Td\n\n[\n(' + old_title.encode('utf-16be') + b')\n]\nTJ'
new_block = b'375.0 464.74875 cm\nBT\n/F2 9 Tf\n0 0 Td\n\n[\n(' + new_title.encode('utf-16be') + b')\n]\nTJ'
assert raw.count(old_block) == 1
updated = raw.replace(old_block, new_block)
exact_stream = DecodedStreamObject()
exact_stream.set_data(updated)
page.replace_contents(exact_stream)
writer.add_metadata({'/Title': 'Figure 3: matched ensembles, clarified panel B heading'})
writer.write(OUT)
assert old_title not in PdfReader(OUT).pages[0].extract_text()
assert new_title in PdfReader(OUT).pages[0].extract_text()
assert sha(SRC) == before_sha

dpi = 300
for tag, p in [('before', SRC), ('after', OUT)]:
    subprocess.run(['pdftoppm', '-singlefile', '-r', str(dpi), '-png', str(p), str(HERE/f'figure3_{tag}')], check=True, capture_output=True)
a = np.array(Image.open(HERE/'figure3_before.png').convert('RGB'))
b = np.array(Image.open(HERE/'figure3_after.png').convert('RGB'))
assert a.shape == b.shape
changed = np.any(a != b, axis=2)
h = float(page.mediabox.height)
box = [373, 462, 681, 475]
x0, y0, x1, y1 = box
allowed = np.zeros(changed.shape, dtype=bool)
allowed[math.floor((h-y1)*dpi/72):math.ceil((h-y0)*dpi/72), math.floor(x0*dpi/72):math.ceil(x1*dpi/72)] = True
outside = int((changed & ~allowed).sum())
assert outside == 0, outside
Image.open(HERE/'figure3_after.png').save(OUT.with_suffix('.png'))
pdffonts = shutil.which('pdffonts') or '/Users/teyian/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/poppler/bin/pdffonts'
fonts = subprocess.run([pdffonts, str(OUT)], check=True, text=True, capture_output=True).stdout
assert all(' no ' not in row for row in fonts.splitlines()[2:]), fonts
(HERE/'figure3_fonts.txt').write_text(fonts)
audit = {'source': str(SRC), 'source_sha256': before_sha, 'output': str(OUT), 'output_sha256': sha(OUT), 'old_title': old_title, 'new_title': new_title, 'font_family_size_unchanged': True, 'font_size_pt': 9, 'title_x_before': 387.174, 'title_x_after': 375.0, 'reason_for_title_shift': 'The new 303.74pt title would overlap C at its original anchor. Only this title moves 12.174pt left, leaving a 9.26pt gap.', 'title_only_box_pt': box, 'pixel_comparison_dpi': dpi, 'changed_pixels': int(changed.sum()), 'outside_title_box_changed_pixels': outside, 'all_fonts_embedded': True, 'all_data_axes_colors_other_panels_unchanged': True, 'source_unchanged': sha(SRC)==before_sha}
(HERE/'figure3_title_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
print(json.dumps(audit, indent=2))
