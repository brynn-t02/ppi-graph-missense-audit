#!/usr/bin/env python3
"""Read-only page-by-page 300-dpi and exact character-geometry PDF audit."""
import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pdfplumber
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "outputs/BIBM_workshop_heading_style_20261025"
NEW = ROOT / "outputs/BIBM_workshop_font_clean_20261025"
OUT = ROOT / "work/font_cleanup_20261025"
REPORT = OUT / "pixel_equality_audit.json"
PAIRS = [
    ("main_named", OLD / "BIBM2026_workshop_named_revised.pdf", NEW / "source/main.pdf", 8, 6),
    ("main_anonymous", OLD / "BIBM2026_workshop_anonymous_revised.pdf", NEW / "source/main_anonymous.pdf", 8, None),
    ("supplement_named", OLD / "BIBM2026_supplementary_named_revised.pdf", NEW / "BIBM2026_supplementary_named_revised.pdf", 18, 3),
    ("supplement_anonymous", OLD / "BIBM2026_supplementary_anonymous_revised.pdf", NEW / "BIBM2026_supplementary_anonymous_revised.pdf", 18, None),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def signature(char):
    return (
        char["text"], char["x0"], char["top"], char["size"],
        re.sub(r"^[A-Z]{6}\+", "", char["fontname"]),
    )


def render(path, page, target):
    subprocess.run([
        "pdftoppm", "-f", str(page), "-l", str(page), "-r", "300",
        "-png", "-singlefile", str(path), str(target),
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return target.with_suffix(".png")


def persist(report):
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


def main():
    reuse_main = "--only-supplements" in sys.argv[1:]
    previous_report = json.loads(REPORT.read_text()) if reuse_main else None
    report = {
        "status": "running",
        "scope": "Read-only comparison of four PDFs after unused-font cleanup; no PDF changed by audit.",
        "raster": {"renderer": "pdftoppm", "dpi": 300, "comparison": "exact RGB channel equality"},
        "character_comparison": {
            "library": "pdfplumber", "fields": ["text", "x0", "top", "size", "fontname"],
            "fontname_normalization": "strip initial six uppercase letters plus plus-sign subset prefix",
            "numeric_tolerance": 0,
        },
        "pairs": [], "total_pages_checked": 0, "total_changed_pixels": 0,
        "representative_pngs": [], "findings": [],
    }
    if reuse_main:
        assert previous_report["status"] == "passed", "Prior audit must have passed."
        report["main_results_reused_after_hash_verification"] = True
        report["refresh_reason"] = "Supplement-only rebuild to remove unused per-page initialization font resource."
        for item in previous_report["pairs"]:
            if not item["label"].startswith("main_"):
                continue
            assert sha(Path(item["old_path"])) == item["old_sha256"], "Old main hash changed."
            assert sha(Path(item["new_path"])) == item["new_sha256"], "New main hash changed."
            assert all(p["changed_pixels"] == 0 and p["exact_character_signature_equality"] for p in item["pages"])
            item["reused_after_refresh_hash_verification"] = True
            report["pairs"].append(item)
            report["total_pages_checked"] += len(item["pages"])
        assert report["total_pages_checked"] == 16
        report["representative_pngs"] = [
            x for x in previous_report["representative_pngs"] if x["label"].startswith("main_")
        ]
    persist(report)
    for label, old_path, new_path, expected, keep_page in (PAIRS[2:] if reuse_main else PAIRS):
        item = {
            "label": label, "old_path": str(old_path), "new_path": str(new_path),
            "old_sha256": sha(old_path), "new_sha256": sha(new_path), "expected_pages": expected,
            "pages": [],
        }
        report["pairs"].append(item)
        with pdfplumber.open(old_path) as old_pdf, pdfplumber.open(new_path) as new_pdf:
            item["old_page_count"] = len(old_pdf.pages)
            item["new_page_count"] = len(new_pdf.pages)
            if len(old_pdf.pages) != expected or len(new_pdf.pages) != expected:
                report["findings"].append({"label": label, "issue": "unexpected page count"})
                report["status"] = "failed"
                persist(report)
                raise SystemExit("FAIL: page-count difference")
            for page_number, (old_page, new_page) in enumerate(zip(old_pdf.pages, new_pdf.pages), 1):
                old_chars = [signature(c) for c in old_page.chars]
                new_chars = [signature(c) for c in new_page.chars]
                chars_equal = old_chars == new_chars
                page_item = {
                    "page": page_number, "old_character_count": len(old_chars),
                    "new_character_count": len(new_chars), "exact_character_signature_equality": chars_equal,
                }
                if not chars_equal:
                    unequal = [
                        {"index": i, "old": a, "new": b}
                        for i, (a, b) in enumerate(zip(old_chars, new_chars)) if a != b
                    ]
                    page_item["first_character_differences"] = unequal[:10]
                with tempfile.TemporaryDirectory(prefix="ppi-font-qa-") as temporary:
                    temp = Path(temporary)
                    old_png = render(old_path, page_number, temp / "old")
                    new_png = render(new_path, page_number, temp / "new")
                    with Image.open(old_png) as old_img, Image.open(new_png) as new_img:
                        old_rgb = np.asarray(old_img.convert("RGB"))
                        new_rgb = np.asarray(new_img.convert("RGB"))
                    page_item["old_rgb_shape"] = list(old_rgb.shape)
                    page_item["new_rgb_shape"] = list(new_rgb.shape)
                    same_shape = old_rgb.shape == new_rgb.shape
                    if same_shape:
                        delta = np.any(old_rgb != new_rgb, axis=2)
                        changed_pixels = int(np.count_nonzero(delta))
                        page_item["changed_pixels"] = changed_pixels
                        if changed_pixels:
                            ys, xs = np.where(delta)
                            page_item["difference_bbox_xyxy"] = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
                    else:
                        changed_pixels = -1
                        page_item["changed_pixels"] = None
                    if page_number == keep_page:
                        kept = OUT / f"font_clean_{label}_page{page_number}_300dpi.png"
                        shutil.copyfile(new_png, kept)
                        report["representative_pngs"].append({"path": str(kept), "sha256": sha(kept), "label": label, "page": page_number})
                item["pages"].append(page_item)
                report["total_pages_checked"] += 1
                report["total_changed_pixels"] += max(changed_pixels, 0)
                if not chars_equal or not same_shape or changed_pixels:
                    report["findings"].append({"label": label, "page": page_number, "issue": "character or raster difference"})
                    report["status"] = "failed"
                    persist(report)
                    raise SystemExit(f"FAIL: {label} page {page_number}; chars_equal={chars_equal}; changed_pixels={changed_pixels}")
                persist(report)
                print(f"PASS {label} page {page_number}: RGB 0 pixels differ; {len(old_chars)} exact character signatures", flush=True)
                old_page.close()
                new_page.close()
        item["source_files_still_same_hash"] = sha(old_path) == item["old_sha256"] and sha(new_path) == item["new_sha256"]
        if not item["source_files_still_same_hash"]:
            report["findings"].append({"label": label, "issue": "source PDF changed during audit"})
            report["status"] = "failed"
            persist(report)
            raise SystemExit("FAIL: concurrent PDF changes")
    report["status"] = "passed"
    report["all_52_pages_exact_rgb_equal"] = report["total_pages_checked"] == 52 and report["total_changed_pixels"] == 0
    report["all_character_signatures_exactly_equal"] = all(p["exact_character_signature_equality"] for item in report["pairs"] for p in item["pages"])
    persist(report)
    print(f"COMPLETE: {report['total_pages_checked']} pages, 0 differing pixels, exact character geometry/font equality", flush=True)


if __name__ == "__main__":
    main()
