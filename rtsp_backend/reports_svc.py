"""
Report generation (Parts 8, 10).

Produces a JSON report always, and a PDF when ReportLab is available (it is a
pure-Python dependency, installed via requirements). The PDF embeds the
annotated image plus component counts, wire summary, and — for inspections —
the mismatch table. If ReportLab is missing the PDF step is skipped and the
caller still gets the JSON; nothing is faked.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional


def _ts_name(prefix: str, ext: str) -> str:
    return f"{prefix}_{int(time.time()*1000)}.{ext}"


def write_json(data_dir: str, subdir: str, payload: dict, prefix: str) -> str:
    rel = f"{subdir}/{_ts_name(prefix, 'json')}"
    path = os.path.join(data_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)
    return rel


def _try_reportlab():
    try:
        from reportlab.lib.pagesizes import A4  # noqa
        from reportlab.pdfgen import canvas  # noqa
        return True
    except Exception:
        return False


def panel_pdf(data_dir: str, result: dict, annotated_rel: Optional[str],
              title: str = "Panel Analysis Report") -> Optional[str]:
    if not _try_reportlab():
        return None
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    rel = f"reports/{_ts_name('panel', 'pdf')}"
    path = os.path.join(data_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4
    y = H - 20 * mm

    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, y, title)
    y -= 8 * mm
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, y, time.strftime("%Y-%m-%d %H:%M:%S"))
    y -= 10 * mm

    # annotated image
    if annotated_rel:
        img_path = os.path.join(data_dir, annotated_rel)
        if os.path.isfile(img_path):
            try:
                img = ImageReader(img_path)
                iw, ih = img.getSize()
                max_w = W - 40 * mm
                scale = min(max_w / iw, (90 * mm) / ih)
                c.drawImage(img, 20 * mm, y - ih * scale, iw * scale, ih * scale)
                y -= ih * scale + 8 * mm
            except Exception:
                pass

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, f"Components detected: {result.get('component_total', 0)}")
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    for name, n in (result.get("component_counts") or {}).items():
        c.drawString(24 * mm, y, f"• {name}: {n}")
        y -= 5 * mm
        if y < 30 * mm:
            c.showPage(); y = H - 20 * mm

    y -= 4 * mm
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, f"Wires detected: {result.get('wire_total', 0)}")
    y -= 6 * mm
    c.setFont("Helvetica", 10)
    for color, n in (result.get("wire_color_counts") or {}).items():
        c.drawString(24 * mm, y, f"• {color}: {n}")
        y -= 5 * mm
        if y < 30 * mm:
            c.showPage(); y = H - 20 * mm

    for note in result.get("notes", []):
        y -= 5 * mm
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(20 * mm, y, f"note: {note[:110]}")
        if y < 25 * mm:
            c.showPage(); y = H - 20 * mm

    c.showPage()
    c.save()
    return rel


def inspection_pdf(data_dir: str, insp: dict, annotated_rel: Optional[str],
                   title: str = "Inspection Report") -> Optional[str]:
    if not _try_reportlab():
        return None
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.lib.utils import ImageReader

    rel = f"reports/{_ts_name('inspection', 'pdf')}"
    path = os.path.join(data_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4
    y = H - 20 * mm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, y, title)
    y -= 8 * mm
    status = insp.get("status", "unknown").upper()
    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, f"Result: {status}   "
                 f"Mismatches: {insp.get('n_mismatches', 0)}")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, y, time.strftime("%Y-%m-%d %H:%M:%S"))
    y -= 10 * mm

    if annotated_rel:
        img_path = os.path.join(data_dir, annotated_rel)
        if os.path.isfile(img_path):
            try:
                img = ImageReader(img_path)
                iw, ih = img.getSize()
                scale = min((W - 40 * mm) / iw, (80 * mm) / ih)
                c.drawImage(img, 20 * mm, y - ih * scale, iw * scale, ih * scale)
                y -= ih * scale + 8 * mm
            except Exception:
                pass

    c.setFont("Helvetica-Bold", 11)
    c.drawString(20 * mm, y, "Mismatches")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    for mm_item in insp.get("mismatches", []):
        line = f"[{mm_item.get('type')}] {mm_item.get('detail', '')}"
        c.drawString(22 * mm, y, line[:120])
        y -= 5 * mm
        if y < 25 * mm:
            c.showPage(); y = H - 20 * mm
    c.showPage()
    c.save()
    return rel
