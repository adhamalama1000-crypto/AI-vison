"""
Report export for image analysis & comparison.

* ``write_json`` — always available (delegates to reports_svc).
* ``comparison_pdf`` — a ReportLab PDF embedding the side-by-side overlay, the
  similarity score, and the full difference table. Returns ``None`` (JSON still
  written) if ReportLab is unavailable — nothing is faked.
"""

from __future__ import annotations

import os
import time
from typing import Optional

from .. import reports_svc


def write_json(data_dir: str, payload: dict, prefix: str) -> str:
    return reports_svc.write_json(data_dir, "reports", payload, prefix)


def comparison_pdf(data_dir: str, result: dict, overlay_rel: Optional[str],
                   title: str = "Image Comparison Report") -> Optional[str]:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas
        from reportlab.lib.utils import ImageReader
    except Exception:
        return None

    rel = f"reports/imgcmp_{int(time.time()*1000)}.pdf"
    path = os.path.join(data_dir, rel)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    c = canvas.Canvas(path, pagesize=A4)
    W, H = A4
    y = H - 20 * mm
    c.setFont("Helvetica-Bold", 18)
    c.drawString(20 * mm, y, title)
    y -= 9 * mm
    c.setFont("Helvetica-Bold", 13)
    c.drawString(20 * mm, y, f"Similarity: {result.get('similarity', 0):.2f}%   "
                             f"Status: {result.get('status', '—').upper()}   "
                             f"Differences: {result.get('n_diffs', 0)}")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    c.drawString(20 * mm, y, time.strftime("%Y-%m-%d %H:%M:%S"))
    y -= 10 * mm

    if overlay_rel:
        img_path = os.path.join(data_dir, overlay_rel)
        if os.path.isfile(img_path):
            try:
                img = ImageReader(img_path)
                iw, ih = img.getSize()
                scale = min((W - 40 * mm) / iw, (95 * mm) / ih)
                c.drawImage(img, 20 * mm, y - ih * scale, iw * scale, ih * scale)
                y -= ih * scale + 8 * mm
            except Exception:
                pass

    c.setFont("Helvetica-Bold", 12)
    c.drawString(20 * mm, y, "Differences")
    y -= 6 * mm
    c.setFont("Helvetica", 9)
    for d in result.get("differences", []):
        line = f"[{d.get('severity','')}] {d.get('diff_type','')}: {d.get('detail','')}"
        c.drawString(22 * mm, y, line[:120])
        y -= 5 * mm
        if y < 25 * mm:
            c.showPage(); y = H - 20 * mm; c.setFont("Helvetica", 9)
    for n in result.get("notes", []):
        y -= 4 * mm
        c.setFont("Helvetica-Oblique", 8)
        c.drawString(20 * mm, y, f"note: {n[:110]}")
        if y < 22 * mm:
            c.showPage(); y = H - 20 * mm
    c.showPage()
    c.save()
    return rel
