"""
Datasheet / schematic understanding.

Ingests a datasheet, single-line diagram or wiring schematic (PDF / PNG / JPG /
DXF / SVG) and extracts the electrical intent:

* **Component IDs**  — IEC device tags (Q1, F2, KM1, KA3, T1, S1, H1, PLC1 …)
* **Terminal IDs**   — terminal references (X1, X1:1, -X2:PE …)
* **Wire IDs**       — wire numbers / labels
* **Connections**    — "A – B" / "A → B" / "A to B" endpoint pairs
* **Expected graph** — a graph built from the parsed IDs + connections, which an
  inspection can be compared against directly.

OCR is used for raster / PDF content. PaddleOCR is preferred when installed,
Tesseract (pytesseract) is the fallback; DXF/SVG carry text natively and are
parsed without OCR. If no OCR engine is available for a raster/PDF input the
extractor returns ``ocr_engine="none"`` with a clear note — it never fabricates
IDs it could not read.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import Any, Optional

# IEC-style device tag, e.g. Q1, KM12, -KA3, F2.1  (letters then digits)
_TAG_RE = re.compile(r"\b-?([A-Z]{1,3})(\d{1,3}(?:\.\d{1,2})?)\b")
# terminal reference, e.g. X1:1, -X2:PE, X10:12
_TERM_RE = re.compile(r"\b-?(X\d{1,3})[:\-.]([0-9A-Z]{1,4})\b")
# wire number labels, e.g. W12, wire 34, L1, N, PE
_WIRE_RE = re.compile(r"\b(W\d{1,4}|wire\s*\d{1,4})\b", re.IGNORECASE)
# connections "A - B", "A -> B", "A → B", "A to B"
_CONN_RE = re.compile(
    r"([A-Z]{1,3}\d{1,3}(?::[0-9A-Z]{1,4})?)\s*(?:-|--|->|=>|→|to)\s*"
    r"([A-Z]{1,3}\d{1,3}(?::[0-9A-Z]{1,4})?)", re.IGNORECASE)

_COMPONENT_PREFIXES = {
    "Q": "circuit_breaker", "F": "fuse", "K": "relay", "KM": "contactor",
    "KA": "relay", "T": "transformer", "X": "terminal_block", "S": "switch",
    "H": "indicator_lamp", "P": "meter", "M": "motor", "G": "generator",
    "U": "converter", "A": "assembly", "PLC": "plc",
}


def _classify(prefix: str) -> str:
    return _COMPONENT_PREFIXES.get(prefix.upper(), "component")


# ---------------------------------------------------------------------------
# text acquisition per format
# ---------------------------------------------------------------------------

def _ocr_image(path: str) -> tuple[str, str]:
    """Return (text, engine). Tries PaddleOCR then Tesseract."""
    try:  # PaddleOCR preferred
        from paddleocr import PaddleOCR  # type: ignore
        import numpy as np  # noqa
        import cv2
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        res = ocr.ocr(path, cls=True)
        lines = []
        for page in res or []:
            for item in page or []:
                lines.append(item[1][0])
        return "\n".join(lines), "paddleocr"
    except Exception:
        pass
    try:  # Tesseract fallback
        import pytesseract  # type: ignore
        from PIL import Image  # type: ignore
        return pytesseract.image_to_string(Image.open(path)), "tesseract"
    except Exception:
        return "", "none"


def _pdf_text(path: str) -> tuple[str, str]:
    """Extract embedded text; if empty, OCR rendered pages when possible."""
    text = ""
    try:
        from pypdf import PdfReader  # type: ignore
        reader = PdfReader(path)
        text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
    except Exception:
        try:
            from PyPDF2 import PdfReader  # type: ignore
            reader = PdfReader(path)
            text = "\n".join((pg.extract_text() or "") for pg in reader.pages)
        except Exception:
            text = ""
    if text.strip():
        return text, "pdf-text"
    # try rasterising for OCR
    try:
        from pdf2image import convert_from_path  # type: ignore
        import tempfile
        pages = convert_from_path(path, dpi=200)
        allt, eng = [], "none"
        with tempfile.TemporaryDirectory() as tmp:
            for i, pg in enumerate(pages[:10]):
                p = os.path.join(tmp, f"p{i}.png")
                pg.save(p)
                t, eng = _ocr_image(p)
                allt.append(t)
        return "\n".join(allt), eng
    except Exception:
        return "", "none"


def _dxf_text(path: str) -> tuple[str, str]:
    try:
        import ezdxf  # type: ignore
        doc = ezdxf.readfile(path)
        msp = doc.modelspace()
        parts = []
        for e in msp:
            if e.dxftype() in ("TEXT", "MTEXT"):
                try:
                    parts.append(e.plain_text() if e.dxftype() == "MTEXT" else e.dxf.text)
                except Exception:
                    pass
        return "\n".join(parts), "dxf"
    except Exception:
        return "", "none"


def _svg_text(path: str) -> tuple[str, str]:
    try:
        tree = ET.parse(path)
        texts = [el.text.strip() for el in tree.iter()
                 if el.tag.endswith("text") and el.text and el.text.strip()]
        return "\n".join(texts), "svg"
    except Exception:
        return "", "none"


_EXT_KIND = {".pdf": "pdf", ".png": "image", ".jpg": "image", ".jpeg": "image",
             ".bmp": "image", ".webp": "image", ".dxf": "dxf", ".svg": "svg"}


def kind_for(path: str) -> str:
    return _EXT_KIND.get(os.path.splitext(path)[1].lower(), "other")


def acquire_text(path: str) -> tuple[str, str]:
    kind = kind_for(path)
    if kind == "image":
        return _ocr_image(path)
    if kind == "pdf":
        return _pdf_text(path)
    if kind == "dxf":
        return _dxf_text(path)
    if kind == "svg":
        return _svg_text(path)
    # plain text or unknown: read as utf-8
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.read(), "text"
    except Exception:
        return "", "none"


# ---------------------------------------------------------------------------
# parsing + graph
# ---------------------------------------------------------------------------

def parse_text(text: str) -> dict[str, Any]:
    components: dict[str, str] = {}
    terminals: set[str] = set()
    wires: set[str] = set()
    connections: list[dict] = []

    for m in _TERM_RE.finditer(text):
        terminals.add(f"{m.group(1)}:{m.group(2)}")
        components.setdefault(m.group(1), "terminal_block")
    for m in _TAG_RE.finditer(text):
        prefix, num = m.group(1), m.group(2)
        tag = f"{prefix}{num}"
        components.setdefault(tag, _classify(prefix))
    for m in _WIRE_RE.finditer(text):
        wires.add(m.group(1).upper().replace(" ", ""))
    for m in _CONN_RE.finditer(text):
        a, b = m.group(1).upper(), m.group(2).upper()
        connections.append({"from": a, "to": b})

    return {
        "component_ids": sorted(components.keys()),
        "component_types": components,
        "terminal_ids": sorted(terminals),
        "wire_ids": sorted(wires),
        "connections": connections,
        "n_components": len(components),
        "n_terminals": len(terminals),
        "n_connections": len(connections),
    }


def build_expected_graph(parsed: dict) -> dict[str, Any]:
    nodes = []
    for tag, ctype in (parsed.get("component_types") or {}).items():
        nodes.append({"id": tag, "kind": "component", "type": ctype, "label": tag})
    for t in parsed.get("terminal_ids") or []:
        nodes.append({"id": t, "kind": "terminal", "label": t})
    known = {n["id"] for n in nodes}
    edges = []
    for i, conn in enumerate(parsed.get("connections") or []):
        for nid in (conn["from"], conn["to"]):
            if nid not in known:
                nodes.append({"id": nid, "kind": "component", "type": "component",
                              "label": nid})
                known.add(nid)
        edges.append({"id": f"e{i}", "kind": "wire",
                      "from": conn["from"], "to": conn["to"]})
    return {"nodes": nodes, "edges": edges,
            "node_count": len(nodes), "edge_count": len(edges)}


def extract(path: str) -> dict[str, Any]:
    """Full pipeline: acquire text -> parse -> expected graph."""
    text, engine = acquire_text(path)
    parsed = parse_text(text or "")
    graph = build_expected_graph(parsed)
    note = None
    if engine == "none":
        note = ("no OCR engine available for this input — install paddleocr or "
                "pytesseract (+ the tesseract binary) to read raster/PDF "
                "schematics; DXF/SVG are parsed without OCR")
    elif not parsed["component_ids"] and not parsed["terminal_ids"]:
        note = ("no electrical IDs recognised in the document text — check the "
                "scan quality or that the schematic uses IEC-style tags")
    return {
        "ocr_engine": engine,
        "text_chars": len(text or ""),
        "parsed": parsed,
        "expected_graph": graph,
        "note": note,
    }
