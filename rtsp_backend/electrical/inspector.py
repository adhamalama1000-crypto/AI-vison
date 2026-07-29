"""
The panel inspection engine.

One entry point, :func:`inspect_panel`, takes an image and produces the complete
inspection result: recognised components with expert annotation, the inferred
panel type and function, the bill of materials, possible omissions, maintenance
observations, confidence statistics and an annotated overlay.

Deliberate design decisions
---------------------------
* **No wiring detection.** The previous engine ran a classical line/colour wire
  tracer over the whole frame, which turned every cabinet seam, duct edge,
  device outline and shadow into a "wire" — hundreds of them per image — and
  drowned the component result. Component recognition is the objective, so the
  wire stage is gone from this path entirely. See
  ``docs/AUDIT_PANEL_INSPECTOR.md``.
* **Honest emptiness.** With no trained (or zero-shot) recogniser available the
  result contains zero components and a note saying precisely why. It never
  fabricates a component list.
* **Every number is traceable.** The result carries the post-processing
  diagnostics — how many raw candidates the model emitted and how many each gate
  stage removed — so a claim like "false positives are down" is checkable rather
  than asserted.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional, Sequence

import numpy as np

from . import expert, panel_type, postprocess as pp, risk, taxonomy as tax

_log = logging.getLogger("rtsp_backend.electrical.inspector")

#: BGR overlay colours per taxonomy category.
CATEGORY_COLORS: dict[str, tuple[int, int, int]] = {
    "protection":     (60, 90, 235),
    "switching":      (235, 160, 40),
    "control":        (215, 120, 235),
    "automation":     (90, 210, 120),
    "hmi":            (60, 200, 235),
    "drives":         (235, 90, 160),
    "power":          (40, 200, 200),
    "instrumentation": (170, 200, 60),
    "network":        (200, 130, 90),
    "infrastructure": (150, 150, 150),
    "cooling":        (200, 200, 120),
}
UNKNOWN_COLOR = (110, 110, 200)


def _color_for(class_id: str) -> tuple[int, int, int]:
    if class_id == tax.UNKNOWN_COMPONENT_ID:
        return UNKNOWN_COLOR
    return CATEGORY_COLORS.get(tax.spec(class_id).category, (180, 180, 180))


def draw_overlay(image_bgr: np.ndarray,
                 findings: Sequence[expert.ComponentFinding]) -> np.ndarray:
    """Annotate a copy of the image with clean, readable component boxes.

    Real panels pack devices shoulder to shoulder, so a full-length label per box
    turns the overlay into unreadable overlapping text. The label therefore
    degrades with the space available: compact class name plus confidence when it
    fits inside the box width, the class name alone when only that fits, and a
    bare index badge (cross-referenced to the component table) when the device is
    too narrow for any text. Labels are also nudged to avoid the label placed
    immediately before them.
    """
    import cv2

    img = image_bgr.copy()
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.34, min(0.55, w / 2000.0))
    thickness = 2 if w >= 900 else 1
    placed: list[tuple[int, int, int, int]] = []

    def overlaps(rect: tuple[int, int, int, int]) -> bool:
        ax1, ay1, ax2, ay2 = rect
        for bx1, by1, bx2, by2 in placed:
            if ax1 < bx2 and bx1 < ax2 and ay1 < by2 and by1 < ay2:
                return True
        return False

    for f in findings:
        x1, y1, x2, y2 = [int(round(v)) for v in f.bbox]
        color = _color_for(f.class_id)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        box_w = max(0, x2 - x1)
        name = tax.short_name(f.class_id)
        candidates = [f"{name} {f.confidence * 100:.0f}%", name, f"#{f.index}"]
        text, tw, th = None, 0, 0
        for cand in candidates:
            (cw, ch), _ = cv2.getTextSize(cand, font, scale, 1)
            if cw + 6 <= max(box_w, 26) or cand.startswith("#"):
                text, tw, th = cand, cw, ch
                break
        if text is None:
            continue

        # Prefer above the box; fall back below, then inside, keeping it in frame
        # and clear of the previously drawn label.
        for top in (y1 - th - 7, y2 + 1, y1 + 1):
            ly1 = max(0, top)
            ly2 = min(h - 1, ly1 + th + 6)
            lx1 = max(0, min(x1, w - tw - 7))
            lx2 = min(w, lx1 + tw + 6)
            rect = (lx1, ly1, lx2, ly2)
            if ly2 <= ly1 or overlaps(rect):
                continue
            cv2.rectangle(img, (lx1, ly1), (lx2, ly2), color, -1)
            cv2.putText(img, text, (lx1 + 3, ly2 - 4), font, scale,
                        (18, 18, 18), 1, cv2.LINE_AA)
            placed.append(rect)
            break
    return img


# --------------------------------------------------------------------------
# main entry point
# --------------------------------------------------------------------------

def inspect_panel(recognizer, image_bgr: np.ndarray, *,
                  annotate: bool = True,
                  read_text: bool = True,
                  gate_config: Optional[pp.GateConfig] = None) -> dict:
    """Run a full inspection of one panel image.

    ``recognizer`` is any object exposing either ``recognize(frame) ->
    GateResult`` (the industrial backends) or the plain ``infer(frame)``
    detector interface; ``None`` or an unready backend yields an honest empty
    result with the reason attached.
    """
    started = time.time()
    t0 = time.perf_counter()
    h, w = image_bgr.shape[:2]

    result: dict[str, Any] = {
        "engine": "madkour_panel_inspector",
        "engine_version": "5.0",
        "image_size": [int(w), int(h)],
        "components": [],
        "component_total": 0,
        "component_counts": {},
        "bill_of_materials": [],
        "panel": {},
        "application": {},
        "missing_components": [],
        "maintenance_notes": [],
        "confidence": {},
        "layout": {"rows": 0, "description": []},
        "diagnostics": {},
        "notes": [],
        "ocr": {"engine": None, "item_count": 0},
        "wire_analysis": {
            "enabled": False,
            "reason": "Wiring detection is disabled by design. The classical "
                      "line tracer produced hundreds of false 'wires' from "
                      "cabinet seams, duct edges and shadows, which degraded "
                      "the whole result. Component recognition is the objective; "
                      "wire tracing will only return with a trained, validated "
                      "instance-segmentation model.",
        },
    }

    # -- recognition -------------------------------------------------------
    gate = None
    if recognizer is None:
        result["notes"].append(
            "No component recognition backend is selected. Choose one on the "
            "AI Models page.")
    elif not getattr(recognizer, "ready", False):
        reason = (getattr(recognizer, "_error", None)
                  or getattr(recognizer, "_reason", None)
                  or "backend not loaded")
        result["notes"].append(f"Component recogniser unavailable: {reason}")
    else:
        try:
            if hasattr(recognizer, "recognize"):
                gate = recognizer.recognize(image_bgr)
            else:
                gate = _adapt_plain_detector(recognizer, image_bgr, gate_config)
        except Exception as exc:
            _log.exception("component recognition failed")
            result["notes"].append(f"Component recognition error: "
                                   f"{type(exc).__name__}: {exc}")

    if gate is None:
        result["diagnostics"] = pp.Diagnostics().to_dict()
        result["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
        result["inspected_at"] = started
        result["panel"] = panel_type.classify({}).to_dict()
        result["confidence"] = pp.confidence_stats([])
        if annotate:
            result["_annotated"] = image_bgr.copy()
        return result

    result["diagnostics"] = gate.diagnostics.to_dict()
    if gate.truncated:
        result["notes"].append(
            f"Detection count exceeded the {len(gate.accepted)}-device cap; only "
            f"the most confident detections are reported. A real panel image "
            f"should not produce this many candidates — review the model.")

    # -- optional OCR for nameplates --------------------------------------
    ocr_items: list[dict] = []
    if read_text and gate.accepted:
        try:
            from ..imaging import ocr as ocr_mod
            ocr_res = ocr_mod.read_text(image_bgr)
            ocr_items = list(ocr_res.get("items") or [])
            result["ocr"] = {"engine": ocr_res.get("engine"),
                             "item_count": len(ocr_items),
                             "note": ocr_res.get("note")}
            if ocr_res.get("engine") in (None, "none"):
                result["notes"].append(
                    "No OCR engine installed — manufacturer and part numbers "
                    "cannot be read. Install easyocr, paddleocr or pytesseract "
                    "to enable nameplate identification.")
        except Exception as exc:
            result["notes"].append(f"OCR unavailable: {exc}")

    # -- expert annotation -------------------------------------------------
    findings = expert.annotate(gate.accepted, (h, w), gate.rows, ocr_items)
    counts = pp.counts(gate.accepted)
    unknown_n = counts.get(tax.UNKNOWN_COMPONENT_ID, 0)

    result["components"] = [f.to_dict() for f in findings]
    result["component_total"] = len(findings)
    result["component_counts"] = {tax.display_name(k): v
                                  for k, v in counts.items()}
    result["component_counts_by_id"] = counts
    result["bill_of_materials"] = expert.quantities(findings)
    result["confidence"] = pp.confidence_stats(gate.accepted)
    result["layout"] = {
        "rows": len(gate.rows),
        "description": expert.layout_description(findings, gate.rows),
    }

    # -- panel understanding ----------------------------------------------
    classification = panel_type.classify(counts)
    result["panel"] = classification.to_dict()

    all_text = [f.nameplate_text for f in findings if f.nameplate_text]
    all_text += [str(i.get("text") or "") for i in ocr_items]
    app = panel_type.infer_application(counts, all_text)
    result["application"] = app.to_dict()

    result["missing_components"] = [
        m.to_dict() for m in panel_type.missing_components(
            classification.panel_type, counts)]
    result["maintenance_notes"] = [
        n.to_dict() for n in panel_type.maintenance_notes(
            counts, classification.panel_type, unknown_n, len(findings))]

    if annotate:
        result["_annotated"] = draw_overlay(image_bgr, findings)

    result["duration_ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
    result["inspected_at"] = started
    return result


def _adapt_plain_detector(detector, image_bgr: np.ndarray,
                          gate_config: Optional[pp.GateConfig]) -> pp.GateResult:
    """Run a legacy ``infer()``-only detector through the new gate.

    Lets any previously-registered component backend benefit from the
    post-processing cascade without changing its code.
    """
    dets = detector.infer(image_bgr) or []
    cands: list[pp.Candidate] = []
    for d in dets:
        cid = (d.extra or {}).get("class_id") or tax.resolve(d.label) \
            or tax.UNKNOWN_COMPONENT_ID
        cands.append(pp.Candidate(
            class_id=cid, score=float(d.confidence),
            box=tuple(float(v) for v in d.bbox.as_list()),
            source=getattr(detector, "backend_id", "legacy"),
            raw_label=d.label, extra=dict(d.extra or {})))
    return pp.run(cands, image_bgr.shape[:2], gate_config or pp.GateConfig())


# --------------------------------------------------------------------------
# Report rendering
# --------------------------------------------------------------------------

_SEVERITY_ORDER = {"important": 0, "advisory": 1, "info": 2}


def build_report(result: dict) -> dict:
    """Structure an inspection result into the report sections.

    Sections match the specification exactly: Inspection Summary, Panel Type,
    Detected Components, Component Count, Possible Function, Possible Missing
    Components, Potential Maintenance Notes, Confidence Statistics, Inspection
    Time.
    """
    panel = result.get("panel") or {}
    app = result.get("application") or {}
    conf = result.get("confidence") or {}
    diag = result.get("diagnostics") or {}
    bom = result.get("bill_of_materials") or []
    total = int(result.get("component_total") or 0)
    unknown = int(conf.get("unknown") or 0)

    identified = total - unknown
    if total == 0:
        headline = ("No industrial components were recognised in this image. "
                    "The recogniser reported the reason below rather than "
                    "guessing.")
    elif panel.get("panel_type") and panel.get("panel_type") != panel_type.UNCLASSIFIED:
        headline = (
            f"{identified} of {total} detected devices were identified across "
            f"{len(bom)} component type(s). The composition indicates a "
            f"{panel.get('panel_type_name')} "
            f"({float(panel.get('confidence') or 0) * 100:.0f}% confidence).")
    else:
        headline = (
            f"{total} device(s) detected but the panel type could not be "
            f"determined: {panel.get('reason') or 'insufficient evidence'}.")

    notes = sorted(result.get("maintenance_notes") or [],
                   key=lambda n: _SEVERITY_ORDER.get(n.get("severity"), 3))
    missing = sorted(result.get("missing_components") or [],
                     key=lambda m: _SEVERITY_ORDER.get(m.get("severity"), 3))

    return {
        "title": "Panel Inspection Report",
        "generator": "Madkour AI Panel Inspector",
        "inspection_summary": {
            "headline": headline,
            "components_detected": total,
            "components_identified": identified,
            "components_unknown": unknown,
            "component_types": len(bom),
            "rows_detected": (result.get("layout") or {}).get("rows", 0),
            "image_size": result.get("image_size"),
            "notes": result.get("notes") or [],
        },
        "panel_type": {
            "id": panel.get("panel_type"),
            "name": panel.get("panel_type_name"),
            "confidence": panel.get("confidence"),
            "evidence": panel.get("evidence") or [],
            "alternatives": [
                {"name": c.get("name"), "confidence": c.get("confidence")}
                for c in (panel.get("candidates") or [])[1:]
            ],
        },
        "detected_components": result.get("components") or [],
        "component_count": {
            "total": total,
            "by_type": {b["name"]: b["quantity"] for b in bom},
            "by_category": _by_category(bom),
            "bill_of_materials": bom,
        },
        "possible_function": {
            "panel_function": panel.get("function"),
            "controlled_process": app.get("application"),
            "process_confidence": app.get("confidence"),
            "evidence": app.get("evidence") or [],
            "layout": (result.get("layout") or {}).get("description") or [],
        },
        "possible_missing_components": missing,
        "potential_maintenance_notes": notes,
        # Aggregate of the two sections above plus detection quality. Reported as
        # 'unknown' rather than 'low' when there is no basis to score — see
        # rtsp_backend.electrical.risk.
        "risk_assessment": risk.assess(result).to_dict(),
        "confidence_statistics": {
            **conf,
            "detection_gate": diag,
        },
        "inspection_time": {
            "inspected_at": result.get("inspected_at"),
            "duration_ms": result.get("duration_ms"),
            "ocr_engine": (result.get("ocr") or {}).get("engine"),
        },
        "wire_analysis": result.get("wire_analysis"),
    }


def _by_category(bom: Sequence[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for b in bom:
        cat = b.get("category") or "other"
        out[cat] = out.get(cat, 0) + int(b.get("quantity") or 0)
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def report_text(report: dict) -> str:
    """Plain-text rendering, used for the PDF body and CLI output."""
    lines: list[str] = []
    add = lines.append
    add("MADKOUR AI PANEL INSPECTOR — INSPECTION REPORT")
    add("=" * 62)
    s = report["inspection_summary"]
    add("")
    add("INSPECTION SUMMARY")
    add("-" * 62)
    add(s["headline"])
    add(f"Devices detected: {s['components_detected']}  "
        f"(identified {s['components_identified']}, "
        f"unknown {s['components_unknown']})")
    add(f"Component types: {s['component_types']}   "
        f"Device rows: {s['rows_detected']}")
    for n in s.get("notes") or []:
        add(f"  · {n}")

    pt = report["panel_type"]
    add("")
    add("PANEL TYPE")
    add("-" * 62)
    conf = pt.get("confidence")
    add(f"{pt.get('name') or 'Unclassified'}"
        + (f"  ({float(conf) * 100:.0f}% confidence)" if conf else ""))
    for e in pt.get("evidence") or []:
        add(f"  · {e}")
    alts = pt.get("alternatives") or []
    if alts:
        add("Alternatives considered: " + ", ".join(
            f"{a['name']} {float(a.get('confidence') or 0) * 100:.0f}%"
            for a in alts))

    add("")
    add("DETECTED COMPONENTS")
    add("-" * 62)
    comps = report.get("detected_components") or []
    if not comps:
        add("  (none)")
    for c in comps:
        loc = f"row {c['row']}" if c.get("row") else c.get("position", "")
        add(f"  [{c['index']:>3}] {c['title']}")
        add(f"        confidence {c['confidence_pct']:.1f}%  ·  {loc}  ·  "
            f"centre ({c['center'][0]:.0f}, {c['center'][1]:.0f})  ·  "
            f"bbox {c['bbox']}")
        if c.get("purpose"):
            add(f"        purpose: {c['purpose']}")

    cc = report["component_count"]
    add("")
    add("COMPONENT COUNT")
    add("-" * 62)
    for name, qty in cc["by_type"].items():
        add(f"  {qty:>3} × {name}")
    add(f"  {'':>3}   ── total {cc['total']}")

    pf = report["possible_function"]
    add("")
    add("POSSIBLE FUNCTION")
    add("-" * 62)
    add(pf.get("panel_function") or "Undetermined.")
    if pf.get("controlled_process"):
        add(f"Likely controlled process: {pf['controlled_process']} "
            f"({float(pf.get('process_confidence') or 0) * 100:.0f}% of evidence)")
    for e in pf.get("evidence") or []:
        add(f"  · {e}")
    for line in pf.get("layout") or []:
        add(f"  {line}")

    add("")
    add("POSSIBLE MISSING COMPONENTS")
    add("-" * 62)
    miss = report.get("possible_missing_components") or []
    if not miss:
        add("  None — every component expected for this panel type was detected.")
    for m in miss:
        add(f"  [{m['severity'].upper()}] {m['name']}")
        add(f"        {m['rationale']}")

    add("")
    add("POTENTIAL MAINTENANCE NOTES")
    add("-" * 62)
    notes = report.get("potential_maintenance_notes") or []
    if not notes:
        add("  No observations raised from the detected inventory.")
    for n in notes:
        add(f"  [{n['severity'].upper()}] {n['message']}")

    cs = report["confidence_statistics"]
    add("")
    add("CONFIDENCE STATISTICS")
    add("-" * 62)
    if cs.get("count"):
        add(f"  mean {cs['mean']:.3f}   median {cs['median']:.3f}   "
            f"min {cs['min']:.3f}   max {cs['max']:.3f}")
        add(f"  detections below 0.50: {cs['below_0_5']}   "
            f"unclassified: {cs['unknown']}")
    else:
        add("  No detections to summarise.")
    gate = cs.get("detection_gate") or {}
    if gate:
        add(f"  Gate: {gate.get('input_count', 0)} raw candidate(s) → "
            f"{gate.get('output_count', 0)} accepted "
            f"({gate.get('dropped_total', 0)} suppressed, "
            f"{gate.get('relabelled_unknown', 0)} demoted to unknown)")
        for reason, n in (gate.get("dropped_by_reason") or {}).items():
            add(f"        {reason}: {n}")

    it = report["inspection_time"]
    add("")
    add("INSPECTION TIME")
    add("-" * 62)
    add(f"  Duration: {it.get('duration_ms')} ms")
    if it.get("inspected_at"):
        add("  Timestamp: " + time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime(float(it["inspected_at"]))))
    add(f"  OCR engine: {it.get('ocr_engine') or 'none'}")

    wa = report.get("wire_analysis") or {}
    if wa and not wa.get("enabled", True):
        add("")
        add("WIRING ANALYSIS")
        add("-" * 62)
        add("  Disabled by design. " + str(wa.get("reason", "")))
    return "\n".join(lines)


__all__ = ["CATEGORY_COLORS", "draw_overlay", "inspect_panel", "build_report",
           "report_text"]
