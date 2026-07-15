"""
Reference-vs-observed comparison & error detection.

Diffs an observed panel analysis against a learned reference template and emits
typed, confidence-scored errors covering the full fault taxonomy:

  Components : missing_component, extra_component, wrong_component,
               moved_component, wrong_rotation
  Wires      : missing_wire, extra_wire, wrong_wire, loose_wire,
               disconnected_wire, broken_wire, wrong_wire_color
  Terminals  : wrong_terminal, wrong_source, wrong_destination

The observed image is first registered to the reference frame with a homography
(``features.align``) so pose / zoom differences don't masquerade as faults.
When alignment fails the comparison falls back to identity (raw pixel
coordinates) and lowers its confidence, recording the reason — it never
silently pretends the panels are aligned.

Every verdict is derived from real geometry. Where a required capability is
absent (e.g. no component model), the affected error classes are suppressed and
a note explains why, instead of reporting false positives.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from . import features as _features

# tuneable thresholds (pixels are in the reference image's coordinate frame)
DEFAULTS = {
    "match_dist": 60.0,          # component centre match radius
    "moved_dist": 28.0,          # beyond this a match counts as "moved"
    "rotation_tol": 18.0,        # deg
    "wire_end_dist": 45.0,       # wire endpoint match radius
    "loose_dist": 40.0,          # endpoint-to-terminal for "connected"
    "broken_ratio": 0.6,         # observed/ref length below this => broken
}


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _warp_center(H, c) -> tuple[float, float]:
    return _features.warp_point(H, float(c["cx"]), float(c["cy"]))


def _warp_pt(H, p) -> tuple[float, float]:
    return _features.warp_point(H, float(p[0]), float(p[1]))


def compare(reference: dict, observed: dict, observed_bgr=None,
            ref_features: Optional[dict] = None,
            params: Optional[dict] = None) -> dict[str, Any]:
    """Compare an observed analysis against a reference template.

    ``reference`` is a learned template dict (``components``/``terminals``/
    ``wires``/``graph``). ``observed`` is the output of
    :func:`panels.template.analyze_image`. ``observed_bgr`` + ``ref_features``
    enable homography registration.
    """
    p = {**DEFAULTS, **(params or {})}
    errors: list[dict] = []
    notes: list[str] = []

    # -- registration ------------------------------------------------------
    H = None
    align_info = {"ok": False, "note": "no image/features supplied"}
    if observed_bgr is not None and ref_features:
        align_info = _features.align(ref_features, observed_bgr)
        if align_info.get("ok"):
            H = align_info["homography"]
        else:
            notes.append(f"registration failed ({align_info.get('note')}); "
                         "comparing in raw coordinates with reduced confidence")
    align_conf = 1.0 if H is not None else 0.65

    ref_comps = reference.get("components") or []
    obs_comps = observed.get("components") or []
    have_component_model = not any("no trained component model" in n
                                   for n in observed.get("notes", []))

    # ================= COMPONENTS =================
    matched_obs: set[int] = set()
    if have_component_model or ref_comps:
        for rc in ref_comps:
            rc_center = (float(rc["cx"]), float(rc["cy"]))
            best_i, best_d, best_type_ok = None, p["match_dist"], False
            for i, oc in enumerate(obs_comps):
                if i in matched_obs:
                    continue
                oc_center = _warp_center(H, oc)
                d = _dist(rc_center, oc_center)
                if d < best_d:
                    best_d, best_i = d, i
                    best_type_ok = (oc.get("comp_type") == rc.get("comp_type"))
            if best_i is None:
                errors.append(_err("missing_component", "error", rc.get("ref_id"),
                                   f"missing {rc.get('comp_type')} at "
                                   f"{rc.get('position')}", align_conf,
                                   rc_center))
                continue
            matched_obs.add(best_i)
            oc = obs_comps[best_i]
            if not best_type_ok:
                errors.append(_err("wrong_component", "error", rc.get("ref_id"),
                                   f"expected {rc.get('comp_type')} but found "
                                   f"{oc.get('comp_type')}",
                                   round(align_conf * float(oc.get("confidence", 0.6)), 3),
                                   rc_center))
                continue
            if best_d > p["moved_dist"]:
                conf = round(align_conf * min(1.0, best_d / (2 * p["moved_dist"])), 3)
                errors.append(_err("moved_component", "warning", rc.get("ref_id"),
                                   f"{rc.get('comp_type')} moved ~{best_d:.0f}px", conf,
                                   rc_center))
            rot_diff = _rot_diff(rc.get("rotation"), oc.get("rotation"))
            if rot_diff is not None and rot_diff > p["rotation_tol"]:
                errors.append(_err("wrong_rotation", "warning", rc.get("ref_id"),
                                   f"{rc.get('comp_type')} rotated {rot_diff:.0f}° "
                                   "vs reference",
                                   round(align_conf * min(1.0, rot_diff / 90.0), 3),
                                   rc_center))
        # extra components
        for i, oc in enumerate(obs_comps):
            if i not in matched_obs:
                errors.append(_err("extra_component", "error", oc.get("label"),
                                   f"unexpected {oc.get('comp_type')} not in reference",
                                   round(align_conf * float(oc.get("confidence", 0.6)), 3),
                                   _warp_center(H, oc)))
    else:
        notes.append("component-level checks skipped: no trained component model "
                     "(wire/topology checks still run)")

    # ================= WIRES =================
    ref_wires = reference.get("wires") or []
    obs_wires = observed.get("wires") or []
    obs_terms = observed.get("terminals") or []
    matched_ow: set[int] = set()

    for rw in ref_wires:
        r_start = (rw["start"][0], rw["start"][1])
        r_end = (rw["end"][0], rw["end"][1])
        best_i, best_score = None, None
        for i, ow in enumerate(obs_wires):
            if i in matched_ow:
                continue
            o_start = _warp_pt(H, ow["start"])
            o_end = _warp_pt(H, ow["end"])
            # orientation-agnostic endpoint distance
            d_direct = _dist(r_start, o_start) + _dist(r_end, o_end)
            d_flip = _dist(r_start, o_end) + _dist(r_end, o_start)
            d = min(d_direct, d_flip)
            if d < 2 * p["wire_end_dist"] and (best_score is None or d < best_score):
                best_score, best_i = d, i
        if best_i is None:
            errors.append(_err("missing_wire", "error", rw.get("wire_uid"),
                               f"missing {rw.get('color')} wire "
                               f"({rw.get('from_terminal') or rw.get('from_component')} → "
                               f"{rw.get('to_terminal') or rw.get('to_component')})",
                               align_conf, _midpoint(r_start, r_end)))
            continue
        matched_ow.add(best_i)
        ow = obs_wires[best_i]
        # broken: matched but far shorter than reference
        if rw.get("length") and ow.get("length"):
            ratio = float(ow["length"]) / float(rw["length"])
            if ratio < p["broken_ratio"]:
                errors.append(_err("broken_wire", "error", rw.get("wire_uid"),
                                   f"wire fragmented ({ratio*100:.0f}% of expected length)",
                                   round(align_conf * (1 - ratio), 3),
                                   _midpoint(r_start, r_end)))
        # wrong colour
        if rw.get("color") and ow.get("color") and rw["color"] != ow["color"] \
                and "unknown" not in (rw["color"], ow["color"]):
            errors.append(_err("wrong_wire_color", "warning", rw.get("wire_uid"),
                               f"expected {rw['color']} wire, found {ow['color']}",
                               round(align_conf * 0.8, 3), _midpoint(r_start, r_end)))
        # terminal / source / destination checks
        _check_endpoints(rw, ow, errors, align_conf, _midpoint(r_start, r_end))
        # loose / disconnected: observed endpoint not near any terminal, but the
        # reference wire *was* connected there.
        _check_connection(rw, ow, obs_terms, H, p, errors, align_conf)

    # extra wires
    for i, ow in enumerate(obs_wires):
        if i not in matched_ow:
            errors.append(_err("extra_wire", "warning", ow.get("wire_uid"),
                               f"unexpected {ow.get('color')} wire not in reference",
                               round(align_conf * float(ow.get("confidence", 0.5)), 3),
                               _midpoint(_warp_pt(H, ow["start"]), _warp_pt(H, ow["end"]))))

    # ================= SCORING =================
    n_err = sum(1 for e in errors if e["severity"] == "error")
    n_warn = sum(1 for e in errors if e["severity"] == "warning")
    ref_elems = max(1, len(ref_comps) + len(ref_wires))
    penalty = (n_err + 0.4 * n_warn) / ref_elems
    score = round(max(0.0, 1.0 - penalty), 3)
    if n_err:
        status = "fail"
    elif n_warn:
        status = "warning"
    else:
        status = "pass"

    return {
        "status": status,
        "score": score,
        "n_errors": n_err,
        "n_warnings": n_warn,
        "errors": errors,
        "alignment": align_info,
        "counts": {
            "reference_components": len(ref_comps),
            "observed_components": len(obs_comps),
            "reference_wires": len(ref_wires),
            "observed_wires": len(obs_wires),
        },
        "component_model_loaded": have_component_model,
        "notes": notes,
    }


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _err(etype, severity, target, detail, confidence, pt=None) -> dict:
    e = {"error_type": etype, "severity": severity, "target": target,
         "detail": detail, "confidence": round(float(confidence), 3)}
    if pt is not None:
        e["x"], e["y"] = round(float(pt[0]), 1), round(float(pt[1]), 1)
    return e


def _midpoint(a, b) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _rot_diff(a, b) -> Optional[float]:
    if a is None or b is None:
        return None
    d = abs(float(a) - float(b)) % 180.0
    return min(d, 180.0 - d)


def _check_endpoints(rw, ow, errors, conf, pt) -> None:
    """wrong_terminal / wrong_source / wrong_destination when a matched wire
    connects to different reference terminals/components than expected."""
    rs_t, re_t = rw.get("from_terminal"), rw.get("to_terminal")
    os_t, oe_t = ow.get("from_terminal"), ow.get("to_terminal")
    # only meaningful when the reference declares terminals
    if rs_t and os_t and rs_t != os_t and oe_t != rs_t:
        errors.append(_err("wrong_source", "error", rw.get("wire_uid"),
                           f"wire source terminal {os_t} ≠ expected {rs_t}",
                           round(conf * 0.8, 3), pt))
    if re_t and oe_t and re_t != oe_t and os_t != re_t:
        errors.append(_err("wrong_destination", "error", rw.get("wire_uid"),
                           f"wire destination terminal {oe_t} ≠ expected {re_t}",
                           round(conf * 0.8, 3), pt))
    rs_c, re_c = rw.get("from_component"), rw.get("to_component")
    os_c, oe_c = ow.get("from_component"), ow.get("to_component")
    if rs_c and os_c and {rs_c, re_c} != {os_c, oe_c} and not (rs_t or os_t):
        errors.append(_err("wrong_terminal", "warning", rw.get("wire_uid"),
                           f"wire connects {os_c}↔{oe_c}, expected {rs_c}↔{re_c}",
                           round(conf * 0.7, 3), pt))


def _check_connection(rw, ow, obs_terms, H, p, errors, conf) -> None:
    """loose_wire / disconnected_wire: reference wire was connected at an end
    but the observed wire's corresponding end floats free of any terminal."""
    ref_connected_start = bool(rw.get("from_terminal") or rw.get("from_component"))
    ref_connected_end = bool(rw.get("to_terminal") or rw.get("to_component"))
    if not obs_terms:
        return
    term_pts = [(float(t["x"]), float(t["y"])) for t in obs_terms
                if t.get("x") is not None]
    if not term_pts:
        return

    def free(pt_obs) -> float:
        wp = _warp_pt(H, pt_obs)
        return min(_dist(wp, tp) for tp in term_pts)

    if ref_connected_start:
        d = free(ow["start"])
        if d > p["loose_dist"]:
            sev = "error" if d > 2 * p["loose_dist"] else "warning"
            etype = "disconnected_wire" if sev == "error" else "loose_wire"
            errors.append(_err(etype, sev, rw.get("wire_uid"),
                               f"wire start not connected (nearest terminal {d:.0f}px)",
                               round(conf * min(1.0, d / (3 * p["loose_dist"])), 3),
                               _warp_pt(H, ow["start"])))
    if ref_connected_end:
        d = free(ow["end"])
        if d > p["loose_dist"]:
            sev = "error" if d > 2 * p["loose_dist"] else "warning"
            etype = "disconnected_wire" if sev == "error" else "loose_wire"
            errors.append(_err(etype, sev, rw.get("wire_uid"),
                               f"wire end not connected (nearest terminal {d:.0f}px)",
                               round(conf * min(1.0, d / (3 * p["loose_dist"])), 3),
                               _warp_pt(H, ow["end"])))
