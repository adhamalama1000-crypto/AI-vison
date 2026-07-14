"""
Reference-vs-observed inspection (Part 10).

Compares a panel-analysis result (from a live camera frame or an uploaded
image) against a reference design's expected specification and reports:

* missing components (expected but not found),
* extra components (found but not expected),
* wrong counts (found N, expected M),
* wrong / unexpected wire colours.

The "expected" spec is either stored explicitly on the reference design (a JSON
of ``component_counts`` / ``wire_color_counts``) or derived by analysing a
reference *image* with the same pipeline. Comparison is pure arithmetic over
real detections — no fabricated verdicts. When no component model is loaded the
report says so instead of pretending the panel is correct.
"""

from __future__ import annotations

from typing import Any


def build_expected_from_analysis(result: dict) -> dict:
    return {
        "component_counts": dict(result.get("component_counts") or {}),
        "wire_color_counts": dict(result.get("wire_color_counts") or {}),
    }


def compare(expected: dict, observed: dict) -> dict:
    exp_c = {k: int(v) for k, v in (expected.get("component_counts") or {}).items()}
    obs_c = {k: int(v) for k, v in (observed.get("component_counts") or {}).items()}
    exp_w = {k: int(v) for k, v in (expected.get("wire_color_counts") or {}).items()}
    obs_w = {k: int(v) for k, v in (observed.get("wire_color_counts") or {}).items()}

    mismatches: list[dict[str, Any]] = []

    all_c = set(exp_c) | set(obs_c)
    for name in sorted(all_c):
        e, o = exp_c.get(name, 0), obs_c.get(name, 0)
        if e and not o:
            mismatches.append({"type": "missing_component", "component": name,
                               "expected": e, "found": 0,
                               "detail": f"missing {name} (expected {e})"})
        elif o and not e:
            mismatches.append({"type": "extra_component", "component": name,
                               "expected": 0, "found": o,
                               "detail": f"unexpected {name} (found {o})"})
        elif e != o:
            mismatches.append({"type": "wrong_count", "component": name,
                               "expected": e, "found": o,
                               "detail": f"{name}: found {o}, expected {e}"})

    all_w = set(exp_w) | set(obs_w)
    for color in sorted(all_w):
        e, o = exp_w.get(color, 0), obs_w.get(color, 0)
        if e and not o:
            mismatches.append({"type": "missing_wire_color", "color": color,
                               "expected": e, "found": 0,
                               "detail": f"no {color} wires (expected {e})"})
        elif o and not e:
            mismatches.append({"type": "unexpected_wire_color", "color": color,
                               "expected": 0, "found": o,
                               "detail": f"unexpected {color} wires (found {o})"})

    has_component_model = not any(
        "no trained component model" in n for n in observed.get("notes", []))

    if not has_component_model and not exp_c:
        status = "warning"
    elif not mismatches:
        status = "pass"
    else:
        # missing/extra/wrong components are failures; wire-colour diffs alone warn
        hard = [m for m in mismatches if m["type"] in (
            "missing_component", "extra_component", "wrong_count")]
        status = "fail" if hard else "warning"

    return {
        "status": status,
        "n_mismatches": len(mismatches),
        "mismatches": mismatches,
        "expected": {"component_counts": exp_c, "wire_color_counts": exp_w},
        "observed": {"component_counts": obs_c, "wire_color_counts": obs_w},
        "component_model_loaded": has_component_model,
    }
