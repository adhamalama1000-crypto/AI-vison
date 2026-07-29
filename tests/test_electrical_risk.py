"""
Panel risk assessment.

Somebody may decide not to open a cabinet because of this output, so the tests here
are mostly about what the module refuses to say.

The dangerous failure mode is not a wrong number — it is a confident ``low`` derived
from an inspection that did not happen. "We found nothing wrong" and "we could not
look" read identically to a human skimming a report while meaning opposite things, so
every no-basis case must come back ``unknown`` and say why.
"""

from __future__ import annotations

import pytest

from rtsp_backend.electrical import risk


# ==========================================================================
# helpers
# ==========================================================================

def _components(**counts):
    out = []
    for cid, n in counts.items():
        out.extend([{"class_id": cid, "confidence": 0.9}] * n)
    return out


def _result(components=None, missing=None, notes=None, unknown=0,
            mean_conf=0.88, panel_type="motor_control_centre",
            model_loaded=True):
    comps = components if components is not None else _components(
        mcb=8, contactor=3, overload_relay=3, earth_bar=1, plc=1)
    return {
        "component_model_loaded": model_loaded,
        "components": comps,
        "confidence": {"unknown": unknown, "mean": mean_conf},
        "missing_components": missing or [],
        "maintenance_notes": notes or [],
        "panel": {"panel_type": panel_type},
    }


def _missing(cid, severity="important"):
    return {"class_id": cid, "severity": severity,
            "rationale": f"{cid} was not detected."}


def _note(code, severity="important", message="something to check"):
    return {"code": code, "severity": severity, "message": message}


# ==========================================================================
# refusal to score — the safety-critical behaviour
# ==========================================================================

def test_no_model_loaded_is_unknown_never_low():
    """A 'low risk' verdict from an inspection that did not happen is the worst
    thing this platform could output."""
    r = risk.assess(_result(components=[], model_loaded=False))
    assert r.level == risk.UNKNOWN_LEVEL
    assert r.level not in risk.LEVELS
    assert r.assessable is False
    assert r.confidence == "none"
    assert "not a low-risk result" in r.headline
    assert r.recommendations, "it must say what to do about it"


def test_no_detections_is_unknown_never_low():
    r = risk.assess(_result(components=[]))
    assert r.level == risk.UNKNOWN_LEVEL
    assert r.assessable is False
    assert "'Nothing found' is not 'nothing wrong'" in r.headline


def test_a_mostly_unidentified_inventory_is_unknown():
    """Scoring the identified minority would understate the uncertainty."""
    comps = (_components(unknown_industrial_component=7)
             + _components(mcb=5))
    r = risk.assess(_result(components=comps, unknown=7))
    assert r.level == risk.UNKNOWN_LEVEL
    assert r.assessable is False
    assert "more than half unknown" in r.headline
    assert any("unidentified" in lim for lim in r.limits)
    assert any("autolabel" in rec for rec in r.recommendations)


def test_unknown_level_is_not_on_the_orderable_scale():
    """It must be impossible to sort 'unknown' as if it were better than 'high'."""
    assert risk.UNKNOWN_LEVEL not in risk.LEVELS
    assert list(risk.LEVELS) == ["low", "moderate", "elevated", "high"]


# ==========================================================================
# scoring
# ==========================================================================

def test_a_clean_panel_scores_low():
    r = risk.assess(_result())
    assert r.assessable is True
    assert r.level == "low"
    assert r.score == 0.0
    assert r.confidence == "high"


def test_a_missing_safety_critical_device_drives_the_level_up():
    clean = risk.assess(_result())
    unsafe = risk.assess(_result(missing=[_missing("emergency_stop")]))
    assert unsafe.score > clean.score
    assert unsafe.level != "low"
    driver = next(d for d in unsafe.drivers if d.class_id == "emergency_stop")
    assert "safety-critical" in driver.message


def test_safety_critical_absences_outweigh_ordinary_ones():
    safety = risk.assess(_result(missing=[_missing("earth_bar")]))
    ordinary = risk.assess(_result(missing=[_missing("indicator_lamp")]))
    assert safety.score > ordinary.score


def test_many_important_findings_reach_high():
    r = risk.assess(_result(
        missing=[_missing("emergency_stop"), _missing("earth_bar"),
                 _missing("overload_relay")],
        notes=[_note("starter_protection_mismatch"),
               _note("drive_thermal_management")]))
    assert r.level == "high"
    assert "needs review" in r.headline


def test_one_important_finding_does_not_condemn_the_panel():
    """A single missing overload relay is worth investigating, not condemning."""
    r = risk.assess(_result(missing=[_missing("overload_relay")]))
    assert r.level in ("moderate", "elevated")
    assert r.level != "high"


def test_severity_ordering_is_respected():
    important = risk.assess(_result(notes=[_note("x", "important")]))
    advisory = risk.assess(_result(notes=[_note("x", "advisory")]))
    info = risk.assess(_result(notes=[_note("x", "info")]))
    assert important.score > advisory.score > info.score


def test_score_thresholds_map_to_the_documented_levels():
    assert risk._level_for(0.0) == "low"
    assert risk._level_for(risk.THRESHOLDS["moderate"]) == "moderate"
    assert risk._level_for(risk.THRESHOLDS["elevated"]) == "elevated"
    assert risk._level_for(risk.THRESHOLDS["high"]) == "high"
    assert risk._level_for(999.0) == "high"


# ==========================================================================
# detection quality bounds the conclusion
# ==========================================================================

def test_a_partly_unidentified_inventory_adds_risk_and_a_limit():
    comps = _components(unknown_industrial_component=3) + _components(mcb=9)
    r = risk.assess(_result(components=comps, unknown=3))
    assert r.assessable is True
    codes = {d.code for d in r.drivers}
    assert "unidentified_devices" in codes
    assert any("unidentified" in lim for lim in r.limits)
    assert r.confidence != "high"


def test_low_mean_confidence_is_itself_a_driver():
    r = risk.assess(_result(mean_conf=0.40))
    assert "low_detection_confidence" in {d.code for d in r.drivers}
    assert any("inventory may be wrong" in lim for lim in r.limits)


def test_too_few_devices_means_absence_is_not_evidence():
    """A photograph of two devices has not shown you the whole cabinet."""
    r = risk.assess(_result(
        components=_components(mcb=1, relay=1),
        missing=[_missing("emergency_stop")]))
    driver = next(d for d in r.drivers if d.class_id == "emergency_stop")
    assert driver.weight == 0.0
    assert "Not scored" in driver.message
    assert r.score == 0.0
    assert any("too few" in lim for lim in r.limits)


def test_a_low_confidence_low_score_does_not_claim_a_clean_panel():
    """The regression: 'consistent with a correctly-populated panel' is a positive
    claim, and it is not supportable from two devices."""
    r = risk.assess(_result(
        components=_components(mcb=1, relay=1),
        missing=[_missing("emergency_stop")]))
    assert r.level == "low"
    assert r.confidence == "low"
    assert "consistent with a correctly-populated panel" not in r.headline
    assert "Do not read this as a pass" in r.headline


def test_a_high_confidence_low_score_may_claim_a_clean_panel():
    r = risk.assess(_result())
    assert r.confidence == "high"
    assert "consistent with a correctly-populated panel" in r.headline


def test_an_unclassified_panel_type_lowers_confidence_and_states_why():
    r = risk.assess(_result(panel_type="unclassified"))
    assert r.confidence != "high"
    assert any("panel type could not be determined" in lim for lim in r.limits)


def test_low_confidence_is_shouted_in_the_headline():
    r = risk.assess(_result(components=_components(mcb=1, relay=1)))
    assert r.confidence == "low"
    assert "LOW" in r.headline


# ==========================================================================
# traceability
# ==========================================================================

def test_every_driver_is_traceable_to_a_finding():
    r = risk.assess(_result(
        missing=[_missing("emergency_stop")],
        notes=[_note("starter_protection_mismatch")]))
    assert r.drivers
    for d in r.drivers:
        assert d.code and d.message
        assert d.category in ("missing_protection", "maintenance",
                              "detection_quality")
        assert d.severity in ("info", "advisory", "important")
        assert d.weight >= 0.0


def test_drivers_are_ordered_by_contribution():
    r = risk.assess(_result(
        missing=[_missing("emergency_stop"), _missing("indicator_lamp",
                                                     "advisory")],
        notes=[_note("x", "info")]))
    weights = [d.weight for d in r.drivers]
    assert weights == sorted(weights, reverse=True)


def test_the_score_is_the_sum_of_its_drivers():
    """No hidden terms — the number must be reconstructable from the JSON."""
    r = risk.assess(_result(
        missing=[_missing("emergency_stop"), _missing("earth_bar")],
        notes=[_note("a"), _note("b", "advisory")]))
    assert r.score == pytest.approx(sum(d.weight for d in r.drivers))


def test_the_assessment_adds_no_findings_of_its_own():
    """It weighs what the rule engine found; it does not invent defects."""
    r = risk.assess(_result())
    assert r.drivers == []
    assert r.score == 0.0


def test_to_dict_is_json_ready_and_carries_the_scale():
    import json

    d = risk.assess(_result(missing=[_missing("earth_bar")])).to_dict()
    json.dumps(d)                                  # must not raise
    assert d["level"] in list(risk.LEVELS) + [risk.UNKNOWN_LEVEL]
    assert d["thresholds"] == risk.THRESHOLDS
    assert d["scale"] == list(risk.LEVELS)
    assert isinstance(d["drivers"][0], dict)


# ==========================================================================
# recommendations
# ==========================================================================

def test_safety_critical_absences_produce_a_priority_recommendation():
    r = risk.assess(_result(missing=[_missing("emergency_stop"),
                                     _missing("earth_bar")]))
    assert any(rec.startswith("PRIORITY") for rec in r.recommendations)
    joined = " ".join(r.recommendations)
    assert "Emergency Stop" in joined


def test_recommendations_always_state_the_photograph_limitation():
    """A photograph cannot see thermal condition, torque or insulation."""
    for r in (risk.assess(_result()),
              risk.assess(_result(missing=[_missing("earth_bar")]))):
        joined = " ".join(r.recommendations)
        assert "thermal" in joined and "torque" in joined


def test_a_clean_panel_still_recommends_physical_inspection():
    r = risk.assess(_result())
    assert r.recommendations
    assert any("not a substitute for a physical inspection" in rec
               for rec in r.recommendations)


def test_recommendations_are_derived_from_the_findings_that_fired():
    """No generic advice that does not match the result."""
    clean = risk.assess(_result())
    assert not any(rec.startswith("PRIORITY") for rec in clean.recommendations)
    assert not any("Confirm presence of" in rec for rec in clean.recommendations)


def test_summary_line_is_readable_for_both_states():
    assessable = risk.summary_line(risk.assess(_result()))
    assert "Risk: LOW" in assessable and "confidence" in assessable
    unknown = risk.summary_line(risk.assess(_result(components=[])))
    assert unknown.startswith("Risk: UNKNOWN")


# ==========================================================================
# integration
# ==========================================================================

def test_inspector_report_includes_the_risk_assessment():
    import numpy as np

    from rtsp_backend.electrical import inspector

    result = inspector.inspect_panel(None, np.zeros((120, 160, 3), np.uint8),
                                     annotate=False, read_text=False)
    report = inspector.build_report(result)
    assert "risk_assessment" in report
    # With no backend there is nothing to assess, and it must say so.
    assert report["risk_assessment"]["level"] == risk.UNKNOWN_LEVEL
    assert report["risk_assessment"]["assessable"] is False


def test_api_exposes_risk_and_recommendations(client):
    import io

    import cv2
    import numpy as np

    img = np.full((240, 320, 3), 120, np.uint8)
    buf = cv2.imencode(".jpg", img)[1].tobytes()
    body = client.post("/api/panel/analyze",
                       files={"file": ("p.jpg", io.BytesIO(buf), "image/jpeg")}
                       ).json()
    assert "risk" in body["report"]
    assert "recommendations" in body["report"]
    if not body["model"]["loaded"]:
        assert body["report"]["risk"]["level"] == risk.UNKNOWN_LEVEL
        assert body["report"]["risk"]["assessable"] is False
