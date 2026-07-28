"""
Panel understanding: type inference, nameplate reading, expert annotation,
reporting and the evaluation metrics.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rtsp_backend.electrical import (expert, inspector, metrics as em,
                                     nameplate, panel_type as ptype,
                                     postprocess as pp, taxonomy as tax)

SHAPE = (768, 1024)


def c(cid, score, box, **kw):
    return pp.Candidate(cid, score, box, **kw)


# ==========================================================================
# panel type inference
# ==========================================================================

def test_motor_control_center_recognised():
    cl = ptype.classify({"contactor": 4, "overload_relay": 4, "mccb": 2,
                         "push_button": 6, "indicator_lamp": 6, "mcb": 4})
    assert cl.panel_type == "motor_control_center"
    assert cl.confidence > ptype.MIN_CLASSIFY_SCORE
    assert any("Contactor" in e for e in cl.evidence)
    assert "motor" in cl.function.lower()


def test_distribution_panel_recognised_and_not_confused_with_mcc():
    cl = ptype.classify({"mcb": 24, "rccb": 4, "busbar": 3, "neutral_bar": 1,
                         "earth_bar": 1, "din_rail": 4})
    assert cl.panel_type == "distribution_panel"
    names = [x.id for x in cl.candidates]
    assert "motor_control_center" not in names[:1]


def test_ats_panel_recognised():
    cl = ptype.classify({"changeover_switch": 1, "ats_controller": 1,
                         "mccb": 2, "indicator_lamp": 4})
    assert cl.panel_type == "automatic_transfer_switch"


def test_plc_cabinet_recognised():
    cl = ptype.classify({"plc": 1, "io_module": 5, "power_supply": 2,
                         "relay": 8, "hmi": 1, "ethernet_switch": 1})
    assert cl.panel_type == "plc_automation_cabinet"


def test_too_little_evidence_refuses_to_classify():
    cl = ptype.classify({"terminal_block": 1})
    assert cl.panel_type == ptype.UNCLASSIFIED
    assert cl.reason == "insufficient_evidence"
    assert cl.confidence == 0.0


def test_empty_inventory_refuses():
    cl = ptype.classify({})
    assert cl.panel_type == ptype.UNCLASSIFIED


def test_unknown_components_are_not_evidence():
    """Unknown detections say nothing about panel type and must not vote."""
    a = ptype.classify({"contactor": 4, "overload_relay": 4, "mccb": 2})
    b = ptype.classify({"contactor": 4, "overload_relay": 4, "mccb": 2,
                        tax.UNKNOWN_COMPONENT_ID: 30})
    assert a.panel_type == b.panel_type
    assert a.confidence == pytest.approx(b.confidence)


def test_candidates_are_ranked_and_serialisable():
    cl = ptype.classify({"vfd": 3, "line_reactor": 2, "cooling_fan": 2,
                         "mccb": 3})
    confs = [x.confidence for x in cl.candidates]
    assert confs == sorted(confs, reverse=True)
    json.dumps(cl.to_dict())


# ==========================================================================
# application inference
# ==========================================================================

def test_application_from_panel_text():
    g = ptype.infer_application({"contactor": 3, "overload_relay": 3},
                                ["PUMP 1 DUTY", "PUMP 2 STANDBY"])
    assert g.application == "pumping"
    assert g.confidence > 0


def test_application_from_composition_only():
    g = ptype.infer_application({"changeover_switch": 1, "ats_controller": 1})
    assert g.application == "standby generation"


def test_application_admits_ignorance():
    g = ptype.infer_application({"terminal_block": 4}, [])
    assert g.application is None
    assert g.evidence


# ==========================================================================
# missing components / maintenance notes
# ==========================================================================

def test_missing_components_for_known_type():
    missing = ptype.missing_components("motor_control_center",
                                       {"contactor": 3, "mcb": 2})
    ids = {m.class_id for m in missing}
    assert "overload_relay" in ids
    assert "earth_bar" in ids
    assert all(m.rationale for m in missing)
    assert any(m.severity == "important" for m in missing)


def test_missing_components_unknown_type_is_empty():
    assert ptype.missing_components(ptype.UNCLASSIFIED, {"mcb": 3}) == []


def test_contactor_without_overload_is_flagged():
    notes = ptype.maintenance_notes({"contactor": 3, "overload_relay": 1},
                                    "motor_control_center", 0, 4)
    codes = {n.code for n in notes}
    assert "starter_protection_mismatch" in codes


def test_vfd_without_cooling_flagged():
    notes = ptype.maintenance_notes({"vfd": 2, "mccb": 2}, "vfd_drive_panel",
                                    0, 4)
    assert "drive_thermal_management" in {n.code for n in notes}


def test_ct_without_meter_flagged_as_important():
    notes = ptype.maintenance_notes({"current_transformer": 3},
                                    "metering_panel", 0, 3)
    hit = [n for n in notes if n.code == "ct_secondary_unloaded"]
    assert hit and hit[0].severity == "important"


def test_estop_without_safety_relay_flagged():
    notes = ptype.maintenance_notes({"emergency_stop": 2, "contactor": 2},
                                    "safety_control_panel", 0, 4)
    assert "estop_not_monitored" in {n.code for n in notes}


def test_high_unknown_ratio_is_reported_as_important():
    notes = ptype.maintenance_notes({"contactor": 2}, "motor_control_center",
                                    unknown_count=6, accepted_total=10)
    hit = [n for n in notes if n.code == "low_recognition_confidence"]
    assert hit and hit[0].severity == "important"


def test_no_spurious_notes_on_a_coherent_panel():
    counts = {"contactor": 3, "overload_relay": 3, "mcb": 4, "power_supply": 1,
              "terminal_block": 4, "wire_duct": 2, "earth_bar": 1,
              "push_button": 4, "indicator_lamp": 4}
    notes = ptype.maintenance_notes(counts, "motor_control_center", 0, 26)
    assert [n.code for n in notes] == []


# ==========================================================================
# nameplate identification
# ==========================================================================

@pytest.mark.parametrize("text,manufacturer,cid", [
    ("LC1D32 24VDC", "Schneider Electric", "contactor"),
    ("LRD 32 SCHNEIDER", "Schneider Electric", "overload_relay"),
    ("SIEMENS 3RT2027-1AL20", "Siemens", "contactor"),
    ("SIMATIC S7-1200 CPU 1214C", "Siemens", "plc"),
    ("ABB ACS580-01", "ABB", "vfd"),
    ("ATV320U15N4", "Schneider Electric", "vfd"),
    ("PNOZ X3", "Pilz", "safety_relay"),
    ("MDR-60-24", "MEAN WELL", "power_supply"),
    ("Compact NSX 250", "Schneider Electric", "mccb"),
    ("H3CR-A8", "Omron", "timer_relay"),
])
def test_part_numbers_identified(text, manufacturer, cid):
    m = nameplate.identify(text, cid)
    assert m.manufacturer == manufacturer
    assert m.implied_class == cid
    assert m.agrees_with_detector is True
    assert m.part_number


def test_brand_only_text_still_yields_manufacturer():
    m = nameplate.identify("SCHNEIDER ELECTRIC MADE IN FRANCE", "contactor")
    assert m.manufacturer == "Schneider Electric"
    assert m.part_number is None


def test_nameplate_disagreement_is_reported_not_hidden():
    m = nameplate.identify("ABB ACS580-01", "contactor")
    assert m.implied_class == "vfd"
    assert m.agrees_with_detector is False


def test_no_text_yields_nothing():
    m = nameplate.identify("", "contactor")
    assert m.manufacturer is None and m.part_number is None
    m2 = nameplate.identify("220V 50Hz IP65", "contactor")
    assert m2.manufacturer is None


def test_text_assigned_to_containing_box():
    boxes = [(100, 100, 200, 200), (300, 100, 400, 200)]
    items = [
        {"text": "LC1D32", "bbox": (120, 130, 180, 150)},
        {"text": "S7-1200", "bbox": (320, 130, 380, 150)},
        {"text": "stray", "bbox": (240, 130, 270, 150)},   # between boxes
    ]
    out = nameplate.text_for_boxes(items, boxes)
    assert out[0] == "LC1D32"
    assert out[1] == "S7-1200"
    assert "stray" not in " ".join(out)


def test_catalogue_summary():
    s = nameplate.catalogue_summary()
    assert s["signature_count"] > 50
    assert "Schneider Electric" in s["manufacturers"]
    assert "Siemens" in s["manufacturers"]


# ==========================================================================
# expert annotation
# ==========================================================================

def test_annotate_produces_full_engineering_record():
    cands = [c("contactor", 0.91, (100, 100, 180, 200))]
    res = pp.run(cands, SHAPE)
    findings = expert.annotate(res.accepted, SHAPE, res.rows, None)
    f = findings[0]
    assert f.class_id == "contactor"
    assert f.function and f.purpose and f.category == "switching"
    assert f.position == "top-left"
    assert f.row == 1 and f.row_position == 1
    assert f.center == pytest.approx((140.0, 150.0))
    json.dumps(f.to_dict())


def test_purpose_is_context_sensitive():
    """The same device means different things depending on its neighbours."""
    with_overload = pp.run([
        c("contactor", 0.9, (100, 100, 180, 205)),
        c("overload_relay", 0.9, (100, 200, 180, 270)),
    ], SHAPE)
    f1 = expert.annotate(with_overload.accepted, SHAPE, with_overload.rows)
    starter = next(x for x in f1 if x.class_id == "contactor")
    assert "motor starter" in starter.purpose.lower()

    with_caps = pp.run([
        c("contactor", 0.9, (100, 100, 180, 205)),
        c("capacitor", 0.9, (100, 210, 190, 340)),
    ], SHAPE)
    f2 = expert.annotate(with_caps.accepted, SHAPE, with_caps.rows)
    cap_switch = next(x for x in f2 if x.class_id == "contactor")
    assert "power-factor" in cap_switch.purpose.lower()


def test_nameplate_corroboration_raises_confidence():
    cands = [c("contactor", 0.70, (100, 100, 180, 200))]
    res = pp.run(cands, SHAPE)
    ocr = [{"text": "LC1D32", "bbox": (120, 140, 170, 160)}]
    f = expert.annotate(res.accepted, SHAPE, res.rows, ocr)[0]
    assert f.manufacturer == "Schneider Electric"
    assert f.part_number == "LC1D32"
    assert f.confidence > 0.70
    assert f.confidence < 1.0
    assert f.identification_basis == "detector+nameplate"
    assert "LC1D32" in f.display_title()


def test_unknown_component_is_labelled_honestly():
    cfg = pp.GateConfig()
    score = cfg.threshold_for("contactor") - 0.05
    res = pp.run([c("contactor", score, (100, 100, 180, 200))], SHAPE)
    f = expert.annotate(res.accepted, SHAPE, res.rows)[0]
    assert f.is_unknown
    assert f.label == tax.UNKNOWN_COMPONENT_NAME
    assert "manual identification" in f.purpose.lower()
    assert any("threshold" in n for n in f.notes)


def test_quantities_rollup():
    res = pp.run([
        c("mcb", 0.8, (100, 100, 120, 160)),
        c("mcb", 0.7, (125, 100, 145, 160)),
        c("plc", 0.9, (400, 100, 600, 200)),
    ], SHAPE)
    f = expert.annotate(res.accepted, SHAPE, res.rows)
    bom = expert.quantities(f)
    by_id = {b["class_id"]: b for b in bom}
    assert by_id["mcb"]["quantity"] == 2
    assert by_id["mcb"]["mean_confidence"] == pytest.approx(0.75)
    assert by_id["plc"]["quantity"] == 1
    assert bom[0]["quantity"] >= bom[-1]["quantity"]


def test_layout_description():
    res = pp.run([
        c("mcb", 0.8, (100, 100, 120, 160)),
        c("mcb", 0.8, (125, 100, 145, 160)),
        c("relay", 0.8, (100, 400, 140, 460)),
    ], SHAPE)
    f = expert.annotate(res.accepted, SHAPE, res.rows)
    lines = expert.layout_description(f, res.rows)
    assert len(lines) == 2
    assert lines[0].startswith("Row 1:")
    assert "2×" in lines[0]


# ==========================================================================
# inspection engine + report
# ==========================================================================

class FakeRecognizer:
    """A recogniser stand-in returning a fixed, plausible MCC inventory."""

    ready = True
    backend_id = "fake"

    def recognize(self, frame):
        cands = [
            c("contactor", 0.93, (100, 100, 180, 200)),
            c("contactor", 0.90, (200, 100, 280, 200)),
            c("contactor", 0.88, (300, 100, 380, 200)),
            c("overload_relay", 0.87, (100, 205, 180, 272)),
            c("overload_relay", 0.85, (200, 205, 280, 272)),
            c("overload_relay", 0.84, (300, 205, 380, 272)),
            c("mccb", 0.80, (600, 100, 760, 260)),
            c("mcb", 0.78, (100, 400, 122, 460)),
            c("mcb", 0.77, (126, 400, 148, 460)),
            c("power_supply", 0.75, (300, 400, 400, 480)),
            c("push_button", 0.70, (700, 420, 730, 450)),
            c("indicator_lamp", 0.68, (750, 420, 775, 445)),
        ]
        return pp.run(cands, frame.shape[:2])


def _blank(h=768, w=1024):
    return np.full((h, w, 3), 60, np.uint8)


def test_inspect_panel_full_result():
    res = inspector.inspect_panel(FakeRecognizer(), _blank(), read_text=False)
    assert res["component_total"] == 12
    assert res["panel"]["panel_type"] == "motor_control_center"
    assert res["layout"]["rows"] >= 2
    assert res["bill_of_materials"]
    assert res["confidence"]["count"] == 12
    assert res["wire_analysis"]["enabled"] is False
    assert res["duration_ms"] >= 0
    assert "_annotated" in res
    assert res["_annotated"].shape == (768, 1024, 3)
    json.dumps({k: v for k, v in res.items() if k != "_annotated"}, default=str)


def test_inspect_panel_with_no_backend_is_honest():
    res = inspector.inspect_panel(None, _blank(), read_text=False)
    assert res["component_total"] == 0
    assert res["components"] == []
    assert any("recognition backend" in n for n in res["notes"])
    assert res["panel"]["panel_type"] == ptype.UNCLASSIFIED


def test_inspect_panel_with_unready_backend_reports_reason():
    class NotReady:
        ready = False
        _error = "weights missing: train a detector first"

    res = inspector.inspect_panel(NotReady(), _blank(), read_text=False)
    assert res["component_total"] == 0
    assert any("weights missing" in n for n in res["notes"])


def test_inspect_panel_survives_a_throwing_backend():
    class Boom:
        ready = True

        def recognize(self, frame):
            raise RuntimeError("cuda exploded")

    res = inspector.inspect_panel(Boom(), _blank(), read_text=False)
    assert res["component_total"] == 0
    assert any("cuda exploded" in n for n in res["notes"])


def test_legacy_detector_interface_is_adapted():
    from rtsp_backend.ai.base import BBox, Detection

    class Legacy:
        ready = True
        backend_id = "legacy"

        def infer(self, frame):
            return [Detection(label="Contactor", confidence=0.9,
                              bbox=BBox(100, 100, 180, 200), kind="component")]

    res = inspector.inspect_panel(Legacy(), _blank(), read_text=False)
    assert res["component_total"] == 1
    assert res["components"][0]["class_id"] == "contactor"


def test_report_has_every_required_section():
    res = inspector.inspect_panel(FakeRecognizer(), _blank(), read_text=False)
    rep = inspector.build_report(res)
    for key in ("inspection_summary", "panel_type", "detected_components",
                "component_count", "possible_function",
                "possible_missing_components", "potential_maintenance_notes",
                "confidence_statistics", "inspection_time"):
        assert key in rep, key
    assert rep["generator"] == "Madkour AI Panel Inspector"
    assert rep["component_count"]["total"] == 12
    assert rep["panel_type"]["name"] == "Motor Control Center (MCC)"
    assert rep["confidence_statistics"]["detection_gate"]["input_count"] == 12


def test_report_text_renders_all_sections():
    res = inspector.inspect_panel(FakeRecognizer(), _blank(), read_text=False)
    text = inspector.report_text(inspector.build_report(res))
    for heading in ("INSPECTION SUMMARY", "PANEL TYPE", "DETECTED COMPONENTS",
                    "COMPONENT COUNT", "POSSIBLE FUNCTION",
                    "POSSIBLE MISSING COMPONENTS",
                    "POTENTIAL MAINTENANCE NOTES", "CONFIDENCE STATISTICS",
                    "INSPECTION TIME", "WIRING ANALYSIS"):
        assert heading in text, heading
    assert "Motor Control Center" in text


def test_report_text_on_empty_result():
    res = inspector.inspect_panel(None, _blank(), read_text=False)
    text = inspector.report_text(inspector.build_report(res))
    assert "No industrial components were recognised" in text
    assert "(none)" in text


def test_overlay_draws_and_stays_in_bounds():
    res = inspector.inspect_panel(FakeRecognizer(), _blank(), read_text=False)
    img = res["_annotated"]
    assert img.shape == (768, 1024, 3)
    # something was actually drawn
    assert not np.all(img == 60)


def test_overlay_label_near_top_edge_does_not_crash():
    class TopEdge:
        ready = True

        def recognize(self, frame):
            return pp.run([c("contactor", 0.9, (0, 0, 90, 80))],
                          frame.shape[:2])

    res = inspector.inspect_panel(TopEdge(), _blank(), read_text=False)
    assert res["component_total"] == 1


# ==========================================================================
# metrics
# ==========================================================================

def _gt(image_id, cid, box):
    return {"image_id": image_id, "class_id": cid, "box": box}


def _pr(image_id, cid, box, score):
    return {"image_id": image_id, "class_id": cid, "box": box, "score": score}


def test_perfect_predictions_score_one():
    gts = [_gt("a", "contactor", (0, 0, 10, 10)),
           _gt("a", "mcb", (20, 0, 30, 20))]
    preds = [_pr("a", "contactor", (0, 0, 10, 10), 0.9),
             _pr("a", "mcb", (20, 0, 30, 20), 0.8)]
    rep = em.evaluate(gts, preds)
    assert rep["overall"]["precision"] == 1.0
    assert rep["overall"]["recall"] == 1.0
    assert rep["overall"]["f1"] == 1.0
    assert rep["map_50"] == pytest.approx(1.0)


def test_spurious_prediction_lowers_precision_only():
    gts = [_gt("a", "contactor", (0, 0, 10, 10))]
    preds = [_pr("a", "contactor", (0, 0, 10, 10), 0.9),
             _pr("a", "contactor", (500, 500, 520, 520), 0.5)]
    rep = em.evaluate(gts, preds)
    assert rep["overall"]["recall"] == 1.0
    assert rep["overall"]["precision"] == pytest.approx(0.5)
    fp = rep["false_positive_analysis"]
    assert fp["total"] == 1
    assert fp["by_cause"]["spurious_detection"] == 1


def test_class_confusion_is_diagnosed_as_such():
    gts = [_gt("a", "mcb", (0, 0, 20, 60))]
    preds = [_pr("a", "mccb", (0, 0, 20, 60), 0.9)]
    rep = em.evaluate(gts, preds)
    fp = rep["false_positive_analysis"]
    assert fp["by_cause"]["class_confusion"] == 1
    assert rep["false_negative_analysis"]["total"] == 1


def test_localisation_error_is_diagnosed():
    gts = [_gt("a", "mcb", (0, 0, 100, 100))]
    # IoU ≈ 0.22 — overlapping the right device but too loosely to count as a hit
    preds = [_pr("a", "mcb", (40, 40, 140, 140), 0.9)]
    rep = em.evaluate(gts, preds)
    assert rep["false_positive_analysis"]["by_cause"]["localisation"] == 1


def test_missed_detection_lowers_recall_only():
    gts = [_gt("a", "contactor", (0, 0, 10, 10)),
           _gt("a", "contactor", (50, 50, 60, 60))]
    preds = [_pr("a", "contactor", (0, 0, 10, 10), 0.9)]
    rep = em.evaluate(gts, preds)
    assert rep["overall"]["precision"] == 1.0
    assert rep["overall"]["recall"] == pytest.approx(0.5)


def test_average_precision_hand_computable():
    # 2 ground truths, predictions: TP, FP, TP
    assert em.average_precision([1, 0, 1], [0.9, 0.8, 0.7], 2) == pytest.approx(
        (1.0 * 0.5) + (2 / 3 * 0.5), abs=1e-6)
    assert em.average_precision([], [], 3) == 0.0
    assert em.average_precision([1], [0.9], 0) == 0.0


def test_confusion_matrix_includes_background():
    gts = [_gt("a", "mcb", (0, 0, 20, 60)),
           _gt("a", "contactor", (100, 0, 180, 100))]
    preds = [_pr("a", "mccb", (0, 0, 20, 60), 0.9),          # confusion
             _pr("a", "relay", (500, 500, 540, 560), 0.6)]   # background FP
    cm = em.confusion_matrix(gts, preds)
    assert "__background__" in cm["labels"]
    labels = cm["labels"]
    mat = cm["matrix"]
    bg = labels.index("__background__")
    # mcb was predicted as mccb
    assert mat[labels.index("mcb")][labels.index("mccb")] == 1
    # contactor was missed -> background column
    assert mat[labels.index("contactor")][bg] == 1
    # relay predicted on nothing -> background row
    assert mat[bg][labels.index("relay")] == 1
    assert cm["per_class_accuracy"]["mcb"] == 0.0


def test_threshold_optimiser_finds_the_separating_threshold():
    gts = [_gt(f"i{i}", "contactor", (0, 0, 10, 10)) for i in range(5)]
    preds = [_pr(f"i{i}", "contactor", (0, 0, 10, 10), 0.9) for i in range(5)]
    # add junk that only appears below 0.4
    preds += [_pr(f"i{i}", "contactor", (500, 500, 510, 510), 0.3)
              for i in range(5)]
    rec = em.optimise_thresholds(gts, preds)
    assert rec["contactor"]["recommended_threshold"] >= 0.35
    assert rec["contactor"]["at_recommended"]["precision"] == pytest.approx(1.0)


def test_threshold_optimiser_respects_precision_floor():
    gts = [_gt("a", "mcb", (0, 0, 20, 60))]
    preds = [_pr("a", "mcb", (0, 0, 20, 60), 0.9),
             _pr("a", "mcb", (300, 300, 320, 360), 0.85)]
    rec = em.optimise_thresholds(gts, preds, min_precision=0.99)
    best = rec["mcb"]["at_recommended"]
    assert best is None or best["precision"] >= 0.99


def test_compare_models_ranks_by_map():
    good = em.evaluate([_gt("a", "mcb", (0, 0, 20, 60))],
                       [_pr("a", "mcb", (0, 0, 20, 60), 0.9)])
    bad = em.evaluate([_gt("a", "mcb", (0, 0, 20, 60))],
                      [_pr("a", "mcb", (300, 0, 320, 60), 0.9)])
    cmp = em.compare_models({"bad": bad, "good": good})
    assert cmp["winner"] == "good"
    table = em.format_table(cmp)
    assert "good" in table and "mAP50" in table


def test_metrics_on_empty_inputs():
    rep = em.evaluate([], [])
    assert rep["overall"]["f1"] == 0.0
    assert rep["map_50"] == 0.0
    assert em.format_table(em.compare_models({})) == "(no models evaluated)"
