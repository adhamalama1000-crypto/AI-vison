"""
Expert annotation: turn a bounding box into an engineering statement.

A detection on its own — ``contactor 0.91 [412, 233, 498, 341]`` — is not an
inspection result. What an engineer wants to read is:

    Schneider Electric TeSys D contactor (LC1D32)
    Confidence 91%  ·  row 2, position 3  ·  centre (455, 287)
    Function: switches the three-phase motor circuit under coil control.
    Estimated purpose: motor starter switching element.

This module builds that record from three independent sources — the detector's
class, the taxonomy's engineering knowledge, and any nameplate text read inside
the box — and is explicit about which source contributed what. When the sources
disagree it says so rather than quietly picking one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from . import nameplate as np_mod
from . import postprocess as pp
from . import taxonomy as tax


@dataclass
class ComponentFinding:
    """One fully-annotated component in the inspection result."""

    index: int
    class_id: str
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]
    position: str
    row: Optional[int] = None
    row_position: Optional[int] = None
    category: str = ""
    domain: str = ""
    function: str = ""
    purpose: str = ""
    mounting: tuple[str, ...] = ()
    manufacturer: Optional[str] = None
    product_family: Optional[str] = None
    part_number: Optional[str] = None
    nameplate_text: Optional[str] = None
    #: "detector" | "detector+nameplate" | "detector (nameplate disagrees)"
    identification_basis: str = "detector"
    notes: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def is_unknown(self) -> bool:
        return self.class_id == tax.UNKNOWN_COMPONENT_ID

    def display_title(self) -> str:
        """Human title, e.g. 'Schneider Electric TeSys D contactor (LC1D32)'."""
        bits: list[str] = []
        if self.manufacturer:
            bits.append(self.manufacturer)
        if self.product_family:
            bits.append(self.product_family)
        else:
            bits.append(self.label)
        title = " ".join(bits)
        if self.part_number and self.part_number.lower() not in title.lower():
            title = f"{title} ({self.part_number})"
        return title

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "class_id": self.class_id,
            "label": self.label,
            "title": self.display_title(),
            "confidence": round(float(self.confidence), 4),
            "confidence_pct": round(float(self.confidence) * 100.0, 1),
            "bbox": [round(float(v), 1) for v in self.bbox],
            "center": [round(float(v), 1) for v in self.center],
            "position": self.position,
            "row": self.row,
            "row_position": self.row_position,
            "category": self.category,
            "domain": self.domain,
            "function": self.function,
            "purpose": self.purpose,
            "mounting": list(self.mounting),
            "manufacturer": self.manufacturer,
            "product_family": self.product_family,
            "part_number": self.part_number,
            "nameplate_text": self.nameplate_text,
            "identification_basis": self.identification_basis,
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }


# --------------------------------------------------------------------------
# Purpose inference — the "estimated purpose" line
# --------------------------------------------------------------------------

def _purpose(class_id: str, neighbours: Sequence[str]) -> str:
    """Estimate a component's role from what sits around it.

    Context changes meaning: a contactor next to an overload relay is a motor
    starter; the same contactor next to capacitors is a capacitor switching
    stage. This is exactly the reasoning that makes an inspection read like it
    was written by someone who understands panels.
    """
    near = set(neighbours)
    sp = tax.spec(class_id)

    if class_id == "contactor":
        if "overload_relay" in near or "motor_starter" in near:
            return "Motor starter switching element (with thermal overload protection)."
        if "capacitor" in near:
            return "Capacitor stage switching contactor for power-factor correction."
        if "timer_relay" in near:
            return "Timed load switching — lighting or star-delta changeover duty."
        return "Load switching element."
    if class_id == "overload_relay":
        return "Thermal protection of the motor fed by the contactor above it."
    if class_id == "mcb":
        if "power_supply" in near or "plc" in near:
            return "Control-circuit supply protection."
        return "Final-circuit overcurrent protection."
    if class_id == "mccb":
        if "vfd" in near or "soft_starter" in near:
            return "Drive input protection and isolation."
        return "Feeder protection and isolation."
    if class_id == "acb":
        return "Main incoming supply protection."
    if class_id == "vfd":
        if "line_reactor" in near:
            return "Variable-speed motor control with harmonic mitigation."
        return "Variable-speed motor control."
    if class_id == "plc":
        if "hmi" in near:
            return "Automation controller with an operator interface."
        return "Automation controller executing the panel's control program."
    if class_id == "io_module":
        return "Field signal interface for the adjacent controller."
    if class_id == "power_supply":
        return "24 V DC control supply for the controller, relays and sensors."
    if class_id == "relay":
        if "plc" in near or "io_module" in near:
            return "Interposing relay between the controller output and the field."
        return "Auxiliary control switching."
    if class_id == "safety_relay":
        return "Functional-safety evaluation of the emergency-stop circuit."
    if class_id == "timer_relay":
        if "contactor" in near:
            return "Sequencing the adjacent contactor (star-delta or delayed start)."
        return "Control sequence timing."
    if class_id == "terminal_block":
        return "Field wiring termination and distribution."
    if class_id == "current_transformer":
        if "energy_meter" in near or "ammeter" in near:
            return "Current measurement input to the adjacent meter."
        return "Current measurement or protection input."
    if class_id == "emergency_stop":
        return "Operator emergency-stop command into the safety circuit."
    if class_id == "indicator_lamp":
        return "State annunciation for the operator."
    if class_id == "push_button":
        return "Manual operator command."
    if class_id == "selector_switch":
        return "Operating-mode selection (e.g. Manual / Off / Auto)."
    if class_id == "cooling_fan":
        return "Enclosure ventilation removing device heat."
    if class_id == tax.UNKNOWN_COMPONENT_ID:
        return ("Unidentified device — requires manual identification. Add this "
                "crop to the training set.")
    # Fall back to the taxonomy's declared role.
    return sp.role


def _neighbour_classes(findings_boxes: Sequence[Sequence[float]],
                       classes: Sequence[str], i: int,
                       radius_factor: float = 1.8) -> list[str]:
    """Classes whose box centre lies within a few device-widths of box ``i``."""
    if not findings_boxes:
        return []
    x1, y1, x2, y2 = findings_boxes[i]
    w, h = max(1.0, x2 - x1), max(1.0, y2 - y1)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    rx, ry = w * radius_factor, h * radius_factor
    out: list[str] = []
    for j, box in enumerate(findings_boxes):
        if j == i:
            continue
        ox, oy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
        if abs(ox - cx) <= rx + w and abs(oy - cy) <= ry + h:
            out.append(classes[j])
    return out


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

def annotate(candidates: Sequence[pp.Candidate], image_shape: Sequence[int],
             rows: Optional[Sequence[Sequence[int]]] = None,
             ocr_items: Optional[Sequence[dict]] = None) -> list[ComponentFinding]:
    """Build the expert findings list from gated candidates.

    ``rows`` is the row grouping from :func:`~.postprocess.group_rows` (indices
    into ``candidates``); ``ocr_items`` are whole-image OCR results, which get
    assigned to the box that contains them.
    """
    boxes = [tuple(float(v) for v in c.box) for c in candidates]
    classes = [c.class_id for c in candidates]

    row_of: dict[int, tuple[int, int]] = {}
    for r_i, row in enumerate(rows or []):
        for pos, idx in enumerate(row):
            row_of[idx] = (r_i + 1, pos + 1)

    texts = (np_mod.text_for_boxes(ocr_items, boxes)
             if ocr_items else ["" for _ in boxes])

    findings: list[ComponentFinding] = []
    for i, cand in enumerate(candidates):
        sp = tax.spec(cand.class_id)
        plate = np_mod.identify(texts[i], cand.class_id)
        neighbours = _neighbour_classes(boxes, classes, i)

        basis = "detector"
        notes: list[str] = []
        confidence = float(cand.score)

        if plate.manufacturer and plate.part_number:
            if plate.agrees_with_detector:
                basis = "detector+nameplate"
                # Corroboration from an independent source justifiably raises
                # stated confidence, but never to certainty.
                confidence = min(0.995, confidence + (1.0 - confidence) * 0.5)
                notes.append(
                    f"Nameplate '{plate.part_number}' corroborates the detected "
                    f"class ({sp.name}).")
            else:
                basis = "detector (nameplate disagrees)"
                notes.append(
                    f"Nameplate '{plate.part_number}' indicates "
                    f"{tax.display_name(plate.implied_class or '')}, while the "
                    f"detector reported {sp.name}. Both are reported; verify "
                    f"manually.")
        elif plate.manufacturer:
            basis = "detector+brand"
            notes.append(plate.note or "Manufacturer read from the nameplate.")

        if sp.notes:
            notes.append(sp.notes)
        if cand.class_id == tax.UNKNOWN_COMPONENT_ID:
            reason = cand.extra.get("demotion_reason")
            if reason:
                notes.append(str(reason))

        label = sp.name
        if plate.implied_class and plate.agrees_with_detector is False:
            label = f"{sp.name} / possibly {tax.display_name(plate.implied_class)}"

        row_no, row_pos = row_of.get(i, (None, None))
        findings.append(ComponentFinding(
            index=i,
            class_id=cand.class_id,
            label=label,
            confidence=confidence,
            bbox=boxes[i],
            center=cand.center,
            position=pp.panel_position(boxes[i], image_shape),
            row=row_no, row_position=row_pos,
            category=sp.category, domain=sp.domain,
            function=sp.function,
            purpose=_purpose(cand.class_id, neighbours),
            mounting=sp.mounting,
            manufacturer=plate.manufacturer,
            product_family=plate.family,
            part_number=plate.part_number,
            nameplate_text=(plate.text or None),
            identification_basis=basis,
            notes=notes,
            extra={**dict(cand.extra), "source": cand.source,
                   "detector_score": round(float(cand.score), 4),
                   "neighbours": sorted(set(neighbours))},
        ))
    return findings


def quantities(findings: Sequence[ComponentFinding]) -> list[dict]:
    """Bill-of-materials style rollup: class, quantity, mean confidence."""
    groups: dict[str, list[ComponentFinding]] = {}
    for f in findings:
        groups.setdefault(f.class_id, []).append(f)
    out: list[dict] = []
    for cid, items in groups.items():
        sp = tax.spec(cid)
        confs = [f.confidence for f in items]
        makers = sorted({f.manufacturer for f in items if f.manufacturer})
        out.append({
            "class_id": cid,
            "name": sp.name,
            "category": sp.category,
            "quantity": len(items),
            "mean_confidence": round(sum(confs) / len(confs), 4),
            "min_confidence": round(min(confs), 4),
            "max_confidence": round(max(confs), 4),
            "manufacturers": makers,
            "function": sp.function,
            "indices": [f.index for f in items],
        })
    # Countable devices first, then structural, each by descending quantity.
    countable = tax.countable_classes()
    out.sort(key=lambda d: (d["class_id"] not in countable, -d["quantity"],
                            d["name"]))
    return out


def layout_description(findings: Sequence[ComponentFinding],
                       rows: Sequence[Sequence[int]]) -> list[str]:
    """One sentence per detected row — the panel's physical structure."""
    lines: list[str] = []
    for r_i, row in enumerate(rows or []):
        names: list[str] = []
        counts: dict[str, int] = {}
        for idx in row:
            if idx < len(findings):
                cid = findings[idx].class_id
                counts[cid] = counts.get(cid, 0) + 1
        for cid, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            names.append(f"{n}× {tax.display_name(cid)}" if n > 1
                         else tax.display_name(cid))
        if names:
            lines.append(f"Row {r_i + 1}: " + ", ".join(names))
    return lines


__all__ = ["ComponentFinding", "annotate", "quantities", "layout_description"]
