"""
Panel-type and panel-function inference.

Detecting components is only half of what an inspecting engineer does. The other
half is reading the *composition* of the panel and concluding what it is for:
three contactors each with an overload relay under it, fed from an MCCB, with a
Manual/Off/Auto selector on the door, is a motor control centre — nobody needs a
label to say so.

This module encodes that reasoning explicitly as weighted evidence rules over
the detected class counts, so every verdict comes with the evidence that
produced it and can be argued with. It is deliberately *not* a neural net: a
rule set over a component inventory is auditable, needs no training data, and
degrades gracefully when a component is missed.

Three outputs:

* :func:`classify` — ranked panel-type candidates with confidence + evidence.
* :func:`infer_application` — what the panel probably controls (pumps, HVAC,
  conveyors, compressors, lighting, generator), from component mix and any OCR
  text found on the panel.
* :func:`missing_components` / :func:`maintenance_notes` — what a panel of this
  type would normally also contain, and engineering observations worth raising.

All three are honest about uncertainty: with too little evidence,
:func:`classify` returns ``unclassified`` rather than a confident guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

from . import taxonomy as tax

# --------------------------------------------------------------------------
# Panel type rules
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelTypeRule:
    id: str
    name: str
    description: str
    #: Purpose statement used in the report ("Possible Function").
    function: str
    #: class_id -> evidence weight contributed per device (count is capped).
    weights: Mapping[str, float]
    #: Classes whose presence argues *against* this type.
    penalties: Mapping[str, float] = field(default_factory=dict)
    #: Classes without which this type is very unlikely; a rule missing all of
    #: them is heavily discounted rather than eliminated (the detector may have
    #: missed one).
    keystones: tuple[str, ...] = ()
    #: How many of one class still add evidence (a 40-way DB is not 40× an MCC).
    count_cap: int = 4
    #: Components a compliant panel of this type is normally expected to have.
    expected: tuple[str, ...] = ()


_RULES: tuple[PanelTypeRule, ...] = (
    PanelTypeRule(
        id="motor_control_center", name="Motor Control Center (MCC)",
        description="Multiple motor starter columns — switching plus thermal "
                    "protection per motor feeder.",
        function="Starts, stops and protects several three-phase motors, each "
                 "through its own contactor and overload relay fed from a "
                 "protected feeder.",
        weights={"contactor": 3.0, "overload_relay": 3.0, "motor_starter": 2.5,
                 "mccb": 1.2, "push_button": 0.6, "indicator_lamp": 0.5,
                 "selector_switch": 0.6, "busbar": 0.5, "mcb": 0.2,
                 "terminal_block": 0.2, "ammeter": 0.4},
        penalties={"capacitor": 0.8, "changeover_switch": 1.0},
        keystones=("contactor", "overload_relay", "motor_starter"),
        expected=("contactor", "overload_relay", "mcb", "push_button",
                  "indicator_lamp", "terminal_block", "earth_bar"),
    ),
    PanelTypeRule(
        id="vfd_drive_panel", name="VFD / Drive Panel",
        description="One or more variable frequency drives with their input "
                    "protection and thermal management.",
        function="Provides variable-speed control of motors, ramping frequency "
                 "and voltage to match process demand and limit starting current.",
        weights={"vfd": 4.0, "line_reactor": 1.5, "soft_starter": 1.5,
                 "cooling_fan": 0.8, "mccb": 0.8, "mcb": 0.2, "plc": 0.6,
                 "terminal_block": 0.2, "power_supply": 0.4},
        penalties={"capacitor": 0.5},
        keystones=("vfd", "soft_starter", "servo_drive"),
        count_cap=6,
        expected=("vfd", "mccb", "cooling_fan", "terminal_block", "earth_bar",
                  "line_reactor"),
    ),
    PanelTypeRule(
        id="plc_automation_cabinet", name="PLC Automation Cabinet",
        description="Programmable controller with IO, a 24 V control supply and "
                    "interposing relays.",
        function="Executes the automation program for a machine or process, "
                 "reading field signals through IO modules and driving actuators "
                 "via interface relays.",
        weights={"plc": 4.0, "io_module": 2.5, "power_supply": 1.5,
                 "relay": 1.0, "ethernet_switch": 1.2, "hmi": 1.2,
                 "signal_isolator": 0.8, "terminal_block": 0.4,
                 "logic_module": 1.5, "industrial_router": 0.6, "mcb": 0.2},
        penalties={"acb": 1.0, "capacitor": 0.8},
        keystones=("plc", "io_module", "logic_module"),
        count_cap=6,
        expected=("plc", "power_supply", "mcb", "terminal_block",
                  "ethernet_switch", "earth_bar"),
    ),
    PanelTypeRule(
        id="distribution_panel", name="Distribution Panel / Distribution Board",
        description="Rows of modular protective devices distributing final "
                    "circuits from a common busbar.",
        function="Splits an incoming supply into protected final circuits, each "
                 "with its own overcurrent and (where required) earth-leakage "
                 "protection.",
        weights={"mcb": 2.0, "rccb": 1.5, "rcbo": 1.5, "busbar": 1.2,
                 "din_rail": 0.4, "neutral_bar": 0.8, "earth_bar": 0.8,
                 "surge_protector": 0.8, "fuse": 0.3, "energy_meter": 0.3},
        penalties={"contactor": 1.2, "overload_relay": 2.0, "vfd": 2.0,
                   "plc": 1.5, "hmi": 1.0},
        keystones=("mcb", "rcbo", "rccb"),
        count_cap=12,
        expected=("mcb", "busbar", "neutral_bar", "earth_bar", "din_rail"),
    ),
    PanelTypeRule(
        id="main_lv_switchboard", name="Main LV Switchboard",
        description="Main incoming protection with metering and heavy current "
                    "distribution.",
        function="Receives and protects the main low-voltage supply, meters it, "
                 "and distributes it to sub-boards through moulded-case feeders.",
        weights={"acb": 4.0, "mccb": 2.0, "current_transformer": 1.5,
                 "energy_meter": 1.5, "busbar": 1.5, "voltage_transformer": 1.0,
                 "protection_relay": 1.0, "ammeter": 0.6, "surge_protector": 0.6,
                 "earth_bar": 0.5},
        penalties={"plc": 0.8, "hmi": 0.8, "vfd": 1.0},
        keystones=("acb", "mccb"),
        count_cap=6,
        expected=("acb", "current_transformer", "energy_meter", "busbar",
                  "earth_bar", "surge_protector"),
    ),
    PanelTypeRule(
        id="automatic_transfer_switch", name="Automatic Transfer Switch (ATS) Panel",
        description="Two supplies, a mechanically interlocked transfer element "
                    "and a controller that sequences the changeover.",
        function="Monitors the mains supply and automatically transfers the load "
                 "to a standby generator on failure, then back on restoration, "
                 "with interlocking that prevents paralleling the sources.",
        weights={"changeover_switch": 4.0, "ats_controller": 3.5,
                 "contactor": 1.0, "protection_relay": 1.0, "mccb": 0.8,
                 "acb": 0.8, "indicator_lamp": 0.5, "selector_switch": 0.5,
                 "energy_meter": 0.4},
        penalties={"overload_relay": 1.0, "vfd": 1.2},
        keystones=("changeover_switch", "ats_controller"),
        expected=("changeover_switch", "ats_controller", "indicator_lamp",
                  "selector_switch", "terminal_block", "earth_bar"),
    ),
    PanelTypeRule(
        id="power_factor_correction", name="Power Factor Correction Panel",
        description="Switched capacitor stages under automatic reactive-power "
                    "control.",
        function="Corrects lagging power factor by switching capacitor stages in "
                 "and out of circuit to hold a target power factor and reduce "
                 "reactive demand charges.",
        weights={"capacitor": 4.0, "pf_controller": 3.5, "contactor": 1.2,
                 "line_reactor": 1.0, "mccb": 0.6, "cooling_fan": 0.5,
                 "current_transformer": 0.8, "fuse": 0.4},
        penalties={"overload_relay": 1.5, "plc": 0.8, "vfd": 1.0},
        keystones=("capacitor", "pf_controller"),
        count_cap=8,
        expected=("capacitor", "pf_controller", "contactor",
                  "current_transformer", "cooling_fan", "earth_bar"),
    ),
    PanelTypeRule(
        id="lighting_control_panel", name="Lighting / Load Control Panel",
        description="Contactors switching lighting or non-motor load circuits "
                    "under timer or BMS control, with no motor protection.",
        function="Switches lighting or general load circuits on schedule or from "
                 "a building management signal, with modular overcurrent "
                 "protection per circuit.",
        weights={"contactor": 2.0, "timer_relay": 2.5, "mcb": 1.2,
                 "selector_switch": 0.8, "indicator_lamp": 0.6,
                 "logic_module": 1.0, "relay": 0.6},
        penalties={"overload_relay": 3.0, "vfd": 2.0, "capacitor": 1.5},
        keystones=("contactor", "timer_relay"),
        count_cap=8,
        expected=("contactor", "mcb", "timer_relay", "selector_switch",
                  "terminal_block"),
    ),
    PanelTypeRule(
        id="metering_panel", name="Metering / Monitoring Panel",
        description="Instrumentation only — meters fed from CTs and VTs, with no "
                    "switching of the load.",
        function="Measures and records electrical parameters (current, voltage, "
                 "power, energy, power quality) for an existing feeder without "
                 "switching it.",
        weights={"energy_meter": 4.0, "current_transformer": 2.5,
                 "voltage_transformer": 2.0, "ammeter": 1.5,
                 "protection_relay": 1.0, "terminal_block": 0.3, "fuse": 0.4},
        penalties={"contactor": 2.0, "vfd": 2.0, "plc": 1.0, "mcb": 0.3},
        keystones=("energy_meter", "ammeter"),
        expected=("energy_meter", "current_transformer", "fuse",
                  "terminal_block", "earth_bar"),
    ),
    PanelTypeRule(
        id="junction_terminal_box", name="Junction / Marshalling Box",
        description="Terminal strips and cable management only — a wiring "
                    "interface with no active devices.",
        function="Marshals and terminates field cabling between the plant and the "
                 "control system; contains no switching or control devices.",
        weights={"terminal_block": 3.0, "wire_duct": 1.5, "din_rail": 1.0,
                 "cable_gland": 1.5, "earth_bar": 0.6, "signal_isolator": 0.8},
        penalties={"contactor": 2.5, "plc": 2.0, "vfd": 2.5, "mcb": 1.0,
                   "energy_meter": 1.5, "hmi": 2.0},
        keystones=("terminal_block",),
        count_cap=10,
        expected=("terminal_block", "wire_duct", "din_rail", "earth_bar",
                  "cable_gland"),
    ),
    PanelTypeRule(
        id="safety_control_panel", name="Safety Control Panel",
        description="Certified safety chain — safety relays evaluating "
                    "emergency-stop and guard circuits and dropping out the "
                    "power contactors.",
        function="Implements the machine's functional-safety circuit: monitors "
                 "emergency stops and guards through redundant safety relays and "
                 "removes power from hazardous motion.",
        weights={"safety_relay": 4.0, "emergency_stop": 2.5, "contactor": 1.0,
                 "relay": 0.6, "power_supply": 0.5, "terminal_block": 0.3},
        penalties={"capacitor": 1.0, "energy_meter": 0.8},
        keystones=("safety_relay",),
        expected=("safety_relay", "emergency_stop", "contactor",
                  "power_supply", "terminal_block"),
    ),
    PanelTypeRule(
        id="motor_starter_panel", name="Single Motor Starter Panel",
        description="One motor feeder: protection, switching and overload for a "
                    "single machine, with local controls.",
        function="Starts, stops and protects a single three-phase motor, with "
                 "local start/stop control and run/trip indication.",
        weights={"contactor": 2.5, "overload_relay": 2.5, "motor_starter": 3.0,
                 "push_button": 1.2, "indicator_lamp": 1.0, "mccb": 0.8,
                 "mcb": 0.4, "selector_switch": 0.6},
        penalties={"vfd": 1.0, "plc": 0.8},
        keystones=("contactor", "motor_starter"),
        count_cap=2,
        expected=("contactor", "overload_relay", "push_button",
                  "indicator_lamp", "terminal_block"),
    ),
)

RULES: dict[str, PanelTypeRule] = {r.id: r for r in _RULES}

UNCLASSIFIED = "unclassified"

#: Below this normalised score we refuse to name the panel type.
MIN_CLASSIFY_SCORE = 0.22
#: Fewer accepted devices than this is not enough evidence for any verdict.
MIN_EVIDENCE_DEVICES = 3


@dataclass
class PanelTypeCandidate:
    id: str
    name: str
    confidence: float
    score: float
    evidence: list[str]
    function: str
    description: str

    def to_dict(self) -> dict:
        return {
            "id": self.id, "name": self.name,
            "confidence": round(self.confidence, 4),
            "score": round(self.score, 3),
            "evidence": self.evidence,
            "function": self.function,
            "description": self.description,
        }


@dataclass
class PanelClassification:
    panel_type: str
    panel_type_name: str
    confidence: float
    function: str
    evidence: list[str]
    candidates: list[PanelTypeCandidate]
    reason: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "panel_type": self.panel_type,
            "panel_type_name": self.panel_type_name,
            "confidence": round(self.confidence, 4),
            "function": self.function,
            "evidence": self.evidence,
            "candidates": [c.to_dict() for c in self.candidates],
            "reason": self.reason,
        }


def _rule_score(rule: PanelTypeRule, counts: Mapping[str, int]
                ) -> tuple[float, list[str]]:
    """Weighted evidence score for one rule, with human-readable evidence."""
    score = 0.0
    evidence: list[str] = []
    for cid, weight in rule.weights.items():
        n = int(counts.get(cid, 0))
        if n <= 0:
            continue
        effective = min(n, rule.count_cap)
        # Diminishing returns: the first device is worth full weight, further
        # ones sqrt-scaled, so 12 MCBs don't swamp every other signal.
        contribution = weight * (effective ** 0.5)
        score += contribution
        if weight >= 1.0:
            evidence.append(f"{n}× {tax.display_name(cid)}")
    for cid, weight in rule.penalties.items():
        n = int(counts.get(cid, 0))
        if n > 0:
            score -= weight * (min(n, rule.count_cap) ** 0.5)
            evidence.append(f"counter-evidence: {n}× {tax.display_name(cid)}")
    if rule.keystones and not any(int(counts.get(k, 0)) > 0 for k in rule.keystones):
        score *= 0.30
        evidence.append(
            "no keystone device detected ("
            + ", ".join(tax.display_name(k) for k in rule.keystones) + ")")
    return max(0.0, score), evidence


def classify(counts: Mapping[str, int], top_k: int = 3) -> PanelClassification:
    """Rank panel types from a component inventory.

    ``counts`` maps canonical class ids to quantities (as produced by
    :func:`~.postprocess.counts`). Unknown-class detections are ignored as
    evidence — they carry no information about panel type.
    """
    clean = {k: int(v) for k, v in counts.items()
             if v and k != tax.UNKNOWN_COMPONENT_ID}
    device_total = sum(v for k, v in clean.items()
                       if k in tax.countable_classes())

    scored: list[tuple[float, PanelTypeRule, list[str]]] = []
    for rule in _RULES:
        s, ev = _rule_score(rule, clean)
        scored.append((s, rule, ev))
    scored.sort(key=lambda t: t[0], reverse=True)

    total = sum(s for s, _, _ in scored)
    candidates = [
        PanelTypeCandidate(
            id=r.id, name=r.name,
            confidence=(s / total) if total > 0 else 0.0,
            score=s, evidence=ev, function=r.function, description=r.description,
        )
        for s, r, ev in scored[:max(1, top_k)]
    ]

    best = candidates[0] if candidates else None
    if best is None or device_total < MIN_EVIDENCE_DEVICES:
        return PanelClassification(
            panel_type=UNCLASSIFIED, panel_type_name="Unclassified Panel",
            confidence=0.0,
            function="Not enough recognised components to determine the panel's "
                     "purpose. Capture a sharper or wider image, or train the "
                     "detector on this panel family.",
            evidence=[f"only {device_total} device(s) recognised"],
            candidates=candidates,
            reason="insufficient_evidence",
        )
    if best.confidence < MIN_CLASSIFY_SCORE:
        return PanelClassification(
            panel_type=UNCLASSIFIED, panel_type_name="Unclassified Panel",
            confidence=round(best.confidence, 4),
            function="The detected component mix does not match a known panel "
                     "archetype closely enough to state a type.",
            evidence=best.evidence,
            candidates=candidates,
            reason="ambiguous_composition",
        )
    return PanelClassification(
        panel_type=best.id, panel_type_name=best.name,
        confidence=best.confidence, function=best.function,
        evidence=best.evidence, candidates=candidates,
    )


# --------------------------------------------------------------------------
# Application / process inference
# --------------------------------------------------------------------------

#: OCR keyword → controlled process. Panel labels and wire markers are the most
#: reliable evidence of what a panel actually drives.
_APPLICATION_KEYWORDS: tuple[tuple[str, str], ...] = (
    (r"\bpump|\bpmp\b|booster|sump|dosing|borehole", "pumping"),
    (r"\bfan\b|\bahu\b|air\s*handl|chiller|\bhvac\b|cooling\s*tower|extract|"
     r"ventilat|fcu\b", "hvac"),
    (r"conveyor|belt\s*drive|\bcnv\b|transfer\s*line", "material handling"),
    (r"compressor|\bcomp\b\s*\d|air\s*receiver", "compressed air"),
    (r"\bmixer|agitator|\bstirrer", "process mixing"),
    (r"lighting|\blight\b|luminaire|street\s*light", "lighting"),
    (r"generator|\bgenset\b|\bdg\b\s*set|\bamf\b", "standby generation"),
    (r"crane|hoist|winch", "lifting"),
    (r"\bfilter\s*press|\bblower|aerat", "water treatment"),
    (r"\bfire\b|sprinkler|jockey", "fire fighting"),
    (r"\bpackag|filler|capper|labell", "packaging line"),
)

#: Component-composition hints, used when there is no readable text.
_COMPOSITION_HINTS: tuple[tuple[str, tuple[str, ...], str], ...] = (
    ("pumping", ("contactor", "overload_relay", "protection_relay"),
     "motor starters with protection relays, typical of duty/standby pump sets"),
    ("hvac", ("vfd", "thermostat", "cooling_fan"),
     "variable-speed drives with temperature control, typical of air handling"),
    ("material handling", ("vfd", "plc", "limit_switch", "sensor", "encoder"),
     "drives plus position feedback, typical of conveying equipment"),
    ("standby generation", ("changeover_switch", "ats_controller"),
     "source transfer equipment, typical of a generator installation"),
    ("lighting", ("contactor", "timer_relay"),
     "scheduled contactor switching without motor overloads"),
)


@dataclass
class ApplicationGuess:
    application: Optional[str]
    confidence: float
    evidence: list[str]

    def to_dict(self) -> dict:
        return {"application": self.application,
                "confidence": round(self.confidence, 4),
                "evidence": self.evidence}


def infer_application(counts: Mapping[str, int],
                      text: Optional[Sequence[str]] = None) -> ApplicationGuess:
    """Guess what process the panel controls, from labels and composition."""
    evidence: list[str] = []
    votes: dict[str, float] = {}

    joined = " ".join(str(t) for t in (text or [])).lower()
    for pattern, app in _APPLICATION_KEYWORDS:
        if joined and re.search(pattern, joined):
            votes[app] = votes.get(app, 0.0) + 2.0
            evidence.append(f"panel text matches '{app}'")

    for app, needed, why in _COMPOSITION_HINTS:
        present = [c for c in needed if int(counts.get(c, 0)) > 0]
        if len(present) >= 2:
            votes[app] = votes.get(app, 0.0) + 0.6 * len(present)
            evidence.append(why)

    if not votes:
        return ApplicationGuess(None, 0.0,
                                ["no label text or composition signature "
                                 "specific enough to name the process"])
    total = sum(votes.values())
    app, best = max(votes.items(), key=lambda kv: kv[1])
    return ApplicationGuess(app, best / total, evidence)


# --------------------------------------------------------------------------
# Expected bill of materials / missing components
# --------------------------------------------------------------------------

@dataclass
class MissingComponent:
    class_id: str
    name: str
    severity: str          # info | advisory | important
    rationale: str

    def to_dict(self) -> dict:
        return {"class_id": self.class_id, "name": self.name,
                "severity": self.severity, "rationale": self.rationale}


#: Components whose absence is genuinely significant rather than cosmetic.
_IMPORTANT_ABSENCES = {"earth_bar", "overload_relay", "emergency_stop",
                       "safety_relay", "surge_protector", "cooling_fan",
                       "current_transformer"}


def missing_components(panel_type: str, counts: Mapping[str, int]
                       ) -> list[MissingComponent]:
    """Components a panel of this type normally has, that were not detected.

    Phrased as *possible* omissions: a component may be present but hidden,
    out of frame, or simply not yet recognisable by the model. This is decision
    support for an inspector, not an assertion of non-existence.
    """
    rule = RULES.get(panel_type)
    if rule is None:
        return []
    out: list[MissingComponent] = []
    for cid in rule.expected:
        if int(counts.get(cid, 0)) > 0:
            continue
        sp = tax.spec(cid)
        severity = "important" if cid in _IMPORTANT_ABSENCES else "advisory"
        out.append(MissingComponent(
            class_id=cid, name=sp.name, severity=severity,
            rationale=f"A {rule.name} normally includes {sp.name.lower()} — "
                      f"{sp.role.rstrip('.').lower()}. Not detected in this "
                      f"image; verify it is present and in frame.",
        ))
    return out


# --------------------------------------------------------------------------
# Engineering / maintenance observations
# --------------------------------------------------------------------------

@dataclass
class MaintenanceNote:
    code: str
    severity: str          # info | advisory | important
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "severity": self.severity,
                "message": self.message}


def maintenance_notes(counts: Mapping[str, int], panel_type: str,
                      unknown_count: int = 0,
                      accepted_total: int = 0) -> list[MaintenanceNote]:
    """Rule-based engineering observations derived from the inventory.

    Each note is a check a competent inspector performs by eye. They are
    observations about *this inventory*, never invented defects: a note only
    fires on a relationship that the detected counts actually support.
    """
    notes: list[MaintenanceNote] = []
    g = lambda cid: int(counts.get(cid, 0))  # noqa: E731

    contactors, overloads = g("contactor"), g("overload_relay")
    if contactors and overloads and overloads < contactors:
        notes.append(MaintenanceNote(
            "starter_protection_mismatch", "important",
            f"{contactors} contactor(s) but only {overloads} overload relay(s) "
            f"detected. Every motor contactor needs thermal protection — check "
            f"whether the remaining feeders use electronic protection, a motor "
            f"protection circuit breaker, or are genuinely unprotected."))

    if g("vfd") and not g("cooling_fan"):
        notes.append(MaintenanceNote(
            "drive_thermal_management", "important",
            "Variable frequency drive(s) detected with no enclosure cooling fan. "
            "Drives are the dominant heat source in a panel; verify forced "
            "ventilation and filter condition against the drive's derating curve."))

    if g("plc") and not g("power_supply"):
        notes.append(MaintenanceNote(
            "control_supply_not_visible", "advisory",
            "A controller was detected but no 24 V DC power supply. Confirm the "
            "control supply is present and within frame."))

    if g("power_supply") and not (g("mcb") or g("fuse") or g("fuse_holder")):
        notes.append(MaintenanceNote(
            "unprotected_control_supply", "important",
            "Power supply detected with no upstream MCB or fuse in view. The "
            "control transformer/supply primary must be protected."))

    if g("emergency_stop") and not g("safety_relay"):
        notes.append(MaintenanceNote(
            "estop_not_monitored", "important",
            "Emergency stop detected but no safety relay. Confirm the E-stop is "
            "evaluated by a certified safety device rather than wired directly "
            "into a standard control relay."))

    if g("safety_relay") and not g("emergency_stop"):
        notes.append(MaintenanceNote(
            "safety_chain_incomplete", "advisory",
            "Safety relay detected but no emergency-stop device in frame. Verify "
            "the safety inputs are terminated to real, accessible E-stops."))

    if g("current_transformer") and not (g("energy_meter") or g("ammeter")
                                         or g("protection_relay")):
        notes.append(MaintenanceNote(
            "ct_secondary_unloaded", "important",
            "Current transformer(s) detected with no meter or protection relay. "
            "An open-circuited CT secondary develops a dangerous voltage — "
            "confirm every CT is either loaded or shorted."))

    if g("capacitor") and not g("pf_controller"):
        notes.append(MaintenanceNote(
            "unregulated_capacitors", "advisory",
            "Power-factor capacitors detected without a power-factor controller. "
            "Fixed compensation can over-correct at light load; verify the "
            "switching scheme."))

    if not g("earth_bar") and accepted_total >= 5:
        notes.append(MaintenanceNote(
            "earthing_not_visible", "advisory",
            "No earth/ground bar detected. Verify the protective-earth bar, door "
            "bonding strap and gland-plate bonding — they are frequently out of "
            "frame in a front-on photograph."))

    if g("terminal_block") and not g("wire_duct") and accepted_total >= 8:
        notes.append(MaintenanceNote(
            "cable_management", "info",
            "Terminal blocks detected without cable ducting in view. Confirm "
            "internal wiring is routed in trunking rather than loose bundles."))

    if accepted_total and unknown_count:
        ratio = unknown_count / accepted_total
        if ratio >= 0.30:
            notes.append(MaintenanceNote(
                "low_recognition_confidence", "important",
                f"{unknown_count} of {accepted_total} detections "
                f"({ratio:.0%}) could not be classified confidently and are "
                f"reported as Unknown Industrial Component. Treat the component "
                f"list as incomplete, and add these crops to the training set."))
        elif ratio > 0:
            notes.append(MaintenanceNote(
                "some_unknown_components", "advisory",
                f"{unknown_count} detection(s) were kept but not classified. "
                f"They are genuine devices the model is not yet trained on."))

    if g("surge_protector") == 0 and panel_type in (
            "main_lv_switchboard", "distribution_panel"):
        notes.append(MaintenanceNote(
            "no_surge_protection", "advisory",
            "No surge protection device detected on a distribution/switchboard "
            "assembly. Check whether transient protection is required by the "
            "installation standard for this location."))

    return notes


def rule_summary() -> dict:
    return {
        "panel_types": [
            {"id": r.id, "name": r.name, "description": r.description,
             "function": r.function, "keystones": list(r.keystones),
             "expected": list(r.expected)}
            for r in _RULES
        ],
        "min_classify_score": MIN_CLASSIFY_SCORE,
        "min_evidence_devices": MIN_EVIDENCE_DEVICES,
    }


__all__ = [
    "PanelTypeRule", "RULES", "UNCLASSIFIED", "PanelTypeCandidate",
    "PanelClassification", "classify", "ApplicationGuess", "infer_application",
    "MissingComponent", "missing_components", "MaintenanceNote",
    "maintenance_notes", "rule_summary", "MIN_CLASSIFY_SCORE",
    "MIN_EVIDENCE_DEVICES",
]
