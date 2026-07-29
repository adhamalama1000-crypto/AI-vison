"""
Panel risk assessment — one auditable level from many pieces of evidence.

:mod:`rtsp_backend.electrical.panel_type` already produces per-finding severities:
missing components carry ``advisory``/``important``, and maintenance notes carry
``info``/``advisory``/``important``. What nothing did was aggregate them, so a report
with one important finding and a report with twelve looked identical at a glance —
which is exactly the glance an inspector takes.

This module produces that aggregate. The design constraints are unusual for a
scoring function, and they come from what the output is used for: somebody may
decide not to open a cabinet because of it.

**It refuses to score when it has no basis.** With no model loaded, or no
components detected, the level is ``unknown`` — never ``low``. A "low risk" verdict
derived from zero detections is the most dangerous output the platform could produce,
because "we found nothing wrong" and "we could not look" are indistinguishable to the
reader while meaning opposite things.

**Detection quality is itself a risk driver.** A panel where 60% of devices came
back as ``unknown_industrial_component`` has not been assessed, and the assessment
says so rather than scoring the 40% it managed to read.

**Every level is traceable.** :class:`RiskAssessment` carries the individual
drivers with their own contributions, so "why is this elevated?" is answerable from
the JSON without re-running anything. Nothing is a black box, and nothing is a
learned weight — these are declared engineering judgements, and they are
overridable.

**It never invents a defect.** Every driver is derived from a finding that already
exists in the inspection result. This module adds no observation of its own; it only
weighs what the detector and the rule engine found.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from . import taxonomy as tax

#: Risk levels, ascending. ``unknown`` is not a level on this scale — it means the
#: assessment could not be made, and it is deliberately not orderable against the
#: others.
LEVELS: tuple[str, ...] = ("low", "moderate", "elevated", "high")
UNKNOWN_LEVEL = "unknown"

#: Score thresholds. A panel accumulates weighted points from its findings; these
#: are the boundaries between levels. Chosen so that a single ``important`` finding
#: reaches ``moderate`` and does not on its own reach ``high`` — one missing overload
#: relay is worth investigating, not worth condemning the panel.
THRESHOLDS: dict[str, float] = {"moderate": 2.0, "elevated": 5.0, "high": 9.0}

#: Weight per severity of a *missing component*. A missing protective device is the
#: most consequential thing this system can observe, because its absence is what
#: turns a fault into a fire.
MISSING_WEIGHTS: dict[str, float] = {
    "important": 3.0, "advisory": 1.0, "info": 0.25,
}

#: Weight per severity of a *maintenance note*.
NOTE_WEIGHTS: dict[str, float] = {
    "important": 2.5, "advisory": 1.0, "info": 0.25,
}

#: Classes whose absence is a safety matter rather than a completeness matter, and
#: which therefore carry an additional weight on top of their severity. These are
#: the devices whose job is to stop somebody being hurt.
SAFETY_CRITICAL: frozenset[str] = frozenset({
    "earth_bar", "emergency_stop", "safety_relay", "overload_relay",
    "surge_protector", "rccb", "rcbo",
})
SAFETY_CRITICAL_BONUS = 1.5

#: Fraction of detections that may be unidentified before the assessment is
#: considered unreliable rather than merely uncertain.
UNKNOWN_RATIO_UNRELIABLE = 0.5
#: ...and the ratio above which it contributes risk at all.
UNKNOWN_RATIO_CONCERN = 0.2

#: Mean detection confidence below which the inventory itself is doubtful.
LOW_CONFIDENCE_THRESHOLD = 0.55

#: Below this many detected devices, a panel photograph has probably not captured
#: the whole cabinet, so absences are not evidence.
MIN_COMPONENTS_FOR_ABSENCE_EVIDENCE = 4


@dataclass
class RiskDriver:
    """One traceable contribution to the risk score."""

    code: str
    category: str            # missing_protection | maintenance | detection_quality
    severity: str            # info | advisory | important
    weight: float
    message: str
    class_id: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "code": self.code, "category": self.category,
            "severity": self.severity, "weight": round(self.weight, 3),
            "message": self.message, "class_id": self.class_id,
        }


@dataclass
class RiskAssessment:
    level: str
    score: float
    confidence: str                     # high | moderate | low | none
    headline: str
    drivers: list = field(default_factory=list)
    recommendations: list = field(default_factory=list)
    limits: list = field(default_factory=list)
    assessable: bool = True

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "score": round(self.score, 3),
            "confidence": self.confidence,
            "headline": self.headline,
            "assessable": self.assessable,
            "drivers": [d.to_dict() if isinstance(d, RiskDriver) else d
                        for d in self.drivers],
            "recommendations": list(self.recommendations),
            "limits": list(self.limits),
            "thresholds": dict(THRESHOLDS),
            "scale": list(LEVELS),
        }


def _level_for(score: float) -> str:
    if score >= THRESHOLDS["high"]:
        return "high"
    if score >= THRESHOLDS["elevated"]:
        return "elevated"
    if score >= THRESHOLDS["moderate"]:
        return "moderate"
    return "low"


def _not_assessable(reason: str, limits: Sequence[str],
                    recommendations: Sequence[str]) -> RiskAssessment:
    return RiskAssessment(
        level=UNKNOWN_LEVEL, score=0.0, confidence="none",
        headline=reason, assessable=False,
        drivers=[], limits=list(limits),
        recommendations=list(recommendations))


def assess(result: Mapping[str, Any]) -> RiskAssessment:
    """Aggregate an inspection result into one risk level.

    ``result`` is the dict from :func:`rtsp_backend.electrical.inspector.inspect_panel`.
    Adds no findings of its own — it weighs only what is already there.
    """
    components = list(result.get("components") or [])
    missing = list(result.get("missing_components") or [])
    notes = list(result.get("maintenance_notes") or [])
    conf = dict(result.get("confidence") or {})
    panel = dict(result.get("panel") or {})
    model_loaded = result.get("component_model_loaded")

    # -- refuse to score without a basis ---------------------------------
    if model_loaded is False:
        return _not_assessable(
            "Risk cannot be assessed: no trained component model is loaded, so "
            "nothing was inspected. This is not a low-risk result — it is an "
            "absence of inspection.",
            limits=["No component detection was performed."],
            recommendations=[
                "Install a trained detector bundle into models/components/ and "
                "re-run the inspection (see docs/ELECTRICAL_MODEL_TRAINING.md).",
                "Until then, this panel must be assessed by a human inspector.",
            ])
    if not components:
        return _not_assessable(
            "Risk cannot be assessed: no components were detected in this image. "
            "Either the panel is not visible, the image is unusable, or the model "
            "does not recognise this panel's devices. 'Nothing found' is not "
            "'nothing wrong'.",
            limits=["No devices were detected, so no inventory exists to assess."],
            recommendations=[
                "Re-photograph with the cabinet door open, the device rows filling "
                "the frame, and even lighting.",
                "If devices are clearly visible and still not detected, this panel "
                "family needs to be added to the training set.",
            ])

    total = len(components)
    unknown = int(conf.get("unknown") or 0)
    unknown_ratio = unknown / total if total else 0.0
    mean_conf = conf.get("mean")

    drivers: list[RiskDriver] = []
    limits: list[str] = []

    # -- detection quality first: it bounds everything else --------------
    if unknown_ratio >= UNKNOWN_RATIO_UNRELIABLE:
        return _not_assessable(
            f"Risk cannot be reliably assessed: {unknown}/{total} detected devices "
            f"({unknown_ratio:.0%}) could not be identified. An inventory that is "
            f"more than half unknown does not support a risk judgement, and "
            f"scoring the identified minority would understate the uncertainty.",
            limits=[
                f"{unknown_ratio:.0%} of detections are unidentified.",
                "Absent-component reasoning is unreliable, because a device the "
                "model could not classify may be exactly the one reported missing.",
            ],
            recommendations=[
                "Have an engineer identify the unknown devices from the annotated "
                "image.",
                "Feed those crops back into the training set — they are precisely "
                "the examples the model is asking for "
                "(python -m training.electrical.cli autolabel).",
            ])

    if unknown_ratio >= UNKNOWN_RATIO_CONCERN:
        drivers.append(RiskDriver(
            code="unidentified_devices", category="detection_quality",
            severity="advisory",
            weight=1.0 + 2.0 * unknown_ratio,
            message=(f"{unknown}/{total} detected devices ({unknown_ratio:.0%}) "
                     f"could not be identified. Any conclusion about what this "
                     f"panel is missing is weakened accordingly — one of the "
                     f"unknowns may be the device reported absent.")))
        limits.append(
            f"{unknown_ratio:.0%} of detections are unidentified; missing-component "
            f"findings should be verified by eye.")

    if isinstance(mean_conf, (int, float)) and mean_conf < LOW_CONFIDENCE_THRESHOLD:
        drivers.append(RiskDriver(
            code="low_detection_confidence", category="detection_quality",
            severity="advisory", weight=1.0,
            message=(f"Mean detection confidence is {mean_conf:.2f}, below "
                     f"{LOW_CONFIDENCE_THRESHOLD:.2f}. The inventory itself is "
                     f"doubtful — treat both the detections and the conclusions "
                     f"drawn from them as provisional.")))
        limits.append("Detection confidence is low; the inventory may be wrong.")

    # -- missing components ----------------------------------------------
    # A photograph showing only a handful of devices has probably not captured the
    # whole cabinet, so "X was not detected" is not evidence that X is absent.
    absence_is_evidence = total >= MIN_COMPONENTS_FOR_ABSENCE_EVIDENCE
    if not absence_is_evidence:
        limits.append(
            f"Only {total} device(s) detected — too few to treat an absence as "
            f"evidence, so missing-component findings do not contribute to the "
            f"score. The image probably does not show the whole panel.")

    for m in missing:
        cid = m.get("class_id")
        severity = str(m.get("severity") or "advisory")
        weight = MISSING_WEIGHTS.get(severity, 1.0)
        safety = cid in SAFETY_CRITICAL
        if safety:
            weight += SAFETY_CRITICAL_BONUS
        if not absence_is_evidence:
            weight = 0.0
        drivers.append(RiskDriver(
            code=f"missing_{cid}", category="missing_protection",
            severity=severity, weight=weight, class_id=cid,
            message=(m.get("rationale")
                     or f"{tax.display_name(cid or '')} was not detected.")
                    + (" This is a safety-critical device."
                       if safety else "")
                    + ("" if absence_is_evidence else
                       " (Not scored: too few devices detected for an absence to "
                       "mean anything.)")))

    # -- maintenance notes -----------------------------------------------
    for n in notes:
        severity = str(n.get("severity") or "info")
        drivers.append(RiskDriver(
            code=str(n.get("code") or "note"), category="maintenance",
            severity=severity,
            weight=NOTE_WEIGHTS.get(severity, 0.25),
            message=str(n.get("message") or "")))

    score = sum(d.weight for d in drivers)
    level = _level_for(score)

    # -- confidence in the assessment itself -----------------------------
    if unknown_ratio < 0.1 and (mean_conf is None
                                or mean_conf >= 0.7) and absence_is_evidence:
        assessment_conf = "high"
    elif unknown_ratio < UNKNOWN_RATIO_CONCERN and absence_is_evidence:
        assessment_conf = "moderate"
    else:
        assessment_conf = "low"

    if panel.get("panel_type") in (None, "unclassified"):
        limits.append(
            "The panel type could not be determined, so expected-component "
            "reasoning had no template to compare against. Missing-component "
            "findings are therefore incomplete rather than exhaustive.")
        if assessment_conf == "high":
            assessment_conf = "moderate"

    return RiskAssessment(
        level=level, score=score, confidence=assessment_conf,
        headline=_headline(level, score, drivers, assessment_conf),
        drivers=sorted(drivers, key=lambda d: -d.weight),
        recommendations=_recommendations(level, drivers, unknown_ratio),
        limits=limits, assessable=True)


def _headline(level: str, score: float, drivers: Sequence[RiskDriver],
              confidence: str) -> str:
    scored = [d for d in drivers if d.weight > 0]
    important = [d for d in scored if d.severity == "important"]
    safety = [d for d in scored if d.class_id in SAFETY_CRITICAL]

    if level == "low":
        # "Consistent with a correctly-populated panel" is a positive claim, and it
        # is only supportable when the assessment itself is sound. With low
        # confidence — a handful of devices, or a third of them unidentified — the
        # honest statement is that nothing was found, not that nothing is wrong.
        base = ("No significant risk indicators. The detected inventory is "
                "consistent with a correctly-populated panel."
                if confidence in ("high", "moderate") else
                "No risk indicators were scored, but this reflects how little "
                "could be established from the image rather than a clean panel. "
                "Do not read this as a pass.")
    elif level == "moderate":
        base = (f"{len(scored)} risk indicator(s) found, "
                f"{len(important)} of them important.")
    elif level == "elevated":
        base = (f"{len(scored)} risk indicator(s), {len(important)} important"
                + (f", including {len(safety)} safety-critical device(s) not "
                   f"detected" if safety else "") + ".")
    else:
        base = (f"{len(scored)} risk indicator(s), {len(important)} important"
                + (f" and {len(safety)} safety-critical" if safety else "")
                + ". This panel needs review before it is relied upon.")

    suffix = {
        "high": "",
        "moderate": " Confidence in this assessment is moderate.",
        "low": " CONFIDENCE IN THIS ASSESSMENT IS LOW — see 'limits'.",
        "none": "",
    }[confidence]
    return f"{base} Risk level: {level.upper()} (score {score:.1f}).{suffix}"


def _recommendations(level: str, drivers: Sequence[RiskDriver],
                     unknown_ratio: float) -> list[str]:
    """Actionable next steps, ordered by what matters most.

    Recommendations are derived from the drivers that actually fired, so this never
    emits generic advice that does not match the findings.
    """
    out: list[str] = []
    scored = [d for d in drivers if d.weight > 0]

    safety_missing = [d for d in scored
                      if d.category == "missing_protection"
                      and d.class_id in SAFETY_CRITICAL]
    if safety_missing:
        names = ", ".join(tax.display_name(d.class_id or "")
                          for d in safety_missing)
        out.append(
            f"PRIORITY: verify by eye whether these safety-critical devices are "
            f"genuinely absent or merely not visible in this image — {names}. If "
            f"genuinely absent, treat as a compliance finding and escalate.")

    important_notes = [d for d in scored
                       if d.category == "maintenance"
                       and d.severity == "important"]
    for d in important_notes[:4]:
        out.append(f"Investigate: {d.message}")

    other_missing = [d for d in scored
                     if d.category == "missing_protection"
                     and d.class_id not in SAFETY_CRITICAL]
    if other_missing:
        names = ", ".join(tax.display_name(d.class_id or "")
                          for d in other_missing[:6])
        out.append(
            f"Confirm presence of: {names}. These are expected for this panel "
            f"type but were not detected; they may be present and out of frame.")

    if unknown_ratio >= UNKNOWN_RATIO_CONCERN:
        out.append(
            "Identify the unidentified devices from the annotated image, then add "
            "those crops to the training set so the next inspection reads them "
            "(python -m training.electrical.cli autolabel).")

    if level == "low" and not out:
        out.append(
            "No action indicated by this image. This is not a substitute for a "
            "physical inspection: thermal condition, torque, insulation and "
            "conductor sizing are not observable from a photograph.")
    else:
        out.append(
            "This assessment is derived from one photograph. It cannot see thermal "
            "condition, terminal torque, insulation resistance or conductor "
            "sizing — confirm on site before acting on it.")
    return out


def summary_line(assessment: RiskAssessment) -> str:
    """One line for a PDF header or a dashboard badge."""
    if not assessment.assessable:
        return f"Risk: UNKNOWN — {assessment.headline}"
    return (f"Risk: {assessment.level.upper()} (score {assessment.score:.1f}, "
            f"confidence {assessment.confidence})")


__all__ = [
    "LEVELS", "UNKNOWN_LEVEL", "THRESHOLDS", "MISSING_WEIGHTS",
    "NOTE_WEIGHTS", "SAFETY_CRITICAL", "SAFETY_CRITICAL_BONUS",
    "UNKNOWN_RATIO_UNRELIABLE", "UNKNOWN_RATIO_CONCERN",
    "LOW_CONFIDENCE_THRESHOLD", "MIN_COMPONENTS_FOR_ABSENCE_EVIDENCE",
    "RiskDriver", "RiskAssessment", "assess", "summary_line",
]
