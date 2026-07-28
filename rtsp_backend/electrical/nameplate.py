"""
Nameplate reading: manufacturer and series identification from panel text.

A trained detector can tell you *what kind* of device it is looking at. It
cannot tell you it is a Schneider TeSys LC1D32 rather than a Siemens 3RT2027 —
that information is printed on the device, and the way an engineer reads it is
off the nameplate.

This module holds a curated catalogue of real manufacturer part-number
signatures for industrial control gear, and matches OCR text found *inside a
component's bounding box* against it. When a signature matches, the component
gains a manufacturer, a product family and — importantly — a *corroboration
check*: if the nameplate says "contactor family" and the detector said
"contactor", confidence in the class goes up; if they disagree, that
disagreement is reported rather than hidden.

Nothing is invented. With no OCR engine installed, or no readable text, every
component simply reports ``manufacturer: None`` and the detector's own class
stands alone.

Signature sources are the manufacturers' published catalogue numbering schemes
(Schneider TeSys/Acti9/Compact/Altivar, Siemens SIRIUS/SIMATIC/SENTRON/SINAMICS,
ABB AF/S200/Tmax/ACS/AC500, Eaton PKZM/DILM/NZM, Omron/Mitsubishi/Phoenix/
WAGO/Weidmüller/Pilz/MeanWell). They are matched conservatively — a signature
must be a reasonably specific token, not a bare two-letter prefix.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from . import taxonomy as tax


@dataclass(frozen=True)
class SeriesSignature:
    """A manufacturer part-number pattern and what it implies."""

    pattern: str
    manufacturer: str
    family: str
    #: Canonical taxonomy class this part number belongs to.
    class_id: str
    #: Extra engineering context printed in the expert analysis.
    note: str = ""
    #: Higher wins when two signatures match the same text.
    specificity: int = 1

    def compiled(self) -> re.Pattern:
        return re.compile(self.pattern, re.IGNORECASE)


_SIGS: tuple[SeriesSignature, ...] = (
    # ---------- Schneider Electric ----------
    SeriesSignature(r"\bLC1[\s-]?[DFK]?\d{2,3}\b", "Schneider Electric",
                    "TeSys D contactor", "contactor",
                    "TeSys D range; coil designation follows the current rating.",
                    specificity=3),
    SeriesSignature(r"\bLC2[\s-]?[DK]?\d{2,3}\b", "Schneider Electric",
                    "TeSys reversing contactor", "contactor",
                    "Mechanically interlocked reversing assembly.", specificity=3),
    SeriesSignature(r"\bLRD[\s-]?\d{2,3}\b", "Schneider Electric",
                    "TeSys LRD thermal overload relay", "overload_relay",
                    "Direct-mount thermal overload for TeSys D contactors.",
                    specificity=3),
    SeriesSignature(r"\bLR9[\s-]?[DF]?\d{2,3}\b", "Schneider Electric",
                    "TeSys electronic overload relay", "overload_relay",
                    "", specificity=3),
    SeriesSignature(r"\bGV2[\s-]?[MPL]?\w{0,4}\b", "Schneider Electric",
                    "TeSys GV2 motor circuit breaker", "motor_starter",
                    "Magneto-thermal motor protection, adjustable trip.",
                    specificity=3),
    SeriesSignature(r"\b(?:iC60|C60[NHL]?|C120[NH]?)\b", "Schneider Electric",
                    "Acti9 miniature circuit breaker", "mcb", "", specificity=3),
    SeriesSignature(r"\bA9[FKN]\w{2,6}\b", "Schneider Electric",
                    "Acti9 modular device", "mcb", "", specificity=2),
    SeriesSignature(r"\b(?:NSX|NS)\s?\d{2,4}\b", "Schneider Electric",
                    "Compact NSX moulded case circuit breaker", "mccb",
                    "Micrologic trip units are field-interchangeable.",
                    specificity=3),
    SeriesSignature(r"\bMasterpact\b|\bMTZ\d\b|\bNW\d{2}\b", "Schneider Electric",
                    "Masterpact air circuit breaker", "acb", "", specificity=3),
    SeriesSignature(r"\bATV\s?\d{2,3}\w*\b", "Schneider Electric",
                    "Altivar variable frequency drive", "vfd", "", specificity=3),
    SeriesSignature(r"\bATS\s?\d{2}\w*\b", "Schneider Electric",
                    "Altistart soft starter", "soft_starter", "", specificity=3),
    SeriesSignature(r"\b(?:ZB4|ZB5|XB4|XB5)\w{0,4}\b", "Schneider Electric",
                    "Harmony operator device", "push_button", "", specificity=2),
    SeriesSignature(r"\bXPS\w{2,6}\b", "Schneider Electric",
                    "Preventa safety relay", "safety_relay", "", specificity=3),
    SeriesSignature(r"\bRXM\d\w*\b|\bRXZ\w*\b", "Schneider Electric",
                    "Zelio interface relay", "relay", "", specificity=3),
    SeriesSignature(r"\bRE\d{2}\w*\b|\bREXL\w*\b", "Schneider Electric",
                    "Zelio timing relay", "timer_relay", "", specificity=2),
    SeriesSignature(r"\bPM\s?[25]\d{3}\b|\bPowerLogic\b", "Schneider Electric",
                    "PowerLogic power meter", "energy_meter", "", specificity=3),
    SeriesSignature(r"\bTM2\d\w*\b|\bTM3\w{2,6}\b|\bModicon\b|\bM2[13]\d\b",
                    "Schneider Electric", "Modicon controller", "plc", "",
                    specificity=3),
    SeriesSignature(r"\bABL\w{2,6}\b|\bPhaseo\b", "Schneider Electric",
                    "Phaseo power supply", "power_supply", "", specificity=3),

    # ---------- Siemens ----------
    SeriesSignature(r"\b3RT\s?[12]?\d{3,4}\b", "Siemens",
                    "SIRIUS 3RT contactor", "contactor",
                    "SIRIUS modular range; 3RH is the auxiliary-contact variant.",
                    specificity=3),
    SeriesSignature(r"\b3RH\s?\d{3,4}\b", "Siemens",
                    "SIRIUS 3RH contactor relay", "relay", "", specificity=3),
    SeriesSignature(r"\b3R[UB]\s?\d{3,4}\b", "Siemens",
                    "SIRIUS overload relay", "overload_relay", "", specificity=3),
    SeriesSignature(r"\b3RV\s?\d{3,4}\b", "Siemens",
                    "SIRIUS motor protection circuit breaker", "motor_starter",
                    "", specificity=3),
    SeriesSignature(r"\b3SK\s?\d\w*\b", "Siemens", "SIRIUS safety relay",
                    "safety_relay", "", specificity=3),
    SeriesSignature(r"\b3RP\s?\d{3,4}\b|\b3RT2\s?TIMER\b", "Siemens",
                    "SIRIUS timing relay", "timer_relay", "", specificity=3),
    SeriesSignature(r"\b5S[YLJ]\d\b|\b5SY\d{2}\b", "Siemens",
                    "5SY miniature circuit breaker", "mcb", "", specificity=3),
    SeriesSignature(r"\b3V[AL]\d\w*\b", "Siemens",
                    "SENTRON moulded case circuit breaker", "mccb", "",
                    specificity=3),
    SeriesSignature(r"\b3W[LT]\d\w*\b", "Siemens", "SENTRON air circuit breaker",
                    "acb", "", specificity=3),
    SeriesSignature(r"\bPAC\s?\d{3,4}\b|\bSENTRON\s?PAC\b", "Siemens",
                    "SENTRON PAC power meter", "energy_meter", "", specificity=3),
    SeriesSignature(r"\bS7[\s-]?1200\b|\b6ES7\s?21\d\w*\b", "Siemens",
                    "SIMATIC S7-1200 controller", "plc",
                    "Compact controller; signal boards and modules expand IO.",
                    specificity=4),
    SeriesSignature(r"\bS7[\s-]?1500\b|\b6ES7\s?5\d{2}\w*\b", "Siemens",
                    "SIMATIC S7-1500 controller", "plc", "", specificity=4),
    SeriesSignature(r"\bS7[\s-]?(?:300|400)\b|\b6ES7\s?3\d{2}\w*\b", "Siemens",
                    "SIMATIC S7-300/400 controller", "plc", "", specificity=4),
    SeriesSignature(r"\bET\s?200\w*\b", "Siemens", "SIMATIC ET 200 remote IO",
                    "io_module", "", specificity=3),
    SeriesSignature(r"\bLOGO!?\s?[78]?\b", "Siemens", "LOGO! logic module",
                    "logic_module", "", specificity=3),
    SeriesSignature(r"\b(?:KTP|TP)\s?\d{3,4}\b|\bSIMATIC\s?HMI\b", "Siemens",
                    "SIMATIC HMI operator panel", "hmi", "", specificity=3),
    SeriesSignature(r"\bSITOP\b|\b6EP\d\w*\b", "Siemens", "SITOP power supply",
                    "power_supply", "", specificity=3),
    SeriesSignature(r"\bSINAMICS\b|\bG1[12]0\b|\bV20\b|\b6SL3\w*\b", "Siemens",
                    "SINAMICS variable frequency drive", "vfd", "", specificity=3),
    SeriesSignature(r"\bSCALANCE\b|\b6GK\d\w*\b", "Siemens",
                    "SCALANCE industrial ethernet switch", "ethernet_switch", "",
                    specificity=3),
    SeriesSignature(r"\b3SU1\w*\b|\b3SB3\w*\b", "Siemens",
                    "SIRIUS ACT operator device", "push_button", "", specificity=3),

    # ---------- ABB ----------
    SeriesSignature(r"\bAF\s?\d{2,3}[\s-]?\d{0,2}\b", "ABB",
                    "AF electronically-controlled contactor", "contactor",
                    "Wide-band electronic coil (AF range).", specificity=3),
    SeriesSignature(r"\bA\d{2,3}[\s-]?30[\s-]?10\b", "ABB", "A-series contactor",
                    "contactor", "", specificity=3),
    SeriesSignature(r"\bT[AF]\d{2,3}DU\d{1,3}\b|\bE\d{2,3}DU\b", "ABB",
                    "TA/E overload relay", "overload_relay", "", specificity=3),
    SeriesSignature(r"\bMS\s?(?:11[26]|13[23]|4\d{2})\b", "ABB",
                    "MS manual motor starter", "motor_starter", "", specificity=3),
    SeriesSignature(r"\bS20[01]\w*\b|\bS80\d\w*\b", "ABB",
                    "S200/S800 miniature circuit breaker", "mcb", "",
                    specificity=3),
    SeriesSignature(r"\bT?max\s?XT\d\b|\bXT[1-7]\w*\b|\bT[1-7]N\b", "ABB",
                    "Tmax moulded case circuit breaker", "mccb", "", specificity=3),
    SeriesSignature(r"\bE[1-6]\.\d\b|\bEmax\b", "ABB", "Emax air circuit breaker",
                    "acb", "", specificity=3),
    SeriesSignature(r"\bACS\s?\d{3}\w*\b", "ABB", "ACS variable frequency drive",
                    "vfd", "", specificity=4),
    SeriesSignature(r"\bPSTX?\s?\d{2,3}\b|\bPSE\s?\d{2,3}\b", "ABB",
                    "PST/PSE soft starter", "soft_starter", "", specificity=3),
    SeriesSignature(r"\bAC500\b|\bPM5[0-9]{2}\b", "ABB", "AC500 controller",
                    "plc", "", specificity=3),
    SeriesSignature(r"\bCM[\s-]?[EMU]\w*\b|\b1SVR\w*\b", "ABB",
                    "CM-range monitoring relay", "protection_relay", "",
                    specificity=2),
    SeriesSignature(r"\bM2M\b|\bM4M\b", "ABB", "M4M network analyser",
                    "energy_meter", "", specificity=3),

    # ---------- Eaton / Moeller ----------
    SeriesSignature(r"\bDILM\d{1,3}\b", "Eaton", "DILM contactor", "contactor",
                    "", specificity=3),
    SeriesSignature(r"\bZB\d{1,3}[\s-]?\d*\b|\bZ[BE]\d\b", "Eaton",
                    "ZB overload relay", "overload_relay", "", specificity=2),
    SeriesSignature(r"\bPKZM\d\w*\b|\bPKE\d{1,2}\b", "Eaton",
                    "PKZM motor protective circuit breaker", "motor_starter", "",
                    specificity=3),
    SeriesSignature(r"\bNZM[1-4]\w*\b", "Eaton", "NZM moulded case breaker",
                    "mccb", "", specificity=3),
    SeriesSignature(r"\bM22[\s-]?\w{1,6}\b", "Eaton", "M22 operator device",
                    "push_button", "", specificity=2),
    SeriesSignature(r"\bEasy[\s-]?E4\b|\bEASY\d{3}\b", "Eaton",
                    "easyE4 logic relay", "logic_module", "", specificity=3),

    # ---------- Omron ----------
    SeriesSignature(r"\bMY[24]N?\b|\bLY[1-4]\b|\bG2R[\s-]?\d\b", "Omron",
                    "MY/LY/G2R plug-in relay", "relay", "", specificity=3),
    SeriesSignature(r"\bH3[CYD][A-Z]?\b|\bH5CX\b", "Omron", "H3/H5 timer",
                    "timer_relay", "", specificity=3),
    SeriesSignature(r"\bG3PB\b|\bG3NA\b", "Omron", "G3 solid state relay",
                    "relay", "", specificity=3),
    SeriesSignature(r"\bCP1[EHLW]\b|\bCJ2[MH]\b|\bNX1P2\b|\bNJ\d{3}\b", "Omron",
                    "SYSMAC controller", "plc", "", specificity=3),
    SeriesSignature(r"\bNB\d\w*\b|\bNA5\b", "Omron", "NB/NA operator terminal",
                    "hmi", "", specificity=3),
    SeriesSignature(r"\bS8V[KS]\b|\bS8FS\b", "Omron", "S8 power supply",
                    "power_supply", "", specificity=3),
    SeriesSignature(r"\bG9S[AP]\b|\bG9SE\b", "Omron", "G9S safety relay unit",
                    "safety_relay", "", specificity=3),
    SeriesSignature(r"\bE3Z\b|\bE2E\b|\bE3F\d\b", "Omron", "E-series sensor",
                    "sensor", "", specificity=3),

    # ---------- Mitsubishi ----------
    SeriesSignature(r"\bFR[\s-]?[ADEF]\d{3}\w*\b", "Mitsubishi Electric",
                    "FREQROL variable frequency drive", "vfd", "", specificity=3),
    SeriesSignature(r"\bFX[35][UGS]\w*\b|\bFX\d[UN]\b", "Mitsubishi Electric",
                    "MELSEC FX controller", "plc", "", specificity=3),
    SeriesSignature(r"\bQ0\d[UH]\w*\b|\bR0\d[CP]\w*\b", "Mitsubishi Electric",
                    "MELSEC Q/R controller", "plc", "", specificity=3),
    SeriesSignature(r"\bGT\d{4}\b|\bGOT\d{4}\b", "Mitsubishi Electric",
                    "GOT operator terminal", "hmi", "", specificity=3),
    SeriesSignature(r"\bS[\s-]?N1[01]\b|\bSD[\s-]?N\d{1,2}\b", "Mitsubishi Electric",
                    "MS-N contactor", "contactor", "", specificity=2),
    SeriesSignature(r"\bMR[\s-]?J[45]\w*\b", "Mitsubishi Electric",
                    "MELSERVO servo amplifier", "servo_drive", "", specificity=3),

    # ---------- Phoenix Contact / WAGO / Weidmüller ----------
    SeriesSignature(r"\bQUINT\b|\bTRIO[\s-]?PS\b|\bSTEP[\s-]?PS\b|\bUNO[\s-]?PS\b",
                    "Phoenix Contact", "QUINT/TRIO power supply", "power_supply",
                    "", specificity=3),
    SeriesSignature(r"\b(?:UT|UK|PT|ST)\s?\d{1,2}(?:[,.]\d)?\b", "Phoenix Contact",
                    "CLIPLINE terminal block", "terminal_block", "", specificity=2),
    SeriesSignature(r"\bPLC[\s-]?RSC\b|\bREL[\s-]?MR\b", "Phoenix Contact",
                    "PLC-INTERFACE relay module", "relay", "", specificity=3),
    SeriesSignature(r"\bFL\s?SWITCH\b", "Phoenix Contact",
                    "FL industrial ethernet switch", "ethernet_switch", "",
                    specificity=3),
    SeriesSignature(r"\bWAGO\b|\b2[02]\d[\s-]\d{3}\b", "WAGO",
                    "TOPJOB S / 221 terminal block", "terminal_block", "",
                    specificity=2),
    SeriesSignature(r"\b750[\s-]\d{3}\b", "WAGO", "750 series remote IO",
                    "io_module", "", specificity=3),
    SeriesSignature(r"\bWDU\s?\d(?:[,.]\d)?\b|\bWEIDM[UÜ]LLER\b|\bA2C\b",
                    "Weidmüller", "Klippon/A-series terminal block",
                    "terminal_block", "", specificity=2),
    SeriesSignature(r"\bIE[\s-]SW\b|\bPROmax\b", "Weidmüller",
                    "IE-SW ethernet switch", "ethernet_switch", "", specificity=3),

    # ---------- Others ----------
    SeriesSignature(r"\bPNOZ\s?\w{1,6}\b", "Pilz", "PNOZ safety relay",
                    "safety_relay",
                    "Certified up to PL e / SIL 3 depending on variant.",
                    specificity=4),
    SeriesSignature(r"\b(?:MDR|NDR|DR|SDR|HDR|EDR)[\s-]?\d{2,3}[\s-]?\d{0,2}\b",
                    "MEAN WELL", "DIN rail power supply", "power_supply", "",
                    specificity=3),
    SeriesSignature(r"\bFINDER\b|\b(?:55|40|46|38)\.\d{2}\b", "Finder",
                    "series relay", "relay", "", specificity=2),
    SeriesSignature(r"\bDANFOSS\b|\bFC[\s-]?\d{3}\b|\bVLT\b", "Danfoss",
                    "VLT variable frequency drive", "vfd", "", specificity=3),
    SeriesSignature(r"\bYASKAWA\b|\b(?:GA|A1000|V1000)\d*\b", "Yaskawa",
                    "variable frequency drive", "vfd", "", specificity=2),
    SeriesSignature(r"\bDELTA\b|\bVFD\d{3}\w*\b|\bMS300\b", "Delta Electronics",
                    "VFD series drive", "vfd", "", specificity=2),
    SeriesSignature(r"\bLS\s?(?:MC|GMC)[\s-]?\d{1,3}\b|\bMETASOL\b", "LS Electric",
                    "MC contactor", "contactor", "", specificity=2),
    SeriesSignature(r"\bHAGER\b|\bMY\d{3}\b|\bNB[NS]\d{3}\b", "Hager",
                    "modular circuit breaker", "mcb", "", specificity=2),
    SeriesSignature(r"\bLEGRAND\b|\bDX³\b|\bDX3\b", "Legrand",
                    "DX³ modular device", "mcb", "", specificity=2),
    SeriesSignature(r"\bCHINT\b|\bNXC[\s-]?\d{2}\b|\bNXB\b", "CHINT",
                    "NXC contactor / NXB breaker", "contactor", "", specificity=2),
    SeriesSignature(r"\bSICK\b|\bWL\d{1,2}\b|\bWTB\d\b", "SICK", "photoelectric sensor",
                    "sensor", "", specificity=2),
    SeriesSignature(r"\bIFM\b|\bIF\d{4}\b|\bOGD\d{3}\b", "ifm electronic",
                    "sensor", "sensor", "", specificity=2),
)

_COMPILED: tuple[tuple[re.Pattern, SeriesSignature], ...] = tuple(
    (s.compiled(), s) for s in _SIGS
)

#: Bare manufacturer names — weaker evidence than a part number, but still
#: useful when the part number is unreadable.
_BRAND_WORDS: tuple[tuple[str, str], ...] = (
    (r"\bschneider\b|\btelemecanique\b|\bmerlin\s*gerin\b", "Schneider Electric"),
    (r"\bsiemens\b|\bsimatic\b|\bsirius\b", "Siemens"),
    (r"\babb\b", "ABB"),
    (r"\beaton\b|\bmoeller\b|\bklockner\b", "Eaton"),
    (r"\bomron\b", "Omron"),
    (r"\bmitsubishi\b|\bmelsec\b", "Mitsubishi Electric"),
    (r"\bphoenix\s*contact\b", "Phoenix Contact"),
    (r"\bwago\b", "WAGO"),
    (r"\bweidm[uü]ller\b", "Weidmüller"),
    (r"\bpilz\b", "Pilz"),
    (r"\bmean\s*well\b|\bmeanwell\b", "MEAN WELL"),
    (r"\bfinder\b", "Finder"),
    (r"\bdanfoss\b", "Danfoss"),
    (r"\byaskawa\b", "Yaskawa"),
    (r"\bdelta\b", "Delta Electronics"),
    (r"\bls\s*electric\b|\blsis\b", "LS Electric"),
    (r"\bhager\b", "Hager"),
    (r"\blegrand\b", "Legrand"),
    (r"\bchint\b", "CHINT"),
    (r"\bsick\b", "SICK"),
    (r"\bifm\b", "ifm electronic"),
    (r"\brockwell\b|\ballen[\s-]?bradley\b", "Rockwell Automation"),
    (r"\bfuji\b", "Fuji Electric"),
    (r"\bteco\b", "TECO"),
)


@dataclass
class NameplateMatch:
    manufacturer: Optional[str]
    family: Optional[str]
    part_number: Optional[str]
    #: Class the nameplate implies (may differ from the detector's class).
    implied_class: Optional[str]
    #: Does the nameplate corroborate the detector's class?
    agrees_with_detector: Optional[bool]
    note: str = ""
    text: str = ""

    def to_dict(self) -> dict:
        return {
            "manufacturer": self.manufacturer,
            "family": self.family,
            "part_number": self.part_number,
            "implied_class": self.implied_class,
            "agrees_with_detector": self.agrees_with_detector,
            "note": self.note,
            "text": self.text,
        }


EMPTY_MATCH = NameplateMatch(None, None, None, None, None, "", "")


def identify(text: str, detected_class: Optional[str] = None) -> NameplateMatch:
    """Match free OCR text against the part-number catalogue.

    ``detected_class`` is optional; when given, the result records whether the
    nameplate agrees with the detector, which the expert layer uses to raise or
    lower stated confidence — and to report a disagreement instead of silently
    preferring one source.
    """
    blob = " ".join(str(text or "").split())
    if not blob:
        return EMPTY_MATCH

    best: Optional[tuple[SeriesSignature, str]] = None
    for pattern, sig in _COMPILED:
        m = pattern.search(blob)
        if not m:
            continue
        if best is None or sig.specificity > best[0].specificity:
            best = (sig, m.group(0).strip())

    if best is not None:
        sig, part = best
        agrees = None
        if detected_class:
            agrees = (detected_class == sig.class_id
                      or _same_group(detected_class, sig.class_id))
        return NameplateMatch(
            manufacturer=sig.manufacturer, family=sig.family, part_number=part,
            implied_class=sig.class_id, agrees_with_detector=agrees,
            note=sig.note, text=blob[:200],
        )

    for pattern, brand in _BRAND_WORDS:
        if re.search(pattern, blob, re.IGNORECASE):
            return NameplateMatch(
                manufacturer=brand, family=None, part_number=None,
                implied_class=None, agrees_with_detector=None,
                note="Manufacturer read from the nameplate; part number not "
                     "resolvable from the visible text.",
                text=blob[:200],
            )
    return NameplateMatch(None, None, None, None, None, "", blob[:200])


def _same_group(a: str, b: str) -> bool:
    """Treat closely related classes as agreement (contactor vs relay block)."""
    from .postprocess import CONFUSABLE_GROUPS
    return any(a in g and b in g for g in CONFUSABLE_GROUPS)


# --------------------------------------------------------------------------
# Assigning OCR items to component boxes
# --------------------------------------------------------------------------

def text_for_boxes(ocr_items: Sequence[dict],
                   boxes: Sequence[Sequence[float]],
                   min_containment: float = 0.6) -> list[str]:
    """Distribute OCR text items to the component box that contains them.

    One OCR pass over the whole panel image, then geometric assignment — far
    cheaper and more accurate than re-running OCR on every crop, and it keeps
    text that straddles a boundary out of both components.
    """
    from .postprocess import containment

    out: list[list[str]] = [[] for _ in boxes]
    for item in ocr_items or []:
        tb = item.get("bbox")
        txt = str(item.get("text") or "").strip()
        if not txt or not tb or len(tb) != 4:
            continue
        best_i, best_c = None, min_containment
        for i, box in enumerate(boxes):
            c = containment(tb, box)
            if c >= best_c:
                best_i, best_c = i, c
        if best_i is not None:
            out[best_i].append(txt)
    return [" ".join(parts) for parts in out]


def catalogue_summary() -> dict:
    return {
        "signature_count": len(_SIGS),
        "manufacturers": sorted({s.manufacturer for s in _SIGS}),
        "classes_covered": sorted({s.class_id for s in _SIGS}),
    }


__all__ = ["SeriesSignature", "NameplateMatch", "EMPTY_MATCH", "identify",
           "text_for_boxes", "catalogue_summary"]
