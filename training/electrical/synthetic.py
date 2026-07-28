"""
Synthetic industrial panel generator.

Public datasets for industrial control-panel components are scarce, small, and
inconsistently labelled. Waiting for one is not a plan. This module builds
labelled training data from two sources, in order of value:

1. **Composition from real device crops** (``compose_from_crops``) — the useful
   mode. Given a library of cropped photographs of real devices, organised one
   directory per taxonomy class, it composites them onto realistic back plates
   with DIN rails and cable ducts, and applies the nuisance factors the
   specification calls out: lighting variation, camera angle (perspective),
   rotation, partial occlusion, dust, shadow and specular reflection. Ten
   photographs of a contactor become thousands of labelled training instances.

2. **Procedural stand-ins** (``synthesise_panel``) — devices drawn from the
   taxonomy's geometric priors with plausible housings, terminals, screws,
   labels and status LEDs. These are **not** a substitute for real photographs
   and will not by themselves produce a model that generalises to a real
   cabinet. They exist to exercise and *measure* the pipeline end to end —
   layout reasoning, the post-processing gate, panel-type inference, the metrics
   harness — on data with exact ground truth, and to pre-train layout priors
   before real images are available. Every artefact records
   ``"source": "procedural"`` so a synthetic-only evaluation can never be
   mistaken for a real-world result.

Output is standard YOLO detection format (``images/`` + ``labels/`` with
normalised ``cls cx cy w h``) plus a ``dataset.yaml`` and a ``classes.json`` that
pins the label order to :data:`rtsp_backend.electrical.taxonomy.CLASS_ORDER`.

Deterministic: every function takes a seed, so a dataset is reproducible.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import cv2
import numpy as np

from rtsp_backend.electrical import taxonomy as tax

# Devices worth synthesising procedurally: those with a reasonably canonical
# rectangular housing. Sensors, encoders and glands are too shape-variable for a
# procedural stand-in to be anything but misleading.
PROCEDURAL_CLASSES: tuple[str, ...] = (
    "mcb", "mccb", "rccb", "rcbo", "fuse_holder", "surge_protector",
    "contactor", "relay", "safety_relay", "timer_relay", "overload_relay",
    "motor_starter", "plc", "io_module", "logic_module", "signal_isolator",
    "power_supply", "ups", "vfd", "soft_starter", "energy_meter",
    "protection_relay", "ethernet_switch", "industrial_router",
    "terminal_block", "thermostat", "pf_controller", "ats_controller",
)

#: Plausible device palettes (BGR) seen in real cabinets.
_BODY_COLORS: tuple[tuple[int, int, int], ...] = (
    (52, 52, 54),      # near-black moulded housing
    (78, 78, 82),      # dark grey
    (128, 130, 132),   # light grey
    (168, 172, 176),   # RAL 7035 grey
    (58, 78, 104),     # dark blue-grey
    (48, 112, 152),    # blue
    (36, 132, 156),    # teal
    (28, 152, 188),    # light blue (Siemens-ish)
    (42, 42, 132),     # dark red
    (36, 116, 196),    # orange-ish
    (200, 202, 204),   # off white
)

_RAIL_COLOR = (176, 178, 180)
_DUCT_COLOR = (150, 152, 154)


@dataclass
class Instance:
    class_id: str
    box: tuple[float, float, float, float]


@dataclass
class GeneratedImage:
    image: np.ndarray
    instances: list[Instance] = field(default_factory=list)
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# nuisance factors
# --------------------------------------------------------------------------

def apply_lighting(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Uneven illumination + global gain, the single biggest real-world nuisance."""
    h, w = img.shape[:2]
    gain = rng.uniform(0.55, 1.45)
    # radial falloff from a random light centre
    cx, cy = rng.uniform(0, w), rng.uniform(0, h)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    dist /= max(1.0, dist.max())
    falloff = 1.0 - rng.uniform(0.15, 0.55) * dist
    out = img.astype(np.float32) * gain * falloff[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_shadow(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """A cast shadow band — cabinet doors and hands produce these constantly."""
    h, w = img.shape[:2]
    mask = np.ones((h, w), np.float32)
    pts = np.array([
        [rng.uniform(-0.2, 0.8) * w, 0],
        [rng.uniform(0.2, 1.2) * w, 0],
        [rng.uniform(0.2, 1.2) * w, h],
        [rng.uniform(-0.2, 0.8) * w, h],
    ], np.int32)
    cv2.fillPoly(mask, [pts], float(rng.uniform(0.45, 0.8)))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=w * 0.03 + 1)
    return np.clip(img.astype(np.float32) * mask[..., None], 0, 255).astype(np.uint8)


def apply_reflection(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Specular highlight — the reflection off a closed cabinet window."""
    h, w = img.shape[:2]
    overlay = np.zeros((h, w), np.float32)
    x0 = rng.uniform(0, w)
    thickness = int(max(6, w * rng.uniform(0.04, 0.14)))
    angle = rng.uniform(-40, 40)
    x1 = x0 + math.tan(math.radians(angle)) * h
    cv2.line(overlay, (int(x0), 0), (int(x1), h), 1.0, thickness)
    overlay = cv2.GaussianBlur(overlay, (0, 0), sigmaX=thickness * 0.6)
    strength = rng.uniform(30, 95)
    return np.clip(img.astype(np.float32) + overlay[..., None] * strength,
                   0, 255).astype(np.uint8)


def apply_dust(img: np.ndarray, rng: random.Random) -> np.ndarray:
    """Dust film + speckle — panels in plant rooms are never clean."""
    h, w = img.shape[:2]
    film = np.full((h, w, 3), rng.uniform(120, 190), np.float32)
    alpha = rng.uniform(0.04, 0.18)
    out = img.astype(np.float32) * (1 - alpha) + film * alpha
    speck = (np.random.default_rng(rng.randrange(1 << 30))
             .random((h, w)) > 0.9985).astype(np.float32)
    speck = cv2.GaussianBlur(speck, (3, 3), 0)[..., None] * 90.0
    return np.clip(out + speck, 0, 255).astype(np.uint8)


def apply_perspective(img: np.ndarray, boxes: Sequence[Instance],
                      rng: random.Random, strength: float = 0.06
                      ) -> tuple[np.ndarray, list[Instance]]:
    """Off-axis camera angle, with the boxes transformed consistently."""
    h, w = img.shape[:2]
    s = strength
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([
        [w * rng.uniform(0, s), h * rng.uniform(0, s)],
        [w * (1 - rng.uniform(0, s)), h * rng.uniform(0, s)],
        [w * (1 - rng.uniform(0, s)), h * (1 - rng.uniform(0, s))],
        [w * rng.uniform(0, s), h * (1 - rng.uniform(0, s))],
    ])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    out: list[Instance] = []
    for inst in boxes:
        x1, y1, x2, y2 = inst.box
        corners = np.float32([[[x1, y1], [x2, y1], [x2, y2], [x1, y2]]])
        t = cv2.perspectiveTransform(corners, M)[0]
        nx1, ny1 = float(t[:, 0].min()), float(t[:, 1].min())
        nx2, ny2 = float(t[:, 0].max()), float(t[:, 1].max())
        nx1, ny1 = max(0.0, nx1), max(0.0, ny1)
        nx2, ny2 = min(float(w), nx2), min(float(h), ny2)
        if nx2 - nx1 >= 5 and ny2 - ny1 >= 5:
            out.append(Instance(inst.class_id, (nx1, ny1, nx2, ny2)))
    return warped, out


def apply_blur_noise(img: np.ndarray, rng: random.Random) -> np.ndarray:
    if rng.random() < 0.45:
        k = rng.choice([3, 5])
        img = cv2.GaussianBlur(img, (k, k), 0)
    if rng.random() < 0.6:
        sigma = rng.uniform(2.0, 9.0)
        noise = np.random.default_rng(rng.randrange(1 << 30)).normal(
            0, sigma, img.shape)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:  # JPEG recompression artefacts
        q = rng.randint(38, 82)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def apply_occlusion(img: np.ndarray, rng: random.Random,
                    n: Optional[int] = None) -> np.ndarray:
    """Partial occlusion: cable bundles, wire markers, a technician's hand."""
    h, w = img.shape[:2]
    for _ in range(n if n is not None else rng.randint(1, 4)):
        kind = rng.random()
        color = (rng.randint(20, 220), rng.randint(20, 220), rng.randint(20, 220))
        if kind < 0.55:   # cable
            p0 = (rng.randint(0, w), rng.randint(0, h))
            p1 = (rng.randint(0, w), rng.randint(0, h))
            mid = ((p0[0] + p1[0]) // 2 + rng.randint(-80, 80),
                   (p0[1] + p1[1]) // 2 + rng.randint(-80, 80))
            pts = np.array([p0, mid, p1], np.int32)
            cv2.polylines(img, [pts], False, color,
                          rng.randint(3, 9), cv2.LINE_AA)
        else:             # opaque patch
            x, y = rng.randint(0, max(1, w - 40)), rng.randint(0, max(1, h - 40))
            bw, bh = rng.randint(20, max(24, w // 8)), rng.randint(15, max(20, h // 8))
            cv2.rectangle(img, (x, y), (x + bw, y + bh), color, -1)
    return img


# --------------------------------------------------------------------------
# procedural device rendering
# --------------------------------------------------------------------------

def _draw_screw(img: np.ndarray, cx: int, cy: int, r: int) -> None:
    cv2.circle(img, (cx, cy), r, (96, 98, 100), -1, cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (58, 58, 60), 1, cv2.LINE_AA)
    cv2.line(img, (cx - r + 1, cy), (cx + r - 1, cy), (40, 40, 42), 1, cv2.LINE_AA)


def render_device(class_id: str, w: int, h: int, rng: random.Random) -> np.ndarray:
    """Draw a plausible device housing for ``class_id`` at ``w×h`` pixels.

    Draws the features the taxonomy says matter — terminals, screws, labels,
    status LEDs, ventilation, dials, handles — so a model trained on these
    learns *structure* rather than a flat colour block. It is still a stand-in
    for a photograph, not a replacement.
    """
    w, h = max(8, int(w)), max(8, int(h))
    body = _BODY_COLORS[rng.randrange(len(_BODY_COLORS))]
    img = np.full((h, w, 3), body, np.uint8)

    # moulding: bevel + shading
    cv2.rectangle(img, (0, 0), (w - 1, h - 1),
                  tuple(int(c * 0.6) for c in body), 1)
    grad = np.linspace(1.12, 0.88, h, dtype=np.float32)[:, None, None]
    img = np.clip(img.astype(np.float32) * grad, 0, 255).astype(np.uint8)

    sp = tax.spec(class_id)
    term = max(2, min(w, h) // 12)

    def label_plate(frac_y: float = 0.42, frac_h: float = 0.26) -> None:
        ly, lh = int(h * frac_y), max(4, int(h * frac_h))
        lx, lw = int(w * 0.12), int(w * 0.76)
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (222, 224, 226), -1)
        cv2.rectangle(img, (lx, ly), (lx + lw, ly + lh), (150, 152, 154), 1)
        for i in range(max(1, lh // 5)):
            yy = ly + 2 + i * 5
            if yy + 1 < ly + lh:
                cv2.line(img, (lx + 3, yy), (lx + lw - 3, yy), (110, 112, 114), 1)

    def top_bottom_terminals(count: int) -> None:
        for i in range(count):
            cx = int(w * (i + 0.5) / count)
            for cy in (term + 2, h - term - 3):
                cv2.rectangle(img, (cx - term, cy - term), (cx + term, cy + term),
                              (140, 142, 144), -1)
                _draw_screw(img, cx, cy, max(1, term - 1))

    def vents(y0: float, y1: float) -> None:
        for yy in range(int(h * y0), int(h * y1), max(3, h // 22)):
            cv2.line(img, (int(w * 0.12), yy), (int(w * 0.88), yy),
                     tuple(int(c * 0.55) for c in body), 1)

    def led(cx: int, cy: int, color: tuple[int, int, int]) -> None:
        cv2.circle(img, (cx, cy), max(1, min(w, h) // 22), color, -1, cv2.LINE_AA)

    if class_id in ("mcb", "rccb", "rcbo", "fuse_holder", "surge_protector"):
        # modular device: toggle lever + single top/bottom terminal per pole
        poles = max(1, round(w / max(1, h) * 3))
        lever_w = max(3, w // (poles * 2))
        for p in range(poles):
            lx = int(w * (p + 0.5) / poles)
            cv2.rectangle(img, (lx - lever_w, int(h * 0.36)),
                          (lx + lever_w, int(h * 0.64)),
                          (36, 36, 38) if rng.random() < 0.5 else (30, 30, 160), -1)
        top_bottom_terminals(poles)
        if class_id in ("rccb", "rcbo") and h > 24:
            cv2.rectangle(img, (int(w * 0.55), int(h * 0.2)),
                          (int(w * 0.85), int(h * 0.32)), (40, 40, 190), -1)
    elif class_id in ("contactor", "motor_starter"):
        top_bottom_terminals(3)
        label_plate(0.38, 0.24)
        # coil block hint
        cv2.rectangle(img, (int(w * 0.05), int(h * 0.68)),
                      (int(w * 0.32), int(h * 0.82)),
                      tuple(int(c * 0.75) for c in body), -1)
    elif class_id == "overload_relay":
        top_bottom_terminals(3)
        # current dial + test/reset buttons
        cv2.circle(img, (int(w * 0.32), int(h * 0.5)), max(3, h // 6),
                   (210, 212, 214), -1, cv2.LINE_AA)
        cv2.circle(img, (int(w * 0.32), int(h * 0.5)), max(3, h // 6),
                   (90, 92, 94), 1, cv2.LINE_AA)
        led(int(w * 0.62), int(h * 0.42), (40, 40, 200))
        led(int(w * 0.76), int(h * 0.42), (40, 200, 60))
    elif class_id in ("relay", "safety_relay", "timer_relay", "protection_relay",
                      "signal_isolator", "thermostat"):
        label_plate(0.30, 0.30)
        for i in range(max(2, w // max(3, term * 3))):
            cx = int(w * (i + 0.5) / max(2, w // max(3, term * 3)))
            cv2.rectangle(img, (cx - term, h - 2 * term - 2),
                          (cx + term, h - 2), (140, 142, 144), -1)
        if class_id == "timer_relay":
            cv2.circle(img, (int(w * 0.5), int(h * 0.22)), max(3, h // 7),
                       (235, 235, 235), -1, cv2.LINE_AA)
        if class_id == "safety_relay":
            img[:] = np.clip(img.astype(np.float32) *
                             np.array([0.55, 1.05, 1.25]), 0, 255).astype(np.uint8)
        led(int(w * 0.2), int(h * 0.12), (40, 200, 60))
    elif class_id in ("plc", "io_module", "logic_module"):
        # terminal strip down one edge + LED bank + port
        n = max(4, h // max(4, term * 3))
        for i in range(n):
            cy = int(h * (i + 0.5) / n)
            cv2.rectangle(img, (w - 3 * term, cy - term), (w - term, cy + term),
                          (150, 152, 154), -1)
        for i in range(min(10, max(3, n // 2))):
            led(int(w * 0.18), int(h * (i + 0.6) / max(3, n // 2)),
                (60, 220, 90) if i % 3 else (40, 180, 220))
        cv2.rectangle(img, (int(w * 0.34), int(h * 0.06)),
                      (int(w * 0.62), int(h * 0.2)), (28, 28, 30), -1)
        label_plate(0.40, 0.22)
    elif class_id in ("vfd", "soft_starter"):
        vents(0.05, 0.30)
        vents(0.72, 0.96)
        # keypad + display
        cv2.rectangle(img, (int(w * 0.18), int(h * 0.34)),
                      (int(w * 0.82), int(h * 0.50)), (60, 90, 70), -1)
        cv2.rectangle(img, (int(w * 0.18), int(h * 0.34)),
                      (int(w * 0.82), int(h * 0.50)), (30, 30, 32), 1)
        for i in range(4):
            cv2.circle(img, (int(w * (0.28 + 0.15 * i)), int(h * 0.60)),
                       max(2, h // 26), (200, 202, 204), -1, cv2.LINE_AA)
        top_bottom_terminals(3)
    elif class_id in ("power_supply", "ups"):
        vents(0.08, 0.34)
        label_plate(0.40, 0.24)
        led(int(w * 0.5), int(h * 0.70), (60, 220, 90))
        for i in range(4):
            cx = int(w * (i + 0.5) / 4)
            cv2.rectangle(img, (cx - term, h - 2 * term - 2),
                          (cx + term, h - 2), (140, 142, 144), -1)
            _draw_screw(img, cx, h - term - 3, max(1, term - 1))
    elif class_id in ("energy_meter", "pf_controller", "ats_controller"):
        cv2.rectangle(img, (int(w * 0.1), int(h * 0.12)),
                      (int(w * 0.9), int(h * 0.55)), (40, 62, 48), -1)
        for i in range(3):
            cv2.line(img, (int(w * 0.16), int(h * (0.2 + 0.12 * i))),
                     (int(w * 0.84), int(h * (0.2 + 0.12 * i))),
                     (120, 220, 150), 1)
        for i in range(4):
            cv2.rectangle(img, (int(w * (0.14 + 0.2 * i)), int(h * 0.66)),
                          (int(w * (0.24 + 0.2 * i)), int(h * 0.82)),
                          (180, 182, 184), -1)
    elif class_id in ("ethernet_switch", "industrial_router"):
        n = max(4, w // max(6, term * 4))
        for i in range(n):
            cx = int(w * (i + 0.5) / n)
            cv2.rectangle(img, (cx - term, int(h * 0.55)),
                          (cx + term, int(h * 0.85)), (28, 28, 30), -1)
            led(cx, int(h * 0.42), (60, 220, 90) if i % 2 else (40, 180, 220))
        label_plate(0.08, 0.22)
    elif class_id == "terminal_block":
        n = max(3, w // max(4, term * 3))
        for i in range(n):
            cx = int(w * (i + 0.5) / n)
            cv2.line(img, (cx, 0), (cx, h), tuple(int(c * 0.7) for c in body), 1)
            _draw_screw(img, cx, int(h * 0.28), max(1, term))
            _draw_screw(img, cx, int(h * 0.74), max(1, term))
    else:
        label_plate()
        top_bottom_terminals(2)

    # light wear so the model does not key on perfectly clean surfaces
    if rng.random() < 0.5:
        img = apply_dust(img, rng)
    if sp.category == "drives" and rng.random() < 0.5:
        vents(0.05, 0.25)
    return img


def _plausible_size(class_id: str, panel_w: int, panel_h: int,
                    rng: random.Random) -> tuple[int, int]:
    """Sample a device size consistent with the taxonomy priors.

    Sampling *inside* the priors means the plausibility gate and the generator
    agree by construction, so a gate failure on synthetic data indicates a real
    bug rather than a disagreement about geometry.
    """
    sp = tax.spec(class_id)
    a_lo, a_hi = sp.rel_area
    ar_lo, ar_hi = sp.aspect_ratio
    panel_area = panel_w * panel_h
    # stay comfortably inside the band so augmentation cannot push us out
    area = panel_area * math.exp(rng.uniform(math.log(a_lo * 1.6),
                                             math.log(min(a_hi * 0.7, 0.25))))
    ar = math.exp(rng.uniform(math.log(max(ar_lo * 1.25, 0.05)),
                              math.log(min(ar_hi * 0.8, 6.0))))
    h = max(10, int(round(math.sqrt(area / max(ar, 1e-6)))))
    w = max(8, int(round(h * ar)))
    return min(w, panel_w // 2), min(h, panel_h // 2)


def synthesise_panel(width: int = 1024, height: int = 768,
                     classes: Optional[Sequence[str]] = None,
                     n_rows: Optional[int] = None,
                     seed: Optional[int] = None,
                     nuisance: bool = True) -> GeneratedImage:
    """Render one procedural panel with exact ground-truth boxes."""
    rng = random.Random(seed)
    pool = [c for c in (classes or PROCEDURAL_CLASSES) if c in tax.SPECS]
    if not pool:
        pool = list(PROCEDURAL_CLASSES)

    # back plate
    base = int(rng.uniform(150, 205))
    img = np.full((height, width, 3), (base, base + 2, base + 4), np.float32)
    # Signed grain must be added in float — casting a negative int16 to uint8
    # wraps to ~255 and produces salt speckle instead of paint texture.
    img += np.random.default_rng(rng.randrange(1 << 30)).normal(
        0.0, 3.0, (height, width, 3))
    img = np.clip(img, 0, 255).astype(np.uint8)

    rows = n_rows if n_rows is not None else rng.randint(3, 6)
    margin = int(height * 0.05)
    band = (height - 2 * margin) // rows
    instances: list[Instance] = []

    for r in range(rows):
        y_top = margin + r * band
        # DIN rail for this row
        rail_y = y_top + int(band * 0.62)
        rail_h = max(4, band // 12)
        cv2.rectangle(img, (int(width * 0.04), rail_y),
                      (int(width * 0.96), rail_y + rail_h), _RAIL_COLOR, -1)
        cv2.rectangle(img, (int(width * 0.04), rail_y),
                      (int(width * 0.96), rail_y + rail_h), (120, 122, 124), 1)
        for hx in range(int(width * 0.06), int(width * 0.95), 24):
            cv2.circle(img, (hx, rail_y + rail_h // 2), max(1, rail_h // 3),
                       (130, 132, 134), -1)

        # a cable duct between some rows
        if rng.random() < 0.45 and r < rows - 1:
            dy = y_top + int(band * 0.90)
            dh = max(6, band // 8)
            cv2.rectangle(img, (int(width * 0.03), dy),
                          (int(width * 0.97), dy + dh), _DUCT_COLOR, -1)
            for sx in range(int(width * 0.05), int(width * 0.96), 14):
                cv2.line(img, (sx, dy + 1), (sx, dy + dh - 1), (120, 122, 124), 1)

        # devices seated on the rail
        x = int(width * rng.uniform(0.05, 0.12))
        guard = 0
        while x < width * 0.92 and guard < 40:
            guard += 1
            cid = pool[rng.randrange(len(pool))]
            dw, dh = _plausible_size(cid, width, height, rng)
            # Fit into the row band and the width budget by scaling BOTH sides by
            # the same factor. Clamping them independently would distort the
            # aspect ratio outside the taxonomy prior, and the generator must
            # never emit a box its own plausibility gate would reject.
            fit = min(1.0, (band * 0.85) / max(1, dh),
                      (width * 0.30) / max(1, dw))
            if fit < 1.0:
                dw, dh = max(8, int(dw * fit)), max(8, int(dh * fit))
            if x + dw > width * 0.95:
                break
            dev = render_device(cid, dw, dh, rng)
            y = rail_y + rail_h - dh + int(dh * rng.uniform(0.35, 0.55))
            y = max(0, min(y, height - dh - 1))
            img[y:y + dh, x:x + dw] = dev
            # contact shadow so devices don't look pasted on
            cv2.rectangle(img, (x, y + dh), (x + dw, min(height - 1, y + dh + 3)),
                          (int(base * 0.7),) * 3, -1)
            instances.append(Instance(cid, (float(x), float(y),
                                            float(x + dw), float(y + dh))))
            x += dw + int(rng.uniform(2, width * 0.05))

    meta = {"source": "procedural", "seed": seed, "rows": rows,
            "class_pool": list(pool)}
    if not nuisance:
        return GeneratedImage(img, instances, meta)

    if rng.random() < 0.7:
        img, instances = apply_perspective(img, instances, rng)
    if rng.random() < 0.35:
        img = apply_occlusion(img, rng)
    img = apply_lighting(img, rng)
    if rng.random() < 0.35:
        img = apply_shadow(img, rng)
    if rng.random() < 0.25:
        img = apply_reflection(img, rng)
    if rng.random() < 0.3:
        img = apply_dust(img, rng)
    img = apply_blur_noise(img, rng)
    return GeneratedImage(img, instances, meta)


# --------------------------------------------------------------------------
# composition from real crops
# --------------------------------------------------------------------------

def load_crop_library(root: str) -> dict[str, list[str]]:
    """Index a crop library: ``root/<class_id or alias>/*.png|jpg``.

    Directory names are resolved through the taxonomy, so a folder called
    ``"magnetic contactor"`` or ``"MCCB"`` lands on the right canonical class.
    """
    lib: dict[str, list[str]] = {}
    if not os.path.isdir(root):
        return lib
    for name in sorted(os.listdir(root)):
        path = os.path.join(root, name)
        if not os.path.isdir(path):
            continue
        cid = tax.resolve(name)
        if cid is None:
            continue
        files = [os.path.join(path, f) for f in sorted(os.listdir(path))
                 if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))]
        if files:
            lib.setdefault(cid, []).extend(files)
    return lib


def compose_from_crops(library: dict[str, list[str]], width: int = 1024,
                       height: int = 768, seed: Optional[int] = None,
                       max_devices: int = 26,
                       nuisance: bool = True) -> GeneratedImage:
    """Composite real device crops onto a synthetic back plate.

    This is the mode that produces training data capable of generalising: the
    device appearance is real, only the arrangement and nuisance factors are
    synthetic. Ten crops per class is enough to start.
    """
    rng = random.Random(seed)
    if not library:
        raise ValueError("crop library is empty — populate it with real device "
                         "photographs, one directory per component class")

    scaffold = synthesise_panel(width, height, classes=list(library.keys()),
                                seed=seed, nuisance=False)
    img = scaffold.image
    instances: list[Instance] = []
    slots = scaffold.instances[:max_devices]

    for slot in slots:
        cid = slot.class_id
        files = library.get(cid)
        if not files:
            continue
        crop = cv2.imread(files[rng.randrange(len(files))], cv2.IMREAD_COLOR)
        if crop is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in slot.box]
        tw, th = max(8, x2 - x1), max(8, y2 - y1)
        # keep the crop's own aspect ratio; fit it into the slot
        ch, cw = crop.shape[:2]
        scale = min(tw / cw, th / ch)
        nw, nh = max(8, int(cw * scale)), max(8, int(ch * scale))
        resized = cv2.resize(crop, (nw, nh),
                             interpolation=cv2.INTER_AREA if scale < 1
                             else cv2.INTER_LINEAR)
        if rng.random() < 0.5:
            angle = rng.uniform(-6, 6)
            M = cv2.getRotationMatrix2D((nw / 2, nh / 2), angle, 1.0)
            resized = cv2.warpAffine(resized, M, (nw, nh),
                                     borderMode=cv2.BORDER_REPLICATE)
        px, py = x1, y2 - nh
        py = max(0, min(py, height - nh))
        px = max(0, min(px, width - nw))
        img[py:py + nh, px:px + nw] = resized
        instances.append(Instance(cid, (float(px), float(py),
                                        float(px + nw), float(py + nh))))

    meta = {"source": "composed_real_crops", "seed": seed,
            "classes": sorted(library.keys())}
    if not nuisance:
        return GeneratedImage(img, instances, meta)

    if rng.random() < 0.75:
        img, instances = apply_perspective(img, instances, rng)
    if rng.random() < 0.4:
        img = apply_occlusion(img, rng)
    img = apply_lighting(img, rng)
    if rng.random() < 0.4:
        img = apply_shadow(img, rng)
    if rng.random() < 0.3:
        img = apply_reflection(img, rng)
    if rng.random() < 0.35:
        img = apply_dust(img, rng)
    img = apply_blur_noise(img, rng)
    return GeneratedImage(img, instances, meta)


# --------------------------------------------------------------------------
# dataset writing
# --------------------------------------------------------------------------

def to_yolo_lines(instances: Sequence[Instance], width: int, height: int,
                  class_index: Optional[dict[str, int]] = None) -> list[str]:
    idx = class_index or tax.class_index()
    lines: list[str] = []
    for inst in instances:
        if inst.class_id not in idx:
            continue
        x1, y1, x2, y2 = inst.box
        cx = ((x1 + x2) / 2.0) / width
        cy = ((y1 + y2) / 2.0) / height
        w = (x2 - x1) / width
        h = (y2 - y1) / height
        if w <= 0 or h <= 0:
            continue
        lines.append(f"{idx[inst.class_id]} "
                     f"{min(max(cx, 0), 1):.6f} {min(max(cy, 0), 1):.6f} "
                     f"{min(max(w, 0), 1):.6f} {min(max(h, 0), 1):.6f}")
    return lines


def write_dataset(out_dir: str, n_train: int = 400, n_val: int = 80,
                  width: int = 1024, height: int = 768,
                  crop_library: Optional[str] = None,
                  seed: int = 1234,
                  progress=None) -> dict:
    """Generate a YOLO-format dataset on disk and return a manifest."""
    library = load_crop_library(crop_library) if crop_library else {}
    idx = tax.class_index()
    counts: dict[str, int] = {}
    splits = {"train": n_train, "val": n_val}
    written = {"train": 0, "val": 0}

    for split, n in splits.items():
        img_dir = os.path.join(out_dir, "images", split)
        lbl_dir = os.path.join(out_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for i in range(int(n)):
            s = seed + (0 if split == "train" else 10 ** 6) + i
            gen = (compose_from_crops(library, width, height, seed=s)
                   if library else
                   synthesise_panel(width, height, seed=s))
            stem = f"{split}_{i:06d}"
            cv2.imwrite(os.path.join(img_dir, stem + ".jpg"), gen.image,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            lines = to_yolo_lines(gen.instances, gen.image.shape[1],
                                  gen.image.shape[0], idx)
            with open(os.path.join(lbl_dir, stem + ".txt"), "w",
                      encoding="utf-8") as fh:
                fh.write("\n".join(lines) + ("\n" if lines else ""))
            for inst in gen.instances:
                counts[inst.class_id] = counts.get(inst.class_id, 0) + 1
            written[split] += 1
            if progress and (i + 1) % 25 == 0:
                progress(split, i + 1, int(n))

    names = {i: cid for cid, i in idx.items()}
    yaml_path = os.path.join(out_dir, "dataset.yaml")
    with open(yaml_path, "w", encoding="utf-8") as fh:
        fh.write("# Generated by training.electrical.synthetic — do not edit by hand.\n")
        fh.write(f"path: {os.path.abspath(out_dir)}\n")
        fh.write("train: images/train\nval: images/val\n")
        fh.write(f"nc: {len(idx)}\nnames:\n")
        for i in sorted(names):
            fh.write(f"  {i}: {names[i]}\n")

    manifest = {
        "source": "composed_real_crops" if library else "procedural",
        "images": written,
        "instances": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "instance_total": sum(counts.values()),
        "class_count": len(idx),
        "classes": list(tax.CLASS_ORDER),
        "image_size": [width, height],
        "seed": seed,
        "dataset_yaml": yaml_path,
        "warning": (None if library else
                    "PROCEDURAL DATA ONLY. Metrics measured on this dataset "
                    "validate the pipeline, not real-world accuracy. Supply a "
                    "crop library of real device photographs before drawing any "
                    "conclusion about field performance."),
    }
    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    with open(os.path.join(out_dir, "classes.json"), "w", encoding="utf-8") as fh:
        json.dump({"classes": list(tax.CLASS_ORDER)}, fh, indent=2)
    return manifest


__all__ = [
    "PROCEDURAL_CLASSES", "Instance", "GeneratedImage", "apply_lighting",
    "apply_shadow", "apply_reflection", "apply_dust", "apply_perspective",
    "apply_blur_noise", "apply_occlusion", "render_device", "synthesise_panel",
    "load_crop_library", "compose_from_crops", "to_yolo_lines", "write_dataset",
]
