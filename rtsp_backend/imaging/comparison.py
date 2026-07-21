"""
Image-to-image comparison — detect every difference between two images.

Pipeline (all classical, weights-free, CPU — works on any image):

1. **Registration / perspective compensation** — ORB features + RANSAC
   homography warps the *current* image into the *reference* frame, so camera
   pose / zoom / small perspective shifts are not mistaken for changes. Falls
   back to a plain resize when too few matches (and says so).
2. **Lighting compensation** — the warped current image's luminance histogram
   is matched to the reference before differencing, so exposure changes don't
   flood the diff.
3. **Structural difference** — SSIM (skimage) produces a per-pixel similarity
   map; ``1 - SSIM`` is thresholded, denoised with morphology (noise tolerance),
   and connected into changed regions. Tiny specks are dropped; both small and
   large changed regions are reported with bounding boxes.
4. **Object-level diff** — objects detected in each image are matched by
   position (in the reference frame) + label → missing / new / moved / changed.
5. **Colour diff** — dominant colour shift inside each changed region.
6. **Text diff** — OCR of both images is token-diffed → added / removed text.
7. **Similarity score** — a weighted blend of SSIM, ORB inlier ratio, colour-
   histogram correlation and perceptual-hash distance, as a 0–100 %.

Returns a JSON-serialisable report plus the raw arrays needed to render the
overlay/heatmap (stripped before persistence).
"""

from __future__ import annotations

import math
from typing import Any, Optional

import cv2
import numpy as np

from . import analysis as _an

try:
    from skimage.metrics import structural_similarity as _ssim
    _HAVE_SSIM = True
except Exception:  # pragma: no cover
    _HAVE_SSIM = False


# --------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------

def align(reference_bgr: np.ndarray, current_bgr: np.ndarray,
          min_matches: int = 12) -> dict[str, Any]:
    """Warp ``current`` into ``reference``'s frame. Returns warped image +
    homography + inlier ratio + note. Never raises."""
    H, ref = reference_bgr, current_bgr
    rh, rw = H.shape[:2]
    info: dict[str, Any] = {"ok": False, "homography": None, "inliers": 0,
                            "match_ratio": 0.0, "note": None}
    try:
        orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
        g1 = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2GRAY)
        k1, d1 = orb.detectAndCompute(g1, None)
        k2, d2 = orb.detectAndCompute(g2, None)
        if d1 is None or d2 is None or len(d1) < min_matches or len(d2) < min_matches:
            raise ValueError("too few features")
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        good = [m for m, n in
                (p for p in bf.knnMatch(d2, d1, k=2) if len(p) == 2)
                if m.distance < 0.75 * n.distance]
        if len(good) < min_matches:
            raise ValueError(f"only {len(good)} good matches")
        src = np.float32([k2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([k1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if M is None:
            raise ValueError("homography failed")
        warped = cv2.warpPerspective(current_bgr, M, (rw, rh))
        inliers = int(mask.sum()) if mask is not None else 0
        info.update(ok=True, homography=M.tolist(), inliers=inliers,
                    match_ratio=round(inliers / max(1, len(good)), 3))
        return {**info, "warped": warped, "n_good": len(good)}
    except Exception as exc:
        info["note"] = f"registration fell back to resize ({exc})"
        warped = cv2.resize(current_bgr, (rw, rh), interpolation=cv2.INTER_AREA)
        return {**info, "warped": warped, "n_good": 0}


def _match_luma(reference_bgr: np.ndarray, current_bgr: np.ndarray) -> np.ndarray:
    """Match the current image's brightness to the reference (lighting comp)."""
    ref_l = cv2.cvtColor(reference_bgr, cv2.COLOR_BGR2LAB)[:, :, 0]
    lab = cv2.cvtColor(current_bgr, cv2.COLOR_BGR2LAB)
    cur_l = lab[:, :, 0]
    rm, rs = float(ref_l.mean()), float(ref_l.std()) or 1.0
    cm, cs = float(cur_l.mean()), float(cur_l.std()) or 1.0
    adj = ((cur_l.astype(np.float32) - cm) * (rs / cs) + rm)
    lab[:, :, 0] = np.clip(adj, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


# --------------------------------------------------------------------------
# difference regions
# --------------------------------------------------------------------------

def _diff_regions(ref: np.ndarray, cur: np.ndarray, params: dict):
    """Return (score_map 0..1 similarity, list of changed-region bboxes)."""
    g1 = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
    if _HAVE_SSIM:
        score, smap = _ssim(g1, g2, full=True)
        diff = (1.0 - smap)
    else:  # fallback: absolute difference
        diff = cv2.absdiff(g1, g2).astype(np.float32) / 255.0
        score = 1.0 - float(diff.mean())
    diff_u8 = np.clip(diff * 255, 0, 255).astype(np.uint8)
    diff_u8 = cv2.GaussianBlur(diff_u8, (5, 5), 0)  # noise tolerance
    thr = int(params.get("diff_thresh", 45))
    _, mask = cv2.threshold(diff_u8, thr, 255, cv2.THRESH_BINARY)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = g1.shape[:2]
    min_area = max(64, int(params.get("min_area_frac", 0.0004) * h * w))
    regions = []
    for c in contours:
        x, y, bw, bh = cv2.boundingRect(c)
        area = bw * bh
        if area < min_area:
            continue
        roi = diff_u8[y:y + bh, x:x + bw]
        intensity = float(roi.mean()) / 255.0
        regions.append({"bbox": [x, y, x + bw, y + bh], "area": int(area),
                        "area_frac": round(area / (h * w), 4),
                        "intensity": round(intensity, 3)})
    regions.sort(key=lambda r: -r["area"])
    return float(score), diff_u8, regions


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def _warp_box(H, box, size):
    if H is None:
        return box
    M = np.array(H, dtype=np.float64)
    pts = np.array([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]],
                   dtype=np.float64).reshape(-1, 1, 2)
    w = cv2.perspectiveTransform(pts, M).reshape(-1, 2)
    x1, y1 = w[:, 0].min(), w[:, 1].min()
    x2, y2 = w[:, 0].max(), w[:, 1].max()
    return [float(x1), float(y1), float(x2), float(y2)]


def _object_diff(ref_objs, cur_objs, H, size, move_thresh) -> list[dict]:
    """missing / new / moved / changed objects, comparing in the ref frame."""
    diffs: list[dict] = []
    cur_mapped = []
    for o in cur_objs:
        cur_mapped.append({**o, "bbox_ref": _warp_box(H, o["bbox"], size)})
    used = set()
    for ro in ref_objs:
        best_j, best_iou = None, 0.25
        for j, co in enumerate(cur_mapped):
            if j in used:
                continue
            i = _iou(ro["bbox"], co["bbox_ref"])
            if i > best_iou:
                best_iou, best_j = i, j
        if best_j is None:
            diffs.append({"diff_type": "missing_object", "severity": "major",
                          "detail": f"missing {ro['label']}", "confidence": round(float(ro.get('confidence', 0.5)), 3),
                          "bbox": ro["bbox"]})
            continue
        used.add(best_j)
        co = cur_mapped[best_j]
        if co["label"] != ro["label"]:
            diffs.append({"diff_type": "changed_object", "severity": "major",
                          "detail": f"{ro['label']} → {co['label']}",
                          "confidence": 0.7, "bbox": ro["bbox"]})
            continue
        rcx, rcy = (ro["bbox"][0] + ro["bbox"][2]) / 2, (ro["bbox"][1] + ro["bbox"][3]) / 2
        ccx, ccy = (co["bbox_ref"][0] + co["bbox_ref"][2]) / 2, (co["bbox_ref"][1] + co["bbox_ref"][3]) / 2
        shift = math.hypot(rcx - ccx, rcy - ccy)
        if shift > move_thresh:
            diffs.append({"diff_type": "moved_object", "severity": "minor",
                          "detail": f"{ro['label']} moved ~{shift:.0f}px",
                          "confidence": round(min(1.0, shift / (3 * move_thresh)), 3),
                          "bbox": ro["bbox"]})
    for j, co in enumerate(cur_mapped):
        if j not in used:
            diffs.append({"diff_type": "new_object", "severity": "major",
                          "detail": f"new {co['label']}", "confidence": round(float(co.get('confidence', 0.5)), 3),
                          "bbox": co["bbox_ref"]})
    return diffs


def _region_color_text(ref, cur, regions) -> list[dict]:
    """Attach a colour-change note to changed regions with a clear colour shift."""
    out = []
    for r in regions:
        x1, y1, x2, y2 = r["bbox"]
        rc = _an.dominant_colors(ref[y1:y2, x1:x2], k=1) if (y2 > y1 and x2 > x1) else []
        cc = _an.dominant_colors(cur[y1:y2, x1:x2], k=1) if (y2 > y1 and x2 > x1) else []
        detail = "region changed"
        dtype = "region_changed"
        if rc and cc and rc[0]["name"] != cc[0]["name"]:
            detail = f"colour {rc[0]['name']} → {cc[0]['name']}"
            dtype = "color_change"
        sev = "major" if r["area_frac"] > 0.02 else "minor"
        out.append({"diff_type": dtype, "severity": sev, "detail": detail,
                    "confidence": round(min(1.0, 0.4 + r["intensity"]), 3),
                    "bbox": r["bbox"]})
    return out


def _text_diff(ref_text: str, cur_text: str) -> list[dict]:
    def toks(t):
        return {w.strip(".,:;|").lower() for w in t.split() if len(w.strip(".,:;|")) >= 2}
    rt, ct = toks(ref_text), toks(cur_text)
    diffs = []
    removed = rt - ct
    added = ct - rt
    if removed:
        diffs.append({"diff_type": "text_change", "severity": "minor",
                      "detail": "text removed: " + ", ".join(sorted(removed))[:80],
                      "confidence": 0.6, "bbox": None})
    if added:
        diffs.append({"diff_type": "text_change", "severity": "minor",
                      "detail": "text added: " + ", ".join(sorted(added))[:80],
                      "confidence": 0.6, "bbox": None})
    return diffs


# --------------------------------------------------------------------------
# top-level
# --------------------------------------------------------------------------

DEFAULTS = {"diff_thresh": 45, "min_area_frac": 0.0004, "move_thresh": 26.0}


def compare(reference_bgr: np.ndarray, current_bgr: np.ndarray,
            ai_manager=None, params: Optional[dict] = None) -> dict[str, Any]:
    p = {**DEFAULTS, **(params or {})}
    rh, rw = reference_bgr.shape[:2]

    reg = align(reference_bgr, current_bgr)
    warped = reg["warped"]
    warped_lit = _match_luma(reference_bgr, warped)

    score, diff_map, regions = _diff_regions(reference_bgr, warped_lit, p)

    ref_an = _an.analyze(reference_bgr, ai_manager)
    cur_an = _an.analyze(current_bgr, ai_manager)
    H = reg["homography"]

    diffs: list[dict] = []
    diffs += _object_diff(ref_an["objects"], cur_an["objects"], H, (rw, rh), p["move_thresh"])
    diffs += _region_color_text(reference_bgr, warped_lit, regions)
    diffs += _text_diff(ref_an["ocr"]["text"], cur_an["ocr"]["text"])

    # similarity: blend SSIM, ORB inlier ratio, colour-hist correlation, pHash
    hist_sim = _hist_corr(reference_bgr, warped_lit)
    phash_sim = _phash_sim(ref_an.get("phash"), cur_an.get("phash"))
    orb_sim = reg["match_ratio"] if reg["ok"] else 0.4
    ssim_sim = max(0.0, min(1.0, (score + 1) / 2)) if score < 0 else max(0.0, min(1.0, score))
    weights = {"ssim": 0.5, "hist": 0.2, "orb": 0.15, "phash": 0.15}
    similarity = (weights["ssim"] * ssim_sim + weights["hist"] * hist_sim +
                  weights["orb"] * orb_sim + weights["phash"] * phash_sim)
    similarity_pct = round(float(similarity) * 100, 2)

    n_major = sum(1 for d in diffs if d["severity"] == "major")
    status = "identical" if not diffs else ("major" if n_major else "minor")

    result = {
        "similarity": similarity_pct,
        "status": status,
        "n_diffs": len(diffs),
        "differences": diffs,
        "changed_regions": regions,
        "registration": {k: reg[k] for k in ("ok", "inliers", "match_ratio", "note")},
        "scores": {"ssim": round(ssim_sim, 3), "hist": round(hist_sim, 3),
                   "orb": round(orb_sim, 3), "phash": round(phash_sim, 3)},
        "reference_analysis": _slim(ref_an),
        "current_analysis": _slim(cur_an),
        "summary": _summarize(similarity_pct, diffs, status),
        "notes": list(dict.fromkeys(ref_an["notes"] + cur_an["notes"])),
        # raw arrays for rendering — stripped before DB persistence
        "_warped": warped_lit, "_diff_map": diff_map,
    }
    return result


def _slim(an: dict) -> dict:
    return {k: an[k] for k in ("image_size", "object_total", "object_counts",
                               "dominant_colors", "tags", "summary")}


def _summarize(sim, diffs, status) -> str:
    from collections import Counter
    if not diffs:
        return f"Images are {sim:.2f}% similar — no significant differences detected."
    c = Counter(d["diff_type"] for d in diffs)
    top = "; ".join(f"{n} {t.replace('_', ' ')}" for t, n in c.most_common())
    return f"{sim:.2f}% similar ({status}). {len(diffs)} difference(s): {top}."


def _hist_corr(a: np.ndarray, b: np.ndarray) -> float:
    ha = cv2.calcHist([cv2.cvtColor(a, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
    hb = cv2.calcHist([cv2.cvtColor(b, cv2.COLOR_BGR2HSV)], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(ha, ha); cv2.normalize(hb, hb)
    return max(0.0, min(1.0, (float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL)) + 1) / 2))


def _phash_sim(a: Optional[str], b: Optional[str]) -> float:
    if not a or not b:
        return 0.5
    try:
        import imagehash  # type: ignore
        d = imagehash.hex_to_hash(a) - imagehash.hex_to_hash(b)
        return max(0.0, 1.0 - d / 64.0)
    except Exception:
        return 0.5
