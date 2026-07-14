"""
Per-frame AI pipeline.

For a given camera frame it runs whichever tasks are enabled (face recognition,
object detection, component detection, wire analysis), draws overlays on a copy
of the frame, persists detections/events, and returns a structured result plus
the annotated JPEG. The MJPEG "AI stream" and the snapshot-with-overlay both use
this. Throttled so inference doesn't run on every single frame at full rate.
"""

from __future__ import annotations

import time
from typing import Any, Optional

import cv2
import numpy as np

from .base import BBox, Detection, Wire
from .tracker import MultiClassTracker

# BGR colours per overlay kind
_COLORS = {
    "face_known": (80, 200, 80),
    "face_unknown": (60, 60, 220),
    "object": (230, 180, 40),
    "component": (200, 120, 240),
    "wire_ok": (0, 200, 200),
    "wire_bad": (0, 0, 255),
    "fire": (0, 128, 255),
    "weapon": (0, 0, 255),
    "violence": (0, 0, 200),
    "fall": (0, 90, 255),
    "ppe_ok": (80, 200, 80),
    "ppe_bad": (0, 0, 255),
    "human": (240, 160, 40),
    "vehicle": (200, 200, 40),
}

# Additional detection modules processed by the generic event loop.
# alert=True -> high-priority event that always saves a snapshot.
_EVENT_TASKS = {
    "fire":     {"alert": True,  "snapshot": True},
    "weapon":   {"alert": True,  "snapshot": True},
    "violence": {"alert": True,  "snapshot": True},
    "fall":     {"alert": True,  "snapshot": True},
    "ppe":      {"alert": False, "snapshot": True},
    "human":    {"alert": False, "snapshot": False},
    "vehicle":  {"alert": False, "snapshot": False},
}


def _draw_box(img, box, label, color, conf=None):
    x1, y1, x2, y2 = [int(v) for v in box.as_list()]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    text = label if conf is None else f"{label} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, text, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (255, 255, 255), 1, cv2.LINE_AA)


class AIPipeline:
    def __init__(self, db, ai_manager, event_sink=None, data_dir: str = "data",
                 min_interval: float = 0.2) -> None:
        self.db = db
        self.ai = ai_manager
        self._emit = event_sink or (lambda ev: None)
        self.data_dir = data_dir
        self.min_interval = min_interval
        self._last_run: dict[str, float] = {}
        self._last_result: dict[str, dict] = {}
        # de-dup identical events within this window (seconds)
        self._event_dedup: dict[str, float] = {}
        # per-(camera, task) trackers giving stable IDs to detections
        self._trackers: dict[str, MultiClassTracker] = {}

    def _tracker(self, camera_id: str, task: str) -> MultiClassTracker:
        key = f"{camera_id}:{task}"
        t = self._trackers.get(key)
        if t is None:
            t = MultiClassTracker(min_hits=2, max_age=30, iou_threshold=0.3)
            self._trackers[key] = t
        return t

    # -- helpers -----------------------------------------------------------

    def _save_snapshot(self, frame, prefix: str) -> Optional[str]:
        import os
        rel = f"snapshots/{prefix}_{int(time.time()*1000)}.jpg"
        path = os.path.join(self.data_dir, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if ok:
            with open(path, "wb") as fh:
                fh.write(buf.tobytes())
            return rel
        return None

    def _log_event(self, etype, camera_id, camera_name, label, conf,
                   employee_id=None, snapshot=None, payload=None, dedup_key=None):
        import json
        now = time.time()
        if dedup_key:
            last = self._event_dedup.get(dedup_key, 0)
            if now - last < 10.0:
                return
            self._event_dedup[dedup_key] = now
        eid = self.db.insert(
            "INSERT INTO events(type,camera_id,camera_name,label,confidence,"
            "employee_id,snapshot,payload,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
            (etype, camera_id, camera_name, label, conf, employee_id, snapshot,
             json.dumps(payload) if payload else None, now),
        )
        self._emit({
            "type": "ai_event", "event_id": eid, "event_type": etype,
            "camera_id": camera_id, "camera_name": camera_name, "label": label,
            "confidence": conf, "employee_id": employee_id,
            "snapshot": snapshot, "timestamp": now,
        })

    # -- main --------------------------------------------------------------

    def process(self, camera_id: str, camera_name: str, frame: np.ndarray,
                annotate: bool = True, force: bool = False) -> dict:
        now = time.monotonic()
        if not force and (now - self._last_run.get(camera_id, 0)) < self.min_interval:
            return self._last_result.get(camera_id, {"faces": [], "objects": [],
                                                     "components": [], "wires": []})
        self._last_run[camera_id] = now

        annotated = frame.copy() if annotate else None
        result: dict[str, Any] = {"faces": [], "objects": [], "components": [],
                                  "wires": [], "camera_id": camera_id}

        # --- face recognition ---
        if self.ai.is_enabled("face") and self.ai.face_service is not None:
            t0 = time.monotonic()
            try:
                faces = self.ai.face_service.recognize_frame(frame)
            except Exception as exc:
                faces = []
                result["face_error"] = str(exc)
            self.ai.record_infer("face", (time.monotonic() - t0) * 1000)
            for f in faces:
                result["faces"].append(f.to_dict())
                known = f.employee_id is not None
                if annotated is not None:
                    _draw_box(annotated, f.bbox, f.label,
                              _COLORS["face_known"] if known else _COLORS["face_unknown"],
                              f.confidence)
                if known:
                    self._log_event("face_recognized", camera_id, camera_name,
                                    f.label, f.confidence, employee_id=f.employee_id,
                                    dedup_key=f"face:{camera_id}:{f.employee_id}")
                else:
                    snap = self._save_snapshot(frame, "unknown")
                    self._log_event("unknown_person", camera_id, camera_name,
                                    "Unknown Person", f.confidence, snapshot=snap,
                                    dedup_key=f"unknown:{camera_id}")

        # --- generic object detection ---
        if self.ai.is_enabled("detection"):
            det = self.ai.backend("detection")
            t0 = time.monotonic()
            try:
                objs = det.infer(frame)
            except Exception as exc:
                objs = []
                result["detection_error"] = str(exc)
            self.ai.record_infer("detection", (time.monotonic() - t0) * 1000)
            id_map = {}
            try:
                id_map = self._tracker(camera_id, "detection").update(objs)
            except Exception:
                id_map = {}
            for i, o in enumerate(objs):
                od = o.to_dict()
                tid = id_map.get(i)
                if tid is not None:
                    od["track_id"] = tid
                result["objects"].append(od)
                if annotated is not None:
                    lbl = o.label if tid is None else f"#{tid} {o.label}"
                    _draw_box(annotated, o.bbox, lbl, _COLORS["object"], o.confidence)

        # --- electrical component detection ---
        comps: list[Detection] = []
        if self.ai.is_enabled("components"):
            cd = self.ai.backend("components")
            t0 = time.monotonic()
            try:
                comps = cd.infer(frame)
            except Exception as exc:
                comps = []
                result["components_error"] = str(exc)
            self.ai.record_infer("components", (time.monotonic() - t0) * 1000)
            for c in comps:
                d = c.to_dict()
                d["position"] = self._panel_position(c.bbox, frame.shape)
                result["components"].append(d)
                if annotated is not None:
                    _draw_box(annotated, c.bbox, c.label, _COLORS["component"], c.confidence)
                self.db.insert(
                    "INSERT INTO components(camera_id,name,comp_type,confidence,"
                    "x1,y1,x2,y2,position,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (camera_id, c.label, c.label, c.confidence, c.bbox.x1, c.bbox.y1,
                     c.bbox.x2, c.bbox.y2, d["position"], time.time()),
                )

        # --- wire analysis ---
        if self.ai.is_enabled("wires"):
            wa = self.ai.backend("wires")
            t0 = time.monotonic()
            try:
                wires = wa.analyze(frame, comps)
            except Exception as exc:
                wires = []
                result["wires_error"] = str(exc)
            self.ai.record_infer("wires", (time.monotonic() - t0) * 1000)
            for w in wires:
                result["wires"].append(w.to_dict())
                if annotated is not None:
                    color = _COLORS["wire_ok"] if w.status in ("ok", "unknown") else _COLORS["wire_bad"]
                    cv2.line(annotated, (int(w.start[0]), int(w.start[1])),
                             (int(w.end[0]), int(w.end[1])), color, 2)
                if w.status not in ("ok", "unknown"):
                    self._log_event("wiring_error", camera_id, camera_name,
                                    f"wire {w.status}", None,
                                    payload=w.to_dict(),
                                    dedup_key=f"wire:{camera_id}:{w.wire_uid}:{w.status}")

        # --- additional detection modules (fire/weapon/violence/fall/ppe/
        #     human/vehicle) processed generically ---
        people: list[Detection] = []
        for task, cfg in _EVENT_TASKS.items():
            if not self.ai.is_enabled(task):
                continue
            backend = self.ai.backend(task)
            if backend is None:
                continue
            t0 = time.monotonic()
            try:
                # fall's heuristic backend classifies person boxes rather than
                # running its own image inference.
                if task == "fall" and hasattr(backend, "analyze_people"):
                    dets = backend.analyze_people(people)
                else:
                    dets = backend.infer(frame)
            except Exception as exc:
                dets = []
                result[f"{task}_error"] = str(exc)
            self.ai.record_infer(task, (time.monotonic() - t0) * 1000)

            if task == "human":
                people = list(dets)

            self._handle_event_task(
                task, cfg, dets, camera_id, camera_name, frame, annotated, result)

        # crowd / occupancy counters from tracked people
        if "human" in result:
            result["crowd_count"] = len(result["human"])

        if annotated is not None:
            result["_annotated"] = annotated
        self._last_result[camera_id] = {k: v for k, v in result.items() if k != "_annotated"}
        return result

    def _handle_event_task(self, task, cfg, dets, camera_id, camera_name,
                           frame, annotated, result) -> None:
        """Track, overlay, persist and (optionally) alert for a module's detections."""
        # stable IDs
        id_map = {}
        try:
            id_map = self._tracker(camera_id, task).update(dets)
        except Exception:
            id_map = {}

        color = _COLORS.get(task, _COLORS["object"])
        out = []
        for i, d in enumerate(dets):
            dd = d.to_dict()
            tid = id_map.get(i)
            if tid is not None:
                dd["track_id"] = tid
            out.append(dd)

            label = d.label.lower()
            is_violation = task == "ppe" and (label.startswith("no_") or label.startswith("no-"))
            draw_color = color
            if task == "ppe":
                draw_color = _COLORS["ppe_bad"] if is_violation else _COLORS["ppe_ok"]

            if annotated is not None:
                lbl = d.label if tid is None else f"#{tid} {d.label}"
                _draw_box(annotated, d.bbox, lbl, draw_color, d.confidence)

            # events
            if task == "ppe":
                if is_violation:
                    snap = self._save_snapshot(frame, "ppe") if cfg["snapshot"] else None
                    self._log_event("ppe_violation", camera_id, camera_name,
                                    d.label, d.confidence, snapshot=snap,
                                    payload=dd, dedup_key=f"ppe:{camera_id}:{d.label}")
            elif cfg["alert"]:
                etype = {"fire": "fire", "weapon": "weapon",
                         "violence": "violence", "fall": "fall"}[task]
                # fire model may emit smoke/explosion as distinct labels
                if task == "fire" and label in ("smoke", "explosion"):
                    etype = label
                snap = self._save_snapshot(frame, task) if cfg["snapshot"] else None
                self._log_event(etype, camera_id, camera_name, d.label, d.confidence,
                                snapshot=snap, payload=dd,
                                dedup_key=f"{task}:{camera_id}:{d.label}")

        result[task] = out

    @staticmethod
    def _panel_position(box, shape) -> str:
        h, w = shape[:2]
        cx, cy = box.center
        col = "left" if cx < w / 3 else ("center" if cx < 2 * w / 3 else "right")
        row = "top" if cy < h / 3 else ("middle" if cy < 2 * h / 3 else "bottom")
        return f"{row}-{col}"

    def annotated_jpeg(self, camera_id, camera_name, frame, quality=80) -> bytes:
        res = self.process(camera_id, camera_name, frame, annotate=True, force=True)
        img = res.get("_annotated", frame)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else b""

    # -- low-latency overlay draw (no inference) ---------------------------

    def draw_overlays(self, camera_id: str, frame: np.ndarray) -> np.ndarray:
        """
        Draw the *most recent* detections (computed by the background AI worker)
        onto a copy of ``frame`` WITHOUT running inference. This is what the live
        AI stream uses: overlays lag inference by at most one worker interval,
        but the streamed frame itself is always the freshest one, so the video
        has the same ultra-low latency as the raw stream.
        """
        res = self._last_result.get(camera_id)
        img = frame.copy()
        if not res:
            return img
        for f in res.get("faces", []):
            known = f.get("employee_id") is not None
            self._draw_dict_box(
                img, f, _COLORS["face_known"] if known else _COLORS["face_unknown"])
        for o in res.get("objects", []):
            self._draw_dict_box(img, o, _COLORS["object"])
        for c in res.get("components", []):
            self._draw_dict_box(img, c, _COLORS["component"])
        for task in _EVENT_TASKS:
            for d in res.get(task, []):
                if task == "ppe":
                    lbl = str(d.get("label", "")).lower()
                    col = _COLORS["ppe_bad"] if lbl.startswith(("no_", "no-")) else _COLORS["ppe_ok"]
                else:
                    col = _COLORS.get(task, _COLORS["object"])
                self._draw_dict_box(img, d, col)
        return img

    @staticmethod
    def _draw_dict_box(img, det: dict, color) -> None:
        box = det.get("bbox")
        if not box or len(box) != 4:
            return
        b = BBox(*box)
        label = det.get("label", "")
        tid = det.get("track_id")
        if tid is not None:
            label = f"#{tid} {label}"
        _draw_box(img, b, label, color, det.get("confidence"))

    def annotated_jpeg_fast(self, camera_id, camera_name, frame, quality=80) -> bytes:
        """Draw cached overlays on the freshest frame and encode. No inference."""
        img = self.draw_overlays(camera_id, frame)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        return buf.tobytes() if ok else b""
