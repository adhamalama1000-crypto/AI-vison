"""
Reference Panel service (Features 1, 2, 5, 6, 8, 9, 10).

Orchestrates the reference-panel lifecycle on top of the normalized SQLite
schema and the :mod:`rtsp_backend.panels` vision engine:

* **create / list / get / delete** a reference panel.
* **add_image**    — persist an image captured from an RTSP camera buffer or
  uploaded, under ``data/reference/panel_<id>/``.
* **learn**        — run the detector stack over every image, build the reusable
  template + electrical graph + feature embedding, and persist them both
  denormalised (JSON blobs) and normalised (reference_components / _terminals /
  _wires / _connections / _graph).
* **compare**      — analyse an observed image, register it to the reference,
  diff it, persist the inspection_results (+ _components / _terminals / _wires /
  _errors), and render the green/yellow/red overlay.

All heavy CPU work is plain synchronous code; the API layer runs it via
``asyncio.to_thread`` so the event loop, camera pipeline and RTSP connections
are never blocked and no second camera connection is ever opened (images are
read straight from the existing camera frame buffer).
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import cv2
import numpy as np

from .panels import comparison as _comparison
from .panels import features as _features
from .panels import graph as _graph
from .panels import overlay as _overlay
from .panels import template as _template
from .errors import RTSPBackendError


class ReferencePanelService:
    def __init__(self, db, ai_manager, data_dir: str = "data") -> None:
        self.db = db
        self.ai = ai_manager
        self.data_dir = data_dir

    # -- CRUD --------------------------------------------------------------

    def create(self, name: str, version: str = "v1",
               description: Optional[str] = None) -> dict:
        now = time.time()
        pid = self.db.insert(
            "INSERT INTO reference_panels(name,version,description,status,"
            "created_at,updated_at) VALUES(?,?,?,?,?,?)",
            (name, version or "v1", description, "draft", now, now))
        return self.get(pid)

    def list_panels(self, limit: int = 100) -> list[dict]:
        rows = self.db.query(
            "SELECT id,name,version,description,status,n_images,n_components,"
            "n_terminals,n_wires,thumbnail,note,created_at,updated_at "
            "FROM reference_panels ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    def get(self, panel_id: int, full: bool = True) -> dict:
        row = self.db.query_one("SELECT * FROM reference_panels WHERE id=?", (panel_id,))
        if not row:
            raise RTSPBackendError("Reference panel not found.", status_code=404,
                                   code="not_found")
        d = dict(row)
        for k in ("template", "features"):
            if d.get(k):
                try:
                    d[k] = json.loads(d[k])
                except (TypeError, json.JSONDecodeError):
                    pass
        if not full:
            d.pop("template", None)
            d.pop("features", None)
            return d
        d["images"] = [dict(r) for r in self.db.query(
            "SELECT id,path,source,camera_id,width,height,is_primary,created_at "
            "FROM reference_images WHERE panel_id=? ORDER BY created_at", (panel_id,))]
        d["components"] = [dict(r) for r in self.db.query(
            "SELECT * FROM reference_components WHERE panel_id=? ORDER BY id", (panel_id,))]
        d["terminals"] = [dict(r) for r in self.db.query(
            "SELECT * FROM reference_terminals WHERE panel_id=? ORDER BY id", (panel_id,))]
        d["wires"] = [_json_cols(dict(r), ("polyline", "payload")) for r in self.db.query(
            "SELECT * FROM reference_wires WHERE panel_id=? ORDER BY id", (panel_id,))]
        graph_row = self.db.query_one(
            "SELECT nodes,edges FROM reference_graph WHERE panel_id=?", (panel_id,))
        if graph_row:
            d["graph"] = {"nodes": _loads(graph_row["nodes"]),
                          "edges": _loads(graph_row["edges"])}
        return d

    def delete(self, panel_id: int) -> None:
        row = self.db.query_one("SELECT id FROM reference_panels WHERE id=?", (panel_id,))
        if not row:
            raise RTSPBackendError("Reference panel not found.", status_code=404,
                                   code="not_found")
        # remove image files
        for img in self.db.query("SELECT path FROM reference_images WHERE panel_id=?",
                                 (panel_id,)):
            full = os.path.join(self.data_dir, img["path"])
            if os.path.isfile(full):
                try:
                    os.remove(full)
                except OSError:
                    pass
        pdir = os.path.join(self.data_dir, "reference", f"panel_{panel_id}")
        if os.path.isdir(pdir):
            try:
                os.rmdir(pdir)
            except OSError:
                pass
        # ON DELETE CASCADE clears child tables
        self.db.execute("DELETE FROM reference_panels WHERE id=?", (panel_id,))

    # -- images ------------------------------------------------------------

    def add_image(self, panel_id: int, image_bgr: np.ndarray, source: str,
                  camera_id: Optional[str] = None) -> dict:
        self.get(panel_id, full=False)  # 404 guard
        h, w = image_bgr.shape[:2]
        rel_dir = os.path.join("reference", f"panel_{panel_id}")
        abs_dir = os.path.join(self.data_dir, rel_dir)
        os.makedirs(abs_dir, exist_ok=True)
        fname = f"img_{int(time.time()*1000)}.jpg"
        rel = os.path.join(rel_dir, fname)
        ok, buf = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        if not ok:
            raise RTSPBackendError("Failed to encode image.", status_code=400,
                                   code="bad_image")
        with open(os.path.join(self.data_dir, rel), "wb") as fh:
            fh.write(buf.tobytes())
        existing = self.db.query_one(
            "SELECT COUNT(*) AS n FROM reference_images WHERE panel_id=?", (panel_id,))
        is_primary = 1 if (existing["n"] == 0) else 0
        img_id = self.db.insert(
            "INSERT INTO reference_images(panel_id,path,source,camera_id,width,"
            "height,is_primary,created_at) VALUES(?,?,?,?,?,?,?,?)",
            (panel_id, rel, source, camera_id, w, h, is_primary, time.time()))
        n = existing["n"] + 1
        thumb = rel if is_primary else None
        if thumb:
            self.db.execute(
                "UPDATE reference_panels SET n_images=?, thumbnail=?, updated_at=? WHERE id=?",
                (n, thumb, time.time(), panel_id))
        else:
            self.db.execute(
                "UPDATE reference_panels SET n_images=?, updated_at=? WHERE id=?",
                (n, time.time(), panel_id))
        return {"id": img_id, "panel_id": panel_id, "path": rel,
                "source": source, "width": w, "height": h, "is_primary": bool(is_primary)}

    # -- learning ----------------------------------------------------------

    def learn(self, panel_id: int, wire_params: Optional[dict] = None) -> dict:
        panel = self.get(panel_id, full=False)
        img_rows = self.db.query(
            "SELECT path FROM reference_images WHERE panel_id=? ORDER BY is_primary DESC, created_at",
            (panel_id,))
        if not img_rows:
            raise RTSPBackendError(
                "Add at least one image before learning the reference panel.",
                status_code=400, code="no_images")
        images = []
        for r in img_rows:
            img = cv2.imread(os.path.join(self.data_dir, r["path"]))
            if img is not None:
                images.append(img)
        if not images:
            raise RTSPBackendError("No decodable reference images found.",
                                   status_code=400, code="bad_image")

        self.db.execute("UPDATE reference_panels SET status='learning', updated_at=? WHERE id=?",
                        (time.time(), panel_id))
        try:
            built = _template.build_template(self.ai, images, wire_params)
        except Exception as exc:
            self.db.execute(
                "UPDATE reference_panels SET status='error', note=?, updated_at=? WHERE id=?",
                (f"learning failed: {exc}", time.time(), panel_id))
            raise RTSPBackendError(f"Learning failed: {exc}", status_code=500,
                                   code="learn_failed")

        tmpl = built["template"]
        feats = built["features"]
        self._persist_template(panel_id, tmpl, feats)
        return self.get(panel_id)

    def _persist_template(self, panel_id: int, tmpl: dict, feats: dict) -> None:
        now = time.time()
        # clear any previous learned rows (re-learn is idempotent)
        for t in ("reference_components", "reference_terminals", "reference_wires",
                  "reference_connections"):
            self.db.execute(f"DELETE FROM {t} WHERE panel_id=?", (panel_id,))
        self.db.execute("DELETE FROM reference_graph WHERE panel_id=?", (panel_id,))

        for c in tmpl["components"]:
            self.db.insert(
                "INSERT INTO reference_components(panel_id,ref_id,comp_type,label,"
                "x1,y1,x2,y2,cx,cy,w,h,rotation,confidence,position,payload,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (panel_id, c["ref_id"], c["comp_type"], c["label"],
                 c["bbox"][0], c["bbox"][1], c["bbox"][2], c["bbox"][3],
                 c["cx"], c["cy"], c["w"], c["h"], c["rotation"], c["confidence"],
                 c.get("position"), json.dumps(c), now))
        for t in tmpl["terminals"]:
            self.db.insert(
                "INSERT INTO reference_terminals(panel_id,ref_id,component_ref,label,"
                "kind,x,y,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (panel_id, t["ref_id"], t.get("component_ref"), t.get("label"),
                 t.get("kind"), t["x"], t["y"], now))
        for wnode in tmpl["wires"]:
            self.db.insert(
                "INSERT INTO reference_wires(panel_id,wire_uid,start_x,start_y,end_x,"
                "end_y,polyline,length,thickness,color,direction,from_terminal,"
                "to_terminal,from_component,to_component,payload,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (panel_id, wnode["wire_uid"], wnode["start"][0], wnode["start"][1],
                 wnode["end"][0], wnode["end"][1], json.dumps(wnode["polyline"]),
                 wnode["length"], wnode["thickness"], wnode["color"], wnode["direction"],
                 wnode.get("from_terminal"), wnode.get("to_terminal"),
                 wnode.get("from_component"), wnode.get("to_component"),
                 json.dumps(wnode), now))
        for conn in _graph.connections(tmpl["graph"]):
            self.db.insert(
                "INSERT INTO reference_connections(panel_id,wire_uid,from_node,to_node,"
                "from_terminal,to_terminal,color,created_at) VALUES(?,?,?,?,?,?,?,?)",
                (panel_id, conn["wire_uid"], conn["from_node"], conn["to_node"],
                 conn["from_terminal"], conn["to_terminal"], conn["color"], now))
        self.db.insert(
            "INSERT INTO reference_graph(panel_id,nodes,edges,created_at) VALUES(?,?,?,?)",
            (panel_id, json.dumps(tmpl["graph"]["nodes"]),
             json.dumps(tmpl["graph"]["edges"]), now))

        note = "; ".join(tmpl.get("notes", [])) or None
        self.db.execute(
            "UPDATE reference_panels SET status='ready', template=?, features=?, "
            "n_components=?, n_terminals=?, n_wires=?, note=?, updated_at=? WHERE id=?",
            (json.dumps(tmpl), json.dumps(feats), len(tmpl["components"]),
             len(tmpl["terminals"]), len(tmpl["wires"]), note, now, panel_id))

    # -- comparison --------------------------------------------------------

    def compare(self, panel_id: int, image_bgr: np.ndarray, source: str,
                camera_id: Optional[str] = None,
                wire_params: Optional[dict] = None) -> dict:
        panel = self.get(panel_id, full=True)
        if panel.get("status") != "ready" or not panel.get("template"):
            raise RTSPBackendError(
                "Reference panel has not been learned yet — call learn first.",
                status_code=409, code="not_learned")
        tmpl = panel["template"]
        feats = panel.get("features") or {}

        observed = _template.analyze_image(self.ai, image_bgr, wire_params)
        result = _comparison.compare(tmpl, observed, image_bgr, feats)

        annotated = _overlay.draw_overlay(image_bgr, observed, result)
        rel_dir = os.path.join("inspections", f"panel_{panel_id}")
        os.makedirs(os.path.join(self.data_dir, rel_dir), exist_ok=True)
        rel = os.path.join(rel_dir, f"insp_{int(time.time()*1000)}.jpg")
        ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if ok:
            with open(os.path.join(self.data_dir, rel), "wb") as fh:
                fh.write(buf.tobytes())
        else:
            rel = None

        full = {**result, "observed": observed,
                "reference": {"id": panel_id, "name": panel["name"]}}
        result_id = self.db.insert(
            "INSERT INTO inspection_results(panel_id,camera_id,source,status,score,"
            "n_errors,n_warnings,result,snapshot,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
            (panel_id, camera_id, source, result["status"], result["score"],
             result["n_errors"], result["n_warnings"], json.dumps(full), rel, time.time()))
        self._persist_inspection(result_id, observed, result)
        return {"id": result_id, "panel_id": panel_id, "status": result["status"],
                "score": result["score"], "n_errors": result["n_errors"],
                "n_warnings": result["n_warnings"], "errors": result["errors"],
                "alignment": result["alignment"], "snapshot": rel,
                "observed": observed, "result": full}

    def _persist_inspection(self, result_id: int, observed: dict, result: dict) -> None:
        now = time.time()
        err_targets = {e.get("target"): e for e in result.get("errors", [])}
        for c in observed.get("components", []):
            e = err_targets.get(c.get("label"))
            status = e["error_type"] if e else "ok"
            self.db.insert(
                "INSERT INTO inspection_components(result_id,comp_type,label,x1,y1,"
                "x2,y2,confidence,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (result_id, c.get("comp_type"), c.get("label"),
                 c["bbox"][0], c["bbox"][1], c["bbox"][2], c["bbox"][3],
                 c.get("confidence"), status, now))
        for t in observed.get("terminals", []):
            self.db.insert(
                "INSERT INTO inspection_terminals(result_id,label,kind,x,y,status,created_at) "
                "VALUES(?,?,?,?,?,?,?)",
                (result_id, t.get("label"), t.get("kind"), t.get("x"), t.get("y"),
                 "ok", now))
        for wnode in observed.get("wires", []):
            e = err_targets.get(wnode.get("wire_uid"))
            status = e["error_type"] if e else "ok"
            self.db.insert(
                "INSERT INTO inspection_wires(result_id,wire_uid,start_x,start_y,end_x,"
                "end_y,color,length,thickness,status,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (result_id, wnode.get("wire_uid"), wnode["start"][0], wnode["start"][1],
                 wnode["end"][0], wnode["end"][1], wnode.get("color"),
                 wnode.get("length"), wnode.get("thickness"), status, now))
        for e in result.get("errors", []):
            self.db.insert(
                "INSERT INTO inspection_errors(result_id,error_type,severity,target,"
                "detail,confidence,x,y,created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (result_id, e["error_type"], e["severity"], e.get("target"),
                 e.get("detail"), e.get("confidence"), e.get("x"), e.get("y"), now))

    def get_result(self, panel_id: int, result_id: Optional[int] = None) -> dict:
        if result_id is not None:
            row = self.db.query_one(
                "SELECT * FROM inspection_results WHERE id=? AND panel_id=?",
                (result_id, panel_id))
        else:
            row = self.db.query_one(
                "SELECT * FROM inspection_results WHERE panel_id=? "
                "ORDER BY created_at DESC LIMIT 1", (panel_id,))
        if not row:
            raise RTSPBackendError("No inspection result found for this panel.",
                                   status_code=404, code="not_found")
        d = dict(row)
        if d.get("result"):
            d["result"] = _loads(d["result"])
        return d

    def list_results(self, panel_id: int, limit: int = 50) -> list[dict]:
        rows = self.db.query(
            "SELECT id,panel_id,camera_id,source,status,score,n_errors,n_warnings,"
            "snapshot,created_at FROM inspection_results WHERE panel_id=? "
            "ORDER BY created_at DESC LIMIT ?", (panel_id, limit))
        return [dict(r) for r in rows]


def _loads(s):
    try:
        return json.loads(s)
    except (TypeError, json.JSONDecodeError):
        return s


def _json_cols(d: dict, cols) -> dict:
    for c in cols:
        if d.get(c):
            d[c] = _loads(d[c])
    return d
