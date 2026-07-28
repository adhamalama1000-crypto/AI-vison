"""
DB-backed orchestration for the image analysis & comparison API.

Persists uploaded images, their analyses (+ objects + OCR), and comparison
reports (+ per-difference rows + rendered overlay/heatmap images). All heavy
work is synchronous and is called from the API via ``asyncio.to_thread`` so the
event loop and camera pipeline never block.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

import cv2
import numpy as np

from ..errors import RTSPBackendError
from . import analysis as _an
from . import comparison as _cmp
from . import export as _export
from . import visualize as _vis

_FMT = {".jpg": "jpg", ".jpeg": "jpg", ".png": "png", ".bmp": "bmp",
        ".webp": "webp", ".tif": "tiff", ".tiff": "tiff"}


class ImageService:
    def __init__(self, db, ai_manager, data_dir: str = "data") -> None:
        self.db = db
        self.ai = ai_manager
        self.data_dir = data_dir

    # -- storage -----------------------------------------------------------

    def store_image(self, raw: bytes, filename: Optional[str] = None) -> dict:
        """Validate + persist an uploaded image. Raises on undecodable input."""
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RTSPBackendError("File is not a decodable image.", status_code=400,
                                   code="bad_image")
        h, w = img.shape[:2]
        name = os.path.basename(filename or "image").replace("/", "_")
        ext = os.path.splitext(name)[1].lower()
        fmt = _FMT.get(ext, "jpg")
        rel = os.path.join("images", f"img_{int(time.time()*1000)}_{name or 'image'}")
        if not os.path.splitext(rel)[1]:
            rel += ".jpg"
        abs_path = os.path.join(self.data_dir, rel)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)
        with open(abs_path, "wb") as fh:
            fh.write(raw)
        now = time.time()
        img_id = self.db.insert(
            "INSERT INTO images(name,path,format,width,height,bytes,sha256,phash,"
            "status,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (name, rel, fmt, w, h, len(raw), _an.sha256_bytes(raw),
             _an.perceptual_hash(img), "uploaded", now, now))
        return {"id": img_id, "name": name, "path": rel, "format": fmt,
                "width": w, "height": h, "bytes": len(raw), "status": "uploaded"}

    def _load(self, image_id: int) -> tuple[dict, np.ndarray]:
        row = self.db.query_one("SELECT * FROM images WHERE id=?", (image_id,))
        if not row:
            raise RTSPBackendError("Image not found.", status_code=404, code="not_found")
        d = dict(row)
        img = cv2.imread(os.path.join(self.data_dir, d["path"]))
        if img is None:
            raise RTSPBackendError("Stored image file is missing/corrupt.",
                                   status_code=404, code="not_found")
        return d, img

    # -- analysis ----------------------------------------------------------

    def analyze(self, image_id: int) -> dict:
        row, img = self._load(image_id)
        self.db.execute("UPDATE images SET status='analyzing', updated_at=? WHERE id=?",
                        (time.time(), image_id))
        try:
            result = _an.analyze(img, self.ai)
        except Exception as exc:
            self.db.execute("UPDATE images SET status='error', updated_at=? WHERE id=?",
                            (time.time(), image_id))
            raise RTSPBackendError(f"Analysis failed: {exc}", status_code=500,
                                   code="analysis_failed")
        self._persist_analysis(image_id, result)
        return self.get(image_id)

    def _persist_analysis(self, image_id: int, result: dict) -> None:
        now = time.time()
        self.db.execute("DELETE FROM image_objects WHERE image_id=?", (image_id,))
        self.db.execute("DELETE FROM image_ocr WHERE image_id=?", (image_id,))
        for o in result["objects"]:
            b = o["bbox"]
            self.db.insert(
                "INSERT INTO image_objects(image_id,label,confidence,x1,y1,x2,y2,source,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (image_id, o["label"], o["confidence"], b[0], b[1], b[2], b[3],
                 o.get("source"), now))
        for it in result["ocr"]["items"]:
            b = it.get("bbox") or [None, None, None, None]
            self.db.insert(
                "INSERT INTO image_ocr(image_id,text,confidence,x1,y1,x2,y2,engine,created_at) "
                "VALUES(?,?,?,?,?,?,?,?,?)",
                (image_id, it["text"], it.get("confidence"), b[0], b[1], b[2], b[3],
                 result["ocr"]["engine"], now))
        self.db.execute(
            "UPDATE images SET dominant_colors=?, metadata=?, summary=?, tags=?, "
            "ocr_text=?, n_objects=?, analysis=?, status='analyzed', updated_at=? WHERE id=?",
            (json.dumps(result["dominant_colors"]),
             json.dumps({"channels": result["channels"], "image_size": result["image_size"]}),
             result["summary"], json.dumps(result["tags"]), result["ocr"]["text"],
             result["object_total"], json.dumps(result), now, image_id))

    def get(self, image_id: int) -> dict:
        row = self.db.query_one("SELECT * FROM images WHERE id=?", (image_id,))
        if not row:
            raise RTSPBackendError("Image not found.", status_code=404, code="not_found")
        d = dict(row)
        for k in ("dominant_colors", "tags", "metadata", "analysis"):
            if d.get(k):
                try: d[k] = json.loads(d[k])
                except (TypeError, json.JSONDecodeError): pass
        d["objects"] = [dict(r) for r in self.db.query(
            "SELECT label,confidence,x1,y1,x2,y2,source FROM image_objects WHERE image_id=? ORDER BY id",
            (image_id,))]
        d["ocr_items"] = [dict(r) for r in self.db.query(
            "SELECT text,confidence,x1,y1,x2,y2,engine FROM image_ocr WHERE image_id=? ORDER BY id",
            (image_id,))]
        return d

    def list_images(self, limit: int = 100) -> list[dict]:
        rows = self.db.query(
            "SELECT id,name,path,format,width,height,n_objects,summary,status,created_at "
            "FROM images ORDER BY created_at DESC LIMIT ?", (limit,))
        return [dict(r) for r in rows]

    # -- comparison --------------------------------------------------------

    def compare(self, ref_image_id: int, cur_image_id: int, make_pdf: bool = True) -> dict:
        _ref_row, ref = self._load(ref_image_id)
        _cur_row, cur = self._load(cur_image_id)
        result = _cmp.compare(ref, cur, self.ai)

        visuals = _vis.build_comparison_visuals(ref, cur, result)
        rel_dir = "images"
        ov = self._save(visuals["overlay"], "overlay")
        hm = self._save(visuals["heatmap"], "heatmap")
        al = self._save(visuals["annotated_current"], "aligned")

        # strip raw arrays before persistence / JSON
        clean = {k: v for k, v in result.items() if not k.startswith("_")}
        pdf_rel = None
        if make_pdf:
            pdf_rel = _export.comparison_pdf(self.data_dir, clean, ov)
        json_rel = _export.write_json(self.data_dir, clean, "imgcmp")

        now = time.time()
        cmp_id = self.db.insert(
            "INSERT INTO image_comparisons(ref_image_id,cur_image_id,similarity,n_diffs,"
            "status,report,overlay_path,heatmap_path,aligned_path,report_pdf,created_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (ref_image_id, cur_image_id, clean["similarity"], clean["n_diffs"],
             clean["status"], json.dumps(clean), ov, hm, al, pdf_rel, now))
        for d in clean["differences"]:
            b = d.get("bbox") or [None, None, None, None]
            self.db.insert(
                "INSERT INTO image_diffs(comparison_id,diff_type,severity,detail,confidence,"
                "x1,y1,x2,y2,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (cmp_id, d["diff_type"], d["severity"], d.get("detail"),
                 d.get("confidence"), b[0], b[1], b[2], b[3], now))
        return self.get_comparison(cmp_id)

    def _save(self, img: np.ndarray, prefix: str) -> Optional[str]:
        rel = os.path.join("images", f"{prefix}_{int(time.time()*1000)}_{int(np.random.randint(1_000_000))}.jpg")
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        if not ok:
            return None
        p = os.path.join(self.data_dir, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "wb") as fh:
            fh.write(buf.tobytes())
        return rel

    def get_comparison(self, cmp_id: int) -> dict:
        row = self.db.query_one("SELECT * FROM image_comparisons WHERE id=?", (cmp_id,))
        if not row:
            raise RTSPBackendError("Comparison not found.", status_code=404, code="not_found")
        d = dict(row)
        if d.get("report"):
            try: d["report"] = json.loads(d["report"])
            except (TypeError, json.JSONDecodeError): pass
        d["diffs"] = [dict(r) for r in self.db.query(
            "SELECT diff_type,severity,detail,confidence,x1,y1,x2,y2 FROM image_diffs "
            "WHERE comparison_id=? ORDER BY id", (cmp_id,))]
        return d

    def list_comparisons(self, limit: int = 100) -> list[dict]:
        rows = self.db.query(
            "SELECT id,ref_image_id,cur_image_id,similarity,n_diffs,status,overlay_path,"
            "report_pdf,created_at FROM image_comparisons ORDER BY created_at DESC LIMIT ?",
            (limit,))
        return [dict(r) for r in rows]

    def delete_image(self, image_id: int) -> None:
        row = self.db.query_one("SELECT path FROM images WHERE id=?", (image_id,))
        if row:
            p = os.path.join(self.data_dir, row["path"])
            if os.path.isfile(p):
                try: os.remove(p)
                except OSError: pass
        self.db.execute("DELETE FROM images WHERE id=?", (image_id,))
