# Industrial Panel Inspection

A production module that turns an electrical control panel into structured,
comparable data: **learn** a reference panel from images, then **inspect** any
live/uploaded panel against it and report wiring / component / terminal faults
with confidence scores.

It is fully additive — the RTSP camera pipeline, employee face recognition,
attendance, events, and every prior API/page/table are untouched.

---

## 1. Concept

```
 images ──▶ detectors ──▶ template (components + terminals + wires + graph)
 (RTSP or                 + ORB feature embedding                 │  store
  upload)                                                         ▼
                                                        reference_panels + children
 live frame ─▶ detectors ─▶ observed ─▶ register (homography) ─▶ compare ─▶ errors
                                                                    │
                                                                    ▼
                                                     inspection_results + children
                                                     + green/yellow/red overlay
```

* **Reference learning** builds a reusable *template* from one or more images of
  a known-good panel.
* **Inspection** analyses a new panel, registers it to the reference with an ORB
  homography (so camera pose/zoom differences aren't mistaken for faults), and
  diffs it.

## 2. Vision engine (`rtsp_backend/panels/`)

| Module | Responsibility |
|---|---|
| `wire_detector.py` | Real wire instances: per-colour HSV/LAB segmentation → adaptive threshold → morphology → skeletonisation → connected components → contour filtering → polyline extraction (longest skeleton path + spur pruning) → Hough fallback → direction-aware merge + overlap dedup. Each wire has start, end, polyline, length, thickness, colour, direction, connections. |
| `terminal_detector.py` | Terminal blocks (from the component model or morphology), screws (Hough circles), and wire-entry points. |
| `features.py` | ORB descriptor + colour-histogram embedding, and RANSAC homography alignment (observed → reference). |
| `graph.py` | Electrical graph: component + terminal nodes, wire edges. |
| `template.py` | Learns a reference template (per-component bbox / centre / size / rotation / grid position, terminals, wires, graph) from images. |
| `comparison.py` | Diffs observed vs reference → the full fault taxonomy with confidence. |
| `datasheet.py` | OCR/parse a schematic (PDF/PNG/JPG/DXF/SVG) into component/terminal/wire IDs + an expected graph. |
| `overlay.py` | Green (correct) / yellow (warning) / red (error) overlay with names, wire IDs and confidence. |

### Fault taxonomy (`comparison.py`)

Components: `missing_component`, `extra_component`, `wrong_component`,
`moved_component`, `wrong_rotation`.
Wires: `missing_wire`, `extra_wire`, `wrong_wire`, `loose_wire`,
`disconnected_wire`, `broken_wire`, `wrong_wire_color`.
Terminals: `wrong_terminal`, `wrong_source`, `wrong_destination`.

Every error carries `severity` (error/warning/info) and a `confidence` in
`[0,1]`. `loose_wire`/`disconnected_wire` are measured *relative to the
reference's own connectivity*, so an identical panel is always clean.

## 3. REST API

Reference panels (`rtsp_backend/api/reference_panels.py`):

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/reference-panels` | create (`{name, version?, description?}`) |
| GET | `/api/reference-panels` | list |
| GET | `/api/reference-panels/{id}` | full panel (images/components/terminals/wires/graph/template) |
| DELETE | `/api/reference-panels/{id}` | delete + data |
| POST | `/api/reference-panels/{id}/capture` | grab a frame from an RTSP camera (form `camera_id`) |
| POST | `/api/reference-panels/{id}/upload` | upload one or more images (multipart `files`) |
| POST | `/api/reference-panels/{id}/learn` | build the template + graph |
| POST | `/api/reference-panels/{id}/compare` | inspect (`file` upload or form `camera_id`) |
| GET | `/api/reference-panels/{id}/result` | latest (or `?result_id=`) result |
| GET | `/api/reference-panels/{id}/results` | inspection history |

Datasheets (`rtsp_backend/api/datasheets.py`): `POST /api/datasheets/upload`,
`POST /api/datasheets/{id}/extract`, `GET /api/datasheets[/{id}]`,
`DELETE /api/datasheets/{id}`.

## 4. Dashboard pages

New **Industrial Inspection** nav section:
* **Reference Panels** — create, capture/upload, learn, inspect, view errors + overlay.
* **Topology Viewer** — SVG electrical graph of a learned panel.
* **Datasheets** — upload a schematic, OCR it into an expected graph.

## 5. Database (all additive, `IF NOT EXISTS`)

`reference_panels`, `reference_images`, `reference_components`,
`reference_terminals`, `reference_wires`, `reference_connections`,
`reference_graph`, `inspection_results`, `inspection_components`,
`inspection_terminals`, `inspection_wires`, `inspection_errors`, `datasheets`.

## 6. Performance

Learning, comparison and OCR run in worker threads via `asyncio.to_thread`, so
the event loop, camera capture and MJPEG/AI streams never block. **Capturing a
reference frame reads the existing camera frame buffer — it never opens a second
RTSP connection.**

## 7. What needs a trained model / optional dependency

The pipeline is complete and honest about its inputs:

* **Component detection** requires a trained detector. Train one (YOLOv8) on the
  electrical classes (`rtsp_backend/ai/components.py::ELECTRICAL_CLASSES`),
  export to ONNX, and drop it into `models/components/` — it auto-loads, no code
  change. Until then components are reported empty (with a note); wire, terminal
  and topology inspection still run on real geometry.
  * Training path: upload a YOLO dataset → Training page → select `yolov8`
    (needs `pip install ultralytics`) → ONNX export is automatic. See
    `rtsp_backend/training_svc.py`.
* **Datasheet OCR** for raster/PDF needs an OCR engine: `pip install paddleocr`
  (preferred) or `pip install pytesseract` + the `tesseract` binary. DXF/SVG are
  parsed without OCR. Without an engine the extractor returns
  `ocr_engine="none"` and says so — it never invents IDs.

## 8. Tests

`tests/test_panels_vision.py` (vision engine: wire geometry/colour/stability,
terminals, features/alignment, graph, template, comparison fault detection,
datasheet parsing, overlay) and `tests/test_reference_panels_api.py` (full REST
flow incl. RTSP capture + datasheets + non-regression of existing endpoints).

```bash
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q
```
