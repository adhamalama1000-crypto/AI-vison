# AI Image Analysis & Comparison

Upload **any** image and have it analysed by AI; upload a reference + a current
image and detect **every difference** between them. Fully additive — the RTSP
camera, face recognition, panel inspection and all prior features are untouched.

## Engine (`rtsp_backend/imaging/`)

| Module | Responsibility |
|---|---|
| `analysis.py` | Single-image analysis: objects + boxes + confidence (ONNX detector), dominant colours (k-means), OCR text, perceptual hash, quality defects (blur/exposure), tags + AI summary, size/metadata. |
| `comparison.py` | Reference-vs-current diff: ORB + RANSAC **registration** (perspective compensation), **luminance matching** (lighting compensation), **SSIM** difference regions (noise-tolerant), object/colour/text diff, and a blended **similarity %**. |
| `ocr.py` | Multi-engine OCR (EasyOCR → PaddleOCR → Tesseract) with graceful fallback. |
| `visualize.py` | Heatmap, annotated boxes, side-by-side overlay. |
| `export.py` | JSON + PDF reports. |
| `service.py` | DB-backed orchestration used by the API. |

**Similarity** blends SSIM (0.5), colour-histogram correlation (0.2), ORB inlier
ratio (0.15) and perceptual-hash distance (0.15) into a 0–100 %.

**Difference types:** `missing_object`, `new_object`, `moved_object`,
`changed_object`, `color_change`, `text_change`, `region_changed` — each with a
severity (`major`/`minor`/`info`), confidence and (where spatial) a bounding box.

## REST API

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/images/upload` | store an image (multipart `file`) |
| POST | `/api/images/analyze` | analyse a `file` or `image_id` |
| POST | `/api/images/compare` | compare `reference`/`current` files or `reference_id`/`current_id` |
| GET | `/api/images` | list images |
| GET | `/api/images/{id}` | one image + analysis (objects, OCR, colours) |
| DELETE | `/api/images/{id}` | delete |
| GET | `/api/images/history` | images + comparisons |
| GET | `/api/images/comparisons` | list comparisons |
| GET | `/api/images/report/{id}` | comparison report (JSON) + `report_pdf` link |

Rendered overlay/heatmap/PDF/JSON are served under `/api/media/...`.

## Dashboard pages (Image AI section)

* **Image Analysis** — drag-&-drop upload, preview with SVG bounding-box overlay,
  AI status/progress, detected objects, dominant colours, OCR, tags, summary,
  recent images.
* **Image Comparison** — two drag-&-drop zones, similarity gauge, status, full
  difference list, side-by-side + heatmap overlay, PDF/JSON/image export, history.

## Database (additive)

`images`, `image_objects`, `image_ocr`, `image_comparisons`, `image_diffs`.

## Performance & security

- Handles large (up to ~8K) images; downscales internally for colour/registration.
- CPU by default; ONNX Runtime uses CUDA automatically when available.
- All heavy work runs in worker threads (`asyncio.to_thread`) — the event loop,
  camera pipeline and streams never block.
- Uploads are size-capped (`max_upload_bytes`) and validated by decoding; a
  non-image is rejected `400`, filenames are sanitised.

## What needs a model / optional dependency

The pipeline is complete and honest about inputs:
- **Object detection & bounding boxes** use the platform's ONNX detector — drop a
  YOLO/GroundingDINO-exported `.onnx` into `models/detection/` and it auto-loads
  (COCO by default). Without weights, object lists are empty (with a note) while
  colours, SSIM regions, OCR, hashing and the full comparison still run.
- **OCR** needs one of `easyocr` / `paddleocr` / `pytesseract` (+ tesseract
  binary). Without any, OCR returns empty with `engine="none"` — never fabricated.
- A CLIP/BLIP hook exists in `analysis._clip_tags` for richer tags when installed.

## Tests

`tests/test_imaging.py` — analysis (colours, hash, defects, shape), comparison
(identical / changed / perspective), and the full REST surface incl. bad-upload
rejection and non-regression of existing endpoints.
