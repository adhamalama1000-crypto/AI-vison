# v4.0.0 — Industrial AI Inspection upgrade

This release extends the AI Vision Platform from an RTSP + face-recognition
backend into an **industrial electrical-panel inspection system**, adding
dataset management, a real training/HPO/comparison/export engine, panel
analysis, reference designs, reference-vs-live inspection, attendance, and a set
of new dashboard pages.

The project's founding principle is unchanged: **nothing is faked**. Every AI
number comes from a real model or a real measurement; where a capability needs a
trained model or data that isn't present, the pipeline runs but returns empty
results with an explicit reason, never invented output.

---

## What is real and verified here

- **Attendance (Part 1)** — a known employee is recorded at most once per a
  configurable timeout (default 8h / once per shift; short timeouts throttle by
  interval, long ones enforce one row per calendar day). Recording is driven by
  the face pipeline's positive matches only. REST API + tests.
- **Face recognition (Part 1)** — the enrol → embed → persist → cosine-match →
  top-k vote pipeline (unchanged core), plus InsightFace ArcFace with
  **automatic `pip install insightface`** when missing, per-frame detection
  caching, and a configurable detection threshold. When the ArcFace *model pack*
  can't be downloaded (e.g. an offline/network-restricted host — as in this
  build), it reports `weights unavailable` and the tested OpenCV embedder stays
  active. It is genuinely wired for a normal deployment.
- **Dataset management (Part 2)** — upload images / videos / zips / whole
  folders; auto-detect **YOLO / COCO / Pascal-VOC / image-classification /
  images / videos / mixed**; validation walks the real files for missing
  labels, corrupt/undecodable images, class counts and **class imbalance**, and
  emits a report. REST API + DB + tests.
- **Training engine (Parts 3, 4, 5)** — a real per-epoch training loop with
  cooperative **pause / resume / stop**, live metrics (epoch, train/val loss,
  accuracy, precision, recall, F1, learning rate, elapsed/ETA, CPU/RAM),
  feature-space **augmentation**, **early stopping**, optional **Optuna** HPO,
  **multi-model comparison**, automatic **best-model selection**, and **ONNX
  export with reload verification** under ONNX Runtime. Verified end-to-end on a
  real demo dataset (scikit-learn digits) — see `tests/test_platform_upgrade.py`.
  Object-detection architectures (YOLOv8–v11, RT-DETR, …) train through an
  **Ultralytics adapter** when that library + a detection dataset are present,
  and are otherwise recorded as `skipped` with the precise reason.
- **Panel analysis (Part 8)** — runs the component detector + wire analyzer on
  an uploaded image (or camera frame), counts components by class, records wire
  colours, builds an electrical topology, renders an annotated image, and writes
  JSON + PDF reports.
- **Reference designs (Part 9)** — upload/store PDF / PNG / JPG / DXF / DWG plus
  an optional expected-spec (component/wire-colour counts).
- **Inspection (Part 10)** — compares an observed panel against a reference
  (explicit spec, or derived by analysing a reference image), reporting missing
  / extra / wrong-count components and wire-colour mismatches with a
  pass/warning/fail verdict, an annotated frame, and a PDF report.
- **Reports registry** and **Metrics** endpoints for the dashboard.
- **Bug fix** — the classical wire analyzer assumed `HoughLinesP` returns
  `(N,1,4)`; on OpenCV builds that return `(N,4)` it raised and produced no
  wires. Now shape-robust.

## Component classes (Part 6)

The component label set is the full 27-class electrical set (MCB, MCCB, ACB,
fuse, relay, contactor, PLC, VFD, CT, VT, power supply, terminal block, push
button, selector switch, indicator lamp, e-stop, energy meter, protection
relay, motor starter, capacitor, DIN rail, busbar, neutral bar, earth bar, cable
tray, wire duct, copper bus). A trained detector dropped into
`models/components/*.onnx` (+ optional `labels.txt`) activates real component
detection with no code changes.

## What still needs YOUR data / hardware for production

- **No electrical dataset ships with the platform.** Component/wire detection
  returns empty until you upload a labelled electrical dataset (Electrical
  Dataset page), train a detector (Training page — install `ultralytics` for
  YOLO/RT-DETR), and let the platform export + load the ONNX model.
- **GPU/CUDA** is absent in this build, so training ran on CPU and TensorRT
  export is unavailable; GPU metrics report `unavailable` rather than estimates.
- **InsightFace ArcFace model pack** downloads from GitHub releases, which is
  blocked in this sandbox; it works on a normal host.

## New REST surface

`/api/attendance*`, `/api/datasets*`, `/api/training*`, `/api/reference*`,
`/api/panels*`, `/api/inspection*`, `/api/reports*` — see the OpenAPI docs at
`/docs`.

## Tests

`tests/test_platform_upgrade.py` adds coverage for attendance throttling,
dataset detection/validation, end-to-end training (train → compare → select →
export → verify), panel analysis, reference + inspection, and the comparison
logic. The full suite remains green.
