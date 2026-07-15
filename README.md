# AI Vision Platform

An RTSP camera backend extended into a full **AI vision platform**: live camera
ingestion, a pluggable AI subsystem (face recognition, object/component
detection, wire analysis), employee management with persistent face embeddings,
a SQLite database, a real-time events feed, and a browser dashboard — all served
by one FastAPI application.

It is designed to sit behind mobile apps, robotics stacks, industrial HMIs,
SCADA systems, and other backend services. Those clients talk to it over HTTP
and WebSockets only — they never touch the cameras directly. A built-in web
dashboard (served at `/`) drives every feature through the same public API.

**Version:** 4.0.0 · **Python:** 3.10+ · **Stack:** FastAPI · Uvicorn · OpenCV (FFmpeg) · ONNX Runtime · scikit-learn · Pydantic v2 · SQLite

> **v4.0.0 — Industrial AI Inspection upgrade.** Adds employee attendance,
> dataset management + validation, a real training / HPO / model-comparison /
> ONNX-export engine, panel analysis, reference designs, and reference-vs-live
> inspection, with new dashboard pages for each. See
> [`CHANGES_v4.0.0.md`](CHANGES_v4.0.0.md) for exactly what is real vs. what
> needs your dataset/GPU.

> **Reference Panel Inspection module.** Learn a known-good electrical panel
> from RTSP/uploaded images (real per-colour wire tracing, terminal detection,
> ORB feature embedding, and an electrical graph), then inspect any panel
> against it and get confidence-scored wiring/component/terminal faults with a
> green/yellow/red overlay. New **Reference Panels**, **Topology Viewer** and
> **Datasheets** dashboard pages and a full `/api/reference-panels/*` +
> `/api/datasheets/*` REST surface. Full details:
> [`docs/INDUSTRIAL_INSPECTION.md`](docs/INDUSTRIAL_INSPECTION.md).

---

## What is real, and what needs a trained model

This platform was built to be **honest about its limits**. Everything that can
be implemented and tested with available libraries is fully wired end to end.
Where a capability depends on a trained model or dataset that does not publicly
exist, the *entire pipeline around it* is implemented (preprocessing, inference
call, postprocessing, visualization, persistence, API, UI) but it returns
**empty results rather than fabricated ones**, and this section says so plainly.

**Fully implemented and tested (see the test suite):**

- RTSP ingestion, reconnect, MJPEG streaming, snapshots, WebSocket telemetry
  (the original backend — unchanged and still covered).
- SQLite database and schema for employees, images, embeddings, events,
  detections, components, wires, model config, and settings.
- The plugin architecture, registry, and AI model manager (enable/disable,
  select backend, tune thresholds, choose device) with **config persisted to
  the database and restored on restart**.
- **Face recognition end to end**: face detection (OpenCV Haar), a deterministic
  embedding, a persistent embedding store, cosine matching with a configurable
  threshold, "Unknown Person" handling, immediate recognition after enrolment,
  and event logging with snapshots. Tested with a real face image.
- The full REST API, the AI overlay pipeline (boxes/labels/scores drawn on the
  live frame), the events feed, dashboard statistics, and the web dashboard.

**Implemented as a pipeline, but inert until you supply a trained model
(no fake output is ever produced):**

- **Generic object detection** — a real YOLOv5/YOLOv8 ONNX inference path. Drop
  a `.onnx` into `models/detection/` and select the `onnx_yolo` backend to get
  real COCO detections. Without weights it reports `no_weights` and detects
  nothing.
- **Electrical component detection** (circuit breakers, MCB/MCCB, contactors,
  relays, PLC modules, busbars, VFDs, …). No suitable public pretrained model
  covers these classes, so **this cannot be validated here**. The ONNX pipeline,
  the 18-class label set, the DB tables, the API, and the UI are all complete;
  drop a trained `models/components/*.onnx` (plus an optional `labels.txt`) to
  activate it.
- **Wire analysis / topology** — instance-level wire tracing and fault
  classification (broken / loose / disconnected / incorrect) is at the frontier
  of computer vision and **has no reliable public model**, so it is not faked.
  A real classical baseline (`classical_wires`) detects wire-like line segments
  and links endpoints to nearby components, but honestly reports each wire's
  fault status as `"unknown"`. The topology data model, DB, visualization, and
  API are complete for a future trained model to populate.
- **Component-to-wire relationship / electrical-path validation** — the data
  model (each wire carries `from_component` / `to_component`) and the topology
  endpoint exist; meaningful validation depends on the two trained models above.
- **Real ArcFace face recognition** — the `insightface_arcface` backend uses the
  InsightFace library when it and its models are available. In an environment
  without them it reports `unavailable` and the manager keeps using the tested
  fallback embedder. (Note: the fallback embedder is a *real* deterministic
  feature extractor sufficient to prove and test the pipeline; it is **not**
  production-accuracy face recognition. Install InsightFace and select this
  backend for that.)

**GPU note:** this environment has no CUDA stack, so GPU metrics are reported as
`unavailable` rather than estimated, and inference runs on CPU. The device
selector and GPU fields are wired for a GPU-equipped deployment.

---

## Activating real AI models

No code changes are needed — the plugin architecture loads weights by convention:

```
models/
  detection/   <your-model>.onnx        # generic object detection (e.g. YOLO COCO)
  components/  <your-model>.onnx         # electrical component detector (custom-trained)
  components/  labels.txt                # optional: one class name per line
```

Then select the backend (or do it from the AI Models page):

```bash
# use a dropped-in YOLO model for object detection
curl -X POST localhost:8000/api/ai/models/detection/select \
     -H 'Content-Type: application/json' -d '{"backend_id":"onnx_yolo"}'
curl -X POST localhost:8000/api/ai/models/detection/enable \
     -H 'Content-Type: application/json' -d '{"enabled":true}'

# real ArcFace embeddings (after `pip install insightface`)
curl -X POST localhost:8000/api/ai/models/face/select \
     -H 'Content-Type: application/json' -d '{"backend_id":"insightface_arcface"}'
```

Adding a brand-new model type is just as clean: subclass one of the interfaces
in `rtsp_backend/ai/base.py` (`Detector`, `FaceEmbedder`, `ComponentDetector`,
`WireAnalyzer`), decorate it with `@register`, and it appears in the catalog,
the API, and the dashboard automatically.

---

## Architecture

```
rtsp_backend/
  app.py                 FastAPI app: RTSP media, AI media, routers, static UI
  db.py                  SQLite layer + schema (stdlib sqlite3, thread-safe)
  config.py              settings incl. data_dir / db_path / models_dir
  camera.py, manager.py  RTSP capture, reconnect, camera manager (original core)
  ai/
    base.py              interfaces + dataclasses (BBox, Detection, Wire)
    registry.py          plugin registry (@register by task)
    embedders.py         opencv_fallback (tested) + insightface_arcface (real, optional)
    detectors.py         onnx_yolo (real ONNX) + null
    components.py        onnx_components (real ONNX) + null_components
    wires.py             classical_wires (real baseline) + null_wires
    face_service.py      enrol / persist / cosine match / threshold
    manager.py           AIModelManager: select/enable/params, metrics, persistence
    pipeline.py          per-frame inference, overlays, event + detection logging
  api/
    employees.py         employee CRUD + face enrolment (upload / capture)
    ai.py                model manager, params, settings, metrics
    analysis.py          events, components, wires/topology, dashboard stats
  web/                   vanilla-JS dashboard (index.html, css/, js/) served at /
```

The **pipeline judges the whole conversation of a frame**: it runs only the
tasks that are enabled and ready, draws overlays, persists structured results,
de-duplicates repeated events, and pushes `ai_event` messages over the existing
WebSocket so the dashboard updates live.

---

## New API surface (added by the platform)

Employee management & enrolment:

```
GET    /api/employees                     list (with images + embedding counts)
POST   /api/employees                     create
GET    /api/employees/{id}                fetch one
PUT    /api/employees/{id}                update
DELETE /api/employees/{id}                delete (cascades images + embeddings)
POST   /api/employees/{id}/images         enrol from an uploaded image (base64)
POST   /api/employees/{id}/capture        enrol from the current camera frame
DELETE /api/employees/{id}/images/{img}   remove one image (+ its embedding)
```

AI model manager, settings & metrics:

```
GET    /api/ai/status                     full status: tasks, backends, resources
GET    /api/ai/catalog                    registered backends per task
GET    /api/ai/metrics                    fps / inference-ms / cpu / ram / gpu
GET    /api/ai/models/{task}              one task's status
POST   /api/ai/models/{task}/select       choose the active backend
POST   /api/ai/models/{task}/enable       enable / disable the task
POST   /api/ai/models/{task}/params       set thresholds / device (persisted)
GET    /api/ai/settings                   all persisted settings
GET/PUT /api/ai/settings/{key}            read / write a persisted setting
```

Detections, events, and stats:

```
GET    /api/events                        recent events (filter by type / camera)
GET    /api/events/types                  event-type histogram
DELETE /api/events                        clear the event log
GET    /api/components                    logged component detections
GET    /api/components/classes            the 18 supported electrical classes
GET    /api/components/summary            counts + average confidence per type
GET    /api/wires                         logged wire records
GET    /api/wires/topology                live nodes (components) + edges (wires)
GET    /api/stats/dashboard               employees / recognition / electrical / cameras / resources
```

AI-annotated media & stored files:

```
GET    /api/cameras/{id}/ai-stream        MJPEG with detection overlays
GET    /api/cameras/{id}/ai-snapshot      single annotated frame
GET    /api/cameras/{id}/analyze          structured results for the current frame (no image)
GET    /api/media/{path}                  serve stored snapshots / employee images (traversal-guarded)
```

Interactive docs for every endpoint remain at `/docs`.

---

## Registration, recognition & model status

### Enrolling employees from the live stream

The Employees page drives the whole flow from the browser, with no manual image
upload required when a camera is available:

1. **Add employee** opens a form with the camera **live inside the page**.
2. Fill in the details and click **Capture** — each capture reads the *latest
   buffered frame* (never a new RTSP connection) and validates it before it is
   kept, reporting one of: face accepted, **no face detected**, **too blurry**,
   or **multiple faces** (a warning — the largest face is used).
3. Capture **multiple** samples; **delete** or **retake** any of them before
   saving.
4. **Save employee** creates the record, enrols every kept sample (generating
   and storing a face embedding per image), and recognition works **immediately**
   — no backend restart.

Blur is measured by the variance of the Laplacian of the face crop; the
threshold is the tunable `min_blur` parameter on the face model (default `40`).
A rejected capture never leaves an orphaned image or embedding behind.

Capture is fully non-blocking: image decode, face detection, embedding, and disk
I/O all run in the threadpool (`asyncio.to_thread`), so the MJPEG stream stays
smooth during enrolment. This is verified under load in
`tests/test_capture_concurrency.py` (12 concurrent real-detection captures while
health pings stay responsive).

### Continuous recognition & attendance

A background worker (started with the app) runs the enabled AI tasks on each
camera's latest frame on a throttled loop, so recognition and attendance logging
happen **automatically and continuously** even when nobody is viewing the
stream. Recognised faces are logged as `face_recognized` events (name +
confidence); unrecognised faces as `unknown_person` (labelled "Unknown Person"
with a saved snapshot). Events are **de-duplicated** per camera/identity, so a
person standing in view does not create one event per frame — verified in
`tests/test_continuous_recognition.py`.

### AI model status vocabulary

Every task on the AI Models page reports one of six states, and — when a model
is unavailable — the **exact reason**, so nothing ever fails silently:

| State | Meaning |
| --- | --- |
| **Loaded** | Backend initialised and ready |
| **Running** | Ready and actively producing inference right now |
| **Loading** | Initialising |
| **Not Loaded** | Selected but not initialised (e.g. weights absent) |
| **Disabled** | Turned off for the pipeline (but still probed, so problems show) |
| **Error** | Initialisation failed |

Reasons include: `weights_missing`, `onnx_file_missing`, `onnxruntime_missing`,
`insightface_missing`, `cuda_unavailable` (falls back to CPU), and
`init_failed`. Present pretrained weights are **loaded automatically at
startup**; missing ones are reported, never faked.

### Training a model

A complete, tested training pipeline lives in `training/` (scikit-learn +
`skl2onnx`): dataset loading (image-folder or a real digits self-test),
train/validation/test split, accuracy/loss/precision/recall/F1, ONNX export, and
a reload check under ONNX Runtime (the backend's engine). Run
`python -m training.train` for the self-test; see `training/README.md` for
training on your own data. It is exercised by `tests/test_training.py`.

## The dashboard

Open `http://localhost:8000/` for a control-room UI covering:

- **Dashboard** — live employee / recognition / component / wiring-error /
  camera / CPU / RAM counters and AI-module status.
- **Live Cameras** — multi-camera switching, the AI overlay stream, snapshot
  capture, full-screen, and a live per-frame results panel.
- **Employees** — full CRUD, plus enrolment by upload or by capturing the live
  camera frame, with per-image thumbnails and deletion.
- **Events** — the real-time feed with snapshots, filterable by type.
- **Components** / **Wire Topology** — the supported classes, detection
  summaries, and a live topology view (honest about model availability).
- **AI Models** — enable/disable each task, switch backends, tune confidence /
  IoU / match-threshold, pick the device, and watch live FPS / inference time /
  resource usage.
- **Settings** — persisted key/value settings and system status.

The dashboard is plain HTML/CSS/JS (no build step) so it is served directly and
every control calls a real endpoint above.

---

## Running the platform

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py                      # or: uvicorn rtsp_backend.app:build_app --factory
# open http://localhost:8000/
```

Storage locations (auto-created) are configurable via YAML or environment:
`RTSP_DATA_DIR` (default `data/`), `RTSP_DB_PATH` (default `data/platform.db`),
`RTSP_MODELS_DIR` (default `models/`), `RTSP_AI_MIN_INTERVAL` (inference throttle,
seconds).

## Testing

```bash
pip install -r requirements-dev.txt   # includes the training deps
pytest -q
```

The suite (80+ tests) covers the database; the face-recognition pipeline with a
**real** face image (enrol, recognise, unknown handling, threshold gating,
persistence across a restart, cache rebuild); capture validation (no-face, blur
rejection, multi-face warning, orphan cleanup); the **non-blocking capture**
under concurrent load; **continuous background recognition** with de-duplication;
the AI manager, its persistence, auto-load of present weights, and the full
status/reason vocabulary; every new API endpoint; WebSocket `ai_event` delivery;
settings persistence across an app restart; the AI overlay renderer;
path-traversal protection; and the **training pipeline** (metrics, ONNX export,
and ONNX-reload verification) — alongside the original RTSP suite. Capabilities
that require an unavailable trained model are explicitly asserted to return
**empty, honest results** rather than fabricated detections.

---

## RTSP core (unchanged)

The remainder of this document describes the original RTSP-only backend, which
is fully retained.

---

## RTSP only — no silent fallback

This is the defining guarantee of the backend, so it is worth stating plainly:

- The only accepted sources are `rtsp://` and `rtsps://` URLs.
- There is **no** support for USB / local capture devices, numeric indices
  (`0`, `1`, …), HTTP(S) streams, or local video files. Those code paths do
  not exist.
- The source **always** comes from configuration (or an explicit API call). If
  a URL is invalid or a camera is unreachable, the backend returns a structured
  JSON error explaining exactly why. It **never** falls back to device `0`, to
  another camera, or to a placeholder image.

Rejection happens at two layers. Invalid URLs are refused up front (fail-fast at
startup, `400` on the API) by a single validator, and an unreachable-but-valid
camera surfaces as a `503` at the media endpoints rather than as a substituted
source.

```jsonc
// POST /cameras  { "id": "cam", "url": "0" }   ->  HTTP 400
{
  "error": {
    "code": "invalid_rtsp_url",
    "message": "Numeric camera index '0' is not allowed. This backend is RTSP-only and never opens USB or local capture devices.",
    "camera_id": "cam",
    "details": { "provided": "0", "reason": "numeric_device_index" }
  }
}
```

The validator distinguishes the failure so callers can react programmatically.
The `details.reason` is one of: `numeric_device_index`, `local_file_or_path`,
`unsupported_scheme`, `missing_host`, or `empty_url`.

---

## Install

```bash
pip install -r requirements.txt
```

OpenCV is used in its headless build (`opencv-python-headless`) and carries its
own FFmpeg, which handles the RTSP transport. No system FFmpeg or GUI libraries
are required.

## Configure

Copy `config.example.yaml` to `config.yaml` and edit it (or point `RTSP_CONFIG`
at any path):

```yaml
host: 0.0.0.0
port: 8000
stats_interval: 5.0          # seconds between WebSocket "stats" telemetry events
active_camera: front_door    # camera used by /snapshot and /stream

cameras:
  - id: front_door
    name: Front Door
    url: rtsp://admin:password@192.168.1.10:554/Streaming/Channels/101
    transport: auto           # auto (default: try tcp, then udp), tcp, or udp
    jpeg_quality: 80

  - id: yard
    name: Back Yard
    url: rtsp://admin:password@192.168.1.11:554/stream1
    reconnect_delay: 2.0        # initial reconnect backoff (seconds)
    max_reconnect_delay: 30.0   # backoff cap (seconds)
    read_timeout: 10.0          # FFmpeg socket timeout (seconds)
    max_read_failures: 30       # consecutive failed reads before a reconnect
    # target_fps: 15            # optional: throttle the capture loop
```

Per-camera fields: `id` (required), `url` (required, RTSP), `name`, `transport`
(`auto` default: tries `tcp` then `udp`; or pin `tcp`/`udp`), `reconnect_delay`, `max_reconnect_delay`,
`open_timeout`, `read_timeout`, `max_read_failures`, `target_fps`,
`jpeg_quality`.

A few settings can be overridden from the environment, which is handy in
containers: `RTSP_CONFIG`, `RTSP_HOST`, `RTSP_PORT`, `RTSP_STATS_INTERVAL`,
`RTSP_ACTIVE_CAMERA`.

These can also live in a `.env` file that is loaded automatically at startup.
Copy `.env.example` to `.env` and edit it. Real environment variables always
take precedence over `.env`, so the file holds this instance's defaults. To run
several independent instances side by side, give each its own file and select it
with `RTSP_ENV_FILE` (use a distinct `RTSP_PORT`, and usually its own
`RTSP_CONFIG`, per instance):

```bash
RTSP_ENV_FILE=instance-a.env python run.py   # e.g. RTSP_PORT=8000, its own cameras
RTSP_ENV_FILE=instance-b.env python run.py   # e.g. RTSP_PORT=8001, its own cameras
```

## Run

```bash
python run.py
# or
uvicorn run:app --host 0.0.0.0 --port 8000
```

Interactive API docs are served at `/docs` (Swagger UI) and `/redoc`.

---

## API

Base URL: `http://<host>:<port>`

| Method | Path | Description |
| ------ | ---- | ----------- |
| `GET` | `/health` | Liveness plus a compact overview of every camera. |
| `GET` | `/cameras` | List all cameras with full status. |
| `POST` | `/cameras` | Add a camera at runtime. Body: `{id, url, ...}`. `201` on success. |
| `GET` | `/cameras/{id}` | One camera's full status. |
| `GET` | `/cameras/{id}/status` | Same status payload (explicit status route). |
| `GET` | `/cameras/{id}/diagnose` | Connection diagnostic: probes the camera over TCP and UDP and returns the real FFmpeg errors plus a verdict. `?timeout=1..60`. |
| `PUT` | `/cameras/{id}` | Update a camera at runtime (e.g. switch its URL). Restarts capture. |
| `DELETE` | `/cameras/{id}` | Stop and remove a camera. |
| `GET` | `/active-camera` | The camera `/snapshot` and `/stream` currently use. |
| `POST` | `/active-camera` | Switch the active camera. Body: `{id}`. |
| `GET` | `/cameras/{id}/snapshot` | Single JPEG frame. `?quality=1..100`. |
| `GET` | `/snapshot` | Snapshot from the active camera. |
| `GET` | `/cameras/{id}/stream` | MJPEG stream (`multipart/x-mixed-replace`). `?quality`, `?fps`. |
| `GET` | `/stream` | MJPEG stream from the active camera. |
| `WS` | `/ws/events` | Live event stream (state changes, telemetry). |

### Camera status

`GET /cameras/{id}` returns everything a monitoring client needs. The stream URL
is redacted so credentials never leak into logs or dashboards:

```json
{
  "id": "front_door",
  "name": "Front Door",
  "url": "rtsp://admin:***@192.168.1.10:554/Streaming/Channels/101",
  "transport": "tcp",
  "state": "connected",
  "healthy": true,
  "has_frame": true,
  "frame_seq": 1423,
  "frame_age_ms": 41.7,
  "connected_for_seconds": 92.4,
  "fps": 24.9,
  "latency": { "last_ms": 38.2, "avg_ms": 41.0, "max_ms": 88.5 },
  "statistics": {
    "frames_captured": 2301,
    "frames_dropped": 3,
    "read_failures_total": 3,
    "reconnect_count": 1,
    "connected_since": 1751731200.0,
    "last_frame_at": 1751731292.4,
    "last_error": null,
    "last_error_at": null,
    "uptime_seconds": 300.1
  }
}
```

`state` moves through `initializing → connecting → connected`, and on trouble
`reconnecting` (with `error` / `stopped` as terminal-ish states). `healthy` is a
quick boolean: connected and delivering fresh frames.

### Errors

Every error uses the same envelope: `{"error": {code, message, camera_id?, details?}}`.

| Status | `code` | When |
| ------ | ------ | ---- |
| `400` | `invalid_rtsp_url` | URL is not a usable RTSP URL (see `details.reason`). |
| `404` | `camera_not_found` | No camera with that id. |
| `404` | `no_active_camera` | No active camera is set / none configured. |
| `409` | `duplicate_camera` | A camera with that id already exists. |
| `503` | `camera_not_connected` | Camera is not connected; no frame to serve. |
| `503` | `frame_unavailable` | Connected but no frame yet, or JPEG encode failed. |
| `500` | `internal_error` | Unexpected server error; the message names the exception and the traceback is in the server log. |

A snapshot from an unreachable camera returns JSON, **not** a placeholder image:

```jsonc
// GET /cameras/front_door/snapshot  ->  HTTP 503, application/json
{
  "error": {
    "code": "camera_not_connected",
    "message": "Camera 'front_door' is not connected (state: reconnecting); no snapshot is available.",
    "camera_id": "front_door",
    "details": { "state": "reconnecting", "last_error": "connection refused" }
  }
}
```

---

## Examples

**Snapshot to a file:**

```bash
curl -s "http://localhost:8000/cameras/front_door/snapshot?quality=90" -o frame.jpg
```

**MJPEG in a browser** — point an `<img>` at the stream endpoint:

```html
<img src="http://localhost:8000/cameras/front_door/stream" alt="live" />
<!-- or the active camera, throttled: -->
<img src="http://localhost:8000/stream?fps=10&quality=70" />
```

**Live events over WebSocket** (browser / Node):

```js
const ws = new WebSocket("ws://localhost:8000/ws/events");

ws.onmessage = (msg) => {
  const evt = JSON.parse(msg.data);
  // evt.type: "hello" | "state" | "stats" (and other lifecycle events)
  if (evt.type === "stats") {
    console.log(evt.camera_id, evt.data.fps, evt.data.latency.avg_ms);
  }
};
```

On connect the server sends a `hello` event containing the current status of
every camera, then streams updates as they happen; telemetry (`stats`) is pushed
on the `stats_interval`.

**Add / switch a camera at runtime:**

```bash
# Add
curl -X POST http://localhost:8000/cameras \
  -H "Content-Type: application/json" \
  -d '{"id":"line2","name":"Line 2","url":"rtsp://10.0.0.20:554/live"}'

# Repoint an existing camera (capture restarts on the new URL)
curl -X PUT http://localhost:8000/cameras/line2 \
  -H "Content-Type: application/json" \
  -d '{"url":"rtsp://10.0.0.21:554/live"}'

# Make it the default for /snapshot and /stream
curl -X POST http://localhost:8000/active-camera \
  -H "Content-Type: application/json" -d '{"id":"line2"}'
```

---

## Integration notes

The backend is a headless service; every consumer uses REST + WebSockets.

- **Mobile apps** — render `/stream` in an image view for a lightweight live feed,
  poll `/snapshot` for thumbnails, and open `/ws/events` for connection state so
  the UI can show "camera offline" accurately.
- **Robotics** — pull frames via `/snapshot` (or consume MJPEG) and watch
  `latency` / `frame_age_ms` to decide whether a frame is fresh enough to act on.
- **Industrial HMIs** — embed the MJPEG stream directly in a panel; drive
  status indicators from `state` / `healthy`.
- **SCADA** — treat `/cameras/{id}/status` as a telemetry source. `state`,
  `reconnect_count`, `frames_dropped`, and `read_failures_total` map cleanly onto
  tags and alarms; `/health` works as a service heartbeat.
- **Other backends** — orchestrate cameras through the CRUD endpoints and react to
  `/ws/events` instead of polling.

Because credentials are embedded in RTSP URLs, run this behind your own
authentication / network boundary (reverse proxy, VPN, or private VLAN). Stream
URLs are redacted everywhere the backend reports them, but the backend itself
does not add an auth layer.

---

## Architecture

```
             +------------------------------------------+
 RTSP cams   |                FastAPI app                |  REST + WS clients
 =========   |                                          |  ================
  cam A ---> |  CameraManager ---- EventBus ----+       |  mobile / HMI /
  cam B ---> |     |     \                       \      |  robotics / SCADA
  cam C ---> |     |      RTSPCamera (per cam)    \     |
             |     |        - capture thread       -> WebSocket /ws/events
             |     |        - FrameBuffer (1 slot)      |
             |     +------> status / snapshot / MJPEG -> REST endpoints
             +------------------------------------------+
```

- **`RTSPCamera`** runs one background daemon thread per camera. It opens *only*
  its configured URL — TCP transport and a 1-frame capture buffer are forced via
  FFmpeg options for low latency — reads frames, records latency and FPS, and on
  repeated read failures reconnects with exponential backoff. Opens are
  serialized by a module lock to avoid FFmpeg races when many cameras start at
  once.
- **`FrameBuffer`** is a thread-safe single-slot buffer (a `Condition` guards the
  latest frame and its sequence number). Readers always get the newest frame and
  can block efficiently for the next one — this is what lets MJPEG push frames
  without busy-waiting and without backing up stale frames.
- **`CameraManager`** owns the camera registry (add / remove / update / set
  active) behind a lock, so runtime changes are safe while requests are in
  flight.
- **`EventBus`** bridges the capture threads to the async world, fanning state
  changes and telemetry out to WebSocket subscribers (dropping the oldest event
  for a slow consumer rather than blocking capture).
- **`app.py`** wires it together, renders all `RTSPBackendError`s as JSON, and
  runs a heartbeat task that emits periodic `stats` events.

### Metrics: latency, frame drops, connections

- **Latency** (`latency.last_ms` / `avg_ms` / `max_ms`) is the time each
  `read()` takes to return a frame — a practical proxy for pipeline delay.
- **FPS** (`fps`) is measured from the spacing of recently captured frames.
- **Frame drops** (`statistics.frames_dropped`) count reads that failed while the
  camera was nominally connected — the low-level signal of a flaky link.
- **Connection stats** (`reconnect_count`, `read_failures_total`,
  `connected_since`, `uptime_seconds`, `last_error`) describe link stability over
  time and are what you graph or alarm on.

---

## Troubleshooting: "it works in VLC but not here"

Run the diagnostic — it probes the URL over both transports and prints the
**real FFmpeg error**, a raw TCP reachability check, and a verdict:

```bash
# CLI (quote the URL — an unquoted & breaks the shell):
python diagnose.py "rtsp://admin:pass@192.168.100.5:554/ch=1&subtype=0"

# or against a configured camera, via the API:
curl "http://localhost:8000/cameras/front_door/diagnose?timeout=10"
```

How to read the result:

- `tcp_port.reachable: false` — network problem (wrong IP/port, camera off,
  firewall/VLAN). Fix the network first; RTSP settings don't matter yet.
- `401` in `ffmpeg_stderr` — wrong username/password.
- `404` / `DESCRIBE failed` — wrong stream path; check the camera's web UI.
- one transport succeeds and the other fails — pin `transport:` to the working
  one (the default `transport: auto` already tries `tcp` then `udp`, which
  covers cameras that only implement one of the two — the most common cause of
  the VLC-works-backend-doesn't symptom, since VLC negotiates transports).
- everything fails with timeouts but VLC is open — **close VLC**: many cameras
  allow only one or two concurrent RTSP clients.

`GET /cameras/{id}` also reports `transport_in_use` (the transport the current
connection actually uses) and `statistics.last_error`.

Version note: the FFmpeg RTSP socket-timeout option was renamed (`stimeout` →
`timeout` in FFmpeg 5). The backend detects the bundled FFmpeg version and
passes the correct one, and additionally enforces `open_timeout` /
`read_timeout` through OpenCV's own capture timeouts.

## Tests

Install the test dependencies first (the suite uses FastAPI's `TestClient`,
which needs `httpx`), then run `pytest`:

```bash
pip install -r requirements-dev.txt
pytest
```

The smoke suite covers URL validation (numeric indices, file paths, and HTTP
URLs are all rejected; `rtsp`/`rtsps` accepted), the health and CRUD endpoints,
the duplicate/not-found/active-camera error paths, and — importantly — that a
snapshot from an unreachable camera returns a JSON `503` rather than an image.

## Project layout

```
rtsp-backend/
├── run.py                    # entrypoint (python run.py / uvicorn run:app)
├── diagnose.py               # CLI: diagnose why an RTSP URL won't open
├── config.example.yaml       # sample configuration
├── .env.example              # sample environment file (copy to .env)
├── requirements.txt          # runtime dependencies
├── requirements-dev.txt      # test dependencies (pytest, httpx)
├── rtsp_backend/
│   ├── app.py                # FastAPI app: routes, MJPEG, WebSocket, error handlers
│   ├── camera.py             # RTSPCamera capture thread + FrameBuffer
│   ├── manager.py            # CameraManager registry
│   ├── config.py             # settings + YAML/.env loading
│   ├── events.py             # EventBus (thread -> async bridge)
│   ├── errors.py             # error types + the RTSP-only URL validator
│   ├── diagnostics.py        # RTSP probe: per-transport attempts + FFmpeg stderr
│   ├── schemas.py            # request models
│   └── __init__.py
└── tests/
    ├── test_smoke.py
    └── test_open_path.py
```
