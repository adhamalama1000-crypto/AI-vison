# v3.2.0 — Optimization pass

This release optimizes the platform in place. It **adds no fabricated behavior**:
every new AI module runs *real* inference when weights are present and honestly
reports `weights_missing` ("model unavailable") otherwise — the same contract the
existing detection/component backends already followed.

Test status: **101 passed, 1 skipped** (`pytest --ignore=tests/test_training.py`).
The 3 training tests require the optional `skl2onnx` package and are unrelated to
this work. 17 new tests were added; no existing test was weakened (one assertion
was widened to reflect the intentionally larger task set).

---

## Priority 1 — Live-stream latency

Root causes addressed in `rtsp_backend/camera.py`, `ai/pipeline.py`, `app.py`:

1. **Capture-side throttle removed.** The old loop slept between `cap.read()`
   calls when `target_fps` was set, which lets the FFmpeg/decoder queue grow and
   *adds* latency. Replaced with a **grab-drain** strategy: when throttling,
   intermediate frames are `grab()`-ed (no decode cost) and discarded, and only
   the newest frame is `retrieve()`-ed and published. Latency no longer
   accumulates regardless of the configured rate.
2. **Ultra-low-latency FFmpeg tuning** (new `low_latency` camera flag, default
   on): `fflags;nobuffer`, `flags;low_delay`, `max_delay;0`,
   `reorder_queue_size;0`, small `probesize`, `analyzeduration;0`.
3. **AI overlay stream no longer blocked by inference.** The `/ai-stream`
   endpoint used to run full inference synchronously per streamed frame
   (`annotated_jpeg`, `force=True`). It now draws the *most recent* detections
   (computed by the background AI worker) onto the freshest frame
   (`annotated_jpeg_fast` → `draw_overlays`). The AI video now has the same
   latency as the raw stream; inference runs only on the background worker and
   never blocks a delivered frame.
4. **Metrics.** Encode time is now measured per JPEG; a new consolidated
   `GET /api/metrics` returns per-camera capture FPS, stream latency (frame
   age), read time, encode time, drops/reconnects; per-task AI FPS + inference
   ms; and host CPU/RAM (GPU honestly reported unavailable here).

Single shared connection, latest-frame buffer, drop-old-frames, no per-request
reconnect — these were already correct in the original design and are preserved.

## Priority 2 — Face detection & recognition (`ai/face_service.py`)

- **Tiny-face filter** (`min_face_size`): faces below the size floor are ignored.
- **Duplicate suppression**: overlapping detector boxes merged via NMS.
- **Per-face quality score** (0..1 from size + Laplacian sharpness), attached to
  each detection's `extra` along with `blur` and `min_side`.
- **Blur gating in recognition**: too-blurry faces are reported but never matched
  (won't identify a smudge as an employee).
- **Multi-embedding top-k voting**: identity is decided by the k nearest stored
  vectors voting, far more stable than a single argmax while people move.
- InsightFace **SCRFD + ArcFace** backend (`insightface_arcface`) was already
  present and is unchanged; it activates when the library + models are installed.

## Priority 3 — New AI modules (`ai/modules.py`)

New tasks, each with its own runtime toggle, metrics, overlays, and events:
`fire` (fire/smoke/explosion), `weapon` (gun/rifle/knife), `ppe`
(helmet/vest/gloves/goggles + `no_*` violations), `violence`, `fall`, `human`
(person/crowd count), `vehicle` (car/truck/bus/motorcycle). All reuse the
existing real ONNX engine (letterbox → forward pass → decode → NMS). `fall` also
offers an opt-in classical aspect-ratio heuristic. Drop weights into
`models/<task>/` to activate — no code changes. Proven end-to-end with a real
ONNX model during development.

## Priority 4 — Tracking (`ai/tracker.py`)

Self-contained **ByteTrack-style** tracker (two-stage association,
constant-velocity motion model, Hungarian assignment via scipy with a greedy
fallback), plus a per-class wrapper so IDs never cross class lines. Wired into
the pipeline so object/human/vehicle/module detections carry a stable
`track_id`. No weights needed; fully unit-tested.

## Priority 5 — Performance

ONNX Runtime is the inference engine throughout; backends request the CUDA
provider when `device: gpu` is set and **fall back to CPU automatically**,
reporting `cuda_unavailable` honestly. FP16/TensorRT/dynamic batching require a
CUDA build + GPU not present in this environment; the code paths select GPU
providers when available but were not runnable here.

## Priority 7 — Events

The generic module loop emits and persists (SQLite) events for `fire`, `smoke`,
`explosion`, `weapon`, `violence`, `fall`, and `ppe_violation`, with snapshots
and 10s de-duplication, alongside the existing `unknown_person`,
`face_recognized`, `wiring_error`, and camera connect/disconnect events.

## Priority 8 — Code quality

The seven new modules share one generic, table-driven pipeline path rather than
seven copy-pasted branches; tracker assignment and overlay drawing are factored
into single helpers used by both the live and cached paths.

---

## Not done in this environment (needs hardware / weights / a live camera)

Being straight about the boundary:

- **No sub-300 ms end-to-end verification** — there is no live RTSP camera here.
  The latency *code* is fixed and unit-tested; the number must be confirmed
  against real hardware.
- **No trained weights** for fire/violence/fall/weapon/PPE — none exist as
  free drop-ins, and fabricating detections was explicitly out of scope. Each
  module is wired and honest; supply weights to activate.
- **No GPU / TensorRT / FP16** validation — CPU-only environment. GPU provider
  selection + CPU fallback are implemented but only the CPU path was exercised.
- **Dashboard redesign (Priority 6) not rebuilt.** The backend metrics the new
  dashboard needs are all exposed at `GET /api/metrics`; the React rebuild was
  out of scope for this pass.

## New / changed files

    rtsp_backend/ai/tracker.py      NEW  ByteTrack tracker
    rtsp_backend/ai/modules.py      NEW  fire/weapon/ppe/human/vehicle/violence/fall
    rtsp_backend/ai/pipeline.py     mod  generic event loop, tracking, fast overlays
    rtsp_backend/ai/manager.py      mod  new tasks, defaults, face params
    rtsp_backend/ai/face_service.py mod  quality, NMS, tiny-face, top-k voting
    rtsp_backend/camera.py          mod  grab-drain, low-latency FFmpeg, encode metric
    rtsp_backend/config.py          mod  low_latency flag
    rtsp_backend/app.py             mod  /api/metrics, fast AI stream
    tests/test_tracker.py           NEW  6 tests
    tests/test_modules.py           NEW  5 tests
    tests/test_runtime_smoke.py     NEW  3 tests
    tests/test_face_recognition.py  mod  +3 quality tests
    tests/test_api_ai.py            mod  task-set assertion widened
    models/<task>/                  NEW  weight drop-in dirs + README update
