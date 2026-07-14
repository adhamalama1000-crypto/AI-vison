# Changelog

## [3.1.5] — Production React frontend + RTSP-camera-only registration

### Frontend — complete rebuild (React + TypeScript + Vite)

The previous plain-HTML/JS interface has been replaced by a modern, responsive
single-page application suitable for commercial deployment. It is pre-built into
`rtsp_backend/web/` and served directly by the backend — **no Node.js is needed
at runtime**. Source lives in `frontend/`.

- **Stack:** React 18 + TypeScript + Vite, Tailwind CSS design system with
  light/dark theme tokens, TanStack Query for data/polling, Recharts for
  charts, lucide-react icons, React Router.
- **Dark & light mode** with a persisted preference and instant toggle.
- **Dashboard** — live stat cards, real-time FPS area chart (fed by the events
  WebSocket), CPU/RAM radial gauges, a recognition donut, per-task AI status,
  and a recent-events feed. All values come from `/api/stats/dashboard` and the
  `/ws/events` socket.
- **Live Cameras** — large RTSP preview with an AI-overlay/raw toggle,
  fullscreen support, one-click snapshot download, a camera switcher with live
  thumbnails, and per-camera stats (fps, latency, frame age, uptime, transport,
  frame counters).
- **Employees** — searchable, paginated table with add / edit / delete. The
  add/edit dialog shows the **live camera inside the dialog**, captures faces
  **directly from the RTSP stream**, supports **multiple captures** with a
  **thumbnail gallery** and per-image **retake/delete**, saves atomically via
  `/api/employees/register`, then switches to a **live verification** view that
  draws recognition overlays and confirms the match on-screen.
- **Events** — filter by event type and camera, snapshot thumbnails with a
  full-size preview dialog, absolute + relative timestamps, and clear-all.
- **AI Models** — a card per task showing status, device, throughput (fps),
  inference time, an enable/disable toggle, a backend selector, and a match
  threshold slider, plus compute-resource gauges.
- **Settings** — RTSP camera CRUD (with `rtsp://` validation), appearance/theme
  controls, and system health.
- **Serving:** the backend redirects `/` → `/app/`, serves the built assets, and
  falls back to `index.html` for client-side routes so deep links survive a
  refresh. The app is fully self-contained (no external CDN/font dependencies).

### Backend

## [3.1.5-backend] — RTSP-camera-only employee registration

This release makes employee enrolment work **entirely from the live RTSP
camera** and guarantees recognition works **immediately after enrolment** — no
manual image upload, no manual model toggle, and no backend restart.

### Fixed

- **Recognition did not start by itself after enrolment.** The `face` task
  shipped *disabled*, so an enrolled employee stored a face vector but was never
  recognised until someone manually enabled the model on the AI Models page.
  - Face recognition is now **enabled by default** on a fresh install.
  - Enrolment now **auto-enables** the face task
    (`AIModelManager.ensure_enabled("face")`), so recognition is live on the
    very next frame.
- **The UI fell back to local file upload.** With no `config.yaml`, zero
  cameras loaded and the wizard offered "Upload image", which is exactly the
  manual path that was not wanted.
  - A working `config.yaml` is now shipped (RTSP camera declared there).
  - The **file-upload path was removed entirely** from the employee wizard.
- **A bad/degenerate frame could return HTTP 500 and interrupt capture.**
  OpenCV's Haar `detectMultiScale` can raise an internal range-check error on
  some frames.
  - `OpenCVFallbackEmbedder.detect_faces` now guards empty frames and swallows
    OpenCV internal errors, returning `[]` instead of crashing.
  - Enrolment wraps the face-service call so a bad frame becomes a clean
    rejection (`enrollment_error` / `no_face_detected` / `blurry`), never a 500.

### Added

- **`POST /api/employees/register`** — atomic RTSP enrolment: creates the
  employee and enrols every captured frame in one call. If no frame yields a
  usable face the whole operation is **rolled back** (no faceless employees).
  Response includes `enrolled`, `rejected`, and `recognition_enabled`.
- **Live in-dialog verification.** After saving, the Add-employee dialog
  switches to the AI-annotated stream (name + confidence drawn on the feed) and
  reports the recognition result live, so the operator confirms it works
  on the spot.
- Regression tests in `tests/test_rtsp_registration.py`:
  auto-enable-on-enrolment, atomic register + immediate recognition, no-face
  rollback, blurry rejection without crashing, and detector robustness on bad
  frames.

### Changed

- The Add-employee wizard is now **RTSP-only**: shows the live camera, captures
  straight from the stream (validated — rejects no-face and blurry frames, warns
  on multiple faces), supports **multiple captures with delete/retake**, then
  saves via `/register`.

### Behaviour preserved

- Capture and snapshot run in the thread pool: the MJPEG stream **does not
  freeze** during capture, and **snapshots do not interrupt** the stream (both
  verified under concurrent load).
- Recognition and unknown-person events (attendance) are written to the
  `events` table automatically by the background worker.
- The RTSP-only source policy is unchanged — there is no USB / local-file
  fallback anywhere.

---

## Verification (this release)

Performed against a live RTSP camera (a real H.264 RTSP stream carrying a real
face) and a fresh database:

1. Clean boot via `python run.py` (reads `config.yaml` + `.env`) — fresh DB.
2. Camera connects and streams (~15 fps, 0 dropped frames); MJPEG live.
3. Face recognition auto-enabled on boot (`state: running`, `ready: true`).
4. Validate live capture — single sharp face accepted; no upload used.
5. Register from captures — 2 enrolled, 0 rejected, recognition auto-enabled.
6. Recognition immediate — enrolled employee matched at ~0.9996 confidence.
7. Attendance/events stored (`face_recognized`, `unknown_person`).
8. AI overlay (`ai-snapshot` / `ai-stream`) draws name + confidence.
9. No freeze — stream keeps advancing while 6 captures + 6 snapshots run
   concurrently (capture max ~0.25 s).
10. AI Models endpoints — enable/disable/select/params/metrics all functional.
11. Full browser UI (headless Chromium) — wizard capture → save → live
    "Recognized ✓ … · 100.0% confidence", zero console errors.

Test suite: **88 passed** (`pytest`).

## Running

```bash
pip install -r requirements.txt        # runtime deps
# edit config.yaml -> cameras[0].url with your RTSP camera URL
python run.py                          # serves on the host/port from .env / config.yaml
# open the dashboard at http://<host>:<port>/  (default port 8090)
```

For tests: `pip install -r requirements-dev.txt && pytest`.

## Rebuilding the frontend (optional)

The UI is already built into `rtsp_backend/web/`, so running the backend is all
that's required. To modify and rebuild it:

```bash
cd frontend
npm install
npm run build     # outputs to ../rtsp_backend/web
# or: npm run dev  (Vite dev server on :5173, proxies the API to :8090)
```
