"""
Batch inference — throughput without changing the answer.

The correctness requirement is ordering. A batched forward pass returns one result
per input, and if that pairing slips, one panel's detections are attributed to
another image. That failure is silent, produces plausible-looking output, and would
be found by a customer rather than a test — so it is asserted here directly, along
with the guards that catch a backend which returns the wrong number of results.

The second property under test is honesty about whether batching happened at all.
A backend without a real batched path falls back to a sequential loop; the response
says so, so a throughput number is never presented as a batching win when it is not.
"""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from rtsp_backend.electrical import postprocess as pp
from rtsp_backend.electrical import recognizer as rec
from rtsp_backend.electrical import taxonomy as tax


# ==========================================================================
# fakes
# ==========================================================================

class _FakeRecognizer(rec.IndustrialRecognizer):
    """A recogniser whose detections encode which frame they came from.

    Each frame is tagged by its mean pixel value, and the returned box position is
    derived from that tag — so a mispaired result is detectable rather than
    plausible.
    """

    backend_id = "fake_industrial"
    requires_weights = False

    def __init__(self, true_batching: bool = False, **params):
        super().__init__(**params)
        self.supports_true_batching = true_batching
        self.single_calls = 0
        self.batch_calls = 0
        self._ready = True
        self._status = "ready"

    def load(self) -> None:
        self._ready = True
        self._status = "ready"

    @staticmethod
    def _tag(frame) -> int:
        return int(round(float(frame[0, 0, 0])))

    def _candidates_for(self, frame) -> list:
        tag = self._tag(frame)
        # A box whose x offset encodes the frame tag.
        return [pp.Candidate(class_id="mcb", score=0.9,
                             box=(float(tag), 10.0, float(tag) + 30.0, 90.0),
                             source=self.backend_id, raw_label="mcb")]

    def raw_candidates(self, frame):
        self.single_calls += 1
        return self._candidates_for(frame)

    def raw_candidates_batch(self, frames):
        if not self.supports_true_batching:
            return super().raw_candidates_batch(frames)
        self.batch_calls += 1
        return [self._candidates_for(f) for f in frames]


class _BrokenBatchRecognizer(_FakeRecognizer):
    """Returns the wrong number of results — the misattribution hazard."""

    backend_id = "broken_batch"

    def raw_candidates_batch(self, frames):
        self.batch_calls += 1
        return [self._candidates_for(frames[0])]        # one result for N frames


class _ExplodingBatchRecognizer(_FakeRecognizer):
    backend_id = "exploding_batch"

    def raw_candidates_batch(self, frames):
        self.batch_calls += 1
        raise RuntimeError("batched kernel blew up")


def _frame(tag: int, w: int = 200, h: int = 150) -> np.ndarray:
    return np.full((h, w, 3), tag, np.uint8)


# ==========================================================================
# ordering — the correctness requirement
# ==========================================================================

def test_batch_returns_one_result_per_frame_in_order():
    backend = _FakeRecognizer(true_batching=True)
    frames = [_frame(t) for t in (20, 40, 60, 80, 100)]
    results = backend.recognize_batch(frames, batch_size=2)

    assert len(results) == len(frames)
    for frame, result in zip(frames, results):
        assert result.accepted, "every frame should produce a detection"
        # The box x offset encodes the source frame; a mispairing shows up here.
        assert result.accepted[0].box[0] == pytest.approx(
            float(backend._tag(frame)))


def test_batch_and_single_inference_agree():
    """Batching is an optimisation; it must not change the answer."""
    backend = _FakeRecognizer(true_batching=True)
    frames = [_frame(t) for t in (30, 55, 90)]
    batched = backend.recognize_batch(frames, batch_size=3)
    singles = [backend.recognize(f) for f in frames]

    for b, s in zip(batched, singles):
        assert len(b.accepted) == len(s.accepted)
        for cb, cs in zip(b.accepted, s.accepted):
            assert cb.class_id == cs.class_id
            assert cb.box == pytest.approx(cs.box)
            assert cb.score == pytest.approx(cs.score)


def test_chunking_does_not_reorder_across_chunk_boundaries():
    backend = _FakeRecognizer(true_batching=True)
    tags = list(range(10, 130, 10))
    frames = [_frame(t) for t in tags]
    for batch_size in (1, 2, 3, 5, 12, 50):
        results = backend.recognize_batch(frames, batch_size=batch_size)
        got = [int(r.accepted[0].box[0]) for r in results]
        assert got == tags, f"reordered at batch_size={batch_size}"


def test_a_backend_returning_the_wrong_result_count_is_refused():
    """Silently pairing them up would misattribute detections across panels."""
    backend = _BrokenBatchRecognizer(true_batching=True)
    with pytest.raises(RuntimeError) as exc:
        backend.recognize_batch([_frame(20), _frame(40), _frame(60)],
                                batch_size=3)
    assert "misattributed" in str(exc.value)


def test_a_failing_batched_path_falls_back_per_frame():
    """One bad chunk must not lose the results for that chunk."""
    backend = _ExplodingBatchRecognizer(true_batching=True)
    frames = [_frame(t) for t in (20, 40, 60)]
    results = backend.recognize_batch(frames, batch_size=3)
    assert len(results) == 3
    assert backend.single_calls == 3, "should have fallen back to per-frame"
    for frame, result in zip(frames, results):
        assert result.accepted[0].box[0] == pytest.approx(
            float(backend._tag(frame)))


# ==========================================================================
# the default sequential path
# ==========================================================================

def test_the_default_batch_path_is_a_correct_sequential_loop():
    """A backend that cannot batch inherits working behaviour, not a wrong one."""
    backend = _FakeRecognizer(true_batching=False)
    frames = [_frame(t) for t in (25, 50, 75, 100)]
    results = backend.recognize_batch(frames, batch_size=2)

    assert len(results) == 4
    assert backend.batch_calls == 0
    assert backend.single_calls == 4
    for frame, result in zip(frames, results):
        assert result.accepted[0].box[0] == pytest.approx(
            float(backend._tag(frame)))


def test_supports_true_batching_is_reported_in_status():
    """A throughput claim must be checkable."""
    assert _FakeRecognizer(true_batching=True).status()[
        "supports_true_batching"] is True
    assert _FakeRecognizer(true_batching=False).status()[
        "supports_true_batching"] is False


def test_the_ultralytics_backend_declares_real_batching():
    """It passes a list source to predict(), which is one forward pass."""
    assert rec.UltralyticsIndustrialRecognizer.supports_true_batching is True


def test_the_base_recognizer_does_not_claim_batching():
    assert rec.IndustrialRecognizer.supports_true_batching is False


# ==========================================================================
# edge cases
# ==========================================================================

def test_an_empty_batch_is_an_empty_list():
    assert _FakeRecognizer(true_batching=True).recognize_batch([]) == []


def test_a_batch_size_below_one_is_refused():
    backend = _FakeRecognizer(true_batching=True)
    with pytest.raises(ValueError):
        backend.recognize_batch([_frame(20)], batch_size=0)


def test_infer_batch_returns_detections_per_frame():
    backend = _FakeRecognizer(true_batching=True)
    out = backend.infer_batch([_frame(20), _frame(50)], batch_size=2)
    assert len(out) == 2
    for dets in out:
        assert dets and dets[0].extra["class_id"] == "mcb"
        assert dets[0].label == tax.display_name("mcb")


def test_frames_of_different_sizes_are_gated_against_their_own_geometry():
    """The plausibility gate is relative to each image, so it cannot be shared."""
    backend = _FakeRecognizer(true_batching=True)
    frames = [_frame(20, w=200, h=150), _frame(40, w=1920, h=1080)]
    results = backend.recognize_batch(frames, batch_size=2)
    assert len(results) == 2
    # Both must have been evaluated; the gate must not have crashed on the
    # mismatched shapes.
    assert all(r.diagnostics is not None for r in results)


# ==========================================================================
# service layer
# ==========================================================================

def test_analyze_batch_falls_back_without_a_batched_backend():
    from rtsp_backend import panel_svc

    class _Manager:
        def backend(self, task):
            return None

    frames = [_frame(20), _frame(40)]
    results = panel_svc.analyze_batch(_Manager(), frames)
    assert len(results) == 2
    for r in results:
        assert r["component_model_loaded"] is False
        assert r["components"] == []
        assert "report" in r


def test_analyze_batch_of_nothing_is_empty():
    from rtsp_backend import panel_svc

    class _Manager:
        def backend(self, task):
            return None

    assert panel_svc.analyze_batch(_Manager(), []) == []


def test_analyze_batch_produces_full_reports_including_risk():
    from rtsp_backend import panel_svc

    class _Manager:
        def __init__(self):
            self._b = _FakeRecognizer(true_batching=True)

        def backend(self, task):
            return self._b

    results = panel_svc.analyze_batch(_Manager(), [_frame(20), _frame(60)])
    assert len(results) == 2
    for r in results:
        assert r["component_model_loaded"] is True
        assert r["components"], "the fake backend detects one MCB per frame"
        # The batch path must run the SAME downstream pipeline as analyze(), not a
        # reduced copy of it.
        assert "report" in r
        assert "risk_assessment" in r["report"]
        assert "topology" in r and "bill_of_materials" in r


# ==========================================================================
# API
# ==========================================================================

def _jpeg(tag: int) -> bytes:
    img = np.full((200, 280, 3), tag, np.uint8)
    for i in range(5):
        cv2.rectangle(img, (15 + i * 52, 70), (55 + i * 52, 140), (40, 40, 45), -1)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_batch_endpoint_returns_one_entry_per_image(client):
    files = [("files", (f"p{i}.jpg", io.BytesIO(_jpeg(60 + i * 10)),
                        "image/jpeg")) for i in range(4)]
    r = client.post("/api/panel/analyze/batch", files=files,
                    params={"batch_size": 2})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["images"] == 4
    assert [x["filename"] for x in body["results"]] == [
        "p0.jpg", "p1.jpg", "p2.jpg", "p3.jpg"]
    for entry in body["results"]:
        assert entry["image"]["width"] == 280
        assert isinstance(entry["components"], list)
    assert body["bbox_format"] == "xyxy_absolute_pixels"
    assert body["ms_per_image"] is not None


def test_batch_endpoint_reports_whether_batching_was_real(client):
    files = [("files", ("a.jpg", io.BytesIO(_jpeg(90)), "image/jpeg"))]
    body = client.post("/api/panel/analyze/batch", files=files).json()
    assert "true_batching" in body
    if not body["true_batching"]:
        assert body["note"] and "sequentially" in body["note"]


def test_batch_endpoint_rejects_one_bad_file_without_failing_the_batch(client):
    files = [
        ("files", ("good.jpg", io.BytesIO(_jpeg(70)), "image/jpeg")),
        ("files", ("bad.txt", io.BytesIO(b"not an image"), "text/plain")),
    ]
    body = client.post("/api/panel/analyze/batch", files=files).json()
    assert body["images"] == 1
    assert len(body["rejected"]) == 1
    assert body["rejected"][0]["filename"] == "bad.txt"


def test_batch_endpoint_fails_when_nothing_decodes(client):
    files = [("files", ("bad.txt", io.BytesIO(b"nope"), "text/plain"))]
    r = client.post("/api/panel/analyze/batch", files=files)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "bad_image"


def test_batch_endpoint_caps_the_request_size(client):
    from rtsp_backend.api import panel_analyze

    files = [("files", (f"x{i}.jpg", io.BytesIO(_jpeg(80)), "image/jpeg"))
             for i in range(panel_analyze.MAX_BATCH_IMAGES + 1)]
    r = client.post("/api/panel/analyze/batch", files=files)
    assert r.status_code == 413
    assert r.json()["error"]["code"] == "batch_too_large"


def test_batch_endpoint_omits_reports_by_default(client):
    files = [("files", ("a.jpg", io.BytesIO(_jpeg(65)), "image/jpeg"))]
    plain = client.post("/api/panel/analyze/batch", files=files).json()
    assert "report" not in plain["results"][0]

    files = [("files", ("a.jpg", io.BytesIO(_jpeg(65)), "image/jpeg"))]
    full = client.post("/api/panel/analyze/batch", files=files,
                       params={"report": True}).json()
    assert "report" in full["results"][0]
    assert "risk" in full["results"][0]["report"]


def test_batch_endpoint_agrees_with_the_single_endpoint(client):
    """Same engine, same thresholds — only throughput differs."""
    payload = _jpeg(75)
    single = client.post(
        "/api/panel/analyze",
        files={"file": ("a.jpg", io.BytesIO(payload), "image/jpeg")},
        params={"persist": False}).json()
    batch = client.post(
        "/api/panel/analyze/batch",
        files=[("files", ("a.jpg", io.BytesIO(payload), "image/jpeg"))]).json()
    assert (batch["results"][0]["component_total"]
            == single["component_total"])
