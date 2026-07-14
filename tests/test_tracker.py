"""Tests for the self-contained ByteTrack-style tracker."""

import numpy as np

from rtsp_backend.ai.tracker import ByteTrack, MultiClassTracker, iou_matrix
from rtsp_backend.ai.base import BBox, Detection


def test_iou_matrix_basic():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[0, 0, 10, 10], [100, 100, 110, 110]], dtype=np.float32)
    m = iou_matrix(a, b)
    assert m.shape == (1, 2)
    assert abs(m[0, 0] - 1.0) < 1e-5   # identical boxes
    assert m[0, 1] == 0.0              # disjoint boxes


def test_stable_id_across_frames():
    """An object moving steadily keeps the same track id."""
    trk = ByteTrack(min_hits=2, max_age=5)
    ids = []
    for step in range(6):
        x = 10 + step * 5
        boxes = np.array([[x, 10, x + 20, 40]], dtype=np.float32)
        scores = np.array([0.9], dtype=np.float32)
        tracks = trk.update(boxes, scores)
        if tracks:
            ids.append(tracks[0].track_id)
    assert len(ids) >= 3
    assert len(set(ids)) == 1  # one stable id the whole time


def test_two_objects_get_distinct_ids():
    trk = ByteTrack(min_hits=1)
    boxes = np.array([[0, 0, 20, 20], [100, 100, 130, 130]], dtype=np.float32)
    scores = np.array([0.9, 0.9], dtype=np.float32)
    tracks = trk.update(boxes, scores)
    tracks = trk.update(boxes, scores)
    assert len({t.track_id for t in tracks}) == 2


def test_track_removed_after_max_age():
    trk = ByteTrack(min_hits=1, max_age=2)
    boxes = np.array([[0, 0, 20, 20]], dtype=np.float32)
    trk.update(boxes, np.array([0.9]))
    assert len(trk.tracks) == 1
    # object disappears; after max_age frames the track is culled
    for _ in range(4):
        trk.update(np.zeros((0, 4), dtype=np.float32), np.zeros((0,)))
    assert len(trk.tracks) == 0


def test_low_confidence_recovery():
    """A track survives a low-confidence frame (ByteTrack second stage)."""
    trk = ByteTrack(min_hits=1, max_age=5, high_thresh=0.5, low_thresh=0.1)
    trk.update(np.array([[10, 10, 30, 30]], dtype=np.float32), np.array([0.9]))
    tid = trk.tracks[0].track_id
    # next frame the detection is weak (occlusion) but should still update
    tracks = trk.update(np.array([[12, 12, 32, 32]], dtype=np.float32),
                        np.array([0.2]))
    assert tracks and tracks[0].track_id == tid
    assert tracks[0].age == 0  # it was updated, not just coasting


def test_multiclass_keeps_ids_separate():
    mt = MultiClassTracker(min_hits=1)
    dets = [
        Detection("person", 0.9, BBox(0, 0, 20, 40)),
        Detection("car", 0.9, BBox(100, 100, 200, 160)),
    ]
    m1 = mt.update(dets)
    m2 = mt.update(dets)
    assert set(m2.keys()) == {0, 1}
    assert m2[0] != m2[1]                 # person and car have different ids
    assert m1 == {} or m2[0] == mt.update(dets)[0]  # id is stable
