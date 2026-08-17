"""
The dataset pipeline: download normalisation, splitting, auto-labelling, export.

What can and cannot be tested here
----------------------------------
The network fetch itself cannot run in CI — it needs a ``ROBOFLOW_API_KEY`` and a
live upstream project — so :func:`training.electrical.download.fetch_roboflow` is
tested for its *failure* contracts (missing key, un-versioned locator), which are
the paths a user actually hits first. Everything downstream of the bytes landing
on disk is tested for real against a Roboflow-shaped export fixture: layout
normalisation, label remapping onto the taxonomy, group-aware splitting, and
bundle export/verification.

The recurring theme in the assertions is honesty. A dataset pipeline has many
opportunities to lose data quietly — an unmapped class folded into a neighbour, an
image dropped for having no label file, a capture group straddling train and val,
an ONNX head whose class count does not match its labels file — and each of those
produces a model that scores well and then fails on site. Each is asserted
against here.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from rtsp_backend.electrical import taxonomy as tax
from training.electrical import autolabel as al
from training.electrical import datasets as ds
from training.electrical import download as dl
from training.electrical import export as ex
from training.electrical import split as sptools


# ==========================================================================
# fixtures
# ==========================================================================

def _img(path: str, w: int = 96, h: int = 72) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, np.full((h, w, 3), 90, np.uint8))


def _labels(path: str, lines) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))


def make_roboflow_export(root: str, names=("contactor", "mcb", "gizmo"),
                         per_split=(4, 2, 2)) -> str:
    """A directory shaped exactly like a Roboflow YOLO export.

    ``<split>/images`` + ``<split>/labels`` + ``data.yaml``, Roboflow's ``valid``
    split name, and its ``_jpg.rf.<hash>`` filename mangling.
    """
    os.makedirs(root, exist_ok=True)
    counter = 0
    for split, n in zip(("train", "valid", "test"), per_split):
        for i in range(n):
            # Distinct panel per image, and a distinct export hash per file —
            # which is what Roboflow actually produces.
            stem = f"panel{counter}_jpg.rf.{counter:016x}{'b' * 8}"
            counter += 1
            _img(os.path.join(root, split, "images", stem + ".jpg"))
            _labels(os.path.join(root, split, "labels", stem + ".txt"),
                    [f"{i % len(names)} 0.5 0.5 0.3 0.4"])
    with open(os.path.join(root, "data.yaml"), "w", encoding="utf-8") as fh:
        fh.write("train: ../train/images\nval: ../valid/images\n"
                 f"nc: {len(names)}\nnames: {list(names)}\n")
    return root


# ==========================================================================
# fetch failure contracts — the paths users hit first
# ==========================================================================

def test_roboflow_without_an_api_key_says_how_to_get_one(tmp_path, monkeypatch):
    monkeypatch.delenv("ROBOFLOW_API_KEY", raising=False)
    with pytest.raises(dl.DownloadError) as exc:
        dl.fetch_roboflow("ws/proj/1", str(tmp_path))
    msg = str(exc.value)
    assert "ROBOFLOW_API_KEY" in msg
    assert "app.roboflow.com" in msg


def test_roboflow_locator_without_a_version_explains_the_fork_remedy(tmp_path):
    """Several real Universe projects have no generated version. Say so."""
    with pytest.raises(dl.DownloadError) as exc:
        dl.fetch_roboflow("control-panel-azure/control-panels", str(tmp_path),
                          api_key="dummy")
    msg = str(exc.value)
    assert "no version" in msg.lower()
    assert "fork" in msg.lower() and "generate" in msg.lower()


def test_kaggle_placeholder_locator_is_refused(tmp_path):
    with pytest.raises(dl.DownloadError) as exc:
        dl.fetch_kaggle("<owner>/<dataset-slug>", str(tmp_path))
    assert "placeholder" in str(exc.value)


def test_url_fetcher_refuses_a_non_url(tmp_path):
    with pytest.raises(dl.DownloadError):
        dl.fetch_url("<some-url>", str(tmp_path))


def test_download_source_reports_unknown_keys_instead_of_raising(tmp_path):
    res = dl.download_source("no_such_source", str(tmp_path))
    assert res.status == "failed"
    assert "unknown source key" in res.reason


def test_download_source_skips_excluded_sources_with_the_reason(tmp_path):
    res = dl.download_source("rf_thermal_panel", str(tmp_path))
    assert res.status == "skipped"
    assert "Thermal" in res.reason or "thermal" in res.reason


def test_download_source_skips_manual_sources(tmp_path):
    res = dl.download_source("vendor_catalogue_crops", str(tmp_path))
    assert res.status == "skipped"
    assert "manual" in res.reason.lower()


# ==========================================================================
# layout normalisation
# ==========================================================================

def test_normalise_converts_roboflow_layout_and_canonicalises_valid(tmp_path):
    src = make_roboflow_export(str(tmp_path / "raw"))
    dst = str(tmp_path / "norm")
    stats = dl.normalise_yolo_layout(src, dst)

    assert stats["images"] == 8
    # Roboflow's 'valid' must land as our 'val', not as a third split name.
    assert set(stats["splits"]) == {"train", "val", "test"}
    assert stats["splits"]["val"] == 2
    for split in ("train", "val", "test"):
        assert os.path.isdir(os.path.join(dst, "images", split))
        assert os.path.isdir(os.path.join(dst, "labels", split))
    assert not os.path.exists(os.path.join(dst, "images", "valid"))


def test_normalise_keeps_unlabelled_images_as_negatives(tmp_path):
    """An image with no label file is a negative example, not a dropped row."""
    src = str(tmp_path / "raw")
    _img(os.path.join(src, "train", "images", "labelled.jpg"))
    _labels(os.path.join(src, "train", "labels", "labelled.txt"),
            ["0 0.5 0.5 0.2 0.2"])
    _img(os.path.join(src, "train", "images", "bare.jpg"))
    dst = str(tmp_path / "norm")
    stats = dl.normalise_yolo_layout(src, dst)

    assert stats["images"] == 2
    assert stats["images_without_labels"] == 1
    # The empty label file must exist — trainers need the pair, and its absence
    # would silently exclude the negative.
    bare = os.path.join(dst, "labels", "train", "bare.txt")
    assert os.path.exists(bare)
    assert open(bare, encoding="utf-8").read().strip() == ""


def test_normalise_handles_a_flat_layout_as_one_train_split(tmp_path):
    src = str(tmp_path / "raw")
    for i in range(3):
        _img(os.path.join(src, "images", f"p{i}.jpg"))
        _labels(os.path.join(src, "labels", f"p{i}.txt"), ["0 0.5 0.5 0.2 0.2"])
    stats = dl.normalise_yolo_layout(src, str(tmp_path / "norm"))
    assert stats["splits"] == {"train": 3}


def test_normalise_refuses_an_unrecognisable_tree(tmp_path):
    src = str(tmp_path / "junk")
    os.makedirs(src)
    (tmp_path / "junk" / "readme.txt").write_text("nothing here")
    with pytest.raises(dl.DownloadError) as exc:
        dl.normalise_yolo_layout(src, str(tmp_path / "out"))
    assert "no recognisable YOLO layout" in str(exc.value)


def test_find_data_yaml_locates_a_nested_export(tmp_path):
    src = make_roboflow_export(str(tmp_path / "wrapper" / "inner"))
    found = dl.find_data_yaml(str(tmp_path / "wrapper"))
    assert found and os.path.dirname(found) == src
    assert ds.read_yolo_names(found) == ["contactor", "mcb", "gizmo"]


def test_archive_extraction_rejects_path_traversal(tmp_path):
    import zipfile
    archive = str(tmp_path / "evil.zip")
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("../escaped.txt", "nope")
    with pytest.raises(dl.DownloadError) as exc:
        dl._extract(archive, str(tmp_path / "out"))
    assert "unsafe path" in str(exc.value)


# ==========================================================================
# splitting — the part that decides whether the metrics mean anything
# ==========================================================================

@pytest.mark.parametrize("filename,expected", [
    ("panel12_jpg.rf.0123456789abcdef.jpg", "panel12"),
    ("panel12_jpg.rf.0123456789abcdef0.jpg", "panel12"),
    ("mcc_board_003.jpg", "mcc_board"),
    ("mcc_board (2).png", "mcc_board"),
    ("cabinet7_rot90.jpg", "cabinet7"),
    ("cabinet7_fliph.jpg", "cabinet7"),
    ("cabinet7_brightness2.jpg", "cabinet7"),
])
def test_group_key_collapses_framings_and_augmentations(filename, expected):
    assert sptools.group_key(filename) == expected


def test_group_key_strips_the_source_prefix():
    assert sptools.group_key("rf_switchgear_potholes_panel4.jpg",
                             ["rf_switchgear_potholes_"]) == "panel4"


def test_group_key_never_returns_empty():
    """A filename that is entirely counter/suffix must still get a key."""
    assert sptools.group_key("003.jpg")
    assert sptools.group_key("_rot90.jpg")


def _multi_group_dataset(root: str, groups: int = 20, per_group: int = 3):
    """``groups`` distinct panels, each photographed ``per_group`` times."""
    idx = tax.class_index()
    for g in range(groups):
        for k in range(per_group):
            stem = f"panel{g}_{k:03d}"
            _img(os.path.join(root, "images", "train", stem + ".jpg"))
            # Rotate through four classes so every split can receive each.
            cls = idx[["mcb", "contactor", "relay", "fuse"][g % 4]]
            _labels(os.path.join(root, "labels", "train", stem + ".txt"),
                    [f"{cls} 0.5 0.5 0.3 0.3", f"{cls} 0.2 0.2 0.1 0.1"])
    return root


def test_split_never_puts_one_capture_group_in_two_splits(tmp_path):
    src = _multi_group_dataset(str(tmp_path / "merged"), groups=24, per_group=3)
    report = sptools.split_dataset(src, str(tmp_path / "final"), seed=7)

    assert report["leaking_groups"] == 0
    # Independently re-derive group membership from the written output.
    seen: dict[str, set] = {}
    for split in ("train", "val", "test"):
        d = os.path.join(str(tmp_path / "final"), "images", split)
        for fn in os.listdir(d):
            seen.setdefault(sptools.group_key(fn), set()).add(split)
    assert seen, "the split produced no images"
    leaked = {g: s for g, s in seen.items() if len(s) > 1}
    assert not leaked, f"capture groups leaked across splits: {leaked}"


def test_split_hits_roughly_80_10_10(tmp_path):
    src = _multi_group_dataset(str(tmp_path / "merged"), groups=40, per_group=2)
    report = sptools.split_dataset(src, str(tmp_path / "final"), seed=3)
    got = report["ratios_achieved"]
    # Group-level assignment cannot hit the ratio exactly; it must be close.
    assert 0.68 <= got["train"] <= 0.90, got
    assert 0.03 <= got["val"] <= 0.22, got
    assert 0.03 <= got["test"] <= 0.22, got
    assert sum(report["images_per_split"].values()) == 80


def test_split_is_deterministic_for_a_seed(tmp_path):
    src = _multi_group_dataset(str(tmp_path / "m"), groups=16, per_group=2)
    a = sptools.split_dataset(src, str(tmp_path / "a"), seed=99)
    b = sptools.split_dataset(src, str(tmp_path / "b"), seed=99)
    assert a["images_per_split"] == b["images_per_split"]
    assert a["instances_per_split"] == b["instances_per_split"]


def test_split_spreads_every_class_across_the_splits(tmp_path):
    """A class absent from val is excluded from mAP — the splitter must try."""
    src = _multi_group_dataset(str(tmp_path / "m"), groups=40, per_group=2)
    report = sptools.split_dataset(src, str(tmp_path / "f"), seed=11)
    assert report["classes_absent_from_val"] == []
    assert report["classes_absent_from_train"] == []


def test_split_names_classes_it_could_not_validate(tmp_path):
    """With one group of a rare class, val cannot get it — say so, loudly."""
    root = str(tmp_path / "m")
    _multi_group_dataset(root, groups=12, per_group=2)
    idx = tax.class_index()
    _img(os.path.join(root, "images", "train", "rare_panel_001.jpg"))
    _labels(os.path.join(root, "labels", "train", "rare_panel_001.txt"),
            [f"{idx['vfd']} 0.5 0.5 0.4 0.4"])
    report = sptools.split_dataset(root, str(tmp_path / "f"), seed=5)
    absent = (set(report["classes_absent_from_val"])
              | set(report["classes_absent_from_test"]))
    assert "vfd" in absent
    assert any("unmeasured" in w or "NONE in" in w for w in report["warnings"])


def test_split_warns_when_there_are_too_few_capture_groups(tmp_path):
    src = _multi_group_dataset(str(tmp_path / "m"), groups=3, per_group=2)
    report = sptools.split_dataset(src, str(tmp_path / "f"), seed=1)
    assert any("capture group" in w for w in report["warnings"])


def test_split_honours_an_explicit_groups_json(tmp_path):
    """Recorded panel ids must beat the filename heuristic."""
    root = str(tmp_path / "m")
    idx = tax.class_index()
    mapping = {}
    for i in range(12):
        # Filenames deliberately share no common stem...
        stem = f"IMG_{1000 + i}"
        _img(os.path.join(root, "images", "train", stem + ".jpg"))
        _labels(os.path.join(root, "labels", "train", stem + ".txt"),
                [f"{idx['mcb']} 0.5 0.5 0.2 0.2"])
        # ...but they are really 3 photographs each of 4 panels.
        mapping[stem + ".jpg"] = f"cabinet{i // 3}"
    gj = str(tmp_path / "groups.json")
    with open(gj, "w", encoding="utf-8") as fh:
        json.dump(mapping, fh)

    report = sptools.split_dataset(root, str(tmp_path / "f"), groups_json=gj,
                                   seed=4)
    assert report["groups"] == 4
    assert report["grouping"] == "explicit groups.json"
    # Every image of one cabinet must share a split.
    placement: dict[str, set] = {}
    for split in ("train", "val", "test"):
        for fn in os.listdir(os.path.join(str(tmp_path / "f"), "images", split)):
            placement.setdefault(mapping[fn], set()).add(split)
    assert all(len(v) == 1 for v in placement.values()), placement


def test_split_detects_leakage_in_the_input(tmp_path):
    root = str(tmp_path / "leaky")
    idx = tax.class_index()
    for split in ("train", "val"):
        stem = f"panel1_{'001' if split == 'train' else '002'}"
        _img(os.path.join(root, "images", split, stem + ".jpg"))
        _labels(os.path.join(root, "labels", split, stem + ".txt"),
                [f"{idx['mcb']} 0.5 0.5 0.2 0.2"])
    grouped = sptools.analyse_groups(root)
    assert grouped["groups"] == 1
    assert grouped["leaking_groups"] == 1


def test_split_rejects_ratios_that_do_not_sum_to_one(tmp_path):
    src = _multi_group_dataset(str(tmp_path / "m"), groups=4, per_group=1)
    with pytest.raises(ValueError):
        sptools.split_dataset(src, str(tmp_path / "f"), ratios=(0.8, 0.8, 0.1))


def test_split_writes_a_dataset_yaml_on_the_canonical_label_space(tmp_path):
    src = _multi_group_dataset(str(tmp_path / "m"), groups=8, per_group=2)
    out = str(tmp_path / "f")
    sptools.split_dataset(src, out, seed=2)
    text = open(os.path.join(out, "dataset.yaml"), encoding="utf-8").read()
    assert f"nc: {len(tax.CLASS_ORDER)}" in text


# ==========================================================================
# auto-annotation
# ==========================================================================

def test_autolabel_skips_honestly_when_no_backend_is_available(tmp_path):
    """No model must mean no labels — never an empty dataset that looks done."""
    images = str(tmp_path / "imgs")
    _img(os.path.join(images, "a.jpg"))
    out = str(tmp_path / "out")
    manifest = al.autolabel_directory(images, out, backends=("null_components",))
    # null_components loads fine but detects nothing: that is an 'empty' verdict,
    # not a fabricated label.
    if manifest["status"] == "labelled":
        assert manifest["boxes_written"] == 0
        assert manifest["by_verdict"].get("empty") == 1
        assert manifest["human_review_required"] is True
    else:
        assert manifest["status"] == "skipped" and manifest["reason"]


def test_autolabel_reports_a_missing_backend_rather_than_inventing_labels(tmp_path):
    images = str(tmp_path / "imgs")
    _img(os.path.join(images, "a.jpg"))
    out = str(tmp_path / "out")
    manifest = al.autolabel_directory(images, out,
                                      backends=("definitely_not_a_backend",))
    assert manifest["status"] == "skipped"
    assert "backend" in manifest["reason"]
    assert not os.path.exists(os.path.join(out, "labels", "train"))


def test_autolabel_rejects_a_review_threshold_above_accept(tmp_path):
    images = str(tmp_path / "imgs")
    _img(os.path.join(images, "a.jpg"))
    with pytest.raises(ValueError):
        al.autolabel_directory(images, str(tmp_path / "o"),
                               accept=0.2, review=0.8)


def test_autolabel_skips_a_missing_directory(tmp_path):
    res = al.autolabel_directory(str(tmp_path / "nope"), str(tmp_path / "o"))
    assert res["status"] == "skipped"


def test_yolo_line_is_normalised_and_clipped():
    line = al._to_yolo_line(3, (-10.0, -10.0, 50.0, 50.0), 100, 100)
    assert line is not None
    cls, cx, cy, w, h = line.split()
    assert cls == "3"
    # The box was clipped to (0,0,50,50), so the centre is (0.25, 0.25).
    assert abs(float(cx) - 0.25) < 1e-6 and abs(float(cy) - 0.25) < 1e-6
    assert all(0.0 <= float(v) <= 1.0 for v in (cx, cy, w, h))


def test_yolo_line_rejects_a_degenerate_box():
    assert al._to_yolo_line(0, (10.0, 10.0, 10.4, 10.4), 100, 100) is None


def test_annotation_instructions_state_the_rules_that_get_broken(tmp_path):
    g = al.annotation_instructions()
    joined = json.dumps(g).lower()
    # The four rules that most reliably ruin an industrial dataset.
    assert "two boxes" in joined                  # contactor + overload
    assert "per pole" in joined                   # terminal strips
    assert tax.UNKNOWN_COMPONENT_ID in json.dumps(g)
    assert "never guess" in joined
    assert g["do_not_label"] and g["quality_control"] and g["workflow"]


# ==========================================================================
# export bundle
# ==========================================================================

def test_write_labels_writes_real_names_not_indices(tmp_path):
    """The exact bug that shipped once: labels.txt full of bare integers."""
    path = ex.write_labels(str(tmp_path))
    lines = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    assert lines == list(tax.CLASS_ORDER)
    assert not any(ln.isdigit() for ln in lines)


def test_labels_and_classes_json_agree_exactly(tmp_path):
    ex.write_labels(str(tmp_path))
    ex.write_classes_json(str(tmp_path))
    labels = [ln.strip() for ln in
              open(tmp_path / "labels.txt", encoding="utf-8") if ln.strip()]
    data = json.load(open(tmp_path / "classes.json", encoding="utf-8"))
    assert labels == data["classes"] == list(tax.CLASS_ORDER)
    assert data["class_count"] == len(tax.CLASS_ORDER)


def test_verify_bundle_rejects_a_numeric_labels_file(tmp_path):
    ex.write_classes_json(str(tmp_path))
    (tmp_path / "labels.txt").write_text("0\n1\n2\n")
    info = ex.verify_bundle(str(tmp_path))
    assert not info["ok"]
    assert any("only integers" in p for p in info["problems"])


def test_verify_bundle_rejects_disagreeing_label_sources(tmp_path):
    ex.write_classes_json(str(tmp_path))
    (tmp_path / "labels.txt").write_text("mcb\ncontactor\n")
    info = ex.verify_bundle(str(tmp_path))
    assert not info["ok"]
    assert any("disagree" in p for p in info["problems"])


def test_verify_bundle_flags_a_reordered_label_space(tmp_path):
    """A permuted label order silently mislabels every single detection.

    It cannot be validated from the bundle alone -- a self-consistent classes.json is
    correct if it truthfully describes the checkpoint's head and catastrophic if it does
    not, and nothing on disk distinguishes those. An arbitrary permutation matches no
    registered profile, so it stays a hard problem.
    """
    scrambled = list(tax.CLASS_ORDER)
    scrambled[0], scrambled[1] = scrambled[1], scrambled[0]
    ex.write_classes_json(str(tmp_path), scrambled)
    ex.write_labels(str(tmp_path), scrambled)
    info = ex.verify_bundle(str(tmp_path))
    assert not info["ok"]
    assert any("nor any registered profile" in p for p in info["problems"])


def test_verify_bundle_accepts_a_registered_profile_label_space(tmp_path):
    """A profile-scoped bundle is legitimate and must not fail verification.

    core8's order is not the taxonomy's and not a prefix of it, but it is safe: the
    runtime reads classes.json in preference to the taxonomy, and the bundle ships it.
    Failing this case failed every profile-scoped export, and a check that fires on
    correct bundles trains people to ignore it.
    """
    from training.electrical import profiles as pf

    names = list(pf.CORE8.classes)
    ex.write_classes_json(str(tmp_path), names)
    ex.write_labels(str(tmp_path), names)
    info = ex.verify_bundle(str(tmp_path))
    assert not any("registered profile" in p for p in info["problems"])
    assert info.get("label_space_profile") == "core8"
    assert "core8" in info["taxonomy_note"]
    assert "Serve it only against this label space" in info["taxonomy_note"]


def test_verify_bundle_accepts_the_core15_profile_label_space(tmp_path):
    from training.electrical import profiles as pf

    names = list(pf.CORE15.classes)
    ex.write_classes_json(str(tmp_path), names)
    ex.write_labels(str(tmp_path), names)
    info = ex.verify_bundle(str(tmp_path))
    assert info.get("label_space_profile") == "core15"


def test_verify_bundle_accepts_an_older_prefix_bundle(tmp_path):
    """CLASS_ORDER is append-only, so a shorter prefix is valid, not broken."""
    prefix = list(tax.CLASS_ORDER[:-1])
    ex.write_classes_json(str(tmp_path), prefix)
    ex.write_labels(str(tmp_path), prefix)
    info = ex.verify_bundle(str(tmp_path))
    assert not any("does not match the taxonomy" in p for p in info["problems"])
    assert "taxonomy_note" in info


def test_verify_bundle_requires_the_label_files(tmp_path):
    info = ex.verify_bundle(str(tmp_path))
    assert not info["ok"]
    assert any("classes.json is missing" in p for p in info["problems"])
    assert any("labels.txt is missing" in p for p in info["problems"])


def test_export_bundle_fails_on_a_missing_checkpoint(tmp_path):
    res = ex.export_bundle(str(tmp_path / "nope.pt"), str(tmp_path / "out"))
    assert res["status"] == "failed"
    assert "not found" in res["reason"]


def test_export_bundle_writes_labels_even_without_ultralytics(tmp_path):
    """A bundle missing ONNX is still useful; a bundle missing labels is not."""
    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not a real checkpoint")
    out = str(tmp_path / "bundle")
    res = ex.export_bundle(str(weights), out, log=lambda m: None)

    assert res["status"] == "exported"
    assert os.path.exists(os.path.join(out, "best.pt"))
    assert os.path.exists(os.path.join(out, "labels.txt"))
    assert os.path.exists(os.path.join(out, "classes.json"))
    card = json.load(open(os.path.join(out, "model_card.json"),
                          encoding="utf-8"))
    assert card["class_count"] == len(tax.CLASS_ORDER)
    assert card["limitations"], "a model card without limitations is marketing"
    # ONNX either exported or reported why not — never silently absent.
    assert res["onnx"]["status"] in ("exported", "skipped", "failed")
    if res["onnx"]["status"] != "exported":
        assert res["onnx"]["reason"]


def test_install_refuses_a_bundle_that_would_mislabel(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "labels.txt").write_text("0\n1\n")
    res = ex.install_bundle(str(bundle), str(tmp_path / "models"))
    assert res["status"] == "refused"
    assert res["problems"]
    assert not os.path.exists(tmp_path / "models" / "labels.txt")


def test_install_copies_a_verified_bundle(tmp_path):
    bundle = str(tmp_path / "bundle")
    ex.write_labels(bundle)
    ex.write_classes_json(bundle)
    install = str(tmp_path / "models" / "components")
    res = ex.install_bundle(bundle, install)
    assert res["status"] == "installed"
    assert os.path.exists(os.path.join(install, "labels.txt"))
    assert os.path.exists(os.path.join(install, "classes.json"))
    assert res["next_step"]


def _ultralytics_run(root: str, epochs: int = 12) -> str:
    """A directory shaped like an Ultralytics training run."""
    os.makedirs(os.path.join(root, "weights"), exist_ok=True)
    header = ["epoch", "train/box_loss", "train/cls_loss", "train/dfl_loss",
              "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)",
              "metrics/mAP50-95(B)", "val/box_loss", "val/cls_loss",
              "val/dfl_loss", "lr/pg0"]
    lines = [",".join(header)]
    for e in range(1, epochs + 1):
        decay = 1.0 / (1 + 0.15 * e)
        lines.append(",".join(str(v) for v in [
            e, 2.4 * decay, 3.1 * decay, 1.5 * decay,
            0.25 + 0.6 * (1 - decay), 0.18 + 0.6 * (1 - decay),
            0.20 + 0.65 * (1 - decay), 0.10 + 0.45 * (1 - decay),
            2.6 * decay, 3.4 * decay, 1.6 * decay, 0.01 * decay]))
    with open(os.path.join(root, "results.csv"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    with open(os.path.join(root, "args.yaml"), "w", encoding="utf-8") as fh:
        fh.write(f"epochs: {epochs}\nimgsz: 960\n")
    weights = os.path.join(root, "weights", "best.pt")
    with open(weights, "w", encoding="utf-8") as fh:
        fh.write("not a real checkpoint")
    return weights


def _evaluation(classes=("mcb", "contactor", "relay")) -> dict:
    cm = {t: {p: (24 if t == p else 2) for p in classes} for t in classes}
    return {
        "status": "evaluated", "map_50": 0.80, "map_50_95": 0.57,
        "overall": {"precision": 0.83, "recall": 0.79, "f1": 0.81},
        "classes": [{"class_id": c, "precision": 0.8, "recall": 0.78,
                     "f1": 0.79, "support": 30} for c in classes],
        "confusion_matrix": cm,
    }


def test_results_csv_is_parsed_into_curves(tmp_path):
    run = str(tmp_path / "run")
    _ultralytics_run(run, epochs=9)
    curves = ex.parse_results_csv(os.path.join(run, "results.csv"))
    assert curves["epochs"] == 9
    assert len(curves["map_50_95"]) == 9
    # Loss must fall and mAP must rise over the run, or the fixture is wrong.
    assert curves["train_box_loss"][0] > curves["train_box_loss"][-1]
    assert curves["map_50_95"][0] < curves["map_50_95"][-1]
    assert curves["learning_rate"]


def test_parse_results_csv_of_a_missing_file_is_empty(tmp_path):
    assert ex.parse_results_csv(str(tmp_path / "nope.csv")) == {}


def test_artifacts_are_collected_from_the_run_directory(tmp_path):
    run = str(tmp_path / "run")
    _ultralytics_run(run)
    out = str(tmp_path / "bundle")
    info = ex.collect_artifacts(run, out)
    assert info["status"] == "collected"
    assert "results.csv" in info["copied"]
    assert "args.yaml" in info["copied"]
    assert os.path.exists(os.path.join(out, "artifacts", "results.csv"))
    # Files Ultralytics did not write are reported missing, not invented.
    assert "confusion_matrix.png" in info["missing"]


def test_collect_artifacts_explains_a_missing_run_directory(tmp_path):
    info = ex.collect_artifacts(str(tmp_path / "nope"), str(tmp_path / "b"))
    assert info["status"] == "skipped"
    assert "--run-dir" in info["reason"]


def test_metrics_json_records_accuracy_and_caveats(tmp_path):
    out = str(tmp_path / "bundle")
    path = ex.write_metrics_json(out, evaluation=_evaluation(),
                                 curves={"epochs": 5, "map_50_95": [0.1] * 5})
    data = json.load(open(path, encoding="utf-8"))
    assert data["headline"]["map_50_95"] == 0.57
    assert data["headline"]["f1"] == 0.81
    assert data["per_class"] and data["confusion_matrix"]
    assert data["training_curves"]["epochs"] == 5
    # The caveats are the point: a headline mAP without them is misleading.
    assert any("absent from the validation split" in c for c in data["caveats"])
    assert any("300-instance" in c for c in data["caveats"])


def test_metrics_json_uses_none_for_unmeasured_not_zero(tmp_path):
    """A metric that was never measured must not read as a bad measurement."""
    path = ex.write_metrics_json(str(tmp_path / "b"))
    data = json.load(open(path, encoding="utf-8"))
    assert data["headline"]["map_50_95"] is None
    assert data["headline"]["f1"] is None
    assert data["per_class"] is None
    assert data["training_curves"] is None
    assert data["runtime"] is None


def _acceptance(passed=True, conf=0.05, constraints_met=True):
    return {
        "best_operating_point": {
            "conf": conf, "map_50": 0.72, "precision": 0.81, "recall": 0.64,
            "fp_per_image": 0.4, "fn_per_image": 1.1,
            "constraints_met": constraints_met,
        },
        "acceptance": {"passed": passed, "target_map_50": 0.70,
                       "achieved_map_50": 0.72,
                       "statement": "meets" if passed else "does NOT meet"},
    }


def test_metrics_json_without_acceptance_says_the_headline_is_not_a_production_claim(
        tmp_path):
    """The overstatement this guards against.

    A bundle whose only accuracy figure comes from the trainer's validator can read as
    a production claim. The trainer scores at a ~0.001 confidence floor with none of
    the gates the API applies, so served accuracy is normally much lower -- and the
    bundle has to say that itself rather than relying on the reader to know.
    """
    path = ex.write_metrics_json(str(tmp_path / "b"), evaluation=_evaluation())
    data = json.load(open(path, encoding="utf-8"))
    assert data["production_acceptance"] is None
    first = data["caveats"][0]
    assert "NO production-path evaluation" in first
    assert "cli accept" in first


def test_metrics_json_with_acceptance_points_the_reader_at_it(tmp_path):
    path = ex.write_metrics_json(str(tmp_path / "b"), evaluation=_evaluation(),
                                 acceptance=_acceptance())
    data = json.load(open(path, encoding="utf-8"))
    assert data["production_acceptance"]["best_operating_point"]["conf"] == 0.05
    first = data["caveats"][0]
    assert "read `production_acceptance` in preference to `headline`" in first
    # The served figures, not the training ones, are the ones quoted.
    assert "0.4 false positives" in first
    assert "1.1" in first


def test_metrics_json_caveat_reports_an_unmet_target(tmp_path):
    path = ex.write_metrics_json(str(tmp_path / "b"),
                                 acceptance=_acceptance(passed=False))
    data = json.load(open(path, encoding="utf-8"))
    assert "does NOT meet" in data["caveats"][0]


def test_metrics_json_caveat_reports_unmet_constraints(tmp_path):
    path = ex.write_metrics_json(str(tmp_path / "b"),
                                 acceptance=_acceptance(constraints_met=False))
    data = json.load(open(path, encoding="utf-8"))
    assert "does NOT satisfy the constraints" in data["caveats"][0]


def test_curves_are_plotted_when_matplotlib_is_available(tmp_path):
    run = str(tmp_path / "run")
    _ultralytics_run(run)
    out = str(tmp_path / "bundle")
    curves = ex.parse_results_csv(os.path.join(run, "results.csv"))
    res = ex.plot_curves(out, curves)
    if res["status"] == "skipped":
        assert "matplotlib" in res["reason"]
        assert "metrics.json" in res["reason"], \
            "a skipped plot must say the numbers are still available"
        return
    assert res["status"] == "plotted"
    assert "loss_curves.png" in res["files"]
    assert "metric_curves.png" in res["files"]
    for name in res["files"]:
        path = os.path.join(out, "artifacts", name)
        assert os.path.getsize(path) > 1000, f"{name} is suspiciously small"


def test_confusion_matrix_is_plotted_over_active_classes_only(tmp_path):
    """A 54x54 grid of mostly zeros is unreadable and says nothing."""
    out = str(tmp_path / "bundle")
    res = ex.plot_confusion_matrix(out, _evaluation()["confusion_matrix"])
    if res["status"] == "skipped":
        assert "matplotlib" in res["reason"]
        return
    assert res["status"] == "plotted"
    assert res["classes_plotted"] == 3, \
        "only classes that appear should be plotted"
    assert os.path.getsize(os.path.join(
        out, "artifacts", "confusion_matrix_normalized.png")) > 1000


def test_confusion_matrix_plot_skips_an_empty_matrix(tmp_path):
    res = ex.plot_confusion_matrix(str(tmp_path / "b"), {})
    assert res["status"] == "skipped"
    res2 = ex.plot_confusion_matrix(str(tmp_path / "b"), None)
    assert res2["status"] == "skipped"


def test_export_bundle_collects_evidence_end_to_end(tmp_path):
    run = str(tmp_path / "run")
    weights = _ultralytics_run(run)
    out = str(tmp_path / "bundle")
    res = ex.export_bundle(weights, out, evaluation=_evaluation(),
                           log=lambda m: None)

    assert res["status"] == "exported"
    # The run directory must be inferred from .../weights/best.pt.
    assert res["artifacts"]["run_dir"] == run
    assert "results.csv" in res["artifacts"]["copied"]
    assert os.path.exists(os.path.join(out, "metrics.json"))
    metrics = json.load(open(os.path.join(out, "metrics.json"),
                             encoding="utf-8"))
    assert metrics["headline"]["map_50_95"] == 0.57
    assert metrics["training_curves"]["epochs"] == 12
    assert metrics["provenance"]["run_dir"] == run


def test_export_warns_when_no_accuracy_evidence_is_supplied(tmp_path):
    """A deployed model with no accuracy record cannot be audited later."""
    run = str(tmp_path / "run")
    weights = _ultralytics_run(run)
    res = ex.export_bundle(weights, str(tmp_path / "bundle"),
                           log=lambda m: None)
    assert any("no measured accuracy" in w for w in res["warnings"])
    assert any("cli eval" in w for w in res["warnings"])


def test_install_carries_the_artifacts_with_the_weights(tmp_path):
    run = str(tmp_path / "run")
    weights = _ultralytics_run(run)
    bundle = str(tmp_path / "bundle")
    ex.export_bundle(weights, bundle, evaluation=_evaluation(),
                     log=lambda m: None)
    install = str(tmp_path / "models" / "components")
    res = ex.install_bundle(bundle, install)

    assert res["status"] == "installed"
    assert os.path.exists(os.path.join(install, "metrics.json"))
    assert os.path.isdir(os.path.join(install, "artifacts"))
    assert os.path.exists(os.path.join(install, "artifacts", "results.csv"))


def test_tensorrt_instructions_explain_why_they_are_not_automated():
    info = ex.tensorrt_instructions()
    assert "not portable" in info["why_not_automated"]
    assert "trtexec" in info["build_from_onnx"]
    assert "engine" in info["build_from_pt"]
    assert info["verify"] and info["prerequisites"]


# ==========================================================================
# end to end: a Roboflow-shaped export becomes a trainable, split dataset
# ==========================================================================

def test_normalise_remap_split_and_report_run_end_to_end(tmp_path):
    raw = make_roboflow_export(str(tmp_path / "raw"),
                               names=("contactor", "mcb", "gizmo"),
                               per_split=(12, 4, 4))
    norm = str(tmp_path / "norm")
    dl.normalise_yolo_layout(raw, norm)

    # Remap onto the taxonomy. 'gizmo' has no home and must be dropped with a
    # count rather than folded into a neighbouring class.
    remapped = str(tmp_path / "remapped")
    stats = ds.remap_yolo_dataset(norm, remapped,
                                  ["contactor", "mcb", "gizmo"],
                                  prefix="src_")
    assert stats["unmapped_source_classes"] == ["gizmo"]
    assert stats["instances_dropped"] > 0
    assert set(stats["per_class"]) == {"contactor", "mcb"}

    final = str(tmp_path / "final")
    report = sptools.split_dataset(remapped, final,
                                   source_prefixes=["src_"], seed=13)
    assert report["leaking_groups"] == 0
    assert sum(report["images_per_split"].values()) == 20

    analysis = ds.analyse_dataset(final)
    assert analysis["instances"] == stats["instances_kept"]

    gap = ds.requirements_report(analysis, priority_only=True)
    # A 20-image dataset must be reported as nowhere near production-ready.
    assert gap["ready_classes"] == []
    assert gap["annotations_required"] > 0
    assert "vfd" in gap["missing_classes"]
    # And every short class comes with somewhere to go and photograph it.
    assert all(r["where_to_find_it"] for r in gap["what_to_collect"])
