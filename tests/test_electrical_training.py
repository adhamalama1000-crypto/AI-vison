"""
Dataset synthesis, label-space unification and the training driver's honesty.

The training stack cannot be *executed* in CI (no GPU, no ultralytics, no
network access to dataset hosts), so what is tested here is everything that can
be: the synthetic generator produces images with correct, in-prior ground truth;
dataset remapping puts heterogeneous labels onto one canonical space and drops —
loudly — what it cannot map; and the training driver reports unavailable
architectures as ``skipped`` with a reason instead of silently substituting one.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from rtsp_backend.electrical import postprocess as pp
from rtsp_backend.electrical import taxonomy as tax
from training.electrical import datasets as ds
from training.electrical import synthetic as syn
from training.electrical import train as tr


# ==========================================================================
# synthetic generation
# ==========================================================================

def test_synthesise_panel_produces_image_and_boxes():
    gen = syn.synthesise_panel(800, 600, seed=1, nuisance=False)
    assert gen.image.shape == (600, 800, 3)
    assert len(gen.instances) >= 5
    assert gen.meta["source"] == "procedural"
    for inst in gen.instances:
        x1, y1, x2, y2 = inst.box
        assert 0 <= x1 < x2 <= 800
        assert 0 <= y1 < y2 <= 600
        assert inst.class_id in tax.SPECS


def test_synthesis_is_deterministic():
    a = syn.synthesise_panel(640, 480, seed=99, nuisance=False)
    b = syn.synthesise_panel(640, 480, seed=99, nuisance=False)
    assert np.array_equal(a.image, b.image)
    assert [i.box for i in a.instances] == [i.box for i in b.instances]


def test_different_seeds_differ():
    a = syn.synthesise_panel(640, 480, seed=1, nuisance=False)
    b = syn.synthesise_panel(640, 480, seed=2, nuisance=False)
    assert not np.array_equal(a.image, b.image)


def test_generated_boxes_satisfy_the_plausibility_gate():
    """The generator and the gate must agree on geometry by construction.

    If they disagree, a gate failure on synthetic data is ambiguous — it could be
    the gate or the generator. Pinning this makes the validation harness sound.
    """
    cfg = pp.GateConfig()
    checked = 0
    for seed in range(6):
        gen = syn.synthesise_panel(1024, 768, seed=seed, nuisance=False)
        area = float(gen.image.shape[0] * gen.image.shape[1])
        for inst in gen.instances:
            cand = pp.Candidate(inst.class_id, 0.9, inst.box)
            ok, reason = pp.plausible(cand, area, cfg)
            assert ok, f"{inst.class_id} box {inst.box} rejected: {reason}"
            checked += 1
    assert checked > 100


def test_generated_panel_survives_the_full_cascade():
    gen = syn.synthesise_panel(1024, 768, seed=5, nuisance=False)
    cands = [pp.Candidate(i.class_id, 0.9, i.box) for i in gen.instances]
    res = pp.run(cands, gen.image.shape[:2])
    # allow for legitimate cross-class dedupe of overlapping confusable devices
    assert len(res.accepted) >= int(0.9 * len(cands))
    assert res.diagnostics.dropped.get("implausible_aspect_ratio", 0) == 0
    assert res.diagnostics.dropped.get("implausible_too_large", 0) == 0


def test_generated_panel_contains_no_wires():
    """The validation harness depends on this: ground truth is zero wires."""
    gen = syn.synthesise_panel(800, 600, seed=3, nuisance=False)
    assert all(i.class_id != "wire" for i in gen.instances)
    assert "wire" not in tax.CLASS_ORDER


def test_render_device_produces_structure_not_a_flat_block():
    img = syn.render_device("contactor", 90, 110, __import__("random").Random(0))
    assert img.shape == (110, 90, 3)
    # terminals, screws and a label plate mean many distinct intensities
    assert len(np.unique(img.reshape(-1, 3), axis=0)) > 20


@pytest.mark.parametrize("cid", syn.PROCEDURAL_CLASSES)
def test_every_procedural_class_renders(cid):
    import random
    img = syn.render_device(cid, 80, 90, random.Random(7))
    assert img.shape == (90, 80, 3)
    assert img.dtype == np.uint8


def test_nuisance_factors_change_the_image_but_keep_boxes_valid():
    import random
    rng = random.Random(4)
    gen = syn.synthesise_panel(800, 600, seed=4, nuisance=False)
    lit = syn.apply_lighting(gen.image, rng)
    assert not np.array_equal(lit, gen.image)
    assert lit.dtype == np.uint8

    warped, boxes = syn.apply_perspective(gen.image, gen.instances, rng)
    assert warped.shape == gen.image.shape
    for b in boxes:
        x1, y1, x2, y2 = b.box
        assert 0 <= x1 < x2 <= 800 and 0 <= y1 < y2 <= 600


def test_all_nuisance_functions_return_valid_images():
    import random
    rng = random.Random(11)
    img = syn.synthesise_panel(400, 300, seed=11, nuisance=False).image
    for fn in (syn.apply_lighting, syn.apply_shadow, syn.apply_reflection,
               syn.apply_dust, syn.apply_blur_noise, syn.apply_occlusion):
        out = fn(img.copy(), rng)
        assert out.shape == img.shape and out.dtype == np.uint8


def test_yolo_lines_are_normalised():
    gen = syn.synthesise_panel(800, 600, seed=6, nuisance=False)
    lines = syn.to_yolo_lines(gen.instances, 800, 600)
    assert len(lines) == len(gen.instances)
    idx = tax.class_index()
    for ln in lines:
        parts = ln.split()
        assert len(parts) == 5
        assert 0 <= int(parts[0]) < len(idx)
        for v in parts[1:]:
            assert 0.0 <= float(v) <= 1.0


def test_write_dataset_creates_a_valid_yolo_tree(tmp_path):
    out = str(tmp_path / "synth")
    manifest = syn.write_dataset(out, n_train=3, n_val=2, width=480, height=360,
                                seed=5)
    assert manifest["images"] == {"train": 3, "val": 2}
    assert manifest["instance_total"] > 0
    assert manifest["source"] == "procedural"
    assert "PROCEDURAL DATA ONLY" in manifest["warning"]

    for split, n in (("train", 3), ("val", 2)):
        imgs = os.listdir(os.path.join(out, "images", split))
        lbls = os.listdir(os.path.join(out, "labels", split))
        assert len(imgs) == n and len(lbls) == n
        for f in imgs:
            assert cv2.imread(os.path.join(out, "images", split, f)) is not None

    assert os.path.exists(os.path.join(out, "dataset.yaml"))
    with open(os.path.join(out, "classes.json"), encoding="utf-8") as fh:
        assert json.load(fh)["classes"] == list(tax.CLASS_ORDER)


def test_crop_library_indexing_resolves_alias_directories(tmp_path):
    root = tmp_path / "crops"
    for name in ("contactor", "MCCB", "magnetic contactor", "not_a_component"):
        d = root / name
        d.mkdir(parents=True)
        cv2.imwrite(str(d / "a.png"), np.full((40, 30, 3), 100, np.uint8))
    lib = syn.load_crop_library(str(root))
    assert "contactor" in lib and "mccb" in lib
    assert len(lib["contactor"]) == 2          # both alias dirs merged
    assert "not_a_component" not in lib


def test_compose_from_crops_uses_the_real_crops(tmp_path):
    root = tmp_path / "crops"
    d = root / "contactor"
    d.mkdir(parents=True)
    marker = np.zeros((60, 50, 3), np.uint8)
    marker[:, :, 1] = 231                       # a colour nothing else uses
    cv2.imwrite(str(d / "c.png"), marker)

    lib = syn.load_crop_library(str(root))
    gen = syn.compose_from_crops(lib, 640, 480, seed=2, nuisance=False)
    assert gen.meta["source"] == "composed_real_crops"
    assert gen.instances
    assert all(i.class_id == "contactor" for i in gen.instances)
    assert int((gen.image[:, :, 1] == 231).sum()) > 100


def test_compose_from_crops_rejects_an_empty_library():
    with pytest.raises(ValueError, match="crop library is empty"):
        syn.compose_from_crops({}, 320, 240, seed=1)


def test_write_dataset_with_crops_is_marked_as_real_sourced(tmp_path):
    root = tmp_path / "crops"
    d = root / "mcb"
    d.mkdir(parents=True)
    cv2.imwrite(str(d / "m.png"), np.full((60, 20, 3), 80, np.uint8))
    out = str(tmp_path / "ds")
    manifest = syn.write_dataset(out, n_train=2, n_val=1, width=480, height=360,
                                crop_library=str(root), seed=3)
    assert manifest["source"] == "composed_real_crops"
    assert manifest["warning"] is None


# ==========================================================================
# dataset sources + label unification
# ==========================================================================

def test_source_registry_is_well_formed():
    assert ds.SOURCES
    for s in ds.SOURCES:
        assert s.key and s.name and s.licence and s.notes
        assert s.kind in ("roboflow", "kaggle", "url", "github", "openimages",
                          "manual")
        for cid in s.provides:
            assert cid in tax.SPECS, f"{s.key} claims unknown class {cid}"
        for src_label, cid in s.label_map.items():
            assert cid in tax.SPECS, f"{s.key} maps {src_label} to unknown {cid}"


def test_every_registry_kind_has_a_fetcher():
    """A registry entry whose kind nothing can fetch is a dead end."""
    from training.electrical import download as dl

    fetchable = {"roboflow": dl.fetch_roboflow, "kaggle": dl.fetch_kaggle,
                 "url": dl.fetch_url, "github": dl.fetch_github,
                 "openimages": dl.fetch_openimages}
    for s in ds.SOURCES:
        if s.kind == "manual":
            continue
        assert s.kind in fetchable, f"{s.key}: no fetcher for kind '{s.kind}'"


def test_a_placeholder_locator_source_promises_no_classes():
    """A template entry must not inflate the coverage forecast.

    `github_dataset_template` and `kaggle_electrical_components` exist so the
    fetchers are reachable, not because a verified dataset sits behind them. If
    either claimed classes in `provides`, plan() would report coverage that does
    not exist.
    """
    for s in ds.SOURCES:
        if s.kind != "manual" and "<" in s.locator:
            assert s.provides == (), \
                (f"{s.key} has a placeholder locator but claims to provide "
                 f"{s.provides}")


def test_open_images_contributes_negatives_not_positives():
    """Open Images has no industrial electrical classes — it must claim none."""
    oi = ds.SOURCE_INDEX["openimages_hard_negatives"]
    assert oi.kind == "openimages"
    assert oi.provides == ()
    assert oi.class_counts == {}
    assert "NEGATIVE" in oi.notes.upper()
    # And it must not appear in any coverage forecast.
    p = ds.plan()
    assert "openimages_hard_negatives" not in {
        row["key"] for row in p["sources"] if row["provides"]}


def test_plan_separates_downloadable_from_manual_coverage():
    p = ds.plan()
    assert 0.0 < p["downloadable_coverage_fraction"] < 1.0, \
        "claiming full downloadable coverage of this taxonomy would be dishonest"
    assert p["classes_needing_manual_capture"]
    assert "will NOT produce a production model" in p["verdict"]


def test_plan_can_be_restricted_to_one_source():
    p = ds.plan(["rf_control_panels_azure"])
    assert len(p["sources"]) == 1
    assert "mcb" in p["classes_from_downloadable_sources"]


def test_every_roboflow_source_has_a_concrete_locator():
    """A '<workspace>/<project>' placeholder is a fake citation, not a source."""
    for s in ds.SOURCES:
        if s.kind != "roboflow":
            continue
        assert "<" not in s.locator, \
            f"{s.key}: locator {s.locator!r} is a placeholder"
        assert s.verified, f"{s.key}: locator was never verified upstream"
        # 'workspace/project' (no version) is legitimate but must be explained,
        # because it cannot be downloaded until somebody forks and generates one.
        if len(s.locator.split("/")) < 3:
            assert "GENERATED VERSION" in s.notes or "version" in s.notes.lower()


def test_excluded_sources_state_why_and_stay_out_of_the_forecast():
    excluded = [s for s in ds.SOURCES if not s.usable]
    assert excluded, "the registry should record the traps it found"
    for s in excluded:
        assert s.excluded_reason and len(s.excluded_reason) > 40
    p = ds.plan()
    keys = {row["key"] for row in p["sources"]}
    for s in excluded:
        assert s.key not in keys
        assert s.key in {row["key"] for row in p["excluded_sources"]}


def test_plan_forecast_is_bounded_by_measured_counts():
    """The forecast must come from observed counts, not from `provides` lists."""
    p = ds.plan()
    forecast = p["forecast_instances_per_class"]
    assert forecast, "no per-class counts were recorded for any source"
    measured = {}
    for s in ds.SOURCES:
        if not s.usable or s.kind == "manual":
            continue
        for cid, n in s.class_counts.items():
            measured[cid] = measured.get(cid, 0) + n
    assert forecast == {k: v for k, v in
                        sorted(measured.items(), key=lambda kv: -kv[1])}
    # The honest headline: public data does not make this taxonomy reliable.
    assert len(p["forecast_reliable_classes"]) < len(tax.CLASS_ORDER) / 2


def test_priority_classes_without_public_data_are_named():
    """The brief asks exactly which classes are missing — so name them."""
    p = ds.plan()
    missing = p["priority_classes_with_no_public_instances"]
    assert missing, "some priority classes genuinely have no public instances"
    for cid in missing:
        assert cid in tax.SPECS
        assert cid in ds.PRIORITY_CLASSES


def test_requirements_report_from_zero_costs_the_whole_taxonomy():
    r = ds.requirements_report(None, priority_only=True)
    assert r["ready_classes"] == []
    assert set(r["missing_classes"]) == set(ds.PRIORITY_CLASSES)
    assert r["annotations_required"] == (
        ds.MIN_INSTANCES_RELIABLE * len(ds.PRIORITY_CLASSES))
    assert r["images_required"] > 0
    # Every short class must come with somewhere to go and find it.
    assert len(r["what_to_collect"]) == len(ds.PRIORITY_CLASSES)
    for row in r["what_to_collect"]:
        assert row["where_to_find_it"]


def test_requirements_report_credits_data_that_exists(tmp_path):
    root = str(tmp_path / "d")
    # 2 splits x 2 images x 100 boxes of class index 0 ('mcb') = 400 instances
    _make_yolo_dataset(root, names_count=1, per_image=100)
    report = ds.requirements_report(ds.analyse_dataset(root))
    mcb = next(r for r in report["per_class"] if r["class_id"] == "mcb")
    assert mcb["have_annotations"] == 400
    assert mcb["need_annotations"] == 0
    assert mcb["status"] == "ready"
    assert "mcb" in report["ready_classes"]
    assert "mcb" not in report["missing_classes"]


def test_priority_classes_are_all_real_taxonomy_classes():
    for cid in ds.PRIORITY_CLASSES:
        assert cid in tax.SPECS, cid


def test_build_index_map_resolves_and_reports_unmappable():
    idx_map, unmapped = ds.build_index_map(
        ["breaker", "magnetic contactor", "gibberish"],
        {"breaker": "mcb"})
    canon = tax.class_index()
    assert idx_map[0] == canon["mcb"]
    assert idx_map[1] == canon["contactor"]
    assert 2 not in idx_map
    assert unmapped == ["gibberish"]


def _make_yolo_dataset(root, names_count: int, per_image: int = 2):
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
        for i in range(2):
            cv2.imwrite(os.path.join(root, "images", split, f"{i}.jpg"),
                        np.zeros((100, 100, 3), np.uint8))
            with open(os.path.join(root, "labels", split, f"{i}.txt"), "w") as fh:
                for k in range(per_image):
                    fh.write(f"{k % names_count} 0.5 0.5 0.2 0.2\n")


def test_remap_rewrites_class_indices(tmp_path):
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    _make_yolo_dataset(src, names_count=2)
    stats = ds.remap_yolo_dataset(src, dst, ["breaker", "magnetic contactor"],
                                  {"breaker": "mcb"})
    assert stats["images"] == 4
    assert stats["instances_kept"] == 8
    assert stats["instances_dropped"] == 0
    assert stats["per_class"] == {"mcb": 4, "contactor": 4}

    canon = tax.class_index()
    with open(os.path.join(dst, "labels", "train", "0.txt")) as fh:
        ids = {int(ln.split()[0]) for ln in fh if ln.strip()}
    assert ids == {canon["mcb"], canon["contactor"]}
    assert os.path.exists(os.path.join(dst, "dataset.yaml"))


def test_remap_drops_unmappable_classes_with_a_count(tmp_path):
    src = str(tmp_path / "src")
    dst = str(tmp_path / "dst")
    _make_yolo_dataset(src, names_count=2)
    stats = ds.remap_yolo_dataset(src, dst, ["contactor", "zzz_unknown_thing"])
    assert stats["instances_kept"] == 4
    assert stats["instances_dropped"] == 4
    assert stats["dropped_by_class"] == {"zzz_unknown_thing": 4}
    assert stats["unmapped_source_classes"] == ["zzz_unknown_thing"]


def test_merge_prefixes_filenames_to_avoid_collisions(tmp_path):
    a, b = str(tmp_path / "a"), str(tmp_path / "b")
    _make_yolo_dataset(a, names_count=1)
    _make_yolo_dataset(b, names_count=1)
    # both datasets use identical file names (0.jpg, 1.jpg)
    for root in (a, b):
        ds.remap_yolo_dataset(root, root + "_r", ["contactor"])
    dst = str(tmp_path / "merged")
    totals = ds.merge([a + "_r", b + "_r"], dst)
    assert totals["images"] == 8            # nothing overwritten
    assert len(os.listdir(os.path.join(dst, "images", "train"))) == 4


def test_dataset_yaml_pins_the_canonical_label_space(tmp_path):
    root = str(tmp_path / "x")
    path = ds.write_dataset_yaml(root)
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    assert f"nc: {len(tax.CLASS_ORDER)}" in text
    assert f"0: {tax.CLASS_ORDER[0]}" in text


def test_analyse_and_coverage_report(tmp_path):
    out = str(tmp_path / "synth")
    syn.write_dataset(out, n_train=4, n_val=2, width=480, height=360, seed=8)
    analysis = ds.analyse_dataset(out)
    assert analysis["images"] == 6
    assert analysis["instances"] > 0
    assert analysis["per_class"]

    cov = ds.coverage_report(analysis)
    total = sum(len(cov[k]) for k in ("reliable", "weak", "untrainable", "absent"))
    assert total == len(tax.CLASS_ORDER)
    assert cov["absent"], "a 6-image dataset cannot cover the whole taxonomy"
    assert "reliable" in cov["summary"]


def test_custom_collection_plan_is_actionable():
    plan = ds.custom_collection_plan()
    assert plan["capture_protocol"] and plan["labelling_rules"]
    assert "PANEL" in plan["split_policy"]
    # the two rules that most often get labelling wrong must be stated
    joined = " ".join(plan["labelling_rules"]).lower()
    assert "two boxes" in joined          # contactor + overload are separate
    assert "per pole" in joined           # terminal strips, not poles


# ==========================================================================
# training driver honesty
# ==========================================================================

def test_supported_archs_cover_the_requested_families():
    assert any(a.startswith("yolo11") for a in tr.SUPPORTED_ARCHS)
    assert any(a.startswith("yolov8") for a in tr.SUPPORTED_ARCHS)
    assert any(a.startswith("rtdetr") for a in tr.SUPPORTED_ARCHS)
    assert any(a.startswith("yolo12") for a in tr.SUPPORTED_ARCHS)


def test_train_reports_skipped_without_ultralytics(tmp_path):
    cfg = tr.TrainConfig(data=str(tmp_path / "nope.yaml"), epochs=1)
    res = tr.train(cfg)
    assert res.status in ("skipped", "failed")
    assert res.reason
    if not tr.ultralytics_available()[0]:
        assert "ultralytics" in res.reason


def test_train_never_substitutes_an_unknown_arch(tmp_path):
    cfg = tr.TrainConfig(data=str(tmp_path / "d.yaml"), arch="not_a_model")
    res = tr.train(cfg)
    assert res.status == "skipped"
    assert res.weights is None


def test_train_config_uses_panel_appropriate_augmentation():
    cfg = tr.TrainConfig(data="d.yaml")
    kw = cfg.to_kwargs()
    # panels are gravity-oriented and nameplates are directional: mirroring is
    # actively harmful, unlike in generic object detection
    assert kw["fliplr"] == 0.0
    assert kw["flipud"] == 0.0
    assert kw["imgsz"] >= 960          # modular devices are small
    assert kw["close_mosaic"] > 0      # restore layout context at the end
    assert kw["hsv_v"] > 0.3           # lighting varies enormously in the field


def test_classes_json_written_beside_an_export(tmp_path):
    tr._write_classes_json(str(tmp_path))
    with open(tmp_path / "classes.json", encoding="utf-8") as fh:
        assert json.load(fh)["classes"] == list(tax.CLASS_ORDER)
    # labels.txt lands beside it, holding real names rather than bare indices —
    # the failure that once made every detection come back labelled "0".
    labels = [ln.strip() for ln in
              open(tmp_path / "labels.txt", encoding="utf-8") if ln.strip()]
    assert labels == list(tax.CLASS_ORDER)


def test_only_one_writer_produces_classes_json(tmp_path):
    """train and export must not drift into writing different files."""
    from training.electrical import export as ex

    a, b = tmp_path / "a", tmp_path / "b"
    tr._write_classes_json(str(a))
    ex.write_classes_json(str(b))
    assert (json.load(open(a / "classes.json", encoding="utf-8"))
            == json.load(open(b / "classes.json", encoding="utf-8")))


def test_load_ground_truth_reads_absolute_pixel_boxes(tmp_path):
    out = str(tmp_path / "synth")
    syn.write_dataset(out, n_train=1, n_val=2, width=480, height=360, seed=9)
    gts = tr.load_ground_truth(out, "val")
    assert gts
    for g in gts:
        x1, y1, x2, y2 = g["box"]
        assert 0 <= x1 < x2 <= 481 and 0 <= y1 < y2 <= 361
        assert g["class_id"] in tax.SPECS


def test_evaluate_backend_skips_cleanly_without_ground_truth(tmp_path):
    rep = tr.evaluate_backend("industrial_onnx", str(tmp_path))
    assert rep["status"] == "skipped"
    assert "no ground truth" in rep["reason"]


def test_evaluate_backend_skips_cleanly_without_weights(tmp_path):
    out = str(tmp_path / "synth")
    syn.write_dataset(out, n_train=1, n_val=1, width=320, height=240, seed=2)
    rep = tr.evaluate_backend("industrial_onnx", out,
                              params={"models_dir": str(tmp_path / "empty")})
    assert rep["status"] == "skipped"
    assert "models/components" in rep["reason"]
    assert "nothing is fabricated" in rep["reason"].lower()


def test_evaluate_disabled_backend_produces_a_real_report(tmp_path):
    """End-to-end metric path: a backend that detects nothing scores zero."""
    out = str(tmp_path / "synth")
    syn.write_dataset(out, n_train=1, n_val=2, width=480, height=360, seed=7)
    rep = tr.evaluate_backend("industrial_disabled", out)
    assert rep["status"] == "evaluated"
    assert rep["overall"]["tp"] == 0
    assert rep["overall"]["recall"] == 0.0
    assert rep["false_negative_analysis"]["total"] > 0
    assert "confusion_matrix" in rep


def test_benchmark_reports_skips_without_crashing(tmp_path):
    out = str(tmp_path / "synth")
    syn.write_dataset(out, n_train=1, n_val=1, width=320, height=240, seed=4)
    res = tr.benchmark(os.path.join(out, "dataset.yaml"), out,
                       archs=["yolo11n"], epochs=1)
    assert "training" in res and "comparison" in res
    assert res["training"]["yolo11n"]["status"] in ("skipped", "failed", "trained")
    assert isinstance(res["table"], str)
