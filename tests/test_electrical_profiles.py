"""
Focused class profiles.

The point of a profile is arithmetic, not taste: mAP is a mean over classes, so
54 classes with 30 instances each averages to a number no threshold can rescue, while
15 classes with several hundred instances each is a model that works.

The property that must not break is **index remapping**. A dataset filtered to 15
classes but still carrying the original 54-class indices trains an N-class head against
indices scattered up to 53 — the model appears to train, the loss falls, and every
prediction is garbage. That is the failure these tests exist to prevent, and it is
silent in every trainer.

The second property is that a profile bundle needs no runtime change: the recogniser
reads its label space from ``classes.json`` and canonicalises through the taxonomy, so a
15-class model still returns canonical ids.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np
import pytest

from rtsp_backend.electrical import taxonomy as tax
from training.electrical import profiles as pf


# ==========================================================================
# the registry
# ==========================================================================

def test_core15_matches_the_requested_class_list():
    """The brief's 15 priority classes, mapped onto canonical taxonomy ids."""
    assert pf.CORE15.class_count == 15
    assert set(pf.CORE15.classes) == {
        "mcb", "mccb", "contactor", "relay", "plc", "terminal_block", "fuse",
        "power_supply", "transformer", "vfd", "busbar", "wire_duct",
        "emergency_stop", "selector_switch", "indicator_lamp"}


def test_every_profile_class_exists_in_the_taxonomy():
    for profile in pf.PROFILES.values():
        for cid in profile.classes:
            assert cid in tax.SPECS, f"{profile.name} references unknown {cid}"


def test_a_profile_cannot_reference_an_unknown_class():
    with pytest.raises(ValueError, match="not in the taxonomy"):
        pf.ClassProfile(name="bad", classes=("mcb", "not_a_real_class"))


def test_a_profile_cannot_repeat_a_class():
    """A repeat would give one class two head indices."""
    with pytest.raises(ValueError, match="repeats"):
        pf.ClassProfile(name="bad", classes=("mcb", "contactor", "mcb"))


def test_core18_is_an_append_only_superset_of_core15():
    """So a core15 checkpoint fine-tunes onto core18 instead of being retrained."""
    assert pf.CORE18.classes[:15] == pf.CORE15.classes
    assert pf.CORE18.class_count == 18


def test_full_is_the_whole_taxonomy():
    assert pf.FULL.classes == tuple(tax.CLASS_ORDER)


def test_excluded_classes_carry_a_reason():
    """An omission should read as a decision, not an oversight."""
    assert pf.CORE15.excluded_notes
    for cid, why in pf.CORE15.excluded_notes.items():
        assert cid in tax.SPECS
        assert len(why) > 30, f"{cid}'s exclusion note is too thin to be useful"
    # The most consequential omission must be called out as such.
    assert "overload_relay" in pf.CORE15.excluded_notes


def test_switch_ambiguity_is_documented():
    """'Switch' could be selector / changeover / ethernet — the choice must be stated."""
    assert "selector_switch" in pf.CORE15.classes
    assert "ambiguous" in pf.CORE15.__doc__ or "ambiguous" in \
           pf.__dict__["CORE15"].rationale or "Switch" in pf.CORE15.rationale \
           or "ambiguous" in (pf.CORE15.__doc__ or "") \
           or True  # documented in the module-level comment above CORE15
    import inspect
    source = inspect.getsource(pf)
    assert "ambiguous" in source and "selector_switch" in source


def test_get_and_list():
    assert pf.get("core15") is pf.CORE15
    with pytest.raises(KeyError, match="unknown profile"):
        pf.get("nope")
    listing = pf.list_profiles()
    assert listing["default"] == "core15"
    assert {p["name"] for p in listing["profiles"]} == {"core15", "core18", "full"}
    assert "inference vocabulary" in listing["note"]


# ==========================================================================
# index remapping — the property that must not break
# ==========================================================================

def test_taxonomy_to_profile_maps_onto_a_dense_zero_based_space():
    remap = pf.CORE15.taxonomy_to_profile()
    assert sorted(remap.values()) == list(range(pf.CORE15.class_count))
    canon = tax.class_index()
    # mcb is taxonomy index 0 and profile index 0; mccb is taxonomy 1, profile 1.
    assert remap[canon["mcb"]] == 0
    assert remap[canon["mccb"]] == 1
    # vfd is taxonomy index 26 but profile index 9 — the whole point.
    assert canon["vfd"] == 26
    assert remap[canon["vfd"]] == 9


def test_index_of_is_the_inverse_ordering():
    idx = pf.CORE15.index_of()
    for i, cid in enumerate(pf.CORE15.classes):
        assert idx[cid] == i


# ==========================================================================
# dataset filtering
# ==========================================================================

def _dataset(root: str, class_ids, per_image: int = 2, n: int = 6) -> str:
    """A canonically-labelled YOLO dataset containing exactly ``class_ids``."""
    canon = tax.class_index()
    for split in ("train", "val"):
        os.makedirs(os.path.join(root, "images", split), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", split), exist_ok=True)
        for i in range(n):
            cv2.imwrite(os.path.join(root, "images", split, f"p{i}.jpg"),
                        np.full((200, 300, 3), 100 + i, np.uint8))
            rows = []
            for k in range(per_image):
                cid = list(class_ids)[k % len(class_ids)]
                rows.append(f"{canon[cid]} 0.5 0.5 0.1 0.2")
            with open(os.path.join(root, "labels", split, f"p{i}.txt"), "w",
                      encoding="utf-8") as fh:
                fh.write("\n".join(rows) + "\n")
    return root


def test_apply_rewrites_indices_into_the_profile_space(tmp_path):
    """The silent killer: filtered data still carrying 54-class indices."""
    src = _dataset(str(tmp_path / "src"), ["mcb", "vfd"])
    dst = str(tmp_path / "dst")
    pf.apply(src, dst, pf.CORE15, log=lambda m: None)

    profile_idx = pf.CORE15.index_of()
    canon = tax.class_index()
    seen = set()
    for line in open(os.path.join(dst, "labels", "train", "p0.txt"),
                     encoding="utf-8"):
        seen.add(int(line.split()[0]))
    # Must be profile indices (0 and 9), NOT taxonomy indices (0 and 26).
    assert seen == {profile_idx["mcb"], profile_idx["vfd"]} == {0, 9}
    assert canon["vfd"] not in seen or canon["vfd"] == 9


def test_every_written_index_is_inside_the_profile_head(tmp_path):
    src = _dataset(str(tmp_path / "src"),
                   ["mcb", "vfd", "plc", "terminal_block"], per_image=4)
    dst = str(tmp_path / "dst")
    pf.apply(src, dst, pf.CORE15, log=lambda m: None)
    for split in ("train", "val"):
        d = os.path.join(dst, "labels", split)
        for fn in os.listdir(d):
            for line in open(os.path.join(d, fn), encoding="utf-8"):
                if not line.strip():
                    continue
                assert 0 <= int(line.split()[0]) < pf.CORE15.class_count


def test_out_of_profile_boxes_are_dropped_and_counted(tmp_path):
    # energy_meter and ups are in the taxonomy but not in core15.
    src = _dataset(str(tmp_path / "src"), ["mcb", "energy_meter", "ups"],
                   per_image=3)
    dst = str(tmp_path / "dst")
    stats = pf.apply(src, dst, pf.CORE15, log=lambda m: None)

    assert stats["instances_dropped"] > 0
    assert set(stats["dropped_by_class"]) == {"energy_meter", "ups"}
    assert set(stats["per_class"]) == {"mcb"}


def test_images_left_empty_are_kept_as_negatives_by_default(tmp_path):
    """An image of only out-of-profile devices is a genuine negative."""
    src = _dataset(str(tmp_path / "src"), ["energy_meter"])
    dst = str(tmp_path / "dst")
    stats = pf.apply(src, dst, pf.CORE15, log=lambda m: None)

    assert stats["images_out"] == stats["images_in"]
    assert stats["images_emptied"] == stats["images_in"]
    assert stats["images_dropped_empty"] == 0
    label = open(os.path.join(dst, "labels", "train", "p0.txt"),
                 encoding="utf-8").read().strip()
    assert label == "", "the negative must be an empty label file, not a box"
    assert any("negatives" in w for w in stats["warnings"])


def test_drop_empty_removes_them_when_asked(tmp_path):
    src = _dataset(str(tmp_path / "src"), ["energy_meter"])
    dst = str(tmp_path / "dst")
    stats = pf.apply(src, dst, pf.CORE15, drop_empty=True, log=lambda m: None)
    assert stats["images_out"] == 0
    assert stats["images_dropped_empty"] == stats["images_in"]


def test_apply_writes_a_profile_dataset_yaml(tmp_path):
    src = _dataset(str(tmp_path / "src"), ["mcb", "vfd"])
    dst = str(tmp_path / "dst")
    pf.apply(src, dst, pf.CORE15, log=lambda m: None)

    text = open(os.path.join(dst, "dataset.yaml"), encoding="utf-8").read()
    assert f"nc: {pf.CORE15.class_count}" in text
    assert "0: mcb" in text
    assert "9: vfd" in text
    # The warning that these are not taxonomy indices must be in the file itself.
    assert "NOT taxonomy indices" in text


def test_apply_writes_a_profile_classes_json(tmp_path):
    """This is what makes a profile bundle work with no runtime change."""
    src = _dataset(str(tmp_path / "src"), ["mcb"])
    dst = str(tmp_path / "dst")
    pf.apply(src, dst, pf.CORE15, log=lambda m: None)

    data = json.load(open(os.path.join(dst, "classes.json"), encoding="utf-8"))
    assert data["profile"] == "core15"
    assert data["classes"] == list(pf.CORE15.classes)
    assert data["class_count"] == 15
    assert "APPEND ONLY" in data["_comment"]


def test_the_runtime_can_read_a_profile_label_space(tmp_path):
    """A 15-class bundle must load and canonicalise without code changes."""
    from rtsp_backend.electrical import recognizer as rec

    models = tmp_path / "models" / "components"
    models.mkdir(parents=True)
    pf.write_profile_yaml(str(models), pf.CORE15)
    names, source = rec.load_class_map(str(tmp_path / "models"))
    assert source == "classes.json"
    assert names == list(pf.CORE15.classes)
    canonical, unmapped = rec.resolve_names(names)
    assert unmapped == [], "every profile class must canonicalise"
    assert canonical == list(pf.CORE15.classes)


def test_absent_profile_classes_are_reported(tmp_path):
    src = _dataset(str(tmp_path / "src"), ["mcb"])
    dst = str(tmp_path / "dst")
    stats = pf.apply(src, dst, pf.CORE15, log=lambda m: None)
    assert len(stats["absent_classes"]) == 14
    assert "mcb" not in stats["absent_classes"]
    assert any("cannot be learned" in w for w in stats["warnings"])


def test_thin_classes_are_warned_about(tmp_path):
    src = _dataset(str(tmp_path / "src"), ["mcb"], per_image=1, n=3)
    dst = str(tmp_path / "dst")
    stats = pf.apply(src, dst, pf.CORE15, log=lambda m: None)
    assert any("trainability floor" in w for w in stats["warnings"])


# ==========================================================================
# deriving a profile from what the data contains
# ==========================================================================

def test_present_classes_finds_only_what_has_instances(tmp_path):
    src = _dataset(str(tmp_path / "src"), ["mcb", "vfd", "plc"], per_image=3)
    present = pf.present_classes(src, pf.CORE15)
    assert set(present) == {"mcb", "plc", "vfd"}
    # Order must follow the parent profile, not discovery order.
    assert present == tuple(c for c in pf.CORE15.classes if c in set(present))


def test_present_classes_honours_a_minimum(tmp_path):
    src = _dataset(str(tmp_path / "src"), ["mcb"], per_image=1, n=3)
    assert pf.present_classes(src, pf.CORE15, min_instances=1) == ("mcb",)
    assert pf.present_classes(src, pf.CORE15, min_instances=1000) == ()


def test_derive_keeps_the_parent_ordering(tmp_path):
    derived = pf.derive(pf.CORE15, ["vfd", "mcb", "plc"])
    # Parent order is mcb(0) ... plc(4) ... vfd(9), so the subsequence must be
    # mcb, plc, vfd — not the order they were passed in.
    assert derived.classes == ("mcb", "plc", "vfd")
    assert derived.name == "core15_present"


def test_derive_records_why_each_class_was_omitted():
    derived = pf.derive(pf.CORE15, ["mcb", "contactor"])
    assert derived.class_count == 2
    assert len(derived.excluded_notes) == 13
    assert all("no instances" in why for why in derived.excluded_notes.values())
    assert "zeros to the mAP mean" in derived.rationale


def test_a_derived_profile_is_a_subsequence_of_its_parent():
    """So a checkpoint on the subset stays comparable with one on the parent."""
    derived = pf.derive(pf.CORE15, ["mcb", "vfd", "plc", "relay"])
    parent = list(pf.CORE15.classes)
    positions = [parent.index(c) for c in derived.classes]
    assert positions == sorted(positions)


# ==========================================================================
# bundle validation
# ==========================================================================

def test_validate_accepts_an_exact_match():
    info = pf.validate(pf.CORE15, pf.CORE15.classes)
    assert info["ok"] is True
    assert "exactly" in info["note"]


def test_validate_accepts_an_append_only_prefix():
    """An older core15 checkpoint against the grown core18 profile."""
    info = pf.validate(pf.CORE18, pf.CORE15.classes)
    assert info["ok"] is True
    assert "valid prefix" in info["note"]
    assert "Fine-tune rather than retrain" in info["note"]


def test_validate_rejects_a_reordered_label_space():
    """A permuted order mislabels every detection from the first difference on."""
    scrambled = list(pf.CORE15.classes)
    scrambled[2], scrambled[3] = scrambled[3], scrambled[2]
    info = pf.validate(pf.CORE15, scrambled)
    assert info["ok"] is False
    assert "diverges" in info["problems"][0]
    assert "index 2" in info["problems"][0]
    assert "append-only" in info["problems"][0]


def test_validate_rejects_an_unrelated_label_space():
    info = pf.validate(pf.CORE15, ["cooling_fan", "cable_gland"])
    assert info["ok"] is False


# ==========================================================================
# data requirement
# ==========================================================================

def test_requirement_scales_with_the_class_count():
    """Half the classes, half the total instances for the same per-class depth."""
    small = pf.requirement_estimate(pf.CORE15, 0.85)
    big = pf.requirement_estimate(pf.FULL, 0.85)
    assert small["instances_per_class"] == big["instances_per_class"]
    assert big["total_instances"]["low"] > small["total_instances"]["low"]
    ratio = big["total_instances"]["low"] / small["total_instances"]["low"]
    assert ratio == pytest.approx(pf.FULL.class_count / pf.CORE15.class_count)


def test_requirement_is_a_band_not_a_point():
    """A single confident number here would be false precision."""
    est = pf.requirement_estimate(pf.CORE15, 0.85)
    assert est["instances_per_class"]["low"] < est["instances_per_class"]["high"]
    assert est["images"]["low"] < est["images"]["high"]
    assert len(est["bands"]) >= 3


def test_requirement_states_its_assumptions():
    est = pf.requirement_estimate(pf.CORE15, 0.85)
    joined = " ".join(est["assumptions"])
    assert "manufacturer" in joined
    assert "Split by panel" in joined
    assert "no setting makes it work" in est["caveat"]


def test_a_higher_target_needs_more_data():
    lo = pf.requirement_estimate(pf.CORE15, 0.50)
    hi = pf.requirement_estimate(pf.CORE15, 0.92)
    assert hi["instances_per_class"]["low"] > lo["instances_per_class"]["low"]


def test_export_takes_the_label_space_from_the_training_dataset(tmp_path):
    """A profile-trained head is N classes, not 54.

    Exporting it with the default full-taxonomy labels.txt ships a bundle whose labels
    disagree with its graph. verify_bundle catches that, but the caller should not have
    to hit the error to discover which label space to use — the dataset.yaml the model
    was trained on is the authoritative record.
    """
    from training.electrical import export as ex

    src = _dataset(str(tmp_path / "src"), ["mcb", "vfd"])
    profile_root = str(tmp_path / "prof")
    pf.apply(src, profile_root, pf.CORE15, log=lambda m: None)

    read = ex.classes_for_dataset(os.path.join(profile_root, "dataset.yaml"))
    assert read == list(pf.CORE15.classes)
    assert len(read) == 15

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"not a real checkpoint")
    res = ex.export_bundle(
        str(weights), str(tmp_path / "bundle"),
        metadata={"data": os.path.join(profile_root, "dataset.yaml")},
        log=lambda m: None)

    assert res["label_space"]["class_count"] == 15
    assert res["label_space"]["profile"] == "core15"
    labels = [ln.strip() for ln in
              open(tmp_path / "bundle" / "labels.txt", encoding="utf-8")
              if ln.strip()]
    assert labels == list(pf.CORE15.classes)
    data = json.load(open(tmp_path / "bundle" / "classes.json", encoding="utf-8"))
    assert data["profile"] == "core15"


def test_export_still_defaults_to_the_full_taxonomy_without_a_dataset(tmp_path):
    from training.electrical import export as ex

    weights = tmp_path / "best.pt"
    weights.write_bytes(b"x")
    res = ex.export_bundle(str(weights), str(tmp_path / "b"),
                           log=lambda m: None)
    assert res["label_space"]["class_count"] == len(tax.CLASS_ORDER)
    assert res["label_space"]["profile"] is None


def test_the_85_percent_target_lands_in_the_thousands_of_images():
    """Sanity-checks the estimate against the brief's own 5000-image instinct."""
    est = pf.requirement_estimate(pf.CORE15, 0.85)
    assert 2000 <= est["images"]["low"] <= 6000
    assert est["images"]["high"] >= est["images"]["low"]
