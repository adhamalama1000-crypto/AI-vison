"""Taxonomy: label resolution, priors, and the honest-ambiguity contract."""

from __future__ import annotations

import pytest

from rtsp_backend.electrical import taxonomy as tax


def test_class_order_is_unique_and_complete():
    assert len(tax.CLASS_ORDER) == len(set(tax.CLASS_ORDER))
    assert len(tax.CLASS_ORDER) >= 40, "the taxonomy must cover the full brief"
    for cid in tax.CLASS_ORDER:
        assert cid in tax.SPECS


def test_every_spec_is_fully_populated():
    for cid, sp in tax.SPECS.items():
        assert sp.name and sp.function and sp.role, cid
        assert sp.category in tax.CATEGORIES, cid
        assert sp.domain in tax.DOMAINS, cid
        assert sp.mounting, cid
        lo, hi = sp.aspect_ratio
        assert 0 < lo < hi, cid
        a_lo, a_hi = sp.rel_area
        assert 0 < a_lo < a_hi <= 1.0, cid
        assert 0.0 < sp.min_conf < 1.0, cid
        # every non-structural class needs zero-shot prompts, otherwise the
        # open-vocabulary backends silently cannot find it
        assert sp.prompts, cid


@pytest.mark.parametrize("label,expected", [
    ("MCB", "mcb"),
    ("mccb", "mccb"),
    ("Moulded Case Circuit Breaker", "mccb"),
    ("magnetic contactor", "contactor"),
    ("Schneider LC1D Contactor", "contactor"),
    ("thermal_overload_relay", "overload_relay"),
    ("overload relay", "overload_relay"),
    ("safety relay", "safety_relay"),
    ("relay", "relay"),
    ("Terminal Blocks", "terminal_block"),
    ("push-button", "push_button"),
    ("E-Stop", "emergency_stop"),
    ("programmable logic controller", "plc"),
    ("inverter", "vfd"),
    ("touch panel", "hmi"),
    ("din rail", "din_rail"),
])
def test_resolve_maps_real_world_labels(label, expected):
    assert tax.resolve(label) == expected


@pytest.mark.parametrize("label", [
    "circuit breaker",      # MCB? MCCB? ACB? RCCB? — genuinely ambiguous
    "circuit-breaker",
    "banana",
    "widget_42",
    "",
    None,
])
def test_resolve_refuses_to_guess(label):
    """An ambiguous or unknown label must not be silently mapped to a class."""
    assert tax.resolve(label) is None


def test_mcb_and_mccb_never_collide():
    assert tax.resolve("mcb") != tax.resolve("mccb")
    assert tax.resolve("MCB 3P") == "mcb"


def test_unknown_spec_is_permissive_and_marked():
    sp = tax.spec(tax.UNKNOWN_COMPONENT_ID)
    assert sp.id == tax.UNKNOWN_COMPONENT_ID
    assert sp.min_conf == 0.0
    # the unknown class must never be filtered out by geometry
    lo, hi = sp.aspect_ratio
    assert lo < 0.05 and hi > 50


def test_spec_falls_back_to_unknown_for_garbage():
    assert tax.spec("not_a_class").id == tax.UNKNOWN_COMPONENT_ID


def test_prompt_and_index_helpers():
    pm = tax.prompt_map()
    assert set(pm).issubset(set(tax.CLASS_ORDER))
    flat = tax.flat_prompts()
    assert all(cid in tax.SPECS for _, cid in flat)
    assert len(flat) > len(pm), "each class should offer several phrasings"

    idx = tax.class_index()
    assert idx[tax.CLASS_ORDER[0]] == 0
    assert sorted(idx.values()) == list(range(len(tax.CLASS_ORDER)))


def test_structural_classes_excluded_from_countable():
    countable = tax.countable_classes()
    assert "din_rail" not in countable
    assert "wire_duct" not in countable
    assert "busbar" not in countable
    assert "contactor" in countable
    assert "plc" in countable


def test_modular_and_door_class_sets():
    assert "mcb" in tax.modular_classes()
    assert "hmi" in tax.door_classes()
    assert "emergency_stop" in tax.door_classes()
    assert "hmi" not in tax.modular_classes()


def test_summary_is_serialisable():
    import json
    s = tax.summary()
    assert s["class_count"] == len(tax.CLASS_ORDER)
    assert s["unknown_class"] == tax.UNKNOWN_COMPONENT_ID
    json.dumps(s)  # must not raise


# --------------------------------------------------------------------------
# compact labels for the annotated overlay
# --------------------------------------------------------------------------

@pytest.mark.parametrize("class_id,expected", [
    ("mcb", "MCB"),
    ("mccb", "MCCB"),
    ("acb", "ACB"),
    ("plc", "PLC"),
    ("hmi", "HMI"),
    ("overload_relay", "OL Relay"),
    ("terminal_block", "Terminals"),
    ("emergency_stop", "E-Stop"),
    ("current_transformer", "CT"),
    ("power_supply", "PSU"),
    (tax.UNKNOWN_COMPONENT_ID, "Unknown"),
])
def test_short_name(class_id, expected):
    assert tax.short_name(class_id) == expected


def test_every_class_has_a_short_label_that_fits_an_overlay():
    """A long label overlaps its neighbours and makes the overlay unreadable."""
    for cid in tax.CLASS_ORDER:
        short = tax.short_name(cid)
        assert short, cid
        assert "(" not in short, cid
        assert len(short) <= 12, f"{cid}: {short!r} is too long for an overlay"
