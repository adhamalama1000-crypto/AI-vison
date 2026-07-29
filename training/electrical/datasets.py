"""
Dataset acquisition and unification for industrial component detection.

The single biggest reason the previous system could not recognise a contactor is
that no model was ever trained, and no model was ever trained because there was
no dataset. This module is the plan for getting one, executed as code.

Three problems it solves:

1. **Where to get data.** :data:`SOURCES` is a curated registry of public
   datasets that contain industrial electrical components, each with its licence,
   the classes it actually covers, and a mapping from its label names onto the
   canonical taxonomy. Nothing is downloaded implicitly — you supply an API key
   or a local copy, and :func:`plan` tells you exactly what you will get.

2. **Merging incompatible label spaces.** Public sets disagree on everything:
   ``"breaker"`` vs ``"MCB"`` vs ``"circuit_breaker_1p"``. :func:`remap_yolo_dataset`
   rewrites any YOLO-format dataset onto :data:`~rtsp_backend.electrical.taxonomy.CLASS_ORDER`
   using each source's declared mapping plus the taxonomy resolver, and *drops*
   (with a count) any class it cannot map rather than guessing. :func:`merge`
   then unions several remapped datasets into one training set.

3. **Knowing whether the result is usable.** :func:`analyse_dataset` reports
   per-class instance counts, images per class, box-size distribution and the
   long tail. :func:`coverage_report` compares that against the taxonomy and
   names the classes that will *not* work yet, so nobody is surprised when the
   model cannot find an ACB it never saw.

Where public data is insufficient — which, for most of this taxonomy, it is —
:func:`custom_collection_plan` emits the concrete capture protocol for building
a proprietary Madkour dataset, and
:mod:`training.electrical.synthetic` multiplies a small crop library into a
large labelled set.
"""

from __future__ import annotations

import json
import os
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Sequence

from rtsp_backend.electrical import taxonomy as tax

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass(frozen=True)
class DatasetSource:
    """A public dataset that contains industrial electrical components."""

    key: str
    name: str
    #: "roboflow" | "kaggle" | "url" | "github" | "openimages" | "manual"
    kind: str
    #: Where to get it. Roboflow entries are ``workspace/project/version``.
    locator: str
    licence: str
    #: Taxonomy classes this source realistically contributes.
    provides: tuple[str, ...]
    #: source label -> taxonomy id. Labels absent here fall back to the resolver.
    label_map: Mapping[str, str] = field(default_factory=dict)
    notes: str = ""
    #: Rough instance count, for planning only.
    approx_instances: Optional[int] = None
    #: Images in the source (base images, before Roboflow augmentation).
    images: Optional[int] = None
    #: Observed per-class instance counts, read off the source project. These
    #: are what :func:`plan` uses to predict coverage *before* downloading
    #: anything, so the shortfall is known in advance rather than discovered
    #: after an afternoon of downloads.
    class_counts: Mapping[str, int] = field(default_factory=dict)
    #: Set when the source must not go into RGB training (e.g. thermal imagery).
    excluded_reason: Optional[str] = None
    #: True when the locator was verified to exist and be downloadable.
    verified: bool = False

    @property
    def usable(self) -> bool:
        return self.excluded_reason is None

    def cli_hint(self) -> str:
        if self.kind == "roboflow":
            parts = self.locator.split("/")
            if len(parts) < 3:
                return (f"roboflow: project '{self.locator}' has no generated "
                        f"version — fork it in the Roboflow UI and generate one, "
                        f"then set the version here")
            return (f"python -m training.electrical.cli download "
                    f"--sources {self.key}   "
                    f"# rf.workspace('{parts[0]}').project('{parts[1]}')"
                    f".version({parts[2]}).download('yolov11')")
        if self.kind == "kaggle":
            return (f"python -m training.electrical.cli download "
                    f"--sources {self.key}   "
                    f"# kaggle datasets download -d {self.locator}")
        if self.kind == "url":
            return (f"python -m training.electrical.cli download "
                    f"--sources {self.key}   # curl -L {self.locator}")
        if self.kind == "github":
            return (f"python -m training.electrical.cli download "
                    f"--sources {self.key}   "
                    f"# github archive/release asset: {self.locator}")
        if self.kind == "openimages":
            return (f"python -m training.electrical.cli download "
                    f"--sources {self.key}   "
                    f"# needs fiftyone; Open Images classes: {self.locator}")
        return "manual acquisition — see notes"


#: Curated source registry — **verified locators**, not placeholders.
#:
#: Every ``roboflow`` entry below was looked up on Roboflow Universe and its
#: project metadata read: image count, generated version number, and the actual
#: per-class instance counts recorded in :attr:`DatasetSource.class_counts`.
#: That is what makes :func:`plan` an honest forecast instead of a wish list.
#:
#: The headline finding from that survey, stated plainly: **public data does not
#: cover this taxonomy.** The whole of public Universe contributes on the order
#: of 2.5k usable panel images, most sources carry only 20–50 instances per
#: class (below :data:`MIN_INSTANCES_TRAINABLE`), the imagery skews hard toward
#: medium-voltage switchgear and German domestic distribution boards rather than
#: LV industrial control panels, and several requested classes — VFD, SMPS,
#: busbar, DIN rail, cable duct, emergency stop — have no usable public source
#: at all. Use this registry to bootstrap; use
#: :func:`custom_collection_plan` and :func:`requirements_report` to plan the
#: capture programme that actually gets the model to production.
SOURCES: tuple[DatasetSource, ...] = (
    # ---------------------------------------------------------------------
    # LV industrial control panels — the deployment domain. Only one usable
    # public source, and it is a single panel photographed many times.
    # ---------------------------------------------------------------------
    DatasetSource(
        key="rf_electrical_panel_imgpro",
        name="Roboflow Universe — IMGPRO 'Electrical panel'",
        kind="roboflow", locator="imgpro-2yjry/electrical-panel-nq9g7/1",
        licence="CC BY 4.0",
        provides=("contactor", "mcb", "relay", "overload_relay", "timer_relay"),
        label_map={
            "contactor": "contactor",
            "mcb": "mcb",
            "relay": "relay",
            # source typo, kept verbatim — the label really is 'overlaod'
            "overlaod": "overload_relay",
            "delay_timer": "timer_relay",
            "digital-timer": "timer_relay",
        },
        images=256, approx_instances=1529,
        class_counts={"contactor": 256, "mcb": 254, "relay": 256,
                      "overload_relay": 255, "timer_relay": 508},
        verified=True,
        notes="The single most on-domain public set: a real LV motor-control "
              "panel with contactors, MCBs, control relays, an overload relay "
              "and timers. ~255 instances of each class across 256 images — but "
              "read that number carefully: it is roughly one instance per image "
              "of what appears to be the SAME panel, so the effective diversity "
              "is one cabinet, not 256. It will teach the model what a contactor "
              "looks like from many angles and will NOT teach it manufacturer or "
              "layout invariance. Split it by image with care and never let it "
              "dominate the merged set.",
    ),
    DatasetSource(
        key="rf_control_panel_anoop",
        name="Roboflow Universe — Anoop 'Control panel'",
        kind="roboflow", locator="anoop-alrrb/control-panel-oozlx/2",
        licence="CC BY 4.0",
        provides=("contactor", "relay", "cooling_fan", "fuse", "indicator_lamp",
                  "mccb", "motor_starter", "earth_bar", "transformer",
                  "terminal_block", "wire_duct", "thermostat"),
        label_map={
            "AUX CONTACT": "relay", "CONTACTOR": "contactor",
            "Cable Trunking": "wire_duct", "Control Relay": "relay",
            "Cooling Fan": "cooling_fan", "FUSE": "fuse",
            "Indication lamp": "indicator_lamp", "MCCB": "mccb",
            "MPCB": "motor_starter", "PFR": "pf_controller",
            "Protective Earth Bar": "earth_bar",
            "insulated Earth bar": "neutral_bar",
            "THERMOSTAT": "thermostat", "Transformer": "transformer",
            "terminal block": "terminal_block",
            # 'objects' and 'POL 648' are not component classes — dropped.
        },
        images=34, approx_instances=None,
        verified=True,
        notes="Only 34 images, but the highest CLASS relevance of anything "
              "public: it is the only source that labels terminal blocks, cable "
              "trunking, earth bars, panel thermostats and control transformers "
              "inside a real LV cabinet. Far too small to train these classes "
              "(34 images cannot produce 300 instances of anything) — its value "
              "is as a labelling reference and as seed crops for the crop "
              "library. Treat it as documentation of what good labelling looks "
              "like for the infrastructure classes.",
    ),
    DatasetSource(
        key="rf_dmm_panel",
        name="Roboflow Universe — DMM control-panel symbols & devices",
        kind="roboflow", locator="dmm-ythvo/my-first-project-6ehqh/5",
        licence="CC BY 4.0",
        provides=("vfd", "cooling_fan", "indicator_lamp", "transformer",
                  "limit_switch", "circuit_breaker", "contactor", "relay",
                  "fuse", "thermostat", "earth_bar"),
        label_map={
            "VFD": "vfd", "fan": "cooling_fan", "lamp": "indicator_lamp",
            "transformer": "transformer", "limit switch nc": "limit_switch",
            "circuit breaker": "circuit_breaker", "contactor": "contactor",
            "control relay coil": "relay", "fuse": "fuse",
            "hygrostat": "thermostat", "earthing": "earth_bar",
            "grounding": "earth_bar", "indication light pilot": "indicator_lamp",
            "changeover connect": "changeover_switch",
            # 'arrow', 'end', 'RM', 'bms', 'dmm' are drawing annotations, not
            # devices — dropped rather than guessed.
        },
        images=51,
        verified=True,
        notes="51 images and the ONLY public source that labels a VFD at all. "
              "Mixed content — some frames are schematic symbols rather than "
              "photographs, which is the wrong distribution for a photo "
              "detector. Inspect before merging and consider keeping only the "
              "photographic frames.",
    ),

    # ---------------------------------------------------------------------
    # Medium-voltage switchgear. Adjacent domain, useful for the shared
    # classes (contactor, CT, fuse, push button, lamp) but the breakers are
    # a different physical device from an LV panel MCB.
    # ---------------------------------------------------------------------
    DatasetSource(
        key="rf_switchgear_varsha",
        name="Roboflow Universe — 'Switchgear Components' (24 classes)",
        kind="roboflow",
        locator="varsha-rao-boinapalli/switchgear-components-dufv5/2",
        licence="CC BY 4.0",
        provides=("acb", "circuit_breaker", "mccb", "mcb", "contactor",
                  "cooling_fan", "current_transformer", "voltage_transformer",
                  "fuse", "indicator_lamp", "push_button", "selector_switch",
                  "protection_relay", "relay", "capacitor", "surge_protector",
                  "energy_meter"),
        label_map={
            "Air Circuit Breaker": "acb",
            "Miniature Circuit Breaker": "mcb",
            "MCCB": "mccb",
            # 'Circuit Breaker', VCB and SF6 CB are medium-voltage breakers
            # with no LV equivalent in the taxonomy. They map to the generic
            # circuit_breaker class rather than being forced onto MCCB.
            "Circuit Breaker": "circuit_breaker",
            "VCB": "circuit_breaker",
            "SF6 Circuit breaker": "circuit_breaker",
            "Contactor": "contactor",
            "Auxiliary Relay": "relay",
            "Protection Relay": "protection_relay",
            "Cooling Fan": "cooling_fan",
            "Current Transformer": "current_transformer",
            "Potential Voltage Transformer": "voltage_transformer",
            "Fuse": "fuse",
            "Indication Lamps": "indicator_lamp",
            "Push Buttons": "push_button",
            "Selector Switch": "selector_switch",
            "Capacitor Bank": "capacitor",
            "Surge Arrester": "surge_protector",
            "Metering Units": "energy_meter",
            # Deliberately UNMAPPED — no honest taxonomy home, dropped with a
            # count by remap_yolo_dataset:
            #   'Annuciator', 'Cable Termination', 'Interlocking Mechanism',
            #   'Earthing Switch', 'Isolator Switch'
        },
        images=723, approx_instances=743,
        class_counts={"acb": 30, "circuit_breaker": 91, "mccb": 36, "mcb": 36,
                      "contactor": 34, "cooling_fan": 29,
                      "current_transformer": 29, "voltage_transformer": 31,
                      "fuse": 30, "indicator_lamp": 31, "push_button": 49,
                      "selector_switch": 30, "protection_relay": 27,
                      "relay": 30, "capacitor": 25, "surge_protector": 30,
                      "energy_meter": 29},
        verified=True,
        notes="723 images / 24 classes, and the widest class coverage in public "
              "data. The catch is density: ~25–49 instances PER CLASS, i.e. "
              "below MIN_INSTANCES_TRAINABLE for almost every class. This is a "
              "one-instance-per-image classification set relabelled as "
              "detection. It is also medium-voltage switchgear, not an LV "
              "control panel, so its 'Circuit Breaker' is a cubicle-mounted VCB "
              "rather than anything on a DIN rail. Genuinely useful for CTs, "
              "protection relays, push buttons and lamps; misleading for "
              "breakers.",
    ),
    DatasetSource(
        key="rf_switchgear_potholes",
        name="Roboflow Universe — 'switchgear_components' (17 classes)",
        kind="roboflow", locator="potholes-ytdve/switchgear_components/1",
        licence="CC BY 4.0",
        provides=("circuit_breaker", "mcb", "contactor", "relay", "fuse",
                  "current_transformer", "voltage_transformer", "push_button",
                  "selector_switch", "capacitor", "surge_protector"),
        label_map={
            "miniature circuit breaker": "mcb",
            "Circuit Breaker": "circuit_breaker",
            "vaccum circuit breaker": "circuit_breaker",
            "sf6 circuit breaker": "circuit_breaker",
            "contactor": "contactor",
            "Auxiliary Relay": "relay",
            "current transformer": "current_transformer",
            "potential voltage transformer": "voltage_transformer",
            "fuse": "fuse",
            "push Button": "push_button",
            "selector switch": "selector_switch",
            "capacitor bank": "capacitor",
            "surge arrester": "surge_protector",
            # UNMAPPED: 'annunciator', 'cabel termination',
            # 'interlocking mechanism', and the empty 'switchgear-components'
            # umbrella class (0 instances).
        },
        images=464, approx_instances=546,
        class_counts={"circuit_breaker": 83, "mcb": 33, "contactor": 34,
                      "relay": 40, "fuse": 36, "current_transformer": 29,
                      "voltage_transformer": 32, "push_button": 43,
                      "selector_switch": 30, "capacitor": 37,
                      "surge_protector": 30},
        verified=True,
        notes="A near-duplicate taxonomy to rf_switchgear_varsha and very "
              "probably overlapping imagery — the class list and the ~30 "
              "instances-per-class signature match almost exactly. Merging both "
              "may duplicate images rather than add diversity, which inflates "
              "validation scores. Deduplicate by perceptual hash after merging, "
              "or pick one. The published version 1 reports 1114 images because "
              "Roboflow augmentation is baked in; the base set is 464.",
    ),

    # ---------------------------------------------------------------------
    # Domestic / commercial distribution boards. Well covered publicly,
    # different domain, but the best available source of modular DIN-rail
    # device imagery in volume.
    # ---------------------------------------------------------------------
    DatasetSource(
        key="rf_control_panels_azure",
        name="Roboflow Universe — 'Control Panels' (German distribution boards)",
        kind="roboflow", locator="control-panel-azure/control-panels",
        licence="CC BY 4.0",
        provides=("mcb", "rccb", "rcbo", "fuse", "fuse_holder", "energy_meter",
                  "ammeter", "surge_protector", "transformer", "relay",
                  "contactor", "terminal_block", "changeover_switch"),
        label_map={
            "circuit breaker 1-pole": "mcb",
            "circuit breaker 2-pole": "mcb",
            "circuit breaker 3-pole": "mcb",
            "circuit breaker 3-pole 32A": "mcb",
            "SLS -selektiver Leitungsschutzschalter-": "mcb",
            "selective circuit breaker": "mcb",
            "Leitungsschutzschalter": "mcb",
            "residual current circuit breaker": "rccb",
            "residual current circuit breaker 2-pole": "rccb",
            "residual current circuit breaker 4-pole": "rccb",
            "RCBO 1-pole": "rcbo",
            "fuse": "fuse", "fuse 3-pole": "fuse",
            "NH-fuse 1-pole": "fuse", "NH-fuse 3-pole": "fuse",
            "neozed fuse 1pole": "fuse", "neozed fuse 3pole": "fuse",
            "fuse socket 3-pole": "fuse_holder",
            "Electric-Meter": "energy_meter",
            "digital electrical meter": "energy_meter",
            "smart energy meter": "energy_meter",
            "power meter": "energy_meter",
            "analog electrical meter": "ammeter",
            "analoger Stromzahler": "ammeter",
            "surge protection device": "surge_protector",
            "surge protection device typ 2": "surge_protector",
            "bell transformer": "transformer",
            "doorbell transformer": "transformer",
            "transformer intercom": "transformer",
            "relay": "relay", "relais": "relay", "staircase relay": "timer_relay",
            "contactor": "contactor",
            "terminal block": "terminal_block",
            "main branch terminals": "terminal_block",
            "equipotential bonding rail": "earth_bar",
            "main switch": "changeover_switch",
            # UNMAPPED (enclosure furniture and non-devices): 'cover -S35S-',
            # 'junction box', 'house connection box', 'distribution box
            # -2-rows-', 'electricalmeter cabinet', 'media field', 'wallbox',
            # 'power outlet', 'CEE-socket', 'cable', 'fire extinguisher',
            # 'eHZ adapter plate', 'ripple control receiver'.
        },
        images=320, approx_instances=1900,
        class_counts={"mcb": 703, "rccb": 211, "rcbo": 14, "fuse": 260,
                      "energy_meter": 261, "ammeter": 149,
                      "surge_protector": 25, "transformer": 40, "relay": 22,
                      "contactor": 4, "terminal_block": 4,
                      "changeover_switch": 84},
        verified=True,
        notes="THE best public source of modular DIN-rail device instances: "
              "~703 MCB and ~211 RCCB boxes, an order of magnitude more than "
              "anything else. Two caveats. (1) It has NO GENERATED VERSION, so "
              "it cannot be downloaded by version — fork the project into your "
              "own Roboflow workspace, generate a version, and put "
              "'<your-workspace>/control-panels/1' in the locator. The "
              "downloader reports this rather than failing obscurely. (2) It is "
              "German domestic/commercial metering boards, so device styling, "
              "layout and the meter-heavy class mix differ from an industrial "
              "control panel. Excellent for teaching 'what a row of modular "
              "breakers looks like'; not a substitute for Madkour panels.",
    ),
    DatasetSource(
        key="rf_gid_mlops",
        name="Roboflow Universe — GID-MLOps master thesis (metering boards)",
        kind="roboflow", locator="gid-qolf2/gid-mlops-master-thesis/5",
        licence="CC BY 4.0",
        provides=("mcb", "rccb", "rcbo", "fuse", "energy_meter", "ammeter",
                  "changeover_switch"),
        label_map={
            "Leitungsschutzschalter": "mcb",
            "Leitungsschutzschalter 1-polig": "mcb",
            "SLS (selektiver Leitungsschutzschalter)": "mcb",
            "FI-Schutzschalter 4-polig": "rccb",
            "FI/LS Kombination 1-polig": "rcbo",
            "RCBO 1-pole": "rcbo",
            "NH-Sicherungen": "fuse", "NH-fuse 1-pole": "fuse",
            "NH-fuse 3-pole": "fuse", "Schmelzsicherungen": "fuse",
            "Neozed Schmelzsicherungen 3-fach": "fuse",
            "Electric-Meter": "energy_meter",
            "Digital Meter": "energy_meter",
            "Analog meter": "ammeter",
            "Hauptschalter": "changeover_switch",
            "Einbausteckdose für Hutschiene": "terminal_block",
            # UNMAPPED: 'cable', 'switch', 'fire extinguisher', 'CEE-socket'.
        },
        images=388,
        verified=True,
        notes="Same domain and probably the same lineage as "
              "rf_control_panels_azure (overlapping German class names). 66 "
              "declared classes over 388 images means a very long tail. Labels "
              "are German; the label_map above is the translation, and anything "
              "not in it is dropped with a count rather than guessed.",
    ),

    # ---------------------------------------------------------------------
    # Single-class / narrow sources. Useful to lift one weak class each.
    # ---------------------------------------------------------------------
    DatasetSource(
        key="rf_plc_dataset",
        name="Roboflow Universe — PLC families (Siemens S7 / Mitsubishi)",
        kind="roboflow", locator="dataset-plc/plc-yuevb/3",
        licence="CC BY 4.0",
        provides=("plc", "io_module"),
        label_map={"S7-1200": "plc", "S7-1500": "plc", "S7-300": "plc",
                   "Mitsubishi": "plc", "I/O Card": "io_module"},
        images=557, approx_instances=569,
        class_counts={"plc": 459, "io_module": 110},
        verified=True,
        notes="The best public PLC source by a wide margin: 557 base images "
              "(version 3 reports 1321 with augmentation) labelled by product "
              "family, which collapses cleanly onto plc + io_module. Studio and "
              "bench photography rather than in-cabinet, so it teaches device "
              "appearance but not panel context — pair it with in-cabinet "
              "captures.",
    ),
    DatasetSource(
        key="rf_plc_fabrica",
        name="Roboflow Universe — Fabrica Inteligente 'PLC'",
        kind="roboflow", locator="fabrica-inteligente-40igs/plc-vp7bx/1",
        licence="CC BY 4.0",
        provides=("plc",),
        label_map={"PLC": "plc"},
        images=252,
        verified=True,
        notes="252 images, single 'PLC' class. Straightforward top-up for the "
              "plc class; check for overlap with rf_plc_dataset before merging.",
    ),
    DatasetSource(
        key="rf_terminal_block",
        name="Roboflow Universe — terminal block / wire numbering",
        kind="roboflow", locator="braker-p7qyl/terminal-block-jtgsl/1",
        licence="CC BY 4.0",
        provides=("terminal_block",),
        label_map={"Terminal": "terminal_block"},
        images=90,
        verified=True,
        notes="90 images of terminal strips. IMPORTANT labelling mismatch: this "
              "source boxes each terminal INDIVIDUALLY, while the Madkour "
              "labelling rule is one box per contiguous strip. Merging it as-is "
              "teaches the model per-pole boxes and will wreck terminal-block "
              "counts in the bill of materials. Either re-label to strips or "
              "keep it out. Its 'Wire' / 'Wire Number' / 'Labels' classes are "
              "not taxonomy components and are dropped.",
    ),
    DatasetSource(
        key="rf_indicator_lamp",
        name="Roboflow Universe — indicator lamp",
        kind="roboflow", locator="orang-scd99/indicator-lamp-zx4yp/2",
        licence="CC BY 4.0",
        provides=("indicator_lamp",),
        label_map={"lamp": "indicator_lamp"},
        images=378,
        verified=True,
        notes="378 images, single 'lamp' class — mapped explicitly here because "
              "the bare word 'lamp' is too vague for the global resolver to "
              "accept. Verify the imagery is panel-mounted pilot lamps and not "
              "room lighting before merging.",
    ),

    # ---------------------------------------------------------------------
    # Excluded — recorded so nobody rediscovers them and merges them by mistake.
    # ---------------------------------------------------------------------
    DatasetSource(
        key="rf_thermal_panel",
        name="Roboflow Universe — 'thermique' (thermal panel imagery)",
        kind="roboflow", locator="mohamed-hannat/thermique/9",
        licence="CC BY 4.0",
        provides=("circuit_breaker", "contactor", "relay", "transformer",
                  "motor_starter", "overload_relay"),
        label_map={"circuit_breaker": "circuit_breaker",
                   "contactor": "contactor", "relay": "relay",
                   "transformer": "transformer", "tansformer": "transformer",
                   "motor_starter": "motor_starter",
                   "thermal_relay": "overload_relay"},
        images=140,
        excluded_reason=(
            "Thermal/IR imagery. The deployment cameras are visible-light RTSP "
            "streams, so these images are a different input distribution "
            "entirely — mixing them in teaches the model nothing about RGB "
            "panels and costs validation accuracy. Keep it for a future "
            "thermal-inspection model, where it would be genuinely valuable."),
        verified=True,
        notes="Listed and deliberately excluded. Revisit if thermal inspection "
              "becomes a product requirement.",
    ),
    DatasetSource(
        key="rf_pushbutton_generic",
        name="Roboflow Universe — generic push-button switch sets",
        kind="roboflow", locator="biiim/push-button-switch/1",
        licence="CC BY 4.0",
        provides=("push_button",),
        label_map={"switch": "push_button", "push_on": "push_button"},
        images=1958,
        excluded_reason=(
            "1958 images, but they are consumer/appliance push buttons in ON/OFF "
            "state-classification framing, not panel-mounted 22 mm industrial "
            "actuators. Training on them would teach the detector to fire on "
            "every round button in the frame — a false-positive generator."),
        verified=True,
        notes="Recorded as a trap. The image count is tempting and the domain is "
              "wrong.",
    ),

    # ---------------------------------------------------------------------
    # Non-Roboflow options and the proprietary programme.
    # ---------------------------------------------------------------------
    DatasetSource(
        key="openimages_hard_negatives",
        name="Open Images V7 — domestic switches/sockets (HARD NEGATIVES ONLY)",
        kind="openimages", locator="Light switch,Power plugs and sockets",
        licence="CC BY 4.0 (annotations) / CC BY 2.0 (images) — attribute Google "
                "and the Flickr photographers",
        provides=(),
        images=None,
        verified=True,
        notes="Verified: Open Images V7 boxable classes 309 'Light switch' and 405 "
              "'Power plugs and sockets' exist (confirmed against the Ultralytics "
              "open-images-v7.yaml class list). Open Images has NO industrial "
              "electrical classes — no MCB, no contactor, no PLC, nothing on a DIN "
              "rail. `provides` is therefore deliberately EMPTY: this source "
              "contributes zero positive instances and plan() must not credit it "
              "with any.\n\n"
              "Its value is as HARD NEGATIVES. A detector trained only on panel "
              "interiors will happily fire 'push_button' on a domestic light "
              "switch and 'indicator_lamp' on a wall socket, and negatives are the "
              "only thing that teaches it not to. Import these images with EMPTY "
              "label files (the downloader's normaliser already writes an empty "
              "label for an unlabelled image, treating it as a negative), and cap "
              "them at roughly 10%% of the training set — beyond that they "
              "suppress genuine detections.\n\n"
              "Needs `pip install fiftyone`; there is no fallback, because "
              "fetching Open Images without partial download means pulling ~500 GB "
              "to keep a few hundred images.",
    ),
    DatasetSource(
        key="github_dataset_template",
        name="GitHub — dataset repository or release asset",
        kind="github", locator="<owner>/<repo>::<asset.zip>@<tag>",
        licence="per-repository — read the LICENSE file before training on it",
        provides=(),
        notes="A working fetcher with no verified source behind it, and that is a "
              "finding rather than an omission. GitHub was searched for electrical "
              "panel / switchgear / circuit-breaker detection datasets and "
              "returned nothing usable: the matches are a Java circuit-breaker "
              "resilience library, a Brawl Stars mod, a switchgear-symbol printing "
              "repo and a Schneider selection tool. None contain annotated panel "
              "imagery. Naming one here would be a fake citation.\n\n"
              "The fetcher is real and ready for when you find one:\n"
              "  --locator owner/repo@v1.0                  (pinned archive)\n"
              "  --locator owner/repo::images.zip@v1.0      (release asset)\n"
              "Always pin a tag or commit. An unpinned default-branch download is "
              "not reproducible — the same command next month gives different "
              "bytes, and your dataset provenance is gone.",
    ),
    DatasetSource(
        key="kaggle_electrical_components",
        name="Kaggle — electrical / electronic component image sets",
        kind="kaggle", locator="<owner>/<dataset-slug>",
        licence="per-dataset — verify (many are CC0 or CC BY-SA)",
        provides=(),
        notes="Kaggle hosts electronic-component sets (resistors, capacitors, "
              "ICs, PCB defects) which are the WRONG SCALE for panel devices, "
              "and a handful of switchgear photo collections that are "
              "classification-only (no boxes). Search 'electrical panel', "
              "'switchgear', 'circuit breaker' and pass "
              "--sources kaggle_electrical_components --locator <owner>/<slug> "
              "once you have found one worth using. Left unresolved on purpose: "
              "no Kaggle dataset was verified to add boxed panel-device "
              "instances, and inventing a slug here would be a fake citation.",
    ),
    DatasetSource(
        key="vendor_catalogue_crops",
        name="Manufacturer catalogue product photography",
        kind="manual", locator="vendor product pages / catalogue PDFs",
        licence="COPYRIGHTED — obtain written permission before training on it",
        provides=tuple(tax.CLASS_ORDER),
        notes="Every manufacturer publishes clean studio photographs of every "
              "product, labelled with the exact part number. As a *crop "
              "library* for training.electrical.synthetic.compose_from_crops "
              "this is the highest-value source per unit of effort: it covers "
              "the whole taxonomy including the classes public datasets miss, "
              "and it comes with ground-truth part numbers for the nameplate "
              "catalogue. It is also copyrighted — clear it with the vendor or "
              "your legal team first. Studio crops alone under-represent real "
              "lighting and occlusion, which is precisely what the synthetic "
              "compositor adds.",
    ),
    DatasetSource(
        key="vendor_catalogue_crops",
        name="Manufacturer catalogue product photography",
        kind="manual", locator="vendor product pages / catalogue PDFs",
        licence="COPYRIGHTED — obtain written permission before training on it",
        provides=tuple(tax.CLASS_ORDER),
        notes="Every manufacturer publishes clean studio photographs of every "
              "product, labelled with the exact part number. As a *crop "
              "library* for training.electrical.synthetic.compose_from_crops "
              "this is the highest-value source per unit of effort: it covers "
              "the whole taxonomy including the classes public datasets miss, "
              "and it comes with ground-truth part numbers for the nameplate "
              "catalogue. It is also copyrighted — clear it with the vendor or "
              "your legal team first. Studio crops alone under-represent real "
              "lighting and occlusion, which is precisely what the synthetic "
              "compositor adds.",
    ),
    DatasetSource(
        key="madkour_field_capture",
        name="Madkour field capture programme (proprietary)",
        kind="manual", locator="internal capture — see custom_collection_plan()",
        licence="proprietary — owned by Madkour",
        provides=tuple(tax.CLASS_ORDER),
        notes="The only source that matches the deployment distribution: real "
              "Madkour panels, real cabinets, real lighting, real dirt. This is "
              "what production accuracy ultimately depends on. Everything else "
              "is a bootstrap.",
    ),
)

SOURCE_INDEX: dict[str, DatasetSource] = {s.key: s for s in SOURCES}


def plan(keys: Optional[Sequence[str]] = None,
         include_excluded: bool = False) -> dict:
    """What each selected source contributes, and what remains uncovered.

    Unlike a wish list, this forecasts *instance* counts from the verified
    :attr:`DatasetSource.class_counts` recorded for each source, so the
    trainability shortfall is known before anything is downloaded. Sources
    carrying an :attr:`~DatasetSource.excluded_reason` are omitted from the
    forecast (and listed separately) — they exist in the registry so they are
    not rediscovered and merged by mistake.
    """
    chosen = [SOURCE_INDEX[k] for k in (keys or list(SOURCE_INDEX))
              if k in SOURCE_INDEX]
    excluded = [s for s in chosen if not s.usable]
    if not include_excluded:
        chosen = [s for s in chosen if s.usable]

    # A "manual" source (vendor catalogues, field capture) nominally provides
    # every class, but only after somebody photographs and labels it. Counting
    # that as coverage would turn this report into wishful thinking, so
    # downloadable and manual coverage are reported separately.
    downloadable: set[str] = set()
    manual: set[str] = set()
    for s in chosen:
        (downloadable if s.kind != "manual" else manual).update(s.provides)
    uncovered = [c for c in tax.CLASS_ORDER if c not in downloadable]
    manual_only = [c for c in uncovered if c in manual]
    nowhere = [c for c in uncovered if c not in manual]

    # Forecast instances per class by summing the observed per-source counts.
    # Only sources that actually reported counts contribute, so the forecast is
    # a floor, not an estimate: classes listed in `provides` without counts show
    # up as covered-but-unquantified rather than being credited with a number
    # nobody measured.
    forecast: Counter = Counter()
    unquantified: set[str] = set()
    for s in chosen:
        if s.kind == "manual":
            continue
        for cid in s.provides:
            n = s.class_counts.get(cid)
            if n:
                forecast[cid] += int(n)
            else:
                unquantified.add(cid)
    images_total = sum(s.images or 0 for s in chosen if s.kind != "manual")

    return {
        "sources": [
            {"key": s.key, "name": s.name, "kind": s.kind,
             "locator": s.locator, "licence": s.licence,
             "verified": s.verified, "images": s.images,
             "provides": list(s.provides), "notes": s.notes,
             "how": s.cli_hint()}
            for s in chosen
        ],
        "excluded_sources": [
            {"key": s.key, "name": s.name, "locator": s.locator,
             "images": s.images, "reason": s.excluded_reason}
            for s in excluded
        ],
        "classes_from_downloadable_sources": sorted(downloadable),
        "classes_needing_manual_capture": manual_only,
        "classes_with_no_source": nowhere,
        "downloadable_coverage_fraction": round(
            len(downloadable) / max(1, len(tax.CLASS_ORDER)), 3),
        "forecast_images": images_total,
        "forecast_instances_per_class": dict(
            sorted(forecast.items(), key=lambda kv: -kv[1])),
        "forecast_unquantified_classes": sorted(unquantified),
        "forecast_reliable_classes": sorted(
            c for c, n in forecast.items() if n >= MIN_INSTANCES_RELIABLE),
        "forecast_weak_classes": sorted(
            c for c, n in forecast.items()
            if MIN_INSTANCES_TRAINABLE <= n < MIN_INSTANCES_RELIABLE),
        "forecast_untrainable_classes": sorted(
            c for c, n in forecast.items() if n < MIN_INSTANCES_TRAINABLE),
        # The blunt answer to "what can public data NOT give me": priority
        # classes for which no selected source contributes a single measured
        # instance. These need a capture programme, full stop.
        "priority_classes_with_no_public_instances": [
            c for c in PRIORITY_CLASSES if not forecast.get(c)],
        "verdict": (
            f"{len(downloadable)} of {len(tax.CLASS_ORDER)} classes have a "
            f"downloadable public source; {len(manual_only)} depend on vendor "
            f"catalogue crops or the Madkour field-capture programme, and "
            f"{len(nowhere)} have no source at all. The public sources total "
            f"~{images_total} images, and on the measured per-class counts only "
            f"{len([c for c, n in forecast.items() if n >= MIN_INSTANCES_RELIABLE])} "
            f"class(es) reach the {MIN_INSTANCES_RELIABLE}-instance reliability "
            f"bar. Public data alone will NOT produce a production model for "
            f"this taxonomy — use it to bootstrap the modular protection "
            f"classes, and run requirements_report() against a merged dataset "
            f"to get the exact capture shortfall."
            if uncovered else
            "Every taxonomy class has a downloadable source."),
    }


# --------------------------------------------------------------------------
# YOLO dataset remapping / merging
# --------------------------------------------------------------------------

def read_yolo_names(dataset_yaml: str) -> list[str]:
    """Read the ``names`` list from a YOLO ``dataset.yaml`` (dict or list form)."""
    import yaml  # PyYAML is already a runtime dependency

    with open(dataset_yaml, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    names = data.get("names")
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
    if isinstance(names, list):
        return [str(n) for n in names]
    raise ValueError(f"{dataset_yaml}: no usable 'names' entry")


def build_index_map(source_names: Sequence[str],
                    label_map: Optional[Mapping[str, str]] = None
                    ) -> tuple[dict[int, int], list[str]]:
    """Map source class indices onto canonical taxonomy indices.

    Returns ``(index_map, unmapped_names)``. A source class that cannot be
    resolved is *omitted* from the map, and its instances are dropped with a
    count — silently folding an unknown class into a known one is how label noise
    gets baked into a model.
    """
    lm = {str(k).strip().lower(): v for k, v in (label_map or {}).items()}
    canon_idx = tax.class_index()
    out: dict[int, int] = {}
    unmapped: list[str] = []
    for i, name in enumerate(source_names):
        explicit = lm.get(str(name).strip().lower())
        cid = explicit or tax.resolve(name)
        if cid and cid in canon_idx:
            out[i] = canon_idx[cid]
        else:
            unmapped.append(str(name))
    return out, unmapped


def remap_yolo_dataset(src_root: str, dst_root: str,
                       source_names: Sequence[str],
                       label_map: Optional[Mapping[str, str]] = None,
                       splits: Sequence[str] = ("train", "val", "test"),
                       copy_images: bool = True,
                       prefix: str = "") -> dict:
    """Rewrite a YOLO dataset onto the canonical label space."""
    index_map, unmapped = build_index_map(source_names, label_map)
    stats = {"images": 0, "instances_kept": 0, "instances_dropped": 0,
             "dropped_by_class": Counter(), "unmapped_source_classes": unmapped,
             "per_class": Counter()}
    inv = {v: k for k, v in tax.class_index().items()}

    for split in splits:
        s_img = os.path.join(src_root, "images", split)
        s_lbl = os.path.join(src_root, "labels", split)
        if not os.path.isdir(s_img):
            continue
        d_img = os.path.join(dst_root, "images", split)
        d_lbl = os.path.join(dst_root, "labels", split)
        os.makedirs(d_img, exist_ok=True)
        os.makedirs(d_lbl, exist_ok=True)

        for fn in sorted(os.listdir(s_img)):
            if not fn.lower().endswith(IMAGE_EXTS):
                continue
            stem, ext = os.path.splitext(fn)
            out_stem = f"{prefix}{stem}" if prefix else stem
            lbl_path = os.path.join(s_lbl, stem + ".txt")
            kept_lines: list[str] = []
            if os.path.exists(lbl_path):
                with open(lbl_path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        try:
                            src_cls = int(float(parts[0]))
                        except ValueError:
                            continue
                        if src_cls not in index_map:
                            stats["instances_dropped"] += 1
                            name = (source_names[src_cls]
                                    if src_cls < len(source_names) else str(src_cls))
                            stats["dropped_by_class"][name] += 1
                            continue
                        new_cls = index_map[src_cls]
                        kept_lines.append(" ".join([str(new_cls)] + parts[1:5]))
                        stats["instances_kept"] += 1
                        stats["per_class"][inv[new_cls]] += 1
            with open(os.path.join(d_lbl, out_stem + ".txt"), "w",
                      encoding="utf-8") as fh:
                fh.write("\n".join(kept_lines) + ("\n" if kept_lines else ""))
            dst_img = os.path.join(d_img, out_stem + ext)
            if copy_images:
                shutil.copy2(os.path.join(s_img, fn), dst_img)
            else:
                if os.path.lexists(dst_img):
                    os.remove(dst_img)
                os.symlink(os.path.abspath(os.path.join(s_img, fn)), dst_img)
            stats["images"] += 1

    stats["dropped_by_class"] = dict(stats["dropped_by_class"])
    stats["per_class"] = dict(stats["per_class"])
    write_dataset_yaml(dst_root)
    return stats


def merge(roots: Sequence[str], dst_root: str,
          splits: Sequence[str] = ("train", "val", "test"),
          copy_images: bool = True) -> dict:
    """Union several already-remapped datasets into one.

    File names are prefixed per source so identically-named images from
    different datasets cannot silently overwrite one another — a real and easy
    way to lose half a dataset.
    """
    totals = {"images": 0, "instances": 0, "per_class": Counter(),
              "per_source": {}}
    for n, root in enumerate(roots):
        src_stats = {"images": 0, "instances": 0}
        for split in splits:
            s_img = os.path.join(root, "images", split)
            s_lbl = os.path.join(root, "labels", split)
            if not os.path.isdir(s_img):
                continue
            d_img = os.path.join(dst_root, "images", split)
            d_lbl = os.path.join(dst_root, "labels", split)
            os.makedirs(d_img, exist_ok=True)
            os.makedirs(d_lbl, exist_ok=True)
            for fn in sorted(os.listdir(s_img)):
                if not fn.lower().endswith(IMAGE_EXTS):
                    continue
                stem, ext = os.path.splitext(fn)
                out_stem = f"s{n}_{stem}"
                if copy_images:
                    shutil.copy2(os.path.join(s_img, fn),
                                 os.path.join(d_img, out_stem + ext))
                else:
                    dst = os.path.join(d_img, out_stem + ext)
                    if os.path.lexists(dst):
                        os.remove(dst)
                    os.symlink(os.path.abspath(os.path.join(s_img, fn)), dst)
                lbl = os.path.join(s_lbl, stem + ".txt")
                lines: list[str] = []
                if os.path.exists(lbl):
                    with open(lbl, "r", encoding="utf-8") as fh:
                        lines = [ln.strip() for ln in fh if ln.strip()]
                with open(os.path.join(d_lbl, out_stem + ".txt"), "w",
                          encoding="utf-8") as fh:
                    fh.write("\n".join(lines) + ("\n" if lines else ""))
                inv = {v: k for k, v in tax.class_index().items()}
                for ln in lines:
                    try:
                        totals["per_class"][inv[int(float(ln.split()[0]))]] += 1
                    except (ValueError, KeyError, IndexError):
                        continue
                totals["images"] += 1
                totals["instances"] += len(lines)
                src_stats["images"] += 1
                src_stats["instances"] += len(lines)
        totals["per_source"][root] = src_stats
    totals["per_class"] = dict(sorted(totals["per_class"].items(),
                                      key=lambda kv: -kv[1]))
    write_dataset_yaml(dst_root)
    return totals


def write_dataset_yaml(root: str) -> str:
    idx = tax.class_index()
    path = os.path.join(root, "dataset.yaml")
    os.makedirs(root, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Canonical label space — rtsp_backend.electrical.taxonomy\n")
        fh.write(f"path: {os.path.abspath(root)}\n")
        fh.write("train: images/train\nval: images/val\n")
        if os.path.isdir(os.path.join(root, "images", "test")):
            fh.write("test: images/test\n")
        fh.write(f"nc: {len(idx)}\nnames:\n")
        for cid, i in sorted(idx.items(), key=lambda kv: kv[1]):
            fh.write(f"  {i}: {cid}\n")
    with open(os.path.join(root, "classes.json"), "w", encoding="utf-8") as fh:
        json.dump({"classes": list(tax.CLASS_ORDER)}, fh, indent=2)
    return path


# --------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------

def label_names(dataset_yaml: Optional[str]) -> Optional[list[str]]:
    """Read the ordered label space a YOLO ``dataset.yaml`` declares.

    A dataset's own yaml is the authoritative record of what its label indices
    mean, and it is not always the taxonomy's. A profile-scoped dataset
    (:func:`training.electrical.profiles.apply`) deliberately remaps its indices
    to ``0..N-1``, so reading its labels through the 54-class taxonomy index
    names every class after the first one wrong.
    """
    if not dataset_yaml or not os.path.exists(dataset_yaml):
        return None
    try:
        import yaml

        with open(dataset_yaml, "r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh) or {}
    except Exception:
        return None
    names = data.get("names")
    if isinstance(names, dict):
        try:
            return [str(names[k]) for k in sorted(names, key=lambda x: int(x))]
        except (TypeError, ValueError):
            return None
    if isinstance(names, list) and names:
        return [str(n) for n in names]
    return None


def _label_index(root: str, names: Optional[Sequence[str]] = None
                 ) -> tuple[dict[int, str], str]:
    """Index → class id map for a dataset root, and where the map came from.

    Preference order is explicit names, then the dataset's own ``dataset.yaml``,
    then the full taxonomy. The source is returned so the caller can report
    which label space the numbers were computed in rather than leaving a reader
    to assume.
    """
    if names:
        return {i: str(n) for i, n in enumerate(names)}, "explicit"
    declared = label_names(os.path.join(root, "dataset.yaml"))
    if declared:
        return {i: n for i, n in enumerate(declared)}, "dataset.yaml"
    return {v: k for k, v in tax.class_index().items()}, "taxonomy"


def analyse_dataset(root: str, splits: Sequence[str] = ("train", "val", "test"),
                    names: Optional[Sequence[str]] = None) -> dict:
    """Per-class instance/image counts and box-size distribution.

    Label indices are interpreted in the dataset's **own** label space — see
    :func:`_label_index`. The result records which space that was, because the
    same label file means different classes in a 54-class taxonomy dataset and
    in an 8-class profile dataset, and a count attached to the wrong class name
    is worse than no count at all.
    """
    inv, label_space = _label_index(root, names)
    per_class: Counter = Counter()
    images_with: defaultdict[str, set] = defaultdict(set)
    sizes: defaultdict[str, list[float]] = defaultdict(list)
    n_images = 0
    per_split: dict[str, int] = {}

    for split in splits:
        lbl_dir = os.path.join(root, "labels", split)
        if not os.path.isdir(lbl_dir):
            continue
        count = 0
        for fn in sorted(os.listdir(lbl_dir)):
            if not fn.endswith(".txt"):
                continue
            count += 1
            n_images += 1
            with open(os.path.join(lbl_dir, fn), "r", encoding="utf-8") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    try:
                        cid = inv[int(float(parts[0]))]
                        w, h = float(parts[3]), float(parts[4])
                    except (ValueError, KeyError):
                        continue
                    per_class[cid] += 1
                    images_with[cid].add(f"{split}/{fn}")
                    sizes[cid].append(w * h)
        per_split[split] = count

    rows = []
    for cid, n in per_class.most_common():
        areas = sizes[cid]
        rows.append({
            "class_id": cid, "name": tax.display_name(cid), "instances": n,
            "images": len(images_with[cid]),
            "mean_rel_area": round(sum(areas) / len(areas), 6) if areas else None,
            "min_rel_area": round(min(areas), 6) if areas else None,
            "max_rel_area": round(max(areas), 6) if areas else None,
        })
    return {"root": root, "images": n_images, "images_per_split": per_split,
            "instances": int(sum(per_class.values())), "per_class": rows,
            "label_space": label_space,
            "label_space_size": len(inv)}


#: Below this many instances a class will not train usefully; below the warn
#: level it will train but stay unreliable. These are working rules of thumb for
#: detection fine-tuning, not guarantees.
MIN_INSTANCES_TRAINABLE = 50
MIN_INSTANCES_RELIABLE = 300


def coverage_report(analysis: Mapping) -> dict:
    """Name the classes that will and will not work, before training starts."""
    counts = {r["class_id"]: r["instances"] for r in analysis.get("per_class", [])}
    reliable, weak, untrainable, absent = [], [], [], []
    for cid in tax.CLASS_ORDER:
        n = counts.get(cid, 0)
        entry = {"class_id": cid, "name": tax.display_name(cid), "instances": n}
        if n == 0:
            absent.append(entry)
        elif n < MIN_INSTANCES_TRAINABLE:
            untrainable.append(entry)
        elif n < MIN_INSTANCES_RELIABLE:
            weak.append(entry)
        else:
            reliable.append(entry)
    return {
        "reliable": reliable, "weak": weak, "untrainable": untrainable,
        "absent": absent,
        "thresholds": {"trainable": MIN_INSTANCES_TRAINABLE,
                       "reliable": MIN_INSTANCES_RELIABLE},
        "summary": (
            f"{len(reliable)} class(es) have enough data to be reliable, "
            f"{len(weak)} will train but stay weak, "
            f"{len(untrainable)} have too few instances, "
            f"{len(absent)} are absent entirely. Detections for anything other "
            f"than the reliable set should be expected to fall through to "
            f"'Unknown Industrial Component'."),
    }


#: The classes the platform is contractually expected to detect — the brief's
#: target list mapped onto canonical taxonomy ids. Everything else in
#: :data:`~rtsp_backend.electrical.taxonomy.CLASS_ORDER` is still trainable and
#: still reported, but the gap report ranks these first because these are the
#: ones that must work in production.
PRIORITY_CLASSES: tuple[str, ...] = (
    "mcb", "mccb", "contactor", "relay", "plc", "power_supply", "vfd", "fuse",
    "terminal_block", "busbar", "push_button", "emergency_stop",
    "selector_switch", "indicator_lamp", "transformer", "current_transformer",
    "circuit_breaker", "timer_relay", "overload_relay", "din_rail",
    "wire_duct", "cooling_fan",
)

#: Mean labelled instances per image observed in real panel photography at row
#: framing. Used to convert an instance shortfall into an image count, because
#: "collect 300 more annotations" is not an instruction anybody can act on
#: whereas "photograph 25 more panels" is.
INSTANCES_PER_IMAGE_ESTIMATE = 12.0


def requirements_report(analysis: Optional[Mapping] = None,
                        target: int = MIN_INSTANCES_RELIABLE,
                        priority_only: bool = False) -> dict:
    """State the dataset shortfall in units somebody can act on.

    Answers the four questions that matter before committing to a training run:
    which classes are missing, how many annotations are still required, how many
    images that implies, and what to go and photograph.

    ``analysis`` is the output of :func:`analyse_dataset`. Passing ``None``
    reports the shortfall against an empty dataset — i.e. the full cost of
    building this from nothing, which is the honest starting position when no
    data has been collected yet.
    """
    counts: dict[str, int] = {}
    if analysis:
        counts = {r["class_id"]: int(r["instances"])
                  for r in analysis.get("per_class", [])}

    classes = (PRIORITY_CLASSES if priority_only else tax.CLASS_ORDER)
    rows: list[dict] = []
    for cid in classes:
        have = counts.get(cid, 0)
        need = max(0, target - have)
        if have == 0:
            status = "missing"
        elif have < MIN_INSTANCES_TRAINABLE:
            status = "untrainable"
        elif have < target:
            status = "weak"
        else:
            status = "ready"
        rows.append({
            "class_id": cid,
            "name": tax.display_name(cid),
            "priority": cid in PRIORITY_CLASSES,
            "have_annotations": have,
            "need_annotations": need,
            "status": status,
        })

    shortfall = sum(r["need_annotations"] for r in rows)
    priority_shortfall = sum(r["need_annotations"] for r in rows
                             if r["priority"])
    # Panel photographs are multi-label: one image of a DIN rail row yields a
    # dozen boxes across several classes. Dividing the total shortfall by the
    # observed instances-per-image gives the image count; the per-class figure is
    # a worst case that assumes no co-occurrence.
    images_needed = int(round(shortfall / INSTANCES_PER_IMAGE_ESTIMATE))
    priority_images_needed = int(round(priority_shortfall
                                       / INSTANCES_PER_IMAGE_ESTIMATE))

    missing = [r for r in rows if r["status"] == "missing"]
    untrainable = [r for r in rows if r["status"] == "untrainable"]
    weak = [r for r in rows if r["status"] == "weak"]
    ready = [r for r in rows if r["status"] == "ready"]

    return {
        "target_instances_per_class": target,
        "thresholds": {"trainable": MIN_INSTANCES_TRAINABLE,
                       "reliable": MIN_INSTANCES_RELIABLE},
        "instances_per_image_estimate": INSTANCES_PER_IMAGE_ESTIMATE,
        "per_class": rows,
        "missing_classes": [r["class_id"] for r in missing],
        "untrainable_classes": [r["class_id"] for r in untrainable],
        "weak_classes": [r["class_id"] for r in weak],
        "ready_classes": [r["class_id"] for r in ready],
        "annotations_required": shortfall,
        "images_required": images_needed,
        "priority": {
            "classes": list(PRIORITY_CLASSES),
            "missing_classes": [r["class_id"] for r in missing if r["priority"]],
            "annotations_required": priority_shortfall,
            "images_required": priority_images_needed,
        },
        "what_to_collect": _collection_advice(missing + untrainable + weak),
        "summary": (
            f"{len(ready)}/{len(rows)} class(es) reach {target} annotations. "
            f"{len(missing)} have none at all, {len(untrainable)} are below the "
            f"{MIN_INSTANCES_TRAINABLE}-instance trainability floor, "
            f"{len(weak)} will train but stay unreliable. Closing the gap needs "
            f"~{shortfall} more annotations, which at "
            f"~{INSTANCES_PER_IMAGE_ESTIMATE:g} boxes per panel photograph is "
            f"~{images_needed} more labelled images "
            f"(~{priority_images_needed} if you only close the "
            f"{len(PRIORITY_CLASSES)} priority classes first)."),
    }


def _collection_advice(rows: Sequence[Mapping]) -> list[dict]:
    """Turn a list of short classes into per-class capture instructions."""
    #: Where each hard-to-source class is actually found, so the capture list is
    #: a route around a factory rather than a restatement of the class name.
    where = {
        "vfd": "Drive cabinets and pump/fan rooms. Photograph the drive door "
               "closed (keypad visible) and open. Cover at least three brands "
               "— an ABB ACS, a Siemens G120 and a Delta/Schneider unit look "
               "nothing alike.",
        "power_supply": "Any control panel: the 24 V DIN-rail SMPS beside the "
                        "PLC. Phoenix Contact, Meanwell and Siemens SITOP are "
                        "the three you will meet most.",
        "busbar": "Distribution boards and MCC incomers with the shroud "
                  "removed, plus busbar chambers. Capture both bare copper and "
                  "insulated/sleeved bar — they look completely different.",
        "din_rail": "Every panel, but frame it deliberately: empty rail "
                    "sections and part-populated rails, so the class is not "
                    "learned as 'the gap between devices'.",
        "wire_duct": "Every panel. Include lid-on and lid-off, and ducts "
                     "crossing behind cable bundles.",
        "emergency_stop": "Machine doors, conveyor pull-cords, panel fascias. "
                          "Mushroom heads with and without the yellow "
                          "backplate, and both latched and released.",
        "terminal_block": "Panel gland area and field-wiring rails. Label as "
                          "one box per contiguous STRIP, never per pole.",
        "cooling_fan": "Cabinet side walls and doors, inside and outside, "
                       "filter fitted and removed.",
        "current_transformer": "MCC incomers and metering sections, around "
                               "cables and around busbar.",
        "transformer": "Control transformers on the back plate of larger "
                       "panels; include the terminal shroud on and off.",
        "timer_relay": "Control sections, on DIN rail beside relays. Include "
                       "both electromechanical dial timers and digital ones.",
        "overload_relay": "Directly under contactors in every motor starter. "
                          "Remember: contactor and overload are TWO boxes.",
        "selector_switch": "Panel fascia. 2-position, 3-position and key "
                           "switches, with and without legend plates.",
        "plc": "Automation cabinets. Photograph the whole rack and each module, "
               "with the terminal covers both open and closed.",
        "safety_relay": "Safety sections — Pilz, Sick and Schneider Preventa "
                        "units beside the PLC.",
        "soft_starter": "Motor starter cabinets for larger pumps and "
                        "compressors.",
    }
    generic = ("Photograph in situ at row framing under real panel lighting, "
               "±30° off-axis, across at least three manufacturers.")
    out = []
    for r in rows:
        cid = r["class_id"]
        out.append({
            "class_id": cid,
            "name": tax.display_name(cid),
            "need_annotations": r["need_annotations"],
            "where_to_find_it": where.get(cid, generic),
        })
    return out


def custom_collection_plan() -> dict:
    """The capture protocol for building the proprietary Madkour dataset.

    Written as a checklist an engineer with a phone camera can execute on site.
    """
    return {
        "objective": (
            f"{MIN_INSTANCES_RELIABLE}+ labelled instances per class that must "
            f"work in production, captured under deployment conditions."),
        "capture_protocol": [
            "Photograph every panel at three framings: whole cabinet with the "
            "door open, each device row filling the frame, and a close-up of "
            "every device nameplate. The row framing is what the model will see "
            "in service; the nameplate close-ups train and validate the "
            "part-number reader.",
            "Vary the camera angle deliberately: straight on, and roughly ±30° "
            "horizontally and vertically. A model trained only on square-on "
            "photographs fails the moment an inspector stands to one side.",
            "Capture in the lighting that actually exists — overhead fluorescent, "
            "torch, flash, and backlit through the cabinet window. Do not "
            "correct or normalise it; that variation is the training signal.",
            "Include the panels that look bad: dusty, oil-filmed, with cable "
            "bundles crossing devices, faded labels, mixed manufacturers, "
            "retrofitted devices. Clean panels alone produce a fragile model.",
            "Photograph the same device family from several manufacturers. "
            "Manufacturer-invariance has to be learned from examples; there is "
            "no shortcut.",
            "Record the ground-truth bill of materials per panel from the as-built "
            "drawing. It validates counts and panel-type inference independently "
            "of the boxes, and costs almost nothing to collect at capture time.",
        ],
        "labelling_rules": [
            "One box per physical device, tight to the housing including "
            "terminals but excluding wiring.",
            "An overload relay bolted under a contactor is TWO boxes, not one — "
            "they are separately replaceable devices and separately reported.",
            "Terminal blocks: label contiguous strips as one box per strip, not "
            "per pole. Per-pole labelling produces hundreds of boxes per image "
            "and destroys the class balance.",
            "Structural items (DIN rail, cable duct, busbar) get one box per "
            "continuous run.",
            "If a labeller cannot identify a device with certainty, label it "
            f"'{tax.UNKNOWN_COMPONENT_ID}'. A wrong label is worse than an "
            "honest unknown, and the unknowns become the next capture list.",
            "Two labellers on a 10% sample; measure agreement. Below ~0.85 IoU "
            "agreement the labelling guide needs work, not the model.",
        ],
        "split_policy": (
            "Split by PANEL, never by image. Multiple framings of the same "
            "cabinet in both train and val leaks and inflates every metric — "
            "this is the most common way an industrial detector is reported as "
            "excellent and then fails on site."),
        "multiplication": (
            "Crop every labelled device into a per-class crop library, then use "
            "training.electrical.synthetic.compose_from_crops to multiply it "
            "with real appearance and synthetic arrangement, lighting, "
            "perspective, occlusion and dirt."),
        "acceptance": (
            "A held-out set of complete Madkour panels never seen in training, "
            "scored with rtsp_backend.electrical.metrics: per-class recall and "
            "precision, plus bill-of-materials count accuracy against the "
            "as-built drawing."),
    }


__all__ = [
    "DatasetSource", "SOURCES", "SOURCE_INDEX", "plan", "read_yolo_names",
    "build_index_map", "remap_yolo_dataset", "merge", "write_dataset_yaml",
    "analyse_dataset", "label_names", "coverage_report",
    "custom_collection_plan",
    "requirements_report", "PRIORITY_CLASSES", "INSTANCES_PER_IMAGE_ESTIMATE",
    "MIN_INSTANCES_TRAINABLE", "MIN_INSTANCES_RELIABLE", "IMAGE_EXTS",
]
