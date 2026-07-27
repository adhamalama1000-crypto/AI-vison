"""
Industrial electrical component taxonomy — the domain knowledge base.

This module is the reason the system can behave like an electrical engineer
rather than a generic object detector. Every recognisable component is declared
once, here, together with everything the rest of the pipeline needs to reason
about it:

* **identity** — canonical id, display name, aliases used by public datasets
  (so heterogeneous datasets can be merged onto one label space);
* **engineering knowledge** — what the device *does*, the role it plays in a
  panel, its electrical domain (power / control / signal);
* **geometric priors** — plausible aspect-ratio band and plausible fraction of
  the panel image it may occupy, plus how it is mounted (DIN rail, back plate,
  door, busbar). ``postprocess`` uses these to reject boxes that cannot
  physically be that device — this is what kills the "random region = wire"
  class of false positive;
* **open-vocabulary prompts** — natural-language queries for zero-shot
  detectors (OWLv2 / Grounding DINO / Florence-2), phrased the way an engineer
  would describe the device so a text-conditioned model can find it without a
  custom-trained checkpoint;
* **per-class acceptance threshold** — small, visually ambiguous devices need a
  higher bar than a large obvious one.

Nothing here performs inference and nothing here fabricates a detection. It is
pure, declarative domain knowledge and is fully unit-tested.

Design rules
------------
1. ``CLASS_ORDER`` is the canonical training label order. **Append only** —
   inserting in the middle invalidates every previously trained checkpoint.
2. Aliases are matched case-insensitively after normalisation, so
   ``"Circuit Breaker"``, ``"circuit-breaker"`` and ``"circuit_breaker"`` all
   resolve to the same class.
3. A component the model is not confident about must become
   :data:`UNKNOWN_COMPONENT_ID`, never a guess.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

# --------------------------------------------------------------------------
# Categories & electrical domains
# --------------------------------------------------------------------------

#: Broad functional groups, used for grouping in reports and the UI.
CATEGORIES = (
    "protection",       # breakers, fuses, RCDs — interrupt fault current
    "switching",        # contactors, relays, starters — make/break load circuits
    "control",          # timers, monitoring/safety relays, logic modules
    "automation",       # PLCs, IO modules, motion controllers
    "hmi",              # operator interfaces & panel-mounted controls
    "drives",           # VFDs, soft starters, servo drives
    "power",            # supplies, transformers, UPS
    "instrumentation",  # meters, CTs/VTs, sensors, encoders
    "network",          # ethernet switches, routers, gateways
    "infrastructure",   # DIN rail, ducts, busbars, terminal blocks
    "cooling",          # fans, thermostats, filters
)

#: Electrical domain a device belongs to. Used by the panel classifier to
#: distinguish a power distribution board from a control/automation cabinet.
DOMAINS = ("power", "control", "signal", "mixed", "passive")

#: Label emitted when a detection is real but the class is not certain.
UNKNOWN_COMPONENT_ID = "unknown_industrial_component"
UNKNOWN_COMPONENT_NAME = "Unknown Industrial Component"


@dataclass(frozen=True)
class ComponentSpec:
    """Everything the platform knows about one class of industrial device."""

    id: str
    name: str
    category: str
    domain: str
    function: str
    role: str
    #: Mounting styles: din_rail | backplate | door | busbar | duct | enclosure
    mounting: tuple[str, ...]
    #: Plausible width/height ratio band for a correctly-framed detection.
    aspect_ratio: tuple[float, float]
    #: Plausible box area as a fraction of the full panel image area.
    rel_area: tuple[float, float]
    #: Natural-language queries for open-vocabulary / text-conditioned models.
    prompts: tuple[str, ...]
    #: Minimum confidence for this class to be accepted as itself.
    min_conf: float = 0.35
    #: Dataset / vendor synonyms that map onto this class.
    aliases: tuple[str, ...] = ()
    #: Devices that usually accompany this one — used for "possible missing
    #: components" reasoning, never to invent a detection.
    companions: tuple[str, ...] = ()
    #: Free-form engineering notes surfaced in the expert analysis.
    notes: str = ""

    @property
    def is_infrastructure(self) -> bool:
        return self.category == "infrastructure"


def _spec(*args, **kwargs) -> ComponentSpec:
    return ComponentSpec(*args, **kwargs)


# --------------------------------------------------------------------------
# The catalogue
# --------------------------------------------------------------------------
# Aspect-ratio and relative-area bands below are deliberately generous: they
# exist to reject the physically impossible (a 60:1 sliver claimed to be a PLC,
# a box covering 80% of the cabinet claimed to be an indicator lamp), not to
# second-guess a well-trained detector. They were chosen from the geometry of
# real devices photographed at typical panel-inspection framing (whole cabinet
# or one section in view).

_SPECS: tuple[ComponentSpec, ...] = (
    # ---------------- protection ----------------
    _spec(
        id="mcb", name="MCB (Miniature Circuit Breaker)", category="protection",
        domain="power",
        function="Interrupts the circuit on overload or short circuit, protecting "
                 "downstream cable and equipment. Thermal element handles sustained "
                 "overload, magnetic element handles short-circuit current.",
        role="Final-circuit protection for control supplies, lighting and small loads.",
        mounting=("din_rail",), aspect_ratio=(0.12, 1.6), rel_area=(0.0004, 0.06),
        prompts=("a miniature circuit breaker on a DIN rail",
                 "a small modular circuit breaker with a toggle lever",
                 "an MCB with an ON/OFF trip lever"),
        min_conf=0.40,
        aliases=("miniature_circuit_breaker", "circuit_breaker_small", "breaker_mcb",
                 "mcb_1p", "mcb_2p", "mcb_3p", "modular_breaker"),
        companions=("din_rail", "terminal_block"),
        notes="Modular width (typically 17.5–18 mm per pole) makes pole count "
              "readable from the box width relative to neighbouring modules.",
    ),
    _spec(
        id="mccb", name="MCCB (Moulded Case Circuit Breaker)", category="protection",
        domain="power",
        function="Moulded-case breaker with adjustable thermal-magnetic or "
                 "electronic trip unit, used for feeder and motor-circuit "
                 "protection at higher current ratings than an MCB.",
        role="Incoming or feeder protection; motor feeder protection.",
        mounting=("backplate", "din_rail"), aspect_ratio=(0.35, 2.2),
        rel_area=(0.004, 0.22),
        prompts=("a moulded case circuit breaker",
                 "a large black industrial circuit breaker with a rotary handle",
                 "an MCCB with an adjustable trip dial"),
        min_conf=0.40,
        aliases=("moulded_case_circuit_breaker", "molded_case_circuit_breaker",
                 "circuit_breaker_large", "breaker_mccb"),
        companions=("busbar", "current_transformer"),
    ),
    _spec(
        id="acb", name="ACB (Air Circuit Breaker)", category="protection",
        domain="power",
        function="Draw-out air circuit breaker with an electronic trip unit for "
                 "main incoming protection at high current ratings, with "
                 "adjustable long-time / short-time / instantaneous / earth-fault "
                 "protection settings.",
        role="Main incomer or tie breaker in an LV switchboard.",
        mounting=("backplate", "enclosure"), aspect_ratio=(0.5, 2.0),
        rel_area=(0.02, 0.45),
        prompts=("an air circuit breaker in a switchboard",
                 "a large draw-out circuit breaker with a digital trip unit",
                 "an ACB with a charging handle and ON/OFF push buttons"),
        min_conf=0.40,
        aliases=("air_circuit_breaker", "power_circuit_breaker"),
        companions=("current_transformer", "busbar", "energy_meter"),
    ),
    _spec(
        id="rccb", name="RCCB (Residual Current Circuit Breaker)",
        category="protection", domain="power",
        function="Detects residual (earth-leakage) current and disconnects the "
                 "circuit, protecting people against electric shock. Provides no "
                 "overcurrent protection on its own.",
        role="Earth-leakage protection for socket and personnel circuits.",
        mounting=("din_rail",), aspect_ratio=(0.3, 2.0), rel_area=(0.001, 0.08),
        prompts=("a residual current circuit breaker with a test button",
                 "an RCCB earth leakage protection device on a DIN rail"),
        min_conf=0.45,
        aliases=("residual_current_circuit_breaker", "rcd", "elcb", "earth_leakage"),
        companions=("mcb",),
        notes="Distinguished from an RCBO by having no overcurrent trip curve "
              "marking; usually carries a prominent TEST button.",
    ),
    _spec(
        id="rcbo", name="RCBO (Residual Current Breaker with Overcurrent)",
        category="protection", domain="power",
        function="Combines earth-leakage detection with overcurrent protection in "
                 "one modular device.",
        role="Combined personnel and circuit protection on a final circuit.",
        mounting=("din_rail",), aspect_ratio=(0.2, 1.8), rel_area=(0.001, 0.07),
        prompts=("an RCBO combined earth leakage and overcurrent breaker",
                 "a modular breaker with both a trip curve marking and a test button"),
        min_conf=0.48,
        aliases=("residual_current_breaker_overcurrent",),
        companions=("din_rail",),
    ),
    _spec(
        id="fuse", name="Fuse", category="protection", domain="power",
        function="Sacrificial overcurrent protection element; the fuse link melts "
                 "to interrupt fault current.",
        role="Semiconductor, transformer or control-circuit protection.",
        mounting=("din_rail", "backplate"), aspect_ratio=(0.15, 3.0),
        rel_area=(0.0002, 0.03),
        prompts=("an electrical fuse", "a cylindrical cartridge fuse",
                 "an industrial NH/HRC fuse link"),
        min_conf=0.45,
        aliases=("hrc_fuse", "cartridge_fuse", "nh_fuse", "fuse_link"),
        companions=("fuse_holder",),
    ),
    _spec(
        id="fuse_holder", name="Fuse Holder / Disconnector", category="protection",
        domain="power",
        function="Holds and isolates a fuse link, allowing safe replacement; often "
                 "with a blown-fuse indicator.",
        role="Fused isolation of a feeder or control circuit.",
        mounting=("din_rail",), aspect_ratio=(0.15, 2.0), rel_area=(0.0004, 0.04),
        prompts=("a DIN rail fuse holder with a hinged fuse carrier",
                 "a fuse disconnector switch"),
        min_conf=0.45,
        aliases=("fuse_base", "fuse_carrier", "fuse_switch_disconnector"),
        companions=("fuse",),
    ),
    _spec(
        id="surge_protector", name="Surge Protection Device (SPD)",
        category="protection", domain="power",
        function="Diverts transient overvoltage (lightning, switching surges) to "
                 "earth, protecting sensitive electronics. Cartridges carry a "
                 "green/red end-of-life status window.",
        role="Transient overvoltage protection at the incomer or on control supplies.",
        mounting=("din_rail",), aspect_ratio=(0.2, 1.8), rel_area=(0.001, 0.06),
        prompts=("a surge protection device with plug-in cartridges",
                 "an SPD with a green and red status window"),
        min_conf=0.48,
        aliases=("spd", "surge_arrester", "lightning_arrester"),
        companions=("earth_bar",),
    ),

    # ---------------- switching ----------------
    _spec(
        id="contactor", name="Contactor", category="switching", domain="power",
        function="Electromagnetically operated switch that makes and breaks the "
                 "three-phase load circuit under coil control, rated for repeated "
                 "switching of motor inrush current.",
        role="Motor and heater switching element of a starter.",
        mounting=("din_rail", "backplate"), aspect_ratio=(0.35, 2.2),
        rel_area=(0.002, 0.14),
        prompts=("an electrical contactor",
                 "a three phase motor contactor with a coil and auxiliary contacts",
                 "a black contactor block with three power terminals on top"),
        min_conf=0.38,
        aliases=("magnetic_contactor", "power_contactor", "motor_contactor",
                 "contactor_3p", "ac_contactor"),
        companions=("overload_relay", "mcb", "push_button"),
        notes="Three top and three bottom power terminals plus a coil (A1/A2) is "
              "the strongest visual signature; a stacked auxiliary contact block "
              "or a mechanically-linked pair indicates a reversing starter.",
    ),
    _spec(
        id="relay", name="Control Relay", category="switching", domain="control",
        function="Low-power electromechanical switch that isolates and multiplies "
                 "control signals, typically plug-in on a socket base.",
        role="Interposing between PLC outputs and field devices.",
        mounting=("din_rail",), aspect_ratio=(0.25, 2.2), rel_area=(0.0004, 0.05),
        prompts=("a plug-in control relay on a socket base",
                 "a small transparent-cased electromechanical relay",
                 "an interface relay module on a DIN rail"),
        min_conf=0.40,
        aliases=("control_relay", "interface_relay", "plug_in_relay",
                 "miniature_relay", "ice_cube_relay", "auxiliary_relay"),
        companions=("terminal_block", "plc"),
    ),
    _spec(
        id="safety_relay", name="Safety Relay", category="control", domain="control",
        function="Redundant, self-monitoring relay that evaluates emergency-stop, "
                 "guard-door and light-curtain circuits and provides a certified "
                 "safe stop up to the required performance level.",
        role="Functional-safety element of the emergency-stop circuit.",
        mounting=("din_rail",), aspect_ratio=(0.2, 2.0), rel_area=(0.001, 0.07),
        prompts=("a safety relay module with dual channel inputs",
                 "an emergency stop safety relay on a DIN rail",
                 "a yellow safety relay module"),
        min_conf=0.45,
        aliases=("safety_module", "emergency_stop_relay", "safety_controller"),
        companions=("emergency_stop", "contactor"),
        notes="A safety relay without a matching emergency-stop device in the "
              "panel or on the door is an inspection finding worth raising.",
    ),
    _spec(
        id="timer_relay", name="Timer Relay", category="control", domain="control",
        function="Relay with an adjustable time function (on-delay, off-delay, "
                 "star-delta, interval) used to sequence control operations.",
        role="Star-delta changeover timing, sequencing, delayed starts.",
        mounting=("din_rail",), aspect_ratio=(0.25, 2.0), rel_area=(0.0006, 0.05),
        prompts=("a timer relay with a time setting dial",
                 "an adjustable on-delay timing relay on a DIN rail",
                 "a timer module with a rotary scale in seconds"),
        min_conf=0.45,
        aliases=("timer", "time_relay", "timing_relay", "on_delay_timer",
                 "star_delta_timer"),
        companions=("contactor",),
    ),
    _spec(
        id="overload_relay", name="Thermal Overload Relay", category="protection",
        domain="power",
        function="Monitors motor current with bimetallic or electronic elements "
                 "and trips the contactor coil on sustained overload, protecting "
                 "the motor winding.",
        role="Motor thermal protection, mounted directly under a contactor.",
        mounting=("din_rail", "backplate"), aspect_ratio=(0.4, 2.4),
        rel_area=(0.001, 0.09),
        prompts=("a thermal overload relay mounted under a contactor",
                 "a motor overload relay with a current setting dial and reset button",
                 "an electronic overload relay with TEST and RESET buttons"),
        min_conf=0.42,
        aliases=("thermal_overload_relay", "thermal_relay", "motor_protection_relay",
                 "overload", "olr"),
        companions=("contactor",),
        notes="Physically bolted to the contactor bottom terminals — a detection "
              "whose box does not sit directly below a contactor is suspicious.",
    ),
    _spec(
        id="motor_starter", name="Motor Starter / Manual Motor Starter",
        category="switching", domain="power",
        function="Combined switching and protection unit for a motor, either a "
                 "manual motor starter with rotary handle or an assembled "
                 "contactor + overload combination.",
        role="Complete motor branch circuit in one device.",
        mounting=("din_rail", "backplate"), aspect_ratio=(0.3, 2.2),
        rel_area=(0.002, 0.13),
        prompts=("a manual motor starter with a rotary handle",
                 "a motor protection circuit breaker",
                 "a combined motor starter unit"),
        min_conf=0.45,
        aliases=("manual_motor_starter", "motor_protection_circuit_breaker",
                 "mpcb", "dol_starter"),
        companions=("contactor",),
    ),
    _spec(
        id="changeover_switch", name="Changeover / Transfer Switch",
        category="switching", domain="power",
        function="Transfers a load between two supplies (mains/generator) either "
                 "manually or automatically, with mechanical interlocking to "
                 "prevent paralleling.",
        role="Source selection in an ATS or manual changeover panel.",
        mounting=("backplate", "door"), aspect_ratio=(0.4, 2.2),
        rel_area=(0.004, 0.25),
        prompts=("an automatic transfer switch",
                 "a changeover switch with I-0-II positions",
                 "a motorised source transfer switch"),
        min_conf=0.45,
        aliases=("ats", "transfer_switch", "changeover", "source_changeover"),
        companions=("ats_controller", "acb", "mccb"),
    ),
    _spec(
        id="ats_controller", name="ATS / Generator Controller", category="control",
        domain="control",
        function="Monitors mains and generator voltage/frequency and commands the "
                 "transfer sequence, generator start and load transfer with "
                 "adjustable timers.",
        role="Brain of an automatic transfer switch or generator control panel.",
        mounting=("door",), aspect_ratio=(0.5, 2.2), rel_area=(0.003, 0.16),
        prompts=("an automatic transfer switch controller with an LCD display",
                 "a generator control module with mains and generator LEDs"),
        min_conf=0.48,
        aliases=("genset_controller", "amf_controller", "transfer_controller"),
        companions=("changeover_switch",),
    ),

    # ---------------- automation ----------------
    _spec(
        id="plc", name="PLC (Programmable Logic Controller)", category="automation",
        domain="control",
        function="Industrial controller executing a cyclic user program, reading "
                 "field inputs and driving outputs deterministically; the logic "
                 "engine of an automated process.",
        role="Primary automation controller of the cabinet.",
        mounting=("din_rail", "backplate"), aspect_ratio=(0.3, 3.2),
        rel_area=(0.003, 0.25),
        prompts=("a programmable logic controller",
                 "a PLC CPU module with status LEDs and an ethernet port",
                 "a modular industrial controller mounted on a DIN rail"),
        min_conf=0.38,
        aliases=("programmable_logic_controller", "plc_cpu", "cpu_module",
                 "controller", "plc_module"),
        companions=("io_module", "power_supply", "hmi", "ethernet_switch"),
        notes="A CPU is identified by RUN/STOP/ERROR LEDs, a programming port and "
              "usually a removable terminal or bus connector on the right edge "
              "for expansion modules.",
    ),
    _spec(
        id="io_module", name="I/O Module", category="automation", domain="signal",
        function="Expansion module providing digital or analogue input/output "
                 "channels to the controller over the backplane or a fieldbus.",
        role="Field signal interface for the PLC.",
        mounting=("din_rail",), aspect_ratio=(0.15, 2.6), rel_area=(0.001, 0.12),
        prompts=("a PLC input output expansion module",
                 "a digital input module with channel status LEDs",
                 "an analog IO module with screw terminals"),
        min_conf=0.45,
        aliases=("io_card", "digital_input_module", "digital_output_module",
                 "analog_module", "expansion_module", "remote_io"),
        companions=("plc",),
    ),
    _spec(
        id="logic_module", name="Compact Logic Module / Smart Relay",
        category="automation", domain="control",
        function="Small programmable relay with an integrated display and keypad "
                 "for simple sequencing where a full PLC is not justified.",
        role="Standalone control of a small machine or building service.",
        mounting=("din_rail",), aspect_ratio=(0.4, 2.4), rel_area=(0.002, 0.10),
        prompts=("a compact programmable logic relay with a small display and buttons",
                 "a smart relay logic module on a DIN rail"),
        min_conf=0.48,
        aliases=("smart_relay", "programmable_relay", "nano_plc"),
        companions=("power_supply",),
    ),
    _spec(
        id="signal_isolator", name="Signal Isolator / Conditioner",
        category="instrumentation", domain="signal",
        function="Galvanically isolates and converts analogue process signals "
                 "(4–20 mA, 0–10 V, RTD, thermocouple) to protect the controller "
                 "and remove ground loops.",
        role="Analogue signal conditioning between field and controller.",
        mounting=("din_rail",), aspect_ratio=(0.1, 1.6), rel_area=(0.0004, 0.04),
        prompts=("a signal isolator module on a DIN rail",
                 "a narrow analog signal converter with terminal screws"),
        min_conf=0.50,
        aliases=("signal_conditioner", "isolation_amplifier", "signal_converter",
                 "transducer_module"),
        companions=("io_module",),
    ),

    # ---------------- HMI & operator interface ----------------
    _spec(
        id="hmi", name="HMI (Human-Machine Interface)", category="hmi",
        domain="control",
        function="Touch-screen operator terminal visualising process state, "
                 "alarms and setpoints, communicating with the controller over a "
                 "fieldbus or ethernet.",
        role="Operator interface, usually door-mounted.",
        mounting=("door",), aspect_ratio=(0.7, 2.4), rel_area=(0.01, 0.45),
        prompts=("an industrial HMI touch screen panel",
                 "an operator interface display mounted on a control panel door",
                 "a machine touch panel with a graphical process screen"),
        min_conf=0.40,
        aliases=("touch_panel", "operator_panel", "hmi_panel", "display_panel",
                 "touchscreen"),
        companions=("plc",),
    ),
    _spec(
        id="push_button", name="Push Button", category="hmi", domain="control",
        function="Momentary operator command device (start, stop, reset, "
                 "acknowledge) with one or more contact blocks.",
        role="Manual command input on the panel door.",
        mounting=("door",), aspect_ratio=(0.45, 2.2), rel_area=(0.0002, 0.02),
        prompts=("a push button on a control panel door",
                 "a green start push button", "a red stop push button",
                 "a round illuminated push button"),
        min_conf=0.42,
        aliases=("pushbutton", "button", "start_button", "stop_button",
                 "momentary_switch"),
        companions=("indicator_lamp",),
    ),
    _spec(
        id="emergency_stop", name="Emergency Stop", category="hmi", domain="control",
        function="Mushroom-head latching device that removes power from hazardous "
                 "motion through the safety circuit; must be manually released "
                 "and cannot be defeated.",
        role="Mandatory safety command device.",
        mounting=("door",), aspect_ratio=(0.5, 2.0), rel_area=(0.0004, 0.035),
        prompts=("a red mushroom head emergency stop button on a yellow background",
                 "an emergency stop push button",
                 "a large red latching mushroom button"),
        min_conf=0.45,
        aliases=("e_stop", "estop", "emergency_button", "mushroom_button"),
        companions=("safety_relay",),
        notes="A red mushroom head on a yellow backing plate is the IEC 60204-1 "
              "signature; absence of a yellow backing is an inspection finding.",
    ),
    _spec(
        id="selector_switch", name="Selector Switch", category="hmi",
        domain="control",
        function="Maintained multi-position rotary command device selecting a "
                 "mode of operation (e.g. Manual / Off / Auto).",
        role="Operating-mode selection.",
        mounting=("door",), aspect_ratio=(0.45, 2.2), rel_area=(0.0002, 0.02),
        prompts=("a rotary selector switch on a control panel",
                 "a two position selector switch with a black knob",
                 "a manual off auto selector switch"),
        min_conf=0.45,
        aliases=("rotary_switch", "mode_switch", "selector", "cam_switch"),
        companions=("push_button",),
    ),
    _spec(
        id="indicator_lamp", name="Indicator Lamp", category="hmi",
        domain="control",
        function="Signals a discrete state (running, tripped, power available) to "
                 "the operator; typically an LED pilot light.",
        role="State annunciation on the panel door.",
        mounting=("door",), aspect_ratio=(0.5, 2.0), rel_area=(0.0001, 0.015),
        prompts=("an indicator pilot lamp on a control panel",
                 "a small round LED signal light",
                 "a green running indicator light"),
        min_conf=0.45,
        aliases=("pilot_lamp", "signal_lamp", "indicator_light", "led_indicator",
                 "signal_light"),
        companions=("push_button",),
    ),
    _spec(
        id="ammeter", name="Analogue Meter (Ammeter / Voltmeter)",
        category="instrumentation", domain="power",
        function="Panel-mounted moving-iron or digital meter displaying line "
                 "current or voltage, fed from a CT or directly from the busbar.",
        role="Local electrical measurement display.",
        mounting=("door",), aspect_ratio=(0.6, 1.8), rel_area=(0.0006, 0.05),
        prompts=("a panel mounted analog ammeter",
                 "a square voltmeter gauge on a control panel door"),
        min_conf=0.48,
        aliases=("voltmeter", "analog_meter", "panel_meter"),
        companions=("current_transformer", "selector_switch"),
    ),

    # ---------------- drives ----------------
    _spec(
        id="vfd", name="VFD (Variable Frequency Drive)", category="drives",
        domain="power",
        function="Converts fixed-frequency mains to a variable voltage and "
                 "frequency output, controlling motor speed and torque with soft "
                 "start, ramping and energy savings.",
        role="Speed control of pumps, fans, conveyors and compressors.",
        mounting=("backplate", "din_rail"), aspect_ratio=(0.25, 1.8),
        rel_area=(0.006, 0.35),
        prompts=("a variable frequency drive",
                 "a motor inverter drive with a keypad and cooling vents",
                 "a VFD with a small display and up down arrow keys"),
        min_conf=0.38,
        aliases=("variable_frequency_drive", "inverter", "frequency_converter",
                 "vsd", "motor_drive", "ac_drive"),
        companions=("mccb", "line_reactor", "cooling_fan", "plc"),
        notes="Tall portrait housing with ventilation grilles top and bottom and "
              "a removable keypad is the strongest signature; heat generation "
              "means an adjacent cooling fan is expected.",
    ),
    _spec(
        id="soft_starter", name="Soft Starter", category="drives", domain="power",
        function="Thyristor-based starter that ramps motor voltage to limit "
                 "starting current and mechanical shock, then bypasses to a "
                 "contactor at full speed.",
        role="Reduced-inrush starting where variable speed is not required.",
        mounting=("backplate", "din_rail"), aspect_ratio=(0.3, 1.9),
        rel_area=(0.005, 0.3),
        prompts=("a motor soft starter",
                 "a thyristor soft start unit with heat sink fins"),
        min_conf=0.45,
        aliases=("softstarter", "soft_start", "reduced_voltage_starter"),
        companions=("contactor", "mccb"),
    ),
    _spec(
        id="servo_drive", name="Servo Drive", category="drives", domain="power",
        function="Closed-loop drive controlling a servo motor's position, speed "
                 "and torque from encoder feedback, used where precise motion is "
                 "required.",
        role="Motion axis control in machine automation.",
        mounting=("backplate",), aspect_ratio=(0.2, 1.6), rel_area=(0.005, 0.28),
        prompts=("a servo drive amplifier",
                 "a motion control servo amplifier with encoder feedback connectors"),
        min_conf=0.48,
        aliases=("servo_amplifier", "motion_drive", "servo_controller"),
        companions=("encoder", "plc"),
    ),
    _spec(
        id="line_reactor", name="Line Reactor / Choke", category="drives",
        domain="power",
        function="Series inductor limiting current rate-of-rise, reducing "
                 "harmonics and protecting a drive's input rectifier.",
        role="Harmonic mitigation and drive input protection.",
        mounting=("backplate",), aspect_ratio=(0.4, 2.2), rel_area=(0.003, 0.16),
        prompts=("a three phase line reactor choke with copper windings",
                 "an iron core inductor mounted in a control panel"),
        min_conf=0.50,
        aliases=("choke", "reactor", "du_dt_filter", "harmonic_filter"),
        companions=("vfd",),
    ),

    # ---------------- power ----------------
    _spec(
        id="power_supply", name="Switch-Mode Power Supply", category="power",
        domain="power",
        function="Converts mains AC to a regulated low-voltage DC rail "
                 "(commonly 24 V DC) for controllers, sensors and relays, with "
                 "overload and short-circuit protection.",
        role="Control-voltage generation for the whole cabinet.",
        mounting=("din_rail", "backplate"), aspect_ratio=(0.2, 2.2),
        rel_area=(0.002, 0.16),
        prompts=("a DIN rail 24V DC switching power supply",
                 "an industrial power supply unit with DC OK LED and output terminals",
                 "a switch mode power supply module in a control panel"),
        min_conf=0.40,
        aliases=("smps", "psu", "dc_power_supply", "24v_power_supply",
                 "switching_power_supply", "power_module"),
        companions=("plc", "mcb", "terminal_block"),
        notes="A 24 V rail with no upstream MCB or fuse is a protection finding.",
    ),
    _spec(
        id="transformer", name="Control / Isolation Transformer", category="power",
        domain="power",
        function="Steps voltage up or down and provides galvanic isolation; a "
                 "control transformer derives a safe control voltage from the "
                 "main supply.",
        role="Control-circuit supply derivation and isolation.",
        mounting=("backplate",), aspect_ratio=(0.4, 2.2), rel_area=(0.004, 0.25),
        prompts=("a control transformer in an electrical panel",
                 "an encapsulated isolation transformer with primary and secondary terminals"),
        min_conf=0.45,
        aliases=("control_transformer", "isolation_transformer", "step_down_transformer"),
        companions=("fuse", "mcb"),
    ),
    _spec(
        id="current_transformer", name="Current Transformer (CT)",
        category="instrumentation", domain="power",
        function="Ring or bar-primary transformer producing a scaled secondary "
                 "current (typically 5 A or 1 A) proportional to the primary, "
                 "feeding meters and protection relays.",
        role="Current measurement and protection input.",
        mounting=("busbar", "backplate"), aspect_ratio=(0.4, 2.2),
        rel_area=(0.0006, 0.07),
        prompts=("a current transformer around a busbar",
                 "a ring type CT with secondary terminals",
                 "a split core current transformer in a panel"),
        min_conf=0.45,
        aliases=("ct", "current_sensor_ct", "measuring_transformer"),
        companions=("energy_meter", "ammeter", "protection_relay"),
        notes="An open-circuited CT secondary is a serious hazard — the presence "
              "of a shorting link or connected meter should be verified.",
    ),
    _spec(
        id="voltage_transformer", name="Voltage Transformer (VT/PT)",
        category="instrumentation", domain="power",
        function="Steps system voltage down to a standard measuring level "
                 "(e.g. 110 V) for metering and protection.",
        role="Voltage measurement input.",
        mounting=("backplate",), aspect_ratio=(0.4, 2.0), rel_area=(0.002, 0.14),
        prompts=("a voltage transformer in a switchboard",
                 "a potential transformer with HV and LV terminals"),
        min_conf=0.50,
        aliases=("vt", "pt", "potential_transformer"),
        companions=("energy_meter",),
    ),
    _spec(
        id="capacitor", name="Power Factor Capacitor", category="power",
        domain="power",
        function="Supplies reactive current to correct lagging power factor, "
                 "reducing demand charges and improving voltage regulation.",
        role="Power-factor correction stage.",
        mounting=("backplate",), aspect_ratio=(0.25, 2.2), rel_area=(0.003, 0.22),
        prompts=("a power factor correction capacitor",
                 "a cylindrical capacitor can in an electrical panel"),
        min_conf=0.48,
        aliases=("pfc_capacitor", "capacitor_bank", "power_capacitor"),
        companions=("contactor", "pf_controller", "line_reactor"),
    ),
    _spec(
        id="pf_controller", name="Power Factor Controller", category="control",
        domain="control",
        function="Measures reactive demand and switches capacitor stages in and "
                 "out to hold a target power factor.",
        role="Automatic control of a capacitor bank.",
        mounting=("door",), aspect_ratio=(0.6, 2.0), rel_area=(0.002, 0.12),
        prompts=("a power factor correction controller with a digital display",
                 "an automatic capacitor bank controller with stage LEDs"),
        min_conf=0.50,
        aliases=("pfc_controller", "reactive_power_controller"),
        companions=("capacitor",),
    ),
    _spec(
        id="ups", name="UPS / Battery Backup", category="power", domain="power",
        function="Maintains the control supply through a mains interruption from "
                 "a battery, allowing a controlled shutdown or ride-through.",
        role="Uninterrupted control-voltage supply.",
        mounting=("din_rail", "backplate"), aspect_ratio=(0.25, 2.2),
        rel_area=(0.004, 0.25),
        prompts=("a DIN rail UPS module with a battery",
                 "an industrial uninterruptible power supply unit"),
        min_conf=0.50,
        aliases=("uninterruptible_power_supply", "battery_module", "dc_ups"),
        companions=("power_supply",),
    ),

    # ---------------- instrumentation ----------------
    _spec(
        id="energy_meter", name="Energy / Power Meter", category="instrumentation",
        domain="power",
        function="Measures voltage, current, power, energy, power factor and "
                 "harmonics, usually with a communications port for the SCADA or "
                 "energy-management system.",
        role="Revenue or sub-metering and power-quality monitoring.",
        mounting=("door", "din_rail"), aspect_ratio=(0.5, 2.0),
        rel_area=(0.001, 0.1),
        prompts=("a digital power meter with a multi line display",
                 "a panel mounted energy meter showing voltage and current",
                 "a multifunction power quality meter"),
        min_conf=0.45,
        aliases=("power_meter", "multifunction_meter", "kwh_meter", "energy_analyzer",
                 "power_analyzer"),
        companions=("current_transformer",),
    ),
    _spec(
        id="protection_relay", name="Protection Relay", category="protection",
        domain="control",
        function="Numerical relay measuring current/voltage and issuing trip "
                 "commands for overcurrent, earth fault, phase failure, "
                 "under/over-voltage and other protection functions.",
        role="Feeder or motor protection intelligence.",
        mounting=("door", "din_rail"), aspect_ratio=(0.4, 2.2),
        rel_area=(0.002, 0.14),
        prompts=("a numerical protection relay with a display and function keys",
                 "a phase failure relay on a DIN rail",
                 "a motor protection relay module"),
        min_conf=0.48,
        aliases=("numerical_relay", "phase_failure_relay", "voltage_monitoring_relay",
                 "monitoring_relay", "earth_fault_relay"),
        companions=("current_transformer",),
    ),
    _spec(
        id="sensor", name="Sensor (Proximity / Photoelectric)",
        category="instrumentation", domain="signal",
        function="Detects presence, position or a process variable and delivers a "
                 "discrete or analogue signal to the controller.",
        role="Field feedback into the control system.",
        mounting=("enclosure", "din_rail"), aspect_ratio=(0.15, 4.0),
        rel_area=(0.0002, 0.04),
        prompts=("an industrial proximity sensor",
                 "a photoelectric sensor with an LED indicator",
                 "a cylindrical inductive sensor"),
        min_conf=0.50,
        aliases=("proximity_sensor", "photoelectric_sensor", "inductive_sensor",
                 "pressure_sensor", "temperature_sensor"),
        companions=("io_module",),
    ),
    _spec(
        id="encoder", name="Encoder", category="instrumentation", domain="signal",
        function="Converts shaft rotation into incremental or absolute position "
                 "pulses for closed-loop speed and position control.",
        role="Motion feedback device.",
        mounting=("enclosure",), aspect_ratio=(0.4, 2.2), rel_area=(0.0006, 0.05),
        prompts=("a rotary encoder with a shaft and cable",
                 "an incremental shaft encoder"),
        min_conf=0.52,
        aliases=("rotary_encoder", "shaft_encoder", "resolver"),
        companions=("servo_drive",),
    ),
    _spec(
        id="limit_switch", name="Limit Switch", category="instrumentation",
        domain="signal",
        function="Mechanically actuated switch confirming that a moving part has "
                 "reached a defined position.",
        role="End-of-travel and position confirmation.",
        mounting=("enclosure",), aspect_ratio=(0.25, 3.0), rel_area=(0.0004, 0.04),
        prompts=("an industrial limit switch with a roller lever",
                 "a mechanical position limit switch"),
        min_conf=0.52,
        aliases=("position_switch", "micro_switch", "travel_switch"),
        companions=("io_module",),
    ),
    _spec(
        id="thermostat", name="Panel Thermostat / Hygrostat", category="cooling",
        domain="control",
        function="Temperature (or humidity) switch that starts the cabinet fan or "
                 "heater to keep the enclosure inside its rated climate window.",
        role="Enclosure climate control.",
        mounting=("din_rail",), aspect_ratio=(0.4, 2.2), rel_area=(0.0004, 0.03),
        prompts=("a DIN rail panel thermostat with a temperature dial",
                 "an enclosure thermostat for a cooling fan"),
        min_conf=0.52,
        aliases=("panel_thermostat", "hygrostat", "temperature_switch"),
        companions=("cooling_fan",),
    ),

    # ---------------- network ----------------
    _spec(
        id="ethernet_switch", name="Industrial Ethernet Switch", category="network",
        domain="signal",
        function="Hardened layer-2 switch connecting controllers, drives, HMIs and "
                 "remote IO on the plant network, with link/activity LEDs per port.",
        role="Control-network backbone inside the cabinet.",
        mounting=("din_rail",), aspect_ratio=(0.2, 3.2), rel_area=(0.001, 0.1),
        prompts=("an industrial ethernet switch on a DIN rail",
                 "a managed network switch with RJ45 ports and link LEDs",
                 "a hardened ethernet switch in a control cabinet"),
        min_conf=0.45,
        aliases=("network_switch", "industrial_switch", "managed_switch", "unmanaged_switch"),
        companions=("plc", "power_supply"),
    ),
    _spec(
        id="industrial_router", name="Industrial Router / Gateway",
        category="network", domain="signal",
        function="Routes between the machine network and a plant/remote network, "
                 "often with VPN, firewall, cellular or protocol-gateway "
                 "functions (Modbus↔Profinet, OPC UA).",
        role="Remote access and protocol translation.",
        mounting=("din_rail",), aspect_ratio=(0.2, 3.0), rel_area=(0.001, 0.09),
        prompts=("an industrial router with antennas on a DIN rail",
                 "a fieldbus protocol gateway module",
                 "a cellular industrial VPN router"),
        min_conf=0.50,
        aliases=("router", "gateway", "protocol_converter", "fieldbus_gateway",
                 "modem"),
        companions=("ethernet_switch",),
    ),

    # ---------------- infrastructure ----------------
    _spec(
        id="terminal_block", name="Terminal Block", category="infrastructure",
        domain="passive",
        function="Provides a screw, spring or push-in connection point where field "
                 "wiring terminates and is distributed inside the panel; grouped "
                 "into labelled rails.",
        role="Field wiring interface.",
        mounting=("din_rail",), aspect_ratio=(0.05, 8.0), rel_area=(0.0002, 0.2),
        prompts=("a row of DIN rail terminal blocks",
                 "grey screw terminal blocks with wire markers",
                 "push in spring terminal strip in a control panel"),
        min_conf=0.42,
        aliases=("terminal", "terminal_strip", "terminal_row", "screw_terminal",
                 "din_terminal", "feed_through_terminal"),
        companions=("din_rail", "wire_duct"),
        notes="Best counted as rows/strips rather than individual poles; the "
              "detector is configured to report contiguous strips.",
    ),
    _spec(
        id="busbar", name="Busbar", category="infrastructure", domain="power",
        function="Rigid copper or aluminium conductor distributing high current "
                 "between the incomer and the outgoing ways with low impedance.",
        role="Main current distribution.",
        mounting=("busbar", "backplate"), aspect_ratio=(0.03, 12.0),
        rel_area=(0.001, 0.3),
        prompts=("a copper busbar in an electrical panel",
                 "insulated busbar system with phase separation",
                 "a bare copper bar bolted to standoff insulators"),
        min_conf=0.48,
        aliases=("bus_bar", "copper_bus", "distribution_bar", "phase_bar"),
        companions=("current_transformer", "mccb"),
    ),
    _spec(
        id="neutral_bar", name="Neutral Bar", category="infrastructure",
        domain="power",
        function="Common connection bar for all neutral conductors, isolated from "
                 "earth in a TN-S system.",
        role="Neutral collection point.",
        mounting=("busbar",), aspect_ratio=(0.05, 12.0), rel_area=(0.0006, 0.12),
        prompts=("a neutral bar with multiple screw terminals",
                 "a blue marked neutral distribution bar"),
        min_conf=0.52,
        aliases=("neutral_link", "n_bar"),
        companions=("earth_bar",),
    ),
    _spec(
        id="earth_bar", name="Earth / Ground Bar", category="infrastructure",
        domain="power",
        function="Common bonding bar to which all protective-earth conductors and "
                 "the enclosure bonding straps connect.",
        role="Protective earthing reference.",
        mounting=("busbar",), aspect_ratio=(0.05, 12.0), rel_area=(0.0006, 0.12),
        prompts=("an earth ground bar in an electrical panel",
                 "a green yellow marked earthing bar with screw terminals"),
        min_conf=0.52,
        aliases=("ground_bar", "pe_bar", "earthing_bar", "earth_link"),
        companions=("neutral_bar",),
        notes="Absence of a visible earth bar, or unbonded door/gland plate, is a "
              "safety finding.",
    ),
    _spec(
        id="din_rail", name="DIN Rail", category="infrastructure", domain="passive",
        function="Standard 35 mm steel mounting rail that carries modular devices "
                 "and defines the horizontal rows of the panel layout.",
        role="Mechanical mounting structure.",
        mounting=("backplate",), aspect_ratio=(2.0, 60.0), rel_area=(0.0004, 0.12),
        prompts=("a 35mm DIN mounting rail in an electrical cabinet",
                 "a perforated metal mounting rail carrying modular devices"),
        min_conf=0.45,
        aliases=("mounting_rail", "top_hat_rail", "rail"),
        companions=("mcb", "terminal_block"),
        notes="Detected rails give the row structure used to describe layout and "
              "to sanity-check which devices can be modular.",
    ),
    _spec(
        id="wire_duct", name="Cable Duct / Wiring Trunk", category="infrastructure",
        domain="passive",
        function="Slotted plastic trunking with a clip-on lid that routes and "
                 "protects internal wiring between device rows.",
        role="Internal cable management.",
        mounting=("backplate",), aspect_ratio=(1.2, 40.0), rel_area=(0.001, 0.2),
        prompts=("a slotted plastic cable duct in a control panel",
                 "grey wiring trunking with a removable cover",
                 "a cable management channel between DIN rails"),
        min_conf=0.45,
        aliases=("cable_duct", "wiring_duct", "trunking", "cable_tray",
                 "cable_channel", "wire_way"),
        companions=("terminal_block",),
    ),
    _spec(
        id="cooling_fan", name="Cooling Fan / Filter Fan", category="cooling",
        domain="power",
        function="Forced-ventilation unit with a dust filter that removes heat "
                 "generated by drives, transformers and supplies to keep the "
                 "enclosure within its rated temperature.",
        role="Enclosure thermal management.",
        mounting=("enclosure", "door"), aspect_ratio=(0.6, 1.7),
        rel_area=(0.001, 0.12),
        prompts=("an enclosure cooling filter fan",
                 "a square axial fan with a grille mounted in a cabinet wall"),
        min_conf=0.45,
        aliases=("filter_fan", "panel_fan", "axial_fan", "exhaust_fan", "fan"),
        companions=("thermostat",),
    ),
    _spec(
        id="cable_gland", name="Cable Gland Plate / Glands",
        category="infrastructure", domain="passive",
        function="Provides strain relief and maintains the enclosure IP rating "
                 "where cables enter the panel.",
        role="Cable entry and sealing.",
        mounting=("enclosure",), aspect_ratio=(0.2, 8.0), rel_area=(0.0004, 0.12),
        prompts=("cable glands on an enclosure gland plate",
                 "a cable entry plate with sealed glands"),
        min_conf=0.55,
        aliases=("gland", "gland_plate", "cable_entry"),
        companions=("terminal_block",),
    ),
)

# --------------------------------------------------------------------------
# Indexes
# --------------------------------------------------------------------------

#: Canonical training label order. **APPEND ONLY.**
CLASS_ORDER: tuple[str, ...] = tuple(s.id for s in _SPECS)

SPECS: dict[str, ComponentSpec] = {s.id: s for s in _SPECS}

#: Unknown is not a trainable class — it is the postprocessor's honest fallback.
UNKNOWN_SPEC = ComponentSpec(
    id=UNKNOWN_COMPONENT_ID, name=UNKNOWN_COMPONENT_NAME,
    category="infrastructure", domain="mixed",
    function="A device was detected inside the panel but the model is not "
             "confident enough about its class to name it. Reported honestly "
             "rather than guessed.",
    role="Requires manual identification or additional training data.",
    mounting=("din_rail", "backplate", "door", "enclosure"),
    aspect_ratio=(0.02, 60.0), rel_area=(0.00005, 0.9),
    prompts=(), min_conf=0.0,
    notes="Feed these crops back into the dataset — they are exactly the "
          "examples the model needs.",
)


def _norm(text: str) -> str:
    """Normalise a label for alias matching: lowercase, non-alnum → '_'."""
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_")


_ALIAS_INDEX: dict[str, str] = {}
for _s in _SPECS:
    _ALIAS_INDEX[_norm(_s.id)] = _s.id
    _ALIAS_INDEX[_norm(_s.name)] = _s.id
    for _a in _s.aliases:
        _ALIAS_INDEX.setdefault(_norm(_a), _s.id)
_ALIAS_INDEX[_norm(UNKNOWN_COMPONENT_ID)] = UNKNOWN_COMPONENT_ID


def resolve(label: str) -> Optional[str]:
    """Map an arbitrary dataset/vendor label onto a canonical class id.

    Returns ``None`` when the label is not part of the taxonomy, so callers can
    decide explicitly (drop it, or map it to the unknown class) instead of a
    silent mislabel.
    """
    if label is None:
        return None
    key = _norm(label)
    if not key:
        return None
    hit = _ALIAS_INDEX.get(key)
    if hit:
        return hit
    if key.endswith("s") and _ALIAS_INDEX.get(key[:-1]):
        return _ALIAS_INDEX[key[:-1]]
    # Loose matching accepts a label that is *more specific* than a known alias
    # ("schneider lc1d contactor" → contactor), longest alias first so
    # "overload relay" resolves to the overload relay rather than to a plain
    # relay. It deliberately does NOT accept the reverse: a label vaguer than
    # any alias ("circuit breaker" — MCB? MCCB? ACB? RCCB?) stays unresolved so
    # the caller reports it as an unknown component instead of picking one.
    for cand in sorted(_ALIAS_INDEX, key=len, reverse=True):
        if len(cand) >= 4 and cand in key:
            return _ALIAS_INDEX[cand]
    return None


def spec(class_id: str) -> ComponentSpec:
    """Return the spec for a class id, or the unknown spec."""
    if class_id == UNKNOWN_COMPONENT_ID:
        return UNKNOWN_SPEC
    return SPECS.get(class_id, UNKNOWN_SPEC)


def display_name(class_id: str) -> str:
    return spec(class_id).name


#: Compact labels for the annotated overlay. A real panel photograph packs
#: dozens of devices side by side, so the full display name ("PLC (Programmable
#: Logic Controller)") overlaps its neighbours and the overlay becomes
#: unreadable. These are the abbreviations an engineer writes on a drawing.
_SHORT_OVERRIDES: dict[str, str] = {
    "relay": "Relay",
    "overload_relay": "OL Relay",
    "motor_starter": "Starter",
    "changeover_switch": "ATS Switch",
    "ats_controller": "ATS Ctrl",
    "protection_relay": "Prot Relay",
    "safety_relay": "Safety Rly",
    "timer_relay": "Timer",
    "signal_isolator": "Isolator",
    "logic_module": "Logic Mod",
    "io_module": "I/O",
    "power_supply": "PSU",
    "ups": "UPS",
    "current_transformer": "CT",
    "voltage_transformer": "VT",
    "energy_meter": "Meter",
    "pf_controller": "PF Ctrl",
    "ethernet_switch": "Eth Switch",
    "industrial_router": "Router",
    "terminal_block": "Terminals",
    "indicator_lamp": "Lamp",
    "selector_switch": "Selector",
    "emergency_stop": "E-Stop",
    "push_button": "Button",
    "surge_protector": "SPD",
    "fuse_holder": "Fuse Hldr",
    "line_reactor": "Reactor",
    "soft_starter": "Soft Start",
    "servo_drive": "Servo",
    "cooling_fan": "Fan",
    "wire_duct": "Duct",
    "cable_gland": "Glands",
    "neutral_bar": "N Bar",
    "earth_bar": "PE Bar",
    "din_rail": "Rail",
    "transformer": "Xfmr",
    "capacitor": "Cap",
    "thermostat": "Thermostat",
    "limit_switch": "Limit Sw",
    "ammeter": "Meter",
    UNKNOWN_COMPONENT_ID: "Unknown",
}


def short_name(class_id: str) -> str:
    """Compact label for overlays and dense tables.

    Falls back to the leading acronym of the display name — "MCB (Miniature
    Circuit Breaker)" becomes "MCB" — which is what the parenthetical form was
    designed for.
    """
    if class_id in _SHORT_OVERRIDES:
        return _SHORT_OVERRIDES[class_id]
    name = spec(class_id).name
    head = name.split("(")[0].strip()
    return head or name


def by_category(category: str) -> tuple[ComponentSpec, ...]:
    return tuple(s for s in _SPECS if s.category == category)


def prompt_map() -> dict[str, tuple[str, ...]]:
    """Open-vocabulary prompt set per class, for text-conditioned detectors."""
    return {s.id: s.prompts for s in _SPECS if s.prompts}


def flat_prompts() -> tuple[tuple[str, str], ...]:
    """``(prompt, class_id)`` pairs, ready to feed a zero-shot detector."""
    out: list[tuple[str, str]] = []
    for s in _SPECS:
        for p in s.prompts:
            out.append((p, s.id))
    return tuple(out)


def class_index() -> dict[str, int]:
    return {cid: i for i, cid in enumerate(CLASS_ORDER)}


def default_thresholds() -> dict[str, float]:
    return {s.id: s.min_conf for s in _SPECS}


def modular_classes() -> frozenset[str]:
    """Classes that are DIN-rail modular devices (used for row reasoning)."""
    return frozenset(s.id for s in _SPECS if "din_rail" in s.mounting)


def door_classes() -> frozenset[str]:
    return frozenset(s.id for s in _SPECS if "door" in s.mounting)


def countable_classes() -> frozenset[str]:
    """Classes worth counting as discrete devices in a bill of materials.

    Structural items (rail, duct, busbar) are reported as present/extent rather
    than as a meaningful integer quantity.
    """
    structural = {"din_rail", "wire_duct", "busbar", "neutral_bar", "earth_bar",
                  "cable_gland"}
    return frozenset(s.id for s in _SPECS if s.id not in structural)


def summary() -> dict:
    """Machine-readable description of the taxonomy, for the API and docs."""
    return {
        "class_count": len(CLASS_ORDER),
        "classes": [
            {
                "id": s.id, "name": s.name, "category": s.category,
                "domain": s.domain, "function": s.function, "role": s.role,
                "mounting": list(s.mounting), "min_conf": s.min_conf,
                "aliases": list(s.aliases), "prompts": list(s.prompts),
                "companions": list(s.companions), "notes": s.notes,
            }
            for s in _SPECS
        ],
        "categories": list(CATEGORIES),
        "unknown_class": UNKNOWN_COMPONENT_ID,
    }


__all__ = [
    "CATEGORIES", "DOMAINS", "CLASS_ORDER", "SPECS", "ComponentSpec",
    "UNKNOWN_COMPONENT_ID", "UNKNOWN_COMPONENT_NAME", "UNKNOWN_SPEC",
    "resolve", "spec", "display_name", "by_category", "prompt_map",
    "flat_prompts", "class_index", "default_thresholds", "modular_classes",
    "door_classes", "countable_classes", "summary",
]
