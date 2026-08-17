"""
Tests for the CLI wiring itself.

These exist because of a bug that bit twice: two ``def cmd_x`` definitions in the module
mean the second one wins, and *both* subparsers that reference the name end up pointing
at the same function. Nothing fails at import — the failure surfaces at runtime as
``AttributeError: 'Namespace' object has no attribute '<some arg>'``, from a subcommand
that looks entirely unrelated to the edit that broke it.

A subparser is also easy to register with a name that already exists; argparse quietly
replaces the first one.
"""

import argparse
import json

import pytest

from training.electrical import cli


@pytest.fixture(scope="module")
def parser():
    return cli.build_parser()


def _subparsers(parser):
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    raise AssertionError("no subparsers on the CLI parser")


# --------------------------------------------------------------------------
# the shadowing guards
# --------------------------------------------------------------------------

def test_every_subcommand_dispatches_to_a_distinct_function(parser):
    """Two subcommands sharing a handler means one was shadowed by a redefinition."""
    seen: dict = {}
    for name, sub in _subparsers(parser).items():
        func = sub.get_default("func")
        assert func is not None, f"subcommand {name!r} has no func default"
        if func in seen:
            raise AssertionError(
                f"subcommands {seen[func]!r} and {name!r} both dispatch to "
                f"{func.__name__} — a duplicate 'def {func.__name__}' in cli.py "
                f"shadowed the earlier one")
        seen[func] = name


def test_every_handler_is_callable_and_module_level(parser):
    for name, sub in _subparsers(parser).items():
        func = sub.get_default("func")
        assert callable(func), f"{name} -> {func!r} is not callable"
        assert getattr(cli, func.__name__, None) is func, \
            f"{name} -> {func.__name__} is not the module-level definition"


def test_no_duplicate_subcommand_names(parser):
    """argparse silently replaces a re-registered name, losing the first subcommand."""
    names = list(_subparsers(parser).keys())
    assert len(names) == len(set(names)), "a subcommand name is registered twice"


def test_the_pipeline_subcommands_are_all_present(parser):
    names = set(_subparsers(parser))
    expected = {
        "plan", "download", "split", "scope", "quality", "gap", "autolabel",
        "labelguide", "export", "analyse-batch", "hpo", "verify", "tensorrt",
        "synth", "remap", "merge", "dedup", "analyse", "train", "bench",
        "profile", "eval", "tune",
        # domain transfer + production-path acceptance
        "mix", "domain-gap", "finetune", "transfer", "expand", "accept",
    }
    missing = expected - names
    assert not missing, f"subcommands disappeared: {sorted(missing)}"


def test_scope_and_profile_are_different_subcommands(parser):
    """`profile` is the runtime latency profiler; `scope` is the class-scoping step.

    They were briefly the same name, and the collision meant one of them could not be
    invoked at all.
    """
    subs = _subparsers(parser)
    assert subs["scope"].get_default("func") is cli.cmd_scope
    assert subs["profile"].get_default("func") is cli.cmd_profile


# --------------------------------------------------------------------------
# argument parsing per subcommand
# --------------------------------------------------------------------------

class TestTransferSubcommandsParse:
    def test_mix(self, parser):
        a = parser.parse_args(["mix", "--real", "r", "--synth", "s", "--dst", "d"])
        assert a.func is cli.cmd_xf_mix
        assert (a.real, a.synth, a.dst) == ("r", "s", "d")
        assert a.copy is False           # symlink by default

    def test_mix_requires_all_three_roots(self, parser):
        with pytest.raises(SystemExit):
            parser.parse_args(["mix", "--real", "r"])

    def test_mix_synth_fraction_default_comes_from_the_module(self, parser):
        from training.electrical import transfer as xf

        a = parser.parse_args(["mix", "--real", "r", "--synth", "s", "--dst", "d"])
        assert a.synth_fraction == xf.DEFAULT_SYNTH_FRACTION

    def test_domain_gap(self, parser):
        a = parser.parse_args(["domain-gap", "--weights", "w.pt",
                               "--synth", "s", "--real", "r"])
        assert a.func is cli.cmd_xf_domaingap
        assert a.split == "val"

    def test_finetune_init_from_is_optional(self, parser):
        a = parser.parse_args(["finetune", "--data", "d.yaml"])
        assert a.func is cli.cmd_xf_finetune
        assert a.init_from is None
        assert a.no_staged is False      # staged by default

    def test_finetune_lr_default_is_well_below_from_scratch(self, parser):
        a = parser.parse_args(["finetune", "--data", "d.yaml"])
        assert a.lr0 <= 0.005

    def test_transfer_strategy_choices_are_validated(self, parser):
        a = parser.parse_args(["transfer", "--real", "r", "--synth", "s",
                               "--strategies", "mixed", "real_only"])
        assert a.strategies == ["mixed", "real_only"]
        with pytest.raises(SystemExit):
            parser.parse_args(["transfer", "--real", "r", "--synth", "s",
                               "--strategies", "wishful_thinking"])

    def test_transfer_defaults_include_the_control(self, parser):
        a = parser.parse_args(["transfer", "--real", "r", "--synth", "s"])
        assert "real_only" in a.strategies

    def test_expand_defaults_to_the_documented_step(self, parser):
        a = parser.parse_args(["expand"])
        assert a.func is cli.cmd_xf_expand
        assert (a.frm, a.to) == ("core8", "core15")
        assert a.data is None


# --------------------------------------------------------------------------
# expand behaviour
# --------------------------------------------------------------------------

class TestExpandCommand:
    def _args(self, parser, argv):
        return parser.parse_args(argv)

    def test_plan_only_run_prints_and_succeeds(self, parser, capsys):
        rc = cli.cmd_xf_expand(self._args(parser, ["expand"]))
        assert rc == 0
        out = capsys.readouterr()
        assert '"index_stable": true' in out.out
        assert "core8 -> core15" in out.err

    def test_an_unknown_profile_is_reported_not_raised(self, parser, capsys):
        rc = cli.cmd_xf_expand(self._args(parser, ["expand", "--to", "core99"]))
        assert rc == 1
        assert "core99" in capsys.readouterr().err

    def test_it_refuses_to_carry_a_checkpoint_across_a_non_prefix_change(
            self, parser, capsys, monkeypatch):
        called = []
        monkeypatch.setattr(cli.xf, "fine_tune",
                            lambda *a, **k: called.append(1))
        rc = cli.cmd_xf_expand(self._args(parser, [
            "expand", "--to", "full", "--data", "d.yaml", "--init-from", "c.pt"]))
        assert rc == 1
        assert not called, "it started training despite the index mismatch"
        out = capsys.readouterr()
        assert "refusing to fine-tune" in out.err
        assert '"status": "refused"' in out.out

    def test_a_non_prefix_change_without_init_from_is_allowed(
            self, parser, capsys, monkeypatch):
        # Training a fresh head across a profile change is fine; it is only reusing the
        # old head that is wrong.
        monkeypatch.setattr(cli.xf, "fine_tune",
                            lambda *a, **k: {"status": "completed",
                                             "weights": "w.pt", "stages": []})
        rc = cli.cmd_xf_expand(self._args(parser, [
            "expand", "--to", "full", "--data", "d.yaml"]))
        assert rc == 0

    def test_a_stable_step_passes_the_checkpoint_through(self, parser, monkeypatch):
        seen = {}

        def fake_ft(data, init_from, **kw):
            seen["data"], seen["init"] = data, init_from
            seen.update(kw)
            return {"status": "completed", "weights": "w.pt", "stages": []}

        monkeypatch.setattr(cli.xf, "fine_tune", fake_ft)
        rc = cli.cmd_xf_expand(self._args(parser, [
            "expand", "--data", "core15.yaml", "--init-from", "core8.pt"]))
        assert rc == 0
        assert seen["data"] == "core15.yaml"
        assert seen["init"] == "core8.pt"
        assert seen["name"] == "expand_core8_to_core15"

    def test_a_failed_training_run_exits_non_zero(self, parser, monkeypatch):
        monkeypatch.setattr(cli.xf, "fine_tune",
                            lambda *a, **k: {"status": "failed",
                                             "reason": "oom", "stages": []})
        rc = cli.cmd_xf_expand(self._args(parser, [
            "expand", "--data", "d.yaml", "--init-from", "c.pt"]))
        assert rc == 1


# --------------------------------------------------------------------------
# mix / domain-gap / transfer command behaviour
# --------------------------------------------------------------------------

class TestMixCommand:
    def test_a_missing_real_val_split_is_reported_not_raised(self, parser, capsys,
                                                            monkeypatch):
        def boom(*a, **k):
            raise ValueError("real has no val split")

        monkeypatch.setattr(cli.xf, "build_mixed", boom)
        rc = cli.cmd_xf_mix(parser.parse_args(
            ["mix", "--real", "r", "--synth", "s", "--dst", "d"]))
        assert rc == 1
        out = capsys.readouterr()
        assert "no val split" in out.err
        assert '"status": "failed"' in out.out

    def test_copy_flag_inverts_symlink(self, parser, monkeypatch):
        seen = {}
        monkeypatch.setattr(cli.xf, "build_mixed",
                            lambda *a, **k: seen.update(k) or {"warnings": []})
        cli.cmd_xf_mix(parser.parse_args(
            ["mix", "--real", "r", "--synth", "s", "--dst", "d", "--copy"]))
        assert seen["symlink"] is False


class TestDomainGapCommand:
    def test_it_fails_when_real_evaluation_did_not_run(self, parser, monkeypatch):
        monkeypatch.setattr(cli.xf, "measure_domain_gap",
                            lambda *a, **k: {"real": {"status": "skipped"},
                                             "interpretation": "no data"})
        rc = cli.cmd_xf_domaingap(parser.parse_args(
            ["domain-gap", "--weights", "w", "--synth", "s", "--real", "r"]))
        assert rc == 1

    def test_it_succeeds_when_real_evaluation_ran(self, parser, monkeypatch):
        monkeypatch.setattr(cli.xf, "measure_domain_gap",
                            lambda *a, **k: {"real": {"status": "evaluated"},
                                             "interpretation": "ok"})
        rc = cli.cmd_xf_domaingap(parser.parse_args(
            ["domain-gap", "--weights", "w", "--synth", "s", "--real", "r"]))
        assert rc == 0


class TestAcceptCommand:
    """The acceptance gate. Its exit code is what a CI job keys on."""

    def _stub(self, monkeypatch, best, acceptance=None, sweep=None):
        from training.electrical import prodeval as pe

        rep = {"sweep": sweep if sweep is not None else [
                   {"conf": 0.05, "status": "evaluated", "map_50": 0.5,
                    "map_50_95": 0.3, "precision": 0.6, "recall": 0.7, "f1": 0.65,
                    "tp": 1, "fp": 1, "fn": 1, "fp_per_image": 0.1,
                    "fn_per_image": 0.1, "unknown_predictions": 0,
                    "is_production_default": False, "per_class": {}}],
               "best_operating_point": best}
        if acceptance:
            rep["acceptance"] = acceptance
        monkeypatch.setattr(pe, "acceptance_report", lambda *a, **k: rep)
        return pe

    def test_defaults_come_from_the_module(self, parser):
        from training.electrical import prodeval as pe

        a = parser.parse_args(["accept", "--weights", "w.pt", "--root", "r"])
        assert a.func is cli.cmd_pe_acceptance
        # None in the parser, resolved from the module in the handler, so the help
        # text and the swept points cannot drift apart.
        assert a.confs is None and a.backend is None
        assert pe.OPERATING_POINTS == (0.01, 0.03, 0.05, 0.10, 0.20)

    def test_it_sweeps_the_module_defaults_when_none_given(self, parser,
                                                          monkeypatch):
        from training.electrical import prodeval as pe

        seen = {}

        def fake(weights, root, split, **kw):
            seen.update(kw)
            return {"sweep": [], "best_operating_point": {"status": "failed",
                                                          "reason": "x"}}

        monkeypatch.setattr(pe, "acceptance_report", fake)
        cli.cmd_pe_acceptance(parser.parse_args(
            ["accept", "--weights", "w.pt", "--root", "r"]))
        assert seen["confs"] == pe.OPERATING_POINTS
        assert seen["backend"] == pe.DEFAULT_BACKEND

    def test_explicit_confs_are_used(self, parser, monkeypatch):
        from training.electrical import prodeval as pe

        seen = {}
        monkeypatch.setattr(pe, "acceptance_report",
                            lambda w, r, s, **kw: seen.update(kw) or {
                                "sweep": [],
                                "best_operating_point": {"status": "failed",
                                                         "reason": "x"}})
        cli.cmd_pe_acceptance(parser.parse_args(
            ["accept", "--weights", "w.pt", "--root", "r",
             "--confs", "0.02", "0.5"]))
        assert seen["confs"] == (0.02, 0.5)

    def test_a_selected_point_exits_zero(self, parser, monkeypatch, capsys):
        self._stub(monkeypatch, {"status": "selected", "conf": 0.05,
                                 "rationale": "because"})
        rc = cli.cmd_pe_acceptance(parser.parse_args(
            ["accept", "--weights", "w.pt", "--root", "r"]))
        assert rc == 0
        out = capsys.readouterr()
        assert json.loads(out.out)["best_operating_point"]["conf"] == 0.05
        assert "best operating point" in out.err

    def test_no_selectable_point_exits_non_zero(self, parser, monkeypatch):
        self._stub(monkeypatch, {"status": "failed", "reason": "nothing scored"})
        rc = cli.cmd_pe_acceptance(parser.parse_args(
            ["accept", "--weights", "w.pt", "--root", "r"]))
        assert rc == 1

    def test_an_unmet_target_exits_non_zero(self, parser, monkeypatch, capsys):
        # The point of --target-map50: CI must not pass by ignoring the shortfall.
        self._stub(monkeypatch,
                   {"status": "selected", "conf": 0.05, "rationale": "r"},
                   acceptance={"passed": False, "statement": "does NOT meet"})
        rc = cli.cmd_pe_acceptance(parser.parse_args(
            ["accept", "--weights", "w.pt", "--root", "r",
             "--target-map50", "0.85"]))
        assert rc == 1
        assert "does NOT meet" in capsys.readouterr().err

    def test_a_met_target_exits_zero(self, parser, monkeypatch):
        self._stub(monkeypatch,
                   {"status": "selected", "conf": 0.05, "rationale": "r"},
                   acceptance={"passed": True, "statement": "meets"})
        rc = cli.cmd_pe_acceptance(parser.parse_args(
            ["accept", "--weights", "w.pt", "--root", "r",
             "--target-map50", "0.40"]))
        assert rc == 0

    def test_an_unmet_constraint_warning_reaches_stderr(self, parser, monkeypatch,
                                                       capsys):
        self._stub(monkeypatch, {"status": "selected", "conf": 0.05,
                                 "rationale": "r",
                                 "warning": "nothing met min_precision"})
        cli.cmd_pe_acceptance(parser.parse_args(
            ["accept", "--weights", "w.pt", "--root", "r",
             "--min-precision", "0.99"]))
        assert "nothing met min_precision" in capsys.readouterr().err

    def test_it_writes_json_and_csv_when_asked(self, parser, monkeypatch, tmp_path):
        self._stub(monkeypatch, {"status": "selected", "conf": 0.05,
                                 "rationale": "r"})
        j, c = str(tmp_path / "a.json"), str(tmp_path / "s.csv")
        cli.cmd_pe_acceptance(parser.parse_args(
            ["accept", "--weights", "w.pt", "--root", "r", "--out", j,
             "--csv", c]))
        assert json.load(open(j))["best_operating_point"]["conf"] == 0.05
        assert open(c).read().startswith("conf,status,map_50")


class TestTransferCommand:
    def test_stdout_stays_parseable_json_with_the_table_on_stderr(
            self, parser, monkeypatch, capsys):
        monkeypatch.setattr(cli.xf, "compare_strategies", lambda *a, **k: {
            "status": "completed", "winner": "mixed",
            "ranking": [{"strategy": "mixed", "map_50": 0.7,
                         "map_50_95": 0.4, "precision": 0.8, "recall": 0.6}],
            "rationale": "mixed wins"})
        rc = cli.cmd_xf_compare(parser.parse_args(
            ["transfer", "--real", "r", "--synth", "s"]))
        assert rc == 0
        out = capsys.readouterr()
        # The documented `> comparison.json` redirect has to keep working.
        assert json.loads(out.out)["winner"] == "mixed"
        assert "mixed" in out.err          # human-readable table went to stderr
