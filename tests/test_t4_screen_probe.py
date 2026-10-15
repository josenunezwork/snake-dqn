"""Fast unit contracts for the T4 batched screen-probe tooling."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from research.t4_screen_probe_20261015 import screen_probe


def _path(wall: float, decisions: dict[int, list[int]], records: dict[int, dict]) -> dict:
    return {"wall_seconds": wall, "decisions": decisions, "records": records}


def _record(mass: float, survival: float, deaths: int) -> dict:
    return {"mass_integral": mass, "survival_fraction": survival, "deaths": deaths}


def test_compare_reports_rare_action_divergence_and_outcome_delta() -> None:
    result = screen_probe.compare_paths(
        _path(12.0, {7: [1, 2, 0], 8: [1]}, {7: _record(3.0, 1.0, 0), 8: _record(2.0, 0.5, 1)}),
        _path(3.0, {7: [1, 5], 8: [1, 0]}, {7: _record(2.0, 0.5, 1), 8: _record(2.0, 0.5, 1)}),
    )
    assert result["decision_agreement_common_time"] == pytest.approx(2 / 3)
    assert result["decision_agreement_union"] == pytest.approx(2 / 5)
    assert result["missing_tail_decisions"] == 2
    assert result["speedup_rowwise_over_batched"] == 4.0
    assert result["outcome_agreement_fraction"] == 0.5
    assert result["max_abs_mass_delta"] == 1.0
    assert result["worlds"][7]["common_decisions"] == 2


def test_compare_requires_identical_world_inventory() -> None:
    with pytest.raises(ValueError, match="same worlds"):
        screen_probe.compare_paths(
            _path(1.0, {7: []}, {7: _record(1.0, 1.0, 0)}),
            _path(1.0, {8: []}, {8: _record(1.0, 1.0, 0)}),
        )


def test_parser_defaults_to_t2_scripted_32_world_short_measurement() -> None:
    args = screen_probe.build_parser().parse_args(["--student", "student.pth", "--out", "out.json"])
    assert args.arm == "student"
    assert (args.mix, args.worlds, args.round_index) == ("scripted", 32, 60)
    assert (args.device, args.v61_backend, args.anchors) == ("auto", "numba", "fast")
    assert (args.path, args.veto, args.ego2s_mask, args.ego2s_forward) == (
        "compare",
        "none",
        "advisory",
        "rowwise",
    )


def test_materialize_worlds_uses_only_the_requested_t2_mix(monkeypatch) -> None:
    args = SimpleNamespace(worlds=2, round_index=60, mix="scripted")
    monkeypatch.setattr(screen_probe.dev_probe, "probe_seeds", lambda *_: [10, 11])
    ctx = {
        "lookup": {"a": ("scripted", "greedy_food"), "b": ("scripted", "random_safe")},
        "ds": SimpleNamespace(
            _design_rows=lambda seeds: [
                {"world_seed": 10, "mix": "scripted", "slots": [{"member_sha256": "a"}]},
                {"world_seed": 11, "mix": "scripted", "slots": [{"member_sha256": "b"}]},
                {"world_seed": 10, "mix": "mixed", "slots": [{"member_sha256": "b"}]},
            ]
        ),
    }
    seeds, rows = screen_probe.materialize_worlds(args, ctx)
    assert seeds == [10, 11]
    assert rows == {10: [("scripted", "greedy_food")], 11: [("scripted", "random_safe")]}


def test_trace_patch_restores_existing_policy_methods() -> None:
    from src.simd_env import ego2s_policy

    original_init = ego2s_policy.Ego2sSimdPolicy.__init__
    original_actions = ego2s_policy.Ego2sSimdPolicy.actions
    trace = screen_probe.DecisionTrace([3])
    with screen_probe.trace_hero_actions(trace, "cpu"):
        assert ego2s_policy.Ego2sSimdPolicy.__init__ is not original_init
        assert ego2s_policy.Ego2sSimdPolicy.actions is not original_actions
    assert ego2s_policy.Ego2sSimdPolicy.__init__ is original_init
    assert ego2s_policy.Ego2sSimdPolicy.actions is original_actions


def test_run_path_injected_eval_preserves_t2_advisory_contract(monkeypatch) -> None:
    args = screen_probe.build_parser().parse_args(
        [
            "--student",
            "student.pth",
            "--out",
            "out.json",
        ]
    )
    monkeypatch.setattr(screen_probe, "resolve_device", lambda _: "mps")
    monkeypatch.setattr(screen_probe.dataclasses, "replace", lambda profile, **_: profile)
    monkeypatch.setattr(
        screen_probe, "materialize_worlds", lambda *_: ([5], {5: [("scripted", "x")]})
    )
    ctx = {
        "profile": object(),
        "lookup": {screen_probe.ni_spec.FRP3_S12[1]: ("checkpoint", "carrier")},
    }
    calls = []

    def fake_eval(*_positional, **kwargs):
        calls.append(kwargs)
        return [{"seed": 5, "mass_integral": 2.0, "survival_fraction": 1.0, "deaths": 0}]

    result = screen_probe.run_path(args, "batched", ctx=ctx, run_eval=fake_eval)
    assert result["device"] == "mps"
    assert calls[0]["hero_ego2s"] == args.student
    assert calls[0]["ego2s_forward"] == "batched"
    assert calls[0]["ego2s_mask"] == "advisory"
    assert calls[0]["sim_engine"] == "grid"
    assert calls[0]["vector61_forward"] == "rowwise"
    assert len(calls) == 2
    assert result["warmup"] == {"worlds": 2, "frames": 8, "performed": True}


def test_run_path_rejects_nonpositive_world_or_frame_count() -> None:
    args = screen_probe.build_parser().parse_args(
        ["--student", "student.pth", "--out", "out.json", "--worlds", "0"]
    )
    with pytest.raises(ValueError, match="positive"):
        screen_probe.run_path(args, "rowwise", ctx={"profile": object(), "lookup": {}})
