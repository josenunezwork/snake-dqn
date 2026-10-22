"""Focused contract tests for the fixed-endpoint P1 evaluator."""

from __future__ import annotations

import json
from pathlib import Path

from research.bl_ppo import p1_eval


def _audit(bank: dict) -> dict:
    return {
        "schema": "bl-ppo-p1-eval-seed-audit/v1",
        "pass": True,
        "bank_semantic_sha256": bank["allocation_sha256"],
        "candidate_count": 384,
        "collisions": [],
        "published_inventories": [{"path": "research/example/SEED_AUDIT.json", "sha256": "a" * 64}],
        "burned_seed_inventories": [{"path": "/safe/burned.json", "sha256": "b" * 64}],
        "historical_audit": {
            "status": "CLEAR",
            "collisions": [],
            "historical_frp_exclusions": [{"path": str(index)} for index in range(7)],
        },
        "excluded": ["snake-dqn-artifacts", "t34-round0-study-01"],
    }


def _result(bank: dict, *, gain: float = 50.0, boost: int = 50) -> dict:
    traces: dict[tuple[float, int, int], list[dict]] = {}

    def trace(mass: float, alive_frames: int, boosts: int) -> list[dict]:
        key = (mass, alive_frames, boosts)
        if key not in traces:
            traces[key] = [
                {
                    "frame": frame,
                    "alive": frame < alive_frames,
                    "mass": mass if frame < alive_frames else 0.0,
                    "action": 3 if frame < boosts else 0,
                }
                for frame in range(5000)
            ]
        return traces[key]

    rows = []
    for index, world in enumerate(bank["rows"]):
        # Slight deterministic variation keeps Student-t bounds well-defined.
        delta = gain + (index % 5 - 2) * 0.25
        b8_trace = trace(100.0, 4500, 1)
        b8 = {
            "checkpoint_sha256": p1_eval.B8_SHA256,
            "mass_integral": 90.0,
            "alive_frame_fraction": 0.90,
            "no_death_episode": False,
            "frames_stepped": 5000,
            "hero_decisions": 5000,
            "boost_decisions": 1,
            "trace": b8_trace,
            "trace_sha256": p1_eval.canonical_sha(b8_trace),
        }
        candidate_mass = 90.0 + delta
        candidate_trace = trace(candidate_mass, 5000, boost)
        candidate = {
            "checkpoint_sha256": "f" * 64,
            "mass_integral": candidate_mass,
            "alive_frame_fraction": 1.0,
            "no_death_episode": True,
            "frames_stepped": 5000,
            "hero_decisions": 5000,
            "boost_decisions": boost,
            "trace": candidate_trace,
            "trace_sha256": p1_eval.canonical_sha(candidate_trace),
        }
        rows.append(
            {
                **{key: world[key] for key in ("world_id", "world_seed", "mix", "horizon")},
                "roster_sha256": p1_eval.canonical_sha(world["opponents"]),
                "controllers": {"F": candidate, "B8": b8},
            }
        )
    return {
        "schema": "bl-ppo-p1-eval-results/v1",
        "complete": True,
        "cap_expired": False,
        "bank_semantic_sha256": bank["allocation_sha256"],
        "checkpoint_sha256": {"F": "f" * 64, "B8": p1_eval.B8_SHA256},
        "endpoint_receipt": {
            "schema": "bl-ppo-training-status/v1",
            "arm": "F",
            "p1_runtime": True,
            "complete": True,
            "status": "COMPLETE",
            "stop_reason": "endpoint",
            "target_transitions": 100_000_000,
            "pause_at": 100_000_000,
            "critic_qualified": True,
            "checkpoint_sha256": "f" * 64,
        },
        "endpoint_receipt_sha256": "e" * 64,
        "e13_parity": {
            "schema": "bl-p1-e13-parity/v1",
            "pass": True,
            "complete": True,
            "checkpoint_sha256": {"F": "f" * 64, "B8": p1_eval.B8_SHA256},
            "bank_semantic_sha256": bank["allocation_sha256"],
        },
        "e13_parity_sha256": "d" * 64,
        "rows": rows,
    }


def test_materialized_bank_is_fresh_complete_and_deterministic() -> None:
    bank = p1_eval.materialize()
    assert bank == p1_eval.materialize()
    assert bank["counts"] == {mix: 128 for mix in p1_eval.MIXES}
    assert {row["horizon"] for row in bank["rows"]} == {5000}
    assert len({row["world_seed"] for row in bank["rows"]}) == 384
    p1_eval.validate_bank(bank)


def test_collision_audit_binds_burned_and_safe_pruned_history(tmp_path: Path) -> None:
    bank = p1_eval.materialize()
    repo = tmp_path / "repo"
    inventory = repo / "research" / "example" / "SEED_AUDIT.json"
    inventory.parent.mkdir(parents=True)
    inventory.write_text(json.dumps({"seeds": [999_999_999]}), encoding="utf-8")
    burned = tmp_path / "burned.json"
    burned.write_text(json.dumps({"seeds": [888_888_888]}), encoding="utf-8")
    receipt = p1_eval.audit_bank(
        bank,
        repo=repo,
        burned_inventories=(burned,),
        historical_audit=lambda *_args, **_kwargs: {
            "status": "CLEAR",
            "collisions": [],
            "historical_frp_exclusions": [{"path": str(index)} for index in range(7)],
        },
    )
    assert receipt["pass"] is True
    assert receipt["excluded"] == ["snake-dqn-artifacts", "t34-round0-study-01"]
    p1_eval.check_audit(bank, receipt)


def test_nomination_requires_all_registered_clauses() -> None:
    bank, audit = p1_eval.materialize(), None
    audit = _audit(bank)
    decision = p1_eval.analyze(bank, audit, _result(bank))
    assert decision["status"] == "NOMINATE_MAC_STRICT"
    assert all(decision["clauses"].values())


def test_ub90_below_bar_rejects_artifact_after_complete_validation() -> None:
    bank = p1_eval.materialize()
    decision = p1_eval.analyze(bank, _audit(bank), _result(bank, gain=20.0))
    assert decision["status"] == "ARTIFACT_REJECTED_AT_BAR"
    assert decision["mass_gain"]["upper"] < 40


def test_invalid_or_incomplete_has_precedence_over_artifact_bar() -> None:
    bank = p1_eval.materialize()
    result = _result(bank, gain=20.0)
    result["complete"] = False
    decision = p1_eval.analyze(bank, _audit(bank), result)
    assert decision["status"] == "INCOMPLETE"
    assert "artifact_status" not in decision


def test_endpoint_controls_fail_before_any_world_evaluation() -> None:
    bank = p1_eval.materialize()
    result = _result(bank)
    result["e13_parity"]["pass"] = False
    try:
        p1_eval._validate_endpoint_controls(bank, result["checkpoint_sha256"], result)
    except ValueError as exc:
        assert "E13" in str(exc)
    else:
        raise AssertionError("bad E13 evidence reached gameplay")


def test_trace_action_counts_bind_delivered_boost() -> None:
    bank = p1_eval.materialize()
    result = _result(bank)
    result["rows"][0]["controllers"]["F"]["boost_decisions"] += 1
    decision = p1_eval.analyze(bank, _audit(bank), result)
    assert decision["status"] == "INCOMPLETE"


def test_out_of_range_boost_fails_nomination() -> None:
    bank = p1_eval.materialize()
    decision = p1_eval.analyze(bank, _audit(bank), _result(bank, boost=501))
    assert decision["status"] == "INCONCLUSIVE"
    assert decision["clauses"]["delivered_boost_in_005_10"] is False


def test_parallel_pair_assembly_is_ordered_and_keeps_both_arms_in_each_worker() -> None:
    worlds = [
        {"world_id": f"w-{index}", "world_seed": index, "mix": "scripted", "horizon": 5000}
        for index in range(4)
    ]

    def evaluator(item: tuple[dict, str, str]) -> dict:
        world, f_checkpoint, b8_checkpoint = item
        return {"world_id": world["world_id"], "pair": (f_checkpoint, b8_checkpoint)}

    serial = p1_eval.evaluate_pairs(
        worlds, Path("f.pt"), Path("b8.pt"), workers=1, evaluator=evaluator
    )
    parallel = p1_eval.evaluate_pairs(
        worlds, Path("f.pt"), Path("b8.pt"), workers=4, evaluator=evaluator
    )
    assert parallel == serial


def test_colab_runtime_exports_pairs_but_cannot_make_the_registered_claim() -> None:
    script = (Path(__file__).with_name("p1_colab.sh")).read_text(encoding="utf-8")
    assert "python -m research.bl_ppo.p1_eval --analyze" not in script
    assert 'payload="evaluation/results.json"' in script


def test_e2_runtime_screen_exports_compute_only_pairs_not_a_decision() -> None:
    source = Path(__file__).with_name("screen.py").read_text(encoding="utf-8")
    assert '"claim_status": "COMPUTE_ONLY_UNCLAIMED"' in source
    assert 'export_json(out / "decision.json", analyze(bank, audit, result))' not in source
