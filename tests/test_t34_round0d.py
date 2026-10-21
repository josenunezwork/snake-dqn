"""Round 0-D tests on tiny synthetic fixtures (one thread; no Colab, no Round 0 outputs)."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

for _name in ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

from research.t34_exit_b8.engine import ActionOutcome, ExitTeacherConfig  # noqa: E402
from research.t34_exit_b8.policy import PolicyEvaluation  # noqa: E402
from research.t34_round0d import analyze, controller, history, labels, sizing  # noqa: E402
from research.t34_round0d.a5d import compare  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth"
)


# --------------------------------------------------------------------------- controller


def _t3_choice(greedy: int, q: np.ndarray, actions: list[int], frames: dict[int, int]) -> int:
    """Verbatim selection expression from research/t3_rollout_labels_20261015/ceiling.py."""
    best_frames = max(frames[a] for a in actions)
    tied = [a for a in actions if frames[a] == best_frames]
    return greedy if greedy in tied else max(tied, key=lambda a: (float(q[a]), -a))


def test_selection_matches_t3_expression_on_random_cases() -> None:
    rng = random.Random(20261021)
    for _ in range(2000):
        actions = sorted(rng.sample(range(6), rng.randint(1, 6)))
        greedy = rng.choice(actions)
        q = np.asarray([rng.choice([0.0, 0.5, 1.0, rng.random()]) for _ in range(6)], np.float32)
        frames = {a: rng.choice([3, 120, 299, 300]) for a in actions}
        assert controller.select_action(greedy, q, actions, frames) == _t3_choice(
            greedy, q, actions, frames
        )


def test_label_roles_follow_amendment_two() -> None:
    assert controller.label_kind(1, 1, {1: True, 2: True}) == "confirm"
    assert controller.label_kind(1, 2, {1: False, 2: True}) == "decisive"
    assert controller.label_kind(1, 2, {1: False, 2: False}) == "delay"
    assert controller.label_kind(1, 1, {1: False, 2: False}) == "no_rescue"


class _Sim:
    def __init__(self, lengths: list[int], advisory: list[bool]) -> None:
        self.lengths = lengths
        self.decision = 0
        self.advisory = np.asarray(advisory, dtype=bool)
        self.alive = np.ones((1, 6), dtype=bool)

    @property
    def length(self) -> np.ndarray:
        return np.full((1, 6), self.lengths[self.decision])

    def get_advisory_action_mask(self) -> np.ndarray:
        return self.advisory[None, None, :].repeat(6, axis=1)


class _Teacher:
    """Records rollout requests; alive steps favour action 4 (a boost) by one frame."""

    def __init__(self) -> None:
        self.config = ExitTeacherConfig(horizon=300)
        self.requests: list[tuple[int, list[int], int]] = []
        self.sim: _Sim | None = None

    def evaluate(self, _world: object) -> PolicyEvaluation:
        q = np.asarray([0.1, 0.9, 0.2, 0.0, 0.3, 0.0], np.float32)
        legal = np.ones(6, bool)
        # B8Policy semantics: select = advisory, normal-only fallback when empty.
        select = self.sim.advisory.copy() if self.sim is not None else legal.copy()
        if not select.any():
            select[:3] = True
        return PolicyEvaluation(q=q, legal=legal, select=select)

    def _rollout_merged(self, worlds: list, choices: list[list[int]]) -> list[dict]:
        assert self.sim is not None
        depth = self.config.horizon
        self.requests.append((self.sim.decision, list(choices[0]), depth))
        outcome = {
            a: ActionOutcome(
                survives=False,
                alive_steps=depth - 2 + int(a == 4),
                mass=10.0 * depth,
                row_steps=depth,
            )
            for a in choices[0]
        }
        return [outcome]


def _run_controller(lengths: list[int], horizon: int, advisory: list[bool] | None = None):
    sim = _Sim(lengths, advisory or [True] * 6)
    teacher = _Teacher()
    teacher.sim = sim
    dense = controller.DenseT3Controller(teacher, horizon=horizon)
    actions, labels_out = [], []
    for decision in range(len(lengths)):
        sim.decision = decision
        action, label = dense.decide(SimpleNamespace(sim=sim), decision)
        actions.append(action)
        if label:
            labels_out.append(label)
    return teacher, dense, actions, labels_out


def test_dense_clock_starts_at_first_long_decision_and_survives_length_dips() -> None:
    lengths = [500] * 7 + [600, 650, 590, 590, 590, 700, 700, 700, 700, 700, 700, 599, 599]
    teacher, dense, actions, labels_out = _run_controller(lengths, horizon=5000)
    # First L>=600 at decision 7; fires at 7, 12, 17 (dips below 600 do not fire or reset).
    assert [request[0] for request in teacher.requests] == [7, 12, 17]
    assert all(request[2] == 300 for request in teacher.requests)
    assert teacher.config.horizon == 300  # restored after every batch
    # All die; the all-die delay (action 4, +1 frame) executes (T3 rule).
    assert [label["action"] for label in labels_out] == [4, 4, 4]
    assert all(
        label["label_kind"] == "delay" and label["label_weight"] == 0.0 for label in labels_out
    )
    assert actions[7] == 4 and actions[8] == 1
    assert dense.stats["action_changes"] == 3 and dense.stats["delay_changes"] == 3


def test_depth_truncates_at_horizon_and_empty_advisory_falls_back_to_bare() -> None:
    teacher, _dense, _actions, labels_out = _run_controller([700] * 10, horizon=8)
    assert [(r[0], r[2]) for r in teacher.requests] == [(0, 8), (5, 3)]
    assert labels_out[1]["horizon"] == 3
    teacher, dense, actions, labels_out = _run_controller([700] * 6, 5000, [False] * 6)
    assert teacher.requests == [] and labels_out == [] and set(actions) == {1}
    assert dense.stats["empty_advisory_triggers"] == 2


def test_candidates_are_advisory_legal_actions_only() -> None:
    advisory = [True, False, True, False, True, True]
    teacher, _dense, _actions, labels_out = _run_controller([700], 5000, advisory)
    assert teacher.requests[0][1] == [0, 2, 4, 5]
    assert sorted(labels_out[0]["outcomes"]) == ["0", "2", "4", "5"]


# --------------------------------------------------------------------------- bank


def test_bank_is_fresh_disjoint_and_reuses_round0_rosters() -> None:
    from research.t34_round0.bank import _rows
    from research.t34_round0d import bank

    value = json.loads((ROOT / "research/t34_round0d/BANK_ROUND0D.json").read_text())
    assert value["namespace"] == "t34-round0d-dense-20261021/v1"
    assert [r["mix"] for r in value["rows"]].count("scripted") == 96
    seeds = {
        r["world_seed"]
        for key in ("rows", "excluded_preflight_rows", "excluded_a5d_rows")
        for r in value[key]
    }
    assert len(seeds) == 240 + 2 + 30
    for item in bank.known_banks().values():
        assert not seeds & item["seeds"]
    expected = _rows(
        "t34-round0d-dense-20261021/v1", {"scripted": 96, "frozen": 72, "mixed": 72}, 5000
    )
    assert value["rows"] == expected
    audit = json.loads((ROOT / "research/t34_round0d/SEED_AUDIT_ROUND0D.json").read_text())
    assert audit["status"] == "CLEAR" and audit["positive_controls"]
    assert "t34-round0-study-01" in audit["excluded_live_study_roots"]
    assert not any(audit["concrete_hits"].values())


def test_whole_allocation_rejects_on_any_collision() -> None:
    from research.t34_round0d import bank

    allocation = {
        "rows": [{"world_seed": 1}],
        "excluded_preflight_rows": [{"world_seed": 2}],
        "excluded_a5d_rows": [{"world_seed": 1}],
    }
    with pytest.raises(RuntimeError, match="internal collision"):
        bank.audit(allocation, scan=lambda seeds: {})
    round0_seed = min(bank.known_banks()["round0"]["seeds"])
    allocation["excluded_a5d_rows"] = [{"world_seed": round0_seed}]

    def scan(seeds: list[int]) -> dict:
        return {"concrete_hits": {"root": list(seeds)}, "historical_frp_exclusions": []}

    with pytest.raises(RuntimeError, match="bank:round0"):
        bank.audit(allocation, scan=scan)


# --------------------------------------------------------------------------- history


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _authoring_repo(tmp_path: Path) -> Path:
    remote, author = tmp_path / "origin.git", tmp_path / "author"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(tmp_path, "init", str(author))
    _git(author, "config", "user.email", "round0d@example.test")
    _git(author, "config", "user.name", "Round 0-D test")
    _git(author, "checkout", "-b", "main")
    _git(author, "remote", "add", "origin", str(remote))
    (author / "README").write_text("fixture\n")
    _git(author, "add", "README")
    _git(author, "commit", "-m", "initial fixture")
    _git(author, "push", "-u", "origin", "main")
    return author


def _registration(tmp_path: Path, author: Path) -> dict:
    root = (tmp_path / "round0d-study").resolve()
    return {
        "study_root": str(root),
        "commit": _git(author, "rev-parse", "HEAD"),
        "approved_at": "1970-01-01T00:00:00+00:00",
        "sizing": {"chunks": [{"start": 0, "count": 10}]},
        "history_ledger": {
            "repo": str(author.resolve()),
            "path": f"research/t34_round0d/ledger/{root.name}.jsonl",
        },
    }


class _Guard:
    def __init__(self, *_args: object) -> None:
        pass

    def __enter__(self) -> "_Guard":
        return self

    def __exit__(self, *_args: object) -> None:
        return None


def test_round0d_claims_never_match_round0_admission_glob(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    attempt = Path(value["study_root"]) / "chunks/chunk-000/attempt-01"
    history.reserve(attempt, value, world_ids=["w0"])
    assert _git(author, "ls-remote", "--refs", "origin", "refs/tags/t34-round0/*") == ""
    assert "refs/tags/t34-round0d/round0d-study/chunks/chunk-000/attempt-01" in _git(
        author, "ls-remote", "--refs", "origin", "refs/tags/t34-round0d/*"
    )


def test_policy_refuses_round0_ledger_path(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    value["history_ledger"]["path"] = "research/t34_round0/ledger/round0d-study.jsonl"
    with pytest.raises(history.HistoryError, match="canonical"):
        history.policy(value)


def test_in_flight_attempt_is_skipped_but_abandoned_claim_is_kill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    attempt = Path(value["study_root"]) / "chunks/chunk-000/attempt-01"
    history.reserve(attempt, value, world_ids=["w0"])
    claim = history.InflightClaim(attempt)
    assert history.in_flight(attempt)
    assert history.require_published_history(value) == []
    monkeypatch.setattr("research.t34_pathb.source_binding_v2.FrozenWorktreeGuard", _Guard)
    with pytest.raises(history.HistoryError, match="still in flight"):
        history.run_phase(value, "analysis", lambda _out: {"decision": "GO"})
    assert not (Path(value["study_root"]) / "analysis").exists()
    claim.release()
    assert not history.in_flight(attempt)
    with pytest.raises(history.HistoryError, match="missing its terminal"):
        history.require_published_history(value)


def test_sealed_attempt_publishes_and_unsealed_analysis_is_kill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    attempt = Path(value["study_root"]) / "chunks/chunk-000/attempt-01"
    history.reserve(attempt, value, world_ids=["w0"])
    claim = history.InflightClaim(attempt)
    history.finalize(attempt, value, "KILL", "operator abort or unexplained interruption")
    claim.release()
    ledger = value["history_ledger"]["path"]
    with pytest.raises(history.HistoryError, match="committed and pushed"):
        history.require_published_history(value)
    _git(author, "add", ledger)
    _git(author, "commit", "-m", "ledger")
    _git(author, "push", "origin", "main")
    terminals = history.require_published_history(value)
    assert [t["status"] for t in terminals] == ["KILL"]
    monkeypatch.setattr("research.t34_pathb.source_binding_v2.FrozenWorktreeGuard", _Guard)
    with pytest.raises(history.HistoryError):
        history.run_phase(value, "analysis", lambda _out: history.require_chunks([], value))
    decision = json.loads((Path(value["study_root"]) / "analysis/decision.json").read_text())
    assert decision["decision"] == "KILL"


# --------------------------------------------------------------------------- analysis


def _records() -> list[dict]:
    rows = []
    for mix, n in {"scripted": 96, "frozen": 72, "mixed": 72}.items():
        for i in range(n):
            arms = {}
            for name in ("bare", "teacher"):
                alive = i < int(n * (0.50 if name == "bare" else 0.75))
                decisions = 5000 if alive else 2500
                boost = decisions // 50
                counts = [decisions // 3, decisions // 3, 0, boost, 0, 0]
                counts[2] = decisions - sum(counts)
                arms[name] = dict(
                    mass_integral=100 + (50 if name == "teacher" else 0) + i % 7,
                    survival_fraction=1.0 if alive else 0.5,
                    no_death=int(alive),
                    deaths=int(not alive),
                    decisions=decisions,
                    boost_decisions=boost,
                    action_counts=counts,
                    unconditional_faults=[],
                    dense=dict(batches=600, action_changes=2, rescue_changes=1, delay_changes=1),
                )
            rows.append(
                dict(
                    world=dict(
                        world_id=f"w{len(rows)}", world_seed=len(rows), mix=mix, horizon=5000
                    ),
                    arms=arms,
                )
            )
    return rows


def test_bars_are_round0s_function_object_and_descriptives_never_gate() -> None:
    from research.t34_round0 import analyze as round0_analyze

    assert analyze.summarize is round0_analyze.summarize
    records = _records()
    result = analyze.summarize(copy.deepcopy(records))
    assert result["decision"] == "GO"
    report = analyze.descriptive(records)
    assert report["authority"] == "DESCRIPTIVE_ONLY_NEVER_A_GATE"
    assert report["per_mix"]["scripted"]["teacher_deaths_with_executed_rescue"] == 24
    assert report["t3_reference_scripted_no_death_gain_b8_s3"] == 0.28


def _set_boosts(records: list[dict], arm: str, boosts_per_world: int) -> None:
    for row in records:
        item = row["arms"][arm]
        counts = item["action_counts"]
        counts[0] += counts[3] - boosts_per_world
        counts[3] = boosts_per_world
        item["boost_decisions"] = boosts_per_world


def test_relative_boost_floor_replaces_the_absolute_floor() -> None:
    from research.t34_round0_amend.rule import amend, pooled_boosts

    assert analyze.amend is amend
    records = _records()
    # B8-like regime: both arms ~0.1% boosts. Registered absolute floor STOPs; the
    # amended relative floor (teacher >= 0.5 x bare) passes and the verdict is GO.
    _set_boosts(records, "bare", 5)
    _set_boosts(records, "teacher", 4)
    registered = analyze.summarize(copy.deepcopy(records))
    assert registered["decision"] == "STOP"
    amended = amend(registered, pooled_boosts(records, "teacher"), pooled_boosts(records, "bare"))
    assert amended["decision"] == "GO" and amended["clauses"]["delivered_boost_share"]
    _set_boosts(records, "teacher", 2)  # below half the paired bare share
    registered = analyze.summarize(copy.deepcopy(records))
    amended = amend(registered, pooled_boosts(records, "teacher"), pooled_boosts(records, "bare"))
    assert amended["decision"] == "STOP"


def test_point_bar_failure_is_stop_exactly_as_round0() -> None:
    records = _records()
    for row in records:
        row["arms"]["teacher"]["mass_integral"] = row["arms"]["bare"]["mass_integral"] + 39.0
    assert analyze.summarize(records)["decision"] == "STOP"


# --------------------------------------------------------------------------- sizing / A5-D


def _receipt(rate: float, **changes: object) -> dict:
    import hashlib

    value = {
        "schema": sizing.SCHEMA,
        "fidelity_status": "PASS",
        "teacher_worlds_per_hour": rate,
        "bare_seconds_per_world": 190.0,
        "metric_seconds_per_pair": 50.0,
        "cpu_slowdown_factor": 1.0,
        "setup_minutes": 15.0,
        "measured_export_bytes_per_pair": 3 * 1024**2,
        "evidence": {"fixture": True},
    }
    value.update(changes)
    value["evidence_sha256"] = hashlib.sha256(
        json.dumps(value["evidence"], sort_keys=True).encode()
    ).hexdigest()
    return value


def test_rate_to_action_table_and_round0_chunk_formula() -> None:
    assert sizing.plan(_receipt(4.9))["action"] == "KILL"
    assert "wall" in sizing.plan(_receipt(5.5))["reason"]
    with pytest.raises(sizing.SizingRefused, match="HOLD"):
        sizing.plan(_receipt(20.0, fidelity_status="DIFFERENCE"))
    assert sizing.plan(_receipt(20.0, fidelity_status="EXPLAINED_TIE"))["action"] == "RATIFY"
    with pytest.raises(sizing.SizingRefused, match="9 MiB"):
        sizing.plan(_receipt(20.0, measured_export_bytes_per_pair=10 * 1024**2))
    result = sizing.plan(_receipt(10.0))
    assert result["action"] == "RATIFY" and result["expected_units"] <= 120
    assert result["wall_hours_forecast"] <= 24 and result["max_in_flight"] == 2
    assert sum(chunk["count"] for chunk in result["chunks"]) == 240
    assert all(chunk["estimate_minutes"] <= 180 for chunk in result["chunks"])
    headroom = max(
        c["unit_cap"] - e for c, e in zip(result["chunks"], result["expected_chunk_units"])
    )
    assert result["study_unit_ceiling"] == pytest.approx(1.2 * result["expected_units"] + headroom)
    replay = result["replay"]
    assert replay["timeout_minutes"] >= max(
        replay["estimate_minutes"] + 25, 2 * replay["estimate_minutes"]
    )


def test_derived_ceiling_never_strands_the_last_chunk_with_one_lane_in_flight() -> None:
    result = sizing.plan(_receipt(8.0))
    chunks, expected = result["chunks"], result["expected_chunk_units"]
    prior = [{"elapsed_seconds": e / 2.0 * 3600.0} for e in expected[:-2]]
    prior.append({"elapsed_seconds": None, "unit_cap": chunks[-2]["unit_cap"]})
    spend = sizing.admission_spend(prior, chunks[-1], result["study_unit_ceiling"])
    assert spend["allowed"]
    settled_over = [{"elapsed_seconds": 3600.0 * result["study_unit_ceiling"]}]
    spend = sizing.admission_spend(settled_over, chunks[-1], result["study_unit_ceiling"])
    assert not spend["allowed"] and spend["infeasible_on_settled_charges"]


def test_a5d_rate_comes_from_process_seconds_and_rows_never_a_window() -> None:
    worlds = [
        {
            "kind": "fidelity",
            "mix": "scripted",
            "teacher_seconds": 7200.0,
            "rollout_row_steps": 900_000,
            "decisions": 5000,
        },
        {
            "kind": "throughput",
            "mix": "frozen",
            "teacher_seconds": 600.0,
            "rollout_row_steps": 50_000,
            "decisions": 1500,
        },
    ]
    rates = sizing.rate_estimates(worlds)
    rho = 7800.0 / (950_000 + 6500)
    assert rates["seconds_per_step"] == pytest.approx(rho)
    assert rates["rows_per_world_used"] == sizing.T3_ROWS_PER_WORLD  # conservative max
    assert rates["r_rows"] == pytest.approx(12 * 3600 / (rho * (859_420 + 5000)))
    assert rates["r_completed"] == pytest.approx(12 * 3600 / 600.0)
    assert rates["teacher_worlds_per_hour"] == min(rates["r_rows"], rates["r_completed"])
    with pytest.raises(sizing.SizingRefused, match="HOLD"):
        sizing.rate_estimates([])


def test_dense_r1_is_label_yield_sized_and_capped_at_200u() -> None:
    plan = sizing.dense_r1_plan(10.0, 0.66, 0.5)
    assert plan["fresh_worlds"] == 455  # ceil(300 / 0.66)
    assert plan["expected_units"] == pytest.approx(455 / 10.0 * 2.0)
    assert plan["fundable"] is True and plan["fit_preconditions"]
    assert sizing.dense_r1_plan(5.0, 0.66, 0.5)["fundable"] is False  # > 24 h on 3 lanes
    assert sizing.dense_r1_plan(20.0, 0.3, 0.5)["reason"].startswith("held-out minimum")
    assert sizing.dense_r1_plan(20.0, 0.0, 0.0)["fundable"] is False


def test_fidelity_tie_rule_is_pre_registered_and_narrow() -> None:
    from research.t34_round0d.a5d import first_divergence, tie_explained

    expected = {"batch_decisions": [10, 15], "changes": [[15, 1, 0]]}
    observed = {"batch_decisions": [10, 15], "changes": [[15, 1, 2]]}
    assert first_divergence(expected, observed) == 15
    near = {
        "decision": 15,
        "q": [1.0, 2.0, 2.0 + 5e-5, 0.0, 0.0, 0.0],
        "legal": [True] * 6,
        "outcomes": {"0": {"alive_steps": 300}, "2": {"alive_steps": 300}},
    }
    assert tie_explained(15, [near])
    far = dict(near, q=[1.0, 2.0, 2.5, 0.0, 0.0, 0.0])
    assert not tie_explained(15, [far])
    assert not tie_explained(16, [near])


def test_fidelity_compare_is_exact() -> None:
    expected = {
        "episode_survival": False,
        "death_t": 4776,
        "decisions": 4777,
        "alive_frame_fraction": 0.9552,
        "mass_integral": 700.5,
        "batch_decisions": [10, 15],
        "changes": [[15, 1, 0]],
    }
    assert compare(expected, copy.deepcopy(expected)) == []
    observed = dict(copy.deepcopy(expected), changes=[[15, 1, 2]], mass_integral=700.4)
    assert len(compare(expected, observed)) == 2


# --------------------------------------------------------------------------- labels / runner


def test_sealed_labels_refuse_any_purpose_but_r1_and_any_missing_go(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    with pytest.raises(labels.SealedLabelsRefused, match="never admitted"):
        labels.admit(value, "screen")
    with pytest.raises(labels.SealedLabelsRefused, match="stay sealed"):
        labels.admit(value, "dense_r1_fit")


def test_public_trace_strips_labels_and_sealed_keeps_them() -> None:
    from research.t34_round0d.runner import public_trace

    full = {"labels": [{"decision": 1}], "teacher_v2": {"batches": 1}, "actions": []}
    public = public_trace(full)
    assert public["labels"] == [] and "teacher_v2" not in public
    assert full["labels"] == [{"decision": 1}]


def test_source_closure_binds_round0_rules_and_plan() -> None:
    from research.t34_round0d.common import source_hashes

    hashes = source_hashes(ROOT)
    for name in (
        "research/t34_round0/PREREGISTRATION_ROUND0.md",
        "research/t34_round0/OC_ROUND0.json",
        "research/t34_round0d/PREREGISTRATION_ROUND0D.md",
        "docs/research/specs/2026-10-20_t34_nogo_branch_plan.md",
    ):
        assert name in hashes


@pytest.mark.slow
@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="released B8 s3 checkpoint not on this host")
def test_real_dense_trace_on_excluded_fixture_with_lowered_trigger(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real merged-rollout engine; min_length is lowered for this test only."""
    monkeypatch.setenv("SNAKE_CKPT_DIR", "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
    from research.t34_round0d.runner import dense_trace

    bank = json.loads((ROOT / "research/t34_round0d/BANK_ROUND0D.json").read_text())
    row = bank["excluded_preflight_rows"][1]
    full = dense_trace(row, CHECKPOINT, frames=25, device="cpu", min_length=5)
    assert full["teacher_v2"]["batches"] >= 1
    assert all(label["horizon"] <= 25 - label["decision"] for label in full["labels"])
    assert len(full["actions"]) == 25 or full["actions"]


# --------------------------------------------------------------------------- re-scope integrity


def test_ceiling_breach_on_settled_charges_seals_inconclusive_not_kill(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    history.seal_ceiling(value, {"settled_units": 99.0, "ceiling": 90.0})
    _git(author, "add", value["history_ledger"]["path"])
    _git(author, "commit", "-m", "ledger")
    _git(author, "push", "origin", "main")
    with pytest.raises(history.CapExpiredHistory):
        history.require_chunks([], value)
    sealed = json.loads((Path(value["study_root"]) / "ceiling/phase_terminal.json").read_text())
    assert sealed["status"] == "INCONCLUSIVE"


def test_two_live_attempts_are_counted_and_shared_probes_do_not_fake_liveness(
    tmp_path: Path,
) -> None:
    import fcntl

    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    root = Path(value["study_root"])
    claims = []
    for number in range(2):
        attempt = root / f"chunks/chunk-{number:03d}/attempt-01"
        history.reserve(attempt, value, world_ids=[f"w{number}"])
        claims.append(history.InflightClaim(attempt))
    assert history.in_flight_count(value) == 2
    claims[0].release()
    assert history.in_flight_count(value) == 1
    # A concurrent shared probe on the dead attempt must not make it look alive.
    dead = root / "chunks/chunk-000/attempt-01"
    with (dead / "inflight.lock").open("a") as other:
        fcntl.flock(other, fcntl.LOCK_SH | fcntl.LOCK_NB)
        assert not history.in_flight(dead)
    claims[1].release()


def test_round0_route_follows_the_go_file_ruling() -> None:
    from research.t34_round0d.registration import round0_route

    base = {"governing": True, "evidence_authority": "TIER_1_DEVELOPMENT", "reason": "x"}
    assert round0_route(dict(base, decision="STOP"), governing=True) == "ROUND0D"
    assert round0_route(dict(base, decision="INCONCLUSIVE"), governing=True) == "ROUND0D"
    assert round0_route(dict(base, decision="GO"), governing=True) == "GO_PLAN_SHELVE_ROUND0D"
    fault = dict(base, decision="STOP", unconditional_faults=[{"fault": "illegal_action"}])
    assert round0_route(fault, governing=True) == "BUG_BRANCH"
    integrity = dict(base, decision="INCONCLUSIVE", evidence_authority="INVALID_OR_INCOMPLETE")
    assert round0_route(integrity, governing=True) == "MECHANICAL_BRANCH"
    kill = dict(base, decision="KILL", evidence_authority="INTEGRITY_FAILURE")
    assert round0_route(kill, governing=True) == "MECHANICAL_BRANCH"
    not_governing = dict(base, decision="STOP", governing=False)
    assert round0_route(not_governing, governing=True) == "NOT_GOVERNING"


def test_imported_round0_and_amendment_bytes_are_pinned() -> None:
    from research.t34_round0d.pins import PINNED, check_pins

    assert "research/t34_round0/analyze.py" in PINNED
    assert "research/t34_round0_amend/rule.py" in PINNED
    check_pins(ROOT)
    amendment = json.loads(
        (ROOT / "research/t34_round0_amend/AMENDMENT_BOOST_FLOOR.json").read_text()
    )
    for name, digest in amendment["registered_sources"].items():
        if name in PINNED:
            assert PINNED[name] == digest


def test_only_labels_py_reads_sealed_exports() -> None:
    allowed = {"labels.py", "runner.py", "validate.py"}
    readers = []
    for path in (ROOT / "research").rglob("*.py"):
        if path.parent.name == "t34_round0d" and path.name in allowed:
            continue
        text = path.read_text(errors="ignore")
        if "round0d" in text and "sealed/" in text:
            readers.append(str(path))
    assert readers == []
