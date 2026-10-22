"""Untimed, mocked contract tests for the PLAN_v3 C1 development runner.

These tests never open a real Watch session and never invoke ``--check``,
``--make-ledger`` or ``--run`` against the owner's artifact root.  They bind
the safety boundaries that must be true before Claude is allowed to operate
the bank.
"""

from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.bl_served_search import c1, c1_bank, c1_reduce, c1_watch
from research.bl_served_search.exports import export_json


def _allocation_data(count: int = 24) -> dict:
    rows = c1_bank.allocation()[:count]
    return {"identity": "test-c1-identity", "allocation": rows}


def _write_unit(root: Path, unit_id: str, value: object | None = None) -> dict:
    path = root / f"{unit_id}.json"
    export_json(path, value if value is not None else {"unit_id": unit_id})
    descriptor = c1.inventory(path, unit_id)
    row = next(r for r in c1_bank.allocation() if r["unit_id"] == unit_id)
    export_json(
        path.with_name(path.stem + ".ops.json"),
        {
            "schema": c1.SCHEMA + "/completed-pair",
            "namespace": c1_bank.NAMESPACE,
            "identity": "test-c1-identity",
            "allocation": row,
            "payload": descriptor,
        },
    )
    return descriptor


def _ledger_payload(data: dict, descriptors: list[dict]) -> dict:
    return {
        "schema": c1.SCHEMA + "/ledger",
        "identity": data["identity"],
        "units": descriptors,
    }


def _freshness_roots(tmp_path: Path) -> tuple[Path, Path, dict[int, int]]:
    repo = tmp_path / "repo"
    artifacts = tmp_path / "artifacts"
    for name in ("research", "docs", "src", "scripts"):
        (repo / name).mkdir(parents=True)
    controls: dict[int, int] = {}
    paths = (
        "research/t34_round0/BANK_ROUND0.json",
        "research/t34_round0d/BANK_ROUND0D.json",
        "research/t34_pathb/RETENTION_BANK_V2.json",
        "research/phase0_search_20261019/WORLD_BANK.json",
    )
    for index, relative in enumerate(paths):
        value = 4_000_000 + index
        controls[index] = value
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"world_seed": value}) + "\n")
    for index in range(7):
        path = artifacts / f"frp-{index}" / "namespaces" / "training_exclusions.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([5_000_000 + index]) + "\n")
    return repo, artifacts, controls


def test_allocation_is_deterministic_unique_and_frozen() -> None:
    first = c1_bank.allocation()
    second = c1_bank.allocation()
    assert first == second
    assert len(first) == 24
    assert [row["unit_id"] for row in first] == [f"session-{i:02d}" for i in range(24)]
    assert len({row["seed"] for row in first}) == 24
    assert all(row["horizon"] == 14_400 for row in first)


def test_freshness_scans_positive_controls_and_accepts_disjoint_temp_bank(tmp_path: Path) -> None:
    repo, artifacts, controls = _freshness_roots(tmp_path)
    proof = c1_bank.freshness(repo=repo, artifacts=artifacts)
    assert proof["status"] == "CLEAR"
    assert proof["count"] == 24
    assert set(proof["controls"][name]["positive_control"] for name in proof["controls"]) == set(
        controls.values()
    )
    assert proof["training_inventory_count"] == 7


def test_freshness_fails_closed_on_candidate_collision_and_scan_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, artifacts, _ = _freshness_roots(tmp_path)
    candidate = c1_bank.allocation()[0]["seed"]
    collision = repo / "research/t34_round0/BANK_ROUND0.json"
    collision.write_text(json.dumps({"world_seed": candidate}) + "\n")
    with pytest.raises(ValueError, match="collision"):
        c1_bank.freshness(repo=repo, artifacts=artifacts)

    repo, artifacts, _ = _freshness_roots(tmp_path / "scan-error")

    def failed_scan(*args: object, **kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(returncode=2, stdout="", stderr="permission denied")

    monkeypatch.setattr(c1_bank.subprocess, "run", failed_scan)
    with pytest.raises(ValueError, match="freshness read failure"):
        c1_bank.freshness(repo=repo, artifacts=artifacts)


def test_verify_ledger_rejects_foreign_duplicate_and_corrupt_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(c1_bank, "STUDY_ROOT", study)
    data = _allocation_data(2)
    descriptors = [_write_unit(study, row["unit_id"]) for row in data["allocation"]]
    ledger = study / "ledger.json"

    export_json(ledger, _ledger_payload(data, descriptors))
    assert set(c1.verify_ledger(ledger, data)) == {"session-00", "session-01"}

    foreign = copy.deepcopy(descriptors)
    foreign[0]["unit_id"] = "session-99"
    export_json(study / "foreign.json", _ledger_payload(data, foreign))
    with pytest.raises(ValueError, match="foreign or duplicate"):
        c1.verify_ledger(study / "foreign.json", data)

    duplicate = [descriptors[0], descriptors[0]]
    export_json(study / "duplicate.json", _ledger_payload(data, duplicate))
    with pytest.raises(ValueError, match="foreign or duplicate"):
        c1.verify_ledger(study / "duplicate.json", data)

    corrupt = copy.deepcopy(descriptors)
    corrupt[0]["size"] += 1
    export_json(study / "corrupt.json", _ledger_payload(data, corrupt))
    with pytest.raises(ValueError, match="path/size/hash mismatch"):
        c1.verify_ledger(study / "corrupt.json", data)


def test_recovery_allows_exactly_one_verified_second_attempt_and_preserves_cap(
    tmp_path: Path,
) -> None:
    study = tmp_path / "study"
    study.mkdir()
    data = _allocation_data(1)
    descriptor = _write_unit(study, "session-00")
    ledger = {"session-00": descriptor}
    previous = study / "attempt-01-receipt.json"
    export_json(
        previous,
        {
            "schema": c1.SCHEMA + "/attempt",
            "identity": data["identity"],
            "attempt_number": 1,
            "status": "INCOMPLETE",
            "verified_stop": True,
            "mechanical_failure": True,
            "completed_unit_exports": ledger,
            "charged_seconds": 123.5,
        },
    )
    assert c1.recovery(previous, ledger, data) == (2, 123.5)
    with pytest.raises(ValueError, match="nonempty ledger"):
        c1.recovery(None, ledger, data)

    exhausted = study / "exhausted.json"
    export_json(
        exhausted,
        {
            "schema": c1.SCHEMA + "/attempt",
            "identity": data["identity"],
            "attempt_number": 1,
            "status": "INCOMPLETE",
            "verified_stop": True,
            "mechanical_failure": True,
            "completed_unit_exports": ledger,
            "charged_seconds": c1.CAP_SECONDS,
        },
    )
    with pytest.raises(ValueError, match="exhausted cumulative"):
        c1.recovery(exhausted, ledger, data)


def test_single_look_refuses_partial_bank_and_double_open(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    study = tmp_path / "study"
    study.mkdir()
    monkeypatch.setattr(c1_bank, "STUDY_ROOT", study)
    data = _allocation_data()
    descriptors = {
        row["unit_id"]: _write_unit(study, row["unit_id"], {"unit_id": row["unit_id"]})
        for row in data["allocation"]
    }
    monkeypatch.setattr(c1, "verify_unit", lambda *args: None)
    monkeypatch.setattr(c1_reduce, "reduce_sessions", lambda units: {"verdict": "INCONCLUSIVE"})

    partial = dict(list(descriptors.items())[:-1])
    with pytest.raises(ValueError, match="no partial"):
        c1.single_look(data, partial)
    assert not (study / "look-started.json").exists()

    result = c1.single_look(data, descriptors)
    assert result["verdict"] == "INCONCLUSIVE"
    assert (study / "look-started.json.complete").is_file()
    with pytest.raises(FileExistsError):
        c1.single_look(data, descriptors)


class _Root:
    def __init__(self, value: int = 0) -> None:
        self.value = value


def test_dense_batch_runs_exact_stop_all_legal_actions_on_independent_clones(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clones: list[_Root] = []

    def clone(root: _Root) -> _Root:
        result = _Root(root.value)
        clones.append(result)
        return result

    monkeypatch.setattr(c1_watch, "_clone_root", clone)
    monkeypatch.setattr(c1_watch, "_root_bytes", lambda root: str(root.value).encode())

    class Stats:
        def as_dict(self) -> dict[str, int]:
            return {
                "kernel_rows": 6,
                "stock_rows": 0,
                "fallback_rows": 0,
                "noncandidate_rows": 0,
            }

    @contextmanager
    def native(mode: str, stats: Stats):
        assert mode == "kernel"
        yield

    monkeypatch.setattr(c1_watch, "RowStats", Stats)
    monkeypatch.setattr(c1_watch, "configure", native)

    seen: list[int] = []

    def rollout(root: _Root, actions: list[int], *_args: object, **_kwargs: object) -> dict:
        assert len(actions) == 1
        seen.append(root.value)
        root.value += 1
        return {actions[0]: {"survives": False, "frames_executed": 4}}

    monkeypatch.setattr(c1_watch, "rollout_all", rollout)
    result = c1_watch.dense_batch(
        _Root(), 0, np.arange(6, 0, -1, dtype=np.float32), np.ones(6, bool)
    )
    assert result["evaluation_order"] == list(range(6))
    assert result["branches_evaluated"] == 6
    assert result["branches_skipped"] == 0
    assert seen == [0] * 6
    # Each branch receives a fresh root at value zero, then mutates only its
    # private copy.  The ``seen`` assertion above proves no branch saw prior
    # carry; these values prove the branch mutation did not leak sideways.
    assert clones[0].value == 0  # parent clone is untouched by every branch
    assert all(root.value == 1 for root in clones[1:])


def test_dense_batch_rejects_zero_kernel_or_any_stock_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(c1_watch, "_clone_root", lambda root: copy.copy(root))
    monkeypatch.setattr(c1_watch, "_root_bytes", lambda root: b"root")

    class Stats:
        def as_dict(self) -> dict[str, int]:
            return {
                "kernel_rows": 0,
                "stock_rows": 1,
                "fallback_rows": 1,
                "noncandidate_rows": 0,
            }

    @contextmanager
    def native(_mode: str, _stats: Stats):
        yield

    monkeypatch.setattr(c1_watch, "RowStats", Stats)
    monkeypatch.setattr(c1_watch, "configure", native)
    monkeypatch.setattr(
        c1_watch,
        "rollout_all",
        lambda _root, actions, *_args, **_kwargs: {
            actions[0]: {"survives": False, "frames_executed": 3}
        },
    )
    with pytest.raises(ValueError, match="native kernel"):
        c1_watch.dense_batch(_Root(), 0, np.arange(6, dtype=np.float32), np.ones(6, bool))


def _attested_unit(*, frames_executed: int = 300, kernel_rows: int = 1) -> dict:
    from web.backend.served_checkpoint import B8_S3_SHA256, FRP3_S12_SHA256

    batch = {
        "frame": 1,
        "life": 1,
        "bare": 0,
        "chosen": 0,
        "native_v8": {
            "kernel_rows": kernel_rows,
            "stock_rows": 0,
            "fallback_rows": 0,
            "noncandidate_rows": 0,
        },
        "evaluation_order": [0],
        "branches_evaluated": 1,
        "branches_skipped": 5,
        "outcomes": {
            "0": {"survives": True, "frames_executed": frames_executed},
            **{str(action): {"status": "NOT_EVALUATED"} for action in range(1, 6)},
        },
    }
    roster = {
        "hero_checkpoint_sha256": B8_S3_SHA256,
        "opponent_checkpoint": {"sha256": FRP3_S12_SHA256},
        "snakes": [{"id": index} for index in range(6)],
    }
    frames = [
        {
            "frame": frame,
            "life": 1,
            "alive": frame == 1,
            "death": False,
            "mass": 10 if frame == 1 else 0,
        }
        for frame in range(1, c1_bank.HORIZON + 1)
    ]

    def decisions(*, first_eligible: bool) -> list[dict]:
        return [
            {
                "frame": 1,
                "life": 1,
                "length": 600 if first_eligible else 599,
                "bare": 0,
                "chosen": 0,
                "q": [6, 5, 4, 3, 2, 1],
                "advisory": [True] * 6,
            }
        ]

    arm = {
        "mirror_identity_frames": c1_bank.HORIZON,
        "roster": roster,
        "frames": frames,
        "decisions": decisions(first_eligible=False),
        "batches": [],
    }
    dense = {
        "mirror_identity_frames": c1_bank.HORIZON,
        "roster": roster,
        "frames": frames,
        "decisions": decisions(first_eligible=True),
        "batches": [batch],
    }
    return {
        "unit_id": "session-00",
        "seed": c1_bank.allocation()[0]["seed"],
        "horizon": c1_bank.HORIZON,
        "schema": c1.SCHEMA + "/unit",
        "identity": "test-c1-identity",
        "namespace": c1_bank.NAMESPACE,
        "worker_threads": {
            "environment": dict.fromkeys(c1.ceiling.THREAD_ENV, "1"),
            "torch": 1,
            "numba": 1,
        },
        "nice": 10,
        "arms": {"bare": arm, "dense": dense},
    }


def test_verify_unit_enforces_per_decision_native_attestation_and_h300_cap() -> None:
    expected = {"unit_id": "session-00", "seed": c1_bank.allocation()[0]["seed"], "horizon": 14_400}
    data = {"identity": "test-c1-identity"}
    c1.verify_unit(_attested_unit(), expected, data)
    with pytest.raises(ValueError, match="native .*attestation"):
        c1.verify_unit(_attested_unit(kernel_rows=0), expected, data)
    with pytest.raises(ValueError, match="short H300"):
        c1.verify_unit(_attested_unit(frames_executed=301), expected, data)


def test_verify_unit_rejects_missing_live_decision_from_own_denominator() -> None:
    expected = {
        "unit_id": "session-00",
        "seed": c1_bank.allocation()[0]["seed"],
        "horizon": 14_400,
    }
    data = {"identity": "test-c1-identity"}
    value = _attested_unit()
    value["arms"]["dense"]["decisions"] = []
    with pytest.raises(ValueError, match="denominator"):
        c1.verify_unit(value, expected, data)


@pytest.mark.parametrize("horizon", [0, 14_401])
def test_run_arm_rejects_any_non_frozen_frame_cap_before_opening_watch(
    horizon: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed cap must fail before a real Watch constructor is reachable."""
    import research.t34_pathb.adapter_v2 as adapter

    def opened(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("open_watch reached before frozen frame-cap validation")

    monkeypatch.setattr(adapter, "open_watch", opened)
    with pytest.raises(ValueError, match="14400|horizon|frame"):
        c1_watch.run_arm(123, False, lambda: None, horizon=horizon)


def test_run_arm_mocked_full_horizon_tracks_death_dead_interval_and_respawn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise the real 14,400-tick bridge without constructing a game."""
    import torch

    import research.t15_mixed_roster_20261017.episodes as episodes
    import research.t34_pathb.adapter_v2 as adapter
    import src.simd_env.ego_raster as ego_raster
    import src.simd_env.eval_engine as eval_engine

    class Hero:
        id = 0
        length = 600
        is_alive = True
        last_action = 0
        auto_respawn = True

        def update(self) -> None:
            return None

    hero = Hero()

    class LiveGame:
        snakes = [hero]
        frame_death_causes: dict[int, str] = {}

    class Policy:
        def _next_dispatch_q(self) -> torch.Tensor:
            return torch.tensor([6, 5, 4, 3, 2, 1], dtype=torch.float32)

    class Live:
        game = LiveGame()
        policy = Policy()
        pathb = None

        def snapshot(self) -> None:
            return None

    class Sim:
        S = 1
        allow_respawn = True

        def __init__(self) -> None:
            self.frame = 0
            self.alive = np.ones((1, 1), dtype=bool)
            self.length = np.full((1, 1), 600, dtype=np.int64)
            self.transition = False

        def get_advisory_action_mask(self) -> np.ndarray:
            return np.ones((1, 1, 6), dtype=bool)

        def get_resolved_action_mask(self) -> np.ndarray:
            return np.ones((1, 1), dtype=bool)

        def get_transition_valid(self) -> np.ndarray:
            return np.asarray([[self.transition]], dtype=bool)

        def step_with_policy(self, callback: object) -> None:
            self.frame += 1
            self.transition = self.frame == 2
            self.alive[0, 0] = self.frame not in (3, 4)
            callback(self)
            if self.frame == 2:
                self.alive[0, 0] = False

    sim = Sim()

    class Runtime:
        def prepare(self, _sim: Sim) -> None:
            return None

        def observe_step(self, _sim: Sim) -> None:
            return None

    world = SimpleNamespace(
        sim=sim,
        runtime=Runtime(),
        hero=object(),
        opponents=object(),
    )

    class Session:
        def __init__(self) -> None:
            self.world = world
            self.live = Live()

        def _step_live(self, actions: np.ndarray) -> None:
            active = sim.frame not in (3, 4)
            if active:
                hero.update()
                hero.last_action = int(actions[0, 0])
                # Frame 2 is the one physical death; frames 3 and 4 are the
                # respawn gap, then frame 5 starts the next life.
                if sim.frame == 2:
                    hero.is_alive = False
                    self.live.game.frame_death_causes = {hero.id: "collision"}
                else:
                    hero.is_alive = True
                    self.live.game.frame_death_causes = {}
            else:
                hero.is_alive = False
                self.live.game.frame_death_causes = {}

            # The bridge installs an observer around this method.  Calling the
            # policy seam here proves the live/mirror Q comparison was active.
            if active:
                self.live.policy._next_dispatch_q()

        def snapshot(self) -> None:
            return None

        def close(self) -> None:
            return None

    session = Session()
    monkeypatch.setattr(adapter, "open_watch", lambda *_args: session)
    monkeypatch.setattr(episodes, "_assert_mixed", lambda *_args: None)
    monkeypatch.setattr(
        episodes,
        "roster_identity",
        lambda _live: {
            "hero_checkpoint_sha256": "b8",
            "opponent_checkpoint": {"sha256": "frp3"},
            "snakes": [{"id": index} for index in range(6)],
        },
    )
    monkeypatch.setattr(ego_raster, "build_ego_raster", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(
        c1_watch, "prepared_q", lambda *_args, **_kwargs: np.array([6, 5, 4, 3, 2, 1])
    )

    def dispatch(_sim: Sim, _mask: np.ndarray, actions: np.ndarray, *_args: object) -> None:
        actions[0, 0] = 0

    monkeypatch.setattr(eval_engine, "_dispatch_actions", dispatch)

    result = c1_watch.run_arm(123, False, lambda: None)
    assert len(result["frames"]) == c1_bank.HORIZON
    assert result["frames"][1]["death"] is True
    assert result["frames"][2]["alive"] is False
    assert result["frames"][3]["alive"] is False
    assert result["frames"][4]["alive"] is True
    assert len(result["decisions"]) == c1_bank.HORIZON - 2
    assert result["decisions"][0]["life"] == 1
    assert result["decisions"][1]["life"] == 1
    assert result["decisions"][2]["life"] == 2
    assert result["batches"] == []


def test_completed_pair_metadata_refuses_foreign_namespace_before_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(c1_bank, "STUDY_ROOT", tmp_path)
    data = _allocation_data()
    _write_unit(tmp_path, "session-00")
    ops = tmp_path / "session-00.ops.json"
    value = c1.read(ops)
    # A separate immutable foreign completion, not an edit to an evidence file.
    foreign_root = tmp_path / "foreign"
    foreign_root.mkdir()
    raw = foreign_root / "session-00.json"
    export_json(raw, {"foreign": True})
    foreign = c1.inventory(raw, "session-00")
    export_json(
        foreign_root / "session-00.ops.json", {**value, "namespace": "foreign", "payload": foreign}
    )
    ledger = tmp_path / "foreign-ledger.json"
    export_json(ledger, _ledger_payload(data, [foreign]))
    with pytest.raises(ValueError, match="foreign namespace"):
        c1.verify_ledger(ledger, data)
