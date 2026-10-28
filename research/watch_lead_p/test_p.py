"""P's delivery, live continuation, process determinism and terminal-look contracts."""

from __future__ import annotations

import concurrent.futures
import copy
import json
import multiprocessing
from dataclasses import replace
from pathlib import Path

import pytest

from research.bl_served_search.exports import export_json
from research.watch_lead_p import engine, protocol, resume
from research.watch_lead_p.live import LiveWorld, search
from research.watch_lead_p.protocol import Certificate, Identity, accept, decision, statistics


def _identity() -> Identity:
    return Identity("s", 48, 1, "b8", "state", (True,) * 6)


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({}, "accepted"),
        ({"frame": 47}, "pending"),
        ({"frame": 49}, "late"),
        ({"session": "foreign"}, "stale"),
        ({"life": 2}, "stale"),
        ({"checkpoint": "foreign"}, "stale"),
        ({"state": "diverged"}, "divergent"),
        ({"mask": (False,) * 6}, "divergent"),
        ({"transport": "foreign"}, "divergent"),
        ({"fallback": "foreign"}, "divergent"),
    ],
)
def test_identity_delivery(changes: dict, expected: str) -> None:
    target = _identity()
    assert accept(Certificate(0, target, 1, 48, 100), replace(target, **changes)) == expected


def test_late_overflow_and_mask_drop() -> None:
    target = _identity()
    cert = Certificate(0, target, 1, 49, 100)
    assert accept(cert, target) == "late"
    assert accept(replace(cert, root_bytes=protocol.MAX_ROOT_BYTES + 1), target) == "overflow"
    assert accept(replace(cert, ready_frame=48), target, legal=False) == "masked"


def _rows(bare: int = 10, other: int = 8, mass: float = 1.0) -> list[dict]:
    return [
        {
            "arms": {
                "bare": {"deaths": bare, "mass": 100},
                "P48": {"deaths": other, "mass": 100 * mass},
            }
        }
        for _ in range(32)
    ]


def test_statistics_synthetic_known_effect_and_reproducibility() -> None:
    result = statistics(_rows())
    assert result == statistics(_rows())
    assert result["benefit"] == pytest.approx(0.2)
    assert result["benefit_lb90"] == pytest.approx(0.2)
    assert result["benefit_ub90"] == pytest.approx(0.2)
    assert result["mass_lower95"] == result["mass_upper95"] == 0
    assert result["bootstrap_resamples"] == 10000
    assert result["bootstrap_seed"] == 3431
    assert decision(result, valid=True, complete=True) == "GO to real paced development"


@pytest.mark.parametrize("bare,other", [(0, 0), (0, 2)])
def test_zero_baseline_insufficient(bare: int, other: int) -> None:
    result = statistics(_rows(bare, other))
    assert not result["support"]
    assert decision(result, valid=True, complete=True) == "INCONCLUSIVE"


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"valid": False, "complete": True}, "INVALID/INCOMPLETE"),
        ({"valid": True, "complete": False}, "INVALID/INCOMPLETE"),
        ({"valid": True, "complete": True}, "STOP this delayed controller"),
    ],
)
def test_invalid_precedes_stop_and_stop_precedes_go(kwargs: dict, expected: str) -> None:
    metrics = dict(
        support=True,
        benefit=0.2,
        benefit_lb90=0.1,
        benefit_ub90=0.09,
        mass_lower95=0.0,
        mass_upper95=0.0,
    )
    assert decision(metrics, **kwargs) == expected


@pytest.mark.parametrize(
    "mass,expected",
    [(0.94, "STOP this delayed controller"), (0.95, "GO to real paced development")],
)
def test_mass_guard_boundaries(mass: float, expected: str) -> None:
    assert decision(statistics(_rows(mass=mass)), valid=True, complete=True) == expected


def test_no_effect_stop_and_inconclusive_boundary() -> None:
    result = statistics(_rows(other=10))
    assert decision(result, valid=True, complete=True) == "STOP this delayed controller"
    result.update(benefit=0.10, benefit_lb90=0.0, benefit_ub90=0.10)
    assert decision(result, valid=True, complete=True) == "INCONCLUSIVE"


def test_snapshot_rng_carry_callbacks_and_search_are_isolated() -> None:
    world = LiveWorld(protocol.FIXTURE_SEED)
    for _ in range(8):
        world.step()
    before = world.digest()
    payload = world.snapshot()
    clone = LiveWorld.restore(payload)
    assert clone.digest() == before
    assert len(payload) <= protocol.MAX_ROOT_BYTES
    assert clone.session.game is not world.session.game
    assert clone.session.policy.net is not world.session.policy.net
    clone.session.game.frame += 1
    assert clone.hero._get_frame() == clone.frame
    assert world.digest() == before
    clone = world.clone()
    assert clone.session.policy.net is world.session.policy.net
    assert clone.session.watch_opponent_policy.dqn is world.session.watch_opponent_policy.dqn
    assert clone.hero.safety_veto is None
    assert world.proposal() == clone.step()
    assert world.digest() == before
    result = search(world, world.proposal())
    assert result["branches_evaluated"] >= 1
    assert world.digest() == before


def test_actual_respawn_life_identity() -> None:
    world = LiveWorld(protocol.FIXTURE_SEED)
    world.hero.die()
    world.hero.respawn_timer = 1
    world.waiting = True
    event = world.step()
    assert event["alive"]
    assert world.life == 1
    assert len(world.session.game.snakes) == 6


def _process_trace(seed: int) -> dict:
    return engine.run_arm(seed, "bare", frames=40)


def test_determinism_across_spawned_processes() -> None:
    serial = [_process_trace(protocol.FIXTURE_SEED)] * 2
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(_process_trace, protocol.FIXTURE_SEED) for _ in range(2)]
        parallel = [future.result(timeout=120) for future in futures]
    assert serial == parallel


def _delayed_trace(seed: int) -> dict:
    world = LiveWorld(seed)
    cert, detail = engine.submit(world, "paired-test", 48)
    assert cert is not None
    source = world.digest()
    for _ in range(48):
        world.step()
    target = engine.identity(world, "paired-test", world.proposal())
    assert accept(cert, target) == "accepted"
    assert target.state != source
    world.step(cert.action)
    return {
        "target": target.record(),
        "action": cert.action,
        "result": detail["result"],
        "after": world.digest(),
    }


def test_shadow_target_matches_actual_across_processes() -> None:
    serial = _delayed_trace(protocol.FIXTURE_SEED)
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=2, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [pool.submit(_delayed_trace, protocol.FIXTURE_SEED) for _ in range(2)]
        assert [future.result(timeout=120) for future in futures] == [serial, serial]


def test_zero_death_baseline_does_not_waive_mass_stop() -> None:
    assert (
        decision(statistics(_rows(0, 0, 0.90)), valid=True, complete=True)
        == "STOP this delayed controller"
    )


class FakeWorld:
    def __init__(self, _seed: int):
        from types import SimpleNamespace

        self.frame = 0
        self.life = 0
        self.waiting = False
        self.hero = SimpleNamespace(length=600, is_alive=True)
        self.roster = {}

    def digest(self) -> str:
        return str(self.frame)

    def proposal(self) -> dict:
        return {"mask": [True] * 6, "bare": 1, "life": self.life, "q": [0] * 6}

    def step(self, action=None) -> dict:
        self.frame += 1
        return {"mass": 600, "death": False, "action": 1 if action is None else action}


@pytest.mark.parametrize("arm,expected", [("P0", [0, 5, 10]), ("P48", [0, 49, 98])])
def test_w1k1_fixed_submission_and_target_action(monkeypatch, arm, expected) -> None:
    submissions = []

    def submit(world, session, delay):
        submissions.append(world.frame)
        target = Identity(
            session, world.frame + delay, 0, engine.B8_SHA256, str(world.frame + delay), (True,) * 6
        )
        return Certificate(world.frame, target, 2, target.frame, 1), {}

    monkeypatch.setattr(engine, "LiveWorld", FakeWorld)
    monkeypatch.setattr(engine, "submit", submit)
    result = engine.run_arm(1, arm, frames=12 if arm == "P0" else 100)
    assert submissions == expected
    assert result["counts"]["changed"] == (3 if arm == "P0" else 2)
    assert result["counts"]["bare_on_miss"] == (9 if arm == "P0" else 98)


def test_resume_blind_integrity_foreign_namespace_and_duplicates(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    path = inputs / "units/session-00.json"
    export_json(path, {"sealed": "content not parsed by ledger builder"})
    ledger = tmp_path / "ledger.json"
    resume.make_ledger([path], ledger)
    assert list(resume.read_ledger(ledger, inputs)) == [0]
    data = json.loads(ledger.read_text())
    for key in ("namespace", "allocation"):
        broken = copy.deepcopy(data)
        broken[key] = "foreign"
        bad = tmp_path / f"{key}.json"
        bad.write_text(json.dumps(broken))
        with pytest.raises(ValueError, match="foreign"):
            resume.read_ledger(bad, inputs)
    data["units"] *= 2
    bad = tmp_path / "duplicate.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        resume.read_ledger(bad, inputs)
    path.write_text("corrupt")
    with pytest.raises(ValueError):
        resume.read_ledger(ledger, inputs)


def test_partial_union_does_not_take_a_look(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(resume, "statistics", lambda *a: pytest.fail("partial look"))
    result = resume.aggregate([], tmp_path / "invalid.json")
    assert result["decision"] == "INVALID/INCOMPLETE"
    assert result["P48_primary"] == result["P0_descriptive_only"] == {}
