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


class BranchWorld:
    """Deterministic transition seam; search/exact-stop/chooser remain real."""

    def __init__(self, deaths, log=None):
        self.deaths = deaths
        self.log = [] if log is None else log
        self.action = None
        self.frame = 0
        self.respawn_timer = 2

    def clone(self):
        return BranchWorld(self.deaths, self.log)

    def step(self, forced=None):
        if forced is not None:
            self.action = forced
        self.frame += 1
        self.log.append((self.action, self.frame))
        return {"death": self.frame == self.deaths.get(self.action)}


def _proposal(q=None):
    return {"bare": 1, "mask": [True, True, True, True, False, False], "q": q or [3, 0, 4, 4, 0, 0]}


@pytest.mark.parametrize("q,chosen", [([3, 0, 4, 4, 0, 0], 2), ([5, 0, 4, 4, 0, 0], 0)])
def test_search_bare_death_rescue_q_then_index(q, chosen):
    world = BranchWorld({1: 2})
    result = search(world, _proposal(q))
    assert result["chosen"] == chosen
    assert result["evaluation_order"] == [1, chosen]
    assert result["outcomes"][1] == {"survives": False, "frames_survived": 1}
    assert result["outcomes"][chosen] == {"survives": True, "frames_survived": 300}


def test_search_all_die_chooser_and_rescue_only_fallback():
    world = BranchWorld({0: 5, 1: 1, 2: 8, 3: 12})
    result = search(world, _proposal())
    assert result["branches_evaluated"] == 4
    assert result["chosen"] == 1
    assert result["outcomes"][3] == {"survives": False, "frames_survived": 11}


def test_h300_fatal_transition_stops_before_respawn():
    world = BranchWorld({0: 300, 1: 2, 2: 1, 3: 3})
    result = search(world, _proposal())
    assert result["outcomes"][0] == {"survives": False, "frames_survived": 299}
    assert max(t for a, t in world.log if a == 1) == 2
    assert max(t for a, t in world.log if a == 0) == 300


def test_constructor_preserves_external_rng():
    import random

    import numpy as np
    import torch

    from research.watch_lead_p.live import rng_state, set_rng

    before = rng_state()
    expected = (random.random(), np.random.random(), torch.rand(3))
    set_rng(before)
    LiveWorld(protocol.FIXTURE_SEED)
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert actual[:2] == expected[:2]
    assert torch.equal(actual[2], expected[2])
    set_rng(before)


@pytest.mark.parametrize("action", [-1, 6])
def test_forced_action_bounds(action):
    world = LiveWorld(protocol.FIXTURE_SEED)
    with pytest.raises(ValueError, match="index out of range"):
        world.step(action)


def test_forced_dead_nonacting_tick_refused():
    world = LiveWorld(protocol.FIXTURE_SEED)
    world.hero.die()
    world.hero.respawn_timer = 100
    with pytest.raises(ValueError, match="non-acting"):
        world.step(1)


def test_forecast_quantizes_pairs_and_schedules_indivisible_arms():
    from research.watch_lead_p.benchmark import project

    result = project(0.007326245693257079, 2.851223332923837, 3.2827097920235246)
    assert result["old_ten_worker_pair_seconds_corrected"] == pytest.approx(17028.4549056245)
    assert result["projected_seconds"] <= 14400
    assert len(result["worker_loads_seconds"]) == 12
    assert result["projected_seconds"] == max(result["worker_loads_seconds"]) + 1200


def test_recipe_ignores_transport_but_binds_science(monkeypatch):
    original = resume.digest
    before = resume.recipe_fingerprint()
    monkeypatch.setattr(resume, "digest", lambda p: "0" * 64 if p.name == "run.py" else original(p))
    assert resume.recipe_fingerprint() == before
    monkeypatch.setattr(
        resume, "digest", lambda p: "0" * 64 if p.name == "live.py" else original(p)
    )
    assert resume.recipe_fingerprint() != before


@pytest.mark.parametrize("during_submit", [False, True])
def test_signal_pair_reaps_spawned_workers(tmp_path, during_submit):
    import signal
    import subprocess
    import sys
    import time

    import psutil

    root = Path(__file__).resolve().parents[2]
    ledger = tmp_path / "ledger.json"
    ledger.write_text(json.dumps(resume.empty_ledger()))
    script = tmp_path / "signal_probe.py"
    script.write_text("""import os,sys,time,signal
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from research.watch_lead_p import run
base=Path(sys.argv[2])
def slow(index, arm):
    (base / ("active-"+str(os.getpid()))).write_text("started")
    time.sleep(60)
Original=run.concurrent.futures.ProcessPoolExecutor
class Pool(Original):
    def submit(self,*a,**k):
        result=super().submit(*a,**k)
        (base / "pids").write_text(" ".join(str(p.pid) for p in self._processes.values()))
        if sys.argv[3]=="True":
            os.kill(os.getpid(),signal.SIGUSR1)
        return result
    def shutdown(self,*a,**k):
        time.sleep(.3)
        return super().shutdown(*a,**k)
run.worker=slow
run.concurrent.futures.ProcessPoolExecutor=Pool
if __name__=="__main__":
    try:
        run.execute(base/"ledger.json",base,base/"out")
    except InterruptedError:
        pass
""")
    log = (tmp_path / "log").open("w")
    proc = subprocess.Popen(
        [sys.executable, str(script), str(root), str(tmp_path), str(during_submit)],
        stdout=log,
        stderr=log,
    )
    try:
        deadline = time.monotonic() + 30
        marker = tmp_path / "out/interruption.json.complete"
        while time.monotonic() < deadline:
            if during_submit:
                ready = (tmp_path / "pids").exists()
            else:
                ready = len(list(tmp_path.glob("active-*"))) >= 2
            if ready:
                break
            if proc.poll() is not None:
                pytest.fail((tmp_path / "log").read_text())
            time.sleep(0.02)
        else:
            pytest.fail("workers did not start")
        if not during_submit:
            proc.send_signal(signal.SIGUSR1)
        time.sleep(0.03)
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=20)
        assert proc.returncode == 0, (tmp_path / "log").read_text()
        assert marker.exists()
        assert json.loads((tmp_path / "out/interruption.json").read_text())["completed"] == []
        for pid in (tmp_path / "pids").read_text().split():
            assert not psutil.pid_exists(int(pid)), f"unreaped worker {pid}"
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        log.close()
