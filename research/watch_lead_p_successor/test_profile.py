"""Outcome boundary, disjoint allocation, exact path and identity smoke helpers."""

import pytest

from research.watch_lead_p.protocol import FIXTURE_SEED, SEEDS
from research.watch_lead_p_successor.host import SCRIPT, require_profile
from research.watch_lead_p_successor.prepare import audit_bank, bank_record
from research.watch_lead_p_successor.profile import measurement


def test_fresh_bank_excludes_closed_P_and_fixture() -> None:
    bank = bank_record()
    assert not set(bank["seeds"]) & (set(SEEDS) | {FIXTURE_SEED})
    assert len(set(bank["seeds"])) == 4
    assert bank["successor_eligible"] is False
    assert audit_bank()["collisions"] == []


def test_export_allowlist_discards_outcomes() -> None:
    row = dict(
        seed=1,
        arm="P0",
        frames=64,
        trace_sha256="opaque",
        initial_state="opaque",
        final_state="opaque",
        deaths=19,
        mass=999,
        counts={"changed": 9},
        roots=[{"outcomes": "SECRET"}],
    )
    output = measurement(row, 8, 6, 4, 2)
    assert not set(output) & {"deaths", "mass", "counts", "roots"}
    assert "SECRET" not in repr(output)
    assert output["seconds_per_frame"] == pytest.approx(0.125)
    assert output["cpu_percent_one_core"] == 75


def test_prepare_allocation_is_deterministic() -> None:
    assert bank_record() == bank_record()
    assert bank_record()["namespace"] != "a100-dev/t34-watch-lead-1028/"


@pytest.mark.parametrize(
    "change",
    [
        None,
        {"script": "foreign.sh"},
        {"timeout_minutes": 40},
        {"canary": {}},
        {"device": "cuda"},
        {"p_profile_source_delivery": "other"},
        {"sandbox_admission": {}},
    ],
)
def test_source_adapter_is_narrow(change: dict | None) -> None:
    from research.watch_lead_p_successor.prepare import ROOT

    manifest = {
        "p_profile_source_delivery": "background-sparse-v1",
        "script": SCRIPT,
        "device": "cpu",
        "timeout_minutes": 30,
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"]},
        "canary": {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT},
        "resume_exemption": {"kind": "short-canary"},
        "env": {"SNAKE_COLAB_CANARY": "1"},
    }
    if change is None:
        require_profile(manifest, ROOT)
    else:
        with pytest.raises(ValueError):
            require_profile({**manifest, **change}, ROOT)


class TinyWorld:
    """Hermetic mid-session control-path fixture, not a gameplay/timing oracle."""

    stale = False

    def __init__(self, seed: int) -> None:
        self.seed, self.frame, self.life, self.waiting = seed, 0, 0, False
        self.actions = []
        self.roster = {"fixture": "hermetic"}

    @property
    def hero(self) -> object:
        from types import SimpleNamespace

        return SimpleNamespace(is_alive=True, length=600 if self.frame >= 3 else 10)

    def digest(self) -> str:
        return repr((self.seed, self.frame, self.life, self.actions))

    def snapshot(self) -> bytes:
        import pickle

        return pickle.dumps(self)

    @staticmethod
    def restore(value: bytes) -> object:
        import pickle

        return pickle.loads(value)

    def proposal(self) -> dict:
        return {"bare": 0, "q": [1.0] * 6, "mask": [True] * 6, "life": self.life}

    def step(self, forced: int | None = None) -> dict:
        self.actions.append(0 if forced is None else forced)
        self.frame += 1
        if self.stale and self.frame == 5:
            self.life += 1
        return {"bare": 0, "mass": 100, "death": False, "frame": self.frame}


@pytest.mark.parametrize("arm", ["bare", "P0", "P48"])
def test_mid_session_exact_engine_suffix(monkeypatch: pytest.MonkeyPatch, arm: str) -> None:
    from research.watch_lead_p import engine

    made = []
    payload_world = TinyWorld(7)
    for _ in range(3):
        payload_world.step()
    payload = payload_world.snapshot()

    class Factory:
        restore = staticmethod(TinyWorld.restore)
        prefix = False

        def __new__(cls, seed: int) -> TinyWorld:
            world = TinyWorld.restore(payload) if cls.prefix else TinyWorld(seed)
            made.append(world)
            return world

    searches = []

    def fast_search(world: TinyWorld, proposal: dict) -> dict:
        searches.append(world.frame)
        return {"chosen": 1}

    monkeypatch.setattr(engine, "LiveWorld", Factory)
    monkeypatch.setattr(engine, "search", fast_search)
    natural = engine.run_arm(7, arm, frames=67)
    Factory.prefix = True
    window = engine.run_arm(7, arm, frames=64)
    assert natural["final_state"] == window["final_state"]
    assert made[0].actions[3:] == made[1].actions[3:]
    if arm != "bare":
        assert searches
        assert window["counts"]["accepted"] > 0


def test_submit_without_search_is_not_support(monkeypatch: pytest.MonkeyPatch) -> None:
    import faulthandler

    from research.watch_lead_p import engine
    from research.watch_lead_p_successor import profile

    monkeypatch.setattr(TinyWorld, "stale", True)
    monkeypatch.setattr(profile, "LiveWorld", TinyWorld)
    monkeypatch.setattr(engine, "LiveWorld", TinyWorld)
    monkeypatch.setattr(engine, "search", lambda *_args: {"chosen": 1})
    world = TinyWorld(7)
    for _ in range(3):
        world.step()
    try:
        row = profile.task(7, "P48", 1, world.snapshot())
        assert row["submit_calls"] == 1 and row["search_calls"] == 0
        assert row["search_seconds"] == 0
    finally:
        faulthandler.cancel_dump_traceback_later()


def test_foreign_bank_refuses_before_output(tmp_path) -> None:
    from research.watch_lead_p_successor.profile import execute

    bank = tmp_path / "bank.json"
    bank.write_text("{}")
    output = tmp_path / "out"
    with pytest.raises(ValueError, match="foreign"):
        execute(bank, output)
    assert not output.exists()


def test_parent_signal_reaps_only_owned_worker() -> None:
    import multiprocessing
    import signal
    import time

    from research.watch_lead_p_successor.profile import coordinator_signals, reap

    process = multiprocessing.get_context("spawn").Process(target=time.sleep, args=(60,))
    process.start()
    old = signal.getsignal(signal.SIGUSR1)
    with pytest.raises(InterruptedError):
        with coordinator_signals():
            try:
                signal.raise_signal(signal.SIGUSR1)
            finally:
                reap([process])
    assert not process.is_alive()
    assert signal.getsignal(signal.SIGUSR1) == old


def test_overlap_concurrency_is_measured() -> None:
    from research.watch_lead_p_successor.profile import concurrency

    rows = [{"start_elapsed": 0, "end_elapsed": 4}, {"start_elapsed": 2, "end_elapsed": 6}]
    concurrency(rows)
    assert [r["mean_concurrent_tasks"] for r in rows] == [1.5, 1.5]
