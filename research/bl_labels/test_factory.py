"""Independent F0 contract checks.

These tests deliberately keep the simulator out of the fixture path.  They verify
the immutable unit and canary admission boundaries that protect the exact-sim
factory from accidental resume, namespace, or budget drift.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from research.bl_labels.exports import (
    aggregate_attempts,
    build_ledger,
    empty_ledger,
    load_ledger,
    publish_json,
    publish_life,
    publish_session,
    publish_unit,
)
from research.bl_labels.manifests import canary_allocation, make_manifest, namespace


def _allocation() -> list[dict[str, object]]:
    return canary_allocation()


def test_canary_allocation_is_exactly_the_registered_population() -> None:
    rows = _allocation()
    assert len(rows) == 36
    assert {row["stream"] for row in rows} == {
        "gate-frozen",
        "gate-mixed",
        "gate-scripted",
        "watch6",
    }
    assert sum(row["stream"] == "gate-frozen" for row in rows) == 12
    assert sum(row["stream"] == "gate-mixed" for row in rows) == 12
    assert sum(row["stream"] == "gate-scripted" for row in rows) == 8
    assert sum(row["stream"] == "watch6" for row in rows) == 4
    assert {row["horizon"] for row in rows if row["stream"] != "watch6"} == {5000}
    assert {row["horizon"] for row in rows if row["stream"] == "watch6"} == {14400}
    assert len({row["allocation_id"] for row in rows}) == 36


def test_namespace_requires_full_lowercase_commit() -> None:
    commit = "a" * 40
    assert namespace(commit) == f"f0-hindsight-labels-{commit}-cpu-canary"
    for bad in ("a" * 39, "A" * 40, "g" * 40, "a" * 41):
        with pytest.raises(ValueError):
            namespace(bad)


def test_empty_ledger_and_publish_unit_round_trip_is_resume_safe(tmp_path: Path) -> None:
    allocation = _allocation()[:2]
    name = "f0-hindsight-labels-" + "b" * 40 + "-cpu-canary"
    ledger = tmp_path / "verified-ledger.json"
    publish_json(ledger, empty_ledger(name, allocation))
    assert load_ledger(ledger, name, allocation) == set()

    value = {
        "schema": "bl-labels-f0-unit/v1",
        "namespace": name,
        "allocation": allocation[0],
        "rows": [],
        "snapshots": [],
        "counters": {"rescue_bearing": 0},
    }
    path = publish_unit(tmp_path, name, allocation, value)
    assert path == tmp_path / "lives" / "gate-frozen-000.json"
    payload = path.read_bytes()
    assert path.with_name(path.name + ".complete").read_text() == "complete\n"
    digest = hashlib.sha256(payload).hexdigest()
    assert path.with_name(path.name + ".sha256").read_text() == f"{digest}  {path.name}\n"

    ledger_value = build_ledger(tmp_path, tmp_path / "ledger-built.json", name, allocation)
    assert ledger_value["units"] == [
        {
            "unit_id": "gate-frozen-000",
            "path": "lives/gate-frozen-000.json",
            "sha256": digest,
            "size": len(payload),
        }
    ]
    publish_json(tmp_path / "ledger-built.json", ledger_value)
    assert load_ledger(tmp_path / "ledger-built.json", name, allocation) == {"gate-frozen-000"}


def test_build_ledger_to_distinct_output_directory_remains_loadable(tmp_path: Path) -> None:
    allocation = [{"allocation_id": "watch6-000", "seed": 1}]
    name = "f0-hindsight-labels-" + "d" * 40 + "-cpu-canary"
    source = tmp_path / "source"
    publish_unit(
        source,
        name,
        allocation,
        {
            "schema": "bl-labels-f0-unit/v1",
            "namespace": name,
            "allocation": allocation[0],
            "rows": [],
        },
    )
    ledger = tmp_path / "ledger-output" / "verified-ledger.json"
    build_ledger(source, ledger, name, allocation)
    assert load_ledger(ledger, name, allocation) == {"watch6-000"}


def test_ledger_rejects_foreign_or_duplicate_units(tmp_path: Path) -> None:
    allocation = _allocation()[:2]
    name = "f0-hindsight-labels-" + "c" * 40 + "-cpu-canary"
    ledger = tmp_path / "verified-ledger.json"
    publish_json(ledger, empty_ledger(name, allocation))
    payload = {
        "schema": "bl-labels-f0-unit/v1",
        "namespace": name,
        "allocation": allocation[0],
        "rows": [],
    }
    publish_unit(tmp_path, name, allocation, payload)
    unit = tmp_path / "lives" / "gate-frozen-000.json"
    digest = hashlib.sha256(unit.read_bytes()).hexdigest()
    row = {
        "unit_id": "gate-frozen-000",
        "path": "lives/gate-frozen-000.json",
        "sha256": digest,
        "size": unit.stat().st_size,
    }
    bad = json.loads(ledger.read_text())
    bad["units"] = [row, row]
    bad_ledger = tmp_path / "bad-ledger.json"
    publish_json(bad_ledger, bad)
    with pytest.raises(ValueError, match="foreign or duplicate"):
        load_ledger(bad_ledger, name, allocation)


def test_aggregate_ledger_is_loadable_without_attempt_path_aliases(tmp_path: Path) -> None:
    allocation = [{"allocation_id": "watch6-000", "seed": 1}]
    name = "f0-hindsight-labels-" + "e" * 40 + "-cpu-canary"
    first, second = tmp_path / "first", tmp_path / "second"
    for root, life in ((first, 0), (second, 1)):
        publish_life(
            root,
            name,
            allocation,
            {
                "schema": "bl-labels-f0-life/v1",
                "namespace": name,
                "allocation_id": "watch6-000",
                "allocation": allocation[0],
                "life": life,
                "death_decision": 10,
                "rows": [],
            },
        )
    ledger = tmp_path / "union.json"
    value = aggregate_attempts([first, second], ledger, name, allocation)
    assert len(value["units"]) == 2
    assert load_ledger(ledger, name, allocation) == {
        "watch6-000--life-000000",
        "watch6-000--life-000001",
    }


def test_h1440_reversal_rate_above_ten_percent_uses_delay_label(tmp_path: Path) -> None:
    allocation = [{"allocation_id": "watch6-000", "seed": 1}]
    name = "f0-hindsight-labels-" + "f" * 40 + "-cpu-canary"
    attempt = tmp_path / "attempt"
    publish_life(
        attempt,
        name,
        allocation,
        {
            "schema": "bl-labels-f0-life/v1",
            "namespace": name,
            "allocation_id": "watch6-000",
            "allocation": allocation[0],
            "life": 0,
            "death_decision": 10,
            "rows": [{"decisive": True}],
            "counters": {"deaths": 1, "h1440": 10, "reversals": 2},
        },
    )
    publish_session(
        attempt,
        name,
        allocation,
        {
            "schema": "bl-labels-f0-session/v1",
            "namespace": name,
            "allocation_id": "watch6-000",
            "allocation": allocation[0],
            "status": "COMPLETE",
            "rows": [],
            "counters": {"deaths": 1, "h1440": 10, "reversals": 2},
            "provenance": {},
        },
    )
    ledger = tmp_path / "union.json"
    value = aggregate_attempts([attempt], ledger, name, allocation)
    assert value["mac_aggregate"]["h1440_reversal_rate"] == pytest.approx(0.2)
    assert value["mac_aggregate"]["derived_effective_label"] == "delay"


def test_canary_manifest_binds_cpu_cap_and_resume_contract(tmp_path: Path) -> None:
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    control = tmp_path / "control.json"
    ledger = tmp_path / "verified-ledger.json"
    b8 = tmp_path / "b8.pth"
    frp3 = tmp_path / "frp3.pth"
    publish_json(ledger, empty_ledger(namespace(commit), _allocation()))
    for path in (control, b8, frp3):
        path.write_bytes(b"fixture")
    control.with_name("SEED_AUDIT.json").write_text("{}")
    manifest = make_manifest(commit, tmp_path / "output", control, ledger, b8, frp3)
    assert manifest["requirements"] == {
        "gpu": "cpu",
        "allowed_gpus": ["cpu"],
        "min_vcpu": 1,
        "est_minutes": 60,
    }
    assert manifest["timeout_minutes"] == 120
    assert manifest["budget"]["unit_cap"] == 0.20
    assert manifest["budget"]["rate_units_per_hour"] == 0.08
    assert manifest["f0"]["base_envelope"] == {"minutes": 70, "units": 0.12, "reserve_units": 0.024}
    assert manifest["f0"]["effective_envelope"] == {
        "minutes": 140,
        "units": 0.24,
        "reserve_units": 0.048,
    }
    assert manifest["f0"]["gpu"] == 0
    assert manifest["resume"] == {
        "unit_kind": "life",
        "ledger_path": "inputs/verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }
    assert manifest["f0"]["kill"] == {
        "digest_or_bare_audit_mismatch": "INVALID",
        "rescue_bearing_lives_per_mac_worker_hour_lt": 8,
    }


def test_counterfactual_rows_carry_mass_per_action() -> None:
    """The ordinary stream's growth signal is part of the T5 row contract."""
    from research.bl_labels.factory import _row, _TapeEntry

    root = _TapeEntry(
        decision=12,
        life=1,
        frame=12,
        length=9,
        action=0,
        q=(6.0, 5.0, 4.0, 3.0, 2.0, 1.0),
        legal=(0, 2),
        digest="a" * 64,
        observation={
            "local": np.zeros((2, 2), dtype=np.uint8),
            "global": np.zeros((2, 2), dtype=np.uint8),
            "scalars": np.zeros(2, dtype=np.float32),
        },
    )
    row = _row(
        root,
        {
            0: {"survives": False, "frames_executed": 3, "mass_integral": 11.5},
            2: {"survives": True, "frames_executed": 300, "mass_integral": 22.25},
        },
        death_decision=15,
        stream="watch6",
        platform_tag="test",
        forced_complete=True,
        audit_selected=False,
    )
    assert row["mass"] == [11.5, None, 22.25, None, None, None]


def test_doom_root_schedule_is_bounded_and_outcome_blind() -> None:
    from research.bl_labels.factory import _select_roots, _TapeEntry

    def entry(decision: int) -> _TapeEntry:
        return _TapeEntry(
            decision=decision,
            life=3,
            frame=decision,
            length=10,
            action=1,
            q=(0.0,) * 6,
            legal=(0, 1, 2),
            digest="a" * 64,
            observation={},
        )

    entries = [entry(decision) for decision in range(400)]
    death = entry(399)
    selected = _select_roots(entries, death, "f0-test")
    # e=399 is the fatal action and f=e+1=400 is the death endpoint.  Doom
    # roots use [f-300, f), so the fatal action itself is eligible.
    assert len(selected) == 84
    assert {root.decision for root in selected} >= set(range(370, 400))
    assert {root.decision for root in selected} >= set(range(100, 400, 5))
    doom = {root.decision for root in selected if root.decision >= 100}
    assert len(doom) == 84  # 60 stride roots union the last 30 contiguous roots.
    assert {root.decision for root in selected} == doom
    assert all(root.life == 3 and root.decision < 400 for root in selected)


def test_dead_branch_mass_is_fixed_h300_with_zero_tail(monkeypatch: pytest.MonkeyPatch) -> None:
    from research.bl_labels import factory

    class FakeDriver:
        def step(self, _forced=None):
            return {"alive": False, "hero_length": 12, "decision": 0}

    def continuation(snapshot, action, depth, *, restore, pulse):
        driver = restore(snapshot)
        driver.step(action)
        return {
            "H300": {"survives": False, "frames_executed": 1},
            "death": {"decision": 0, "hero_length": 12},
            "continuation": None,
            "native_v8": {},
        }

    monkeypatch.setattr("research.bl_served_search.b2_engine.continue_branch", continuation)
    result = factory._branch(b"root", 2, factory.HORIZON, restore=lambda _payload: FakeDriver())
    assert result["final_length"] == 12
    assert result["mass_integral"] == pytest.approx(12 / factory.HORIZON)


def test_restore_root_replays_bare_tape_and_fails_closed_on_digest() -> None:
    from research.bl_labels import factory
    from research.bl_labels.factory import _TapeEntry

    class FakeDriver:
        def __init__(self) -> None:
            self.decision = 0
            self.actions: list[int] = []
            self.prepares = 0

        def step(self, action: int) -> None:
            self.actions.append(action)
            self.decision += 1

        def prepare(self) -> None:
            self.prepares += 1

    root = _TapeEntry(3, 3, 3, 10, 2, (0.0,) * 6, (0, 1, 2), "ok", {})
    tape = {
        index: _TapeEntry(index, 3, index, 10, index, (0.0,) * 6, (0, 1, 2), "x", {})
        for index in range(3)
    }
    fake = FakeDriver()
    monkeypatch = pytest.MonkeyPatch()
    try:
        monkeypatch.setattr(factory.WatchDriver, "restore", lambda _payload: fake)
        monkeypatch.setattr(factory, "_digest_driver", lambda *_args: "ok")
        restored = factory._restore_root(
            {0: b"anchor"}, tape, root, "b" * 64, restore=lambda _payload: fake
        )
        assert restored is fake
        assert fake.actions == [0, 1, 2]
        # Restore must enter the prepared selector phase at the root, including
        # after replaying the final preceding action.
        assert fake.prepares == 3

        monkeypatch.setattr(factory, "_digest_driver", lambda *_args: "wrong")
        with pytest.raises(factory.FactoryInvalid, match="digest mismatch"):
            factory._restore_root(
                {0: b"anchor"}, tape, root, "b" * 64, restore=lambda _payload: fake
            )
    finally:
        monkeypatch.undo()


def test_collect_unit_selected_bare_mass_integral_mismatch_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A selected natural-tape reroll mismatch must invalidate the whole unit."""
    from types import SimpleNamespace

    from research.bl_labels import factory

    class FakeDriver:
        def __init__(self, *, terminal: bool = False) -> None:
            self.decision = 1 if terminal else 0
            self.life = 0
            self.awaiting_respawn = False
            self.last_root_q = np.zeros(6, dtype=np.float32)
            self.sim = SimpleNamespace(
                alive=np.array([[not terminal]], dtype=np.bool_),
                frame=np.array([0], dtype=np.int64),
                length=np.array([[9]], dtype=np.int64),
            )

        @classmethod
        def restore(cls, _payload: bytes) -> "FakeDriver":
            return cls(terminal=True)

        def prepare(self) -> None:
            self.sim.alive[0, 0] = True

        def legal_and_bare(self) -> tuple[tuple[int, ...], int]:
            return (0, 1), 0

        def prepared_decision(self):
            from research.bl_served_search.b2_engine import PreparedDecision

            return PreparedDecision(
                self.sim,
                self.decision,
                0,
                {},
                self.last_root_q.tobytes(),
                (0, 1),
                0,
            )

        def snapshot(self) -> bytes:
            return b"fake-snapshot"

        def step(self, _forced: int | None = None, packet=None) -> dict[str, object]:
            del packet
            self.sim.alive[0, 0] = False
            self.decision = 1
            return {
                "decision": 0,
                "actions": [0],
                "hero_length": 9,
                "death_cause": "wall",
            }

    def branch(_snapshot: bytes, action: int, _depth: int, **_kwargs: object) -> dict[str, object]:
        if action == 0:
            return {
                "survives": False,
                "frames_executed": 1,
                "death_decision": 0,
                "final_length": 9,
                "mass_integral": 10 / factory.HORIZON,  # differs only in F4's mass field
                "extended": {"survives": False},
            }
        return {
            "survives": True,
            "frames_executed": factory.HORIZON,
            "death_decision": None,
            "final_length": 9,
            "mass_integral": 9.0,
            "extended": {"survives": False},
        }

    monkeypatch.setattr(factory, "make_driver", lambda *_args: FakeDriver())
    monkeypatch.setattr(
        factory,
        "digest",
        lambda path: factory.ceiling.B8_SHA256 if "b8" in str(path) else factory.FRP3,
    )
    monkeypatch.setattr(factory, "_raw_observation", lambda _driver: {})
    monkeypatch.setattr(factory, "_env_digest", lambda _sim: "env")
    monkeypatch.setattr(factory, "_digest_driver", lambda *_args: "root-digest")
    monkeypatch.setattr(factory, "driver_stats", lambda _driver: {})
    monkeypatch.setattr(factory, "attest", lambda _stats: None)
    monkeypatch.setattr(factory, "_restore_root", lambda *_args, **_kwargs: FakeDriver())
    monkeypatch.setattr(factory, "_branch", branch)
    monkeypatch.setattr(factory, "_selection_hash", lambda *_parts: "0" * 64)

    published: list[dict[str, object]] = []
    allocation = {
        "allocation_id": "watch6-000",
        "stream": "watch6",
        "seed": 0,
        "horizon": 1,
        "publish_life": published.append,
    }
    with pytest.raises(factory.FactoryInvalid, match="bare audit mismatch"):
        factory.collect_unit(allocation, Path("b8"), Path("frp3"), "f0-test")
    assert published == []
