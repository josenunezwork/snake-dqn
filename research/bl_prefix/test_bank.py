"""Focused bank-builder tests; real simulator preflight is exercised by the operator entrypoint."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from research.bl_ppo.exports import verify_export
from research.bl_prefix import bank


def _source(
    seed: int, mix: str, *, death: int | None, lengths: dict[int, int], survived: bool
) -> bank.Source:
    trace = np.zeros(bank.HORIZON + 1, dtype=np.int16)
    for frame, length in lengths.items():
        trace[frame] = length
    return bank.Source(
        seed,
        mix,
        {"seed": seed, "mix": mix, "specs": []},
        death,
        700 if death else None,
        survived,
        trace,
    )


def test_matching_uses_death_root_length_and_allocation_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(bank, "LATE_QUOTAS", {"scripted": 1, "frozen": 1, "mixed": 1})
    deaths = [
        _source(91, "scripted", death=2100, lengths={2000: 900}, survived=False),
        _source(92, "frozen", death=2100, lengths={2000: 900}, survived=False),
        _source(93, "mixed", death=2100, lengths={2000: 900}, survived=False),
    ]
    # Seed 3 comes first in the allocation even though its numeric value is larger.
    ordinary = [
        _source(30, "scripted", death=None, lengths={2000: 850}, survived=True),
        _source(3, "scripted", death=None, lengths={2000: 850}, survived=True),
        _source(31, "frozen", death=None, lengths={2000: 900}, survived=True),
        _source(32, "mixed", death=None, lengths={2000: 900}, survived=True),
    ]
    _late, pairs = bank._select([*deaths, ordinary[1], ordinary[0], *ordinary[2:]])
    assert pairs[0][1].seed == 3


def test_preflight_build_requires_audit_before_real_fixture_collection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    planned = False

    def no_audit(_plan: dict) -> dict:
        raise bank.BankIncomplete("audit missing")

    def mark_plans(_seeds: list[int]) -> list[dict]:
        nonlocal planned
        planned = True
        return []

    monkeypatch.setattr(bank, "_audit", no_audit)
    monkeypatch.setattr(bank, "_plans", mark_plans)
    result = bank.run(
        {"experiment": "bank", "preflight": True, "seed_audit": "missing"},
        tmp_path / "out",
        preflight=True,
    )
    assert result["status"] == "INCOMPLETE"
    assert planned is False


def test_preflight_bank_shape_and_supervisor_sidecars(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit_path = tmp_path / "audit.json"
    audit_path.write_text("{}")
    fixture_seeds = list(range(11, 11 + bank.PREFLIGHT_POOL))
    fake_audit = {"allocation": {"preflight": {"seeds": fixture_seeds}}}
    plans = [
        {"seed": seed, "mix": bank.MIXES[index % 3]} for index, seed in enumerate(fixture_seeds)
    ]

    monkeypatch.setattr(bank, "_audit", lambda _plan: fake_audit)
    monkeypatch.setattr(bank, "_b8_path", lambda _plan: audit_path)
    monkeypatch.setattr(bank, "_plans", lambda _seeds: plans)
    monkeypatch.setattr(
        bank,
        "_run_source",
        lambda plan, _path, **_kwargs: _source(
            plan["seed"], plan["mix"], death=None, lengths={}, survived=True
        ),
    )
    monkeypatch.setattr(
        bank,
        "_capture",
        lambda *_args, **_kwargs: {"200": object(), "150": object(), "100": object()},
    )
    monkeypatch.setattr(bank, "_write_root", lambda _root, path: path.name)
    monkeypatch.setattr(bank, "load_root", lambda _path: object())
    monkeypatch.setattr(bank, "verify_root_parity", lambda _root, frames: {"frames": frames})
    out = tmp_path / "fixture"
    value = bank.build(
        {"experiment": "bank", "preflight": True, "seed_audit": str(audit_path)}, out
    )
    assert len(value["roots"]) == len(value["early_roots"]) == 3
    assert all(set(row["ancestors"]) == {"100", "150", "200"} for row in value["early_roots"])
    assert verify_export(out / "bank.json")
    assert json.loads((out / "bank.json").read_text())["scientific"] is False


def test_preflight_selection_uses_only_first_48_and_records_early_deaths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    plans = [
        {"seed": seed, "mix": bank.MIXES[index % 3]}
        for index, seed in enumerate(range(100, 100 + bank.PREFLIGHT_POOL))
    ]
    called: list[int] = []

    def fake_run(plan: dict, _path: Path, *, stop_frame: int) -> bank.Source:
        assert stop_frame == bank.PREFLIGHT_SURVIVAL_FRAME
        called.append(plan["seed"])
        rejected = plan["seed"] in {100, 101, 102}
        return _source(
            plan["seed"],
            plan["mix"],
            death=131 if rejected else None,
            lengths={},
            survived=not rejected,
        )

    monkeypatch.setattr(bank, "_run_source", fake_run)
    selected, rejected = bank._select_preflight_sources(plans, tmp_path / "b8.pth")
    assert [item.mix for item in selected] == list(bank.MIXES)
    assert [item.seed for item in selected] == [103, 104, 105]
    assert called == [100, 101, 102, 103, 104, 105]
    assert rejected == [
        {"seed": 100, "mix": "scripted", "early_death_frame": 131},
        {"seed": 101, "mix": "frozen", "early_death_frame": 131},
        {"seed": 102, "mix": "mixed", "early_death_frame": 131},
    ]


def test_preflight_selection_rejects_wrong_fixture_pool_size(tmp_path: Path) -> None:
    with pytest.raises(bank.BankIncomplete, match="exactly"):
        bank._select_preflight_sources([], tmp_path / "b8.pth")


def test_preflight_accepts_full_audit_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    audit = tmp_path / "audit.json"
    from research.bl_ppo.exports import export_json

    allocation = {
        name: {"seeds": [index + 1]} for index, name in enumerate(bank.seeds.FROZEN_COUNTS)
    }
    export_json(audit, {"status": "PASS", "domain": bank.seeds.DOMAIN, "allocation": allocation})
    monkeypatch.setattr(bank.seeds, "verify_receipt", lambda *_args, **_kwargs: None)
    assert bank._audit({"preflight": True, "seed_audit": str(audit)})["status"] == "PASS"


def test_scientific_worker_bounds_and_fixed_task_order(tmp_path: Path) -> None:
    assert bank._workers({}, preflight=False) == 12
    assert bank._workers({"workers": 2}, preflight=True) == 2
    with pytest.raises(bank.BankIncomplete, match="exactly 12"):
        bank._workers({"workers": 11}, preflight=False)
    with pytest.raises(bank.BankIncomplete, match="1..2"):
        bank._workers({"workers": 3}, preflight=True)
    plans = [{"seed": 9, "mix": "mixed"}, {"seed": 2, "mix": "scripted"}]
    tasks = bank._source_tasks(plans, tmp_path / "b8.pth")
    assert [item[0]["seed"] for item in tasks] == [9, 2]
    assert [item[0]["seed"] for item in tasks] == [
        item[0]["seed"] for item in bank._source_tasks(plans, tmp_path / "b8.pth")
    ]
    pairs = [
        (
            _source(19, "scripted", death=2100, lengths={2000: 900}, survived=False),
            _source(29, "scripted", death=None, lengths={2000: 900}, survived=True),
        ),
        (
            _source(11, "mixed", death=2200, lengths={2100: 900}, survived=False),
            _source(21, "mixed", death=None, lengths={2100: 900}, survived=True),
        ),
    ]
    capture_tasks = bank._capture_tasks(pairs, tmp_path / "b8.pth", tmp_path / "out")
    assert [(item[0].seed, item[1].seed) for item in capture_tasks] == [(19, 29), (11, 21)]


def test_high_range_roster_adapter_preserves_full_rotation(monkeypatch: pytest.MonkeyPatch) -> None:
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from research.t3_rollout_labels_20261015 import bank_engine

    context = {
        "ds": dev_screen,
        "lookup": dev_screen.agent_lookup(
            {sha: f"/unused/{name}" for name, sha in dev_screen.POOL}
        ),
    }
    monkeypatch.setattr(gid, "_context", lambda _threads: context)
    count = 3 * len(dev_screen.POOL) * 2
    ordinary = list(range(100, 100 + count))
    high = [(1 << 40) + value for value in ordinary]
    actual = bank._plans(high)
    expected = bank_engine.planned_rosters(
        context, ordinary, [bank.MIXES[i % 3] for i in range(count)]
    )
    assert [row["specs"] for row in actual] == [row["specs"] for row in expected]
    assert [row["seed"] for row in actual] == high


def test_capture_includes_last_requested_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np

    world = SimpleNamespace(sim=SimpleNamespace(frame=np.array([0]), alive=np.array([[True]])))
    monkeypatch.setattr(bank, "initial_world", lambda **_kwargs: world)
    monkeypatch.setattr(bank, "B8Controller", lambda _path: object())
    monkeypatch.setattr(bank, "capture_root", lambda value, history: int(value.sim.frame[0]))

    def advance(value, _controller):
        value.sim.frame[0] += 1
        return SimpleNamespace(
            observation={}, action=0, legal=np.ones(6, bool), advisory=np.ones(6, bool)
        )

    monkeypatch.setattr(bank, "step", advance)
    source = bank.Source(1, "scripted", {}, None, None, True, np.zeros(5001, np.int16))
    result = bank._capture(
        source, Path("unused"), {"200": 128, "150": 178, "100": 228}, stop_at_last=True
    )
    assert result == {"200": 128, "150": 178, "100": 228}
    assert int(world.sim.frame[0]) == 228
