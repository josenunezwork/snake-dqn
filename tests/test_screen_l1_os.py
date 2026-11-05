"""Bank-v2 screen: allocation, resume/skip, atomic results, checkpoints, analysis rule."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.watch_replay import analyze_screen
from research.watch_replay import screen_l1_os as screen
from research.watch_replay.seed_audit import (
    BANK_V2_DEV_NAMESPACE,
    BANK_V2_SCREEN_NAMESPACE,
    seed_for,
)
from tools.ops.package_spec import (
    PackageRefused,
    empty_ledger,
    publish_json,
    unit_id,
    verify_immutable,
)

ARMS = ("exact-stop-l1", "exact-stop-l1-os")


def test_allocation_pairs_arms_on_identical_screen_seeds():
    contract = screen.allocation()
    assert contract["namespace"] == BANK_V2_SCREEN_NAMESPACE
    assert contract["count"] == 48 and contract["horizon"] == 6000
    units = contract["units"]
    assert len(units) == 96 and [u["index"] for u in units] == list(range(96))
    for world in range(48):
        pair = [u for u in units if u["world"] == world]
        assert [u["arm"] for u in pair] == list(ARMS)
        assert {u["seed"] for u in pair} == {seed_for(BANK_V2_SCREEN_NAMESPACE, world)}
    assert screen.identity(contract) == screen.identity(screen.allocation())
    with pytest.raises(ValueError):
        screen.allocation(count=0)


def _fake_result(contract: dict, unit: dict) -> dict:
    return {
        "schema": screen.SCHEMA,
        "unit_id": unit_id(contract["namespace"], screen.ALLOCATION, unit["index"]),
        "allocation": screen.identity(contract),
        "index": unit["index"],
        "world": unit["world"],
        "arm": unit["arm"],
        "seed": unit["seed"],
        "horizon": contract["horizon"],
        "hero_mass_integral": 100.0,
        "hero_deaths": 0,
    }


def test_run_skips_verified_units_and_retries_failed_ones(tmp_path, monkeypatch):
    contract = screen.allocation("t34-bank-v2-unit-test", 3, 10)
    calls: list[int] = []
    fail = {3}

    def fake_run_unit(contract, unit, out, check, **kwargs):
        calls.append(unit["index"])
        if unit["index"] in fail:
            raise RuntimeError("mechanical")
        result = _fake_result(contract, unit)
        publish_json(out / "units" / f"{unit['index']:06d}.json", result)
        return result

    monkeypatch.setattr(screen, "run_unit", fake_run_unit)
    out = tmp_path / "screen"
    first = screen.run(contract, out, 1)
    # Longest-first: l1-os units (odd indices) before l1 units.
    assert calls == [1, 3, 5, 0, 2, 4]
    assert first["completed_now"] == [0, 1, 2, 4, 5]
    assert [e["index"] for e in first["exclusions"]] == [3]
    calls.clear()
    fail.clear()
    second = screen.run(contract, out, 1)
    assert calls == [3] and second["skipped_verified"] == [0, 1, 2, 4, 5]
    calls.clear()
    third = screen.run(contract, out, 1)
    assert calls == [] and third["skipped_verified"] == list(range(6))
    for i in range(6):
        verify_immutable(out / "units" / f"{i:06d}.json")
    # Operational records are append-only (never overwritten).
    assert sorted(p.name for p in out.glob("operational-*.json")) == [
        "operational-000.json",
        "operational-001.json",
        "operational-002.json",
    ]


def test_run_refuses_foreign_contract_dir_and_tampered_contract(tmp_path):
    contract = screen.allocation("t34-bank-v2-unit-test", 2, 10)
    out = tmp_path / "screen"
    out.mkdir()
    publish_json(out / "contract.json", screen.allocation("t34-bank-v2-other", 2, 10))
    with pytest.raises(PackageRefused, match="different contract"):
        screen.run(contract, out, 1)
    tampered = json.loads(json.dumps(contract))
    tampered["units"][0]["seed"] += 1
    with pytest.raises(PackageRefused, match="frozen allocation"):
        screen.run(tampered, tmp_path / "other", 1)


def test_ledger_skip_and_foreign_ledger_refusal(tmp_path):
    contract = screen.allocation("t34-bank-v2-unit-test", 2, 10)
    ledger = empty_ledger(contract["namespace"], screen.ALLOCATION, screen.identity(contract))
    ledger["units"] = [
        {
            "unit_id": unit_id(contract["namespace"], screen.ALLOCATION, 2),
            "index": 2,
            "sha256": "0" * 64,
            "size": 10,
        }
    ]
    publish_json(tmp_path / "ledger.json", ledger)
    pending, skipped = screen.pending_units(contract, tmp_path / "out", tmp_path / "ledger.json")
    assert skipped == [2] and [u["index"] for u in pending] == [1, 3, 0]
    foreign = empty_ledger("t34-bank-v2-other", screen.ALLOCATION, screen.identity(contract))
    publish_json(tmp_path / "foreign.json", foreign)
    with pytest.raises(PackageRefused):
        screen.pending_units(contract, tmp_path / "out", tmp_path / "foreign.json")


def test_partial_publication_is_not_complete_and_is_set_aside(tmp_path):
    contract = screen.allocation("t34-bank-v2-unit-test", 1, 10)
    path = tmp_path / "units" / "000000.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")  # payload without .sha256/.complete (crash mid-publication)
    pending, skipped = screen.pending_units(contract, tmp_path, None)
    assert skipped == [] and 0 in [u["index"] for u in pending]
    screen._set_aside(path)
    assert not path.exists() and Path(f"{path}.stale-0").exists()
    publish_json(path, _fake_result(contract, contract["units"][0]))
    _, skipped = screen.pending_units(contract, tmp_path, None)
    assert skipped == [0]


def test_checkpoint_archive_roundtrip_and_unsafe_member_refusal(tmp_path):
    import io
    import tarfile

    contract = screen.allocation("t34-bank-v2-unit-test", 1, 10)
    src = tmp_path / "out" / "checkpoints"
    for frame in (5, 9):
        publish_json(
            src / "000001" / f"frame-{frame:06d}.json",
            {
                "schema": screen.CHECKPOINT_SCHEMA,
                "allocation": screen.identity(contract),
                "index": 1,
                "frame": frame,
            },
        )
    publish_json(src / "000000" / "frame-000003.json", {"index": 0})
    archive = tmp_path / "ck.tar.gz"
    assert screen.pack_checkpoint_resume([src], archive, exclude={0}) == 1
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    payload = archive.read_bytes()
    from tools.ops.package_spec import publish_immutable

    publish_immutable(inputs / "checkpoints-resume.tar.gz", payload)
    staged = screen.stage_checkpoint_resume(inputs)
    best = screen.latest_checkpoint([staged], 1, contract)
    assert best["frame"] == 9
    assert screen.latest_checkpoint([staged], 0, contract) is None
    # Foreign checkpoint (other allocation) is refused.
    publish_json(
        tmp_path / "f" / "000001" / "frame-000020.json",
        {"schema": screen.CHECKPOINT_SCHEMA, "allocation": {"x": "y"}, "index": 1, "frame": 20},
    )
    with pytest.raises(PackageRefused, match="foreign checkpoint"):
        screen.latest_checkpoint([tmp_path / "f"], 1, contract)
    # Path traversal in an uploaded archive is refused.
    bad = io.BytesIO()
    with tarfile.open(fileobj=bad, mode="w:gz") as tar:
        info = tarfile.TarInfo("../evil.json")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    other = tmp_path / "inputs2"
    other.mkdir()
    publish_immutable(other / "checkpoints-resume.tar.gz", bad.getvalue())
    with pytest.raises(PackageRefused, match="unsafe"):
        screen.stage_checkpoint_resume(other)


# --- real engine: resumed == uninterrupted ----------------------------------------------


def _saved_snakes():
    from research.watch6_alllen_screen.protocol import ROOT

    for candidate in (ROOT.parent / "snake-dqn/saved_snakes", ROOT / "saved_snakes"):
        if (candidate / "b8_s3_t7_20261016.pth").is_file():
            return candidate
    pytest.skip("product checkpoints unavailable in test environment")


VOLATILE = ("compute_seconds", "replay_seconds", "resumed_from_frame", "checkpoints_rejected")


def _stable(result: dict) -> dict:
    return {k: v for k, v in result.items() if k not in VOLATILE}


@pytest.fixture
def real_env(monkeypatch):
    from research.t1_commitment_20261015.replay import configure_cpu

    monkeypatch.setenv("SNAKE_CKPT_DIR", str(_saved_snakes()))
    monkeypatch.setenv("SNAKE_DQN_DEVICE", "cpu")
    configure_cpu()


@pytest.mark.parametrize("arm_index", [0, 1])
def test_resumed_unit_equals_uninterrupted(tmp_path, monkeypatch, real_env, arm_index):
    contract = screen.allocation(BANK_V2_DEV_NAMESPACE, 1, 24)
    unit = contract["units"][arm_index]
    whole = screen.run_unit(contract, unit, tmp_path / "a", lambda: None)

    frames = {"n": 0}
    original = screen._normalize

    def counting(snapshot):
        frames["n"] += 1
        return original(snapshot)

    def interrupting_check():
        if frames["n"] >= 15:
            raise InterruptedError("simulated preemption")

    monkeypatch.setattr(screen, "_normalize", counting)
    out = tmp_path / "b"
    with pytest.raises(InterruptedError):
        screen.run_unit(contract, unit, out, interrupting_check, checkpoint_frames=frozenset({8}))
    ckpts = sorted((out / "checkpoints" / f"{unit['index']:06d}").glob("frame-*.json"))
    assert [p.name for p in ckpts] == ["frame-000008.json", "frame-000015.json"]
    assert not (out / "units" / f"{unit['index']:06d}.json").exists()
    resumed = screen.run_unit(contract, unit, out, lambda: None)
    assert resumed["resumed_from_frame"] == 15 and resumed["checkpoints_rejected"] == 0
    assert _stable(resumed) == _stable(whole)
    verify_immutable(out / "units" / f"{unit['index']:06d}.json")


def test_mismatched_checkpoint_is_rejected_and_unit_restarts(tmp_path, real_env):
    contract = screen.allocation(BANK_V2_DEV_NAMESPACE, 1, 16)
    unit = contract["units"][1]
    whole = screen.run_unit(contract, unit, tmp_path / "a", lambda: None)
    out = tmp_path / "b"
    publish_json(
        out / "checkpoints" / "000001" / "frame-000010.json",
        {
            "schema": screen.CHECKPOINT_SCHEMA,
            "allocation": screen.identity(contract),
            "index": 1,
            "arm": unit["arm"],
            "seed": unit["seed"],
            "frame": 10,
            "frames_sha256": "f" * 64,
            "counters": {name: 0 for name in screen.COUNTERS},
            "overrides": {},
            "compute_seconds": 0.0,
        },
    )
    result = screen.run_unit(contract, unit, out, lambda: None)
    assert result["checkpoints_rejected"] == 1 and result["resumed_from_frame"] is None
    assert _stable(result) == _stable(whole)


# --- pre-registered analysis rule ---------------------------------------------------------


def _rows(deltas, deaths_l1, deaths_os, base=500.0):
    rows = {}
    for w, d in enumerate(deltas):
        rows[(w, ARMS[0])] = {
            "hero_mass_integral": base,
            "hero_deaths": deaths_l1[w],
            "open_stop_fires": 0,
            "scaled_horizon_roots": 0,
            "compute_seconds": 3600.0,
        }
        rows[(w, ARMS[1])] = {
            "hero_mass_integral": base + d,
            "hero_deaths": deaths_os[w],
            "open_stop_fires": 2,
            "scaled_horizon_roots": 5,
            "compute_seconds": 7200.0,
        }
    return rows


def test_analysis_adopts_only_when_both_bars_hold():
    n = 48
    zeros = [0] * n
    one_death = [1] + [0] * (n - 1)
    adopt = analyze_screen.analyze(_rows([20.0] * n, one_death, zeros), n, 6000)
    assert adopt["verdict"] == "ADOPT" and adopt["paired_mass_delta"] == pytest.approx(20.0)
    assert adopt["mass_bar_met"] and adopt["deaths_bar_met"]
    lo, hi = adopt["paired_mass_delta_ci95"]
    assert lo == pytest.approx(20.0) and hi == pytest.approx(20.0)
    # Equal deaths satisfy "<=".
    assert analyze_screen.analyze(_rows([15.0] * n, zeros, zeros), n, 6000)["verdict"] == "ADOPT"
    # Mass bar: +14.99 fails.
    low = analyze_screen.analyze(_rows([14.99] * n, zeros, zeros), n, 6000)
    assert low["verdict"] == "DO_NOT_ADOPT" and not low["mass_bar_met"]
    # Deaths bar: one extra l1-os death fails even with a large mass gain.
    worse = analyze_screen.analyze(_rows([100.0] * n, zeros, one_death), n, 6000)
    assert worse["verdict"] == "DO_NOT_ADOPT" and not worse["deaths_bar_met"]
    assert worse["deaths_per_1k"]["exact-stop-l1-os"] == pytest.approx(1000 / (48 * 6000))


def test_analysis_bootstrap_is_seeded_and_brackets_the_mean():
    n = 48
    deltas = [float((w * 37) % 41 - 5) for w in range(n)]
    a = analyze_screen.analyze(_rows(deltas, [0] * n, [0] * n), n, 6000)
    b = analyze_screen.analyze(_rows(deltas, [0] * n, [0] * n), n, 6000)
    assert a == b
    lo, hi = a["paired_mass_delta_ci95"]
    assert lo < a["paired_mass_delta"] < hi
    assert a["bootstrap"] == {"seed": 20261105, "draws": 10000, "unit": "world"}


def test_analysis_refuses_incomplete_pairs_and_foreign_units(tmp_path):
    n = 48
    rows = _rows([20.0] * n, [0] * n, [0] * n)
    del rows[(7, ARMS[1])]
    with pytest.raises(ValueError, match="incomplete pairs"):
        analyze_screen.analyze(rows, n, 6000)
    contract = screen.allocation("t34-bank-v2-unit-test", 1, 10)
    for unit in contract["units"]:
        bad = {**_fake_result(contract, unit), "seed": 1}
        publish_json(tmp_path / "units" / f"{unit['index']:06d}.json", bad)
    with pytest.raises(ValueError, match="foreign or mismatched"):
        analyze_screen.load_units(tmp_path, contract)


def test_resume_replays_logged_overrides_exactly(tmp_path, monkeypatch, real_env):
    """Force open-stop 'fires' so the action log carries real overrides across a resume."""
    import numpy as np

    from research.watch_replay import engine_l1_os

    def firing(world, decision, q, legal):
        order = np.argsort(-np.where(legal, q.astype(np.float64), -np.inf), kind="stable")
        bare = int(order[0])
        chosen = int(order[1]) if decision % 3 == 1 and legal.sum() > 1 else bare
        return {"chosen": chosen, "bare": bare, "branches_evaluated": 1}

    monkeypatch.setattr(engine_l1_os, "open_stop_batch", firing)
    contract = screen.allocation(BANK_V2_DEV_NAMESPACE, 1, 24)
    unit = contract["units"][1]
    whole = screen.run_unit(contract, unit, tmp_path / "a", lambda: None)
    assert whole["open_stop_fires"] >= 3

    frames = {"n": 0}
    original = screen._normalize

    def counting(snapshot):
        frames["n"] += 1
        return original(snapshot)

    def interrupting_check():
        if frames["n"] >= 13:
            raise InterruptedError("simulated preemption")

    monkeypatch.setattr(screen, "_normalize", counting)
    out = tmp_path / "b"
    with pytest.raises(InterruptedError):
        screen.run_unit(contract, unit, out, interrupting_check)
    ckpt = json.loads((out / "checkpoints" / "000001" / "frame-000013.json").read_text())
    assert ckpt["overrides"] and ckpt["counters"]["open_stop_fires"] >= 1
    resumed = screen.run_unit(contract, unit, out, lambda: None)
    assert resumed["resumed_from_frame"] == 13 and resumed["checkpoints_rejected"] == 0
    assert _stable(resumed) == _stable(whole)
