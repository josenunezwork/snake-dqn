"""Parallel ordering, legacy reservoir replay, and fail-closed partial imports."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from research.q0q1_20261017 import audit_b8_init, audit_t23_mass, mass_execution
from research.q0q1_20261017.audit_common import canonical, sha256_file, write_json


def initialize_fixture() -> None:
    pass


def fixture_episode(task: dict) -> dict:
    # Completion order differs from plan order on two workers.
    time.sleep(task["delay"])
    number = task["number"]
    candidates = [
        {
            "mix": ("scripted", "frozen", "mixed")[number % 3],
            "action_a": 1,
            "action_b": 1,
            "short_lt700": index < 4100,
            "long_ge800": index >= 4100,
            "open": index >= 4100,
            "crowded": index < 4100,
            "boost_eligible": index % 2 == 0,
            "fixture_index": [number, index],
        }
        for index in range(4200)
    ]
    return {"row": {"episode": number}, "candidates": candidates, "excluded_trapped": number}


def run_fixture(tmp_path: Path, workers: int, **kwargs) -> tuple[list, Path]:
    tasks = [{"number": number, "delay": (3 - number) * 0.01} for number in range(3)]
    results = mass_execution.execute(
        tasks,
        worker=fixture_episode,
        initializer=initialize_fixture,
        initargs=(),
        binding={"fixture": "tiny-worlds/v1"},
        workers=workers,
        deadline=time.monotonic() + 45,
        partials_dir=tmp_path,
        **kwargs,
    )
    rows, reservoir = audit_t23_mass.merge_episodes(results)
    receipt = tmp_path.parent / f"{tmp_path.name}.json"
    write_json(
        receipt,
        {
            "rows": rows,
            "corpus": reservoir.rows,
            "seen": reservoir.seen,
            "excluded": reservoir.excluded_trapped,
        },
    )
    return results, receipt


def test_serial_vs_two_workers_receipt_is_byte_identical(tmp_path: Path) -> None:
    serial, serial_receipt = run_fixture(tmp_path / "serial", 1)
    _, parallel_receipt = run_fixture(tmp_path / "parallel", 2)
    assert serial_receipt.read_bytes() == parallel_receipt.read_bytes()
    # Also compare against the previous serial reservoir loop, including >cap,
    # protected late witnesses and RNG replacements across world boundaries.
    legacy = audit_b8_init.CoverageReservoir()
    for result in serial:
        legacy.excluded_trapped += result["excluded_trapped"]
        for row in result["candidates"]:
            legacy.add(row)
    merged = json.loads(serial_receipt.read_text())
    assert canonical(merged["corpus"]) == canonical(legacy.rows)
    assert merged["seen"] == legacy.seen == 12600
    assert merged["excluded"] == legacy.excluded_trapped == 3


def test_resume_into_fresh_destination_skips_verified_episodes(tmp_path: Path) -> None:
    _, original = run_fixture(tmp_path / "original", 1)
    manifest = tmp_path / "original/manifest.json"
    _, resumed = run_fixture(
        tmp_path / "resumed", 2, resume_partials=manifest, resume_sha256=sha256_file(manifest)
    )
    assert original.read_bytes() == resumed.read_bytes()
    # An all-resumed packet must never submit a world.
    tasks = [{"number": number, "delay": (3 - number) * 0.01} for number in range(3)]

    def unexpected(_: dict) -> dict:
        raise AssertionError("completed world executed twice")

    mass_execution.execute(
        tasks,
        worker=unexpected,
        initializer=initialize_fixture,
        initargs=(),
        binding={"fixture": "tiny-worlds/v1"},
        workers=1,
        deadline=time.monotonic() + 5,
        resume_partials=manifest,
        resume_sha256=sha256_file(manifest),
    )
    with pytest.raises(FileExistsError):
        run_fixture(tmp_path / "original", 1)


@pytest.mark.parametrize("corruption", ["manifest", "shard", "task", "binding", "plan"])
def test_resume_rejects_changed_bytes_and_identities(tmp_path: Path, corruption: str) -> None:
    run_fixture(tmp_path / "old", 1)
    manifest = tmp_path / "old/manifest.json"
    expected = sha256_file(manifest)
    tasks = [{"number": number, "delay": (3 - number) * 0.01} for number in range(3)]
    binding = {"fixture": "tiny-worlds/v1"}
    if corruption == "manifest":
        manifest.write_text(manifest.read_text() + " ")
    elif corruption == "shard":
        shard = manifest.parent / "episode-0000.json"
        shard.write_text(shard.read_text() + " ")
    elif corruption == "binding":
        binding["fixture"] = "foreign"
    elif corruption == "plan":
        tasks[0]["number"] = 5
    else:
        shard = manifest.parent / "episode-0000.json"
        packet = json.loads(shard.read_text())
        packet["task"]["number"] = 123
        shard.write_text(canonical(packet))
        index = json.loads(manifest.read_text())
        index["episodes"]["0"]["sha256"] = sha256_file(shard)
        manifest.write_text(canonical(index))
        expected = sha256_file(manifest)
    with pytest.raises(ValueError, match="mismatch"):
        mass_execution.load_partials(manifest, expected, binding, tasks)


def test_interrupted_run_keeps_finished_episode_for_retry(tmp_path: Path) -> None:
    tasks = [{"number": number, "delay": 0} for number in range(3)]

    def interrupted(task: dict) -> dict:
        if task["number"] == 1:
            raise RuntimeError("interrupted")
        return fixture_episode(task)

    with pytest.raises(RuntimeError, match="interrupted"):
        mass_execution.execute(
            tasks,
            worker=interrupted,
            initializer=initialize_fixture,
            initargs=(),
            binding={"fixture": 1},
            workers=1,
            deadline=time.monotonic() + 30,
            partials_dir=tmp_path / "attempt",
        )
    manifest = tmp_path / "attempt/manifest.json"
    saved = mass_execution.load_partials(manifest, sha256_file(manifest), {"fixture": 1}, tasks)
    assert list(saved) == [0]


def test_progress_while_serial_episode_runs(tmp_path: Path, capsys) -> None:
    mass_execution.execute(
        [{"number": 0, "delay": 0.08}],
        worker=fixture_episode,
        initializer=initialize_fixture,
        initargs=(),
        binding={},
        workers=1,
        deadline=time.monotonic() + 30,
        progress_seconds=0.02,
    )
    output = capsys.readouterr().out
    assert output.count("episodes 0/1 done") >= 2
    assert "episodes 1/1 done" in output and "elapsed=" in output and "ETA=" in output


def test_parent_checks_opponent_and_profile_pins_before_resumed_dispatch(tmp_path, monkeypatch):
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"fixture")
    monkeypatch.setattr(audit_t23_mass, "sha256_file", lambda _: audit_t23_mass.B8_SHA256)
    monkeypatch.setattr(audit_t23_mass.audit_q0_bank, "build_world_bank", lambda: {})
    monkeypatch.setattr(audit_t23_mass.audit_q0_bank, "audit_world_bank", lambda: {"pass": True})

    def reject_context(_: int) -> None:
        raise ValueError("opponent/config bytes changed")

    monkeypatch.setattr(gid, "_context", reject_context)
    with pytest.raises(ValueError, match="opponent/config bytes changed"):
        audit_t23_mass.run_development(
            tmp_path,
            b8,
            workers=2,
            resume_partials=tmp_path / "old/manifest.json",
            resume_partials_sha256="0" * 64,
        )
