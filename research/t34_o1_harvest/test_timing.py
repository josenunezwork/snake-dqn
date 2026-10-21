"""Excluded timing regression: early terminal death must not require late roots."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import pytest

from research.t34_o1_harvest import runner
from research.t34_o1_harvest.bank import materialize
from research.t34_o1_harvest.timing import TIMING_FIXTURE, validate_timing


def fixtures() -> list[dict[str, Any]]:
    return [
        {
            "row": row,
            "frames": 5000,
            "horizon": 500,
            "complete": True,
            "accounting": True,
            "cfg": {"mechanics_version": 2},
            "base_steps": steps,
            "base_survives": False,
            "events": [],
            "roots": [
                {"decision": d, "time_root": True, "event_root": False, "horizon": 500}
                for d in (10, 20)
            ],
        }
        for row, steps in zip(materialize()["excluded_timing_rows"], (4479, 2216))
    ]


def test_terminal_before_3000_without_events_still_supplies_fixed_roots():
    validate_timing(fixtures(), materialize()["excluded_timing_rows"])


@pytest.mark.parametrize("damage", ["dead_before_20", "late_roots", "event_root", "mac_rows"])
def test_exact_timing_validator_refuses_substitutions(damage):
    worlds = fixtures()
    if damage == "dead_before_20":
        worlds[1]["roots"].pop()
    elif damage == "late_roots":
        for root, decision in zip(worlds[1]["roots"], (1000, 3000)):
            root["decision"] = decision
    elif damage == "event_root":
        worlds[1]["roots"][1]["event_root"] = True
    else:
        worlds[1]["row"] = materialize()["excluded_preflight_rows"][1]
    with pytest.raises(ValueError, match="timing fixture"):
        validate_timing(worlds, materialize()["excluded_timing_rows"])


@pytest.mark.parametrize("workers", [1, 11])
def test_shared_collector_uses_exact_l4_worlds_and_full_horizons(tmp_path, monkeypatch, workers):
    calls = []

    def collect(rows, checkpoint, count, **kwargs):
        calls.append((rows, checkpoint, count, kwargs))
        return fixtures()

    monkeypatch.setattr(runner, "collect_many", collect)
    bank = materialize()
    runner.collect_timing_fixture(bank, "fixture-only", workers, tmp_path, 9999.0)
    assert calls == [
        (
            bank["excluded_timing_rows"],
            "fixture-only",
            workers,
            {
                "payload_dir": tmp_path,
                "prefix": "timing-world",
                "frames": 5000,
                "horizon": 500,
                "time_roots": (10, 20),
                "event_limit": 0,
                "deadline": 9999.0,
            },
        )
    ]


def test_failed_root_fixture_stops_before_study_worlds(tmp_path: Path, monkeypatch: Any):
    import argparse

    bank = materialize()
    monkeypatch.setattr(
        runner, "require_registration", lambda _: {"bank": bank, "source_commit": "x"}
    )
    monkeypatch.setattr(runner, "sha256", lambda _: runner.B8_SHA)
    monkeypatch.setattr(runner, "ordinary_remaining", lambda _: 11000.0)
    monkeypatch.setenv("COLAB_CLI_ROOT", str(tmp_path))
    calls = []

    def collect(rows, *args, **kwargs):
        calls.append(rows)
        worlds = copy.deepcopy(fixtures())
        worlds[1]["roots"].pop()
        return worlds

    monkeypatch.setattr(runner, "collect_many", collect)
    args = argparse.Namespace(
        registration=tmp_path / "registration.json",
        checkpoint=tmp_path / "b8.pth",
        out=tmp_path / "run",
        mode="campaign",
        workers=11,
        catalog=tmp_path / "public.json",
        input_root=tmp_path / "inputs",
    )
    with pytest.raises(runner.CapExceeded, match="roots 10/20"):
        runner.run(args)
    assert calls == [bank[TIMING_FIXTURE["rows"]]]
    assert not list(args.out.glob("world-*.json"))
    assert (args.out / "failure.json").is_file()
