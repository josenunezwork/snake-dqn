"""Focused unit coverage for the first served-tick diagnostic package."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.bl_served_search import served_tick_probe as probe


def _manifest() -> dict:
    return json.loads((Path(probe.__file__).with_name("served_tick_manifest.json")).read_text())


def test_preflight_validates_domain_separated_seed_panel() -> None:
    value = probe.preflight(_manifest())
    assert value["status"] == "PREFLIGHT_OK_NOT_TIMING"
    assert len(value["effective_numpy_seeds"]) == 4


def test_preflight_rejects_changed_exposure() -> None:
    value = _manifest()
    value["measured_ticks"] = 120
    with pytest.raises(ValueError, match="frozen"):
        probe.preflight(value)


def test_pair_requires_actions_state_and_detached_dto() -> None:
    left = {"actions": [[1]], "state_digest": ["a"], "dto_digest": ["b"]}
    assert probe._verify_pair(left, copy.deepcopy(left))["pass"]
    right = copy.deepcopy(left)
    right["dto_digest"] = ["changed"]
    result = probe._verify_pair(left, right)
    assert not result["pass"] and result["mismatches"] == ["dto_digest"]


def test_canonical_excludes_only_declared_runtime_duration_fields() -> None:
    baseline = {"timeout_seconds": 10, "apply_seconds_total": 1.0}
    assert probe._digest(baseline) == probe._digest(
        {"timeout_seconds": 10, "apply_seconds_total": 2.0}
    )
    assert probe._digest(baseline) != probe._digest(
        {"timeout_seconds": 11, "apply_seconds_total": 1.0}
    )


def test_reducer_reports_separate_engineering_bar() -> None:
    raw = {
        "manifest": {"measured_ticks": 2, "fps": 24},
        "cells": [
            {
                "cell": {"arm": "inline"},
                "result": {
                    "trace": [
                        {
                            "tick": 0,
                            "target_ns": 0,
                            "wake_ns": 0,
                            "start_ns": 0,
                            "end_ns": 1,
                            "spans": {"game_update": {"wall_ns": 1_000_000, "calls": 1}},
                        },
                        {
                            "tick": 1,
                            "target_ns": probe.FRAME_NS,
                            "wake_ns": probe.FRAME_NS,
                            "start_ns": probe.FRAME_NS,
                            "end_ns": probe.FRAME_NS * 2,
                            "spans": {"game_update": {"wall_ns": 12_000_000, "calls": 1}},
                        },
                    ]
                },
            }
        ],
    }
    report = probe.reduce_trace(raw)
    assert report["verdict"] == "INVALID"


def test_run_panel_uses_create_only_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    manifest["measured_ticks"] = 1
    manifest["order"] = [{"arm": "timer", "seed": manifest["seeds"][0]}]
    monkeypatch.setattr(
        probe,
        "run_inline_arm",
        lambda spec: {
            "trace": [{"target_ns": 0, "end_ns": 1, "spans": {}}],
            "actions": [],
            "state_digest": [],
            "dto_digest": [],
        },
    )
    raw = probe.run_panel(manifest, tmp_path)
    assert raw["cells"][0]["result"]["trace"]
    assert not (tmp_path / "raw.json").exists()
    assert (tmp_path / "cell-00.json").exists()
    with pytest.raises(FileExistsError):
        probe.run_panel(manifest, tmp_path)
