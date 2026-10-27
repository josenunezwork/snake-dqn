"""Fail-closed report checks independent of any long Mac timing execution."""

import json
from copy import deepcopy
from pathlib import Path

import pytest

from research.bl_served_search import served_tick_probe as probe


def fixture_raw() -> dict:
    manifest = json.loads(Path(probe.__file__).with_name("served_tick_manifest.json").read_text())
    binding = {"commit": "a" * 40, "tracked_dirty": "", "source_sha256": {"probe": "b" * 64}}
    result = {
        "actions": [[1]],
        "state_digest": {"initial": "a", "final": "b"},
        "dto_digest": {"final": "c"},
        "processing_wall_ns": [10] * 120,
        "processing_thread_cpu_ns": [5] * 120,
    }
    return {
        "manifest": manifest,
        "source_before": binding,
        "source_after": deepcopy(binding),
        "host": {"status": "COMPLETE"},
        "cells": [
            {
                "cell": cell,
                "result": {
                    **deepcopy(result),
                    "trace": [],
                    "seed": manifest["seeds"][cell["seed_index"]],
                    "fps": 24,
                    "warmup_ticks": 200,
                    "measured_ticks": 2400,
                },
            }
            for cell in manifest["order"]
        ],
        "observer_control": {
            "ticks": 120,
            "arms": [
                {"name": name, "result": deepcopy(result)}
                for name in manifest["instrumented_observer_order"]
            ],
        },
    }


def test_report_refuses_missing_cells_before_any_pass_label() -> None:
    raw = fixture_raw()
    raw["cells"].pop()
    assert probe.reduce_trace(raw)["verdict"] == "INCOMPLETE"


def test_report_refuses_source_or_host_tamper() -> None:
    raw = fixture_raw()
    raw["source_after"]["commit"] = "d" * 40
    assert probe.reduce_trace(raw)["verdict"] == "INVALID"
    raw = fixture_raw()
    raw["host"]["status"] = "INVALID"
    assert probe.reduce_trace(raw)["verdict"] == "INVALID"


def test_report_recomputes_parity_and_requires_quantified_overhead(monkeypatch) -> None:
    from research.bl_served_search import tick_trace

    monkeypatch.setattr(tick_trace, "reduce_trace", lambda *a, **kw: {"verdict": "PASS"})
    monkeypatch.setattr(probe, "_production_summary", lambda *a: {"verdict": "PASS"})
    raw = fixture_raw()
    assert probe.reduce_trace(raw)["verdict"] == "PASS"
    raw["cells"][1]["result"]["state_digest"]["final"] = "changed"
    assert probe.reduce_trace(raw)["verdict"] == "INVALID"
    raw = fixture_raw()
    raw["observer_control"]["arms"][1]["result"]["processing_wall_ns"] = []
    assert probe.reduce_trace(raw)["verdict"] == "INVALID"


def test_empty_parity_fields_never_equal_valid_evidence() -> None:
    assert not probe._verify_pair({}, {})["pass"]


@pytest.mark.parametrize("fps", [24, 30])
def test_classified_report_attaches_all_cells_without_overriding_overall_fail(monkeypatch, fps):
    from research.bl_served_search import background_report, tick_trace

    monkeypatch.setattr(tick_trace, "reduce_trace", lambda *a, **kw: {"verdict": "FAIL"})
    monkeypatch.setattr(probe, "_production_summary", lambda *a: {"verdict": "FAIL"})
    if fps != 24:
        # Exercise forwarding for a hypothetical manifest without changing the frozen panel.
        monkeypatch.setattr(probe, "preflight", lambda manifest: {})
    calls = []

    def summarize(result, host, *, production=False, fps=24):
        calls.append((production, fps))
        return {"timing": {"low_background": {"count": 0}}}

    monkeypatch.setattr(background_report, "summarize_background", summarize)
    raw = fixture_raw()
    raw["load_policy"] = raw["host"]["load_policy"] = "classified-v1"
    raw["host"]["schema"] = "served-tick-host/v2"
    raw["manifest"]["fps"] = fps
    for cell in raw["cells"]:
        cell["result"]["fps"] = fps
    report = probe.reduce_trace(raw)
    assert report["verdict"] == "FAIL"
    assert len(calls) == len(raw["manifest"]["order"])
    assert sum(production for production, actual_fps in calls) == 3
    assert all(actual_fps == fps for production, actual_fps in calls)
    assert all(
        arm["background_load"]["timing"]["low_background"]["count"] == 0 for arm in report["arms"]
    )


def test_classified_report_refuses_missing_or_invalid_cpu_evidence():
    raw = fixture_raw()
    raw["load_policy"] = "classified-v1"
    assert probe.reduce_trace(raw)["reason"] == "host load policy binding differs"
    raw["host"]["load_policy"] = "classified-v1"
    assert probe.reduce_trace(raw)["reason"] == "classified host schema differs"
    raw["host"]["schema"] = "served-tick-host/v2"
    report = probe.reduce_trace(raw)
    assert report["verdict"] == "INVALID"
    assert "missing CPU samples" in report["reason"]


def test_report_refuses_matching_but_unknown_policy():
    raw = fixture_raw()
    raw["load_policy"] = raw["host"]["load_policy"] = "classified-v2"
    assert probe.reduce_trace(raw)["reason"] == "unsupported host load policy"


def test_prospective_reducer_retains_every_cell_and_authoritative_fail(monkeypatch):
    from research.bl_served_search import supervision_io, supervision_report, tick_trace

    monkeypatch.setattr(tick_trace, "reduce_trace", lambda *a, **kw: {"verdict": "FAIL"})
    monkeypatch.setattr(probe, "_production_summary", lambda *a: {"verdict": "FAIL"})
    monkeypatch.setattr(
        supervision_io,
        "replay_host",
        lambda *a: {"intervals": [{"start_ns": 0, "end_ns": 65_000_000_000}]},
    )
    calls = []

    def summarize(result, host, replay, *, production=False, fps=24):
        calls.append((production, fps))
        return {
            "counts": {"poll_overlap": 7},
            "timing": {"overall": {"count": 2400}, "low_background": {"count": 0}},
        }

    monkeypatch.setattr(supervision_report, "summarize_supervision", summarize)
    monkeypatch.setattr(supervision_report, "summarize_scope", lambda *a: {"complete": True})
    raw = fixture_raw()
    raw["load_policy"] = raw["host"]["load_policy"] = "classified-supervised-v2"
    raw["host"]["schema"] = "served-tick-host/v3"
    result = probe.reduce_trace(raw)
    assert result["verdict"] == "FAIL"
    assert len(calls) == len(raw["manifest"]["order"])
    assert sum(production for production, _ in calls) == 3
    assert result["supervision_overall"]["poll_overlap_ticks"] == 7 * len(calls)
    assert all(
        arm["background_load"]["timing"]["overall"]["count"] == 2400 for arm in result["arms"]
    )


def test_prospective_reducer_refuses_unavailable_live_observer(monkeypatch):
    from research.bl_served_search import supervision_io

    def refuse(*args):
        raise ValueError("observer not qualified")

    monkeypatch.setattr(supervision_io, "replay_host", refuse)
    raw = fixture_raw()
    raw["load_policy"] = raw["host"]["load_policy"] = "classified-supervised-v2"
    raw["host"]["schema"] = "served-tick-host/v3"
    assert probe.reduce_trace(raw) == {
        "verdict": "INVALID",
        "reason": "supervision evidence: observer not qualified",
    }
