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


def desktop_raw():
    from research.bl_served_search import desktop_policy as policy
    from research.bl_served_search.desktop_apps import DEFAULT_REGISTRY, DesktopRegistry

    raw = fixture_raw()
    enrollment = DesktopRegistry().read_export()
    registry = enrollment["registry"]
    root = str(Path(probe.__file__).resolve().parents[2])
    registry_file_sha = __import__("hashlib").sha256(DEFAULT_REGISTRY.read_bytes()).hexdigest()
    hashes = raw["source_before"]["source_sha256"]
    hashes["research/bl_served_search/" + policy.RULING.name] = policy.RULING_SHA256
    hashes["research/bl_served_search/" + policy.AMENDMENT.name] = "c" * 64
    hashes["research/bl_served_search/" + DEFAULT_REGISTRY.name] = registry_file_sha
    for item in enrollment["registry_inputs"]:
        hashes["research/bl_served_search/" + item["path"]] = item["sha256"]
    raw["source_after"] = deepcopy(raw["source_before"])
    raw["source_before"]["worktree"] = root
    raw["source_after"]["worktree"] = root
    raw["load_policy"] = raw["host"]["load_policy"] = policy.POLICY
    raw["host"].update(
        schema="served-tick-host/v4",
        desktop_registry=registry,
        desktop_registry_sha256=policy.digest(registry),
        desktop_registry_evidence={"registry_inputs": enrollment["registry_inputs"]},
    )
    raw["desktop_intent"] = {
        "schema": "served-desktop-intent/v1",
        "policy": policy.POLICY,
        "registry": deepcopy(registry),
        "registry_sha256": policy.digest(registry),
        "registry_inputs": enrollment["registry_inputs"],
        "registry_input": {"path": str(DEFAULT_REGISTRY), "sha256": registry_file_sha},
        "ruling_sha256": policy.RULING_SHA256,
        "amendment_sha256": "c" * 64,
        "source": deepcopy(raw["source_before"]),
        "execution_worktree": root,
        "prior_attempts": {"07": "INVALID", "08": "INVALID"},
        "manifest_sha256": policy.digest(raw["manifest"]),
    }
    return raw


@pytest.mark.parametrize(
    "field",
    [
        "policy",
        "registry_sha256",
        "ruling_sha256",
        "amendment_sha256",
        "source",
        "manifest_sha256",
        "prior_attempts",
        "execution_worktree",
        "registry_input",
    ],
)
def test_desktop_reducer_refuses_intent_binding_mismatch(field):
    raw = desktop_raw()
    raw["desktop_intent"][field] = None
    assert probe.reduce_trace(raw)["verdict"] == "INVALID"
    assert "desktop evidence" in probe.reduce_trace(raw)["reason"]


def test_desktop_reducer_keeps_all_cells_and_fail(monkeypatch):
    from research.bl_served_search import background_report, desktop_policy, tick_trace

    monkeypatch.setattr(tick_trace, "reduce_trace", lambda *a, **kw: {"verdict": "FAIL"})
    monkeypatch.setattr(probe, "_production_summary", lambda *a: {"verdict": "FAIL"})
    calls = []

    def summarize(result, host, **kwargs):
        calls.append(result)
        return {
            "timing": {"overall": {"count": 2400}, "low_background": {"count": 0}},
            "counts": {"ticks": 2400},
            "applications": [{"entry": "outlook", "role": "mail", "affected_tick_count": 7}],
        }

    monkeypatch.setattr(background_report, "summarize_background", summarize)
    monkeypatch.setattr(background_report, "validate_desktop_host", lambda *a: None)
    monkeypatch.setattr(
        background_report,
        "desktop_overall",
        lambda *a: {
            "combined": 30,
            "applications": [{"entry": "outlook", "role": "mail", "observed_interval_count": 3}],
        },
    )
    raw = desktop_raw()
    raw["desktop_intent"]["host_observation"] = {"path": "/observation", "sha256": "d" * 64}
    monkeypatch.setattr(
        desktop_policy, "observation_binding", lambda *a: raw["desktop_intent"]["host_observation"]
    )
    report = probe.reduce_trace(raw)
    assert report["verdict"] == "FAIL"
    assert len(calls) == 12
    assert report["desktop_overall"]["combined"] == 30
    assert report["desktop_overall"]["counts"]["ticks"] == 28800
    assert report["desktop_overall"]["applications"][0]["affected_tick_count"] == 84
    assert report["desktop_overall"]["applications"][0]["observed_interval_count"] == 3
    assert all(
        arm["background_load"]["timing"]["overall"]["count"] == 2400 for arm in report["arms"]
    )


@pytest.mark.parametrize(
    "argv",
    [
        ["--desktop-app-registry", "registry", "--classified-host-load"],
        ["--desktop-app-registry", "registry", "--output", "panel"],
        [
            "--desktop-app-registry",
            "registry",
            "--desktop-observation-receipt",
            "receipt",
            "--host-only-seconds",
            "120",
        ],
        ["--desktop-app-registry", "registry", "--remote-supervision-registry", "remote"],
        ["--desktop-app-registry", "registry", "--host-only-seconds", "601"],
        ["--desktop-app-registry", "registry", "--host-only-seconds", "0"],
        ["--desktop-app-registry", "registry", "--host-only-seconds", "120", "--smoke"],
    ],
)
def test_desktop_cli_refuses_combined_policy_or_unbounded_observation(argv):
    with pytest.raises(SystemExit):
        probe.main(argv)


def test_host_only_observation_never_acquires_lease_or_runs_game(monkeypatch, tmp_path):
    from research.bl_paced_identity.record import export_json, read_export
    from research.bl_served_search import background_report, desktop_policy, probe_host

    binding = {"commit": "a" * 40}
    monkeypatch.setattr(probe, "preflight", lambda *a: {})
    monkeypatch.setattr(probe, "source_binding", lambda **kw: binding)
    monkeypatch.setattr(
        desktop_policy, "freeze_intent", lambda *a: {"policy": desktop_policy.POLICY}
    )
    monkeypatch.setattr(probe_host, "preflight", lambda **kw: {"admitted": True})
    monkeypatch.setattr(background_report, "desktop_overall", lambda *a: {"checked": True})

    def prohibited(*args, **kwargs):
        raise AssertionError("host-only observation must not acquire a lease or run a game")

    monkeypatch.setattr(probe_host, "guarded_run", prohibited)
    monkeypatch.setattr(probe, "run_panel", prohibited)
    monkeypatch.setattr(probe, "run_inline_arm", prohibited)

    class Monitor:
        def __init__(self, output, **kwargs):
            self.output = output

        def __enter__(self):
            return self

        def check(self):
            pass

        def __exit__(self, *args):
            export_json(self.output / "host.json", {"status": "COMPLETE"})

    monkeypatch.setattr(probe_host, "HostMonitor", Monitor)
    ticks = iter(range(10))
    monkeypatch.setattr(probe.time, "monotonic", lambda: next(ticks))
    monkeypatch.setattr(probe.time, "sleep", lambda _: None)
    output = tmp_path / "observation"
    assert (
        probe.main(
            [
                "--desktop-app-registry",
                str(tmp_path / "registry.json"),
                "--host-only-seconds",
                "1",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    observation = read_export(output / "observation.json")
    assert observation["status"] == "COMPLETE"
    assert observation["source_before"] == observation["source_after"] == binding
    assert observation["observation_seconds"] == 1


@pytest.mark.parametrize("change", [None, "source", "registry", "status", "no_app"])
def test_panel_requires_sealed_same_source_actual_app_observation(monkeypatch, tmp_path, change):
    from research.bl_paced_identity.record import export_json
    from research.bl_served_search import background_report, desktop_policy

    intent = desktop_raw()["desktop_intent"]
    host = {
        "status": "COMPLETE",
        "desktop_registry": intent["registry"],
        "desktop_registry_sha256": intent["registry_sha256"],
        "desktop_registry_evidence": {"registry_inputs": intent["registry_inputs"]},
        "cpu_samples": [{"processes": [{"kind": "desktop_app"}]}],
    }
    observation = {
        "schema": "served-desktop-host-observation/v1",
        "status": "COMPLETE",
        "load_policy": desktop_policy.POLICY,
        "desktop_intent": deepcopy(intent),
        "source_before": deepcopy(intent["source"]),
        "source_after": deepcopy(intent["source"]),
        "host": host,
    }
    if change == "source":
        observation["source_after"]["commit"] = "b" * 40
    elif change == "registry":
        observation["desktop_intent"]["registry_sha256"] = "a" * 64
    elif change == "status":
        observation["status"] = "INVALID"
    elif change == "no_app":
        host["cpu_samples"][0]["processes"] = [{"kind": "unknown"}]
    monkeypatch.setattr(background_report, "validate_desktop_host", lambda *a: None)
    export_json(tmp_path / "host.json", host)
    export_json(tmp_path / "intent.json", observation["desktop_intent"])
    path = tmp_path / "observation.json"
    export_json(path, observation)
    if change is not None:
        with pytest.raises(ValueError):
            desktop_policy.observation_binding(path, intent)
    else:
        assert desktop_policy.observation_binding(path, intent) == {
            "path": str(path),
            "sha256": desktop_policy.digest(observation),
        }
