"""Offline checks for the I0 waiver admission adapter; no provider or runtime calls."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tools import i0_small_bench as i0
from tools import small_bench_register as register
from tools import v8_small_bench as v8


def arms(slow: float = 0.3) -> dict:
    row = {
        "R": {"plain": [slow], "diagnostic": [slow + 0.02]},
        "I": {"plain": [0.2], "diagnostic": [0.19]},
    }
    return {
        "frozen": row,
        "mixed": json.loads(json.dumps(row)),
        "scripted": {
            "R": {"plain": [0.1], "diagnostic": [0.1]},
            "I": {"plain": [0.1], "diagnostic": [0.1]},
        },
    }


def revised(slow: float = 0.3, **kwargs) -> dict:
    values = dict(
        arms=arms(slow),
        receipt_elapsed=500.0,
        producer_elapsed=300.0,
        native_bound=200.0,
        download_bytes=1e6,
        download_seconds=1.0,
        max_export_bytes=64e6,
        warmups=3,
        repeats=3,
        continuous_ticks=32,
    )
    values.update(kwargs)
    return i0.revised_forecast(**values)


def test_old_forecast_reproduces_frozen_98_minutes() -> None:
    pair = 1.070093621000069
    units = [pair, 2.9788105499999 - pair]
    old = i0.old_forecast(pair, units, 537.1574484159937)
    assert math.isclose(old["forecast_seconds"], 5822.414110720054, rel_tol=1e-9)
    assert old["forecast_minutes"] == 98
    assert old["continuity_pair_equivalents"] == 4752


def test_revised_counts_actual_calls_and_slowest_roster() -> None:
    value = revised()
    assert value["calls_per_arm_per_ordinary_cell"] == 9
    assert value["ordinary_call_pair_equivalents"] == 3 * 9 * (16 + 4 + 1 + 1)
    assert value["continuity_calls_per_arm_per_roster"] == 1 + 3 + 3 * 33
    assert value["continuity_call_pair_equivalents"] == 3 * 103 * 16
    assert math.isclose(value["charged_call_pair_seconds"], 0.32 + 0.2)
    with_margin = value["components"]["continuity_call_seconds_with_margin"]
    assert math.isclose(with_margin, 0.52 * 4944 * i0.PROGRESSING_STATE_MARGIN)
    # E6 is never charged below one E64 call.
    assert value["ordinary_call_pair_equivalents"] > 3 * 9 * (16 + 4 + 1 + 6 / 64)


def test_forecast_excess_refuses_and_small_fits() -> None:
    reasons = i0.sizing_verdict(revised())
    assert any(r.startswith("HOLD_SIZING") for r in reasons)
    assert any(r.startswith("HOLD_SUPERVISOR") for r in reasons)
    tiny = revised(
        slow=0.001,
        receipt_elapsed=60.0,
        producer_elapsed=30.0,
        native_bound=29.0,
        arms={
            k: {
                "R": {"plain": [0.001], "diagnostic": [0.001]},
                "I": {"plain": [0.001], "diagnostic": [0.001]},
            }
            for k in ("frozen", "mixed", "scripted")
        },
    )
    assert i0.sizing_verdict(tiny) == []


def test_malformed_timing_refuses() -> None:
    bad = arms()
    bad["frozen"]["I"]["plain"] = [float("nan")]
    with pytest.raises(i0.Hold):
        revised(arms=bad)
    with pytest.raises(i0.Hold):
        revised(native_bound=400.0)
    with pytest.raises(i0.Hold):
        revised(arms={"frozen": arms()["frozen"]})


def test_cleanup_envelope_refuses_tail_and_accepts_fit() -> None:
    history = [
        {"id": "canary", "pair": {"units": 1.49, "wall_minutes": 8.95}},
        {"id": "compile", "pair": {"units": 0.69, "wall_minutes": 13.76}},
    ]
    held = i0.envelope(history, {"units": 5.0, "wall_minutes": 30.0})
    assert any(r.startswith("HOLD_CLEANUP") for r in held["reasons"])
    assert any(r.startswith("HOLD_SCOPE") for r in held["reasons"])
    fit = i0.envelope(history, {"units": 0.5, "wall_minutes": 1.0})
    assert fit["reasons"] == []


def test_native_lower_bound_uses_ordered_log() -> None:
    log = (
        "== 7 passed in 17.82s ==\n"
        '{"event": "i0_heartbeat", "elapsed_seconds": 110.0}\n'
        "=== test session starts ===\n== 7 passed, 7 deselected in 62.05s (0:01:02) ==\n"
    )
    assert math.isclose(i0.native_lower_bound([log])["seconds"], 172.05)
    with pytest.raises(i0.Hold):
        i0.native_lower_bound(["no heartbeat"])


def test_waiver_identity_refuses_changed_bytes(tmp_path: Path) -> None:
    ruling = tmp_path / "ruling.md"
    ruling.write_text("WAIVE")
    with pytest.raises(i0.Hold):
        i0.require_hash(ruling, i0.WAIVER_SHA, "waiver")
    assert i0.require_hash(ruling, i0.sha(ruling), "waiver")["sha256"] == i0.sha(ruling)


def template_pair(tmp_path: Path) -> tuple[dict, dict]:
    data = tmp_path / "metadata.tar"
    data.write_bytes(b"x")
    inputs = [{"path": str(data), "sha256": i0.sha(data), "dest": "metadata.tar"}]
    env = dict(i0.FLAGS, METADATA_ARCHIVE_SHA256=i0.METADATA_SHA)
    common = {
        "commit": i0.PIN,
        "inputs": inputs,
        "source_paths": ["a"],
        "transport_pin": "t",
        "mac_preflight": {"p": 1},
        "component_admission": {"c": 1},
        "script": "s",
    }
    template = dict(
        common,
        job_id=i0.JOB,
        scientific_admission=False,
        env=env,
        timeout_minutes=120,
        budget={"unit_cap": 20.0, "rate_units_per_hour": 10.0},
        requirements={"gpu": "A100", "allowed_gpus": ["A100"], "est_minutes": 50},
    )
    return template, json.loads(json.dumps(common))


def test_template_identity_and_flags(tmp_path: Path) -> None:
    template, canary = template_pair(tmp_path)
    i0.check_template(template, canary)
    for key, value in (
        ("GPU_SIM_V8_INDEXED", "1"),
        ("GPU_SIM_VECTOR_FLOOD", "sparse"),
        ("INTEGRATION_BENCH_SECONDS", "9000"),
    ):
        changed = json.loads(json.dumps(template))
        changed["env"][key] = value
        with pytest.raises(i0.Hold):
            i0.check_template(changed, canary)
    pinned = dict(template, commit="0" * 40)
    with pytest.raises(i0.Hold):
        i0.check_template(pinned, canary)
    extra = json.loads(json.dumps(template))
    extra["inputs"].append({"path": extra["inputs"][0]["path"], "sha256": "0" * 64, "dest": "x"})
    with pytest.raises(i0.Hold):
        i0.check_template(extra, canary)
    raised = dict(template, timeout_minutes=196)
    with pytest.raises(i0.Hold):
        i0.check_template(raised, canary)


def test_executed_main_identity_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(i0, "OPS", tmp_path / "ops")
    monkeypatch.setattr(i0, "PACKET", tmp_path / "packet")
    monkeypatch.setattr(i0, "ART", tmp_path / "art")
    i0.no_main_executed(str(tmp_path / "art/main-output"))
    (tmp_path / "ops/manifests").mkdir(parents=True)
    (tmp_path / "ops/manifests" / f"{i0.JOB}.json").write_text("{}")
    with pytest.raises(i0.Hold):
        i0.no_main_executed(str(tmp_path / "art/main-output"))


def initial(tmp_path: Path, charges: list[dict]) -> Path:
    path = tmp_path / "register.jsonl"
    source = [{"path": "/x", "sha256": "0" * 64}]
    register.append(
        path,
        {
            "type": "init",
            "id": "init",
            "scopes": {
                "global": {
                    "ceiling": {"units": 10, "wall_minutes": 10},
                    "spent": {"units": 0, "wall_minutes": 0},
                }
            },
            "holds": [],
            "completeness_attestation": "test",
            "sources": source,
            "historical_charges": charges,
        },
    )
    return path


def closure(ruling: str = i0.WAIVER_SHA) -> dict:
    return {
        "type": "delta",
        "id": i0.V8_CLOSURE_ID,
        "status": "CLOSED_UNNEEDED_FOR_I0",
        "ruling": {"path": "/r", "sha256": ruling},
        "closed_route": {"packet_id": "V8_SPLIT_INITIAL_MAIN_20261024"},
        "sources": [{"path": "/r", "sha256": ruling}],
    }


def test_v8_closure_required_and_bound(tmp_path: Path) -> None:
    path = initial(tmp_path, [])
    with pytest.raises(i0.Hold):
        i0.check_v8_closed(register.read(path))
    assert v8.closed(path) is None
    register.append(path, closure("1" * 64))
    with pytest.raises(i0.Hold):
        i0.check_v8_closed(register.read(path))
    with pytest.raises(ValueError):
        v8.closed(path)


def test_v8_adapter_refuses_closed_route(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = initial(tmp_path, [])
    register.append(path, closure())
    assert i0.check_v8_closed(register.read(path))["revision"] == 1
    monkeypatch.setattr(v8, "SPRINT", tmp_path)
    for operation in ("admit", "dispatch", "check", "prepare"):
        monkeypatch.setattr("sys.argv", ["v8_small_bench", operation])
        with pytest.raises(SystemExit, match="closed route"):
            v8.main()


def test_i0_history_refuses_missing_or_foreign_identity(tmp_path: Path) -> None:
    pair = {"units": 1.0, "wall_minutes": 6.0}
    good = [{"id": name, "pair": pair, "basis": "b"} for name in i0.I0_HISTORY_IDS]
    assert len(i0.i0_history(register.read(initial(tmp_path, good)))) == 2
    with pytest.raises(i0.Hold):
        i0.i0_history(register.read(initial(tmp_path / "a", good[:1])))
    foreign = good + [{"id": "gpu-integration-i0-canary-other", "pair": pair, "basis": "b"}]
    with pytest.raises(i0.Hold):
        i0.i0_history(register.read(initial(tmp_path / "b", foreign)))


def test_unrecorded_operational_identity(tmp_path: Path) -> None:
    (tmp_path / "gpu-integration-i0-canary-968f0276.json").write_text("{}")
    assert i0.unrecorded_i0_identities(tmp_path) == []
    (tmp_path / "gpu-integration-i0-main-968f0276.json").write_text("{}")
    assert i0.unrecorded_i0_identities(tmp_path) == ["gpu-integration-i0-main-968f0276.json"]


def test_admit_and_dispatch_refuse_without_admissible_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    monkeypatch.setattr(i0, "CONFIG", tmp_path / "absent.json")
    for operation in ("admit", "dispatch"):
        monkeypatch.setattr("sys.argv", ["i0_small_bench", operation])
        with pytest.raises(SystemExit) as raised:
            i0.main()
        assert raised.value.code == 2
        assert "REFUSED" in capsys.readouterr().out


def test_candidate_manifest_changes_only_admission_fields(tmp_path: Path) -> None:
    template, _ = template_pair(tmp_path)
    template["admission_reason"] = "TEMPLATE_ONLY"
    value = i0.candidate_manifest(template, {"path": "/r", "sha256": "0" * 64}, 55, tmp_path / "o")
    assert value["requirements"]["est_minutes"] == 55
    assert value["scientific_admission"] is True and "admission_reason" not in value
    assert value["inputs"] == template["inputs"] and value["env"] == template["env"]
    assert template["requirements"]["est_minutes"] == 50
