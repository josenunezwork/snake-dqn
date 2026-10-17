"""Mutation coverage for the fail-closed T12B Check-4 evidence audit."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.t12b_check4_20261017 import audit


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


@pytest.fixture
def completed_run(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    source_names = (
        "web/backend/ego2s_policy.py",
        "web/backend/session.py",
        "web/backend/serving_speed.py",
        "configs/free_space_v2.yaml",
    )
    for name in source_names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("amendment", encoding="utf-8")
    checkpoint = tmp_path / "b8.pth"
    checkpoint.write_bytes(b"b8")
    checkpoint_sha = _sha(checkpoint)
    monkeypatch.setattr(audit, "REPO", repo)
    monkeypatch.setattr(audit, "PROTOCOL", protocol)
    monkeypatch.setattr(audit, "PROTOCOL_SHA256", _sha(protocol))
    monkeypatch.setattr(audit, "CHECKPOINT_SHA256", checkpoint_sha)

    prior_audit = tmp_path / "prior-audit.json"
    _write(
        prior_audit,
        {"criteria": {str(n): {"pass": n != 4} for n in range(1, 9)}},
    )
    prior_receipt = tmp_path / "prior-receipt.json"
    _write(prior_receipt, {"intent_sha256": "a" * 64, "files": {"latency.json": "b" * 64}})
    seeds = audit._registered_seeds()
    seed_audit = tmp_path / "seed-audit.json"
    banks = {"latency": {"count": 4, "seeds": seeds, "seed_digest": audit._seed_digest(seeds)}}
    _write(
        seed_audit,
        {
            "schema": "t12b-check4-seed-audit/v1",
            "pass": True,
            "namespace": audit.NAMESPACE,
            "allocation": {"banks": banks},
            "prior_audit": {"pass": True},
            "explicit_exclusions": {"collision_seeds": []},
            "unexcluded_t12b_artifact_scan": {"hits": []},
        },
    )
    validation = tmp_path / "validation-input.json"
    _write(validation, {"passed": True})
    root = tmp_path / "run"
    root.mkdir()
    intent = {
        "schema": audit.INTENT_SCHEMA,
        "protocol_sha256": audit.PROTOCOL_SHA256,
        "namespace": "t12b-check4-20261017/v1",
        "git": {"commit": "abc", "dirty": False},
        "sources": {name: _sha(repo / name) for name in source_names},
        "producer": {"run_sha256": "c" * 64},
        "checkpoint": {"name": "b8-s3", "sha256": checkpoint_sha, "path": str(checkpoint)},
        "released_default": "frp3-s12",
        "seed_audit": {"path": str(seed_audit), "sha256": _sha(seed_audit)},
        "banks": banks,
        "validation_sha256": _sha(validation),
        "prior": {
            "audit_path": str(prior_audit),
            "audit_sha256": _sha(prior_audit),
            "receipt_path": str(prior_receipt),
            "receipt_sha256": _sha(prior_receipt),
        },
    }
    _write(root / "intent.json", intent)
    (root / "validation.json").write_bytes(validation.read_bytes())
    intent_sha = _sha(root / "intent.json")

    def paced(request: int, actual: int, ceiling: int) -> dict[str, object]:
        return {
            "set_speed_request_fps": request,
            "actual_clamped_fps": float(actual),
            "server_speed_ceiling_fps": float(ceiling),
            "ticks": 120,
            "raw_tick_ms": [1.0] * 120,
            "overruns": 0,
        }

    quiet = {
        "quiet": True,
        "power": {"ac_power": True, "lid_closed": False},
        "processes": {"returncode": 0, "stdout": "1 0 launchd\n"},
        "suspicious_processes": [],
        "tmux": {"returncode": 1, "stdout": ""},
        "active_workload_panes": [],
    }
    rows = []
    for (name, (mode, snakes)), seed in zip(audit.CASES.items(), seeds):
        cap = 24 if snakes <= 6 else 12
        rows.append(
            {
                "case": name,
                "mode": mode,
                "snakes": snakes,
                "seed": seed,
                "samples": {"update_ms": [1.0] * 1500, "tick_ms": [1.0] * 1500},
                "advancing_measured_frames": 1500,
                "served": {"registry": {"name": "b8-s3", "sha256": checkpoint_sha}},
                "roster": {
                    "all_snakes_b8": True,
                    "all_ai_share_served_policy": True,
                    "ai_count": snakes if mode == "watch" else snakes - 1,
                    "total_snakes": snakes,
                    "human_id": None if mode == "watch" else 0,
                    "ai_veto_wrappers": [],
                },
                "quiet_before": quiet,
                "quiet_after": quiet,
                "default_paced": paced(12, 12, cap),
                "ceiling_paced_attempts": [paced(120, cap, cap)],
                "clamped_ceiling_fps": cap,
            }
        )
    latency = {
        "schema": audit.RESULT_SCHEMA,
        "intent_sha256": intent_sha,
        "kind": "latency",
        "seed_or_seeds": seeds,
        "checkpoint_sha256": checkpoint_sha,
        "started_utc": "now",
        "status": "complete",
        "error": None,
        "wall_seconds": 1.0,
        "data": {
            "frames_per_case": 1500,
            "preconditions": quiet,
            "cases": rows,
        },
    }
    _write(root / "latency.json", latency)
    for index, row in enumerate(rows):
        _write(
            root / "cases" / f"latency-{index:03d}.json",
            {
                "schema": audit.RESULT_SCHEMA,
                "intent_sha256": intent_sha,
                "kind": "latency_case",
                "index": index,
                "seed_or_seeds": seeds[index],
                "checkpoint_sha256": checkpoint_sha,
                "status": "complete",
                "data": row,
            },
        )
    files = {str(path.relative_to(root)): _sha(path) for path in root.rglob("*") if path.is_file()}
    _write(root / "receipt.json", {"intent_sha256": intent_sha, "files": files})
    return root


def _refresh_receipt(root: Path) -> None:
    receipt = json.loads((root / "receipt.json").read_text())
    receipt["files"] = {
        str(path.relative_to(root)): _sha(path)
        for path in root.rglob("*")
        if path.is_file() and path.name != "receipt.json"
    }
    _write(root / "receipt.json", receipt)


def test_completed_evidence_passes(completed_run):
    assert audit.audit(completed_run)["verdict"] == "SERVING_PASS"


def test_default_12_can_be_accepted_below_a_higher_24_fps_server_cap(completed_run):
    document = json.loads((completed_run / "latency.json").read_text())
    default = document["data"]["cases"][0]["default_paced"]
    ceiling = document["data"]["cases"][0]["ceiling_paced_attempts"][0]
    assert (
        default["set_speed_request_fps"],
        default["actual_clamped_fps"],
        default["server_speed_ceiling_fps"],
    ) == (12, 12.0, 24.0)
    assert (ceiling["set_speed_request_fps"], ceiling["actual_clamped_fps"]) == (120, 24.0)
    assert audit.audit(completed_run)["verdict"] == "SERVING_PASS"


@pytest.mark.parametrize(
    "value", ([1.0] * 1499, [1.0] * 1499 + [float("nan")], [1.0] * 1499 + [float("inf")])
)
def test_missing_or_nonfinite_update_samples_are_completed_failures(completed_run, value):
    path = completed_run / "latency.json"
    document = json.loads(path.read_text())
    document["data"]["cases"][0]["samples"]["update_ms"] = value
    _write(path, document)
    case = completed_run / "cases/latency-000.json"
    item = json.loads(case.read_text())
    item["data"] = document["data"]["cases"][0]
    _write(case, item)
    _refresh_receipt(completed_run)
    assert audit.audit(completed_run)["verdict"] == "INVALID"


def test_forged_summary_cannot_hide_raw_p95_failure(completed_run):
    path = completed_run / "latency.json"
    document = json.loads(path.read_text())
    document["data"]["cases"][0]["samples"]["update_ms"] = [100.0] * 1500
    document["data"]["cases"][0]["update_ms"] = {"p95": 1.0}
    _write(path, document)
    case = completed_run / "cases/latency-000.json"
    item = json.loads(case.read_text())
    item["data"] = document["data"]["cases"][0]
    _write(case, item)
    _refresh_receipt(completed_run)
    assert audit.audit(completed_run)["verdict"] == "SERVING_FAIL"


def test_changed_source_or_environment_evidence_is_invalid(completed_run):
    (audit.REPO / "web/backend/session.py").write_text("changed", encoding="utf-8")
    assert audit.audit(completed_run)["verdict"] == "INVALID"


def test_missing_quiet_machine_evidence_is_a_completed_hold(completed_run):
    path = completed_run / "latency.json"
    document = json.loads(path.read_text())
    document["data"]["cases"][2]["quiet_after"] = {"quiet": False}
    _write(path, document)
    case = completed_run / "cases/latency-002.json"
    item = json.loads(case.read_text())
    item["data"] = document["data"]["cases"][2]
    _write(case, item)
    _refresh_receipt(completed_run)
    assert audit.audit(completed_run)["verdict"] == "INVALID"


def test_forged_pacing_summary_or_unenforced_cap_fails(completed_run):
    path = completed_run / "latency.json"
    document = json.loads(path.read_text())
    attempt = document["data"]["cases"][0]["ceiling_paced_attempts"][0]
    attempt["raw_tick_ms"] = [100.0] * 120
    attempt["overruns"] = 0
    attempt["tick_ms"] = {"max": 1.0}
    _write(path, document)
    case = completed_run / "cases/latency-000.json"
    item = json.loads(case.read_text())
    item["data"] = document["data"]["cases"][0]
    _write(case, item)
    _refresh_receipt(completed_run)
    assert audit.audit(completed_run)["verdict"] == "INVALID"
