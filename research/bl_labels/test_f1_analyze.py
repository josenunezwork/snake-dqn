"""Registered placement boundaries and refusal of incomplete/foreign timing banks."""

from __future__ import annotations

import copy

import pytest

from research.bl_labels.f1_analyze import _is_refund_upload_carveout, decide, transport_faults

ROOT_IDS = [f"root-{i:03d}" for i in range(64)]


def attempt(platform: str, elapsed: float = 100.0) -> dict:
    nproc = 2 if platform == "cpu" else 12
    return {
        "platform": platform,
        "source_commit": "d" * 40,
        "complete": True,
        "transport_faults": [],
        "configs": [
            {
                "source_commit": "d" * 40,
                "schedule_label": label,
                "workers": workers,
                "cpu": {"nproc": nproc},
                "root_count": 64,
                "elapsed_seconds": elapsed,
                "roots_per_hour": 64 * 3600 / elapsed,
                "records": [{"root_id": key} for key in ROOT_IDS],
                "integrity": {"digest_checks": 64},
            }
            for label, workers in (("one", 1), ("half", max(1, nproc // 2)), ("all", nproc))
        ],
    }


@pytest.mark.parametrize(
    ("cpu_seconds", "verdict"),
    [(400.0, "CPU_ALL_THREE_SLOTS"), (400.0001, "L4_LABELS_CPU_OVERFLOW")],
)
def test_quarter_rate_boundary(cpu_seconds: float, verdict: str) -> None:
    result = decide([attempt("cpu", cpu_seconds), attempt("l4")], ROOT_IDS)
    assert result["verdict"] == verdict


def test_recovered_transport_fault_still_excludes_platform() -> None:
    cpu = attempt("cpu")
    cpu["transport_faults"] = [{"command": "exec", "status": "failed"}]
    result = decide([cpu, attempt("l4")], ROOT_IDS)
    assert result["verdict"] == "L4_ONLY_CPU_EXCLUDED"


def test_missing_schedule_cannot_decide_ratio() -> None:
    cpu = attempt("cpu")
    cpu["configs"].pop()
    assert decide([cpu, attempt("l4")], ROOT_IDS)["verdict"] == "INCOMPLETE"


@pytest.mark.parametrize("defect", ["duplicate", "foreign", "capacity", "rate", "digest"])
def test_refuse_unregistered_or_malformed_schedule(defect: str) -> None:
    cpu = attempt("cpu")
    if defect == "duplicate":
        cpu["configs"].append(copy.deepcopy(cpu["configs"][0]))
    elif defect == "foreign":
        cpu["configs"][0]["records"][0]["root_id"] = "foreign"
    elif defect == "capacity":
        cpu["configs"][0]["workers"] = 12
    elif defect == "rate":
        cpu["configs"][0]["roots_per_hour"] = 100000
    else:
        cpu["configs"][0]["integrity"]["digest_checks"] = 63
    with pytest.raises(ValueError):
        decide([cpu, attempt("l4")], ROOT_IDS)


def test_l4_not_run_requires_explicit_declaration() -> None:
    assert decide([attempt("cpu")], ROOT_IDS)["verdict"] == "INCOMPLETE"
    assert decide([attempt("cpu")], ROOT_IDS, l4_not_run=True)["verdict"] == "CPU_ONLY_L4_NOT_RUN"


def test_transport_census_requires_evidence_and_counts_budget_exhaustion() -> None:
    with pytest.raises(ValueError):
        transport_faults({"cli_calls": []})
    call = {"status": "completed", "elapsed_seconds": 120.0, "timeout_seconds": 120.0}
    assert transport_faults({"cli_calls": [call]}) == [call]
    call["elapsed_seconds"] = 1.0
    assert transport_faults({"cli_calls": [call]}) == []


def refund_heartbeat_receipt() -> dict:
    generation = "a" * 32
    return {
        "generation": generation,
        "contents_transfers": [
            {
                "direction": "upload",
                "status": "completed",
                "remote": f"/content/snake-cli-{generation}-release.json",
            },
            {
                "direction": "upload",
                "status": "failed",
                "remote": f"/content/snake-cli/{generation}/lease-heartbeat.json",
                "size_bytes": None,
                "elapsed_seconds": 0.4,
            },
        ],
        "cli_calls": [
            {
                "attempt": 1,
                "command": command,
                "status": "completed",
                "elapsed_seconds": 0.1,
                "timeout_seconds": 120.0,
            }
            for command in ("version", "sessions", "usage", "new", "exec", "upload")
        ]
        + [
            {
                "attempt": 1,
                "command": "upload",
                "status": "failed",
                "elapsed_seconds": 0.4,
                "timeout_seconds": 60.0,
            },
            {
                "attempt": 1,
                "command": "exec",
                "status": "completed",
                "elapsed_seconds": 0.1,
                "timeout_seconds": 120.0,
            },
        ],
    }


def test_refund_carveout_is_only_the_initial_lease_heartbeat_upload() -> None:
    receipt = refund_heartbeat_receipt()
    call = receipt["cli_calls"][6]
    assert _is_refund_upload_carveout(receipt, 6, call)
    assert transport_faults(receipt) == [call]
    assert transport_faults(receipt, allow_refund_upload_carveout=True) == []


@pytest.mark.parametrize(
    "change",
    [
        lambda receipt: receipt["contents_transfers"][1].update(
            {"remote": "/content/snake-cli/foreign/lease-heartbeat.json"}
        ),
        lambda receipt: receipt["contents_transfers"][1].update({"size_bytes": 1}),
        lambda receipt: receipt["cli_calls"][6].update({"timeout_seconds": 61.0}),
        lambda receipt: receipt["cli_calls"][6].update({"elapsed_seconds": 60.0}),
        lambda receipt: receipt["cli_calls"][4].update({"command": "upload"}),
    ],
)
def test_refund_carveout_refuses_any_nonidentical_call(change) -> None:
    receipt = refund_heartbeat_receipt()
    change(receipt)
    assert transport_faults(receipt, allow_refund_upload_carveout=True) == [receipt["cli_calls"][6]]


def test_refund_carveout_does_not_hide_a_later_or_different_fault() -> None:
    receipt = refund_heartbeat_receipt()
    receipt["cli_calls"].append(
        {
            "attempt": 1,
            "command": "exec",
            "status": "failed",
            "elapsed_seconds": 0.2,
            "timeout_seconds": 120.0,
        }
    )
    assert transport_faults(receipt, allow_refund_upload_carveout=True) == [
        receipt["cli_calls"][-1]
    ]


def test_probe_receipt_contributes_fault_census_without_scoring(tmp_path) -> None:
    import hashlib
    import json

    from research.bl_labels.exports import _verify, publish_json
    from research.bl_labels.f1_analyze import load_attempt

    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"env": {"F1_PLATFORM": "cpu"}, "commit": "d" * 40}))
    runtime = tmp_path / "exports/runtime-result.json"
    publish_json(
        runtime,
        {
            "platform": "cpu",
            "bank_sha256": "a" * 64,
            "source_commit": "d" * 40,
            "status": "COMPLETE",
            "transport_probe": True,
            "configs": [{"root_count": 1}],
        },
    )
    receipt = {
        "status": "PASS",
        "stop_verified": True,
        "cleanup_error": None,
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "cli_calls": [{"status": "completed", "elapsed_seconds": 1.0, "timeout_seconds": 120.0}],
        "exports": [
            {
                "path": "runtime-result.json",
                "sha256": _verify(runtime),
                "size": runtime.stat().st_size,
            }
        ],
    }
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    probe = load_attempt(tmp_path, "a" * 64)
    assert probe["complete"] is False
    assert probe["configs"] == []
    assert (
        decide([probe, attempt("cpu"), attempt("l4")], ROOT_IDS)["verdict"] == "CPU_ALL_THREE_SLOTS"
    )
    receipt["cli_calls"][0]["status"] = "failed"
    (tmp_path / "receipt.json").write_text(json.dumps(receipt))
    probe = load_attempt(tmp_path, "a" * 64)
    assert (
        decide([probe, attempt("cpu"), attempt("l4")], ROOT_IDS)["verdict"]
        == "L4_ONLY_CPU_EXCLUDED"
    )
