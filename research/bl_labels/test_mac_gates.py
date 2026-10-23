"""Exercise one-look and receipt-bound Mac admission on synthetic immutable units."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.bl_labels import mac_gates as gates
from research.bl_labels.exports import publish_json, publish_life, publish_session

PIN = "a" * 40


def fixture_pack(tmp_path: Path, monkeypatch, *, elapsed: float = 2700) -> tuple[Path, Path]:
    pack, attempt = tmp_path / "pack", tmp_path / "attempt"
    monkeypatch.setattr(gates, "GLOBAL_LOOK_ROOT", tmp_path / "global-look")
    allocation = [
        {"allocation_id": f"parent-{i:03}", "seed": i, "stream": "gate-scripted", "horizon": 3}
        for i in range(36)
    ]
    control = {
        "namespace": "synthetic-mac-gates",
        "allocation": allocation,
        "source_commit": PIN,
        "mac_gate": {"stage": "canary-b", "status": "HOLD", "package_root": str(pack)},
    }
    publish_json(pack / "control.json", control)
    d256 = copy.deepcopy(control)
    d256["mac_gate"]["stage"] = "d256"
    publish_json(pack / "d256-control.json", d256)
    for item in allocation:
        parent = item["allocation_id"]
        publish_life(
            attempt,
            control["namespace"],
            allocation,
            {
                "schema": "bl-labels-f0-life/v1",
                "namespace": control["namespace"],
                "status": "COMPLETE",
                "allocation_id": parent,
                "allocation": item,
                "life": 0,
                "rows": [{"decisive": True}],
                "counters": {"h1440": 0, "reversals": 0},
            },
        )
        publish_session(
            attempt,
            control["namespace"],
            allocation,
            {
                "schema": "bl-labels-f0-session/v1",
                "namespace": control["namespace"],
                "status": "COMPLETE",
                "allocation_id": parent,
                "allocation": item,
                "counters": {"deaths": 1},
                "provenance": {"elapsed_seconds": 0.1},
            },
        )
    state = tmp_path / "state"
    terminal_name = "attempt-fixture.terminal.json"
    receipt = {
        "namespace": control["namespace"],
        "workers": 6,
        "status": "COMPLETE",
        "elapsed_seconds": elapsed,
        "platform_attestation": {"code_sha": PIN},
        "state_dir": str(state),
        "attempts_ledger": [terminal_name],
    }
    publish_json(attempt / "runtime-result.json", receipt)
    publish_json(state / terminal_name, receipt)
    publish_json(
        state / "attempt-fixture.start.json",
        {
            "namespace": control["namespace"],
            "workers": 6,
            "deadline_seconds": 10800,
            "platform_attestation": {"code_sha": PIN},
            "output": str(attempt),
        },
    )
    monkeypatch.setattr(gates, "_source_pin", lambda _pack: PIN)
    return pack, attempt


def test_canonical_look_claim_prevents_second_read_at_different_output(
    tmp_path, monkeypatch
) -> None:
    pack, attempt = fixture_pack(tmp_path, monkeypatch)
    ledger = tmp_path / "ledger/verified-ledger.json"
    value = gates.publish_ledger(pack, [attempt], ledger)
    assert "mac_aggregate" not in value
    out = tmp_path / "look/decision.json"
    result = gates.look_canary(pack, ledger, [attempt], out)
    assert result["worker_hours"] == 4.5
    assert result["status"] == "PASS"  # equality at the frozen rate floor is admitted
    assert result["analysis"]["rescue_bearing_lives_per_mac_worker_hour"] == 8
    with pytest.raises(ValueError, match="already claimed"):
        gates.look_canary(pack, ledger, [attempt], tmp_path / "different/decision.json")


def test_below_eight_kills_without_new_look(tmp_path, monkeypatch) -> None:
    pack, attempt = fixture_pack(tmp_path, monkeypatch, elapsed=2701)
    ledger = tmp_path / "ledger/verified-ledger.json"
    gates.publish_ledger(pack, [attempt], ledger)
    result = gates.look_canary(pack, ledger, [attempt], tmp_path / "look/decision.json")
    assert result["status"] == "KILL"


def test_no_look_on_incomplete_population_or_foreign_ledger(tmp_path, monkeypatch) -> None:
    pack, attempt = fixture_pack(tmp_path, monkeypatch)
    path = attempt / "lives/parent-035--session.json"
    for suffix in ("", ".sha256", ".complete"):
        Path(str(path) + suffix).unlink()
    ledger = tmp_path / "ledger/verified-ledger.json"
    gates.publish_ledger(pack, [attempt], ledger)
    with pytest.raises(ValueError, match="36/36"):
        gates.look_canary(pack, ledger, [attempt], tmp_path / "look/decision.json")
    assert not (pack / "f0-look-claim.json").exists()


def test_duplicate_immutable_units_are_not_pooled(tmp_path, monkeypatch) -> None:
    pack, attempt = fixture_pack(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="duplicate"):
        gates.publish_ledger(pack, [attempt, attempt], tmp_path / "ledger/verified-ledger.json")


def test_hold_control_cannot_be_dispatched(tmp_path, monkeypatch) -> None:
    pack, _ = fixture_pack(tmp_path, monkeypatch)
    control = json.loads((pack / "d256-control.json").read_text())
    with pytest.raises(ValueError, match="HOLD"):
        gates.require_admission(control)


def test_admission_binds_f1_and_control_bytes(tmp_path, monkeypatch) -> None:
    pack, _ = fixture_pack(tmp_path, monkeypatch)
    output = tmp_path / "f1-output"
    publish_json(output / "runtime-result.json", {"synthetic_fixture": True})
    monkeypatch.setattr(gates, "_f1", lambda *_args: ({}, {"status": "PASS"}))
    control = gates.admit_canary(pack, output, tmp_path / "admitted.json")
    gates.require_admission(control)
    changed = copy.deepcopy(control)
    changed["allocation"][0]["seed"] += 1
    with pytest.raises(ValueError, match="frozen template"):
        gates.require_admission(changed)


def test_cumulative_wall_cap_and_attempt_maximum_are_not_reset(tmp_path, monkeypatch) -> None:
    pack, attempt = fixture_pack(tmp_path, monkeypatch, elapsed=10801)
    with pytest.raises(ValueError, match="three-hour"):
        gates._attempts(pack, [attempt], PIN)
    with pytest.raises(ValueError, match="one original"):
        gates._attempts(pack, [attempt, attempt, attempt], PIN)


def test_omitted_zero_leaf_attempt_cannot_inflate_the_rate(tmp_path, monkeypatch) -> None:
    pack, attempt = fixture_pack(tmp_path, monkeypatch)
    state = tmp_path / "state"
    publish_json(
        state / "attempt-earlier.start.json",
        {
            "namespace": "synthetic-mac-gates",
            "workers": 6,
            "deadline_seconds": 10800,
            "platform_attestation": {"code_sha": PIN},
            "output": str(tmp_path / "earlier"),
        },
    )
    publish_json(
        state / "attempt-earlier.terminal.json",
        {
            "namespace": "synthetic-mac-gates",
            "workers": 6,
            "elapsed_seconds": 60,
        },
    )
    with pytest.raises(ValueError, match="omitted or mismatched"):
        gates._attempts(pack, [attempt], PIN)


def test_unsealed_old_payload_stays_out_of_verified_result_set(tmp_path, monkeypatch) -> None:
    pack, attempt = fixture_pack(tmp_path, monkeypatch)
    # A process can stop between atomic payload and completion-marker publication.
    (attempt / "lives/parent-000--life-000099.json").write_text("partial unsealed bytes")
    ledger = tmp_path / "ledger/verified-ledger.json"
    gates.publish_ledger(pack, [attempt], ledger)
    result = gates.look_canary(pack, ledger, [attempt], tmp_path / "look/decision.json")
    assert result["status"] == "PASS"
    assert (attempt / "lives/parent-000--life-000099.json").read_text() == "partial unsealed bytes"


def test_d256_admission_requires_actual_f4_binding_and_mac_look(tmp_path, monkeypatch) -> None:
    """Use the real F4 approval checker; all outcome data here are synthetic."""
    from research.bl_labels.mac_package import (
        F4_JUDGMENT_SHA256,
        F4_RECIPE_SHA256,
        F4_REGISTRATION_SHA256,
    )

    pack, attempt = fixture_pack(tmp_path, monkeypatch)
    ledger = tmp_path / "ledger/verified-ledger.json"
    gates.publish_ledger(pack, [attempt], ledger)
    look = tmp_path / "look/decision.json"
    gates.look_canary(pack, ledger, [attempt], look)
    output = tmp_path / "f1-output"
    publish_json(output / "runtime-result.json", {"synthetic_fixture": True})
    monkeypatch.setattr(gates, "_f1", lambda *_args: ({}, {"status": "PASS"}))
    approval = tmp_path / "judgment.json"
    approval.write_text(
        json.dumps(
            {
                "schema": "bl-residual-f4-judgment/v1",
                "decision": "APPROVE",
                "judgment_sha256": F4_JUDGMENT_SHA256,
                "registration_sha256": F4_REGISTRATION_SHA256,
                "recipe_sha256": F4_RECIPE_SHA256,
            }
        )
    )
    with pytest.raises(ValueError, match="sidecars"):
        gates.admit_d256(pack, output, look, approval, tmp_path / "premature.json")
    allocation = json.loads((pack / "d256-control.json").read_text())["allocation"]
    publish_json(
        pack / "d256-seed-audit.json",
        {
            "schema": "bl-labels-f0-d256-seed-audit/v1",
            "seeds": [r["seed"] for r in allocation],
            "outcome_blind": True,
            "hits_by_root": {"synthetic-repo": [], "synthetic-artifacts": []},
        },
    )
    control = gates.admit_d256(pack, output, look, approval, tmp_path / "admitted-d256.json")
    assert control["operational_canary_rate"] == 48
    gates.require_admission(control)
    changed = json.loads(approval.read_text())
    changed["recipe_sha256"] = "b" * 64
    approval.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="approval changed"):
        gates.require_admission(control)
