"""Focused contract tests for the offline sandbox preflight."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.sandbox.manifest import ManifestError, load_manifest, validate_manifest
from tools.sandbox.preflight import (
    PreflightError,
    _runtime_updates,
    _validate_b_heads_runtime_recipe,
    verify_committed_manifest,
)


def _payload() -> dict:
    return {
        "schema": "standing-a100-development-sandbox/v1",
        "study": "sandbox-b-heads-v1",
        "inputs": [
            {
                "kind": "closed-f0",
                "dose": "D1024",
                "path": "research/sandbox_b_heads/inputs/admitted.npz",
                "sha256": "a" * 64,
            },
            {
                "kind": "frozen-checkpoint",
                "path": "saved_snakes/b8.pth",
                "sha256": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
            },
            {
                "kind": "worlds",
                "study": "sandbox-b-heads-v1",
                "indices": [0, 1],
                "seeds": [0, 18446744073709551615],
            },
        ],
        "salt": "a100-dev/sandbox-b/split-v1",
        "endpoint": {"kind": "updates", "value": 15000},
        "timeout": {"measured_estimate_seconds": 45.5, "seconds": 91.0},
        "excluded_studies": ["F5", "F4-fold-0", "R30", "strict", "probe", "serving"],
        "owner_rule": "docs/research/governance_a100_sandbox_20261026/ADOPTED.md#2",
        "outputs": {
            "label": "simulator-D",
            "paths": ["research/sandbox_b_heads/results"],
            "broker_manifest": {
                "path": "research/sandbox_b_heads/jobs/sandbox-b.json",
                "sha256": "b" * 64,
            },
        },
        "crossing": {"max_frozen_checkpoint_shas": 1},
    }


def _write_ten_lines(path: Path, payload: dict) -> None:
    keys = list(payload)
    assert len(keys) == 10
    lines = []
    for index, key in enumerate(keys):
        prefix = "{" if index == 0 else ""
        suffix = "}" if index == len(keys) - 1 else ","
        lines.append(
            prefix
            + json.dumps(key)
            + ":"
            + json.dumps(payload[key], separators=(",", ":"))
            + suffix
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_manifest_accepts_full_uint64_seed_and_exact_ten_lines(tmp_path: Path) -> None:
    path = tmp_path / "sandbox-manifest.json"
    _write_ten_lines(path, _payload())
    assert load_manifest(path)["inputs"][2]["seeds"][-1] == 2**64 - 1


@pytest.mark.parametrize(
    "change",
    [
        lambda payload: payload.update({"transport": "not-an-allowed-eleventh-key"}),
        lambda payload: payload["inputs"].append(["closed-f0", "array-not-record"]),
        lambda payload: payload["timeout"].update({"seconds": float("inf")}),
        lambda payload: payload["timeout"].update({"seconds": True}),
        lambda payload: payload["endpoint"].update({"value": False}),
        lambda payload: payload["inputs"][2].update({"seeds": [2**64, 1]}),
        lambda payload: payload["outputs"].update({"paths": ["big-lever/exp/f5-secret"]}),
    ],
)
def test_manifest_refuses_malformed_or_protected_values(change) -> None:
    payload = _payload()
    change(payload)
    with pytest.raises(ManifestError):
        validate_manifest(payload)


def test_loader_refuses_non_ten_physical_line_representation(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(_payload()), encoding="utf-8")
    with pytest.raises(ManifestError, match="exactly ten"):
        load_manifest(path)


def test_committed_check_has_test_bypass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import tools.sandbox.preflight as preflight_module

    monkeypatch.setattr(preflight_module, "REPO_ROOT", tmp_path)
    path = tmp_path / "untracked.json"
    path.write_text("{}", encoding="utf-8")
    verify_committed_manifest(path, require_committed=False)
    with pytest.raises(PreflightError):
        verify_committed_manifest(path)


def test_broker_digest_cannot_admit_foreign_upload(tmp_path):
    """Binding a hash also requires semantic equality with registry inputs."""
    import hashlib

    from tools.sandbox.preflight import _verify_broker_binding

    broker = tmp_path / "broker.json"
    broker.write_text(
        json.dumps(
            {
                "script": "research/sandbox_b_heads/run_job.sh",
                "inputs": [
                    {
                        "path": str(tmp_path / "foreign.npz"),
                        "dest": "foreign.npz",
                        "sha256": "a" * 64,
                    }
                ],
                "env": {},
                "source_paths": [],
                "output_dir": str(tmp_path / "output"),
            }
        )
    )
    payload = _payload()
    payload["outputs"]["broker_manifest"] = {
        "path": str(broker),
        "sha256": hashlib.sha256(broker.read_bytes()).hexdigest(),
    }
    with pytest.raises(PreflightError, match="absent from sandbox admission"):
        _verify_broker_binding(payload)


def test_digest_hex_is_not_mistaken_for_a_protected_path():
    payload = _payload()
    payload["inputs"][0]["sha256"] = "f5" * 32
    validate_manifest(payload)


def test_runaway_timeout_is_explicit_and_doubled():
    payload = _payload()
    payload["timeout"] = {"base_seconds": 3600, "seconds": 7200, "purpose": "runaway-guard"}
    validate_manifest(payload)
    payload["timeout"]["seconds"] = 3600
    with pytest.raises(ManifestError, match="2x"):
        validate_manifest(payload)


def test_path_guard_accepts_random_f5_substring_but_refuses_alias(tmp_path):
    from tools.sandbox.preflight import _assert_safe_path

    benign = tmp_path / "r_nk05n12vg4v5f_rxyf65v40000gn" / "output"
    assert _assert_safe_path(benign, "test") == benign
    alias = tmp_path / "alias"
    alias.symlink_to(tmp_path / "f5" / "output")
    with pytest.raises(PreflightError):
        _assert_safe_path(alias, "test")


def test_b_heads_recipe_must_match_full_runtime_recipe(tmp_path: Path) -> None:
    """The admission recipe binds code hashes and every runtime hyperparameter."""
    from research.sandbox_b_heads import resume

    data, b8 = tmp_path / "data.npz", tmp_path / "b8.pth"
    data.write_bytes(b"closed-f0")
    b8.write_bytes(b"frozen-b8")
    payload = _payload()
    payload["inputs"] = [
        {"kind": "closed-f0", "path": str(data), "sha256": resume.sha256(data)},
        {"kind": "frozen-checkpoint", "path": str(b8), "sha256": resume.sha256(b8)},
    ]
    payload["endpoint"] = {"kind": "updates", "value": 15000}
    recipe = resume.recipe(data, b8, salt=payload["salt"])
    ledger = {
        "schema": resume.SCHEMA,
        "namespace": resume.NAMESPACE,
        "allocation": resume.ALLOCATION,
        "recipe": recipe,
        "fingerprint": resume.fingerprint(recipe),
    }
    packet = {"env": {"SANDBOX_B_SALT": payload["salt"]}}
    _validate_b_heads_runtime_recipe(
        payload, packet, ledger, payload["inputs"][:1], payload["inputs"][1:]
    )

    ledger["recipe"] = {**recipe, "learning_rate": 0.01}
    with pytest.raises(PreflightError, match="full shipped runtime recipe"):
        _validate_b_heads_runtime_recipe(
            payload, packet, ledger, payload["inputs"][:1], payload["inputs"][1:]
        )


def test_b_heads_recipe_rejects_runtime_salt_or_endpoint_drift(tmp_path: Path) -> None:
    from research.sandbox_b_heads import resume

    data, b8 = tmp_path / "data.npz", tmp_path / "b8.pth"
    data.write_bytes(b"closed-f0")
    b8.write_bytes(b"frozen-b8")
    payload = _payload()
    payload["inputs"] = [
        {"kind": "closed-f0", "path": str(data), "sha256": resume.sha256(data)},
        {"kind": "frozen-checkpoint", "path": str(b8), "sha256": resume.sha256(b8)},
    ]
    recipe = resume.recipe(data, b8, salt=payload["salt"])
    ledger = {
        "schema": resume.SCHEMA,
        "namespace": resume.NAMESPACE,
        "allocation": resume.ALLOCATION,
        "recipe": recipe,
        "fingerprint": resume.fingerprint(recipe),
    }
    packet = {"env": {"SANDBOX_B_SALT": "a100-dev/sandbox-b/other"}}
    with pytest.raises(PreflightError, match="runtime salt"):
        _validate_b_heads_runtime_recipe(
            payload, packet, ledger, payload["inputs"][:1], payload["inputs"][1:]
        )

    assert _runtime_updates(resume, None, recipe) == 15000


def test_b3_recipe_binds_its_new_ledger_identity_and_configured_updates(tmp_path: Path) -> None:
    """B-3 cannot inherit B-2's ledger identity or endpoint by accident."""
    from research.sandbox_b_heads import resume_b3

    data, b8 = tmp_path / "data.npz", tmp_path / "b8.pth"
    data.write_bytes(b"closed-f0")
    b8.write_bytes(b"frozen-b8")
    payload = _payload()
    payload["study"] = "sandbox-b-heads-v3"
    payload["inputs"] = [
        {"kind": "closed-f0", "path": str(data), "sha256": resume_b3.sha256(data)},
        {"kind": "frozen-checkpoint", "path": str(b8), "sha256": resume_b3.sha256(b8)},
    ]
    recipe = resume_b3.recipe(data, b8, salt=payload["salt"])
    payload["endpoint"] = {
        "kind": "updates",
        "value": _runtime_updates(resume_b3, "research.sandbox_b_heads.core_b3", recipe),
    }
    ledger = {
        "schema": resume_b3.SCHEMA,
        "namespace": resume_b3.NAMESPACE,
        "allocation": resume_b3.ALLOCATION,
        "recipe": recipe,
        "fingerprint": resume_b3.fingerprint(recipe),
    }
    packet = {"env": {"SANDBOX_B_SALT": payload["salt"]}}
    _validate_b_heads_runtime_recipe(
        payload, packet, ledger, payload["inputs"][:1], payload["inputs"][1:]
    )
    assert payload["endpoint"]["value"] == 6000

    ledger["schema"] = "sandbox-b-verified-ledger/v4"
    with pytest.raises(PreflightError, match="foreign resume ledger identity"):
        _validate_b_heads_runtime_recipe(
            payload, packet, ledger, payload["inputs"][:1], payload["inputs"][1:]
        )


def test_snapshot_derivative_kind_is_admitted_only_for_train01() -> None:
    payload = _payload()
    payload["inputs"].append(
        {
            "kind": "training-export-derivative",
            "path": "exports/iteration-0000.pt.xz",
            "sha256": "c" * 64,
        }
    )
    with pytest.raises(ManifestError, match="exclusive to sandbox-a-train01"):
        validate_manifest(payload)
    payload["study"] = "sandbox-a-train01"
    payload["inputs"][2]["study"] = "sandbox-a-train01"
    assert validate_manifest(payload)["inputs"][-1]["kind"] == "training-export-derivative"
    payload["inputs"][-1].pop("sha256")
    with pytest.raises(ManifestError):
        validate_manifest(payload)
