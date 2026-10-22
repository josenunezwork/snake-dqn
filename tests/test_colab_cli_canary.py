"""Canary receipts must prove the actual broker/scanner/decision path offline."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.colab_cli import canary, runtime


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value))


def _fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu: str = "cpu"):
    data = tmp_path / "input"
    data.write_bytes(b"real bytes")
    template = {
        "script": "research/bl_ppo/colab.sh",
        "commit": "a" * 40,
        "job_id": "science",
        "env": {"BL_PPO_MODE": "ruler"},
        "inputs": [{"path": str(data), "dest": "fixture.bin", "sha256": canary.digest(data)}],
        "requirements": {"gpu": gpu, "allowed_gpus": [gpu], "est_minutes": 1},
        "timeout_minutes": 20,
        "required_exports": ["result.json"],
    }
    output = tmp_path / "probe-output"
    output.mkdir()
    probe = canary.prepare(template, package="bl_ppo", output=output, job_id="probe")
    _json(output / "manifest.json", probe)
    _json(output / "upload-measurement.json", {"elapsed_seconds": 149})
    _json(output / "hardware.json", {"hardware": {"kind": gpu}})
    remote = tmp_path / "remote"
    (remote / "output").mkdir(parents=True)
    payload = remote / "output/result.json"
    payload.write_text('{"actual": true}')
    Path(str(payload) + ".sha256").write_text(canary.digest(payload) + "  result.json\n")
    Path(str(payload) + ".complete").write_text("complete\n")
    inventory = runtime._exports(remote, 1024)
    (output / "exports").mkdir()
    (output / "exports/result.json").write_bytes(payload.read_bytes())
    receipt = {
        "status": "PASS",
        "stop_verified": True,
        "failure_class": None,
        "generation": "testgen",
        "commit": probe["commit"],
        "job_id": probe["job_id"],
        "manifest_sha256": canary.digest(output / "manifest.json"),
        "exports": inventory,
        "terminal": {
            "job_id": probe["job_id"],
            "generation": "testgen",
            "returncode": 0,
            "status": "succeeded",
            "export_scanner": {"name": "runtime._exports", "inventory": inventory},
            "canary_observations": [],
        },
        "transfers": [
            {
                "direction": "upload",
                "status": "verified",
                "remote": "/content/root/inputs/fixture.bin",
                "size_bytes": data.stat().st_size,
                "sha256": canary.digest(data),
            }
        ],
    }
    monkeypatch.setattr(canary, "executable_identity", lambda *args: {"transport.py": "hash"})
    return template, probe, receipt, output


def _bind(template, probe, receipt, output, tmp_path):
    receipt["canary_evidence"] = canary.evidence(probe, receipt, output, tmp_path)
    _json(output / "receipt.json", receipt)
    template["canary"] = {
        "receipt": {
            "path": str(output / "receipt.json"),
            "sha256": canary.digest(output / "receipt.json"),
        }
    }


def test_actual_scanner_transport_release_admits(tmp_path, monkeypatch):
    template, probe, receipt, output = _fixture(tmp_path, monkeypatch)
    _bind(template, probe, receipt, output, tmp_path)
    canary.check(template, tmp_path)


@pytest.mark.parametrize(
    "change",
    [
        "commit",
        "family",
        "missing",
        "larger",
        "count",
        "hardware",
        "export",
        "tamper",
        "retained",
        "substitute",
        "foreign",
        "allocation_target",
    ],
)
def test_receipt_cannot_admit_an_uncovered_dispatch(tmp_path, monkeypatch, change):
    template, probe, receipt, output = _fixture(tmp_path, monkeypatch)
    if change == "retained":
        receipt["stop_verified"] = False
    _bind(template, probe, receipt, output, tmp_path)
    if change == "commit":
        template["commit"] = "b" * 40
    elif change == "family":
        template["env"]["BL_PPO_MODE"] = "train"
    elif change == "missing":
        template.pop("canary")
    elif change == "larger":
        Path(template["inputs"][0]["path"]).write_bytes(b"x" * 100)
    elif change == "count":
        template["inputs"].append({**template["inputs"][0], "dest": "extra"})
    elif change == "allocation_target":
        template["requirements"]["gpu"] = "H100"
    elif change == "hardware":
        template["requirements"]["allowed_gpus"] = ["L4"]
    elif change == "export":
        template["required_exports"].append("missing.json")
    elif change == "substitute":
        other = tmp_path / "substitute"
        other.write_bytes(b"fake bytes")
        template["inputs"][0].update(path=str(other), sha256=canary.digest(other))
    elif change == "foreign":
        receipt["terminal"]["generation"] = "foreign"
        _json(output / "receipt.json", receipt)
        template["canary"]["receipt"]["sha256"] = canary.digest(output / "receipt.json")
    elif change == "tamper":
        (output / "exports/result.json").write_bytes(b"changed")
    with pytest.raises(canary.CanaryError):
        canary.check(template, tmp_path)


@pytest.mark.parametrize("fault", ["slow", "scanner", "upload", "sidecars", "cuda"])
def test_canary_proof_requires_real_observations(tmp_path, monkeypatch, fault):
    template, probe, receipt, output = _fixture(
        tmp_path, monkeypatch, "L4" if fault == "cuda" else "cpu"
    )
    if fault == "slow":
        _json(output / "upload-measurement.json", {"elapsed_seconds": 151})
    elif fault == "scanner":
        receipt["terminal"].pop("export_scanner")
    elif fault == "upload":
        receipt["transfers"] = []
    elif fault == "sidecars":
        receipt["exports"] = []
    with pytest.raises(canary.CanaryError):
        canary.evidence(probe, receipt, output, tmp_path)


def test_gpu_decision_observation_is_accepted(tmp_path, monkeypatch):
    template, probe, receipt, output = _fixture(tmp_path, monkeypatch, "L4")
    receipt["terminal"]["canary_observations"] = [
        {
            "cuda_decisions": 1,
            "device": "cuda:0",
            "package": "research/bl_ppo",
            "caller": "research/bl_ppo/model.py:277:act",
            "operation": "Tensor.argmax",
            "shape": [1, 6],
            "root": "/content/snake-cli/testgen",
        }
    ]
    _bind(template, probe, receipt, output, tmp_path)
    canary.check(template, tmp_path)


def test_probe_retains_real_input_inventory_and_real_entrypoint(tmp_path, monkeypatch):
    template, probe, receipt, output = _fixture(tmp_path, monkeypatch)
    assert probe["inputs"] == template["inputs"]
    assert probe["script"] == template["script"]
    canary.check(probe, tmp_path)
    probe["timeout_minutes"] = 31
    with pytest.raises(canary.CanaryError, match="30 minutes"):
        canary.check(probe, tmp_path)


def test_acceptance_bootstrap_is_not_a_family_receipt(tmp_path):
    canary.check({"script": canary.ACCEPTANCE_SCRIPT}, tmp_path)
    with pytest.raises(canary.CanaryError):
        canary.check({"script": canary.ACCEPTANCE_SCRIPT, "canary": {"mode": "probe"}}, tmp_path)


def test_broker_refuses_science_before_any_allocator_call(tmp_path, monkeypatch):
    import subprocess

    from research.colab_cli import supervisor

    repo = Path(__file__).resolve().parents[1]
    monkeypatch.setattr(supervisor, "ACCOUNT_ROOT", tmp_path / "account")
    template = {
        "job_id": "missing-canary",
        "commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip(),
        "script": "research/bl_ppo/colab.sh",
        "requirements": {"gpu": "L4", "min_vcpu": 1, "est_minutes": 1},
        "timeout_minutes": 3,
        "budget": {"unit_cap": 1, "rate_units_per_hour": 2, "other_assignments": {}},
        "required_exports": ["e4-ruler.json"],
        "output_dir": str(tmp_path / "fresh-output"),
    }
    calls = []
    monkeypatch.setattr(supervisor.CLI, "call", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(supervisor.Refused, match="SHA-bound family canary"):
        supervisor.Broker(tmp_path / "state", "never-run").run(template)
    assert calls == []
    assert not Path(template["output_dir"]).exists()
