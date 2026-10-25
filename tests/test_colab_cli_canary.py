"""Canary receipts must prove the actual broker/scanner/decision path offline."""

from __future__ import annotations

import json
import subprocess
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
        "env": {"BL_PPO_MODE": "ruler", "BL_PPO_DEVICE": "cpu" if gpu == "cpu" else "cuda"},
        "inputs": [{"path": str(data), "dest": "fixture.bin", "sha256": canary.digest(data)}],
        "requirements": {"gpu": gpu, "allowed_gpus": [gpu], "est_minutes": 1},
        "timeout_minutes": 20,
        "required_exports": ["result.json"],
    }
    output = tmp_path / "probe-output"
    output.mkdir()
    probe = canary.prepare(template, package="bl_ppo", output=output, job_id="probe")
    _json(output / "manifest.json", probe)
    _json(
        output / "upload-measurement.json",
        {"elapsed_seconds": 149, "transfer_seconds": 21, "source_delivery_seconds": 120},
    )
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
    monkeypatch.setattr(canary, "_entrypoint", lambda *args: "")
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
        "device",
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
    elif change == "device":
        template["env"]["BL_PPO_DEVICE"] = "cuda"
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


@pytest.mark.parametrize("gpu", ["cpu", "L4"])
@pytest.mark.parametrize("fault", ["slow", "scanner", "upload", "sidecars"])
def test_canary_proof_requires_real_observations(tmp_path, monkeypatch, fault, gpu):
    template, probe, receipt, output = _fixture(tmp_path, monkeypatch, gpu)
    probe["env"]["BL_PPO_DEVICE"] = "cpu"
    if fault == "slow":
        _json(
            output / "upload-measurement.json",
            {"elapsed_seconds": 301, "transfer_seconds": 21, "source_delivery_seconds": 120},
        )
    elif fault == "scanner":
        receipt["terminal"].pop("export_scanner")
    elif fault == "upload":
        receipt["transfers"] = []
    elif fault == "sidecars":
        receipt["exports"] = []
    with pytest.raises(canary.CanaryError):
        canary.evidence(probe, receipt, output, tmp_path)


def test_staging_accepts_slow_source_delivery_when_transfers_are_within_cap(tmp_path, monkeypatch):
    _, probe, receipt, output = _fixture(tmp_path, monkeypatch)
    _json(
        output / "upload-measurement.json",
        {"elapsed_seconds": 177.8, "transfer_seconds": 21.0, "source_delivery_seconds": 156.8},
    )

    proof = canary.evidence(probe, receipt, output, tmp_path)

    assert proof["upload_elapsed_seconds"] == 177.8
    assert proof["transfer_seconds"] == 21.0
    assert proof["source_delivery_seconds"] == 156.8


@pytest.mark.parametrize(
    "measurement",
    [
        {"elapsed_seconds": 300, "transfer_seconds": 150, "source_delivery_seconds": 150},
        {"elapsed_seconds": 150, "transfer_seconds": 150, "source_delivery_seconds": 0},
    ],
)
def test_staging_timing_accepts_exact_caps(tmp_path, monkeypatch, measurement):
    _, probe, receipt, output = _fixture(tmp_path, monkeypatch)
    _json(output / "upload-measurement.json", measurement)

    canary.evidence(probe, receipt, output, tmp_path)


@pytest.mark.parametrize(
    ("measurement", "message"),
    [
        (
            {"elapsed_seconds": 200, "transfer_seconds": 150.000001, "source_delivery_seconds": 49},
            "transfers must complete within 150",
        ),
        (
            {
                "elapsed_seconds": 300.000001,
                "transfer_seconds": 150,
                "source_delivery_seconds": 150,
            },
            "staging must complete within 300",
        ),
        (
            {"elapsed_seconds": 200, "transfer_seconds": 21, "source_delivery_seconds": 180},
            "components exceed",
        ),
        (
            {"elapsed_seconds": 100, "transfer_seconds": 60, "source_delivery_seconds": 60},
            "components exceed",
        ),
    ],
)
def test_staging_timing_refuses_cap_or_component_overrun(
    tmp_path, monkeypatch, measurement, message
):
    _, probe, receipt, output = _fixture(tmp_path, monkeypatch)
    _json(output / "upload-measurement.json", measurement)

    with pytest.raises(canary.CanaryError, match=message):
        canary.evidence(probe, receipt, output, tmp_path)


@pytest.mark.parametrize(
    "measurement",
    [
        {},
        {"elapsed_seconds": 1, "transfer_seconds": 1},
        {"elapsed_seconds": 1, "transfer_seconds": 1, "source_delivery_seconds": float("nan")},
        {"elapsed_seconds": 1, "transfer_seconds": float("inf"), "source_delivery_seconds": 0},
        {"elapsed_seconds": True, "transfer_seconds": 1, "source_delivery_seconds": 0},
        {"elapsed_seconds": 1, "transfer_seconds": False, "source_delivery_seconds": 0},
        {"elapsed_seconds": 1, "transfer_seconds": 1, "source_delivery_seconds": True},
        {"elapsed_seconds": -1, "transfer_seconds": 1, "source_delivery_seconds": 0},
        {"elapsed_seconds": 1, "transfer_seconds": -1, "source_delivery_seconds": 0},
        {"elapsed_seconds": 1, "transfer_seconds": 1, "source_delivery_seconds": -1},
    ],
)
def test_staging_timing_requires_finite_nonboolean_fields(tmp_path, monkeypatch, measurement):
    _, probe, receipt, output = _fixture(tmp_path, monkeypatch)
    _json(output / "upload-measurement.json", measurement)

    with pytest.raises(canary.CanaryError):
        canary.evidence(probe, receipt, output, tmp_path)


def test_gpu_job_without_cuda_decision_refuses(tmp_path, monkeypatch):
    _, probe, receipt, output = _fixture(tmp_path, monkeypatch, "L4")
    with pytest.raises(canary.CanaryError, match="real CUDA action decision"):
        canary.evidence(probe, receipt, output, tmp_path)


@pytest.mark.parametrize("gpu", ["L4", "H100", "cpu"])
def test_cpu_job_on_any_host_admits_with_real_transport_and_exports(tmp_path, monkeypatch, gpu):
    template, probe, receipt, output = _fixture(tmp_path, monkeypatch, gpu)
    for manifest in (template, probe):
        manifest["env"]["BL_PPO_DEVICE"] = "cpu"
    _json(output / "manifest.json", probe)
    receipt["manifest_sha256"] = canary.digest(output / "manifest.json")
    _bind(template, probe, receipt, output, tmp_path)
    assert receipt["canary_evidence"]["device"] == "cpu"
    assert receipt["canary_evidence"]["cuda_observations"] == []
    canary.check(template, tmp_path)


@pytest.mark.parametrize(
    "declarations, expected",
    [
        ({"device": "cpu"}, "cpu"),
        ({"device": "cuda:1"}, "cuda"),
        ({"env": {"SNAKE_DQN_DEVICE": "cpu"}}, "cpu"),
        ({"env": {"BL_PPO_DEVICE": "cuda", "SNAKE_DQN_DEVICE": "cpu"}}, "cuda"),
        ({"env": {"ARGS": "--device cuda:0 --workers 10"}}, "cuda"),
        ({"env": {"ARGS": "--device=cpu"}}, "cpu"),
        ({"env": {"SNAKE_DQN_DEVICE": "cpu", "ARGS": "--device cuda"}}, "cuda"),
    ],
)
def test_job_device_resolves_declarations(declarations, expected):
    assert canary.job_device(declarations) == expected


@pytest.mark.parametrize(
    "script, env, expected",
    [
        ("python run.py --device cuda", {"SNAKE_DQN_DEVICE": "cpu"}, "cuda"),
        ("python run.py --device=cpu", {}, "cpu"),
        ('python run.py --device "${BL_PPO_DEVICE}"', {"BL_PPO_DEVICE": "cuda"}, "cuda"),
        ('python run.py --device "$BL_PPO_DEVICE"', {"BL_PPO_DEVICE": "cpu"}, "cpu"),
        ("# --device cuda\npython run.py --device cpu", {}, "cpu"),
        ("export SNAKE_DQN_DEVICE=cuda\npython run.py", {"SNAKE_DQN_DEVICE": "cpu"}, "cuda"),
    ],
)
def test_entrypoint_device_flags_override_cpu_defaults(
    tmp_path, monkeypatch, script, env, expected
):
    monkeypatch.setattr(canary, "_entrypoint", lambda *args: script)
    assert canary.job_device({"env": env}, tmp_path) == expected


def test_unresolved_entrypoint_device_refuses_even_with_cpu_declaration(tmp_path, monkeypatch):
    monkeypatch.setattr(
        canary, "_entrypoint", lambda *args: 'python run.py --device "${local_device}"'
    )
    with pytest.raises(canary.CanaryError, match="ambiguous job device"):
        canary.job_device({"env": {"SNAKE_DQN_DEVICE": "cpu"}}, tmp_path)


@pytest.mark.parametrize("choice", ["cuda", '"${UNKNOWN}"'])
def test_reassigned_entrypoint_device_cannot_use_original_manifest_value(
    tmp_path, monkeypatch, choice
):
    monkeypatch.setattr(
        canary,
        "_entrypoint",
        lambda *args: f'BL_PPO_DEVICE={choice}\npython run.py --device "$BL_PPO_DEVICE"',
    )
    with pytest.raises(canary.CanaryError, match="ambiguous job device"):
        canary.job_device({"env": {"BL_PPO_DEVICE": "cpu"}}, tmp_path)


def test_hardcoded_cuda_entrypoint_requires_real_cuda_decision(tmp_path, monkeypatch):
    _, probe, receipt, output = _fixture(tmp_path, monkeypatch, "L4")
    probe["env"]["BL_PPO_DEVICE"] = "cpu"
    monkeypatch.setattr(canary, "_entrypoint", lambda *args: "python run.py --device cuda")
    with pytest.raises(canary.CanaryError, match="real CUDA action decision"):
        canary.evidence(probe, receipt, output, tmp_path)


def test_real_pinned_e16_cuda_argument_cannot_be_hidden_by_cpu_env(tmp_path, monkeypatch):
    read_entrypoint = canary._entrypoint
    _, probe, receipt, output = _fixture(tmp_path, monkeypatch, "L4")
    repo = Path(__file__).resolve().parents[1]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    probe.update(script="research/bl_e16/colab.sh", commit=commit)
    probe["env"] = {"SNAKE_DQN_DEVICE": "cpu"}
    receipt["commit"] = commit
    monkeypatch.setattr(canary, "_entrypoint", read_entrypoint)
    with pytest.raises(canary.CanaryError, match="real CUDA action decision"):
        canary.evidence(probe, receipt, output, repo)


def test_observed_cuda_cannot_waive_attribution_for_declared_cpu(tmp_path, monkeypatch):
    _, probe, receipt, output = _fixture(tmp_path, monkeypatch, "L4")
    probe["env"]["BL_PPO_DEVICE"] = "cpu"
    receipt["terminal"]["canary_observations"] = [{"device": "cuda:0"}]
    with pytest.raises(canary.CanaryError, match="real CUDA action decision"):
        canary.evidence(probe, receipt, output, tmp_path)


@pytest.mark.parametrize(
    "declarations",
    [
        {},
        {"device": "auto"},
        {"device": "mps"},
        {"device": None},
        {"env": {"SNAKE_DQN_DEVICE": ""}},
        {"env": {"SNAKE_DQN_DEVICE": "cpu", "BL_PPO_DEVICE": "auto"}},
        {"env": {"ARGS": "--device"}},
        {"env": {"ARGS": "--device=${DEVICE}"}},
        {"env": {"ARGS": '--device "cuda'}},
    ],
)
def test_ambiguous_device_refuses_probe_and_production(tmp_path, monkeypatch, declarations):
    template, probe, receipt, output = _fixture(tmp_path, monkeypatch, "L4")
    for manifest in (template, probe):
        manifest.pop("env")
        manifest.update(declarations)
        with pytest.raises(canary.CanaryError, match="ambiguous job device"):
            canary.check(manifest, tmp_path)
    with pytest.raises(canary.CanaryError, match="ambiguous job device"):
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
        "env": {"BL_PPO_DEVICE": "cuda"},
        "output_dir": str(tmp_path / "fresh-output"),
    }
    calls = []
    monkeypatch.setattr(supervisor.CLI, "call", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(supervisor.Refused, match="SHA-bound family canary"):
        supervisor.Broker(tmp_path / "state", "never-run").run(template)
    assert calls == []
    assert not Path(template["output_dir"]).exists()


def _d1024_fixture(tmp_path, monkeypatch):
    from research.colab_cli import transfer_policy as policy

    template, probe, receipt, output = _fixture(tmp_path, monkeypatch, "L4")
    rows = []
    for dest, (size, sha) in policy.FROZEN_INPUTS.items():
        row = {"dest": dest, "sha256": sha}
        if dest == "b8.pth":
            row.update(url=policy.B8_URL, size_bytes=size)
        else:
            path = tmp_path / dest
            with path.open("wb") as stream:
                stream.truncate(size)
            row["path"] = str(path)
        rows.append(row)
    for manifest, mode in ((template, "main"), (probe, "canary")):
        manifest.update(
            script=policy.FIT_SCRIPT,
            device="cuda:0",
            inputs=rows,
            transfer_exception=policy.authority_binding("fit"),
            f4={"dose": 1024, "registered_updates": 2000},
            job_id=f"f4-d1024-{'fit' if mode == 'main' else 'canary'}-aaaaaaaa-04",
            env={
                "F4_MODE": mode,
                "F4_DOSE": "1024",
                "F4_DEVICE": "cuda:0",
                "METADATA_ARCHIVE_SHA256": policy.FROZEN_INPUTS["metadata.tar"][1],
            },
        )
    probe["env"].update(SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/bl_residual")
    probe["canary"]["family"] = policy.FIT_SCRIPT
    receipt["job_id"] = receipt["terminal"]["job_id"] = probe["job_id"]
    receipt["transfers"] = [
        {
            "direction": "upload",
            "status": "verified",
            "remote": "/content/root/inputs/" + dest,
            "size_bytes": size,
            "sha256": sha,
        }
        for dest, (size, sha) in policy.FROZEN_INPUTS.items()
        if dest != "b8.pth"
    ]
    receipt["terminal"]["canary_observations"] = [
        {
            "cuda_decisions": 1,
            "device": "cuda:0",
            "package": "research/bl_residual",
            "caller": "research/bl_residual/f4_canary.py",
            "operation": "Tensor.argmax",
            "shape": [1, 6],
            "root": "/content/snake-cli/testgen",
        }
    ]
    _json(output / "manifest.json", probe)
    receipt["manifest_sha256"] = canary.digest(output / "manifest.json")
    _json(
        output / "upload-measurement.json",
        {
            "elapsed_seconds": 236,
            "transfer_seconds": 222,
            "source_delivery_seconds": 10,
        },
    )
    return template, probe, receipt, output


def test_d1024_authorized_222_transfer_236_staging_and_consumer(tmp_path, monkeypatch):
    template, probe, receipt, output = _d1024_fixture(tmp_path, monkeypatch)
    _bind(template, probe, receipt, output, tmp_path)
    proof = receipt["canary_evidence"]
    assert proof["transfer_seconds"] == 222
    assert proof["upload_elapsed_seconds"] == 236
    assert proof["transfer_exception"] == template["transfer_exception"]
    canary.check(template, tmp_path)


@pytest.mark.parametrize(
    "transfer,staging,source,passes",
    [
        (300, 300, 0, True),
        (222, 300.000001, 10, False),
        (300.000001, 300, 0, False),
        (222, 236, 15, False),
    ],
)
def test_d1024_independent_caps_and_unmoved_clock(
    tmp_path, monkeypatch, transfer, staging, source, passes
):
    _, probe, receipt, output = _d1024_fixture(tmp_path, monkeypatch)
    _json(
        output / "upload-measurement.json",
        {
            "elapsed_seconds": staging,
            "transfer_seconds": transfer,
            "source_delivery_seconds": source,
        },
    )
    if passes:
        canary.evidence(probe, receipt, output, tmp_path)
    else:
        with pytest.raises(canary.CanaryError):
            canary.evidence(probe, receipt, output, tmp_path)


@pytest.mark.parametrize(
    "fault",
    [
        "missing",
        "foreign",
        "malformed",
        "malformed_context",
        "other_family",
        "dose",
        "mode",
        "hardware",
        "device",
        "recipe",
        "inventory",
        "hash",
        "metadata",
        "foreign_job",
    ],
)
def test_d1024_exception_cannot_escape_its_authority(tmp_path, monkeypatch, fault):
    _, probe, receipt, output = _d1024_fixture(tmp_path, monkeypatch)
    if fault == "missing":
        probe.pop("transfer_exception")
    elif fault == "foreign":
        probe["transfer_exception"]["ruling_sha256"] = "a" * 64
    elif fault == "malformed":
        probe["transfer_exception"] = None
    elif fault == "malformed_context":
        probe["transfer_exception"]["context"] = []
    elif fault == "other_family":
        probe["script"] = "research/bl_ppo/colab.sh"
    elif fault == "dose":
        probe["env"]["F4_DOSE"] = "4096"
    elif fault == "mode":
        probe["env"]["F4_MODE"] = "eval"
    elif fault == "hardware":
        probe["requirements"]["gpu"] = "A100"
    elif fault == "device":
        probe["env"]["F4_DEVICE"] = "cpu"
    elif fault == "recipe":
        probe["f4"]["registered_updates"] = 2001
    elif fault == "inventory":
        probe["inputs"].pop()
    elif fault == "hash":
        probe["inputs"][1]["sha256"] = "b" * 64
    elif fault == "metadata":
        probe["env"]["METADATA_ARCHIVE_SHA256"] = "b" * 64
    else:
        probe["job_id"] = "another-lane"
    with pytest.raises(canary.CanaryError):
        canary.evidence(probe, receipt, output, tmp_path)


def test_other_lane_still_refuses_222_seconds(tmp_path, monkeypatch):
    _, probe, receipt, output = _fixture(tmp_path, monkeypatch)
    _json(
        output / "upload-measurement.json",
        {
            "elapsed_seconds": 236,
            "transfer_seconds": 222,
            "source_delivery_seconds": 10,
        },
    )
    with pytest.raises(canary.CanaryError, match="within 150"):
        canary.evidence(probe, receipt, output, tmp_path)


def test_d1024_consumer_refuses_authority_removed_from_main(tmp_path, monkeypatch):
    template, probe, receipt, output = _d1024_fixture(tmp_path, monkeypatch)
    _bind(template, probe, receipt, output, tmp_path)
    template.pop("transfer_exception")
    with pytest.raises(canary.CanaryError, match="authority differs"):
        canary.check(template, tmp_path)


def test_d1024_binding_cannot_disguise_a_main_as_excluded_canary(tmp_path, monkeypatch):
    from research.colab_cli import transfer_policy as policy

    _, probe, _, _ = _d1024_fixture(tmp_path, monkeypatch)
    probe["env"]["F4_MODE"] = "main"
    probe["job_id"] = "f4-d1024-fit-aaaaaaaa-04"
    with pytest.raises(ValueError, match="package/mode/dose/recipe"):
        policy.transfer_limit(probe)


def test_d1024_binding_cannot_use_canary_workload_for_main(tmp_path, monkeypatch):
    from research.colab_cli import transfer_policy as policy

    main, _, _, _ = _d1024_fixture(tmp_path, monkeypatch)
    main["env"]["F4_MODE"] = "canary"
    main["job_id"] = "f4-d1024-canary-aaaaaaaa-04"
    with pytest.raises(ValueError, match="package/mode/dose/recipe"):
        policy.transfer_limit(main)
