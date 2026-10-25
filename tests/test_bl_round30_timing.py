"""Synthetic redaction and full-plan scheduling proof; no campaign payloads."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_round30 import timing


def config() -> dict:
    from research.bl_round30.timing_pack import identity_paths

    return {
        "schema": "r30-timing-config/v1",
        "ruling_sha256": timing.RULING_SHA,
        "amendment_sha256": timing.AMENDMENT_SHA,
        "deadlines": copy.deepcopy(timing.DEADLINES),
        "source_commit": "a" * 40,
        "source_hashes": {name: digest(timing.ROOT / name) for name in identity_paths()},
        "bundle_sha256": "b" * 64,
        "control_sha256": "c" * 64,
        "input_hashes": {
            name: "d" * 64
            for name in ("b8.pth", "train.npz", "sampler-plan.npz", "sampler-plan.json")
        },
        "schedule": copy.deepcopy(timing.SCHEDULE),
        "synthetic": True,
    }


def report() -> dict:
    c = config()
    return {
        "schema": timing.SCHEMA,
        "complete": True,
        "synthetic": True,
        "ruling_sha256": timing.RULING_SHA,
        "amendment_sha256": timing.AMENDMENT_SHA,
        "deadlines": copy.deepcopy(timing.DEADLINES),
        "config_sha256": "e" * 64,
        **{
            key: c[key]
            for key in (
                "source_commit",
                "source_hashes",
                "input_hashes",
                "bundle_sha256",
                "control_sha256",
                "schedule",
            )
        },
        "device": "cpu",
        "threads": 1,
        "row_count": 4,
        "feature_shape": [4, 256],
        "parameter_count": 1542,
        "phase_seconds": {name: 0.01 for name in timing.PHASES},
        "worker_seconds": 2.0,
    }


@pytest.mark.parametrize(
    "key",
    [
        "parameters",
        "gradient",
        "loss",
        "diagnostics",
        "actions",
        "certificate_bound",
        "certified",
        "optimizer_state",
        "cache",
        "feature_singular_values",
        "unit_id",
    ],
)
def test_export_rejects_every_scientific_field(key: str) -> None:
    value = report()
    value[key] = {"SECRET_SCIENTIFIC_SENTINEL": torch.tensor([12345.6789])}
    with pytest.raises(ValueError, match="keys"):
        timing.validate_report(value)


@pytest.mark.parametrize(
    "path,bad",
    [
        (("phase_seconds", "certificate"), {"loss": 12345.6789}),
        (("phase_seconds", "certificate"), float("nan")),
        (("phase_seconds", "certificate"), 600.0),
        (("phase_seconds", "certificate"), True),
        (("input_hashes", "train.npz"), "SECRET_SCIENTIFIC_SENTINEL"),
        (("source_hashes", "foreign-path"), "a" * 64),
    ],
)
def test_nested_redaction_and_timeout_refusal(path: tuple, bad: object) -> None:
    value = report()
    value[path[0]][path[1]] = bad
    with pytest.raises(ValueError):
        timing.validate_report(value)


def test_fixed_full_plan_schedule_and_no_scientific_return(monkeypatch: pytest.MonkeyPatch) -> None:
    from research.bl_ppo import exports
    from research.bl_round30 import certificate, objective
    from src.model.ego2s_network import Ego2sNet
    from tests.test_bl_round30_objective import _cache

    calls = []
    updates = 0
    original_step = torch.optim.Adam.step
    original_evaluate = objective.evaluate_objective
    original_diagnostics = objective.component_diagnostics
    original_publish = exports._publish
    scratch_publications = []

    def publish(path: Path, payload: bytes) -> None:
        scratch_publications.append(path.name)
        if path.name == "synthetic-io.bin":
            assert len(payload) == 128 * 1024**2
            assert payload.count(0) == len(payload)
        else:
            assert path.name in {"synthetic-io.bin.sha256", "synthetic-io.bin.complete"}
            assert b"SECRET" not in payload
        original_publish(path, payload)

    def step(self: object, *args: object, **kwargs: object) -> object:
        nonlocal updates
        result = original_step(self, *args, **kwargs)
        updates += 1
        return result

    def evaluate(
        theta: torch.Tensor, cache: dict, plan: object, recipe: object, **kwargs: object
    ) -> object:
        calls.append((updates, tuple(plan.indices.shape), plan.seed, theta.dtype))
        original_evaluate(theta, cache, plan, recipe, **kwargs)
        return {"SECRET_GRADIENT_LOSS_SENTINEL": torch.tensor([12345.6789])}

    original_certificate = certificate.gradient_certificate

    def cert(
        theta: torch.Tensor, cache: dict, plan: object, recipe: object, **kwargs: object
    ) -> dict:
        assert updates == 30 and kwargs["max_seconds"] == 600
        assert tuple(plan.indices.shape) == (2000, 256)
        original_certificate(theta, cache, plan, recipe, **kwargs)
        return {"SECRET_SCIENTIFIC_SENTINEL": 12345.6789, "certified": False}

    monkeypatch.setattr(torch.optim.Adam, "step", step)
    monkeypatch.setattr(objective, "evaluate_objective", evaluate)
    monkeypatch.setattr(certificate, "gradient_certificate", cert)

    def diagnostics(*args: object, **kwargs: object) -> dict:
        original_diagnostics(*args, **kwargs)
        return {"SECRET_DIAGNOSTIC_SENTINEL": 12345.6789}

    monkeypatch.setattr(objective, "component_diagnostics", diagnostics)
    monkeypatch.setattr(exports, "_publish", publish)
    timer = timing.Timer()
    rows = timing.workload(_cache(), Ego2sNet(local_channels=7, n_scalars=60), timer)
    assert rows == 4 and updates == 30
    assert [x[0] for x in calls] == [0, 10, 30, 30]  # last is main auxiliary cost
    assert all(x[1:] == ((2000, 256), 3403, torch.float64) for x in calls)
    assert "SECRET" not in json.dumps(timer.seconds)
    assert all(type(v) is float for v in timer.seconds.values())
    assert scratch_publications == [
        "synthetic-io.bin",
        "synthetic-io.bin.sha256",
        "synthetic-io.bin.complete",
    ]


def test_fixed_row_b8_masked_action_discards_values_and_uses_six_actions() -> None:
    """The hermetic analogue preserves the real probe's fixed-row contract."""
    from research.bl_round30.fixture import synthetic_table
    from src.model.ego2s_network import Ego2sNet

    model = Ego2sNet(local_channels=7, n_scalars=60).eval()
    table = synthetic_table(model)
    table["action_mask"][0] = False  # exercise the fixed normal-action fallback.
    assert timing._b8_masked_action_decision(model, table, device="cpu", synthetic=True) is None
    with pytest.raises(ValueError, match="requires one CUDA"):
        timing._b8_masked_action_decision(model, table, device="cpu", synthetic=False)


def test_observer_record_for_real_masked_q_contains_operational_metadata_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Hermetic observer seam; it does not manufacture runtime evidence."""
    from research.bl_round30.fixture import synthetic_table
    from research.colab_cli import canary_observer
    from src.model.ego2s_network import Ego2sNet

    class CudaDevice:
        type = "cuda"

        def __str__(self) -> str:
            return "cuda:0"

    model = Ego2sNet(local_channels=7, n_scalars=60).eval()
    table = synthetic_table(model)
    with torch.no_grad():
        q = model(*[torch.as_tensor(table[key][:1]) for key in ("local", "global", "scalars")])
        masked_q = q.masked_fill(~torch.as_tensor(table["action_mask"][:1]), -torch.inf)

    root = tmp_path / "root"
    package = "research/bl_round30"
    (root / "repo" / package).mkdir(parents=True)
    carrier = SimpleNamespace(shape=masked_q.shape, device=CudaDevice())
    monkeypatch.setattr(
        canary_observer,
        "_caller",
        lambda package_root, repository: (
            "research/bl_round30/timing.py:999:_b8_masked_action_decision"
        ),
    )
    canary_observer._observe(
        "Tensor.argmax",
        carrier,
        lambda: masked_q.argmax(dim=1),
        root=root,
        package_root=root / "repo" / package,
        package=package,
        torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: None)),
    )
    record = json.loads(next((root / "canary-observations").glob("*.json")).read_text())
    assert set(record) == {
        "version",
        "pid",
        "root",
        "package",
        "cuda_decisions",
        "operation",
        "caller",
        "device",
        "shape",
    }
    encoded = json.dumps(record, sort_keys=True)
    assert "SECRET_SCIENTIFIC_SENTINEL" not in encoded
    assert '"action":' not in encoded and '"q":' not in encoded


def test_real_deadline_reserves_cleanup_export_from_supervisor_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    now = 10_000.0
    (tmp_path / "spec.json").write_text(json.dumps({"timeout_seconds": 1642.5}))
    (tmp_path / "state.json").write_text(json.dumps({"started_at": now - 100}))
    monkeypatch.setenv("COLAB_CLI_ROOT", str(tmp_path))
    monkeypatch.setattr(timing.time, "time", lambda: now)
    assert (
        timing._worker_deadline_seconds(input_dir=tmp_path, synthetic=False, launch_age=0) == 1242.5
    )
    assert timing._worker_deadline_seconds(input_dir=tmp_path, synthetic=True, launch_age=0) == 1500


@pytest.mark.parametrize("field,bad", [("amendment_sha256", "0" * 64), ("deadlines", {})])
def test_export_rejects_unbound_amendment_or_deadlines(field: str, bad: object) -> None:
    value = report()
    value[field] = bad
    with pytest.raises(ValueError):
        timing.validate_report(value)


def test_cli_real_synthetic_workload_and_only_timing_export(tmp_path: Path) -> None:
    inputs, output = tmp_path / "inputs", tmp_path / "output"
    inputs.mkdir()
    export_json(inputs / "timing-config.json", config())
    result = subprocess.run(
        [
            str(timing.ROOT / "venv/bin/python"),
            "-m",
            "research.bl_round30.timing",
            "--input-dir",
            str(inputs),
            "--output",
            str(output),
            "--config-sha256",
            digest(inputs / "timing-config.json"),
            "--synthetic",
        ],
        cwd=timing.ROOT,
        capture_output=True,
        text=True,
        timeout=150,
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
    )
    assert result.returncode == 0, (result.stdout, result.stderr)
    assert result.stdout == '{"schema":"r30-timing-operational/v1","complete":true}\n'
    assert result.stderr == ""
    assert {p.name for p in output.iterdir()} == {
        "timing.json",
        "timing.json.sha256",
        "timing.json.complete",
    }
    verify_export(output / "timing.json")
    timing.validate_report(json.loads((output / "timing.json").read_text()))


def test_exception_native_logs_and_numerical_outputs_are_redacted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture
) -> None:
    inputs, output = tmp_path / "inputs", tmp_path / "output"
    inputs.mkdir()
    export_json(inputs / "timing-config.json", config())

    def bad(*args: object, **kwargs: object) -> None:
        print("SECRET_SCIENTIFIC_SENTINEL 12345.6789")
        os.write(2, b"SECRET_NATIVE_GRADIENT 12345.6789\n")
        raise TimeoutError("SECRET_LOSS 12345.6789")

    monkeypatch.setattr(timing, "workload", bad)
    with pytest.raises(TimeoutError):
        timing.run(
            inputs, output, config_sha256=digest(inputs / "timing-config.json"), synthetic=True
        )
    assert capfd.readouterr() == ("", "")
    assert list(output.iterdir()) == []


def test_schedule_change_refused_before_any_training_read(tmp_path: Path) -> None:
    c = config()
    c["schedule"]["fit_batches"] = 3
    export_json(tmp_path / "timing-config.json", c)
    with pytest.raises(ValueError, match="schedule"):
        timing.run(
            tmp_path,
            tmp_path / "out",
            config_sha256=digest(tmp_path / "timing-config.json"),
            synthetic=True,
        )


def test_marker_last_immutable_and_never_scientific_ledger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.bl_ppo import exports
    from research.bl_round30.resume import build

    order = []
    publish = exports._publish

    def spy(path: Path, payload: bytes) -> None:
        order.append(path.name)
        publish(path, payload)

    monkeypatch.setattr(exports, "_publish", spy)
    value = report()
    timing.validate_report(value)
    export_json(tmp_path / "timing.json", value)
    assert order == ["timing.json", "timing.json.sha256", "timing.json.complete"]
    with pytest.raises(FileExistsError):
        export_json(tmp_path / "timing.json", value)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "control.json").write_text(json.dumps({"namespace": "science", "commit": "a" * 40}))
    with pytest.raises(ValueError, match="missing or unsafe"):
        build(inputs, [tmp_path], tmp_path / "ledger.json")
    assert not (tmp_path / "ledger.json").exists()


def test_deadline_never_returns_partial_completion() -> None:
    import time

    with pytest.raises(TimeoutError):
        with timing.deadline(0.01):
            time.sleep(0.05)


def test_current_qualified_transport_has_no_forty_minute_calibration_route(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from research.colab_cli import canary

    manifest = {
        "script": "research/bl_round30/timing_colab.sh",
        "device": "cuda",
        "required_exports": ["timing.json"],
        "timeout_minutes": 40,
        "env": {},
        "commit": "a" * 40,
    }
    monkeypatch.setattr(canary, "_entrypoint", lambda *args: "python -m timing")
    with pytest.raises(canary.CanaryError, match="SHA-bound family canary receipt"):
        canary.check(manifest, timing.ROOT)
    manifest["canary"] = {"mode": "probe", "schema": canary.SCHEMA, "family": manifest["script"]}
    with pytest.raises(canary.CanaryError, match="<=30 minutes"):
        canary.check(manifest, timing.ROOT)


def test_producer_rejects_oversized_valid_metadata_before_publication(tmp_path, monkeypatch):
    from research.bl_round30 import timing_pack

    value = report()
    value["source_hashes"] = {f"source-{index}.py": "a" * 64 for index in range(8000)}
    monkeypatch.setattr(timing_pack, "identity_paths", lambda: list(value["source_hashes"]))
    assert len(json.dumps(value).encode()) > 524_288
    with pytest.raises(ValueError, match="timing payload exceeds 512 KiB"):
        timing.publish_report(tmp_path, value)
    assert list(tmp_path.iterdir()) == []


def test_output_admission_rejects_oversize_and_extra_scientific_exports(tmp_path):
    export_json(tmp_path / "timing.json", report())
    timing.validate_output_inventory(tmp_path)
    export_json(tmp_path / "scientific-unit.json", {"SECRET": 1})
    with pytest.raises(ValueError, match="inventory differs"):
        timing.validate_output_inventory(tmp_path)
    for path in tmp_path.glob("scientific-unit.json*"):
        path.unlink()
    (tmp_path / "timing.json").write_bytes(bytes(524_289))
    with pytest.raises(ValueError, match="timing payload exceeds 512 KiB"):
        timing.validate_output_inventory(tmp_path)
