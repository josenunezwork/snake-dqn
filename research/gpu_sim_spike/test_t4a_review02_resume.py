"""Offline T4A completion-attempt regression; no provider operations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from research.colab_cli import canary as qualified_canary
from research.colab_cli import supervisor as qualified_supervisor
from research.gpu_sim_spike import native_compile, t4a_analyze, t4a_pack, t4a_units
from research.gpu_sim_spike.make_manifests import ManifestRefused
from research.gpu_sim_spike.test_t4a_evidence import _content as t4a_canary_content
from tools.ops import dispatch as ops_dispatch
from tools.ops import qualification as ops_qualification


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _publish_failure(exports: Path) -> None:
    t4a_units.publish_json(
        exports / "t4a-failure.json",
        {
            "status": "INCOMPLETE",
            "reason": "OFFLINE_FIXTURE: RPC_EXEC_POLL_TIMEOUT",
        },
    )


def _interrupted_main(root: Path, *, commit: str, unit: str = "roster-frozen") -> tuple[Path, Path]:
    """Publish one completed roster payload and a sealed mechanical stop receipt."""
    exports = root / "prior-main" / "exports"
    namespace = t4a_units.namespace(commit)
    cells = t4a_units.allocation()
    t4a_units.publish_unit(
        exports,
        namespace,
        cells,
        unit,
        {"status": "COMPLETE", "fixture": "OFFLINE_FIXTURE", "unit_id": unit},
    )
    _publish_failure(exports)
    manifest = {
        "commit": commit,
        "script": t4a_pack.SCRIPT,
        "job_id": f"gpu-t4a-dev-01-main-{commit[:12]}-01",
        "env": {"T4A_BENCH_CANARY": "0"},
        "t4a_envelope": {"attempt": 1},
        "fixture_marker": "OFFLINE_FIXTURE",
    }
    manifest_path = exports.parent / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt = {
        "status": "INCOMPLETE",
        "stop_verified": True,
        "failure_class": "RPC_EXEC_POLL_TIMEOUT",
        "error": "OFFLINE_FIXTURE: mechanical interruption",
        "commit": commit,
        "manifest_sha256": _sha256(manifest_path),
        "fixture_marker": "OFFLINE_FIXTURE",
    }
    receipt_path = exports.parent / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt_path, exports


def _fixture_manifests(root: Path, *, commit: str, canary: dict, main: dict) -> tuple[dict, dict]:
    """Derive CPU-only CLI fixtures from the real packet input inventories."""

    def cpu_fixture(value: dict, *, output_name: str) -> dict:
        result = json.loads(json.dumps(value))
        result.update(
            {
                "device": "cpu",
                "requirements": {
                    "gpu": "cpu",
                    "allowed_gpus": ["cpu"],
                    "min_vcpu": 1,
                    "est_minutes": 1,
                },
                "timeout_minutes": 3,
                "budget": {
                    "unit_cap": 0.05,
                    "reserve_units": 0.0,
                    "rate_units_per_hour": 1.0,
                    "other_assignments": {},
                },
                "output_dir": str((root / output_name).resolve()),
                "fixture_marker": "OFFLINE_FIXTURE",
                "scientific_admission": t4a_pack.scientific_admission(),
            }
        )
        result["env"].pop("SNAKE_COLAB_CANARY", None)
        result["env"].pop("SNAKE_COLAB_CANARY_PACKAGE", None)
        result["env"]["T4A_BENCH_DEVICE"] = "cpu"
        return result

    probe = cpu_fixture(canary, output_name="canary-cli-output")
    probe.update(
        {
            "job_id": f"gpu-t4a-dev-01-canary-{commit[:12]}-02",
            "env": {
                **probe["env"],
                "SNAKE_COLAB_CANARY": "1",
                "SNAKE_COLAB_CANARY_PACKAGE": str(Path(t4a_pack.SCRIPT).parent),
            },
            "canary": {
                "schema": qualified_canary.SCHEMA,
                "mode": "probe",
                "family": qualified_canary.family(probe),
            },
        }
    )
    resumed = cpu_fixture(main, output_name="main-cli-output")
    resumed.update(
        {
            "job_id": f"gpu-t4a-dev-01-main-{commit[:12]}-02",
            "canary": {},
        }
    )
    return probe, resumed


def _offline_canary_receipt(
    root: Path, *, manifest: dict, repo: Path, t4a_canary: bool = False
) -> tuple[Path, dict]:
    """Create broker-shaped observations from bounded fixture files only."""
    output = root / "broker-observation"
    exports_dir = output / "exports"
    exports_dir.mkdir(parents=True)
    if t4a_canary:
        namespace = t4a_units.namespace(manifest["commit"], canary=True)
        cells = t4a_units.allocation(canary=True)
        for unit in cells:
            content = t4a_canary_content(t4a_units.parse(unit), canary=True)
            content.update(
                {
                    "event_seconds": 0.1,
                    "elapsed_seconds": 8.0,
                }
            )
            for record in content["records"]:
                record.update(
                    {
                        "build_seconds": 0.1,
                        "tick_seconds": [0.01, 0.01],
                        "comparison_seconds": 0.0,
                    }
                )
            t4a_units.publish_unit(
                exports_dir,
                namespace,
                cells,
                unit,
                content,
            )
        t4a_units.build_ledger(
            exports_dir,
            exports_dir / "verified-ledger.json",
            commit=manifest["commit"],
            canary=True,
        )
        t4a_units.publish_json(
            exports_dir / "native-parity.json",
            {"status": "PASS", "commit": manifest["commit"], "native_cuda_executed": True},
        )
        t4a_units.publish_json(
            exports_dir / "t4a-bench.json",
            {
                "status": "MEASURED_PENDING_MAC_REVIEW",
                "completed_units": list(cells),
                "native_elapsed_seconds": 1.0,
                "units": [
                    {
                        "horizon": 2,
                        "records": [
                            {
                                "build_seconds": 0.1,
                                "tick_seconds": [0.01, 0.01],
                                "comparison_seconds": 0.0,
                            }
                            for _record in range(18)
                        ],
                        "event_seconds": 0.1,
                        "elapsed_seconds": 8.0,
                    }
                    for _unit in cells
                ],
            },
        )
    for name in manifest["required_exports"]:
        payload = exports_dir / name
        if payload.is_file():
            continue
        payload.parent.mkdir(parents=True, exist_ok=True)
        payload.write_bytes(b"OFFLINE_FIXTURE\n")
    export_rows = [
        {
            "path": payload.relative_to(exports_dir).as_posix(),
            "size": payload.stat().st_size,
            "sha256": _sha256(payload),
        }
        for payload in sorted(exports_dir.rglob("*"))
        if payload.is_file() and not payload.name.endswith((".sha256", ".complete"))
    ]
    inputs = qualified_canary.input_inventory(manifest)
    transfers = [
        {
            "direction": "upload",
            "remote": f"/content/snake-cli/OFFLINE_FIXTURE/inputs/{row['dest']}",
            "status": "verified",
            "size_bytes": row["size"],
            "sha256": row["sha256"],
        }
        for row in inputs
        if row["transport"] == "upload"
    ]
    (output / "upload-measurement.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 1.0,
                "transfer_seconds": 0.5,
                "source_delivery_seconds": 0.25,
            },
            sort_keys=True,
        )
    )
    (output / "hardware.json").write_text(
        json.dumps({"hardware": {"kind": manifest["requirements"]["gpu"]}})
    )
    generation = "OFFLINE_FIXTURE"
    terminal = {
        "job_id": manifest["job_id"],
        "generation": generation,
        "returncode": 0,
        "status": "completed",
        "export_scanner": {"name": "runtime._exports", "inventory": export_rows},
        "canary_observations": (
            [
                {
                    "cuda_decisions": 1,
                    "device": "cuda:0",
                    "package": str(Path(manifest["script"]).parent),
                    "caller": manifest["script"],
                    "operation": "Tensor.argmax",
                    "shape": [1, 6],
                    "root": f"/content/snake-cli/{generation}",
                }
            ]
            if manifest.get("device") == "cuda"
            else []
        ),
    }
    receipt = {
        "status": "PASS",
        "stop_verified": True,
        "commit": manifest["commit"],
        "job_id": manifest["job_id"],
        "generation": generation,
        "terminal": terminal,
        "exports": export_rows,
        "transfers": transfers,
        "fixture_marker": "OFFLINE_FIXTURE",
    }
    proof = qualified_canary.evidence(manifest, receipt, output, repo)
    receipt["canary_evidence"] = proof
    manifest_path = output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    receipt["manifest_sha256"] = _sha256(manifest_path)
    receipt_path = output / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return receipt_path, receipt


def _synthetic_a100_compile_receipt(root: Path, *, commit: str, repo: Path) -> Path:
    """Build a complete, receipt-shaped compile prerequisite without an allocation."""
    compile_root = root / "native-compile"
    manifest = t4a_pack.native_compile_template(commit, compile_root)
    manifest_path = compile_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    hashes, specs = native_compile.inventory(repo, commit)
    exports = compile_root / "exports"
    binaries = []
    for spec in specs:
        name = str(spec["name"])
        relative = f"native-binaries/{name}.so"
        binary = exports / relative
        binary.parent.mkdir(parents=True, exist_ok=True)
        binary.write_bytes(f"OFFLINE_FIXTURE {name}\n".encode())
        binary_sha256 = _sha256(binary)
        binary.with_name(binary.name + ".sha256").write_text(f"{binary_sha256}  {binary.name}\n")
        binary.with_name(binary.name + ".complete").write_text("complete\n")
        binaries.append(
            {
                "name": name,
                "path": relative,
                "sha256": binary_sha256,
                "size": binary.stat().st_size,
            }
        )
    report = {
        "schema": native_compile.SCHEMA,
        "status": "PASS",
        "commit": commit,
        "source_sha256": hashes,
        "extensions": specs,
        "architectures": "8.0",
        "architecture_mode": "a100-8.0",
        "kernels_launched": False,
        "torch": "OFFLINE_FIXTURE",
        "cuda": "12.6",
        "nvcc": "OFFLINE_FIXTURE",
        "binaries": binaries,
    }
    report_path = exports / "native-compile.json"
    t4a_units.publish_json(report_path, report)
    export_rows = [
        {
            "path": path.relative_to(exports).as_posix(),
            "sha256": _sha256(path),
            "size": path.stat().st_size,
        }
        for path in sorted(exports.rglob("*"))
        if path.is_file() and not path.name.endswith((".sha256", ".complete"))
    ]
    receipt = {
        "status": "PASS",
        "stop_verified": True,
        "commit": commit,
        "job_id": manifest["job_id"],
        "manifest_sha256": _sha256(manifest_path),
        "terminal": {
            "returncode": 0,
            "job_id": manifest["job_id"],
            "exports": export_rows,
        },
        "exports": export_rows,
        "fixture_marker": "OFFLINE_FIXTURE_NO_ALLOCATION",
    }
    receipt_path = compile_root / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    assert native_compile.binding(receipt_path, commit, repo)["sha256"] == _sha256(receipt_path)
    return receipt_path


def test_attempt2_resume_keeps_verified_unit_and_passes_unchanged_cli_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One completed unit resumes through fresh canary/main admission offline."""
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    interrupted, exports = _interrupted_main(tmp_path, commit=commit)

    explicit_root = tmp_path / "verified-input"
    explicit_ledger = explicit_root / "verified-ledger.json"
    t4a_units.build_ledger(exports, explicit_ledger, commit=commit)
    with pytest.raises(ManifestRefused, match="completion"):
        t4a_pack.attempt_ledgers(
            tmp_path / "ledger-only-packet",
            commit,
            attempt=2,
            ledger=explicit_ledger,
        )
    packet = tmp_path / "packet"
    main_ledger, canary_ledger = t4a_pack.attempt_ledgers(
        packet,
        commit,
        attempt=2,
        ledger=explicit_ledger,
        completion_from=interrupted,
    )
    assert list(
        t4a_units.load_ledger(main_ledger, t4a_units.namespace(commit), t4a_units.allocation())
    ) == ["roster-frozen"]
    assert not t4a_units.load_ledger(
        canary_ledger,
        t4a_units.namespace(commit, canary=True),
        t4a_units.allocation(canary=True),
    )

    # Exercise packet assembly so the test pins the per-attempt archive names.
    fixture_input = tmp_path / "fixture-checkpoint.bin"
    fixture_input.write_bytes(b"OFFLINE_FIXTURE checkpoint\n")
    monkeypatch.setattr(
        t4a_pack,
        "checkpoint_inputs",
        lambda: [
            {
                "path": str(fixture_input),
                "dest": "fixture/checkpoint.bin",
                "sha256": _sha256(fixture_input),
            }
        ],
    )
    preflight = {"fixture_marker": "OFFLINE_FIXTURE"}
    probe = t4a_pack.manifest(
        commit=commit,
        output_dir=packet / "canary-output-attempt-02",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=True,
        preflight=preflight,
        attempt=2,
    )
    template = t4a_pack.manifest(
        commit=commit,
        output_dir=packet / "main-output-attempt-02",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        template=True,
        attempt=2,
    )
    assert (packet / "metadata-canary-attempt-02.tar").is_file()
    assert (packet / "metadata-main-attempt-02.tar").is_file()
    assert probe["t4a_envelope"]["attempt"] == template["t4a_envelope"]["attempt"] == 2
    assert probe["t4a_envelope"]["remaining_main_units"] == ["roster-mixed", "roster-scripted"]

    # Assemble the admitted main through the real non-template path. Only the
    # native receipt/reducer is fixture-bound; the main input identity check is
    # left intact and must match the same-attempt canary manifest.
    native_fixture = tmp_path / "native-compile.json"
    native_fixture.write_bytes(b"OFFLINE_FIXTURE native compile receipt\n")
    native_row = {
        "path": str(native_fixture.resolve()),
        "sha256": _sha256(native_fixture),
        "dest": "inputs/native/native-compile.json",
    }
    proof = {"path": str(native_fixture.resolve()), "sha256": _sha256(native_fixture)}

    def bind_fixture_native(value: dict, _proof: dict) -> None:
        value["inputs"].append(native_row)

    monkeypatch.setattr(t4a_pack, "_bind_native_inputs", bind_fixture_native)
    monkeypatch.setattr(
        t4a_pack,
        "canary_binding",
        lambda receipt, bound_commit, bound_preflight, *, attempt, remaining_units: {
            "path": str(receipt.resolve()),
            "sha256": _sha256(receipt),
            "native_cuda_compile": {"receipt": proof},
            "operational_amendment": t4a_pack._amendment(t4a_pack.COMPILE_REPAIR),
        },
    )
    probe["native_cuda_compile"] = {"receipt": proof}
    probe["inputs"].append(native_row)
    amendment = t4a_pack._amendment(t4a_pack.COMPILE_REPAIR)
    probe["inputs"].append(
        {
            "path": amendment["path"],
            "sha256": amendment["sha256"],
            "dest": "inputs/authority/t4a-operational-amendment.md",
        }
    )
    release = tmp_path / "canary-release"
    release.mkdir()
    canary_receipt = release / "receipt.json"
    canary_receipt.write_text(json.dumps({"fixture_marker": "OFFLINE_FIXTURE"}))
    (release / "manifest.json").write_text(json.dumps(probe, sort_keys=True) + "\n")
    resumed = t4a_pack.manifest(
        commit=commit,
        output_dir=packet / "main-output-attempt-02",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        canary_receipt=canary_receipt,
        attempt=2,
    )
    assert resumed["t4a_envelope"]["attempt"] == 2
    assert sorted((row["dest"], row["sha256"]) for row in resumed["inputs"]) == sorted(
        (row["dest"], row["sha256"]) for row in probe["inputs"]
    )

    # Use the real qualified canary checker on a broker-shaped fixture receipt.
    cli_probe, cli_main = _fixture_manifests(tmp_path, commit=commit, canary=probe, main=resumed)
    receipt_path, receipt = _offline_canary_receipt(tmp_path, manifest=cli_probe, repo=repo)
    cli_main["canary"] = {"receipt": {"path": str(receipt_path), "sha256": _sha256(receipt_path)}}
    qualified_canary.check(cli_probe, repo)
    qualified_canary.check(cli_main, repo)
    assert receipt["fixture_marker"] == "OFFLINE_FIXTURE"
    assert cli_probe["t4a_envelope"]["attempt"] == cli_main["t4a_envelope"]["attempt"] == 2
    assert qualified_canary.input_inventory(cli_probe) == qualified_canary.input_inventory(cli_main)

    # Dispatch wrapper validation is the unchanged admission path; no provider
    # client, allocation, tmux session, or Colab call is permitted here.
    ops = tmp_path / "ops"
    ops.mkdir()
    qualification = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json"
    )
    shutil.copy2(qualification, ops / qualification.name)
    probe_path = tmp_path / "attempt2-canary.json"
    main_path = tmp_path / "attempt2-main.json"
    probe_path.write_text(json.dumps(cli_probe, indent=2, sort_keys=True) + "\n")
    main_path.write_text(json.dumps(cli_main, indent=2, sort_keys=True) + "\n")

    def qualified_adapter(
        broker: Path, _ops: Path, deadline: float | None = None, ref: str = "HEAD"
    ) -> str:
        del deadline
        return ops_qualification.check(broker, ops / "colab-cli-qualified.json", ref=ref)

    def validate_adapter(
        broker: Path,
        cwd: Path,
        state: Path,
        manifest: Path,
        log: Path,
        deadline: float | None = None,
    ) -> None:
        del cwd, state, log, deadline
        qualified_supervisor.validate(json.loads(manifest.read_text()), repo=broker)

    monkeypatch.setattr(ops_dispatch, "qualified", qualified_adapter)
    monkeypatch.setattr(ops_dispatch, "validate", validate_adapter)
    monkeypatch.setenv("OPS_DIR", str(ops))
    for name, manifest_path in (
        ("t4a-resume-canary-offline", probe_path),
        ("t4a-resume-main-offline", main_path),
    ):
        args = argparse.Namespace(
            name=name,
            manifest=manifest_path,
            broker=repo,
            cwd=repo,
            state_dir=tmp_path / f"state-{name}",
            dispatch_txt=tmp_path / f"{name}.txt",
            dry_run=True,
            colab_control_nice0=False,
            small_bench_config=None,
        )
        try:
            result = ops_dispatch.dispatch(args)
        except (OSError, ValueError, TimeoutError, subprocess.SubprocessError) as exc:
            if not Path("/usr/bin/pmset").is_file():
                pytest.skip(f"Mac host power guard unavailable: {exc}")
            raise
        evidence = tmp_path / f"{name}.dry-run.txt"
        evidence.write_text(result + "\n")
        assert result.startswith("PASS_DRY_RUN ")


def test_attempt2_a100_chain_uses_real_compile_binding_and_shell_dry_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the A100 native admission branch for a resumed attempt offline.

    The compile and canary receipts are deliberately marked synthetic, but their
    byte-level receipt/binary bindings are validated by the production code. The
    final admission calls the public shell entrypoint, never the dispatcher
    module directly, so this cannot silently fall back to the prior CPU fixture.
    """
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    interrupted, exports = _interrupted_main(tmp_path, commit=commit)
    explicit_ledger = tmp_path / "verified-input" / "verified-ledger.json"
    t4a_units.build_ledger(exports, explicit_ledger, commit=commit)
    packet = tmp_path / "packet"
    main_ledger, canary_ledger = t4a_pack.attempt_ledgers(
        packet,
        commit,
        attempt=2,
        ledger=explicit_ledger,
        completion_from=interrupted,
    )

    fixture_input = tmp_path / "fixture-checkpoint.bin"
    fixture_input.write_bytes(b"OFFLINE_FIXTURE checkpoint\n")
    monkeypatch.setattr(
        t4a_pack,
        "checkpoint_inputs",
        lambda: [
            {
                "path": str(fixture_input),
                "dest": "fixture/checkpoint.bin",
                "sha256": _sha256(fixture_input),
            }
        ],
    )
    preflight = {"fixture_marker": "OFFLINE_FIXTURE"}
    canary = t4a_pack.manifest(
        commit=commit,
        output_dir=packet / "canary-output-attempt-02",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=True,
        preflight=preflight,
        attempt=2,
    )
    t4a_pack._write(packet / "t4a-canary-attempt-02.json", canary)
    compile_receipt = _synthetic_a100_compile_receipt(tmp_path, commit=commit, repo=repo)
    compiled_path = t4a_pack.bind_compile(
        packet, compile_receipt, t4a_pack.COMPILE_REPAIR, attempt=2
    )
    compiled = json.loads(compiled_path.read_text())
    assert compiled["requirements"] == {
        "gpu": "A100",
        "allowed_gpus": ["A100"],
        "min_vcpu": 12,
        "est_minutes": 10,
    }
    assert compiled["timeout_minutes"] == 20
    assert (
        native_compile.binding(compile_receipt, commit, repo)
        == compiled["native_cuda_compile"]["receipt"]
    )

    canary_receipt, _ = _offline_canary_receipt(
        tmp_path, manifest=compiled, repo=repo, t4a_canary=True
    )
    reduced = t4a_analyze.aggregate(canary_receipt.parent / "exports", commit, canary=True)
    assert reduced["verdict"] == "PASS_CANARY", reduced
    main = t4a_pack.manifest(
        commit=commit,
        output_dir=packet / "main-output-attempt-02",
        ledger=main_ledger,
        canary_ledger=canary_ledger,
        canary=False,
        preflight=preflight,
        canary_receipt=canary_receipt,
        attempt=2,
    )
    assert compiled["t4a_envelope"]["attempt"] == main["t4a_envelope"]["attempt"] == 2
    assert compiled["timeout_minutes"] == 20 and main["timeout_minutes"] == 210
    assert compiled["requirements"]["gpu"] == main["requirements"]["gpu"] == "A100"

    ops = tmp_path / "ops"
    ops.mkdir()
    qualification = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json"
    )
    shutil.copy2(qualification, ops / qualification.name)
    canary_path = tmp_path / "attempt2-a100-canary.json"
    main_path = tmp_path / "attempt2-a100-main.json"
    canary_path.write_text(json.dumps(compiled, indent=2, sort_keys=True) + "\n")
    main_path.write_text(json.dumps(main, indent=2, sort_keys=True) + "\n")
    environment = {
        **os.environ,
        "OPS_DIR": str(ops),
        "PATH": str(repo / "venv/bin") + os.pathsep + os.environ["PATH"],
    }
    dispatch = repo / "tools/ops/dispatch.sh"
    for name, manifest in (
        ("t4a-resume-a100-canary-offline", canary_path),
        ("t4a-resume-a100-main-offline", main_path),
    ):
        result = subprocess.run(
            [
                "bash",
                str(dispatch),
                name,
                str(manifest),
                "--broker",
                str(repo),
                "--cwd",
                str(repo),
                "--state-dir",
                str(tmp_path / f"state-{name}"),
                "--dispatch-txt",
                str(tmp_path / f"{name}.txt"),
                "--dry-run",
            ],
            cwd=repo,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )
        evidence = tmp_path / f"{name}.dry-run.txt"
        evidence.write_text(result.stdout)
        assert result.returncode == 0, result.stderr
        assert result.stdout.startswith(f"PASS_DRY_RUN {name} allocations=0")


def test_attempt2_rejects_attempt1_canary_receipt(tmp_path: Path) -> None:
    """A completion main cannot bind an original-attempt canary receipt."""
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "commit": "a" * 40,
                "script": t4a_pack.SCRIPT,
                "mac_preflight": {},
                "t4a_envelope": {"attempt": 1},
            },
            sort_keys=True,
        )
    )
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "status": "PASS",
                "stop_verified": True,
                "commit": "a" * 40,
                "manifest_sha256": _sha256(manifest),
                "fixture_marker": "OFFLINE_FIXTURE",
            }
        )
    )
    with pytest.raises(ManifestRefused, match="SAME attempt"):
        t4a_pack.canary_binding(
            receipt,
            "a" * 40,
            {},
            attempt=2,
            remaining_units=("roster-mixed",),
        )


def test_attempt2_forecast_scales_only_remaining_rosters() -> None:
    """A verified frozen roster is excluded from the completion forecast."""
    units = []
    for _index in range(3):
        units.append(
            {
                "horizon": 2,
                "records": [
                    {
                        "build_seconds": 0.1,
                        "tick_seconds": [0.01, 0.01],
                        "comparison_seconds": 0.0,
                    }
                    for _record in range(18)
                ],
                "event_seconds": 0.1,
                "elapsed_seconds": 100.0,
            }
        )
    bench = {"units": units, "native_elapsed_seconds": 1.0}
    full = t4a_pack._main_projection(bench, {}, remaining_units=t4a_units.allocation())
    remaining = ("roster-mixed", "roster-scripted")
    resumed = t4a_pack._main_projection(bench, {}, remaining_units=remaining)
    assert resumed["remaining_units"] == list(remaining)
    assert resumed["projected_main_seconds"] < full["projected_main_seconds"]
