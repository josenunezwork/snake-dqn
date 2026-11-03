"""Tests for research.runtime_probe manifest builders, input staging, and probe output schema."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from research.colab_cli import canary, schema, supervisor
from research.runtime_probe import (
    DEFAULT_HORIZON,
    DEFAULT_INTERVAL,
    DEFAULT_SEED,
    PROBE_SCHEMA,
    RUNTIME_SPECS,
    TARGETS,
    build_manifest,
    check_stack_support,
    collect_system_info,
    normalize_gpu_target,
    prepare_probe_package,
    validate_probe_output,
)
from research.runtime_probe.bench import (
    CONTRACT_SCHEMA,
    UNIT_ID,
    _check_resume_contract,
)
from research.runtime_probe.bench import main as bench_main
from research.runtime_probe.bench import (
    read_cpu_model_name,
    read_total_ram_bytes,
)
from research.runtime_probe.prepare import (
    SCRIPT_PATH,
    _git_head,
    empty_ledger,
    stage_probe_metadata,
)
from tools.ops import dispatch
from tools.ops.package_spec import publish_immutable, publish_json, verify_immutable


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CPU", "cpu"),
        ("cpu", "cpu"),
        ("T4", "T4"),
        ("t4", "T4"),
        ("L4", "L4"),
        ("l4", "L4"),
        ("A100", "A100"),
        ("a100", "A100"),
    ],
)
def test_normalize_gpu_target(raw: str, expected: str) -> None:
    assert normalize_gpu_target(raw) == expected


def test_normalize_gpu_target_rejects_invalid() -> None:
    with pytest.raises(ValueError, match="unsupported gpu target"):
        normalize_gpu_target("V100")


@pytest.mark.parametrize("target", TARGETS)
def test_build_manifest_all_targets_validate_schema_and_policies(
    tmp_path: Path, target: str
) -> None:
    meta_file = tmp_path / "input-metadata.tar.gz"
    pub = publish_immutable(meta_file, b"synthetic-metadata-bytes")
    metadata_input = {
        "path": str(meta_file),
        "dest": "input-metadata.tar.gz",
        "sha256": pub["sha256"],
        "size_bytes": pub["size"],
    }
    output_dir = tmp_path / "outputs" / target.lower()
    manifest = build_manifest(
        target,
        commit="e9bcec242bbb6357c14520f35efd051e48ee24d8",
        metadata_input=metadata_input,
        output_dir=output_dir,
    )

    schema.validate_manifest(manifest)
    dispatch.require_resume_policy(manifest)

    hw = normalize_gpu_target(target)
    assert manifest["requirements"]["gpu"] == hw
    assert manifest["requirements"]["allowed_gpus"] == [hw]
    assert manifest["device"] == "cpu"
    assert manifest["env"]["SNAKE_DQN_DEVICE"] == "cpu"
    assert manifest["script"] == SCRIPT_PATH
    assert manifest["required_exports"] == ["probe.json"]
    assert manifest["timeout_minutes"] <= 15.0
    assert manifest["timeout_minutes"] >= 2.0 * manifest["requirements"]["est_minutes"]
    assert manifest["budget"]["unit_cap"] <= 0.5
    assert manifest["budget"]["rate_units_per_hour"] == RUNTIME_SPECS[hw]["rate_units_per_hour"]
    script_text = (Path.cwd() / SCRIPT_PATH).read_text(encoding="utf-8")
    with patch("research.colab_cli.canary._entrypoint", return_value=script_text):
        assert canary.job_device(manifest, Path.cwd()) == "cpu"
        assert canary.check(manifest, Path.cwd()) is None


def test_prepare_probe_package_and_stage_metadata_roundtrip(tmp_path: Path) -> None:
    pkg_dir = tmp_path / "pkg"
    written = prepare_probe_package(
        pkg_dir,
        commit=_git_head(),
        gpus=TARGETS,
        release_evidence_bytes=b"synthetic-release-evidence",
        verify_pins=False,
    )
    assert set(written.keys()) == {"cpu", "T4", "L4", "A100", "G4"}

    for hw, manifest_path in written.items():
        verified = verify_immutable(manifest_path)
        assert verified["size"] > 0
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        schema.validate_manifest(manifest)
        dispatch.require_resume_policy(manifest)
        validated = supervisor.validate(manifest)
        assert validated["requirements"]["gpu"] == hw

        vm_input_dir = tmp_path / f"vm-{hw.lower()}"
        vm_input_dir.mkdir()
        src_archive = Path(manifest["inputs"][0]["path"])
        (vm_input_dir / "input-metadata.tar.gz").write_bytes(src_archive.read_bytes())

        stage_probe_metadata(
            vm_input_dir,
            manifest["env"]["RUNTIME_PROBE_METADATA_SHA256"],
            verify_pins=False,
            materialize_checkpoints=False,
        )
        for name in (
            "probe-contract.json",
            "verified-ledger.json",
            "release-evidence.tar.gz",
            "public-inputs.json",
        ):
            verify_immutable(vm_input_dir / name)


def test_system_info_and_output_schema_validation(tmp_path: Path) -> None:
    fake_cpuinfo = tmp_path / "cpuinfo"
    fake_cpuinfo.write_text(
        "processor\t: 0\nvendor_id\t: GenuineIntel\nmodel name\t: Intel(R) Xeon(R) CPU @ 2.20GHz\n",
        encoding="utf-8",
    )
    assert read_cpu_model_name(fake_cpuinfo) == "Intel(R) Xeon(R) CPU @ 2.20GHz"

    fake_meminfo = tmp_path / "meminfo"
    fake_meminfo.write_text("MemTotal:       13191680 kB\nMemFree:         1000000 kB\n", "utf-8")
    assert read_total_ram_bytes(fake_meminfo) == 13191680 * 1024

    info = collect_system_info()
    assert info["cpu_count"] >= 1
    assert info["sched_getaffinity_count"] >= 1
    assert isinstance(info["cpu_model_name"], str) and info["cpu_model_name"]
    assert info["total_ram_bytes"] > 0
    assert info["total_ram_gib"] > 0
    assert isinstance(info["gpu_name"], str) and info["gpu_name"]
    assert isinstance(info["python_version"], str) and info["python_version"]
    assert isinstance(info["torch_version"], str) and info["torch_version"]

    valid_payload: dict[str, Any] = {
        "schema": PROBE_SCHEMA,
        "mode": "local",
        **info,
        "benchmark": {
            "workload": "watch6_l1_confirm_bare_plus_exact_stop_l1",
            "seed": DEFAULT_SEED,
            "horizon": DEFAULT_HORIZON,
            "interval": DEFAULT_INTERVAL,
            "arms": ["bare", "l1"],
            "decision_points_per_slice": 3,
            "warmup_seconds": 1.25,
            "single_worker": {
                "workers": 1,
                "wall_seconds": 4.0,
                "slices_per_hour": 900.0,
                "decisions_per_second": 0.75,
                "seconds_per_decision": 1.333333,
            },
            "parallel_cpu_count": {
                "workers": info["cpu_count"],
                "init_and_warmup_seconds": 2.0,
                "wall_seconds": 5.0,
                "worker_seconds_min": 4.5,
                "worker_seconds_mean": 4.75,
                "worker_seconds_max": 5.0,
                "aggregate_slices_per_hour": round(info["cpu_count"] * 720.0, 4),
                "aggregate_decisions_per_second": round(info["cpu_count"] * 0.6, 6),
                "speedup_vs_single_worker": round(info["cpu_count"] * 0.8, 4),
                "per_worker_efficiency": 0.8,
            },
            "slice_verification": {
                "seed": DEFAULT_SEED,
                "horizon": DEFAULT_HORIZON,
                "interval": DEFAULT_INTERVAL,
                "arms": ["bare", "l1"],
                "bare_alive_frames": 15,
                "bare_terminal_reason": None,
                "l1_alive_frames": 15,
                "l1_terminal_reason": None,
                "l1_decision_count": 3,
                "l1_intervention_count": 0,
                "l1_fallback_count": 0,
                "slice_sha256": "a" * 64,
            },
        },
    }
    validate_probe_output(valid_payload)

    bad_payload = copy.deepcopy(valid_payload)
    bad_payload["benchmark"]["single_worker"]["wall_seconds"] = 0.0
    with pytest.raises(ValueError, match="wall_seconds"):
        validate_probe_output(bad_payload)


def test_resume_contract_and_bench_cli_with_mocked_parallel_run(tmp_path: Path) -> None:
    contract_path = tmp_path / "probe-contract.json"
    ledger_path = tmp_path / "verified-ledger.json"
    out_path = tmp_path / "probe.json"

    publish_json(
        contract_path,
        {
            "schema": CONTRACT_SCHEMA,
            "namespace": "ns-test",
            "allocation": "alloc-test",
            "target_gpu": "cpu",
            "unit_id": UNIT_ID,
            "seed": DEFAULT_SEED,
            "horizon": 5,
            "interval": 5,
        },
    )
    publish_json(ledger_path, empty_ledger("ns-test", "alloc-test"))

    contract, skipped = _check_resume_contract(contract_path, ledger_path)
    assert contract["target_gpu"] == "cpu"
    assert skipped is False

    fake_slice = {
        "seed": DEFAULT_SEED,
        "horizon": 5,
        "interval": 5,
        "arms": ["bare", "l1"],
        "bare_alive_frames": 5,
        "bare_terminal_reason": None,
        "l1_alive_frames": 5,
        "l1_terminal_reason": None,
        "l1_decision_count": 1,
        "l1_intervention_count": 0,
        "l1_fallback_count": 0,
        "slice_sha256": "b" * 64,
    }
    fake_workers = [
        {
            "worker_id": 0,
            "ok": True,
            "warmup_seconds": 0.001,
            "wall_seconds": 0.001,
            "slice": fake_slice,
        },
        {
            "worker_id": 1,
            "ok": True,
            "warmup_seconds": 0.001,
            "wall_seconds": 0.001,
            "slice": fake_slice,
        },
    ]

    with (
        patch("research.runtime_probe.bench.run_watch6_slice", return_value=fake_slice),
        patch(
            "research.runtime_probe.bench._run_parallel_workers",
            return_value=(0.002, 0.001, fake_workers),
        ),
        patch(
            "research.runtime_probe.bench.time.perf_counter",
            side_effect=[0.0, 0.001, 1.0, 1.001],
        ),
    ):
        rc = bench_main(
            [
                "--local",
                "--contract",
                str(contract_path),
                "--ledger",
                str(ledger_path),
                "--out",
                str(out_path),
                "--workers",
                "2",
            ]
        )
    assert rc == 0
    verify_immutable(out_path)
    written = json.loads(out_path.read_text(encoding="utf-8"))
    validate_probe_output(written)
    assert written["benchmark"]["parallel_cpu_count"]["workers"] == 2
    assert written["benchmark"]["parallel_cpu_count"]["speedup_vs_single_worker"] == 2.0

    # Verify skip behavior when unit is already recorded in verified-ledger.json.
    verified_ledger = tmp_path / "verified-ledger-done.json"
    done_payload = empty_ledger("ns-test", "alloc-test")
    done_payload["verified_units"] = [UNIT_ID]
    publish_json(verified_ledger, done_payload)
    _, skipped_done = _check_resume_contract(contract_path, verified_ledger)
    assert skipped_done is True

    # Verify refusal on foreign namespace.
    foreign_ledger = tmp_path / "verified-ledger-foreign.json"
    publish_json(foreign_ledger, empty_ledger("other-ns", "alloc-test"))
    with pytest.raises(ValueError, match="foreign namespace"):
        _check_resume_contract(contract_path, foreign_ledger)


def test_stack_support_audit_matches_supervisor_and_dispatch(tmp_path: Path) -> None:
    report = check_stack_support()
    assert set(report.keys()) == {"cpu", "T4", "L4", "A100"}
    assert report["cpu"]["dispatch_live_session_supported"] is True
    assert report["T4"]["dispatch_live_session_supported"] is False

    log_path = tmp_path / "job.log"
    log_path.write_text("EXIT=1\n", encoding="utf-8")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "receipt.json").write_text(
        json.dumps(
            {
                "status": "FAILED",
                "session": "cli-test",
                "error": (
                    "Refused(\"colab new failed (1): [colab] Creating session 'cli-test'...\\n"
                    "ColabRequestError: Failed to issue request POST "
                    "https://colab.research.google.com/tun/m/assign"
                    '?nbh=abc&variant=DEFAULT: 503 Service Unavailable\\n")'
                ),
                "cli_calls": ["usage", "sessions", "new", "sessions"],
            }
        ),
        encoding="utf-8",
    )
    assert not dispatch.allocation_unavailable(out_dir, log_path, "cpu")
    assert hasattr(supervisor, "validate")
