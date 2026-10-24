"""Independent Codex-V checks for the source-bound O1 acceptance package."""

from __future__ import annotations

import copy
import json
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from research.colab_cli import retention
from research.colab_cli import retention_acceptance as acceptance
from research.colab_cli import source


def test_legacy_boolean_evidence_can_never_qualify() -> None:
    evidence = {
        "transport_regression": "PASS",
        "source_inventory_match": True,
        "sole_a100_boundary": True,
        "faults": {name: "PASS" for name in acceptance.REQUIRED_FAULTS},
        "native_packets": ["PASS", "PASS"],
        "metrics": {
            "second_setup_reduction_percent": 100,
            "allocated_wall_saving_percent": 100,
        },
    }
    result = acceptance.reduce_evidence(evidence)
    assert result["status"] == "INCOMPLETE"
    assert result["adoption"] == "REFUSED_UNTIL_PASS"


def test_required_fault_inventory_is_complete_and_stable() -> None:
    assert acceptance.REQUIRED_FAULTS == (
        "mac_disappearance_lease_expiry",
        "stale_generation",
        "child_failure",
        "cap_during_handoff",
        "accounting_refusal",
        "export_failure",
        "interrupted_second_job_launch",
        "recovery",
    )


def test_prepare_refuses_dirty_or_uncommitted_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fake_git(_repo: Path, *args: str) -> bytes:
        if args[:2] == ("rev-parse", "HEAD"):
            return b"a" * 40
        if args[:3] == ("status", "--porcelain", "--untracked-files=all"):
            return b"dirty"
        return b""

    monkeypatch.setattr(acceptance, "git", fake_git)
    monkeypatch.setattr(acceptance, "verify_inventory", lambda *_args: None)
    with pytest.raises(ValueError, match="clean committed candidate"):
        acceptance.prepare(Path.cwd(), tmp_path / "o1")
    assert not (tmp_path / "o1" / "acceptance-plan.json").exists()


def test_verify_inventory_refuses_changed_executable_inventory() -> None:
    inventory = acceptance.executable_inventory(Path.cwd())
    changed = dict(inventory)
    changed["research/colab_cli/retention_acceptance.py"] = "0" * 64
    with pytest.raises(ValueError, match="inventory changed"):
        acceptance.verify_inventory(Path.cwd(), "0" * 40, changed)


def test_reduce_plan_returns_incomplete_when_fixed_evidence_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan_path = tmp_path / "acceptance-plan.json"
    plan = {
        "schema": acceptance.SCHEMA,
        "repo": str(Path.cwd()),
        "commit": "0" * 40,
        "source_inventory": {},
        "regression_plan": {"path": str(tmp_path / "missing-regression.json"), "sha256": "0" * 64},
    }
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setattr(acceptance, "verify_inventory", lambda *_args: None)

    result = acceptance.reduce_plan(plan_path)

    assert result["schema"] == "snake-dqn-a100-retention-acceptance-result/v1"
    assert result["status"] == "INCOMPLETE"
    assert result["native_a100_two_packet"] is False
    assert result["no_registry_update"] is True
    assert result["reasons"]


def test_reduce_plan_success_fixture_derives_bars_from_bound_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A complete fixed fixture produces PASS from receipts and probe evidence."""
    run = tmp_path
    repo = Path.cwd()
    commit = "a" * 40
    probe_sha = "p" * 64
    ledger_sha = "l" * 64
    cache_pins = {
        "python_abi": "cpython-312",
        "torch_cuda": "2.9.1|12.6",
        "dependency_lock_sha256": "d" * 64,
        "source_sha256": "s" * 64,
        "compiler": "nvcc",
        "compiler_flags": '{"CXXFLAGS":"","TORCH_CUDA_ARCH_LIST":"8.0"}',
        "sm": "8.0",
    }
    cache_key = "k" * 64

    def write(path: Path, value: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True))

    regression_manifest = run / "regression" / "job" / "manifest.json"
    write(regression_manifest, {"profile": "final-only"})
    regression_plan = run / "regression-plan.json"
    write(regression_plan, {"jobs": [{"manifest": str(regression_manifest)}]})
    write(regression_manifest.parent.parent / "acceptance-result.json", {"status": "PASS"})

    packet_rows: list[dict[str, str]] = []
    packet_manifests: dict[str, dict[str, object]] = {}
    packets = ["native-canary", "cold-1", "cold-2", "retained-1", "retained-2"]
    packets += ["fault-" + name.replace("_", "-") for name in acceptance.REQUIRED_FAULTS]
    for packet in packets:
        manifest = run / "templates" / f"{packet}.json"
        output = run / "outputs" / packet
        value = {
            "packet": packet,
            "output_dir": str(output),
            "inputs": [
                {"dest": "probe-input.json", "sha256": probe_sha},
                {"dest": "verified-ledger.json", "sha256": ledger_sha},
            ],
        }
        write(manifest, value)
        write(output / "manifest.json", value)
        packet_rows.append(acceptance.binding(manifest) | {"packet": packet})
        packet_manifests[packet] = value

    plan_path = run / "acceptance-plan.json"
    write(
        plan_path,
        {
            "schema": acceptance.SCHEMA,
            "repo": str(repo),
            "commit": commit,
            "source_inventory": {},
            "regression_plan": acceptance.binding(regression_plan),
            "jobs": packet_rows,
            "chain": {"retention_minutes": 40, "retention_units": 40 * 10 / 60},
        },
    )
    canary_output = Path(packet_manifests["native-canary"]["output_dir"])
    canary_probe = {
        "device": "cuda",
        "root": "/content/snake-cli/g-canary",
        "bootstrap": {
            "pins": cache_pins,
            "cache_key": cache_key,
            "cache_status": "MISS",
            "setup_seconds": 1,
        },
    }
    write(canary_output / "exports/probe.json", canary_probe)
    canary_receipt = {
        "generation": "g-canary",
        "endpoint": "canary-endpoint",
        "canary_evidence": {"status": "PASS"},
    }

    receipts = {
        "cold-1": {
            "generation": "g-cold-1",
            "endpoint": "cold-1",
            "assignment_allocated_elapsed_seconds": 60,
        },
        "cold-2": {
            "generation": "g-cold-2",
            "endpoint": "cold-2",
            "assignment_allocated_elapsed_seconds": 60,
        },
        "retained-1": {
            "generation": "g-retained-1",
            "endpoint": "retained",
            "assignment_allocated_elapsed_seconds": 40,
        },
        "retained-2": {
            "generation": "g-retained-2",
            "endpoint": "retained",
            "assignment_allocated_elapsed_seconds": 40,
        },
    }

    def native_record(_plan: dict, packet: str, _run: Path, _pins: dict) -> tuple[dict, Path, dict]:
        output = Path(packet_manifests[packet]["output_dir"])
        if packet.startswith("fault-"):
            fault = packet[6:].replace("-", "_")
            receipt = {
                "status": "INCOMPLETE",
                "generation": "fault-generation",
                "endpoint": packet,
            }
            if fault == "child_failure":
                receipt["terminal"] = {"returncode": 23}
            elif fault in {"mac_disappearance_lease_expiry", "stale_generation"}:
                write(
                    output / "fault-observation.json",
                    {
                        "fault": fault,
                        "expiry_monotonic": 1,
                        "observed_monotonic": 2,
                        "before_fallback_cleanup": True,
                        "readings": [{"sessions": []}],
                        "endpoint": packet,
                    },
                )
            elif fault == "accounting_refusal":
                write(
                    output / "fault-observation.json",
                    {
                        "fault": fault,
                        "actual_usage": {"rate": 10},
                        "injected_usage": {"rate": 110},
                        "error": "rate changed",
                    },
                )
            elif fault == "export_failure":
                write(
                    output / "fault-observation.json",
                    {
                        "fault": fault,
                        "observation": {
                            "failure-and-corrupt-export": {"corrupt_export_rejected": True}
                        },
                    },
                )
            return packet_manifests.get(packet, {}), output, receipt
        receipt = receipts[packet]
        setup = 5 if packet.startswith("cold") else 2
        write(
            output / "exports/probe.json",
            {
                "device": "cuda",
                "root": "/content/snake-cli/" + receipt["generation"],
                "bootstrap": {
                    "pins": cache_pins,
                    "cache_key": cache_key,
                    "cache_status": "HIT" if packet == "retained-2" else "MISS",
                    "setup_seconds": setup,
                },
                "input_sha256": probe_sha,
                "ledger_sha256": ledger_sha,
                "compile_load_seconds": setup,
                "extension_cache": [
                    {
                        "status": (
                            "HIT_VERIFIED_IMPORT" if packet == "retained-2" else "MISS_COMPILED"
                        ),
                        "key": "a" * 64,
                        "sha256": "b" * 64,
                        "module_name": "kernel",
                        "size": 100,
                    }
                ],
            },
        )
        return packet_manifests[packet], output, receipt

    monkeypatch.setattr(acceptance, "verify_inventory", lambda *_args: None)
    monkeypatch.setattr(acceptance, "_offline_faults", lambda *_args: {})
    monkeypatch.setattr(acceptance, "_receipt", lambda *_args, **_kwargs: canary_receipt)
    monkeypatch.setattr(acceptance, "_native_record", native_record)
    monkeypatch.setattr(acceptance, "binding", lambda path: {"path": str(path), "sha256": "e" * 64})
    monkeypatch.setattr("research.colab_cli.retention.source_inventory", lambda _repo: {})
    monkeypatch.setattr("research.colab_cli.admission.require_final_only", lambda *_args: {})
    monkeypatch.setattr(
        "research.colab_cli.canary.evidence",
        lambda *_args: canary_receipt["canary_evidence"],
    )
    write(run / "admitted/retained-2.json", packet_manifests["retained-2"])
    write(
        run / "outputs/retained-2/session-release.json",
        {
            "assignment_allocated_elapsed_seconds": 60,
            "chain_error": None,
            "continuous_accounting": {"rate": 10},
        },
    )
    write(run / "sessions-after.json", {"sessions": []})

    result = acceptance.reduce_plan(plan_path)

    assert result["status"] == "PASS", result["reasons"]
    assert result["native_a100_two_packet"] is True
    assert result["metrics"]["second_setup_reduction_percent"] >= 50
    assert result["metrics"]["allocated_wall_saving_percent"] >= 25
    assert result["no_registry_update"] is True


def test_reduce_plan_refuses_mutated_atomic_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    regression = tmp_path / "regression.json"
    regression.write_text(json.dumps({"jobs": []}))
    plan = tmp_path / "acceptance-plan.json"
    plan.write_text(
        json.dumps(
            {
                "schema": acceptance.SCHEMA,
                "repo": str(Path.cwd()),
                "commit": "a" * 40,
                "source_inventory": {},
                "regression_plan": acceptance.binding(regression),
            }
        )
    )
    regression.write_text(json.dumps({"jobs": [], "mutated": True}))
    monkeypatch.setattr(acceptance, "verify_inventory", lambda *_args: None)

    result = acceptance.reduce_plan(plan)

    assert result["status"] == "INCOMPLETE"
    assert any("SHA-bound" in reason or "changed" in reason for reason in result["reasons"])


def test_boundary_requires_hash_bound_current_admission(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan": "fixture"}))
    admission_path = tmp_path / "admission.json"
    admission_path.write_text(
        json.dumps(
            {
                "schema": "o1-boundary-admission/v1",
                "plan": acceptance.binding(plan_path),
                "original_sealed_admission": {
                    "path": str(tmp_path / "sealed.json"),
                    "sha256": "0" * 64,
                },
            }
        )
    )
    with pytest.raises((ValueError, OSError), match="absent|changed|boundary"):
        acceptance.check_boundary(plan_path, {}, admission_path)


def _write_boundary_fixture(
    tmp_path: Path,
    *,
    sealed_t0: Any,
    boundary_t0: Any | None = None,
    pointer: str | None = "/admission/t0",
) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({"plan": "fixture"}))
    sealed_path = tmp_path / "sealed-admission.json"
    sealed_path.write_text(json.dumps({"admission": {"t0": sealed_t0}}))
    value = {
        "schema": "o1-boundary-admission/v1",
        "plan": acceptance.binding(plan_path),
        "original_sealed_admission": acceptance.binding(sealed_path),
        "t0_epoch": sealed_t0 if boundary_t0 is None else boundary_t0,
        "safe_a100_boundary": True,
        "required_work_not_displaced": True,
        "unencumbered_units": 48,
        "unencumbered_runtime_minutes": 288,
        "prior_attempt_receipts": [],
        "other_assignments": {},
    }
    if pointer is not None:
        value["t0_json_pointer"] = pointer
    boundary_path = tmp_path / "boundary-admission.json"
    boundary_path.write_text(json.dumps(value))
    return plan_path, boundary_path


def test_boundary_accepts_equal_t0_from_hash_bound_sealed_admission(tmp_path: Path) -> None:
    t0 = time.time() - 60
    plan_path, boundary_path = _write_boundary_fixture(tmp_path, sealed_t0=t0)

    value = acceptance.check_boundary(plan_path, {}, boundary_path)

    assert value["t0_epoch"] == t0


def test_boundary_refuses_invented_t0_even_when_sealed_file_is_bound(tmp_path: Path) -> None:
    t0 = time.time() - 60
    plan_path, boundary_path = _write_boundary_fixture(tmp_path, sealed_t0=t0, boundary_t0=t0 + 1)

    with pytest.raises(ValueError, match="t0|sealed|match"):
        acceptance.check_boundary(plan_path, {}, boundary_path)


def test_boundary_refuses_missing_t0_json_pointer(tmp_path: Path) -> None:
    t0 = time.time() - 60
    plan_path, boundary_path = _write_boundary_fixture(tmp_path, sealed_t0=t0, pointer=None)

    with pytest.raises(ValueError, match="t0|pointer|sealed"):
        acceptance.check_boundary(plan_path, {}, boundary_path)


def test_boundary_refuses_future_sealed_t0(tmp_path: Path) -> None:
    future = time.time() + 3600
    plan_path, boundary_path = _write_boundary_fixture(tmp_path, sealed_t0=future)

    with pytest.raises(ValueError, match="future|t0|deadline"):
        acceptance.check_boundary(plan_path, {}, boundary_path)


def test_boundary_accepts_timezone_aware_iso8601_t0_with_offset_or_z(
    tmp_path: Path,
) -> None:
    aware = datetime.fromtimestamp(time.time() - 60, timezone(timedelta(hours=-7)))
    epoch = aware.timestamp()
    offset = aware.isoformat()
    utc_z = aware.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    for index, sealed_t0 in enumerate((offset, utc_z)):
        plan_path, boundary_path = _write_boundary_fixture(
            tmp_path / f"aware-{index}", sealed_t0=sealed_t0, boundary_t0=epoch
        )
        value = acceptance.check_boundary(plan_path, {}, boundary_path)
        assert value["t0_epoch"] == epoch


def test_boundary_refuses_timezone_naive_iso8601_t0(tmp_path: Path) -> None:
    epoch = time.time() - 60
    naive = datetime.fromtimestamp(epoch, timezone.utc).replace(tzinfo=None)
    naive_text = naive.isoformat()
    plan_path, boundary_path = _write_boundary_fixture(
        tmp_path, sealed_t0=naive_text, boundary_t0=naive.timestamp()
    )

    with pytest.raises(ValueError, match="timezone|sealed T0|malformed"):
        acceptance.check_boundary(plan_path, {}, boundary_path)


def test_boundary_refuses_near_original_deadline_without_full_effective_envelope(
    tmp_path: Path,
) -> None:
    # The original utility deadline remains in the future, but fewer than the
    # full 4.8-hour acceptance envelope remains available.
    t0 = time.time() - (120 * 3600 - 60)
    plan_path, boundary_path = _write_boundary_fixture(tmp_path, sealed_t0=t0)

    with pytest.raises(ValueError, match="effective envelope|deadline"):
        acceptance.check_boundary(plan_path, {}, boundary_path)


def test_series_remaining_refuses_next_job_past_absolute_deadline(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="first-cap|deadline|next allocation"):
        acceptance._series_remaining(
            tmp_path,
            time.monotonic(),
            2,
            time.time() + 60,
        )


def test_native_manifest_binds_a100_and_per_manifest_cap(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "probe-input.json").write_text("{}\n")
    (inputs / "verified-ledger.json").write_text("{}\n")
    manifest = acceptance._native_manifest(Path.cwd(), tmp_path, "a" * 40, "cold-1")
    assert manifest["requirements"]["gpu"] == "A100"
    assert manifest["requirements"]["allowed_gpus"] == ["A100"]
    assert manifest["timeout_minutes"] <= 60
    assert manifest["resume_exemption"]["kind"] == "acceptance"
    assert manifest["budget"]["rate_units_per_hour"] == 10.0


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("python_abi", "cpython-311"),
        ("torch_cuda", "2.8.0|12.6"),
        ("dependency_lock_sha256", "a" * 64),
        ("installed_distributions_sha256", "d" * 64),
        ("source_sha256", "b" * 64),
        ("compiler", "f" * 64),
        (
            "compiler_flags",
            '{"CXXFLAGS":"-O3","TORCH_CUDA_ARCH_LIST":"8.0"}',
        ),
        ("sm", "8.6"),
    ],
)
def test_runtime_cache_refuses_each_stale_canary_cache_pin(
    tmp_path: Path, field: str, replacement: str
) -> None:
    """A valid canary receipt cannot authorize reuse after any cache-pin drift."""
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "requirements.txt").write_text("example==1.0\n")
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Retention Test"], cwd=repo, check=True)
    subprocess.run(["git", "add", "requirements.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "freeze retention fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    source_sha = retention.pinned_tree_sha256(repo, commit)
    dependency_sha = acceptance.sha256_file(repo / "requirements.txt")
    canary = repo / "canary-receipt.json"
    cache = {
        "python_abi": "cpython-312",
        "torch_cuda": "2.9.1|12.6",
        "dependency_lock_sha256": dependency_sha,
        "installed_distributions_sha256": "c" * 64,
        "source_sha256": source_sha,
        "compiler": "e" * 64,
        "compiler_flags": '{"CXXFLAGS":"","TORCH_CUDA_ARCH_LIST":"8.0"}',
        "sm": "8.0",
    }
    canary.write_text(json.dumps({"status": "PASS", "cache_evidence": {"pins": cache}}))
    manifest = {
        "commit": commit,
        "requirements": {"gpu": "A100"},
        "canary": {"receipt": acceptance.binding(canary)},
        "retention_chain": {
            "enabled": True,
            "shared": {
                "chain_id": "cache-fixture",
                "retention_minutes": 40,
                "retention_units": 40 * 10 / 60,
            },
            "cache": cache,
            "qualification": {},
        },
    }
    pins, key = retention.runtime_cache(manifest, repo)
    assert pins == cache
    assert len(key) == 64

    stale = copy.deepcopy(manifest)
    stale["retention_chain"]["cache"][field] = replacement
    with pytest.raises(retention.RetentionError):
        retention.runtime_cache(stale, repo)


def _source_identity_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, object], str]:
    """Build and install a real sparse source package for identity checks."""
    repo = tmp_path / "source-repo"
    repo.mkdir()
    files = {
        "research/colab_cli/source.py": "# source fixture\n",
        "research/colab_cli/runtime.py": "# runtime fixture\n",
        "research/colab_cli/bootstrap.sh": "#!/bin/sh\n",
        "research/colab_cli/release.py": "# release fixture\n",
        "research/study/run.sh": "#!/bin/sh\necho run\n",
        "requirements.txt": "pytest\n",
        "pyproject.toml": "[project]\nname='identity-fixture'\n",
        "src/pkg.py": "VALUE = 1\n",
    }
    for name, text in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    (repo / "research/study/run.sh").chmod(0o755)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "identity@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "Identity Fixture"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "identity fixture"], cwd=repo, check=True)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
    manifest = {
        "commit": commit,
        "script": "research/study/run.sh",
        "env": {},
        "budget": {"rate_units_per_hour": 1.0, "unit_cap": 1.0, "reserve_units": 0.0},
    }
    delivery = tmp_path / "source-delivery"
    plan = source.prepare(repo, manifest, delivery)
    digest = retention.pinned_tree_sha256(repo, commit)

    runtime = tmp_path / "fallback-runtime"
    runtime.mkdir()
    (runtime / source.ARCHIVE_NAME).write_bytes((delivery / source.ARCHIVE_NAME).read_bytes())
    installed = source.install_fallback(runtime, plan)
    assert installed["inventory_verified"] is True
    identity = {
        "commit": commit,
        "tree": plan["tree"],
        "tree_sha256": digest,
    }
    spec = {
        "generation": runtime.name,
        "root": str(runtime),
        "retention_source_identity": identity,
    }
    (runtime / "spec.json").write_text(json.dumps(spec, sort_keys=True))
    (runtime / "retention-source-delivery.json").write_text(
        json.dumps({**installed, "generation": runtime.name, "tree_sha256": digest}, sort_keys=True)
    )
    return runtime, repo, identity, digest


def test_runtime_source_identity_attests_real_sparse_fallback(tmp_path: Path) -> None:
    runtime, _repo, identity, digest = _source_identity_fixture(tmp_path)

    actual, origin = retention.runtime_source_identity(runtime, runtime / "repo")

    assert actual == digest == identity["tree_sha256"]
    assert origin == "mac_verified_sparse_inventory"


def test_runtime_source_identity_accepts_attested_pinned_git_tree(tmp_path: Path) -> None:
    _runtime, repo, identity, digest = _source_identity_fixture(tmp_path)
    runtime = tmp_path / "git-runtime"
    runtime.mkdir()
    (runtime / "spec.json").write_text(
        json.dumps(
            {
                "generation": runtime.name,
                "root": str(runtime),
                "retention_source_identity": identity,
            }
        )
    )
    (runtime / "retention-source-delivery.json").write_text(
        json.dumps(
            {
                "generation": runtime.name,
                "transport": "github",
                "commit": identity["commit"],
                "tree": identity["tree"],
                "tree_sha256": digest,
            }
        )
    )

    actual, origin = retention.runtime_source_identity(runtime, repo)

    assert actual == digest
    assert origin == "runtime_git_tree"


@pytest.mark.parametrize("inventory_value", [None, False])
def test_runtime_source_identity_refuses_missing_or_unverified_fallback_inventory(
    tmp_path: Path, inventory_value: bool | None
) -> None:
    runtime, _repo, _identity, _digest = _source_identity_fixture(tmp_path)
    delivery = json.loads((runtime / "retention-source-delivery.json").read_text())
    if inventory_value is None:
        delivery.pop("inventory_verified")
    else:
        delivery["inventory_verified"] = inventory_value
    (runtime / "retention-source-delivery.json").write_text(json.dumps(delivery))

    with pytest.raises(retention.RetentionError, match="sparse source|inventory"):
        retention.runtime_source_identity(runtime, runtime / "repo")


@pytest.mark.parametrize("mutation", ["foreign-generation", "wrong-tree-digest"])
def test_runtime_source_identity_refuses_foreign_or_wrong_mac_attestation(
    tmp_path: Path, mutation: str
) -> None:
    runtime, _repo, _identity, digest = _source_identity_fixture(tmp_path)
    delivery = json.loads((runtime / "retention-source-delivery.json").read_text())
    if mutation == "foreign-generation":
        delivery["generation"] = "foreign-generation"
    else:
        delivery["tree_sha256"] = "0" * 64
    (runtime / "retention-source-delivery.json").write_text(json.dumps(delivery))

    with pytest.raises(retention.RetentionError, match="identity|generation|digest|source"):
        retention.runtime_source_identity(runtime, runtime / "repo")


@pytest.mark.parametrize("missing_fault", acceptance.REQUIRED_FAULTS)
def test_codex_v_fault_review_refuses_any_missing_fault(tmp_path: Path, missing_fault: str) -> None:
    report_log = tmp_path / "fault-tests.log"
    report_log.write_text("all mocked fault tests completed\n")
    report = tmp_path / "fault-test-report.json"
    report.write_text(
        json.dumps(
            {
                "exit_code": 0,
                "failed": 0,
                "passed": 8,
                "log": acceptance.binding(report_log),
            }
        )
    )
    cases = {fault: f"mock:{fault}" for fault in acceptance.REQUIRED_FAULTS}
    cases.pop(missing_fault)
    review = tmp_path / "codex-v-fault-review.json"
    review.write_text(
        json.dumps(
            {
                "schema": "o1-codex-v-fault-review/v1",
                "independent_reviewer": "Codex-V",
                "source_inventory": {"fixture.py": "a" * 64},
                "test_report": acceptance.binding(report),
                "fault_tests": cases,
            }
        )
    )
    plan = {"source_inventory": {"fixture.py": "a" * 64}}
    with pytest.raises(ValueError, match="each fault"):
        acceptance._offline_faults(plan, tmp_path)


def test_codex_v_fault_review_rejects_unbound_fault_labels(tmp_path: Path) -> None:
    log = tmp_path / "fault-tests.log"
    log.write_text("complete\n")
    report = tmp_path / "report.json"
    report.write_text(
        json.dumps({"exit_code": 0, "failed": 0, "passed": 8, "log": acceptance.binding(log)})
    )
    review = tmp_path / "codex-v-fault-review.json"
    review.write_text(
        json.dumps(
            {
                "schema": "o1-codex-v-fault-review/v1",
                "independent_reviewer": "Codex-V",
                "source_inventory": {"fixture.py": "a" * 64},
                "test_report": acceptance.binding(report),
                "fault_tests": {fault: f"mock:{fault}" for fault in acceptance.REQUIRED_FAULTS},
            }
        )
    )
    with pytest.raises(ValueError, match="executable test for each fault|SHA-bound"):
        acceptance._offline_faults({"source_inventory": {"fixture.py": "a" * 64}}, tmp_path)


def test_receipt_rejects_spoofed_export_hash(tmp_path: Path) -> None:
    """A forged export digest cannot make a native packet admissible."""
    output = tmp_path / "packet"
    exports = output / "exports"
    exports.mkdir(parents=True)
    manifest = output / "manifest.json"
    manifest.write_text(json.dumps({"commit": "a" * 40, "job_id": "packet-1"}))
    payload = exports / "probe.json"
    payload.write_text("actual\n")
    (Path(str(payload) + ".complete")).write_text("")
    (output / "receipt.json").write_text(
        json.dumps(
            {
                "manifest_sha256": acceptance.sha256_file(manifest),
                "commit": "a" * 40,
                "job_id": "packet-1",
                "status": "PASS",
                "stop_verified": True,
                "accounting": {"own_rate_units_per_hour": 10},
                "elapsed_seconds": 1,
                "exports": [
                    {
                        "path": "probe.json",
                        "size": payload.stat().st_size,
                        "sha256": "0" * 64,
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="export changed"):
        acceptance._receipt(output)
