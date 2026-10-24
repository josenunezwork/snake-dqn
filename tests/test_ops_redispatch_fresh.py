"""Offline, fail-closed tests for explicit fresh GPU allocation preparation."""

from __future__ import annotations

import copy
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from tools.ops import redispatch_fresh as fresh


def write(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Model the actual supervisor first-readiness failure, without provider access."""
    pack = tmp_path / "gpu-sprint"
    pack.mkdir()
    out = pack / "canary-output"
    out.mkdir()
    manifest = {
        "job_id": "sparse-canary-12345678",
        "commit": "a" * 40,
        "script": "research/gpu_sim_spike/colab.sh",
        "output_dir": str(out),
        "requirements": {"gpu": "A100", "min_vcpu": 12, "est_minutes": 10},
        "budget": {
            "unit_cap": 4,
            "reserve_units": 0.5,
            "rate_units_per_hour": 10,
            "other_assignments": {},
        },
        "timeout_minutes": 24,
        "env": {"SPARSE_BENCH_CANARY": "1"},
        "inputs": [{"path": "/unchanged/fixture", "dest": "fixture", "sha256": "b" * 64}],
        "resume": {
            "unit_kind": "cell",
            "ledger_path": "ledger.json",
            "skip_verified": True,
            "max_outcome_blind_retries": 2,
        },
    }
    write(pack / "canary.json", manifest)
    live = copy.deepcopy(manifest)
    live["budget"]["other_assignments"] = {"other": 5.3}
    write(out / "manifest.json", live)
    receipt = {
        "job_id": manifest["job_id"],
        "commit": manifest["commit"],
        "manifest_sha256": fresh.digest(out / "manifest.json"),
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "error": "Refused('colab exec failed (1): GuardError: " + fresh.KERNEL_ERROR + "')",
        "stop_verified": True,
        "cleanup_error": None,
        "terminal": None,
        "endpoint": "gpu-a100-test",
        "exports": [],
        "transfers": [],
        "contents_transfers": [],
        "accounting": {"allocated_endpoint": None, "own_rate_units_per_hour": 10},
        "elapsed_seconds": 31,
        "warmup_seconds": 23,
        "cli_calls": [
            {"attempt": 1, "command": command, "status": status}
            for command, status in [
                ("version", "completed"),
                ("sessions", "completed"),
                ("usage", "completed"),
                ("new", "completed"),
                ("exec", "failed"),
                ("stop", "completed"),
                ("sessions", "completed"),
                ("usage", "completed"),
            ]
        ],
    }
    write(out / "receipt.json", receipt)
    write(
        out / "warmup-measurement.json",
        {
            "attempts": 1,
            "status": "INCOMPLETE",
            "error": receipt["error"],
            "elapsed_seconds": 23,
        },
    )
    ops = tmp_path / "ops"
    ops.mkdir()
    write(ops / "colab-cli-qualified.json", {"test": True})
    monkeypatch.setenv("OPS_DIR", str(ops))
    monkeypatch.setattr(fresh.qualification, "check", lambda *a, **kw: "qualified-tree")
    monkeypatch.setattr(fresh, "validate", lambda manifest, repo: manifest)
    return pack


def change_receipt(package: Path, key: str, value: Any) -> None:
    path = package / "canary-output/receipt.json"
    receipt = fresh.load(path)
    receipt[key] = value
    write(path, receipt)


def test_fresh_sibling_changes_only_namespace_and_never_dispatches(
    package: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    before = {p: p.read_bytes() for p in package.rglob("*") if p.is_file()}
    validated = []
    monkeypatch.setattr(fresh, "validate", lambda m, repo: validated.append(copy.deepcopy(m)))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("unexpected subprocess"))
    sibling, command = fresh.prepare(package, broker=fresh.ROOT, suffix="fresh-01")

    old = fresh.load(package / "canary.json")
    new = fresh.load(sibling / "canary.json")
    assert {k for k in old if old[k] != new[k]} == {"job_id", "output_dir"}
    assert new["job_id"] == old["job_id"] + "-fresh-01"
    assert validated == [new]
    assert not Path(new["output_dir"]).exists()
    assert not (sibling / "state").exists()
    argv = shlex.split(command)
    assert argv[0] == "env"  # Retain an explicit OPS_DIR override in printed command.
    assert argv[2:5] == [
        str(fresh.ROOT / "tools/ops/dispatch.sh"),
        new["job_id"],
        str(sibling / "canary.json"),
    ]
    assert str(sibling / "state") in argv
    proof = fresh.load(sibling / "PREPARATION.json")
    assert proof["workload_gpu_seconds"] == 0
    assert proof["prior_elapsed_rate_units"] == pytest.approx(31 * 10 / 3600)
    for path in sibling.iterdir():
        if path.suffix in (".json", ".txt"):
            assert Path(str(path) + ".complete").read_text() == "complete\n"
            assert Path(str(path) + ".sha256").read_text().split()[0] == fresh.digest(path)
    assert {p: p.read_bytes() for p in before} == before
    with pytest.raises(fresh.Refused, match="already exists"):
        fresh.prepare(package, broker=fresh.ROOT, suffix="fresh-01")


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "PASS"),
        ("failure_class", "JOB"),
        ("stop_verified", False),
        ("stop_verified", 1),
        ("cleanup_error", "stop failed"),
        ("terminal", {}),
        ("error", "CUDA out of memory"),
        ("exports", [{"path": "result"}]),
        ("transfers", [{}]),
        ("contents_transfers", [{}]),
        ("exports", None),
        ("endpoint", None),
        ("gpu_seconds", 0.01),
        ("gpu_seconds", False),
        ("gpu_time_seconds", float("nan")),
        ("workload_gpu_seconds", "0"),
        ("cli_calls", []),
        ("cli_calls", ["exec"]),
        ("accounting", {"allocated_endpoint": "gpu-test", "own_rate_units_per_hour": 10}),
    ],
)
def test_other_failure_or_uncertain_work_refused_before_creating_sibling(
    package: Path, key: str, value: Any
) -> None:
    change_receipt(package, key, value)
    with pytest.raises(fresh.Refused):
        fresh.prepare(package, broker=fresh.ROOT, suffix="refuse")
    assert not package.with_name(package.name + "-refuse").exists()


@pytest.mark.parametrize("name", ["exports", "hardware.json", "runtime-status.json"])
def test_unreported_runtime_artifacts_are_refused(package: Path, name: str) -> None:
    (package / "canary-output" / name).write_text("unexpected")
    with pytest.raises(fresh.Refused, match="unexpected output"):
        fresh.prepare(package, broker=fresh.ROOT, suffix="refuse")


def test_receipt_and_workload_identity_are_bound(package: Path) -> None:
    source = package / "canary.json"
    original = fresh.load(source)
    original["timeout_minutes"] = 30
    write(source, original)
    with pytest.raises(fresh.Refused, match="source differs"):
        fresh.prepare(package, broker=fresh.ROOT)


def test_saved_manifest_tamper_is_refused(package: Path) -> None:
    saved = package / "canary-output/manifest.json"
    saved.write_text(saved.read_text() + " ")
    with pytest.raises(fresh.Refused, match="bind saved manifest"):
        fresh.prepare(package, broker=fresh.ROOT)


@pytest.mark.parametrize(
    "terminal",
    ["ReadTimeout: timed out", "GuardError: kernel creation returned malformed kernel identity"],
)
def test_source_excerpt_cannot_impersonate_terminal_exception(package: Path, terminal: str) -> None:
    detail = (
        "colab exec failed (1): source traceback\n"
        f'| raise GuardError("{fresh.KERNEL_ERROR}") |\n{terminal}\n'
    )
    error = repr(RuntimeError(detail)).replace("RuntimeError(", "Refused(", 1)
    change_receipt(package, "error", error)
    path = package / "canary-output/warmup-measurement.json"
    warmup = fresh.load(path)
    warmup["error"] = error
    write(path, warmup)
    with pytest.raises(fresh.Refused, match="kernel-creation failure"):
        fresh.prepare(package, broker=fresh.ROOT)


def test_multiline_rich_terminal_guard_is_accepted() -> None:
    detail = (
        f'colab exec failed (1): | raise GuardError("{fresh.KERNEL_ERROR}") |\n'
        f"GuardError: {fresh.KERNEL_ERROR}\n"
    )
    assert fresh.is_kernel_creation_failure("Refused(" + repr(detail) + ")")
    assert not fresh.is_kernel_creation_failure("RuntimeError(" + repr(detail) + ")")


@pytest.mark.parametrize("field,value", [("attempts", 2), ("status", "PASS"), ("error", "other")])
def test_warmup_must_prove_first_probe_failure(package: Path, field: str, value: Any) -> None:
    path = package / "canary-output/warmup-measurement.json"
    evidence = fresh.load(path)
    evidence[field] = value
    write(path, evidence)
    with pytest.raises(fresh.Refused, match="warmup"):
        fresh.prepare(package, broker=fresh.ROOT)


def test_missing_proof_is_not_zero(package: Path) -> None:
    receipt_path = package / "canary-output/receipt.json"
    receipt = fresh.load(receipt_path)
    del receipt["exports"]
    write(receipt_path, receipt)
    with pytest.raises(fresh.Refused, match="explicitly empty"):
        fresh.prepare(package, broker=fresh.ROOT)


@pytest.mark.parametrize("gate", ["qualification", "validation", "resume"])
def test_offline_gate_failure_leaves_no_new_package(
    package: Path, monkeypatch: pytest.MonkeyPatch, gate: str
) -> None:
    def fail(*args: Any, **kwargs: Any) -> None:
        raise ValueError("gate refused")

    if gate == "qualification":
        monkeypatch.setattr(fresh.qualification, "check", fail)
    elif gate == "validation":
        monkeypatch.setattr(fresh, "validate", fail)
    else:
        monkeypatch.setattr(fresh, "require_resume_policy", fail)
    with pytest.raises(ValueError, match="gate refused"):
        fresh.prepare(package, broker=fresh.ROOT, suffix="refuse")
    assert not package.with_name(package.name + "-refuse").exists()


def test_blocked_package_cannot_be_laundered(package: Path) -> None:
    (package / "DISPATCH.txt").write_text("HOLD: review first\n")
    with pytest.raises(fresh.Refused, match="blocking instructions"):
        fresh.prepare(package, broker=fresh.ROOT)


def test_ambiguous_manifests_require_explicit_choice(package: Path) -> None:
    (package / "duplicate.json").write_bytes((package / "canary.json").read_bytes())
    with pytest.raises(fresh.Refused, match="exactly one"):
        fresh.prepare(package, broker=fresh.ROOT)
    sibling, _ = fresh.prepare(package, broker=fresh.ROOT, manifest_name="canary.json")
    assert sibling.exists()


def test_job_id_length_bound_and_default_names_are_unique(package: Path) -> None:
    one, _ = fresh.prepare(package, broker=fresh.ROOT)
    two, _ = fresh.prepare(package, broker=fresh.ROOT)
    assert one != two
    assert len(fresh.load(one / "canary.json")["job_id"]) <= 64


def test_cli_refusal_has_no_dispatch_command(
    package: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    change_receipt(package, "stop_verified", False)
    assert fresh.main([str(package)]) == 2
    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.startswith("REFUSED:")


def test_real_cli_help_is_standalone() -> None:
    result = subprocess.run(
        [sys.executable, str(fresh.ROOT / "tools/ops/redispatch_fresh.py"), "--help"],
        cwd="/tmp",
        capture_output=True,
        text=True,
        check=True,
    )
    assert "--manifest" in result.stdout
