"""End-to-end tests for the one-command TRAIN-01 continuation packet builder."""

from __future__ import annotations

import copy
import json
import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from research.sandbox_a_resident.training_compact import (
    materialize_compact_resume,
    verify_decoded_state,
)
from research.sandbox_a_resident.training_continuation import (
    ContinuationRefused,
    _derive_identities,
    build_continuation_packet,
    compute_remaining_envelope,
    run_dispatch_dry_run,
)
from research.sandbox_a_resident.training_units import (
    allocation,
    load_compact_allowlist,
    load_ledger,
    publish_snapshot,
)
from research.sandbox_a_resident.units import publish
from tools.sandbox.registry import TRANSPORT_TREE, digest

ROOT = Path(__file__).resolve().parents[2]


def _write_ten_line_sandbox(path: Path, sandbox: dict) -> None:
    lines = [
        ("{" if i == 0 else " ") + json.dumps(k) + ": " + json.dumps(v) + ("," if i < 9 else "}")
        for i, (k, v) in enumerate(sandbox.items())
    ]
    path.write_text("\n".join(lines) + "\n")


def _rebind_packet_to_fixture_repo(repo: Path, packet_name: str) -> None:
    packet = repo / "research/sandbox_a_resident" / packet_name
    manifest_path = packet / "dispatch-manifest.json"
    sandbox_path = packet / "sandbox.json"
    manifest = json.loads(manifest_path.read_text())
    primary = ROOT.parent / "snake-dqn"
    registry_path = repo / "tools/sandbox/SANDBOX_INPUT_REGISTRY.json"
    ledger_path = packet / "verified-ledger.json"
    fixed_map = {
        "b8-s3.pth": str(primary / "saved_snakes/b8_s3_t7_20261016.pth"),
        "frp3-s12.pth": str(primary / "saved_snakes/frp3_m3_s12_u60000_20261005.pth"),
        "champion.pth": str(primary / "saved_snakes/champion_a5_freespace_20260621.pth"),
        "registry.json": str(registry_path),
        "verified-ledger.json": str(ledger_path),
    }
    kept_inputs = []
    for row in manifest["inputs"]:
        if row["dest"].startswith("resume/"):
            continue
        row["path"] = fixed_map[row["dest"]]
        row["sha256"] = digest(Path(row["path"]))
        kept_inputs.append(row)
    manifest["inputs"] = kept_inputs
    manifest["sandbox_admission"] = {
        "schema": "standing-a100-development-sandbox-admission/v1",
        "manifest_path": str(sandbox_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    sandbox = json.loads(sandbox_path.read_text())
    worlds = [row for row in sandbox["inputs"] if row["kind"] == "worlds"]
    sandbox["inputs"] = worlds + [
        {
            "kind": (
                "frozen-checkpoint"
                if row["dest"] == "b8-s3.pth"
                else (
                    "registry"
                    if row["dest"] == "registry.json"
                    else (
                        "verified-ledger"
                        if row["dest"] == "verified-ledger.json"
                        else "frozen-opponent"
                    )
                )
            ),
            "path": row["path"],
            "sha256": row["sha256"],
        }
        for row in manifest["inputs"]
    ]
    sandbox["outputs"]["broker_manifest"] = {
        "path": str(manifest_path),
        "sha256": digest(manifest_path),
    }
    _write_ten_line_sandbox(sandbox_path, sandbox)


@pytest.fixture(scope="module")
def fixture_workspace(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Create a fast, clean standalone git repo with exact qualified transport tree."""
    base = tmp_path_factory.mktemp("train01-cont-ws").resolve()
    repo = base / "snake-dqn-fixture"
    repo.mkdir(parents=True)

    archive = subprocess.Popen(
        [
            "git",
            "-C",
            str(ROOT),
            "archive",
            "HEAD",
            ".gitignore",
            "pyproject.toml",
            "src/__init__.py",
            "src/core",
            "src/evaluation",
            "src/model",
            "src/simd_env",
            "src/utils",
            "tools/sandbox",
            "tools/ops",
            "research/colab_cli",
            "research/gpu_sim_spike",
        ],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    subprocess.run(["tar", "-x", "-C", str(repo)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    assert archive.wait(timeout=30) == 0

    src_pkg = ROOT / "research/sandbox_a_resident"
    dst_pkg = repo / "research/sandbox_a_resident"
    dst_pkg.mkdir(parents=True, exist_ok=True)
    for item in src_pkg.iterdir():
        if item.name == "__pycache__" or item.name.startswith("test_"):
            continue
        if item.is_file() and item.suffix in {".py", ".sh", ".json", ".txt"}:
            shutil.copy2(item, dst_pkg / item.name)
    for pkt in ("packet-train01-15", "packet-train01-16"):
        shutil.copytree(src_pkg / pkt, dst_pkg / pkt)
        _rebind_packet_to_fixture_repo(repo, pkt)

    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Argon Test"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "argon@test.local"], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "fixture base"], check=True)
    tree = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD:research/colab_cli"], text=True
    ).strip()
    assert tree == TRANSPORT_TREE
    return base


def _populate_attempt(
    attempt_dir: Path,
    source_packet_dir: Path,
    iterations: list[int],
    *,
    receipt_overrides: dict,
) -> Path:
    output = attempt_dir / "output"
    exports = output / "exports/training"
    exports.mkdir(parents=True)
    manifest = json.loads((source_packet_dir / "dispatch-manifest.json").read_text())
    control = json.loads((source_packet_dir / "control.json").read_text())
    executed_path = output / "manifest.json"
    executed_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    for iteration in iterations:
        state = {
            "schema": "sandbox-a-train-state/v1",
            "iteration": iteration,
            "actor": {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4) + iteration},
            "trainer": {"actor_rows": iteration * 1024 * 256},
            "preparation": {"stage": "ready", "temperature": 0.00843973},
        }
        meta = {
            "allocation": allocation(control),
            "source_commit": control["source_commit"],
            "prepare_phase": "ready",
        }
        if iteration == 0:
            meta["phase"] = "pre-actor"
        publish_snapshot(
            exports,
            iteration,
            lambda path, value=state: torch.save(value, path),
            metadata=meta,
        )
        publish(
            exports / f"iteration-{iteration:04d}-actor.pth",
            lambda path, idx=iteration: path.write_bytes(f"actor-{idx:04d}".encode()),
        )

    receipt = {
        "job_id": manifest["job_id"],
        "commit": manifest["commit"],
        "generation": "gen-fixture",
        "manifest_sha256": digest(executed_path),
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "terminal": None,
        "error": "SupervisorError('whole-session wall cap reached')",
        "stop_verified": True,
        "cleanup_error": None,
        **receipt_overrides,
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    return attempt_dir


def test_remaining_envelope_math_scales_updates_and_proxies_with_2x_rule() -> None:
    env_192 = compute_remaining_envelope(192)
    assert env_192["remaining_updates"] == 192
    assert env_192["scheduled_proxy_updates"] == [256, 320, 384]
    assert env_192["proxy_count"] == 3
    assert env_192["est_minutes"] == 600
    assert env_192["timeout_minutes"] == 1200
    assert env_192["unit_cap"] == 161.0
    assert env_192["unit_cap"] >= (env_192["timeout_minutes"] / 60.0) * 8.0

    env_176 = compute_remaining_envelope(176)
    assert env_176["remaining_updates"] == 208
    assert env_176["scheduled_proxy_updates"] == [192, 256, 320, 384]
    assert env_176["proxy_count"] == 4
    assert env_176["timeout_minutes"] == 2 * env_176["est_minutes"]

    with pytest.raises(ContinuationRefused):
        compute_remaining_envelope(384)


def test_continuation_builder_seals_train01_15_wall_timeout_and_verifies_launch_bound(
    fixture_workspace: Path,
) -> None:
    repo = fixture_workspace / "snake-dqn-fixture"
    artifacts = fixture_workspace / "snake-dqn-artifacts/sandbox-a-resident"
    source_packet = repo / "research/sandbox_a_resident/packet-train01-15"
    attempt_15 = _populate_attempt(
        artifacts / "train01-attempt-15-src",
        source_packet,
        [16, 176, 192],
        receipt_overrides={
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "SupervisorError('whole-session wall cap reached')",
        },
    )

    # Outcome-blind rule: requesting an earlier non-latest snapshot (176 vs 192) is refused.
    with pytest.raises(ContinuationRefused, match="not the latest verified resumable snapshot"):
        build_continuation_packet(
            attempt_15,
            176,
            repo_root=repo,
            checkout_parent=fixture_workspace,
            artifacts_root=artifacts,
            date_tag="20261028",
            skip_dry_run=True,
        )

    result = build_continuation_packet(
        attempt_15,
        192,
        repo_root=repo,
        checkout_parent=fixture_workspace,
        artifacts_root=artifacts,
        date_tag="20261028",
        skip_dry_run=True,
    )
    assert result["status"] == "PREPARED_NOT_DISPATCHED"
    assert result["allocations"] == 0
    assert result["packet_name"] == "packet-train01-15-c1-i0192"
    assert result["completion_attempt_index"] == 1
    assert result["resume_from_iteration"] == 192
    assert result["state_exists"] is False
    assert result["output_exists"] is False
    assert result["launch_exec_bound"]["status"] == "PASS"
    assert all(run["launch_seconds"] < 2.0 for run in result["launch_exec_bound"]["runs"])
    assert result["upload_staging_projection"]["status"] == "PASS"
    assert result["upload_staging_projection"]["total_upload_file_count"] == 18
    assert result["remaining_envelope"]["remaining_updates"] == 192
    assert result["remaining_envelope"]["scheduled_proxy_updates"] == [256, 320, 384]

    packet_dir = Path(result["packet_dir"])
    runtime_checkout = Path(result["runtime_checkout"])
    broker_checkout = Path(result["broker_checkout"])
    assert runtime_checkout.is_dir() and broker_checkout.is_dir()
    assert (runtime_checkout / "venv/bin/python").exists()
    assert (broker_checkout / "venv/bin/python").exists()

    # Verify compact allowlist entry committed in runtime_checkout and round-trip decode works.
    allowlist_path = runtime_checkout / "research/sandbox_a_resident/compact_allowlist.json"
    table = load_compact_allowlist(allowlist_path)
    orig_sha = result["compact_allowlist_entry"]["original"]["sha256"]
    assert ("packet-train01-15", 192, orig_sha) in table

    broker_manifest = json.loads((packet_dir / "dispatch-manifest.json").read_text())
    successor_control = json.loads((packet_dir / "control.json").read_text())
    ledger_payload = json.loads((packet_dir / "verified-ledger.json").read_text())
    assert successor_control["allocation_index"] == 0
    assert successor_control["train_seed_offset"] == 0
    assert broker_manifest["sandbox_a"]["seed_allocations"][3] == {
        "phase": "monitor",
        "offset": 400_000_000,
        "count": 288,
    }
    assert broker_manifest["sandbox_a"]["seed_allocations"][4] == {
        "phase": "nomination",
        "offset": 500_000_000,
        "count": 768,
    }

    vm_inputs = fixture_workspace / "vm-inputs-15-c1"
    for row in broker_manifest["inputs"]:
        if row["dest"].startswith("resume/") or row["dest"] == "verified-ledger.json":
            target = vm_inputs / row["dest"]
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(row["path"], target)
    decoded_manifest = materialize_compact_resume(
        ledger_payload, vm_inputs, allowlist_path=allowlist_path
    )
    decoded_pt = vm_inputs / "resume/iteration-0192.pt"
    verify_decoded_state(
        torch.load(decoded_pt, map_location="cpu", weights_only=False), decoded_manifest
    )
    verified = load_ledger(
        vm_inputs / "verified-ledger.json",
        successor_control,
        input_root=vm_inputs,
        allowlist_path=allowlist_path,
    )
    assert list(verified) == [192]


def test_continuation_builder_supports_train01_16_child_timeout_and_c1_to_c2_chain(
    fixture_workspace: Path,
) -> None:
    from research.sandbox_a_resident import training_continuation, training_preflight
    from research.sandbox_a_resident.training_completion import prepare_completion
    from research.sandbox_a_resident.training_units import register_compact_allowlist_entry

    repo = fixture_workspace / "snake-dqn-fixture"
    artifacts = fixture_workspace / "snake-dqn-artifacts/sandbox-a-resident"
    source_packet_16 = repo / "research/sandbox_a_resident/packet-train01-16"

    # Scientific tripwire / crash receipt (reason=None) must be refused before worktrees.
    bad_attempt = _populate_attempt(
        artifacts / "train01-attempt-16-crash",
        source_packet_16,
        [16],
        receipt_overrides={
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "sandbox-a-train01-20261027-16",
                "generation": "gen-fixture",
                "status": "failed",
                "returncode": 1,
                "reason": None,
            },
        },
    )
    with pytest.raises(ContinuationRefused, match="non-mechanical child failure"):
        build_continuation_packet(
            bad_attempt,
            16,
            repo_root=repo,
            checkout_parent=fixture_workspace,
            artifacts_root=artifacts,
            date_tag="20261028",
            skip_dry_run=True,
        )

    attempt_16 = _populate_attempt(
        artifacts / "train01-attempt-16-src",
        source_packet_16,
        [0, 176],
        receipt_overrides={
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "sandbox-a-train01-20261027-16",
                "generation": "gen-fixture",
                "status": "failed",
                "returncode": -15,
                "reason": "timeout",
            },
        },
    )
    c1_control = json.loads((source_packet_16 / "control.json").read_text())
    assert c1_control["allocation_index"] == 1
    assert c1_control["train_seed_offset"] == 1_000_000_000

    allowlist_16 = fixture_workspace / "c16-allowlist.json"
    shutil.copy2(repo / "research/sandbox_a_resident/compact_allowlist.json", allowlist_16)
    src_exports = attempt_16 / "output/exports/training"
    c1_entry = training_continuation.encode_or_verify_compact_snapshot(
        src_exports / "iteration-0176.pt",
        source_packet_id="packet-train01-16",
        iteration=176,
        source_commit=c1_control["source_commit"],
    )
    register_compact_allowlist_entry(c1_entry, path=allowlist_16)

    c1_packet_dir = fixture_workspace / "packet-train01-16-c1-i0176"
    prepare_completion(
        source_packet_16,
        attempt_16 / "output/receipt.json",
        src_exports,
        c1_packet_dir,
        index=1,
        job_id="sandbox-a-train01-20261028-16-c1-i0176",
        output=artifacts / "train01-attempt-16-c1/output",
        compact_source_packet_id="packet-train01-16",
        allowlist_path=allowlist_16,
    )
    c1_manifest = json.loads((c1_packet_dir / "dispatch-manifest.json").read_text())
    assert c1_manifest["completion_attempt"]["index"] == 1
    training_preflight.validate_uploaded_ledger(
        c1_packet_dir, c1_manifest, c1_control, allowlist_path=allowlist_16
    )

    # Simulate c1 stopping mechanically at iteration 320, verify c2 sealing (index=2),
    # and prove a 3rd continuation (c3) fails closed at build_continuation_packet.
    attempt_16_c1 = _populate_attempt(
        artifacts / "train01-attempt-16-c1-src",
        c1_packet_dir,
        [320],
        receipt_overrides={
            "failure_class": "RPC_EXEC_POLL_TIMEOUT",
            "terminal": None,
        },
    )
    c1_exports = attempt_16_c1 / "output/exports/training"
    c2_allowlist_entry = training_continuation.encode_or_verify_compact_snapshot(
        c1_exports / "iteration-0320.pt",
        source_packet_id="packet-train01-16-c1-i0176",
        iteration=320,
        source_commit=c1_control["source_commit"],
    )
    register_compact_allowlist_entry(c2_allowlist_entry, path=allowlist_16)
    c2_packet_dir = fixture_workspace / "packet-train01-16-c2-i0320"
    prepare_completion(
        c1_packet_dir,
        attempt_16_c1 / "output/receipt.json",
        c1_exports,
        c2_packet_dir,
        index=2,
        job_id="sandbox-a-train01-20261028-16-c2-i0320",
        output=artifacts / "train01-attempt-16-c2/output",
        compact_source_packet_id="packet-train01-16-c1-i0176",
        allowlist_path=allowlist_16,
    )
    c2_manifest = json.loads((c2_packet_dir / "dispatch-manifest.json").read_text())
    assert c2_manifest["completion_attempt"]["index"] == 2
    assert c2_manifest["completion_attempt"]["resume_from_iteration"] == 320
    env_320 = compute_remaining_envelope(320)
    assert env_320["remaining_updates"] == 64
    assert env_320["scheduled_proxy_updates"] == [384]
    training_preflight.validate_uploaded_ledger(
        c2_packet_dir, c2_manifest, c1_control, allowlist_path=allowlist_16
    )

    # A 3rd continuation (c3) exceeds max_outcome_blind_retries=2 and must fail closed.
    attempt_16_c2 = _populate_attempt(
        artifacts / "train01-attempt-16-c2-src",
        c2_packet_dir,
        [352],
        receipt_overrides={"failure_class": "RPC_EXEC_POLL_TIMEOUT", "terminal": None},
    )
    with pytest.raises(ContinuationRefused, match="only two manual outcome-blind"):
        build_continuation_packet(
            attempt_16_c2,
            352,
            source_packet=c2_packet_dir,
            repo_root=repo,
            checkout_parent=fixture_workspace,
            artifacts_root=artifacts,
            date_tag="20261028",
            skip_dry_run=True,
        )


def test_run_dispatch_dry_run_enforces_dry_run_flag_and_zero_allocations(tmp_path: Path) -> None:
    work = tmp_path.resolve()
    broker = work / "broker"
    runtime = work / "runtime"
    packet = broker / "research/sandbox_a_resident/packet-train01-15-c1-i0192"
    ops_dir = broker / "tools/ops"
    packet.mkdir(parents=True)
    ops_dir.mkdir(parents=True)
    runtime.mkdir(parents=True)
    (packet / "dispatch-manifest.json").write_text("{}")
    (packet / "DISPATCH.txt").write_text("ok\n")
    state_dir = work / "state"

    stub = ops_dir / "dispatch.sh"
    stub.write_text(
        "#!/usr/bin/env bash\n"
        'if [[ "${*: -1}" != "--dry-run" ]]; then exit 99; fi\n'
        'echo "PASS_DRY_RUN $1 allocations=0 live_accounting=NOT_CHECKED"\n'
    )
    stub.chmod(0o755)
    out = run_dispatch_dry_run(
        broker, runtime, packet, state_dir, "sandbox-a-train01-20261028-15-c1-i0192"
    )
    assert "PASS_DRY_RUN" in out and "allocations=0" in out
    assert not state_dir.exists()

    bad = copy.copy(stub)
    bad.write_text('#!/usr/bin/env bash\necho "UNEXPECTED_OUTPUT"\nexit 0\n')
    with pytest.raises(ContinuationRefused, match="dispatch.sh --dry-run failed"):
        run_dispatch_dry_run(
            broker, runtime, packet, state_dir, "sandbox-a-train01-20261028-15-c1-i0192"
        )


def test_derive_identities_supports_train01_17_and_c1_c2_continuations() -> None:
    slug_c1, suffix_c1, job_c1 = _derive_identities(
        "packet-train01-17",
        completion_index=1,
        iteration=176,
        slug_override=None,
        job_id_override=None,
        date_tag="20261028",
    )
    assert slug_c1 == "train01-17-c1-i0176"
    assert suffix_c1 == "17-c1-i0176"
    assert job_c1 == "sandbox-a-train01-20261028-17-c1-i0176"

    slug_c2, suffix_c2, job_c2 = _derive_identities(
        f"packet-{slug_c1}",
        completion_index=2,
        iteration=320,
        slug_override=None,
        job_id_override=None,
        date_tag="20261028",
    )
    assert slug_c2 == "train01-17-c2-i0320"
    assert suffix_c2 == "17-c2-i0320"
    assert job_c2 == "sandbox-a-train01-20261028-17-c2-i0320"
