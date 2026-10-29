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
    MAX_CONTINUATION_PRIVATE_INPUT_BYTES,
    STAGING_DESIGN_NOMINAL_SECONDS,
    STAGING_DESIGN_WORST_SECONDS,
    STAGING_HARD_CAP_SECONDS,
    ContinuationRefused,
    _derive_identities,
    build_continuation_packet,
    compute_remaining_envelope,
    compute_staging_projection,
    encode_or_verify_compact_snapshot,
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


def test_original_packet_seal_is_git_verified_without_continuation_proof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from research.sandbox_a_resident import training_preflight

    seal = tmp_path / "SEAL.json"
    seal.write_text(json.dumps({"schema": "sandbox-a-train01-resident-prep-seal/v1"}))
    verified = []
    monkeypatch.setattr(training_preflight, "verify_committed_manifest", verified.append)
    training_preflight._validate_continuation_stop(tmp_path, {})
    assert verified == [seal]
    with pytest.raises(ValueError, match="unsupported continuation"):
        training_preflight._validate_continuation_stop(tmp_path, {"completion_attempt": {}})


@pytest.mark.parametrize("prior_name", ["", ".", "..", "../packet", None])
def test_review_r4_prior_packet_name_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prior_name: str | None
) -> None:
    from research.sandbox_a_resident import training_preflight

    (tmp_path / "SEAL.json").write_text(
        json.dumps(
            {
                "schema": "sandbox-a-train01-continuation-seal/v1",
                "prior_stop_evidence_kind": "supervisor_receipt",
                "prior_packet": prior_name,
            }
        )
    )
    monkeypatch.setattr(training_preflight, "verify_committed_manifest", lambda _path: None)
    with pytest.raises(ValueError, match="prior_packet must be a packet name"):
        training_preflight._validate_continuation_stop(tmp_path, {"completion_attempt": {}})


@pytest.mark.parametrize("symlink", [False, True])
def test_review_r4_tracked_seal_refuses_external_or_symlinked_prior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, symlink: bool
) -> None:
    from research.sandbox_a_resident import training_preflight

    packet = tmp_path / "packet"
    packet.mkdir()
    foreign = tmp_path / "external/prior"
    foreign.mkdir(parents=True)
    source = foreign
    if symlink:
        source = tmp_path / "prior"
        source.symlink_to(foreign, target_is_directory=True)
    seal = packet / "SEAL.json"
    seal.write_text(
        json.dumps(
            {
                "schema": "sandbox-a-train01-continuation-seal/v1",
                "prior_stop_evidence_kind": "supervisor_receipt",
                "prior_packet": "prior",
                "prior_packet_path": str(source),
            }
        )
    )
    subprocess.run(["git", "init", "-q", str(packet)], check=True)
    subprocess.run(["git", "-C", str(packet), "add", "SEAL.json"], check=True)
    monkeypatch.setattr(training_preflight, "verify_committed_manifest", lambda _path: None)
    with pytest.raises(ValueError, match="prior_packet_path differs"):
        training_preflight._validate_continuation_stop(packet, {"completion_attempt": {}})


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
    attempt_slug = (
        "train01-attempt-15" if packet_name == "packet-train01-15" else "train01-attempt-16"
    )
    attempt_out = str(
        (repo.parent / "snake-dqn-artifacts/sandbox-a-resident" / attempt_slug / "output").resolve()
    )
    manifest["inputs"] = kept_inputs
    manifest["output_dir"] = attempt_out
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
    sandbox["outputs"]["paths"] = [attempt_out]
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
            "research/compute",
        ],
        stdout=subprocess.PIPE,
    )
    assert archive.stdout is not None
    subprocess.run(["tar", "-x", "-C", str(repo)], stdin=archive.stdout, check=True)
    archive.stdout.close()
    assert archive.wait(timeout=30) == 0

    for item in (ROOT / "tools/sandbox").glob("*.py"):
        shutil.copy2(item, repo / "tools/sandbox" / item.name)

    src_pkg = ROOT / "research/sandbox_a_resident"
    dst_pkg = repo / "research/sandbox_a_resident"
    dst_pkg.mkdir(parents=True, exist_ok=True)
    for item in src_pkg.iterdir():
        if item.name == "__pycache__" or item.name.startswith("test_"):
            continue
        if item.is_file() and item.suffix in {".py", ".sh", ".json", ".txt"}:
            shutil.copy2(item, dst_pkg / item.name)
    allowlist_file = dst_pkg / "compact_allowlist.json"
    allowlist_payload = json.loads(allowlist_file.read_text())
    allowlist_payload["entries"] = [
        row
        for row in allowlist_payload.get("entries", [])
        if row.get("source_packet_id") == "packet-train01-11"
    ]
    allowlist_file.write_text(json.dumps(allowlist_payload, indent=2, sort_keys=True) + "\n")
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


def test_review_r4_completion_seal_survives_clean_detached_checkout(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Use real Git binding checks in the relocated checkout, without monkeypatches."""
    import sys

    from research.sandbox_a_resident.training_completion import prepare_completion

    repo = fixture_workspace / "snake-dqn-fixture"
    package = repo / "research/sandbox_a_resident"
    original = package / "packet-train01-15"
    attempt = _populate_attempt(
        tmp_path / "portable-attempt",
        original,
        [16],
        receipt_overrides={"failure_class": "RPC_EXEC_POLL_TIMEOUT", "error": None},
    )
    successor = package / "packet-review-r4-portable"
    prepare_completion(
        original,
        attempt / "output/receipt.json",
        attempt / "output/exports/training",
        successor,
        index=1,
        job_id="review-r4-portable",
        output=tmp_path / "successor-output",
    )
    assert "prior_packet_path" not in json.loads((successor / "SEAL.json").read_text())
    subprocess.run(["git", "-C", str(repo), "add", str(successor)], check=True)
    subprocess.run(
        ["git", "-C", str(repo), "commit", "-qm", "portable completion fixture"], check=True
    )
    detached = tmp_path / "detached"
    subprocess.run(
        ["git", "-C", str(repo), "worktree", "add", "--detach", str(detached), "HEAD"], check=True
    )
    code = (
        "import json; from pathlib import Path; "
        "from research.sandbox_a_resident.training_preflight import _validate_continuation_stop; "
        "p=Path('research/sandbox_a_resident/packet-review-r4-portable').resolve(); "
        "_validate_continuation_stop(p,json.loads((p/'dispatch-manifest.json').read_text()))"
    )
    subprocess.run([sys.executable, "-c", code], cwd=detached, check=True)
    # Tracked-file discovery uses the packet's checkout even if its seal vanishes.
    (detached / "research/sandbox_a_resident/packet-review-r4-portable/SEAL.json").unlink()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            code.replace("json.loads((p/'dispatch-manifest.json').read_text())", "{}"),
        ],
        cwd=detached,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0 and "continuation proof" in result.stderr


def test_remaining_envelope_math_scales_updates_and_proxies_with_2x_rule() -> None:
    env_192 = compute_remaining_envelope(192)
    assert env_192["remaining_updates"] == 192
    assert env_192["scheduled_proxy_updates"] == [256, 320, 384]
    assert env_192["proxy_count"] == 3
    assert env_192["est_minutes"] == 840
    assert env_192["timeout_minutes"] == 1680
    assert env_192["unit_cap"] == 225.0
    assert env_192["unit_cap"] >= (env_192["timeout_minutes"] / 60.0) * 8.0

    env_192_deferred = compute_remaining_envelope(192, deferred_proxy=True)
    assert env_192_deferred["proxy_count"] == 4
    assert env_192_deferred["est_minutes"] == 870
    assert env_192_deferred["timeout_minutes"] == 1740

    env_48 = compute_remaining_envelope(48)
    assert env_48["remaining_updates"] == 336
    assert env_48["scheduled_proxy_updates"] == [64, 128, 192, 256, 320, 384]
    assert env_48["proxy_count"] == 6
    assert env_48["timeout_minutes"] == 2 * env_48["est_minutes"]
    assert env_48["unit_cap"] >= (env_48["timeout_minutes"] / 60.0) * 8.0

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

    # A receipt-backed c1 must reopen its bound stop proof on each preflight.
    from research.sandbox_a_resident.training_continuation import run_packet_validation

    prior_receipt = attempt_15 / "output/receipt.json"
    saved_receipt = prior_receipt.read_bytes()
    for mutation in (None, {"stop_verified": False}, {"failure_class": "CHILD_NONZERO_EXIT"}):
        if mutation is None:
            prior_receipt.unlink()
        else:
            prior_receipt.write_text(json.dumps({**json.loads(saved_receipt), **mutation}))
        with pytest.raises(ContinuationRefused):
            run_packet_validation(broker_checkout, packet_dir)
        prior_receipt.write_bytes(saved_receipt)
    retained_fault = attempt_15 / "output/exports/training/job.log"
    retained_fault.write_text("FloatingPointError: non-finite loss tripwire\n")
    with pytest.raises(ContinuationRefused, match="explicit scientific/non-mechanical fault"):
        run_packet_validation(broker_checkout, packet_dir)
    retained_fault.unlink()
    seal_path = packet_dir / "SEAL.json"
    saved_seal = seal_path.read_bytes()
    seal = json.loads(saved_seal)
    for mutation in (None, {"prior_stop_evidence_kind": "unknown"}):
        if mutation is None:
            seal_path.unlink()
        else:
            seal_path.write_text(json.dumps({**seal, **mutation}))
        with pytest.raises(ContinuationRefused):
            run_packet_validation(broker_checkout, packet_dir)
        seal_path.write_bytes(saved_seal)


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


def test_occupied_simulator_ring_periodic_state_and_67mb_staging_bounds(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """Occupied periodic simulator-ring state round-trips and ~66.6 MB inputs pass staging."""
    from research.sandbox_a_resident.training_resume import attach_compact_resume, build_ledger
    from research.sandbox_a_resident.training_units import (
        COMPACT_ALLOWLIST_SCHEMA,
        register_compact_allowlist_entry,
    )

    repo = fixture_workspace / "snake-dqn-fixture"
    source_packet = repo / "research/sandbox_a_resident/packet-train01-15"
    src_control = json.loads((source_packet / "control.json").read_text())
    dst_control = {**src_control, "source_commit": "f" * 40}

    gen = torch.Generator().manual_seed(20261028)
    worlds, snakes, ring_cap, food_cap = 64, 6, 512, 512
    bodies = torch.zeros((worlds, snakes, ring_cap, 2), dtype=torch.int64)
    head_idx = torch.randint(0, ring_cap, (worlds, snakes), generator=gen, dtype=torch.int64)
    lengths = torch.randint(12, 110, (worlds, snakes), generator=gen, dtype=torch.int64)
    for w in range(worlds):
        for s in range(snakes):
            length = int(lengths[w, s].item())
            start_head = int(head_idx[w, s].item())
            coords = torch.randint(1, 127, (length, 2), generator=gen, dtype=torch.int64)
            slots = (start_head - torch.arange(length, dtype=torch.int64)) % ring_cap
            bodies[w, s, slots] = coords
    nonzero_body_fraction = float((bodies != 0).any(dim=-1).float().mean().item())
    assert 0.10 <= nonzero_body_fraction <= 0.25

    food_cells = torch.zeros((worlds, food_cap, 2), dtype=torch.int64)
    food_present = torch.rand((worlds, food_cap), generator=gen) < 0.40
    active_food = int(food_present.sum().item())
    food_cells[food_present] = torch.randint(
        1, 127, (active_food, 2), generator=gen, dtype=torch.int64
    )
    random_slots = torch.randint(
        0, 6, (worlds, 3, ring_cap, 3), generator=gen, dtype=torch.int64
    ) * (torch.rand((worlds, 3, ring_cap, 1), generator=gen) < 0.39).to(torch.int64)

    periodic_state = {
        "schema": "sandbox-a-train-state/v1",
        "iteration": 48,
        "actor": {
            "trunk.weight": torch.randn((128, 64), generator=gen, dtype=torch.float32),
            "head.weight": torch.randn((6, 128), generator=gen, dtype=torch.float32),
            "_temperature": torch.tensor(0.00843973, dtype=torch.float32),
        },
        "critic": {
            "trunk.weight": torch.randn((128, 64), generator=gen, dtype=torch.float32),
            "mu": torch.tensor([0.125], dtype=torch.float64),
            "sigma": torch.tensor([1.875], dtype=torch.float64),
        },
        "trainer": {
            "actor_rows": 48 * 1024 * 256,
            "exp_avg": torch.randn((128, 64), generator=gen, dtype=torch.float32),
            "exp_avg_sq": torch.rand((128, 64), generator=gen, dtype=torch.float32),
        },
        "task": {
            "sim": {
                "bodies": bodies,
                "head_idx": head_idx,
                "lengths": lengths,
                "food_cells": food_cells,
                "food_present": food_present,
            },
            "groups": {2: {"random_slots": [random_slots, None]}},
            "age": torch.randint(0, 4800, (worlds,), generator=gen, dtype=torch.int64),
        },
        "rng": {"torch": torch.get_rng_state(), "collection": gen.get_state()},
        "metrics": [
            {"iteration": 0, "preparation": {"stage": "ready"}},
            {
                "iteration": 16,
                "collector_seconds": 204.78,
                "proxy": {"kind": "diagnostic-update-16"},
            },
            {"iteration": 48, "collector_seconds": 204.30, "learner_seconds": 8.59},
        ],
        "preparation": {"stage": "ready", "temperature": 0.00843973},
        "action_window": torch.randint(0, 6, (100_000,), generator=gen, dtype=torch.long),
    }

    exports = tmp_path / "exports"
    exports.mkdir(parents=True)
    publish_snapshot(
        exports,
        48,
        lambda path: torch.save(periodic_state, path),
        metadata={
            "allocation": allocation(src_control),
            "source_commit": src_control["source_commit"],
            "prepare_phase": "ready",
        },
    )
    publish(exports / "iteration-0048-actor.pth", lambda path: path.write_bytes(b"actor-0048"))

    entry = encode_or_verify_compact_snapshot(
        exports / "iteration-0048.pt",
        source_packet_id="packet-train01-15",
        iteration=48,
        source_commit=src_control["source_commit"],
    )
    assert entry["derivative"]["size"] > 0
    allowlist_path = tmp_path / "compact_allowlist.json"
    allowlist_path.write_text(json.dumps({"schema": COMPACT_ALLOWLIST_SCHEMA, "entries": []}))
    register_compact_allowlist_entry(entry, path=allowlist_path)

    ledger, uploads, latest = build_ledger(
        exports, dst_control, source_control=src_control, allowlist_path=allowlist_path
    )
    assert latest == 48
    ledger, uploads = attach_compact_resume(
        ledger,
        uploads,
        exports,
        source_packet_id="packet-train01-15",
        allowlist_path=allowlist_path,
    )
    vm_inputs = tmp_path / "vm-inputs"
    for row in uploads:
        target = vm_inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(row["path"], target)
    decoded_manifest = materialize_compact_resume(ledger, vm_inputs, allowlist_path=allowlist_path)
    assert (vm_inputs / "resume/iteration-0048.pt").read_bytes() == (
        exports / "iteration-0048.pt"
    ).read_bytes()
    verify_decoded_state(
        torch.load(vm_inputs / "resume/iteration-0048.pt", map_location="cpu", weights_only=False),
        decoded_manifest,
    )

    # Reconstruct the exact ~66.60 MB 15-private-input periodic profile (e.g. real iteration-0048:
    # 38,332,660 B .pt.xz + 93,518 B .pt.xz.manifest + fixed checkpoints) and verify staging PASS.
    base_manifest = json.loads((source_packet / "dispatch-manifest.json").read_text())
    fixed_inputs = [
        row
        for row in base_manifest["inputs"]
        if not row["dest"].startswith("resume/") and row["dest"] != "verified-ledger.json"
    ]
    real_scale_dir = tmp_path / "real-scale-periodic"
    real_scale_dir.mkdir()
    real_pt_xz = real_scale_dir / "iteration-0048.pt.xz"
    real_pt_xz.write_bytes(b"\xfd7zXZ\x00" + b"\x00" * (38_332_660 - 6))
    real_manifest = real_scale_dir / "iteration-0048.pt.xz.manifest"
    real_manifest.write_bytes(b" " * 93_518)
    real_actor = real_scale_dir / "iteration-0048-actor.pth"
    real_actor.write_bytes(b"\x00" * 6_574_173)
    real_ledger = real_scale_dir / "verified-ledger.json"
    real_ledger.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")

    periodic_inputs = list(fixed_inputs) + [
        {"dest": "verified-ledger.json", "path": str(real_ledger)}
    ]
    for row in uploads:
        if row["dest"].endswith(".pt.xz"):
            periodic_inputs.append({"dest": row["dest"], "path": str(real_pt_xz)})
        elif row["dest"].endswith(".pt.xz.manifest"):
            periodic_inputs.append({"dest": row["dest"], "path": str(real_manifest)})
        elif row["dest"].endswith("-actor.pth"):
            periodic_inputs.append({"dest": row["dest"], "path": str(real_actor)})
        else:
            periodic_inputs.append({"dest": row["dest"], "path": row["path"]})
    assert len(periodic_inputs) == 15

    projection = compute_staging_projection(
        {"inputs": periodic_inputs},
        repo,
        source_tar_gz_bytes=1_000_000,
    )
    assert projection["status"] == "PASS"
    assert projection["total_upload_file_count"] == 18
    assert 66_000_000 < projection["private_input_bytes"] <= MAX_CONTINUATION_PRIVATE_INPUT_BYTES
    assert projection["nominal_seconds"] <= STAGING_DESIGN_NOMINAL_SECONDS
    assert projection["worst_case_seconds"] <= STAGING_DESIGN_WORST_SECONDS
    assert projection["worst_case_seconds"] < STAGING_HARD_CAP_SECONDS

    # Payloads exceeding MAX_CONTINUATION_PRIVATE_INPUT_BYTES (>100 MB or raw 1.34 GB) fail closed.
    oversized = real_scale_dir / "oversized.pt.xz"
    oversized.write_bytes(b"\x00" * (MAX_CONTINUATION_PRIVATE_INPUT_BYTES + 1))
    bad_inputs = [
        {
            "dest": row["dest"],
            "path": str(oversized) if row["dest"].endswith(".pt.xz") else row["path"],
        }
        for row in periodic_inputs
    ]
    with pytest.raises(ContinuationRefused, match="exceed .* whole-staging ceiling"):
        compute_staging_projection({"inputs": bad_inputs}, repo, source_tar_gz_bytes=1_000_000)


def test_continuation_builder_refuses_timeout_with_scientific_fault_or_poll_fault(
    fixture_workspace: Path,
) -> None:
    repo = fixture_workspace / "snake-dqn-fixture"
    artifacts = fixture_workspace / "snake-dqn-artifacts/sandbox-a-resident"
    source_packet = repo / "research/sandbox_a_resident/packet-train01-15"

    # Case 1: CHILD_NONZERO_EXIT with reason="timeout" AND explicit scientific tripwire traceback.
    raced_attempt = _populate_attempt(
        artifacts / "train01-attempt-15-timeout-race",
        source_packet,
        [48],
        receipt_overrides={
            "failure_class": "CHILD_NONZERO_EXIT",
            "terminal": {
                "job_id": "sandbox-a-train01-20261027-15",
                "generation": "gen-fixture",
                "status": "failed",
                "returncode": 1,
                "reason": "timeout",
                "last_traceback_line": "FloatingPointError: non-finite loss tripwire",
            },
        },
    )
    with pytest.raises(ContinuationRefused, match="explicit scientific/non-mechanical fault"):
        build_continuation_packet(
            raced_attempt,
            48,
            repo_root=repo,
            checkout_parent=fixture_workspace,
            artifacts_root=artifacts,
            date_tag="20261028",
            skip_dry_run=True,
        )

    # Case 2: SUPERVISOR wall-cap receipt with terminal=None, but poll log records non-finite fault.
    poll_fault_attempt = _populate_attempt(
        artifacts / "train01-attempt-15-poll-fault",
        source_packet,
        [48],
        receipt_overrides={
            "failure_class": "SUPERVISOR",
            "terminal": None,
            "error": "SupervisorError('whole-session wall cap reached')",
        },
    )
    (poll_fault_attempt / "output/poll-000042.json").write_text(
        json.dumps(
            {
                "job_log_tail": (
                    'TRAIN01_PROGRESS {"finite": false, ' '"first_nonfinite_term": "policy_loss"}\n'
                ),
                "exports": [],
            }
        )
        + "\n"
    )
    with pytest.raises(ContinuationRefused, match="explicit scientific/non-mechanical fault"):
        build_continuation_packet(
            poll_fault_attempt,
            48,
            repo_root=repo,
            checkout_parent=fixture_workspace,
            artifacts_root=artifacts,
            date_tag="20261028",
            skip_dry_run=True,
        )


def test_storage_projection_admits_compact_u64_and_refuses_raw_or_low_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real u64 scale: 20 compact exports < 1 GB pass on 13 GiB free; 20 raw exports refuse."""
    from collections import namedtuple

    from research.sandbox_a_resident import training_continuation
    from research.sandbox_a_resident.training_continuation import compute_storage_projection

    DiskUsage = namedtuple("DiskUsage", ["total", "used", "free"])
    free_13_gib = 13 * 1024**3
    total_bytes = 500 * 1024**3
    monkeypatch.setattr(
        training_continuation.shutil,
        "disk_usage",
        lambda _path: DiskUsage(total_bytes, total_bytes - free_13_gib, free_13_gib),
    )

    u64_kwargs = {
        "remaining_snapshot_count": 20,
        "reference_original_pt_bytes": 1_343_234_243,
        "reference_derivative_bytes": 38_396_880,
        "reference_manifest_bytes": 93_518,
        "reference_actor_bytes": 6_562_549,
        "reference_json_bytes": 20_031,
    }
    compact_proj = compute_storage_projection(
        tmp_path / "attempt-15-c1/output",
        snapshot_format="compact",
        **u64_kwargs,
    )
    assert compact_proj["status"] == "PASS"
    assert compact_proj["remaining_exports_nominal_bytes"] < 1_000_000_000
    assert compact_proj["under_1gb_nominal_for_remaining_exports"] is True
    assert compact_proj["export_byte_reduction_factor"] > 29.0
    assert compact_proj["free_above_safety_margin_at_temp_peak_gib"] > 6.0

    # Raw 1.34 GB periodic snapshots (~27.0 GB exports) fail closed on 13 GiB free.
    with pytest.raises(ContinuationRefused, match="insufficient Mac storage"):
        compute_storage_projection(
            tmp_path / "attempt-15-c1-raw/output",
            snapshot_format="raw",
            **u64_kwargs,
        )

    # Low-disk filesystem (< required_conservative_with_margin) fails closed even for compact.
    free_5_gib = 5 * 1024**3
    monkeypatch.setattr(
        training_continuation.shutil,
        "disk_usage",
        lambda _path: DiskUsage(total_bytes, total_bytes - free_5_gib, free_5_gib),
    )
    with pytest.raises(ContinuationRefused, match="insufficient Mac storage"):
        compute_storage_projection(
            tmp_path / "attempt-15-c1-low/output",
            snapshot_format="compact",
            **u64_kwargs,
        )


def test_continuation_builder_admits_receipt_less_host_enospc_and_chains_compact_at_source_c2(
    fixture_workspace: Path, tmp_path: Path
) -> None:
    """End-to-end receipt-less ENOSPC c1 builder + compact-at-source c1 -> c2 continuation."""
    from research.sandbox_a_resident.training_compact import publish_compact_snapshot
    from research.sandbox_a_resident.training_continuation import (
        build_host_enospc_recovery_certificate,
    )
    from research.sandbox_a_resident.training_resume import (
        ACCOUNT_RELEASE_CAPTURE_SCHEMA,
        HOST_ENOSPC_FAILURE_CLASS,
        HOST_ENOSPC_RECOVERY_SCHEMA,
    )

    repo = fixture_workspace / "snake-dqn-fixture"
    artifacts = fixture_workspace / "snake-dqn-artifacts/sandbox-a-resident"
    source_packet = repo / "research/sandbox_a_resident/packet-train01-15"

    enospc_attempt = _populate_attempt(
        artifacts / "train01-attempt-15",
        source_packet,
        [0, 16, 32, 48, 64],
        receipt_overrides={},
    )
    # Remove receipt.json to reproduce the live receipt-less host ENOSPC failure.
    (enospc_attempt / "output/receipt.json").unlink()
    assert not (enospc_attempt / "output/receipt.json").exists()

    # Without --recovery-evidence, build_continuation_packet refuses missing receipt.json.
    with pytest.raises(ContinuationRefused, match="missing or symlinked source attempt evidence"):
        build_continuation_packet(
            enospc_attempt,
            64,
            slug="train01-15-c1-noev",
            repo_root=repo,
            checkout_parent=fixture_workspace,
            artifacts_root=artifacts,
            date_tag="20261028",
            skip_dry_run=True,
        )

    exec_manifest = json.loads((enospc_attempt / "output/manifest.json").read_text())
    job_id = exec_manifest["job_id"]
    gen = "59e68cce673b4f90a8c2e1d0b7c6a5f4"
    session = f"snake-{job_id}-{gen[:12]}"
    endpoint = "m-s-enospc64"
    state_dir = enospc_attempt / "state"
    state_dir.mkdir(parents=True)
    assignment = {
        "endpoint": endpoint,
        "generation": gen,
        "session": session,
        "output": str((enospc_attempt / "output").resolve()),
    }
    (state_dir / "assignment.json").write_text(json.dumps(assignment, sort_keys=True) + "\n")

    account_root = tmp_path / "account/snake"
    account_root.mkdir(parents=True)
    (account_root / f"active-{session}.json").write_text(
        json.dumps(assignment, sort_keys=True) + "\n"
    )

    for idx in range(1, 11):
        (enospc_attempt / f"output/poll-{idx:06d}.json").write_text(
            json.dumps(
                {
                    "job": job_id,
                    "generation": gen,
                    "status": "running",
                    "job_log_tail": (
                        f'TRAIN01_PROGRESS {{"stage": "ppo", "iteration": {idx * 6}}}\n'
                    ),
                    "exports": [],
                }
            )
            + "\n"
        )

    sup_log = fixture_workspace / "snake-dqn-artifacts/ops/logs" / f"{job_id}.log"
    sup_log.parent.mkdir(parents=True, exist_ok=True)
    enospc_output_resolved = (enospc_attempt / "output").resolve()
    sup_lines = (
        ["PREFLIGHT_OK"]
        + [
            json.dumps(
                {
                    "event": "poll",
                    "job": job_id,
                    "job_log_tail": (
                        f'TRAIN01_PROGRESS {{"stage": "ppo", "iteration": {idx * 6}}}\n'
                    ),
                }
            )
            for idx in range(1, 11)
        ]
        + [
            f"[Errno 28] No space left on device: '{enospc_output_resolved}/.receipt-ihix0kmn'",
            "EXIT=1",
        ]
    )
    sup_log.write_text("\n".join(sup_lines) + "\n")

    recovery_dir = tmp_path / "enospc-recovery-evidence"
    recovery_dir.mkdir(parents=True)
    capture_file = recovery_dir / "account-release-status.json"
    capture_payload = {
        "schema": ACCOUNT_RELEASE_CAPTURE_SCHEMA,
        "captured_at_utc": "2026-10-28T13:15:00Z",
        "captured_unix_seconds": sup_log.stat().st_mtime + 5.0,
        "status": {
            "sessions": {},
            "usage": {"assignments": 0, "balance": 1080.0, "rate": 0.0},
            "unresolved": [],
        },
    }
    publish(capture_file, lambda dst: dst.write_text(json.dumps(capture_payload) + "\n"))

    (account_root / f"active-{session}.json").unlink()
    recovered = enospc_attempt / "output/recovery-qualified.json"
    recovered.write_text(json.dumps({"endpoint": endpoint, "stop_verified": True}) + "\n")
    # Freeze derivatives before inventory/certificate creation; later additions
    # to the retained evidence set must invalidate admission.
    encode_or_verify_compact_snapshot(
        enospc_attempt / "output/exports/training/iteration-0064.pt",
        source_packet_id="packet-train01-15",
        iteration=64,
        source_commit=json.loads((source_packet / "control.json").read_text())["source_commit"],
    )
    cert_file = recovery_dir / "host-enospc-recovery-certificate.json"
    cert_summary = build_host_enospc_recovery_certificate(
        enospc_attempt,
        account_status_capture_path=capture_file,
        certificate_output_path=cert_file,
        recover_receipt_path=recovered,
        supervisor_log_path=sup_log,
        account_root=account_root,
        source_packet=source_packet,
        repo_root=repo,
    )
    assert cert_summary["failure_class"] == HOST_ENOSPC_FAILURE_CLASS
    assert cert_summary["original_receipt_present"] is False

    c1_result = build_continuation_packet(
        enospc_attempt,
        64,
        slug="train01-15-c1-enospc64",
        job_id="sandbox-a-train01-20261028-15-c1-enospc64",
        repo_root=repo,
        checkout_parent=fixture_workspace,
        artifacts_root=artifacts,
        date_tag="20261028",
        skip_dry_run=True,
        launch_runs=1,
        recovery_evidence=cert_file,
        supervisor_log=sup_log,
    )
    assert c1_result["status"] == "PREPARED_NOT_DISPATCHED"
    assert c1_result["allocations"] == 0
    assert c1_result["prior_stop_evidence_kind"] == "host_enospc_recovery_certificate"
    assert c1_result["prior_receipt_present"] is False
    assert not (enospc_attempt / "output/receipt.json").exists()

    c1_packet_dir = Path(c1_result["packet_dir"])
    assert (c1_packet_dir / "recovery-evidence.json").is_file()
    c1_seal = json.loads((c1_packet_dir / "SEAL.json").read_text())
    assert c1_seal["prior_stop_evidence_kind"] == "host_enospc_recovery_certificate"
    assert c1_seal["prior_receipt_present"] is False
    assert c1_seal["prior_receipt_sha256"] is None
    assert c1_seal["prior_failure_class"] == HOST_ENOSPC_FAILURE_CLASS
    assert c1_seal["prior_recovery_evidence_schema"] == HOST_ENOSPC_RECOVERY_SCHEMA
    assert c1_seal["prior_recovery_evidence_sha256"] == digest(
        c1_packet_dir / "recovery-evidence.json"
    )
    assert c1_seal["prior_release_proof"]["original_cleanup_error"] == "UNKNOWN_UNAVAILABLE"
    assert c1_seal["prior_release_proof"]["post_hoc_release_verified"] is True
    assert c1_seal["periodic_export_format"] == "compact"
    assert c1_seal["mac_storage_projection"]["status"] == "PASS"

    # Review continuation3 preflight-revalidation probe: a valid seal cannot
    # hide expired release evidence or a newly retained fault at dispatch time.
    import sys

    from research.sandbox_a_resident.training_continuation import (
        _single_thread_process_env,
        run_packet_validation,
    )

    stale = subprocess.run(
        [
            sys.executable,
            "-c",
            "import time; from pathlib import Path; "
            "from research.sandbox_a_resident.training_preflight import validate_packet; "
            f"time.time = lambda: {capture_payload['captured_unix_seconds'] + 172800!r}; "
            f"validate_packet(Path({str(c1_packet_dir)!r}))",
        ],
        cwd=Path(c1_result["broker_checkout"]),
        env=_single_thread_process_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert stale.returncode != 0 and "stale account_status_capture" in stale.stderr
    committed_certificate = c1_packet_dir / "recovery-evidence.json"
    saved_certificate = committed_certificate.read_bytes()
    committed_certificate.unlink()
    with pytest.raises(ContinuationRefused, match="admission proof is missing"):
        run_packet_validation(Path(c1_result["broker_checkout"]), c1_packet_dir)
    seal_path = c1_packet_dir / "SEAL.json"
    saved_seal = seal_path.read_bytes()
    for mutation in ("missing", "dangling", "receipt", "unknown"):
        if mutation == "missing":
            seal_path.unlink()
        elif mutation == "dangling":
            seal_path.unlink()
            seal_path.symlink_to(c1_packet_dir / "absent-seal.json")
        else:
            seal_path.write_text(
                json.dumps({**json.loads(saved_seal), "prior_stop_evidence_kind": mutation})
            )
        with pytest.raises(ContinuationRefused):
            run_packet_validation(Path(c1_result["broker_checkout"]), c1_packet_dir)
        if seal_path.is_symlink():
            seal_path.unlink()
        seal_path.write_bytes(saved_seal)
    committed_certificate.write_bytes(saved_certificate)
    late_fault = enospc_attempt / "output/exports/training/job.log.partial"
    late_fault.write_text("FloatingPointError: non-finite loss tripwire\n")
    with pytest.raises(ContinuationRefused, match="inventory/hashes drifted"):
        run_packet_validation(Path(c1_result["broker_checkout"]), c1_packet_dir)
    late_fault.unlink()

    # Now simulate c1 running on Colab with compact-at-source periodic exports (zero raw .pt files)
    # and stopping mechanically at iteration-0128, then build c2 from c1's broker_checkout!
    c1_broker_repo = Path(c1_result["broker_checkout"])
    c1_control = json.loads((c1_packet_dir / "control.json").read_text())
    c1_exec_manifest = json.loads((c1_packet_dir / "dispatch-manifest.json").read_text())
    c1_attempt = artifacts / "train01-attempt-15-c1-compactsrc"
    c1_exports = c1_attempt / "output/exports/training"
    c1_exports.mkdir(parents=True)
    c1_exec_manifest["output_dir"] = str((c1_attempt / "output").resolve())
    c1_exec_path = c1_attempt / "output/manifest.json"
    c1_exec_path.write_text(json.dumps(c1_exec_manifest, indent=2, sort_keys=True) + "\n")

    state_128 = {
        "schema": "sandbox-a-train-state/v1",
        "iteration": 128,
        "actor": {"weight": torch.arange(16, dtype=torch.float32).reshape(4, 4) + 128},
        "trainer": {"actor_rows": 128 * 1024 * 256},
        "preparation": {"stage": "ready", "temperature": 0.00843973},
    }
    publish_compact_snapshot(
        c1_exports,
        128,
        lambda path: torch.save(state_128, path),
        metadata={
            "allocation": allocation(c1_control),
            "source_commit": c1_control["source_commit"],
            "prepare_phase": "ready",
        },
    )
    publish(
        c1_exports / "iteration-0128-actor.pth",
        lambda path: path.write_bytes(b"actor-0128"),
    )
    assert not (c1_exports / "iteration-0128.pt").exists()
    c1_receipt = {
        "job_id": c1_exec_manifest["job_id"],
        "commit": c1_exec_manifest["commit"],
        "generation": "71ffadd059e68cce673b4f90a8c2e1d0",
        "manifest_sha256": digest(c1_exec_path),
        "status": "INCOMPLETE",
        "failure_class": "RPC_EXEC_POLL_TIMEOUT",
        "terminal": None,
        "stop_verified": True,
        "cleanup_error": None,
    }
    (c1_attempt / "output/receipt.json").write_text(
        json.dumps(c1_receipt, indent=2, sort_keys=True) + "\n"
    )

    c2_result = build_continuation_packet(
        c1_attempt,
        128,
        source_packet=c1_packet_dir,
        slug="train01-15-c2-compact128",
        job_id="sandbox-a-train01-20261029-15-c2-compact128",
        repo_root=c1_broker_repo,
        checkout_parent=fixture_workspace,
        artifacts_root=artifacts,
        date_tag="20261029",
        skip_dry_run=True,
        launch_runs=1,
    )
    assert c2_result["status"] == "PREPARED_NOT_DISPATCHED"
    assert c2_result["completion_attempt_index"] == 2
    assert c2_result["resume_from_iteration"] == 128
    c2_manifest = json.loads((Path(c2_result["packet_dir"]) / "dispatch-manifest.json").read_text())
    c2_dests = {row["dest"] for row in c2_manifest["inputs"]}
    assert "resume/iteration-0128.pt.xz" in c2_dests
    assert "resume/iteration-0128.pt.xz.manifest" in c2_dests
    assert "resume/iteration-0128.pt" not in c2_dests
    assert "resume/iteration-0128.pt.sha256" not in c2_dests
    assert "resume/iteration-0128.pt.complete" not in c2_dests


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("packet_family", ["s2-01", "s2-02", "s2-03"])
def test_second_seed_preference_continuation_preserves_domain_and_whole_timeout_price(
    fixture_workspace: Path, fallback: bool, packet_family: str
) -> None:
    """Both admitted GPU identities can continue S2 with the same audited recipe."""
    from research.compute.gpu_preference import prepare_preference, variants
    from research.sandbox_a_resident.training_task import seed_allocations
    from research.sandbox_a_resident.training_units import empty_ledger
    from tools.sandbox.registry import world_seed

    repo = fixture_workspace / "snake-dqn-fixture"
    packet = repo / f"research/sandbox_a_resident/packet-train01-{packet_family}"
    if not packet.exists():
        shutil.copytree(packet.with_name("packet-train01-16"), packet)
        control = json.loads((packet / "control.json").read_text())
        control.update(allocation_index=2, train_seed_offset=2_000_000_000)
        (packet / "control.json").write_text(json.dumps(control))
        (packet / "verified-ledger.json").write_text(json.dumps(empty_ledger(control)))
        allocations = seed_allocations(control)
        audit = json.loads((packet / "registry-check.json").read_text())
        audit["allocations"] = allocations
        (packet / "registry-check.json").write_text(json.dumps(audit))
        manifest_path = packet / "dispatch-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["job_id"] = f"sandbox-a-train01-20261028-{packet_family}"
        manifest["env"].update(
            SANDBOX_A_CONTROL_JSON=json.dumps(control),
            SANDBOX_A_SNAPSHOT_FORMAT="compact",
            SANDBOX_A_PACKET_ID=packet.name,
        )
        manifest["sandbox_a"].update(
            seed_allocations=allocations, seed_audit_sha256=digest(packet / "registry-check.json")
        )
        prepare_preference(manifest)
        manifest_path.write_text(json.dumps(manifest))
        sandbox = json.loads((packet / "sandbox.json").read_text())
        worlds = next(row for row in sandbox["inputs"] if row["kind"] == "worlds")
        worlds["indices"] = [row["offset"] for row in allocations]
        worlds["seeds"] = [world_seed("sandbox-a-train01", index) for index in worlds["indices"]]
        _write_ten_line_sandbox(packet / "sandbox.json", sandbox)
        _rebind_packet_to_fixture_repo(repo, packet.name)
        subprocess.run(["git", "-C", str(repo), "add", str(packet)], check=True)
        subprocess.run(
            ["git", "-C", str(repo), "commit", "-qm", "S2 preference fixture"], check=True
        )
    artifacts = fixture_workspace / "snake-dqn-artifacts/sandbox-a-resident"
    suffix = "fallback" if fallback else "primary"
    attempt = _populate_attempt(
        artifacts / f"{packet_family}-{suffix}", packet, [64], receipt_overrides={}
    )
    if fallback:
        executed_path = attempt / "output/manifest.json"
        selected = variants(json.loads(executed_path.read_text()))[1]
        executed_path.write_text(json.dumps(selected))
        receipt_path = attempt / "output/receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt.update(job_id=selected["job_id"], manifest_sha256=digest(executed_path))
        receipt_path.write_text(json.dumps(receipt))
    result = build_continuation_packet(
        attempt,
        64,
        source_packet=packet,
        repo_root=repo,
        checkout_parent=fixture_workspace,
        artifacts_root=artifacts,
        slug=f"train01-{packet_family}-c1-i0064-{suffix}",
        skip_dry_run=True,
    )
    successor = Path(result["packet_dir"])
    manifest = json.loads((successor / "dispatch-manifest.json").read_text())
    control = json.loads((successor / "control.json").read_text())
    assert control["allocation_index"] == 2 and control["train_seed_offset"] == 2_000_000_000
    assert manifest["requirements"]["prefer"] == ["H100", "A100"]
    assert manifest["env"]["SANDBOX_A_PACKET_ID"] == successor.name
    assert manifest["budget"]["unit_cap"] == manifest["timeout_minutes"] * 12 / 60
    assert variants(manifest)[1]["budget"]["unit_cap"] == manifest["timeout_minutes"] * 8 / 60
    if packet_family in {"s2-02", "s2-03"}:
        assert manifest["requirements"]["est_minutes"] == 690
        assert manifest["timeout_minutes"] == 1380
        assert manifest["budget"]["unit_cap"] == 276
        assert variants(manifest)[1]["budget"]["unit_cap"] == 184
        assert result["remaining_envelope"]["full_remaining_est_minutes"] > 690
        assert result["remaining_envelope"]["may_require_further_completion"] is True
