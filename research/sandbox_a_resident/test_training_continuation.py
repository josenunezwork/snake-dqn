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
