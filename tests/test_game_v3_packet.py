"""Mac preflight tests for `research.game_v3.prepare`, `job.sh`, and `cuda_check` exports."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch

import research.game_v3.cuda_check as cuda_check_mod
from research.arena_ppo.runtime import sha256_file, verify_completed_artifact
from research.colab_cli import canary as colab_canary
from research.colab_cli import schema as colab_schema
from research.colab_cli import source as colab_source
from research.colab_cli import supervisor as colab_supervisor
from research.game_v3.config import GameV3Config
from research.game_v3.cuda_check import (
    CUDA_CHECK_EXPORT_FILENAME,
    CUDA_CHECK_REQUIRED_EXPORTS,
    benchmark_rebase_and_reset_memory,
    build_cuda_check_plan,
    run_cuda_check,
)
from research.game_v3.prepare import (
    CHECK_BASE_MINUTES,
    CHECK_TIMEOUT_MINUTES,
    CHECK_WALL_GUARD_SECONDS,
    ROOT,
)
from research.game_v3.prepare import main as prepare_main
from research.game_v3.prepare import (
    prepare_game_v3_packet,
)
from research.game_v3.scripted import cardinal6_safe_actions
from research.game_v3.sim import GameV3Sim
from tools.ops import dispatch as ops_dispatch


def _stage_manifest_inputs(mf: dict[str, Any], remote_inputs_dir: Path) -> Path:
    """Copy manifest `inputs` into simulated remote `$COLAB_INPUT_DIR`."""
    remote_inputs_dir.mkdir(parents=True, exist_ok=True)
    for inp in mf["inputs"]:
        dst = remote_inputs_dir / str(inp["dest"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inp["path"], dst)
    return remote_inputs_dir


def _build_synthetic_canary_receipt(
    canary_mf: dict[str, Any],
    gpu_kind: str,
) -> Path:
    """Create a broker-verified canary receipt from real `run_cuda_check` exports."""
    canary_out = Path(canary_mf["output_dir"])
    canary_exports_dir = canary_out / "exports"
    work_json = canary_out / "work" / "cuda_check_live.json"
    _stage_manifest_inputs(canary_mf, canary_out / "remote_inputs")

    run_cuda_check(
        device="cpu",
        parity_envs=4,
        parity_ticks=50,
        det_parity_ticks=25,
        parity_snakes=12,
        parity_arena_radius=120.0,
        rebase_threshold=24,
        bench_envs=(8,),
        bench_snakes_list=(12,),
        learner_slots=4,
        bench_arena_radius=180.0,
        warmup_steps=2,
        bench_steps=4,
        stage_steps=2,
        rebase_reset_envs=8,
        rebase_reset_snakes=12,
        json_output=work_json,
        export_dir=canary_exports_dir,
        require_complete=True,
    )

    saved_probe_mf = canary_out / "manifest.json"
    saved_probe_mf.write_text(
        json.dumps(canary_mf, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (canary_out / "upload-measurement.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 2.0,
                "transfer_seconds": 0.5,
                "source_delivery_seconds": 0.8,
                "staging_completed": True,
            }
        ),
        encoding="utf-8",
    )
    (canary_out / "hardware.json").write_text(
        json.dumps({"hardware": {"kind": gpu_kind}}), encoding="utf-8"
    )

    export_inventory = []
    for fpath in sorted(canary_exports_dir.rglob("*")):
        if fpath.is_file():
            export_inventory.append(
                {
                    "path": fpath.relative_to(canary_exports_dir).as_posix(),
                    "size": fpath.stat().st_size,
                    "sha256": sha256_file(fpath),
                }
            )
    colab_schema.validate_export_inventory(canary_mf, export_inventory)

    transfers = [
        {
            "direction": "upload",
            "remote": f"/content/snake-cli/gen-test-01/inputs/{inp['dest']}",
            "status": "verified",
            "size_bytes": Path(inp["path"]).stat().st_size,
            "sha256": inp["sha256"],
        }
        for inp in canary_mf["inputs"]
    ]
    receipt_obj: dict[str, Any] = {
        "status": "PASS",
        "stop_verified": True,
        "error": None,
        "cleanup_error": None,
        "failure_class": None,
        "commit": canary_mf["commit"],
        "job_id": canary_mf["job_id"],
        "generation": "gen-test-01",
        "manifest_sha256": sha256_file(saved_probe_mf),
        "exports": export_inventory,
        "transfers": transfers,
        "terminal": {
            "job_id": canary_mf["job_id"],
            "generation": "gen-test-01",
            "returncode": 0,
            "status": "completed",
            "export_scanner": {"name": "runtime._exports", "inventory": export_inventory},
            "canary_observations": [
                {
                    "cuda_decisions": 10,
                    "device": "cuda:0",
                    "package": "research/game_v3",
                    "caller": "research/game_v3/scripted.py:cardinal6_safe_actions",
                    "operation": "Tensor.argmax",
                    "shape": [4, 12, 6],
                    "root": "/content/snake-cli/gen-test-01",
                }
            ],
        },
    }
    receipt_obj["canary_evidence"] = colab_canary.evidence(canary_mf, receipt_obj, canary_out, ROOT)
    receipt_path = canary_out / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt_obj, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return receipt_path


def test_prepare_canary_and_check_modes_validate_with_colab_validators(tmp_path: Path) -> None:
    """Validate canary and check packets on L4 & A100 with schema, dispatch, supervisor, canary."""
    for rel in CUDA_CHECK_REQUIRED_EXPORTS:
        assert not rel.endswith(".sha256")
        assert not rel.endswith(".complete")

    # T4 is unsupported and must be rejected.
    with pytest.raises(ValueError, match="T4"):
        prepare_game_v3_packet(
            tmp_path / "pkt_t4_refused",
            commit="HEAD",
            mode="canary",
            gpu="T4",
            output_dir=tmp_path / "out_t4",
            allow_unmerged_commit=True,
        )

    # Unmerged HEAD must be refused when allow_unmerged_commit=False.
    head_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    is_merged = (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", head_sha, "origin/main"],
            cwd=ROOT,
            capture_output=True,
        ).returncode
        == 0
    )
    if not is_merged:
        with pytest.raises(ValueError, match="origin/main"):
            prepare_game_v3_packet(
                tmp_path / "pkt_unmerged_refused",
                commit="HEAD",
                mode="canary",
                gpu="L4",
                output_dir=tmp_path / "out_unmerged_refused",
                allow_unmerged_commit=False,
            )

    # 1. Prepare `canary` packets on L4 and A100.
    canary_l4 = prepare_game_v3_packet(
        tmp_path / "pkt_canary_l4",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_canary_l4",
        allow_unmerged_commit=True,
    )
    canary_a100 = prepare_game_v3_packet(
        tmp_path / "pkt_canary_a100",
        commit="HEAD",
        mode="canary",
        gpu="A100",
        output_dir=tmp_path / "out_canary_a100",
        allow_unmerged_commit=True,
    )
    mf_canary_l4 = json.loads(Path(canary_l4["manifest_path"]).read_text(encoding="utf-8"))
    mf_canary_a100 = json.loads(Path(canary_a100["manifest_path"]).read_text(encoding="utf-8"))

    for mf in (mf_canary_l4, mf_canary_a100):
        assert mf["timeout_minutes"] <= colab_canary.MAX_CANARY_MINUTES
        assert mf["env"]["SNAKE_COLAB_CANARY"] == "1"
        assert mf["env"]["SNAKE_COLAB_CANARY_PACKAGE"] == "research/game_v3"
        assert mf["env"]["GAME_V3_MODE"] == "canary"
        colab_schema.validate_manifest(mf)
        ops_dispatch.require_resume_policy(mf)
        colab_supervisor.validate(mf)
        colab_canary.check(mf, ROOT)

    # Verify sparse source archive contains required runtime files and excludes unrelated code.
    source_plan = colab_source.prepare(ROOT, mf_canary_l4, tmp_path / "source_plan_l4")
    assert "research/game_v3/job.sh" in source_plan["files"]
    assert "research/game_v3/requirements.txt" in source_plan["files"]
    assert "research/game_v3/cuda_check.py" in source_plan["files"]
    assert "research/arena_ppo/runtime.py" in source_plan["files"]
    assert not any(name.startswith("web/") for name in source_plan["files"])

    rcpt_l4 = _build_synthetic_canary_receipt(mf_canary_l4, "L4")
    rcpt_a100 = _build_synthetic_canary_receipt(mf_canary_a100, "A100")

    # Verify canary execution exercised forced rebase_subticks + reset_worlds + rebase_reset stage.
    canary_export_json = Path(mf_canary_l4["output_dir"]) / "exports" / CUDA_CHECK_EXPORT_FILENAME
    assert verify_completed_artifact(canary_export_json)
    canary_payload = json.loads(canary_export_json.read_text(encoding="utf-8"))
    assert canary_payload["status"] == "completed"
    assert canary_payload["parity"]["rebases"] >= 1
    assert canary_payload["parity"]["world_resets"] >= 1
    assert canary_payload["parity_deterministic"]["rebases"] >= 1
    assert canary_payload["parity_deterministic"]["world_resets"] >= 1
    assert canary_payload["rebase_reset_benchmark"]["status"] == "ok"

    # 2. Prepare `check` packets bound to their canary receipts on L4 and A100.
    check_l4 = prepare_game_v3_packet(
        tmp_path / "pkt_check_l4",
        commit="HEAD",
        mode="check",
        gpu="L4",
        canary_receipt=rcpt_l4,
        output_dir=tmp_path / "out_check_l4",
        allow_unmerged_commit=True,
    )
    check_a100 = prepare_game_v3_packet(
        tmp_path / "pkt_check_a100",
        commit="HEAD",
        mode="check",
        gpu="A100",
        canary_receipt=rcpt_a100,
        output_dir=tmp_path / "out_check_a100",
        allow_unmerged_commit=True,
    )
    assert check_l4["check_plan"]["within_l4_budget"] is True
    assert check_l4["check_plan"]["estimated_l4_wall_minutes"] <= 15.0

    mf_check_l4 = json.loads(Path(check_l4["manifest_path"]).read_text(encoding="utf-8"))
    mf_check_a100 = json.loads(Path(check_a100["manifest_path"]).read_text(encoding="utf-8"))
    for mf in (mf_check_l4, mf_check_a100):
        assert "SNAKE_COLAB_CANARY" not in mf["env"]
        assert mf["env"]["GAME_V3_MODE"] == "check"
        assert mf["requirements"]["est_minutes"] == CHECK_BASE_MINUTES
        assert mf["timeout_minutes"] == CHECK_TIMEOUT_MINUTES
        assert mf["env"]["GAME_V3_MAX_WALL_SECONDS"] == str(int(CHECK_WALL_GUARD_SECONDS))
        assert mf["resume_exemption"]["kind"] == "acceptance"
        colab_schema.validate_manifest(mf)
        ops_dispatch.require_resume_policy(mf)
        colab_supervisor.validate(mf)
        colab_canary.check(mf, ROOT)


def test_canary_check_binding_refusals_missing_wrong_gpu_wrong_commit(tmp_path: Path) -> None:
    """Verify check mode refuses missing receipts, wrong GPU receipts, and wrong commit receipts."""
    # 1. Missing receipt refused via Python API and CLI.
    with pytest.raises(ValueError, match="canary_receipt"):
        prepare_game_v3_packet(
            tmp_path / "pkt_missing_arg",
            commit="HEAD",
            mode="check",
            gpu="L4",
            output_dir=tmp_path / "out_missing_arg",
            allow_unmerged_commit=True,
        )
    assert not (tmp_path / "pkt_missing_arg").exists()

    with pytest.raises(SystemExit):
        prepare_main(
            [
                "--packet",
                str(tmp_path / "pkt_cli_missing"),
                "--mode",
                "check",
                "--gpu",
                "L4",
                "--output-dir",
                str(tmp_path / "out_cli_missing"),
                "--allow-unmerged-commit",
            ]
        )

    # 2. Non-existent receipt path refused.
    with pytest.raises(FileNotFoundError, match="Canary receipt not found"):
        prepare_game_v3_packet(
            tmp_path / "pkt_nonexistent_rcpt",
            commit="HEAD",
            mode="check",
            gpu="L4",
            canary_receipt=tmp_path / "does_not_exist_receipt.json",
            output_dir=tmp_path / "out_nonexistent_rcpt",
            allow_unmerged_commit=True,
        )
    assert not (tmp_path / "pkt_nonexistent_rcpt").exists()

    # 3. Passing --canary-receipt to --mode canary is refused.
    canary_l4 = prepare_game_v3_packet(
        tmp_path / "pkt_canary_base",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_canary_base",
        allow_unmerged_commit=True,
    )
    mf_canary_l4 = json.loads(Path(canary_l4["manifest_path"]).read_text(encoding="utf-8"))
    rcpt_l4 = _build_synthetic_canary_receipt(mf_canary_l4, "L4")

    with pytest.raises(ValueError, match="cannot bind --canary-receipt"):
        prepare_game_v3_packet(
            tmp_path / "pkt_canary_with_rcpt",
            commit="HEAD",
            mode="canary",
            gpu="L4",
            canary_receipt=rcpt_l4,
            output_dir=tmp_path / "out_canary_with_rcpt",
            allow_unmerged_commit=True,
        )

    # 4. Wrong GPU (L4 canary receipt passed with --gpu A100) is refused.
    with pytest.raises(ValueError, match="does not match canary receipt hardware"):
        prepare_game_v3_packet(
            tmp_path / "pkt_wrong_gpu",
            commit="HEAD",
            mode="check",
            gpu="A100",
            canary_receipt=rcpt_l4,
            output_dir=tmp_path / "out_wrong_gpu",
            allow_unmerged_commit=True,
        )
    assert not (tmp_path / "pkt_wrong_gpu").exists()

    # 5. Receipt from another commit is refused by colab_canary.check during prepare.
    wrong_commit_dir = tmp_path / "wrong_commit_rcpt_dir"
    shutil.copytree(rcpt_l4.parent, wrong_commit_dir)
    bad_commit_rcpt_path = wrong_commit_dir / "receipt.json"
    bad_rcpt_data = json.loads(bad_commit_rcpt_path.read_text(encoding="utf-8"))
    other_commit = "0" * 40
    bad_rcpt_data["commit"] = other_commit
    bad_rcpt_data["canary_evidence"]["commit"] = other_commit
    bad_commit_rcpt_path.write_text(
        json.dumps(bad_rcpt_data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    with pytest.raises(colab_canary.CanaryError, match="source commit or transport identity"):
        prepare_game_v3_packet(
            tmp_path / "pkt_wrong_commit",
            commit="HEAD",
            mode="check",
            gpu="L4",
            canary_receipt=bad_commit_rcpt_path,
            output_dir=tmp_path / "out_wrong_commit",
            allow_unmerged_commit=True,
        )
    assert not (tmp_path / "pkt_wrong_commit").exists()


def test_job_sh_executes_canary_end_to_end_and_cardinal6_uses_shape6_argmax(
    tmp_path: Path,
) -> None:
    """Run `job.sh` in canary mode on CPU and verify 6-action argmax + sidecar exports."""
    # Verify cardinal6_safe_actions uses Tensor.argmax on shape (E, N, 6) for canary_observer.
    observed_shapes: list[tuple[int, ...]] = []
    orig_argmax = torch.Tensor.argmax

    def _spy_argmax(self: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        observed_shapes.append(tuple(int(v) for v in self.shape))
        return orig_argmax(self, *args, **kwargs)

    sim = GameV3Sim(GameV3Config(arena_radius=90.0, num_snakes=10), num_envs=3, device="cpu")
    torch.Tensor.argmax = _spy_argmax  # type: ignore[method-assign]
    try:
        acts = cardinal6_safe_actions(sim)
    finally:
        torch.Tensor.argmax = orig_argmax  # type: ignore[method-assign]

    assert acts.shape == (3, 10)
    assert (3, 10, 6) in observed_shapes
    assert set(acts.unique().tolist()).issubset({0, 2, 4, 5, 7, 9})

    # Execute research/game_v3/job.sh end-to-end on CPU in simulated Colab runtime root.
    rt_root = tmp_path / "colab_rt"
    in_dir = rt_root / "inputs"
    out_dir = rt_root / "output"
    in_dir.mkdir(parents=True)
    out_dir.mkdir(parents=True)
    (in_dir / "verified-ledger.json").write_text(
        json.dumps({"schema": "game-v3-verified-ledger/v1", "verified_units": []}) + "\n",
        encoding="utf-8",
    )

    env = dict(os.environ)
    env.update(
        {
            "COLAB_CLI_ROOT": str(rt_root),
            "COLAB_INPUT_DIR": str(in_dir),
            "GAME_V3_SOURCE_COMMIT": "HEAD",
            "GAME_V3_MODE": "canary",
            "SNAKE_DQN_DEVICE": "cpu",
            "GAME_V3_PARITY_ENVS": "2",
            "GAME_V3_PARITY_TICKS": "24",
            "GAME_V3_DET_PARITY_TICKS": "20",
            "GAME_V3_PARITY_SNAKES": "8",
            "GAME_V3_BENCH_ENVS": "4",
            "GAME_V3_BENCH_SNAKES": "8",
            "GAME_V3_REBASE_RESET_ENVS": "4",
            "GAME_V3_REBASE_RESET_SNAKES": "8",
            "PYTHONPATH": str(ROOT),
            "PATH": f"{Path(sys.executable).parent}:{env.get('PATH', '')}",
        }
    )
    proc = subprocess.run(
        ["bash", str(ROOT / "research/game_v3/job.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"job.sh failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"

    exported_main = out_dir / CUDA_CHECK_EXPORT_FILENAME
    assert verify_completed_artifact(exported_main)
    stage_jsons = sorted((out_dir / "stages").glob("*.json"))
    assert len(stage_jsons) == 4
    for sj in stage_jsons:
        assert verify_completed_artifact(sj)


def test_cuda_check_non_oom_failure_writes_failed_status_and_wall_guard_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task B.1, B.2 & Task A: failed status + re-raise, 256-world plan/stage, wall guard."""
    # 1. Default plan includes 256-world R=600 rebase/reset stage and early-game memory note.
    plan = build_cuda_check_plan(
        device=torch.device("cuda"),
        parity_envs=32,
        parity_ticks=1000,
        det_parity_ticks=250,
        parity_snakes=24,
        parity_arena_radius=180.0,
        rebase_threshold=400,
        bench_envs=(64, 256),
        bench_snakes_list=(24, 64),
        learner_slots=8,
        bench_arena_radius=600.0,
        warmup_steps=10,
        bench_steps=50,
        stage_steps=20,
    )
    assert plan["within_l4_budget"] is True
    assert "early-game memory" in plan["memory_measurement_note"]
    rr_phases = [p for p in plan["phases"] if p["phase"] == "bench_rebase_reset_N24_E256"]
    assert len(rr_phases) == 1
    assert rr_phases[0]["num_envs"] == 256
    assert rr_phases[0]["arena_radius"] == 600.0

    # Verify benchmark_rebase_and_reset_memory directly on CPU.
    rr_res = benchmark_rebase_and_reset_memory(
        GameV3Config(arena_radius=90.0, num_snakes=8),
        num_envs=4,
        device=torch.device("cpu"),
        warmup_steps=2,
    )
    assert rr_res["status"] == "ok"
    assert rr_res["rebase_subticks"]["shift_applied"] > 0
    assert rr_res["rebase_subticks"]["max_memory_allocated_after_bytes"] > 0
    assert rr_res["reset_worlds"]["worlds_reset"] == 2
    assert rr_res["reset_worlds"]["max_memory_allocated_after_bytes"] > 0
    assert "early-game memory" in rr_res["memory_measurement_note"]

    # 2. Task B.1: Non-OOM failure writes status="failed" + error text, then re-raises.
    def _boom_bench(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("synthetic non-OOM benchmark kernel bug")

    monkeypatch.setattr(cuda_check_mod, "benchmark_single_grid", _boom_bench)
    fail_json = tmp_path / "failed_live.json"
    fail_export_dir = tmp_path / "failed_exports"
    with pytest.raises(RuntimeError, match="synthetic non-OOM benchmark kernel bug"):
        run_cuda_check(
            device="cpu",
            parity_envs=2,
            parity_ticks=6,
            det_parity_ticks=6,
            parity_snakes=8,
            parity_arena_radius=90.0,
            bench_envs=(4,),
            bench_snakes_list=(8,),
            bench_arena_radius=90.0,
            warmup_steps=1,
            bench_steps=2,
            stage_steps=1,
            rebase_reset_envs=0,
            json_output=fail_json,
            export_dir=fail_export_dir,
        )

    failed_payload = json.loads(fail_json.read_text(encoding="utf-8"))
    assert failed_payload["status"] == "failed"
    assert failed_payload["failed_stage"] == "bench_N8_E4"
    assert failed_payload["error_type"] == "RuntimeError"
    assert "synthetic non-OOM benchmark kernel bug" in failed_payload["error"]
    assert failed_payload["parity"]["passed"] is True
    assert verify_completed_artifact(fail_export_dir / CUDA_CHECK_EXPORT_FILENAME)

    # 3. Cooperative wall guard saves partial results cleanly with .sha256 + .complete sidecars.
    monkeypatch.undo()
    wall_json = tmp_path / "wall_live.json"
    wall_export_dir = tmp_path / "wall_exports"
    partial_payload = run_cuda_check(
        device="cpu",
        parity_envs=2,
        parity_ticks=6,
        det_parity_ticks=6,
        parity_snakes=8,
        parity_arena_radius=90.0,
        bench_envs=(4,),
        bench_snakes_list=(8,),
        bench_arena_radius=90.0,
        warmup_steps=1,
        bench_steps=2,
        stage_steps=1,
        rebase_reset_envs=4,
        rebase_reset_snakes=8,
        json_output=wall_json,
        export_dir=wall_export_dir,
        max_wall_seconds=0.0,
    )
    assert partial_payload["status"] == "stopped_wall_guard"
    assert partial_payload["parity"]["passed"] is True
    assert partial_payload["parity_deterministic"] is None
    assert verify_completed_artifact(wall_export_dir / CUDA_CHECK_EXPORT_FILENAME)
