"""Preflight tests for `research.v3_ppo.prepare`, `job.sh`, VRAM refusals, and CLI exit codes."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from research.arena_ppo.runtime import sha256_file, verify_completed_artifact
from research.colab_cli import canary as colab_canary
from research.colab_cli import schema as colab_schema
from research.colab_cli import source as colab_source
from research.colab_cli import supervisor as colab_supervisor
from research.v3_ppo.ppo import main as ppo_main
from research.v3_ppo.prepare import (
    ROOT,
)
from research.v3_ppo.prepare import main as prepare_main
from research.v3_ppo.prepare import (
    prepare_v3_ppo_packet,
)
from research.v3_ppo.ruler import (
    CANARY_REQUIRED_EXPORTS,
    RULER_REQUIRED_EXPORTS,
    TRAIN_REQUIRED_EXPORTS,
)
from research.v3_ppo.ruler import main as ruler_main
from research.v3_ppo.ruler import (
    run_canary_suite,
)
from tools.ops import dispatch as ops_dispatch


def _stage_manifest_inputs(mf: dict[str, Any], remote_inputs_dir: Path) -> Path:
    remote_inputs_dir.mkdir(parents=True, exist_ok=True)
    for inp in mf["inputs"]:
        dst = remote_inputs_dir / str(inp["dest"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inp["path"], dst)
    return remote_inputs_dir


def _build_synthetic_canary_receipt(
    canary_mf: dict[str, Any],
    gpu_kind: str,
    *,
    shared_exports_src: Path | None = None,
) -> Path:
    """Create a broker-verified canary receipt from real `run_canary_suite` exports."""
    canary_out = Path(canary_mf["output_dir"])
    canary_exports_dir = canary_out / "exports"
    remote_in = _stage_manifest_inputs(canary_mf, canary_out / "remote_inputs")

    if shared_exports_src is not None:
        shutil.copytree(shared_exports_src, canary_exports_dir)
    else:
        run_canary_suite(
            output_dir=canary_exports_dir,
            device="cpu",
            input_dir=remote_in,
            ledger_path=remote_in / "verified-ledger.json",
            canary_envs=2,
            canary_arena_radius=60.0,
            canary_num_snakes=6,
            canary_rollout_len=4,
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
                    "cuda_decisions": 12,
                    "device": "cuda:0",
                    "package": "research/v3_ppo",
                    "caller": "research/v3_ppo/ruler.py:_verify_cardinal6_canary_decisions",
                    "operation": "Tensor.argmax",
                    "shape": [2, 6, 6],
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


def test_prepare_canary_ruler_and_train_validate_with_colab_cli(tmp_path: Path) -> None:
    """Validate L4 and A100 canary, ruler, and train packets with all colab_cli validators."""
    for rel in (*CANARY_REQUIRED_EXPORTS, *RULER_REQUIRED_EXPORTS, *TRAIN_REQUIRED_EXPORTS):
        assert not rel.endswith(".sha256")
        assert not rel.endswith(".complete")

    canary_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_canary_l4",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_canary_l4",
        allow_unmerged_commit=True,
    )
    canary_a100 = prepare_v3_ppo_packet(
        tmp_path / "pkt_canary_a100",
        commit="HEAD",
        mode="canary",
        gpu="A100",
        output_dir=tmp_path / "out_canary_a100",
        allow_unmerged_commit=True,
    )
    mf_c_l4 = json.loads(Path(canary_l4["manifest_path"]).read_text(encoding="utf-8"))
    mf_c_a100 = json.loads(Path(canary_a100["manifest_path"]).read_text(encoding="utf-8"))

    for mf in (mf_c_l4, mf_c_a100):
        assert mf["timeout_minutes"] <= colab_canary.MAX_CANARY_MINUTES
        assert mf["env"]["SNAKE_COLAB_CANARY"] == "1"
        assert mf["env"]["SNAKE_COLAB_CANARY_PACKAGE"] == "research/v3_ppo"
        assert mf["env"]["V3_PPO_MODE"] == "canary"
        colab_schema.validate_manifest(mf)
        ops_dispatch.require_resume_policy(mf)
        colab_supervisor.validate(mf)
        colab_canary.check(mf, ROOT)

    # Verify sparse source archive excludes legacy grid sim & old obs modules!
    source_plan = colab_source.prepare(ROOT, mf_c_l4, tmp_path / "source_plan_l4")
    files_set = set(source_plan["files"])
    assert "research/v3_ppo/job.sh" in files_set
    assert "research/v3_ppo/ppo.py" in files_set
    assert "research/v3_ppo/ruler.py" in files_set
    assert "research/game_v3/sim.py" in files_set
    assert "research/arena_ppo/runtime.py" in files_set
    assert "research/arena_ppo/sim.py" not in files_set
    assert "research/arena_ppo/obs.py" not in files_set
    assert "research/arena_ppo/net.py" not in files_set
    assert "research/arena_ppo/ppo.py" not in files_set
    assert not any(
        name.startswith(("src/", "web/", "research/gpu_sim_spike/")) for name in files_set
    )

    rcpt_l4 = _build_synthetic_canary_receipt(mf_c_l4, "L4")
    l4_exports = Path(mf_c_l4["output_dir"]) / "exports"
    rcpt_a100 = _build_synthetic_canary_receipt(mf_c_a100, "A100", shared_exports_src=l4_exports)

    # Bind ruler & default train on L4, and E=256 train on A100
    ruler_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_ruler_l4",
        commit="HEAD",
        mode="ruler",
        gpu="L4",
        canary_receipt=rcpt_l4,
        output_dir=tmp_path / "out_ruler_l4",
        allow_unmerged_commit=True,
    )
    train_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_train_l4",
        commit="HEAD",
        mode="train",
        gpu="L4",
        hours=2.0,
        expected_steps_per_sec=25000.0,
        canary_receipt=rcpt_l4,
        output_dir=tmp_path / "out_train_l4",
        allow_unmerged_commit=True,
    )
    train_a100 = prepare_v3_ppo_packet(
        tmp_path / "pkt_train_a100",
        commit="HEAD",
        mode="train",
        gpu="A100",
        num_envs=256,
        net_size="M",
        hours=2.0,
        canary_receipt=rcpt_a100,
        output_dir=tmp_path / "out_train_a100",
        allow_unmerged_commit=True,
    )

    for summary, expected_mode in (
        (ruler_l4, "ruler"),
        (train_l4, "train"),
        (train_a100, "train"),
    ):
        mf = json.loads(Path(summary["manifest_path"]).read_text(encoding="utf-8"))
        assert "SNAKE_COLAB_CANARY" not in mf["env"]
        assert mf["env"]["V3_PPO_MODE"] == expected_mode
        colab_schema.validate_manifest(mf)
        ops_dispatch.require_resume_policy(mf)
        colab_supervisor.validate(mf)
        colab_canary.check(mf, ROOT)


def test_prepare_refuses_bad_configs_and_oom_vram_budgets(tmp_path: Path) -> None:
    """Verify `prepare` refuses VRAM overflows, missing/wrong receipts, T4, and bad configs."""
    # 1. E=256 x 24 learners x T=32 (~26.7 GB) MUST be refused on 22 GB L4, accepted on 40 GB A100
    with pytest.raises(ValueError, match="exceeds L4 memory budget"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_oom_l4",
            commit="HEAD",
            mode="train",
            gpu="L4",
            num_envs=256,
            learner_slots=24,
            rollout_len=32,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_oom_l4",
            allow_unmerged_commit=True,
        )
    assert not (tmp_path / "pkt_oom_l4").exists(), "Failed prepare must clean up packet_dir!"

    # 2. E=512 x 24 learners x T=32 (~51.2 GB) MUST be refused on 40 GB A100, accepted on 80 GB H100
    with pytest.raises(ValueError, match="exceeds A100 memory budget"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_oom_a100",
            commit="HEAD",
            mode="train",
            gpu="A100",
            num_envs=512,
            learner_slots=24,
            rollout_len=32,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_oom_a100",
            allow_unmerged_commit=True,
        )

    ok_h100 = prepare_v3_ppo_packet(
        tmp_path / "pkt_ok_h100",
        commit="HEAD",
        mode="train",
        gpu="H100",
        num_envs=512,
        learner_slots=24,
        rollout_len=32,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_ok_h100",
        allow_unmerged_commit=True,
    )
    assert ok_h100["memory_estimate"]["fits_gpu_budget"] is True

    # 3. Ruler packet where zero cells fit on 22 GB L4 (`E=1024`) is refused
    with pytest.raises(ValueError, match="zero requested ruler cells fit"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_oom_ruler_l4",
            commit="HEAD",
            mode="ruler",
            gpu="L4",
            ruler_envs="1024",
            ruler_learners="24",
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_oom_ruler_l4",
            allow_unmerged_commit=True,
        )

    # 4. Unsupported GPU (`T4`) refused
    with pytest.raises(ValueError, match="T4"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_t4",
            commit="HEAD",
            mode="canary",
            gpu="T4",
            output_dir=tmp_path / "out_t4",
            allow_unmerged_commit=True,
        )

    # 5. Missing `--canary-receipt` refused on ruler & train via both Python API and CLI
    for mode in ("ruler", "train"):
        with pytest.raises(ValueError, match="canary_receipt"):
            prepare_v3_ppo_packet(
                tmp_path / f"pkt_no_rcpt_{mode}",
                commit="HEAD",
                mode=mode,
                gpu="L4",
                output_dir=tmp_path / f"out_no_rcpt_{mode}",
                allow_unmerged_commit=True,
            )
        with pytest.raises(SystemExit):
            prepare_main(
                [
                    "--mode",
                    mode,
                    "--gpu",
                    "L4",
                    "--packet",
                    str(tmp_path / f"cli_no_rcpt_{mode}"),
                    "--allow-unmerged-commit",
                ]
            )

    # 6. Wrong GPU hardware in canary receipt refused (binding L4 receipt to A100 packet)
    c_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_c_l4_hw",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        output_dir=tmp_path / "out_c_l4_hw",
        allow_unmerged_commit=True,
    )
    mf_c_l4 = json.loads(Path(c_l4["manifest_path"]).read_text(encoding="utf-8"))
    rcpt_l4 = _build_synthetic_canary_receipt(mf_c_l4, "L4")
    with pytest.raises(ValueError, match="does not match canary receipt hardware"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_wrong_hw",
            commit="HEAD",
            mode="ruler",
            gpu="A100",
            canary_receipt=rcpt_l4,
            output_dir=tmp_path / "out_wrong_hw",
            allow_unmerged_commit=True,
        )

    # 7. Unmerged commit refused when `allow_unmerged_commit=False`
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
            prepare_v3_ppo_packet(
                tmp_path / "pkt_unmerged",
                commit="HEAD",
                mode="canary",
                gpu="L4",
                output_dir=tmp_path / "out_unmerged",
                allow_unmerged_commit=False,
            )


def test_non_pass_outcomes_exit_nonzero(tmp_path: Path) -> None:
    """Both `ppo` and `ruler` CLIs must return non-zero exit code when status != 'PASS'."""
    out_ppo = tmp_path / "ppo_fail_out"
    rc_ppo = ppo_main(
        [
            "--output-dir",
            str(out_ppo),
            "--work-dir",
            str(tmp_path / "ppo_fail_work"),
            "--device",
            "cpu",
            "--num-envs",
            "2",
            "--num-snakes",
            "6",
            "--arena-radius",
            "60.0",
            "--rollout-len",
            "4",
            "--total-updates",
            "5",
            "--eval-every-updates",
            "0",
            "--max-wall-seconds",
            "0.0001",
        ]
    )
    assert rc_ppo != 0
    train_res_path = out_ppo / "train_result.json"
    verify_completed_artifact(train_res_path)
    train_res = json.loads(train_res_path.read_text(encoding="utf-8"))
    assert train_res["status"] == "FAIL"

    out_ruler = tmp_path / "ruler_fail_out"
    rc_ruler = ruler_main(
        [
            "--output-dir",
            str(out_ruler),
            "--device",
            "cpu",
            "--envs",
            "2",
            "--nets",
            "S",
            "--learners",
            "6",
            "--arena-radius",
            "60.0",
            "--num-snakes",
            "6",
            "--max-wall-seconds",
            "0.0001",
        ]
    )
    assert rc_ruler != 0
    ruler_res_path = out_ruler / "ruler_result.json"
    verify_completed_artifact(ruler_res_path)
    ruler_res = json.loads(ruler_res_path.read_text(encoding="utf-8"))
    assert ruler_res["status"] == "FAIL"
