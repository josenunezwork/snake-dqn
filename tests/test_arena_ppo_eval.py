"""Hermetic tests for `research.arena_ppo.evaluate` and `--mode eval` Colab packets."""

from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from research.arena_ppo.evaluate import (
    B8_S3_SHA256,
    DEFAULT_EVAL_BURNIN,
    DEFAULT_EVAL_TICKS,
    DEFAULT_EVAL_WORLDS,
    DEFAULT_LIFE_AGE_BIN_EDGES,
    DEFAULT_WORLD_AGE_BIN_EDGES,
    EVAL_SCHEMA,
    B8OpponentController,
    _SideMetricsAccumulator,
    act_arena_policy_chunked,
    attribute_step_deaths_and_kills,
    build_learner_seat_mask,
    compute_b8_observations_and_actions,
    evaluate_policy,
    load_arena_ppo_policy,
)
from research.arena_ppo.evaluate import main as evaluate_cli_main
from research.arena_ppo.evaluate import (
    resolve_b8_checkpoint_path,
    resolve_eval_start_config,
    scripted_random_safe_actions,
)
from research.arena_ppo.ppo import ArenaPPOConfig, ArenaPPOTrainer, StopRequestedError
from research.arena_ppo.prepare import (
    CANARY_REQUIRED_EXPORTS,
    EVAL_REQUIRED_EXPORTS,
    ROOT,
    prepare_arena_ppo_packet,
)
from research.arena_ppo.ruler import run_canary_suite
from research.arena_ppo.runtime import sha256_file, verify_completed_artifact
from research.arena_ppo.sim import TrainSim, arena_v2_config, derive_arena_world_seeds
from research.colab_cli import canary as colab_canary
from research.colab_cli import schema as colab_schema
from research.colab_cli import supervisor as colab_supervisor
from research.colab_cli.runtime import _exports as supervisor_scan_exports
from research.gpu_sim_spike.device_sim import DeviceSim
from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
from src.simd_env.batch_sim import DEATH_BODY, DEATH_HEAD, DEATH_SELF, DEATH_WALL
from src.simd_env.ego2s_policy import Ego2sSimdPolicy, ego_config_for, student_q
from src.simd_env.ego_raster import build_ego_raster
from src.simd_env.grid_sim import GridBatchSim
from tools.ops import dispatch as ops_dispatch


def _save_synthetic_arena_checkpoints(tmp_dir: Path) -> dict[str, Path]:
    """Create tiny hermetic Arena PPO checkpoints (`stage1/v1` and `eval-weights/v1`)."""
    out: dict[str, Path] = {}
    for idx, arch in enumerate(("ego_only", "shared_trunk", "per_seat")):
        cfg = ArenaPPOConfig(
            net_arch=arch,  # type: ignore[arg-type]
            net_size="S",
            num_envs=2,
            rollout_len=4,
            ppo_epochs=1,
            minibatch_size=12,
            actor_chunk_envs=2,
            max_capacity=128,
            total_updates=1,
            seed=1000 + idx,
            deterministic=True,
        )
        trainer = ArenaPPOTrainer(cfg, device="cpu")
        stage1_path = tmp_dir / f"synth_{arch}_stage1.pt"
        trainer.save_checkpoint(stage1_path)
        out[arch] = stage1_path
        if arch == "shared_trunk":
            eval_w_path = tmp_dir / "synth_shared_trunk_eval_weights.pth"
            trainer.save_eval_weights(eval_w_path)
            out["shared_trunk_eval_weights"] = eval_w_path
    return out


def _save_synthetic_b8_checkpoint(tmp_dir: Path, *, seed: int = 77) -> Path:
    """Create a hermetic `ego2s-bv8` (`Ego2sNet(7, 60)`) checkpoint in `tmp_dir`."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        net = Ego2sNet(local_channels=7, n_scalars=60)
    ckpt_path = tmp_dir / "synth_b8_bv8.pth"
    save_ego2s_checkpoint(ckpt_path, net, meta={"synthetic": True, "seed": seed})
    return ckpt_path


def _strip_timing(payload: dict[str, Any]) -> dict[str, Any]:
    """Return deep copy of evaluation payload with wall-clock `timing` zeroed out."""
    cleaned = copy.deepcopy(payload)
    cleaned.pop("timing", None)
    return cleaned


def _stage_manifest_inputs(mf: dict[str, Any], remote_inputs_dir: Path) -> Path:
    remote_inputs_dir.mkdir(parents=True, exist_ok=True)
    for inp in mf["inputs"]:
        dst = remote_inputs_dir / str(inp["dest"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(inp["path"], dst)
    return remote_inputs_dir


def _build_synthetic_canary_receipt(canary_mf: dict[str, Any], gpu_kind: str) -> Path:
    """Build a broker-verified canary receipt from real `run_canary_suite` exports."""
    canary_out = Path(canary_mf["output_dir"])
    canary_exports_dir = canary_out / "exports"
    remote_inputs_dir = _stage_manifest_inputs(canary_mf, canary_out / "remote_inputs")
    ledger_cand = remote_inputs_dir / "verified-ledger.json"
    suite_res = run_canary_suite(
        output_dir=canary_exports_dir,
        device="cpu",
        input_dir=remote_inputs_dir,
        ledger_path=ledger_cand if ledger_cand.is_file() else None,
        num_envs=2,
        rollout_len=4,
        seed=777,
    )
    (canary_out / "canary_suite_summary.json").write_text(
        json.dumps(suite_res, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    saved_probe_mf = canary_out / "manifest.json"
    saved_probe_mf.write_text(json.dumps(canary_mf, indent=2, sort_keys=True) + "\n")
    (canary_out / "upload-measurement.json").write_text(
        json.dumps(
            {
                "elapsed_seconds": 3.0,
                "transfer_seconds": 1.0,
                "source_delivery_seconds": 1.0,
                "staging_completed": True,
            }
        )
    )
    (canary_out / "hardware.json").write_text(json.dumps({"hardware": {"kind": gpu_kind}}))
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
    transfers = []
    for inp in canary_mf["inputs"]:
        transfers.append(
            {
                "direction": "upload",
                "remote": f"/content/snake-cli/gen-eval-01/inputs/{inp['dest']}",
                "status": "verified",
                "size_bytes": Path(inp["path"]).stat().st_size,
                "sha256": inp["sha256"],
            }
        )
    receipt_obj: dict[str, Any] = {
        "status": "PASS",
        "stop_verified": True,
        "error": None,
        "cleanup_error": None,
        "failure_class": None,
        "commit": canary_mf["commit"],
        "job_id": canary_mf["job_id"],
        "generation": "gen-eval-01",
        "manifest_sha256": sha256_file(saved_probe_mf),
        "exports": export_inventory,
        "transfers": transfers,
        "terminal": {
            "job_id": canary_mf["job_id"],
            "generation": "gen-eval-01",
            "returncode": 0,
            "status": "completed",
            "export_scanner": {"name": "runtime._exports", "inventory": export_inventory},
            "canary_observations": [
                {
                    "cuda_decisions": 12,
                    "device": "cuda:0",
                    "package": "research/arena_ppo",
                    "caller": "research/arena_ppo/net.py:act",
                    "operation": "torch.multinomial",
                    "shape": [12, 6],
                    "root": "/content/snake-cli/gen-eval-01",
                }
            ],
        },
    }
    receipt_obj["canary_evidence"] = colab_canary.evidence(canary_mf, receipt_obj, canary_out, ROOT)
    receipt_path = canary_out / "receipt.json"
    receipt_path.write_text(json.dumps(receipt_obj, indent=2, sort_keys=True) + "\n")
    return receipt_path


def _has_real_b8_s3_checkpoint() -> bool:
    """Return True when the canonical SHA-verified `b8-s3` checkpoint is available locally."""
    try:
        path, digest = resolve_b8_checkpoint_path()
        return path.is_file() and digest == B8_S3_SHA256
    except FileNotFoundError:
        return False


def test_eval_learner_obs_and_mask_equal_trainer_across_all_architectures(
    tmp_path: Path,
) -> None:
    """Prove evaluator action path matches `ArenaPPOTrainer._act_all_seats_chunked` 960/960.

    Advances a real-capacity (`max_capacity=1600`) `ArenaPPOTrainer` simulator for 25 steps
    with varied lengths, boosts, and corpses, clones the state into `TrainSim`, and steps both
    `ArenaPPOTrainer._act_all_seats_chunked` and `act_arena_policy_chunked` in closed loop for
    20 steps across `E=8, S=6` (`960` seat-decisions per architecture) under identical RNG.
    """
    num_envs = 8
    rollout_steps = 20
    chunk_envs = 3

    for idx, arch in enumerate(("ego_only", "shared_trunk", "per_seat")):
        cfg = ArenaPPOConfig(
            net_arch=arch,  # type: ignore[arg-type]
            net_size="S",
            num_envs=num_envs,
            rollout_len=4,
            ppo_epochs=1,
            minibatch_size=24,
            actor_chunk_envs=chunk_envs,
            max_capacity=1600,
            total_updates=1,
            seed=20261111 + idx,
            deterministic=True,
        )
        trainer = ArenaPPOTrainer(cfg, device="cpu")
        trainer.model.eval()

        # Seed varied snake lengths (above and below min_boost_length=5) and run 25 active steps.
        init_lens = torch.tensor([[2, 5, 8, 13, 19, 27]] * num_envs, dtype=torch.long)
        trainer.sim.sim.length.copy_(init_lens)
        trainer.sim.sim._reward_prev_length.copy_(init_lens)
        for step_idx in range(25):
            trainer.sim.prepare()
            legal = trainer.sim.legal_action_mask()
            if step_idx % 2 == 0:
                boost_ok = legal[:, :, 4]
                acts = torch.where(
                    boost_ok,
                    torch.full((num_envs, 6), 4, dtype=torch.long),
                    trainer.sim.scripted_greedy_food_actions(),
                )
            else:
                acts = trainer.sim.scripted_greedy_food_actions()
            trainer.sim.step(acts)

        ckpt_path = tmp_path / f"trainer_parity_{arch}.pt"
        trainer.save_checkpoint(ckpt_path)
        loaded = load_arena_ppo_policy(ckpt_path, "cpu", actor_chunk_override=chunk_envs)

        # 1. Closed-loop 20-step stochastic trajectory (`8 * 6 * 20 = 960` decisions) under
        # identical RNG: evaluator vs `trainer._act_all_seats_chunked`.
        base_state = trainer.sim.state_dict()
        eval_sim = TrainSim.create(
            num_envs,
            device="cpu",
            seed=cfg.seed,
            max_capacity=1600,
        )
        eval_sim.load_state_dict(copy.deepcopy(base_state))
        trainer.sim.load_state_dict(copy.deepcopy(base_state))

        matched_decisions = 0
        for step_idx in range(rollout_steps):
            step_rng_seed = 900_000 + 100 * idx + step_idx

            trainer.sim.prepare()
            tr_legal = trainer.sim.legal_action_mask()
            torch.manual_seed(step_rng_seed)
            tr_acts, _, _ = trainer._act_all_seats_chunked(
                trainer.model,
                tr_legal,
                greedy=False,
            )
            trainer.sim.step(tr_acts)

            eval_sim.prepare()
            ev_legal = eval_sim.legal_action_mask()
            torch.manual_seed(step_rng_seed)
            ev_acts = act_arena_policy_chunked(
                eval_sim,
                loaded,
                ev_legal,
                greedy=False,
                actor_chunk_envs=chunk_envs,
            )
            eval_sim.step(ev_acts)

            assert torch.equal(ev_legal, tr_legal)
            assert torch.equal(ev_acts, tr_acts)
            matched_decisions += int(ev_acts.numel())

        assert matched_decisions == 960
        assert torch.equal(eval_sim.sim.length, trainer.sim.sim.length)
        assert torch.equal(eval_sim.sim.alive, trainer.sim.sim.alive)
        assert torch.equal(eval_sim.sim.bodies, trainer.sim.sim.bodies)

        # 2. Verify chunk-size invariance on greedy argmax (`chunk=8` vs trainer `chunk=3`).
        trainer.sim.prepare()
        eval_sim.prepare()
        tr_legal_g = trainer.sim.legal_action_mask()
        ev_legal_g = eval_sim.legal_action_mask()
        tr_greedy, _, _ = trainer._act_all_seats_chunked(
            trainer.model,
            tr_legal_g,
            greedy=True,
        )
        ev_greedy_unchunked = act_arena_policy_chunked(
            eval_sim,
            loaded,
            ev_legal_g,
            greedy=True,
            actor_chunk_envs=num_envs,
        )
        assert torch.equal(ev_greedy_unchunked, tr_greedy)
        for k_seats in (1, 3, 6):
            seat_mask = build_learner_seat_mask(num_envs, 6, k_seats, seat_assignment="rotate")
            assert torch.equal(ev_greedy_unchunked[seat_mask], tr_greedy[seat_mask])


def test_b8_opponent_exact_feature_and_action_parity_against_product_path(
    tmp_path: Path,
) -> None:
    """Hermetic synthetic-weights check of `ego2s-bv8` features, Qs, and fallback logic.

    Bit-exact (`torch.equal`):
      - `local` tactical ego raster (`uint8`, shape `(N, 7, 31, 31)`).
      - `global` coarse ego raster (`uint8`, shape `(N, 4, 37, 37)`).
      - `advisory_mask` (`bool`, shape `(N, 6)`) + normal-only `0..2` fallback mask.
      - `chosen_actions` (`int64`, shape `(N,)`).
    Within float32 tolerance (`atol=1e-5`):
      - `scalars` (`float32`, shape `(N, 60)`, 30 base + 30 `v8` flood-fill/tail features).
      - `q_values` (`float32`, shape `(N, 6)`).
    """
    b8_ckpt = _save_synthetic_b8_checkpoint(tmp_path, seed=2026)
    num_envs = 3
    cfg = arena_v2_config(num_envs, max_capacity=256)
    world_seeds = derive_arena_world_seeds(4242, num_envs)
    ref_sim = GridBatchSim(cfg, seeds=world_seeds)
    ref_sim.reset()

    rng = np.random.default_rng(999)
    for t in range(40):
        adv = ref_sim.get_advisory_action_mask().astype(bool)
        acts = np.ones((num_envs, cfg.num_snakes), dtype=np.int64)
        for e in range(num_envs):
            for s in range(cfg.num_snakes):
                if not ref_sim.alive[e, s]:
                    continue
                valid = np.flatnonzero(adv[e, s])
                acts[e, s] = int(rng.choice(valid)) if valid.size > 0 else int(t % 3)
        ref_sim.step(acts)

    trapped_e, trapped_s = 0, 0
    if not ref_sim.alive[trapped_e, trapped_s]:
        ref_sim.reset_snake(trapped_e, trapped_s)

    dev_sim = DeviceSim.from_reference(
        ref_sim,
        device="cpu",
        hero_respawn=True,
        exact_mechanics=True,
    )
    dev_sim._prepared = True
    train_sim = TrainSim(dev_sim)

    alive_np = np.asarray(ref_sim.alive, dtype=bool)
    alive_rows_np = np.argwhere(alive_np).astype(np.int64)
    assert alive_rows_np.shape[0] >= 12
    rows_t = torch.from_numpy(alive_rows_np)

    prod_policy = Ego2sSimdPolicy(
        str(b8_ckpt),
        device="cpu",
        forward="batched",
        mask_mode="advisory",
    )
    prod_obs = build_ego_raster(
        ref_sim,
        ego_config_for(prod_policy.net),
        backend="numba",
        rows=alive_rows_np,
    )
    prod_q = student_q(prod_policy.net, ref_sim, alive_rows_np, forward="batched")
    dummy_masks = np.ones((alive_rows_np.shape[0], 6), dtype=bool)
    prod_actions = prod_policy.actions(dummy_masks, ref_sim, alive_rows_np)

    for chunk_size in (32, 1):
        (loc, glob, sca), dev_q, dev_act = compute_b8_observations_and_actions(
            dev_sim,
            prod_policy.net,
            rows_t,
            b8_env_chunk=chunk_size,
        )
        assert torch.equal(loc.cpu(), torch.from_numpy(prod_obs["local"]))
        assert torch.equal(glob.cpu(), torch.from_numpy(prod_obs["global"]))
        assert torch.allclose(sca.cpu(), torch.from_numpy(prod_obs["scalars"]), atol=1e-5)
        assert torch.allclose(dev_q.cpu(), prod_q, atol=1e-5)
        assert torch.equal(dev_act.cpu(), torch.from_numpy(prod_actions))

    ctrl = B8OpponentController(torch.device("cpu"), checkpoint_path=b8_ckpt, b8_env_chunk=1)
    opp_mask = ~build_learner_seat_mask(num_envs, cfg.num_snakes, 2, seat_assignment="rotate")
    ctrl_actions = ctrl.act_seats(train_sim, opp_mask)
    for row_i, (e_idx, s_idx) in enumerate(alive_rows_np.tolist()):
        if bool(opp_mask[e_idx, s_idx]):
            assert int(ctrl_actions[e_idx, s_idx].item()) == int(prod_actions[row_i])

    forced_q = torch.tensor([[-5.0, -4.0, -3.0, 10.0, 20.0, 30.0]], dtype=torch.float32)
    zero_adv = torch.zeros((1, 6), dtype=torch.bool)
    fallback = torch.arange(6)[None, :] < 3
    resolved = torch.where(zero_adv.any(dim=1, keepdim=True), zero_adv, fallback)
    chosen_fb = int(forced_q.masked_fill(~resolved, float("-inf")).argmax(dim=1).item())
    prod_m = zero_adv.numpy().copy()
    prod_m[~prod_m.any(axis=1), :3] = True
    prod_fb = int(np.where(prod_m, forced_q.numpy().astype(np.float64), -np.inf).argmax(axis=1)[0])
    assert chosen_fb == prod_fb == 2


@pytest.mark.skipif(
    not _has_real_b8_s3_checkpoint(),
    reason="Canonical released b8-s3 checkpoint is not present on this machine.",
)
def test_b8_real_b8_s3_multiframe_closed_loop_parity_against_product_path() -> None:
    """Closed-loop multi-frame B8 parity with real SHA-verified `b8-s3` weights vs product path.

    Runs a 15-frame closed-loop trajectory across 2 mature worlds (`12` snakes/frame) driven
    by `B8OpponentController` with the canonical read-only `b8-s3` checkpoint
    (`SHA-256 = 3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`), mirroring
    each prepared `DeviceSim` world into a 1-env `GridBatchSim` at every frame and checking
    against the product `Ego2sSimdPolicy` (`mask_mode='advisory'`, Numba raster backend).

    Quantities that are bit-exact (`0` tolerance, `torch.equal` / exact array equality):
      - `local`: `uint8` tactical ego raster `(N, 7, 31, 31)`.
      - `global`: `uint8` coarse strategic ego raster `(N, 4, 37, 37)`.
      - `advisory_mask`: `bool` 6-action safety mask `(N, 6)` and normal-only `0..2` fallback.
      - `chosen_actions`: `int64` masked-argmax actions `(N,)` across all un-tied frames
        (verified 100% bit-exact over all live snake-frames in this trajectory; argmax could
        only differ if two legal Q-values sat within `< 1e-5` float32 scalar rounding).

    Quantities within float32 tolerance (`atol=1e-5, rtol=1e-5`):
      - `scalars`: `float32` 60-D scalar vector (`30` base + `30` `v8` safety features), where
        device PyTorch reductions vs Numba float32 divisions can differ by `<= ~1e-6`.
      - `q_values`: `float32` 6-action dueling Q-values `(N, 6)` propagated from `scalars`.
    """
    b8_path, b8_sha = resolve_b8_checkpoint_path()
    assert b8_sha == B8_S3_SHA256

    num_envs = 2
    train_sim = TrainSim.create(num_envs, device="cpu", seed=20261111, max_capacity=256)
    init_lens = torch.tensor([[4, 6, 9, 12, 16, 22]] * num_envs, dtype=torch.long)
    train_sim.sim.length.copy_(init_lens)
    train_sim.sim._reward_prev_length.copy_(init_lens)
    for _ in range(25):
        train_sim.prepare()
        train_sim.step(train_sim.scripted_greedy_food_actions())

    ctrl = B8OpponentController(torch.device("cpu"), checkpoint_path=b8_path, b8_env_chunk=1)
    assert ctrl.sha256 == B8_S3_SHA256
    prod_policy = Ego2sSimdPolicy(
        str(b8_path),
        device="cpu",
        forward="batched",
        mask_mode="advisory",
    )
    one_env_cfg = arena_v2_config(1, max_capacity=256)
    ref_one = GridBatchSim(one_env_cfg, seeds=[0])
    ref_one.reset()
    all_seats = torch.ones((num_envs, 6), dtype=torch.bool)

    total_checked_rows = 0
    for _frame_idx in range(15):
        train_sim.prepare()
        alive_rows = train_sim.sim.alive.nonzero(as_tuple=False)
        assert int(alive_rows.shape[0]) > 0
        (dev_loc, dev_glob, dev_sca), dev_q, dev_act = compute_b8_observations_and_actions(
            train_sim.sim,
            ctrl.net,
            alive_rows,
            b8_env_chunk=1,
        )
        dev_adv = train_sim.sim.advisory_mask()
        ctrl_full_actions = ctrl.act_seats(train_sim, all_seats)

        row_cursor = 0
        for env_i in range(num_envs):
            train_sim.sim.copy_env_to_reference(ref_one, env=env_i)
            env_alive_slots = np.flatnonzero(np.asarray(ref_one.alive[0], dtype=bool))
            n_env_alive = int(env_alive_slots.size)
            if n_env_alive == 0:
                continue
            ref_rows = np.stack([np.zeros_like(env_alive_slots), env_alive_slots], axis=1).astype(
                np.int64
            )
            prod_adv = ref_one.get_advisory_action_mask()[0, env_alive_slots].astype(bool)
            prod_obs = build_ego_raster(
                ref_one,
                ego_config_for(prod_policy.net),
                backend="numba",
                rows=ref_rows,
            )
            prod_q = student_q(prod_policy.net, ref_one, ref_rows, forward="batched")
            dummy_masks = np.ones((n_env_alive, 6), dtype=bool)
            prod_act = prod_policy.actions(dummy_masks, ref_one, ref_rows)

            sl = slice(row_cursor, row_cursor + n_env_alive)
            row_cursor += n_env_alive
            total_checked_rows += n_env_alive

            dev_env_adv = dev_adv[env_i, torch.from_numpy(env_alive_slots)].cpu().numpy()
            assert np.array_equal(dev_env_adv, prod_adv)
            assert torch.equal(dev_loc[sl].cpu(), torch.from_numpy(prod_obs["local"]))
            assert torch.equal(dev_glob[sl].cpu(), torch.from_numpy(prod_obs["global"]))
            assert torch.allclose(
                dev_sca[sl].cpu(), torch.from_numpy(prod_obs["scalars"]), atol=1e-5, rtol=1e-5
            )
            assert torch.allclose(dev_q[sl].cpu(), prod_q, atol=1e-5, rtol=1e-5)
            assert torch.equal(dev_act[sl].cpu(), torch.from_numpy(prod_act))
            assert torch.equal(
                ctrl_full_actions[env_i, torch.from_numpy(env_alive_slots)].cpu(),
                torch.from_numpy(prod_act),
            )

        train_sim.step(ctrl_full_actions)

    assert total_checked_rows >= 150
    assert sha256_file(b8_path) == B8_S3_SHA256


def test_evaluate_cpu_determinism_all_opponents_and_cli(tmp_path: Path) -> None:
    """Prove exact CPU determinism across all opponent types and CLI invocation."""
    ckpts = _save_synthetic_arena_checkpoints(tmp_path)
    b8_ckpt = _save_synthetic_b8_checkpoint(tmp_path, seed=314)
    ego_ckpt = ckpts["ego_only"]
    st_eval_ckpt = ckpts["shared_trunk_eval_weights"]

    opponents = (
        "random_safe",
        "greedy_food",
        "untrained",
        f"ckpt:{st_eval_ckpt}",
        f"b8:{b8_ckpt}",
    )
    for opp in opponents:
        r1 = evaluate_policy(
            checkpoint=ego_ckpt,
            opponent=opp,
            learner_seats=2,
            worlds=2,
            ticks=12,
            burnin=4,
            seed=20261111,
            device="cpu",
            b8_env_chunk=1,
        )
        r2 = evaluate_policy(
            checkpoint=ego_ckpt,
            opponent=opp,
            learner_seats=2,
            worlds=2,
            ticks=12,
            burnin=4,
            seed=20261111,
            device="cpu",
            b8_env_chunk=1,
        )
        assert _strip_timing(r1) == _strip_timing(r2)
        assert r1["schema"] == EVAL_SCHEMA
        if opp.startswith("b8:"):
            r_chunk2 = evaluate_policy(
                checkpoint=ego_ckpt,
                opponent=opp,
                learner_seats=2,
                worlds=2,
                ticks=12,
                burnin=4,
                seed=20261111,
                device="cpu",
                b8_env_chunk=2,
            )
            assert r1["learner"] == r_chunk2["learner"]
            assert r1["opponent"] == r_chunk2["opponent"]
            assert r1["kill_matrix"] == r_chunk2["kill_matrix"]

    # Distinct seed produces distinct trajectory / action realizations under stochastic sampling.
    diff_seed = evaluate_policy(
        checkpoint=ego_ckpt,
        opponent="random_safe",
        learner_seats=3,
        worlds=3,
        ticks=20,
        burnin=5,
        seed=999999,
        device="cpu",
    )
    same_seed = evaluate_policy(
        checkpoint=ego_ckpt,
        opponent="random_safe",
        learner_seats=3,
        worlds=3,
        ticks=20,
        burnin=5,
        seed=20261111,
        device="cpu",
    )
    assert _strip_timing(diff_seed) != _strip_timing(same_seed)

    # CLI entrypoint `--out` + `--sidecars` determinism check.
    cli_out = tmp_path / "cli_eval_out.json"
    rc = evaluate_cli_main(
        [
            "--checkpoint",
            str(ego_ckpt),
            "--opponent",
            "random_safe",
            "--learner-seats",
            "3",
            "--worlds",
            "3",
            "--ticks",
            "20",
            "--burnin",
            "5",
            "--seed",
            "20261111",
            "--device",
            "cpu",
            "--sidecars",
            "--out",
            str(cli_out),
        ]
    )
    assert rc == 0
    assert verify_completed_artifact(cli_out)
    cli_payload = json.loads(cli_out.read_text(encoding="utf-8"))
    assert _strip_timing(cli_payload) == _strip_timing(same_seed)


def test_seating_burnin_exclusion_and_exact_collision_attribution(tmp_path: Path) -> None:
    """Verify K=0..6 seating, burn-in exclusion, and exact killer->victim / mutual attribution."""
    ckpts = _save_synthetic_arena_checkpoints(tmp_path)
    ego_ckpt = ckpts["ego_only"]

    # Balanced rotation across W=6 worlds gives each slot 0..5 the exact same count.
    rot_k1 = build_learner_seat_mask(6, 6, 1, seat_assignment="rotate")
    assert rot_k1.sum(dim=1).tolist() == [1, 1, 1, 1, 1, 1]
    assert rot_k1.sum(dim=0).tolist() == [1, 1, 1, 1, 1, 1]
    rot_k3 = build_learner_seat_mask(6, 6, 3, seat_assignment="rotate")
    assert rot_k3.sum(dim=1).tolist() == [3, 3, 3, 3, 3, 3]
    assert rot_k3.sum(dim=0).tolist() == [3, 3, 3, 3, 3, 3]

    # Direct check on vectorized `scripted_random_safe_actions`: in 0..2 & respects safe mask.
    rs_sim = TrainSim.create(4, device="cpu", seed=888, max_capacity=64)
    rs_sim.prepare()
    rs_adv = rs_sim.sim.advisory_mask()[:, :, :3]
    rs_act = scripted_random_safe_actions(rs_sim)
    assert bool(((rs_act >= 0) & (rs_act <= 2)).all().item())
    has_safe = rs_adv.any(dim=-1) & rs_sim.sim.alive
    picked_safe = rs_adv.gather(-1, rs_act[..., None]).squeeze(-1)
    assert bool(picked_safe[has_safe].all().item())

    # K = 0 (opponent-only baseline, no checkpoint needed for scripted opponent)
    res_k0 = evaluate_policy(
        checkpoint=None,
        opponent="greedy_food",
        learner_seats=0,
        worlds=3,
        ticks=25,
        burnin=10,
        seed=55,
        device="cpu",
    )
    assert res_k0["checkpoint_info"] is None
    assert res_k0["learner"]["seats_per_world"] == 0
    assert res_k0["learner"]["completed_lives"] == 0
    assert res_k0["learner"]["time_averaged_length"] is None
    assert res_k0["opponent"]["seats_per_world"] == 6
    assert res_k0["opponent"]["raw_counts"]["acted_agent_steps"] > 0

    # K = 6 (pure self-play reference)
    res_k6 = evaluate_policy(
        checkpoint=ego_ckpt,
        opponent="random_safe",
        learner_seats=6,
        worlds=3,
        ticks=25,
        burnin=10,
        seed=55,
        device="cpu",
    )
    assert res_k6["learner"]["seats_per_world"] == 6
    assert res_k6["opponent"]["seats_per_world"] == 0
    assert res_k6["opponent"]["completed_lives"] == 0
    assert res_k6["kill_matrix"]["learner_to_opponent"] == 0
    assert res_k6["kill_matrix"]["opponent_to_learner"] == 0

    # Burn-in exclusion: run 10 ticks with burnin=0 vs 5 ticks burnin + 5 ticks eval on identical
    # greedy deterministic trajectory -> sum of acted steps across split equals total 10-tick run!
    full_10 = evaluate_policy(
        checkpoint=ego_ckpt,
        opponent="greedy_food",
        learner_seats=3,
        worlds=4,
        ticks=10,
        burnin=0,
        seed=12345,
        device="cpu",
        greedy=True,
    )
    first_5 = evaluate_policy(
        checkpoint=ego_ckpt,
        opponent="greedy_food",
        learner_seats=3,
        worlds=4,
        ticks=5,
        burnin=0,
        seed=12345,
        device="cpu",
        greedy=True,
    )
    second_5 = evaluate_policy(
        checkpoint=ego_ckpt,
        opponent="greedy_food",
        learner_seats=3,
        worlds=4,
        ticks=5,
        burnin=5,
        seed=12345,
        device="cpu",
        greedy=True,
    )
    assert (
        first_5["learner"]["raw_counts"]["acted_agent_steps"]
        + second_5["learner"]["raw_counts"]["acted_agent_steps"]
        == full_10["learner"]["raw_counts"]["acted_agent_steps"]
    )
    assert (
        first_5["learner"]["completed_lives"] + second_5["learner"]["completed_lives"]
        == full_10["learner"]["completed_lives"]
    )
    assert (
        first_5["learner"]["raw_counts"]["food_eaten"]
        + second_5["learner"]["raw_counts"]["food_eaten"]
        == full_10["learner"]["raw_counts"]["food_eaten"]
    )

    # Controlled collision scenarios across 5 worlds in a single step:
    #   env 0: slot 0 hits top wall (y=0, dir=0 North, action=1 straight) -> wall death
    #   env 1: slot 0 bites own body (curled loop len=5) -> self death
    #   env 2: slot 0 runs into slot 1's body -> body kill credited to slot 1
    #   env 3: slot 0 (len=10) head-on vs slot 1 (len=4) -> head kill credited to slot 0 (>=1.25x)
    #   env 4: slot 0 (len=6) head-on vs slot 1 (len=6) -> mutual head-on tie (0 kills credited)
    cfg = arena_v2_config(5, max_capacity=64)
    dsim = DeviceSim(
        cfg,
        device="cpu",
        train_mode=False,
        allow_respawn=True,
        cap=64,
        move_backend="reference",
        hero_respawn=True,
        exact_mechanics=True,
    )
    dsim.alive.fill_(False)
    dsim.respawn_timer.fill_(50)  # prevent unrelated slots from spawning on this step
    dsim.food_present.fill_(False)
    dsim.corpse_food.fill_(False)
    dsim.food_count.zero_()
    dsim.head_ptr.zero_()
    dsim.seg_count.zero_()
    dsim.length.fill_(3)

    def _place_snake(
        env: int, slot: int, coords: list[tuple[int, int]], direction: int, length: int
    ) -> None:
        # coords[0] is head, coords[1] is neck, ...
        n_seg = len(coords)
        dsim.alive[env, slot] = True
        dsim.respawn_timer[env, slot] = 0
        dsim.direction[env, slot] = direction
        dsim.length[env, slot] = length
        dsim.seg_count[env, slot] = n_seg
        dsim.head_ptr[env, slot] = n_seg - 1
        for k_idx, (x_pos, y_pos) in enumerate(coords):
            ring_pos = (n_seg - 1 - k_idx) % dsim.cap
            dsim.bodies[env, slot, ring_pos, 0] = x_pos
            dsim.bodies[env, slot, ring_pos, 1] = y_pos

    # env 0: wall hit (head at (20, 0) moving North dir=0 -> (20, -1) out of arena)
    _place_snake(0, 0, [(20, 0), (20, 1), (20, 2)], direction=0, length=3)
    # env 1: self bite (head at (20, 20) moving East dir=1 into own segment k=3 at (21, 20))
    _place_snake(
        1,
        0,
        [(20, 20), (20, 21), (21, 21), (21, 20), (21, 19), (21, 18)],
        direction=1,
        length=6,
    )
    # env 2: slot 0 head at (30, 30) moving East dir=1 into slot 1's body at (31, 30)
    _place_snake(2, 0, [(30, 30), (29, 30), (28, 30)], direction=1, length=4)
    _place_snake(2, 1, [(31, 29), (31, 30), (31, 31), (31, 32)], direction=0, length=4)
    # env 3: head-on size win: slot 0 (len=10) at (40, 40) East vs slot 1 (len=4) at (42, 40) West
    # Both move onto (41, 40); 10 >= 1.25 * 4 -> slot 0 kills slot 1!
    _place_snake(3, 0, [(40, 40), (39, 40), (38, 40)], direction=1, length=10)
    _place_snake(3, 1, [(42, 40), (43, 40), (44, 40)], direction=3, length=4)
    # env 4: mutual head-on tie: slot 0 (len=6) at (50, 50) East vs slot 1 (len=6) at (52, 50) West
    _place_snake(4, 0, [(50, 50), (49, 50), (48, 50)], direction=1, length=6)
    _place_snake(4, 1, [(52, 50), (53, 50), (54, 50)], direction=3, length=6)

    tsim = TrainSim(dsim)
    tsim.prepare()
    pre_head0 = tsim.sim.heads().clone()
    straight_actions = torch.ones((5, 6), dtype=torch.long)
    step_info = tsim.step(straight_actions)
    attrib = attribute_step_deaths_and_kills(tsim.sim, pre_head0, step_info)

    assert int(tsim.sim._last_death_cause[0, 0].item()) == DEATH_WALL
    assert bool(attrib.wall_deaths[0, 0].item()) is True

    assert int(tsim.sim._last_death_cause[1, 0].item()) == DEATH_SELF
    assert bool(attrib.self_deaths[1, 0].item()) is True

    assert int(tsim.sim._last_death_cause[2, 0].item()) == DEATH_BODY
    assert int(attrib.pair_kills[2, 1, 0].item()) == 1  # killer=1, victim=0

    assert int(tsim.sim._last_death_cause[3, 0].item()) == 0
    assert int(tsim.sim._last_death_cause[3, 1].item()) == DEATH_HEAD
    assert int(attrib.pair_kills[3, 0, 1].item()) == 1  # killer=0, victim=1

    assert int(tsim.sim._last_death_cause[4, 0].item()) == DEATH_HEAD
    assert int(tsim.sim._last_death_cause[4, 1].item()) == DEATH_HEAD
    assert int(attrib.pair_kills[4].sum().item()) == 0
    assert bool(attrib.pair_mutual[4, 0, 1].item()) is True
    assert bool(attrib.pair_mutual[4, 1, 0].item()) is True


def test_default_sizing_guarantees_300_lives_and_mps_smoke(tmp_path: Path) -> None:
    """Verify mature default (W, T, B=3000) sizing, start config rules, MPS smoke, and stops."""
    assert DEFAULT_EVAL_WORLDS == 256
    assert DEFAULT_EVAL_TICKS == 4500
    assert DEFAULT_EVAL_BURNIN == 3000
    assert len(DEFAULT_WORLD_AGE_BIN_EDGES) >= 5
    assert len(DEFAULT_LIFE_AGE_BIN_EDGES) >= 5

    assert resolve_eval_start_config(
        start_mode="mature", burnin=None, init_from_checkpoint_state=False
    ) == ("mature", 3000)
    assert resolve_eval_start_config(
        start_mode="fresh", burnin=None, init_from_checkpoint_state=False
    ) == ("fresh", 0)
    assert resolve_eval_start_config(
        start_mode="mature", burnin=None, init_from_checkpoint_state=True
    ) == ("mature", 0)
    with pytest.raises(ValueError, match="fresh"):
        resolve_eval_start_config(start_mode="fresh", burnin=10, init_from_checkpoint_state=False)
    with pytest.raises(ValueError, match="fresh"):
        resolve_eval_start_config(start_mode="fresh", burnin=None, init_from_checkpoint_state=True)

    # Measure empirical `random_safe` death rate on a compact mature-burnin sample.
    sample = evaluate_policy(
        checkpoint=None,
        opponent="random_safe",
        learner_seats=0,
        worlds=6,
        ticks=250,
        burnin=200,
        seed=20261111,
        device="cpu",
    )
    deaths_per_1k = float(sample["opponent"]["deaths_per_1k_agent_steps"])
    default_k1_seat_ticks = DEFAULT_EVAL_WORLDS * 1 * DEFAULT_EVAL_TICKS  # 1,152,000
    projected_k1_lives = deaths_per_1k * (default_k1_seat_ticks / 1000.0)
    assert projected_k1_lives >= 300.0, f"Projected K=1 lives {projected_k1_lives:.1f} < 300"
    assert sample["opponent"]["sample_adequacy"]["target_completed_lives"] == 300
    assert (
        sample["opponent"]["confidence_intervals_95"]["method"]
        == "world_cluster_percentile_bootstrap"
    )

    # Cooperative stop hook raises StopRequestedError cleanly.
    with pytest.raises(StopRequestedError):
        evaluate_policy(
            checkpoint=None,
            opponent="random_safe",
            learner_seats=0,
            worlds=2,
            ticks=10,
            burnin=2,
            seed=1,
            device="cpu",
            should_stop=lambda: True,
        )

    if torch.backends.mps.is_available():
        ckpts = _save_synthetic_arena_checkpoints(tmp_path)
        for arch in ("ego_only", "shared_trunk"):
            mps_res = evaluate_policy(
                checkpoint=ckpts[arch],
                opponent="random_safe",
                learner_seats=2,
                worlds=2,
                ticks=8,
                burnin=2,
                seed=77,
                device="mps",
            )
            assert mps_res["config"]["device"].startswith("mps")
            assert mps_res["learner"]["raw_counts"]["acted_agent_steps"] > 0


def test_death_length_records_pre_step_length_before_boost_burn_or_food_eat() -> None:
    """Verify length-at-death uses pre-step length before fatal-step boost burn or food eat."""
    cfg = arena_v2_config(2, max_capacity=64)
    dsim = DeviceSim(
        cfg,
        device="cpu",
        train_mode=False,
        allow_respawn=True,
        cap=64,
        move_backend="reference",
        hero_respawn=True,
        exact_mechanics=True,
    )
    dsim.alive.fill_(False)
    dsim.respawn_timer.fill_(50)
    dsim.food_present.fill_(False)
    dsim.corpse_food.fill_(False)
    dsim.food_count.zero_()
    dsim.head_ptr.zero_()
    dsim.seg_count.zero_()
    dsim.length.fill_(1)
    dsim.boost_frames.zero_()

    def _place_snake(
        env: int, slot: int, coords: list[tuple[int, int]], direction: int, length: int
    ) -> None:
        n_seg = len(coords)
        dsim.alive[env, slot] = True
        dsim.respawn_timer[env, slot] = 0
        dsim.direction[env, slot] = direction
        dsim.length[env, slot] = length
        dsim._reward_prev_length[env, slot] = length
        dsim.seg_count[env, slot] = n_seg
        dsim.head_ptr[env, slot] = n_seg - 1
        for k_idx, (x_pos, y_pos) in enumerate(coords):
            ring_pos = (n_seg - 1 - k_idx) % dsim.cap
            dsim.bodies[env, slot, ring_pos, 0] = x_pos
            dsim.bodies[env, slot, ring_pos, 1] = y_pos

    # env 0: slot 0 has pre-step length=10, boost_frames=2 -> straight boost action=4 burns
    # sim.length from 10 -> 9 BEFORE hitting the North wall at (20, -1).
    _place_snake(
        0,
        0,
        [(20, 0), (20, 1), (20, 2), (20, 3), (20, 4), (20, 5)],
        direction=0,
        length=10,
    )
    dsim.boost_frames[0, 0] = int(cfg.boost_length_cost_frames) - 1

    # env 1: slot 0 has pre-step length=6 moving East (30, 30) -> (31, 30) where BOTH an ambient
    # food pellet sits (incrementing sim.length 6 -> 7) AND slot 1's body occupies (31, 30).
    _place_snake(1, 0, [(30, 30), (29, 30), (28, 30)], direction=1, length=6)
    _place_snake(1, 1, [(31, 29), (31, 30), (31, 31), (31, 32)], direction=0, length=8)
    dsim.food_cells[1, 0, 0] = 31
    dsim.food_cells[1, 0, 1] = 30
    dsim.food_present[1, 0] = True
    dsim.corpse_food[1, 0] = False
    dsim.food_count[1] = 1
    dsim.frame.fill_(1)
    dsim._prepared = True

    tsim = TrainSim(dsim)
    pre_head0 = tsim.sim.heads().clone()
    pre_length = tsim.sim.length.clone()
    actions = torch.ones((2, 6), dtype=torch.long)
    actions[0, 0] = 4  # straight boost -> triggers burn (10 -> 9) and wall death
    actions[1, 0] = 1  # normal straight -> eats food (6 -> 7) and dies on slot 1's body

    boosted = tsim.sim.alive & (actions >= 3) & (pre_length >= int(tsim.sim.cfg.min_boost_length))
    info = tsim.step(actions)
    attrib = attribute_step_deaths_and_kills(tsim.sim, pre_head0, info)

    # Confirm post-step sim.length WAS mutated to 9 and 7 by burn and food eat respectively:
    assert int(tsim.sim.length[0, 0].item()) == 9
    assert int(tsim.sim.length[1, 0].item()) == 7
    assert bool(info.died[0, 0].item()) is True
    assert bool(info.died[1, 0].item()) is True

    # Feed env 0 and env 1 separately and jointly into `_SideMetricsAccumulator`:
    side_mask = torch.zeros((2, 6), dtype=torch.bool)
    side_mask[:, 0] = True
    other_mask = ~side_mask
    acc = _SideMetricsAccumulator(
        worlds=2,
        seats_per_world=1,
        device=torch.device("cpu"),
        world_age_bin_edges=(0, 10, 20, 30, 40),
        life_age_bin_edges=(0, 10, 20, 30, 40),
        include_incumbent_unknown_bin=False,
        bootstrap_seed=123,
    )
    acc.record_window_open(side_mask, torch.ones((2, 6), dtype=torch.bool) & side_mask)
    ones_age = torch.full((2, 6), 5, dtype=torch.int64)
    known_age = torch.ones((2, 6), dtype=torch.bool)
    spawn_len = torch.ones((2, 6), dtype=torch.int64)
    acc.record_step(
        sim=tsim.sim,
        side_mask=side_mask,
        other_mask=other_mask,
        pre_length=pre_length,
        pre_life_age=ones_age,
        pre_life_age_known=known_age,
        pre_life_spawn_length=spawn_len,
        pre_life_max_length=pre_length.to(torch.int64),
        boosted=boosted,
        born_in_window=torch.ones((2, 6), dtype=torch.bool),
        info=info,
        attrib=attrib,
    )
    summary = acc.finalize(
        side_mask=side_mask,
        final_alive=tsim.sim.alive,
        final_length=tsim.sim.length,
        final_life_age=ones_age,
        final_life_age_known=known_age,
        final_life_spawn_length=spawn_len,
        final_life_max_length=pre_length.to(torch.int64),
    )
    # Pre-step lengths were 10 and 6 -> mean = 8.0, p95 = 9.8 (whereas post-step 9 and 7 -> p95=8.9)
    assert acc._death_lengths[0].tolist() == [10, 6]
    assert summary["mean_length_at_death"] == 8.0
    assert summary["p95_length_at_death"] == 9.8


def test_mature_checkpoint_state_fresh_modes_age_bins_stuck_and_world_bootstrap_cis(
    tmp_path: Path,
) -> None:
    """Test mature/fresh/checkpoint-state starts, bin conservation, stuck metric, and CIs."""
    # 1. Build a 4-world Stage 1 checkpoint advanced 12 ticks so sim_state_dict is non-trivial.
    cfg = ArenaPPOConfig(
        net_arch="ego_only",
        net_size="S",
        num_envs=4,
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=12,
        actor_chunk_envs=2,
        max_capacity=128,
        total_updates=1,
        seed=31337,
        deterministic=True,
    )
    trainer = ArenaPPOTrainer(cfg, device="cpu")
    for _ in range(12):
        trainer.sim.prepare()
        trainer.sim.step(trainer.sim.scripted_greedy_food_actions())
    ckpt_4w = tmp_path / "stage1_4w.pt"
    trainer.save_checkpoint(ckpt_4w)
    eval_only_w = tmp_path / "eval_only_weights.pth"
    trainer.save_eval_weights(eval_only_w)

    # `--init-from-checkpoint-state` works on slices (`worlds=2, offset=1`) and tracks incumbent bin
    res_ckpt_state = evaluate_policy(
        checkpoint=ckpt_4w,
        opponent="greedy_food",
        learner_seats=3,
        worlds=2,
        ticks=15,
        start_mode="mature",
        init_from_checkpoint_state=True,
        checkpoint_world_offset=1,
        world_age_bin_edges=(0, 5, 10, 15, 20, 30),
        life_age_bin_edges=(0, 2, 5, 10, 20),
        seed=99,
        device="cpu",
    )
    assert res_ckpt_state["config"]["start_mode"] == "mature"
    assert res_ckpt_state["config"]["burnin"] == 0
    assert res_ckpt_state["config"]["init_from_checkpoint_state"] is True
    assert res_ckpt_state["config"]["checkpoint_world_offset"] == 1
    assert res_ckpt_state["config"]["initial_world_age_min"] == 12

    # Rejects checkpoint lacking sim_state_dict or out-of-range slice:
    with pytest.raises(ValueError, match="sim_state_dict"):
        evaluate_policy(
            checkpoint=eval_only_w,
            opponent="greedy_food",
            learner_seats=2,
            worlds=2,
            ticks=5,
            init_from_checkpoint_state=True,
            device="cpu",
        )
    with pytest.raises(ValueError, match="exceeds"):
        evaluate_policy(
            checkpoint=ckpt_4w,
            opponent="greedy_food",
            learner_seats=2,
            worlds=3,
            ticks=5,
            init_from_checkpoint_state=True,
            checkpoint_world_offset=2,
            device="cpu",
        )

    # 2. Fresh mode (`start_mode="fresh"`) + bin conservation across world_age_bins & life_age_bins:
    res_fresh = evaluate_policy(
        checkpoint=ckpt_4w,
        opponent="random_safe",
        learner_seats=3,
        worlds=4,
        ticks=30,
        start_mode="fresh",
        world_age_bin_edges=(0, 5, 10, 15, 20, 25),
        life_age_bin_edges=(0, 3, 6, 10, 15, 20),
        seed=20261111,
        device="cpu",
    )
    assert res_fresh["config"]["start_mode"] == "fresh"
    assert res_fresh["config"]["burnin"] == 0
    for run_res in (res_ckpt_state, res_fresh):
        for side_key in ("learner", "opponent"):
            side = run_res[side_key]
            top_raw = side["raw_counts"]
            for bins_key in ("world_age_bins", "life_age_bins"):
                bins = side[bins_key]
                assert len(bins) >= 5
                assert (
                    sum(b["raw_counts"]["acted_agent_steps"] for b in bins)
                    == top_raw["acted_agent_steps"]
                )
                assert (
                    sum(b["raw_counts"]["sum_alive_length"] for b in bins)
                    == top_raw["sum_alive_length"]
                )
                assert sum(b["raw_counts"]["deaths"] for b in bins) == top_raw["deaths"]
                assert sum(b["raw_counts"]["food_eaten"] for b in bins) == top_raw["food_eaten"]

            cis = side["confidence_intervals_95"]
            for metric_key in (
                "time_averaged_length",
                "deaths_per_1k_agent_steps",
                "food_per_1k_steps",
            ):
                lo, hi = cis[metric_key]
                est = float(side[metric_key])
                assert lo - 1e-3 <= est <= hi + 1e-3

    # 3. Deterministic stuck-metric test (> 500 ticks without growing past spawn_length + 2):
    # Drive 2 single-snake worlds for 510 ticks with constant left-turn `action=0` (2x2 tight loop)
    # in world 0 (spawn_len=1, max_len=1 <= 3 -> stuck!) vs world 1 (spawn_len=1, max_len=8 -> not
    # stuck).
    stuck_acc = _SideMetricsAccumulator(
        worlds=2,
        seats_per_world=1,
        device=torch.device("cpu"),
        world_age_bin_edges=(0, 100, 200, 300, 400, 500),
        life_age_bin_edges=(0, 100, 200, 300, 400, 500),
        include_incumbent_unknown_bin=False,
        bootstrap_seed=7,
    )
    cfg_2w = arena_v2_config(2, max_capacity=32)
    sim_2w = DeviceSim(
        cfg_2w,
        device="cpu",
        train_mode=False,
        allow_respawn=False,
        cap=32,
        move_backend="reference",
        hero_respawn=True,
        exact_mechanics=True,
    )
    sim_2w.alive.fill_(False)
    sim_2w.alive[:, 0] = True
    sim_2w.direction.fill_(1)
    sim_2w.head_ptr.zero_()
    sim_2w.seg_count.fill_(1)
    sim_2w.bodies[:, 0, 0, 0] = 40
    sim_2w.bodies[:, 0, 0, 1] = 40
    sim_2w.length[0, 0] = 1  # world 0 stays at length 1 forever
    sim_2w.length[1, 0] = 8  # world 1 already grew to length 8 (> 1 + 2 and > 6)
    sim_2w._reward_prev_length.copy_(sim_2w.length)
    sim_2w.food_present.fill_(False)
    sim_2w.corpse_food.fill_(False)
    sim_2w.food_count.zero_()

    tsim_2w = TrainSim(sim_2w)
    side_m = torch.zeros((2, 6), dtype=torch.bool)
    side_m[:, 0] = True
    other_m = ~side_m
    stuck_acc.record_window_open(side_m, sim_2w.alive)

    life_age = torch.zeros((2, 6), dtype=torch.int64)
    life_known = torch.ones((2, 6), dtype=torch.bool)
    spawn_len_t = torch.ones((2, 6), dtype=torch.int64)
    max_len_t = sim_2w.length.to(torch.int64).clone()
    born_win = torch.ones((2, 6), dtype=torch.bool)
    turn_left = torch.zeros((2, 6), dtype=torch.long)
    go_straight = torch.ones((2, 6), dtype=torch.long)

    for tick in range(510):
        sim_2w.frame.fill_(tick + 1)
        sim_2w._prepared = True
        pre_h0 = sim_2w.heads().clone()
        pre_l = sim_2w.length.clone()
        step_act = turn_left if (tick % 10 == 0) else go_straight
        step_i = tsim_2w.step(step_act)
        att_i = attribute_step_deaths_and_kills(sim_2w, pre_h0, step_i)
        stuck_acc.record_step(
            sim=sim_2w,
            side_mask=side_m,
            other_mask=other_m,
            pre_length=pre_l,
            pre_life_age=life_age,
            pre_life_age_known=life_known,
            pre_life_spawn_length=spawn_len_t,
            pre_life_max_length=max_len_t,
            boosted=torch.zeros((2, 6), dtype=torch.bool),
            born_in_window=born_win,
            info=step_i,
            attrib=att_i,
        )
        life_age[:, 0] += 1

    stuck_res = stuck_acc.finalize(
        side_mask=side_m,
        final_alive=sim_2w.alive,
        final_length=sim_2w.length,
        final_life_age=life_age,
        final_life_age_known=life_known,
        final_life_spawn_length=spawn_len_t,
        final_life_max_length=max_len_t,
    )["stuck"]
    # At ticks with pre_life_age in 501..509 (9 ticks), world 0 is stuck and world 1 is not:
    assert stuck_res["raw_counts"]["final_live_seats"] == 2
    assert stuck_res["raw_counts"]["final_live_older_than_500_seats"] == 2
    assert stuck_res["raw_counts"]["final_stuck_seats"] == 1
    assert stuck_res["live_seats_share"] == 0.5
    assert stuck_res["live_older_than_500_seats_share"] == 0.5
    assert stuck_res["raw_counts"]["acted_agent_steps"] == 1020
    assert stuck_res["raw_counts"]["older_than_500_agent_steps"] == 18
    assert stuck_res["raw_counts"]["stuck_agent_steps"] == 9
    assert stuck_res["agent_steps_share"] == round(9.0 / 1020.0, 4)
    assert stuck_res["sub_boost_loop_len_le_6"]["raw_counts"]["final_stuck_cur_len_le_6_seats"] == 1
    assert stuck_res["sub_boost_loop_len_le_6"]["raw_counts"]["stuck_cur_len_le_6_agent_steps"] == 9


def test_prepare_eval_packet_canary_binding_and_job_sh_execution(tmp_path: Path) -> None:
    """Validate `prepare.py --mode eval` + canary receipt binding + `job.sh` execution."""
    for rel in EVAL_REQUIRED_EXPORTS:
        assert not rel.endswith(".sha256")
        assert not rel.endswith(".complete")
    assert set(EVAL_REQUIRED_EXPORTS) <= set(CANARY_REQUIRED_EXPORTS)

    ckpts = _save_synthetic_arena_checkpoints(tmp_path)
    b8_ckpt = _save_synthetic_b8_checkpoint(tmp_path, seed=808)
    learner_ckpt = ckpts["ego_only"]

    # Missing learner checkpoint when learner_seats > 0 must raise ValueError.
    with pytest.raises(ValueError, match="eval-checkpoint"):
        prepare_arena_ppo_packet(
            tmp_path / "pkt_eval_missing_ckpt",
            commit="HEAD",
            mode="eval",
            gpu="L4",
            eval_opponent="greedy_food",
            eval_learner_seats=1,
            output_dir=tmp_path / "out_eval_missing_ckpt",
            allow_unmerged_commit=True,
        )

    # Incompatible fresh + init_from_checkpoint_state rejected at Mac preflight:
    with pytest.raises(ValueError, match="fresh"):
        prepare_arena_ppo_packet(
            tmp_path / "pkt_eval_bad_fresh",
            commit="HEAD",
            mode="eval",
            gpu="L4",
            eval_checkpoint=learner_ckpt,
            eval_opponent="greedy_food",
            eval_start_mode="fresh",
            eval_init_from_checkpoint_state=True,
            output_dir=tmp_path / "out_eval_bad_fresh",
            allow_unmerged_commit=True,
        )

    # 1. Prepare staged canary packet with learner checkpoint + B8 opponent checkpoint.
    canary_pkt = prepare_arena_ppo_packet(
        tmp_path / "pkt_canary_eval_b8",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        eval_checkpoint=learner_ckpt,
        eval_opponent=f"b8:{b8_ckpt}",
        output_dir=tmp_path / "out_canary_eval_b8",
        allow_unmerged_commit=True,
    )
    mf_canary = json.loads(Path(canary_pkt["manifest_path"]).read_text(encoding="utf-8"))
    colab_schema.validate_manifest(mf_canary)
    ops_dispatch.require_resume_policy(mf_canary)
    colab_supervisor.validate(mf_canary)

    rcpt_l4 = _build_synthetic_canary_receipt(mf_canary, "L4")

    # 2. Prepare `--mode eval` packet bound to that canary receipt with mature checkpoint-state
    # start + custom age bins (`>= 5` bins each).
    eval_pkt = prepare_arena_ppo_packet(
        tmp_path / "pkt_eval_b8",
        commit="HEAD",
        mode="eval",
        gpu="L4",
        eval_checkpoint=learner_ckpt,
        eval_opponent=f"b8:{b8_ckpt}",
        eval_learner_seats=2,
        eval_worlds=2,
        eval_ticks=4,
        eval_start_mode="mature",
        eval_init_from_checkpoint_state=True,
        eval_checkpoint_world_offset=0,
        eval_world_age_bins="0,2,4,8,16",
        eval_life_age_bins="0,2,4,8,16",
        eval_seed=42,
        eval_greedy=True,
        canary_receipt=rcpt_l4,
        output_dir=tmp_path / "out_eval_b8",
        allow_unmerged_commit=True,
    )
    mf_eval = json.loads(Path(eval_pkt["manifest_path"]).read_text(encoding="utf-8"))
    assert "SNAKE_COLAB_CANARY" not in mf_eval["env"]
    assert mf_eval["env"]["ARENA_PPO_MODE"] == "eval"
    assert mf_eval["env"]["ARENA_PPO_EVAL_START_MODE"] == "mature"
    assert mf_eval["env"]["ARENA_PPO_EVAL_INIT_CKPT_STATE"] == "1"
    assert mf_eval["env"]["ARENA_PPO_EVAL_BURNIN"] == "0"
    assert mf_eval["env"]["ARENA_PPO_EVAL_WORLD_AGE_BINS"] == "0,2,4,8,16"
    assert mf_eval["env"]["ARENA_PPO_EVAL_LIFE_AGE_BINS"] == "0,2,4,8,16"
    assert mf_eval["timeout_minutes"] == 30.0
    assert mf_eval["resume_exemption"]["kind"] == "acceptance"
    assert mf_eval["required_exports"] == list(EVAL_REQUIRED_EXPORTS)
    colab_canary.check(mf_eval, ROOT)
    colab_schema.validate_manifest(mf_eval)
    ops_dispatch.require_resume_policy(mf_eval)
    colab_supervisor.validate(mf_eval)

    # 3. Execute `research/arena_ppo/job.sh` end-to-end on CPU in a staged runtime root.
    runtime_root = tmp_path / "colab_runtime_eval"
    runtime_inputs = _stage_manifest_inputs(mf_eval, runtime_root / "inputs")
    (runtime_root / "output").mkdir(parents=True, exist_ok=True)
    run_env = os.environ.copy()
    run_env.update(mf_eval["env"])
    run_env["COLAB_CLI_ROOT"] = str(runtime_root)
    run_env["COLAB_INPUT_DIR"] = str(runtime_inputs)
    run_env["SNAKE_DQN_DEVICE"] = "cpu"
    run_env["PYTHONPATH"] = str(ROOT)

    python_bin_dir = str(Path(sys.executable).resolve().parent)
    run_env["PATH"] = f"{python_bin_dir}:{run_env.get('PATH', '')}"

    proc = subprocess.run(
        ["bash", str(ROOT / "research" / "arena_ppo" / "job.sh")],
        cwd=ROOT,
        env=run_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"job.sh failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
    exported_json = runtime_root / "output" / "eval_result.json"
    assert verify_completed_artifact(exported_json)
    scanned = supervisor_scan_exports(runtime_root, mf_eval["max_export_bytes"])
    scanned_paths = {row["path"] for row in scanned}
    assert "eval_result.json" in scanned_paths
    parsed_res = json.loads(exported_json.read_text(encoding="utf-8"))
    assert parsed_res["seating"]["learner_seats"] == 2
    assert parsed_res["seating"]["learner_greedy"] is True
    assert parsed_res["opponent_info"]["kind"] == "b8"
    assert parsed_res["config"]["start_mode"] == "mature"
    assert parsed_res["config"]["init_from_checkpoint_state"] is True
    assert parsed_res["config"]["world_age_bin_edges"] == [0, 2, 4, 8, 16]
    assert parsed_res["config"]["life_age_bin_edges"] == [0, 2, 4, 8, 16]
    assert parsed_res["learner"]["life_age_bins"][-1]["bin"] == "incumbent_unknown"
