"""Local CPU fixture timing probe for C/T (safe, doom, long-body, topology, and 3-arm fit).

Measures inclusive local 1-thread CPU timings (nice -n 15) on full 83x145 arena,
cap=10,400 fixtures across safe, doom (with and without dead-hero compaction), and
long-body (L=2400) states, plus Numba dynamic-topology extraction and 3-arm fit
minibatches, then projects conservative NVIDIA A100-SXM4-40GB wall-clock envelope
against measured TRAIN-01 A100 multi-mix throughput anchors.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from research.gpu_sim_spike import r2_cache
from research.sandbox_a_resident.ct_branching import score_root_chunk
from research.sandbox_a_resident.ct_fit import (
    float64_masked_kl_from_logits,
    forbidden_margin_loss,
)
from research.sandbox_a_resident.ct_mechanics import CTDeviceSim, CTOpponentRuntime
from research.sandbox_a_resident.ct_models import CT_ARM_NAMES, CTPolicyArm
from research.sandbox_a_resident.ct_snapshots import (
    MAX_LEGAL_BODY_CAPACITY,
    LosslessBoundedSnapshotStore,
    OmitIndexRootMeta,
    capture_world_snapshot,
    observe_hero_bv8,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.ct_task import CTControl
from research.sandbox_a_resident.ct_topology import compute_topology_features
from research.sandbox_a_resident.training_rosters import (
    CHAMPION_SHA256,
    FRP3_S12_SHA256,
    training_rosters,
)
from src.model.apex_network import ApexNetwork
from src.model.ego2s_network import Ego2sNet
from src.simd_env.batch_sim import BatchSimConfig


def _place_folded_snake(
    sim: CTDeviceSim,
    env: int,
    slot: int,
    *,
    head_x: int,
    head_y: int,
    direction: int,
    num_segments: int,
    y_min: int,
    y_max: int,
    x_start: int,
) -> None:
    """Lay a legal non-self-overlapping serpentine snake body of exact ``num_segments``."""
    cells: list[tuple[int, int]] = [(head_x, head_y)]
    cx, cy = x_start, head_y
    going_down = True
    while len(cells) < num_segments:
        if (cx, cy) != (head_x, head_y) and 1 <= cx < sim.grid_w - 1 and 1 <= cy < sim.grid_h - 1:
            cells.append((cx, cy))
        if going_down:
            if cy + 1 <= y_max:
                cy += 1
            else:
                cx += 1
                going_down = False
        else:
            if cy - 1 >= y_min:
                cy -= 1
            else:
                cx += 1
                going_down = True
        if cx >= sim.grid_w - 1:
            cx = 2

    n = len(cells)
    sim.alive[env, slot] = True
    sim.direction[env, slot] = int(direction)
    sim.seg_count[env, slot] = n
    sim.length[env, slot] = n
    sim.head_ptr[env, slot] = n - 1
    body_arr = torch.tensor(list(reversed(cells)), dtype=sim.bodies.dtype, device=sim.device)
    sim.bodies[env, slot, :n] = body_arr


@torch.inference_mode()
def _build_fixture_store(
    kind: str,
    *,
    b8: Ego2sNet,
    ckpts: dict[str, torch.nn.Module],
    num_roots: int = 3,
) -> LosslessBoundedSnapshotStore:
    cfg = BatchSimConfig(
        num_envs=num_roots,
        num_snakes=6,
        max_capacity=MAX_LEGAL_BODY_CAPACITY,
        body_storage_capacity=MAX_LEGAL_BODY_CAPACITY,
        mechanics_version=1,
    )
    sim = CTDeviceSim(cfg, device="cpu", train_mode=False, allow_respawn=True)
    r2_cache.enable(sim)
    sim.alive.fill_(False)
    sim.terminal_slots.fill_(False)
    sim.rng_key.copy_(torch.arange(50_000, 50_000 + num_roots, dtype=torch.long))
    sim.rng_counter.zero_()
    sim.reset_fresh_worlds()
    sim.terminal_slots[:, 0] = True

    rosters = training_rosters(6)[:num_roots]
    for e in range(num_roots):
        sim.bodies[e].zero_()
        sim.seg_count[e].zero_()
        sim.head_ptr[e].zero_()
        if kind == "safe":
            # L=700 hero in open center facing East with distant opponents
            _place_folded_snake(
                sim,
                e,
                0,
                head_x=40,
                head_y=40,
                direction=1,
                num_segments=700,
                y_min=55,
                y_max=80,
                x_start=10,
            )
        elif kind == "doom":
            # L=700 hero at (20, 2) facing North (y=1 is free at h=0 so North & Boost-North are
            # legal at h=0, then Boost-North hits OOB y=0/-1 at h=0 or North hits top wall at h=1!)
            _place_folded_snake(
                sim,
                e,
                0,
                head_x=20,
                head_y=2,
                direction=0,
                num_segments=700,
                y_min=4,
                y_max=37,
                x_start=18,
            )
        elif kind == "long_body":
            # L=2400 hero + long-bodied opponents on full 10,400 ring capacity
            _place_folded_snake(
                sim,
                e,
                0,
                head_x=45,
                head_y=20,
                direction=1,
                num_segments=2400,
                y_min=30,
                y_max=80,
                x_start=5,
            )
        else:
            raise ValueError(f"unknown fixture kind={kind!r}")

        for s in range(1, 6):
            if kind == "doom" and s == 1:
                # Seal East side of (20,2) with an enemy tunnel at x=22 so turning Right
                # (East to 21,2) is legal at h=0 and boxed in within 2 steps!
                _place_folded_snake(
                    sim,
                    e,
                    s,
                    head_x=22,
                    head_y=1,
                    direction=2,
                    num_segments=80,
                    y_min=1,
                    y_max=8,
                    x_start=22,
                )
            else:
                _place_folded_snake(
                    sim,
                    e,
                    s,
                    head_x=70 + s * 10,
                    head_y=10,
                    direction=2,
                    num_segments=200 if kind == "long_body" else 60,
                    y_min=12,
                    y_max=26,
                    x_start=70 + s * 10,
                )

    sim.prepare()
    opp = CTOpponentRuntime(
        sim,
        rosters,
        b8_network=b8,
        checkpoint_networks=ckpts,
        opp_rng_keys=[60_000 + i for i in range(num_roots)],
    )
    local, glob, scalars, legal, q_b8, b8_act = observe_hero_bv8(sim, b8)
    planes, summaries = compute_topology_features(sim)
    store = LosslessBoundedSnapshotStore(
        grid_h=sim.grid_h,
        grid_w=sim.grid_w,
        max_roots=num_roots + 4,
        max_bytes=128 * 1024 * 1024,
        max_body_capacity=MAX_LEGAL_BODY_CAPACITY,
    )
    for e in range(num_roots):
        snap = capture_world_snapshot(
            sim,
            opp,
            e,
            world_id=e,
            life_id=1,
            tick=100,
            max_body_capacity=MAX_LEGAL_BODY_CAPACITY,
        )
        store.add_root(
            meta=OmitIndexRootMeta(
                root_uid=f"probe-{kind}-{e}",
                kind="death_lookback",
                partition="train",
                world_id=e,
                mix=rosters[e].mix,
                roster_index=0,
                life_id=1,
                tick=100,
                hero_length=int(sim.length[e, 0].item()),
                death_event_id=e,
                lookback=100,
                death_tick=200,
                death_length=int(sim.length[e, 0].item()),
                b8_action=int(b8_act[e].item()),
            ),
            snapshot=snap,
            local_obs=local[e],
            glob_obs=glob[e],
            scalars_obs=scalars[e],
            legal_mask=legal[e],
            q_b8=q_b8[e],
            topo_planes_u16=planes[e],
            topo_summaries_f32=summaries[e],
        )
    return store


def run_timing_probe(*, horizon: int = 8, num_roots: int = 1) -> dict[str, Any]:
    """Run local 1-thread CPU fixture timings and compute conservative A100 projections."""
    torch.set_num_threads(1)
    torch.manual_seed(20261028)
    b8 = Ego2sNet(local_channels=7, n_scalars=60).eval()
    ckpts: dict[str, torch.nn.Module] = {
        FRP3_S12_SHA256: ApexNetwork(input_size=61, hidden_size=512, output_size=6).eval(),
        CHAMPION_SHA256: ApexNetwork(input_size=61, hidden_size=512, output_size=6).eval(),
    }

    # Warm up Numba @njit compilation before timing.
    warm_store = _build_fixture_store("safe", b8=b8, ckpts=ckpts, num_roots=1)
    _ = warm_store.finalize_topology()

    fixtures_out: dict[str, Any] = {}
    ctrl = CTControl(
        worlds=6,
        collection_ticks=30,
        max_body_capacity=MAX_LEGAL_BODY_CAPACITY,
        branch_horizon=int(horizon),
        scoring_chunks=1,
        branch_batch_size=128,
    )

    for kind in ("safe", "doom", "long_body"):
        t_build0 = time.perf_counter()
        store = _build_fixture_store(kind, b8=b8, ckpts=ckpts, num_roots=num_roots)
        build_and_topo_sec = time.perf_counter() - t_build0

        # Snapshot restore benchmark on full cap=10,400
        cfg_n = BatchSimConfig(
            num_envs=num_roots,
            num_snakes=6,
            max_capacity=MAX_LEGAL_BODY_CAPACITY,
            body_storage_capacity=MAX_LEGAL_BODY_CAPACITY,
            mechanics_version=1,
        )
        rsim = CTDeviceSim(cfg_n, device="cpu", train_mode=False, allow_respawn=True)
        r2_cache.enable(rsim)
        ropp = CTOpponentRuntime(
            rsim, training_rosters(6)[:num_roots], b8_network=b8, checkpoint_networks=ckpts
        )
        t_rest0 = time.perf_counter()
        restore_world_snapshots_into(store.snapshots, rsim, ropp)
        restore_ms_per_row = ((time.perf_counter() - t_rest0) * 1000.0) / float(num_roots)

        t_topo0 = time.perf_counter()
        for _ in range(4):
            compute_topology_features(rsim)
        topo_ms_per_row = ((time.perf_counter() - t_topo0) * 1000.0) / float(4 * num_roots)

        chunk_compact = score_root_chunk(
            chunk_index=0,
            root_indices=list(range(num_roots)),
            store=store,
            control=ctrl,
            b8_network=b8,
            checkpoint_networks=ckpts,
            device="cpu",
            compaction_interval=2,
        )
        assert chunk_compact is not None
        rec: dict[str, Any] = {
            "hero_length": store.metadata[0].hero_length,
            "snapshot_bytes_per_root": store.snapshots[0].nbytes,
            "restore_ms_per_row_cpu_1t": round(restore_ms_per_row, 3),
            "topology_ms_per_row_cpu_1t": round(topo_ms_per_row, 3),
            "build_and_topo_sec": round(build_and_topo_sec, 3),
            "branch_horizon_measured": int(horizon),
            "logical_branch_steps": chunk_compact.telemetry["logical_branch_steps"],
            "executed_sim_steps_compacted": chunk_compact.telemetry["executed_sim_steps"],
            "compaction_step_savings_ratio": chunk_compact.telemetry[
                "compaction_step_savings_ratio"
            ],
            "wall_sec_compacted_cpu_1t": chunk_compact.telemetry["wall_sec"],
            "logical_steps_per_sec_cpu_1t": round(
                chunk_compact.telemetry["logical_branch_steps"]
                / max(1e-6, chunk_compact.telemetry["wall_sec"]),
                2,
            ),
            "executed_steps_per_sec_cpu_1t": round(
                chunk_compact.telemetry["executed_sim_steps"]
                / max(1e-6, chunk_compact.telemetry["wall_sec"]),
                2,
            ),
        }
        if kind == "doom":
            chunk_dense = score_root_chunk(
                chunk_index=0,
                root_indices=list(range(num_roots)),
                store=store,
                control=ctrl,
                b8_network=b8,
                checkpoint_networks=ckpts,
                device="cpu",
                compaction_interval=0,
            )
            assert chunk_dense is not None
            assert chunk_compact.selection_freeze_sha256 == chunk_dense.selection_freeze_sha256
            rec["wall_sec_uncompacted_cpu_1t"] = chunk_dense.telemetry["wall_sec"]
            rec["compaction_exact_parity"] = True
        fixtures_out[kind] = rec

    # 3-arm distillation minibatch CPU timing (scaled batch B_s=16, B_a=48 on 1-thread CPU)
    fit_timings: dict[str, float] = {}
    rng = torch.Generator().manual_seed(7)
    s_loc = torch.randint(0, 2, (16, 7, 31, 31), generator=rng, dtype=torch.uint8)
    s_glb = torch.randint(0, 2, (16, 4, 37, 37), generator=rng, dtype=torch.uint8)
    s_scl = torch.randn((16, 60), generator=rng, dtype=torch.float32)
    s_leg = torch.ones((16, 6), dtype=torch.bool)
    s_tgt = torch.randn((16, 6), generator=rng, dtype=torch.float64)
    s_ast = torch.zeros(16, dtype=torch.long)
    s_frb = torch.zeros((16, 6), dtype=torch.bool)
    s_frb[:, 2] = True
    s_tp = torch.rand((16, 4, 83, 145), generator=rng, dtype=torch.float32)
    s_ts = torch.rand((16, 6), generator=rng, dtype=torch.float32)

    a_loc = torch.randint(0, 2, (48, 7, 31, 31), generator=rng, dtype=torch.uint8)
    a_glb = torch.randint(0, 2, (48, 4, 37, 37), generator=rng, dtype=torch.uint8)
    a_scl = torch.randn((48, 60), generator=rng, dtype=torch.float32)
    a_leg = torch.ones((48, 6), dtype=torch.bool)
    a_tgt = torch.randn((48, 6), generator=rng, dtype=torch.float64)
    a_tp = torch.rand((48, 4, 83, 145), generator=rng, dtype=torch.float32)
    a_ts = torch.rand((48, 6), generator=rng, dtype=torch.float32)

    for arm_name in CT_ARM_NAMES:
        arm = CTPolicyArm(b8, arm_name).train()
        opt = arm.build_optimizer(trunk_lr=1e-5, head_lr=1e-4)
        needs_t = arm_name == "full_bv8_plus_topology"
        t_f0 = time.perf_counter()
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            zs = arm(
                s_loc,
                s_glb,
                s_scl,
                topo_planes=s_tp if needs_t else None,
                topo_summaries=s_ts if needs_t else None,
            )
            za = arm(
                a_loc,
                a_glb,
                a_scl,
                topo_planes=a_tp if needs_t else None,
                topo_summaries=a_ts if needs_t else None,
            )
            loss = (
                float64_masked_kl_from_logits(s_tgt, zs, s_leg).mean()
                + forbidden_margin_loss(zs, s_ast, s_frb, margin=0.25).mean()
                + 0.05 * float64_masked_kl_from_logits(a_tgt, za, a_leg).mean()
            )
            loss.backward()
            opt.step()
        fit_timings[arm_name] = round(((time.perf_counter() - t_f0) * 1000.0) / 3.0, 2)

    # Conservative NVIDIA A100-SXM4-40GB projection anchored to measured colab-limits telemetry:
    # - Measured TRAIN-01 3-mix E1024 on A100: 1,945.6 - 1,996.8 rows/s (cap=2,000);
    #   with cap=10,400 + LookbackRing + 52,560 topology rows (~52,560 * topo_ms):
    #   conservative rate = 1,250 rows/s -> 2,621,440 / 1,250 = 2,097 s (~35.0 min) + ~2.5 min CPU.
    # - Counterfactual H300 branching (<=2,560 roots * <=6 actions * 8 suffixes * 300 = 36,864,000;
    #   typical ~4.8 legal actions = 29,491,200; 15% dead-hero compaction -> ~25.1M-31.3M steps):
    #   homogeneous single-roster B=512 slabs on A100 run between E360 (863/s) and E1024 (1946/s);
    #   conservative 1,150 steps/s -> worst-case 36.86M / 1,150 = 32,055 s = 8.90 h;
    #   expected ~26.5M executed steps @ 1,350/s = 19,630 s = 5.45 h.
    # - 3-arm distillation fit: 3 arms * 2,000 minibatches * 1,024 rows = 6,144,000 learner rows;
    #   on A100 (~35k-80k rows/s including T-arm (4,83,145) conv tower) = ~3-6 min + bootstrap 1 m.
    topo_52k_sec_1t = (52_560.0 * fixtures_out["long_body"]["topology_ms_per_row_cpu_1t"]) / 1000.0
    a100_projection = {
        "anchor_sources": {
            "train01_a100_e1024_3mix_rows_per_sec": 1945.6,
            "train01_a100_e360_3mix_rows_per_sec": 862.85,
            "a100_burn_rate_units_per_hour": 6.87,
        },
        "stage1_collection_and_topology_minutes": {
            "expected": round(35.0 + (topo_52k_sec_1t / 60.0), 1),
            "conservative_upper": 55.0,
        },
        "stages2_to_9_h300_branching_hours": {
            "max_logical_steps": 36_864_000,
            "expected_executed_steps_with_compaction": 26_500_000,
            "expected_hours_at_1350_steps_per_sec": round(26_500_000 / 1350.0 / 3600.0, 2),
            "conservative_upper_hours_at_1150_steps_per_sec_full_36p86m": round(
                36_864_000 / 1150.0 / 3600.0, 2
            ),
        },
        "stage10_three_arm_fit_and_gates_minutes": {
            "expected": 8.0,
            "conservative_upper": 18.0,
        },
        "bootstrap_upload_export_overhead_minutes": {
            "expected": 12.0,
            "conservative_upper": 25.0,
        },
        "total_projected_a100_hours": {
            "expected_hours": round(
                (35.0 + (topo_52k_sec_1t / 60.0) + 8.0 + 12.0) / 60.0
                + (26_500_000 / 1350.0 / 3600.0),
                2,
            ),
            "conservative_upper_hours": round(
                (55.0 + 18.0 + 25.0) / 60.0 + (36_864_000 / 1150.0 / 3600.0),
                2,
            ),
            "base_estimate_hours": 12.0,
            "effective_2x_timeout_hours": 24.0,
            "effective_2x_unit_cap": 193.0,
        },
    }

    return {
        "schema": "t34-cf-local-fixture-timing-probe/v1",
        "cpu_fixtures": fixtures_out,
        "cpu_fit_minibatch_64rows_ms": fit_timings,
        "projected_52560_rows_topology_cpu_1t_sec": round(topo_52k_sec_1t, 2),
        "a100_projection": a100_projection,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--num-roots", type=int, default=1)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()
    report = run_timing_probe(horizon=args.horizon, num_roots=args.num_roots)
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.write_text(rendered + "\n")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
