"""Fast hermetic tests for ``research.v3_ppo.arena_eval`` and ``prepare --mode eval``."""

from __future__ import annotations

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

from research.colab_cli import schema as colab_schema
from research.colab_cli import supervisor as colab_supervisor
from research.game_v3.config import FP_ONE, GameV3Config
from research.game_v3.obs import allocate_obs_batch, build_obs
from research.game_v3.sim import GameV3Sim
from research.v3_ppo.arena_eval import (
    DEATH_CAUSE_NAMES,
    TURN_LEVEL_NAMES,
    V3_ARENA_EVAL_SCHEMA,
    act_policy_chunked,
)
from research.v3_ppo.arena_eval import build_arg_parser as build_eval_arg_parser
from research.v3_ppo.arena_eval import (
    build_rotated_seat_assignment,
    classify_step_deaths,
    discover_round_robin_checkpoints,
    evaluate_head_to_head,
    evaluate_long_game,
    evaluate_round_robin,
    fit_bradley_terry_elo,
    format_eval_markdown,
    load_eval_policy,
)
from research.v3_ppo.arena_eval import main as arena_eval_main
from research.v3_ppo.arena_eval import (
    write_weights_only_eval_bundle,
)
from research.v3_ppo.config import V3PPOConfig
from research.v3_ppo.prepare import (
    ROOT,
)
from research.v3_ppo.prepare import main as prepare_main
from research.v3_ppo.prepare import (
    prepare_v3_ppo_packet,
    runtime_source_paths,
)
from research.v3_ppo.trainer import V3PPOTrainer
from tests.test_v3_ppo_packet import (
    _build_synthetic_canary_receipt,
    _walk_transitive_project_imports,
)
from tools.ops import dispatch as ops_dispatch


def _make_tiny_light_checkpoint(
    out_dir: Path,
    *,
    name: str,
    net_size: str = "S",
    seed: int = 101,
    arena_radius: float = 80.0,
    num_snakes: int = 6,
    actor_chunk_learners: int | None = 4,
    reward_alive_const: float = 0.0,
) -> tuple[Path, V3PPOTrainer]:
    """Create a real verified LIGHT checkpoint using ``V3PPOTrainer.save_checkpoint``."""
    cfg = V3PPOConfig(
        game_config=GameV3Config(arena_radius=arena_radius, num_snakes=num_snakes),
        num_envs=2,
        learner_slots=num_snakes,
        net_size=net_size,  # type: ignore[arg-type]
        rollout_len=4,
        ppo_epochs=1,
        minibatch_size=8,
        actor_chunk_learners=actor_chunk_learners,
        amp=False,
        use_channels_last=False,
        deterministic=True,
        seed=seed,
        reward_alive_const=reward_alive_const,
        eval_every_minutes=0.0,
        eval_every_updates=0,
    )
    trainer = V3PPOTrainer(cfg, device="cpu")
    trainer.train_step()
    ckpt_path = out_dir / name / "checkpoints" / "light_u0000001.pt"
    trainer.save_light_checkpoint(ckpt_path)
    return ckpt_path, trainer


def _strip_wall_times(obj: Any) -> Any:
    """Recursively strip wall-clock timing keys for exact determinism comparison."""
    if isinstance(obj, dict):
        return {
            k: _strip_wall_times(v)
            for k, v in obj.items()
            if k not in {"wall_seconds", "world_ticks_per_sec", "agent_steps_per_sec"}
        }
    if isinstance(obj, list):
        return [_strip_wall_times(x) for x in obj]
    return obj


def test_action_parity_with_trainer_sampled_and_greedy(tmp_path: Path) -> None:
    """``arena_eval`` policy loading, obs building, and ``act_policy_chunked`` match trainer bit-for-bit."""
    ckpt_path, trainer = _make_tiny_light_checkpoint(
        tmp_path,
        name="parity_s",
        net_size="S",
        seed=2026,
        num_snakes=6,
        actor_chunk_learners=4,
    )
    loaded = load_eval_policy(ckpt_path, device="cpu", label="parity-s")

    # Force mixed boost-mask eligibility (some snakes == min_boost_length, some > min_boost_length)
    min_bl = int(trainer.sim.cfg.min_boost_length)
    trainer.sim.length[0, 0] = min_bl
    trainer.sim.length[0, 1] = min_bl + 15
    trainer.sim.length[1, 2] = min_bl + 40

    full_obs = allocate_obs_batch(trainer.sim)
    learner_obs = allocate_obs_batch(trainer.sim, trainer.learner_index)

    for step_idx in range(4):
        build_obs(trainer.sim, learner_index=trainer.learner_index, out=learner_obs)
        build_obs(trainer.sim, out=full_obs)

        # Full-world obs flattened in row-major (env, slot) order must equal learner_index obs
        assert torch.equal(full_obs.raster_s1.flatten(0, 1), learner_obs.raster_s1)
        assert torch.equal(full_obs.raster_s4.flatten(0, 1), learner_obs.raster_s4)
        assert torch.equal(full_obs.raster_s16.flatten(0, 1), learner_obs.raster_s16)
        assert torch.equal(full_obs.scalars.flatten(0, 1), learner_obs.scalars)
        assert torch.equal(full_obs.action_mask.flatten(0, 1), learner_obs.action_mask)

        # Sampled action parity on identical RNG seed
        rng_seed = 90_000 + step_idx
        torch.manual_seed(rng_seed)
        tr_sampled, _, _ = trainer._act_learners_chunked(
            learner_obs.raster_s1,
            learner_obs.raster_s4,
            learner_obs.raster_s16,
            learner_obs.scalars,
            learner_obs.action_mask,
            greedy=False,
        )

        gen = torch.Generator(device="cpu")
        gen.manual_seed(rng_seed)
        ev_sampled = act_policy_chunked(
            loaded,
            full_obs.raster_s1.flatten(0, 1),
            full_obs.raster_s4.flatten(0, 1),
            full_obs.raster_s16.flatten(0, 1),
            full_obs.scalars.flatten(0, 1),
            full_obs.action_mask.flatten(0, 1),
            greedy=False,
            generator=gen,
        )
        assert torch.equal(ev_sampled, tr_sampled)

        # Greedy action parity
        tr_greedy, _, _ = trainer._act_learners_chunked(
            learner_obs.raster_s1,
            learner_obs.raster_s4,
            learner_obs.raster_s16,
            learner_obs.scalars,
            learner_obs.action_mask,
            greedy=True,
        )
        ev_greedy = act_policy_chunked(
            loaded,
            learner_obs.raster_s1,
            learner_obs.raster_s4,
            learner_obs.raster_s16,
            learner_obs.scalars,
            learner_obs.action_mask,
            greedy=True,
        )
        assert torch.equal(ev_greedy, tr_greedy)

        # Advance simulator to test next state
        trainer.sim.step(ev_sampled.view(trainer.sim.E, trainer.sim.N))


def test_seat_rotation_balances_seats_and_spawn_slots() -> None:
    """Seat assignment splits 24 seats evenly and rotates every spawn slot across policies."""
    num_snakes = 24
    for num_policies in (2, 3, 4, 5, 6):
        num_envs = 2 * num_policies
        assign = build_rotated_seat_assignment(
            num_envs, num_snakes, num_policies, device="cpu"
        ).numpy()
        assert assign.shape == (num_envs, num_snakes)

        # Within every world, policy seat counts differ by at most 1 (0 when P divides 24)
        for w in range(num_envs):
            counts = np.bincount(assign[w], minlength=num_policies)
            if num_snakes % num_policies == 0:
                assert np.all(counts == (num_snakes // num_policies))
            else:
                assert int(counts.max() - counts.min()) == 1
                assert set(counts.tolist()) == {4, 5}

        # Across 2*P worlds, every policy occupies every spawn slot s the exact same number of times
        for slot in range(num_snakes):
            slot_counts = np.bincount(assign[:, slot], minlength=num_policies)
            assert np.all(slot_counts == 2)

        # Adjacent worlds never have identical seat maps
        assert not np.array_equal(assign[0], assign[1])


def test_controlled_collisions_kill_attribution_and_death_causes() -> None:
    """Controlled collisions classify head_on, head_into_body, and wall deaths accurately."""
    cfg = GameV3Config(arena_radius=100.0, num_snakes=6)
    sim = GameV3Sim(cfg, num_envs=3, device="cpu", seeds=[11, 22, 33])

    # Park unused snakes safely far from origin so they cannot interfere
    half_ang = cfg.angle_steps // 2
    quarter_ang = cfg.angle_steps // 4
    park_coords_cells = [(-50, -50), (-50, 50), (50, -50), (50, 50), (-50, 25), (50, -25)]
    for e in range(3):
        for s in range(6):
            px, py = park_coords_cells[s]
            sim.head_pos[e, s, 0] = px * FP_ONE
            sim.head_pos[e, s, 1] = py * FP_ONE
            sim.theta[e, s] = 0
            sim.dist_accum_fp[e, s] = 0

    # Advance 14 ticks so parked snakes' bodies only occupy their parked corners,
    # while positioning world 1 snake 1 along y = 0 so it lays a clean horizontal body.
    sim.head_pos[1, 1, 0] = -8 * FP_ONE
    sim.head_pos[1, 1, 1] = 0
    sim.theta[1, 1] = 0  # moving +x along y = 0
    sim.length[1, 1] = 30
    straight_all = torch.full((3, 6), 2, dtype=torch.long)
    for _ in range(14):
        sim.step(straight_all)

    # World 0: Head-on collision at origin between snake 0 (+x) and snake 1 (-x)
    spd_fp = int(sim._speed_fp_table[int(sim.length[0, 0].item())].item())
    sim.head_pos[0, 0, 0] = -(spd_fp // 2)
    sim.head_pos[0, 0, 1] = 0
    sim.theta[0, 0] = 0
    sim.dist_accum_fp[0, 0] = spd_fp  # guarantees point emission on this tick

    sim.head_pos[0, 1, 0] = spd_fp // 2
    sim.head_pos[0, 1, 1] = 0
    sim.theta[0, 1] = half_ang
    sim.dist_accum_fp[0, 1] = spd_fp

    # World 1: Snake 0 positioned just below snake 1's mid-body at y = 0, facing +y into body
    body_mid_x_fp = int(sim.head_pos[1, 1, 0].item()) - 6 * FP_ONE
    sim.head_pos[1, 0, 0] = body_mid_x_fp
    sim.head_pos[1, 0, 1] = -(spd_fp // 2)
    sim.theta[1, 0] = quarter_ang  # +y into snake 1's retained body along y = 0
    sim.dist_accum_fp[1, 0] = spd_fp

    # World 2: Snake 0 right at the +x arena wall facing +x
    sim.head_pos[2, 0, 0] = int(cfg.arena_radius_fp) - (spd_fp // 4)
    sim.head_pos[2, 0, 1] = 0
    sim.theta[2, 0] = 0
    sim.dist_accum_fp[2, 0] = spd_fp

    acted_es = sim.alive & (~sim.truncated_pending)
    pre_hx = sim.head_pos[:, :, 0].clone()
    pre_hy = sim.head_pos[:, :, 1].clone()
    pre_theta = sim.theta.clone()
    pre_len = sim.length.clone()
    pre_accum = sim.dist_accum_fp.clone()
    pre_bt = sim.boost_ticks.clone()

    info = sim.step(straight_all)
    causes = classify_step_deaths(
        sim,
        acted_es=acted_es,
        world_actions=straight_all,
        pre_hx=pre_hx,
        pre_hy=pre_hy,
        pre_theta=pre_theta,
        pre_len=pre_len,
        pre_accum=pre_accum,
        pre_boost_ticks=pre_bt,
        died=info.died,
        killer_id=info.killer_id,
    )

    # World 0: mutual head-on kills both snake 0 and snake 1, crediting each other
    assert bool(info.died[0, 0].item()) is True
    assert bool(info.died[0, 1].item()) is True
    assert int(info.killer_id[0, 0].item()) == 1
    assert int(info.killer_id[0, 1].item()) == 0
    assert DEATH_CAUSE_NAMES[int(causes[0, 0].item())] == "head_on"
    assert DEATH_CAUSE_NAMES[int(causes[0, 1].item())] == "head_on"

    # World 1: snake 0 dies hitting snake 1's body; snake 1 survives and gets +1 kill credit
    assert bool(info.died[1, 0].item()) is True
    assert bool(info.died[1, 1].item()) is False
    assert int(info.killer_id[1, 0].item()) == 1
    assert int(info.kill_credit[1, 1].item()) == 1
    assert DEATH_CAUSE_NAMES[int(causes[1, 0].item())] == "head_into_body"
    assert int(causes[1, 1].item()) == -1

    # World 2: snake 0 dies hitting arena wall (killer_id == -1)
    assert bool(info.died[2, 0].item()) is True
    assert int(info.killer_id[2, 0].item()) == -1
    assert DEATH_CAUSE_NAMES[int(causes[2, 0].item())] == "wall"


def test_cpu_determinism_heterogeneous_nets_and_provenance(tmp_path: Path) -> None:
    """Repeated CPU runs with identical (checkpoints, seed, config) are bit-for-bit identical."""
    ckpt_s, _ = _make_tiny_light_checkpoint(
        tmp_path,
        name="pol_s",
        net_size="S",
        seed=11,
        arena_radius=70.0,
        num_snakes=6,
        reward_alive_const=0.01,
    )
    ckpt_m, _ = _make_tiny_light_checkpoint(
        tmp_path,
        name="pol_m",
        net_size="M",
        seed=22,
        arena_radius=70.0,
        num_snakes=6,
        reward_alive_const=0.0,
    )

    pol_s = load_eval_policy(ckpt_s, device="cpu", label="small-rwa")
    pol_m = load_eval_policy(ckpt_m, device="cpu", label="medium-def")

    h2h_1 = evaluate_head_to_head(
        [pol_s, pol_m],
        num_envs=4,
        burnin_ticks=6,
        window_ticks=14,
        mass_bin_ticks=5,
        greedy=False,
        seed=424242,
        bootstrap_samples=200,
    )
    # Perturb global RNG state between runs to prove isolation
    torch.manual_seed(999)
    np.random.seed(888)
    h2h_2 = evaluate_head_to_head(
        [pol_s, pol_m],
        num_envs=4,
        burnin_ticks=6,
        window_ticks=14,
        mass_bin_ticks=5,
        greedy=False,
        seed=424242,
        bootstrap_samples=200,
    )
    assert _strip_wall_times(h2h_1) == _strip_wall_times(h2h_2)

    # Greedy mode determinism + weights-only bundle parity
    bundle_m = tmp_path / "pol_m.eval.pt"
    write_weights_only_eval_bundle(ckpt_m, bundle_m)
    assert bundle_m.stat().st_size < ckpt_m.stat().st_size
    pol_m_bundle = load_eval_policy(bundle_m, device="cpu", label="medium-def")
    assert pol_m_bundle.source_checkpoint_sha256 == pol_m.source_checkpoint_sha256

    h2h_bundle = evaluate_head_to_head(
        [pol_s, pol_m_bundle],
        num_envs=4,
        burnin_ticks=6,
        window_ticks=14,
        mass_bin_ticks=5,
        greedy=False,
        seed=424242,
        bootstrap_samples=200,
    )
    assert _strip_wall_times(h2h_bundle)["per_policy"] == _strip_wall_times(h2h_1)["per_policy"]
    assert _strip_wall_times(h2h_bundle)["pairwise"] == _strip_wall_times(h2h_1)["pairwise"]

    # Also verify compatibility with older v2 LIGHT checkpoints that omit top-level obs_layout
    legacy_ckpt = tmp_path / "pol_m_no_obs_layout.pt"
    raw_m = torch.load(ckpt_m, map_location="cpu", weights_only=False)
    raw_m.pop("obs_layout", None)
    torch.save(raw_m, legacy_ckpt)
    pol_m_legacy = load_eval_policy(legacy_ckpt, device="cpu", label="medium-def")
    assert pol_m_legacy.obs_layout == pol_m.obs_layout

    # Provenance & kill-matrix accounting invariants
    assert h2h_1["schema"] == V3_ARENA_EVAL_SCHEMA
    assert len(h2h_1["policies"]) == 2
    assert h2h_1["policies"][0]["net_size"] == "S"
    assert h2h_1["policies"][1]["net_size"] == "M"
    assert h2h_1["policies"][0]["training_recipe"]["reward_alive_const"] == pytest.approx(0.01)
    assert h2h_1["policies"][0]["recipe_overrides_from_default"][
        "reward_alive_const"
    ] == pytest.approx(0.01)
    assert "reward_alive_const" not in h2h_1["policies"][1]["recipe_overrides_from_default"]
    assert h2h_1["policies"][0]["updates_completed"] == 1
    assert h2h_1["policies"][0]["agent_steps_collected"] > 0

    km = h2h_1["kill_matrix"]["victim_by_killer"]
    for p_row in h2h_1["per_policy"]:
        lbl = p_row["label"]
        assert km[lbl]["self"] == 0
        assert km[lbl]["total_deaths"] == p_row["deaths"]


def test_long_game_life_age_world_age_bins_and_round_robin(tmp_path: Path) -> None:
    """Long-game life-age/world-age bins partition window steps cleanly; round-robin resumes pairs."""
    ckpt_a, _ = _make_tiny_light_checkpoint(
        tmp_path, name="rr_a", net_size="S", seed=1, arena_radius=60.0, num_snakes=6
    )
    ckpt_b, _ = _make_tiny_light_checkpoint(
        tmp_path, name="rr_b", net_size="S", seed=2, arena_radius=60.0, num_snakes=6
    )
    ckpt_c, _ = _make_tiny_light_checkpoint(
        tmp_path, name="rr_c", net_size="S", seed=3, arena_radius=60.0, num_snakes=6
    )

    pol_a = load_eval_policy(ckpt_a, device="cpu", label="pol-a")
    lg = evaluate_long_game(
        pol_a,
        num_envs=3,
        burnin_ticks=10,
        window_ticks=24,
        learner_slots=4,
        opponent_policy="food_seeking",
        num_world_age_bins=6,
        life_age_edges=(4, 8, 14, 20, 28),
        victim_length_edges=(12, 20, 40),
        seed=555,
    )
    lg_repeat = evaluate_long_game(
        pol_a,
        num_envs=3,
        burnin_ticks=10,
        window_ticks=24,
        learner_slots=4,
        opponent_policy="food_seeking",
        num_world_age_bins=6,
        life_age_edges=(4, 8, 14, 20, 28),
        victim_length_edges=(12, 20, 40),
        seed=555,
    )
    assert _strip_wall_times(lg) == _strip_wall_times(lg_repeat)

    win_wbins = [b for b in lg["world_age_bins"] if b["phase"] == "window"]
    assert len(win_wbins) == 6
    assert len(lg["life_age_bins"]) == 6

    ws = lg["window_summary"]
    max_possible_steps = 3 * 4 * 24
    assert 0 < ws["acted_steps"] <= max_possible_steps
    assert sum(b["acted_steps"] for b in win_wbins) == ws["acted_steps"]
    assert sum(b["acted_steps"] for b in lg["life_age_bins"]) == ws["acted_steps"]
    assert sum(b["deaths"] for b in lg["life_age_bins"]) == ws["deaths"]
    assert lg["death_causes"]["overall"]["total_deaths"] == ws["deaths"]
    assert lg["death_causes"]["overall"]["counts"]["self"] == 0

    for b in lg["life_age_bins"]:
        assert sum(b["turn_counts"][t] for t in TURN_LEVEL_NAMES) == b["acted_steps"]

    md = format_eval_markdown(lg)
    assert "Metrics Binned by Snake Life Age" in md
    assert "Death Causes by Victim Length Bin" in md

    # Round-robin across 3 policies + pair-unit resume skip
    list_file = tmp_path / "policies.txt"
    list_file.write_text(f"{ckpt_a}\n{ckpt_b}\n{ckpt_c}\n", encoding="utf-8")
    discovered = discover_round_robin_checkpoints([list_file])
    assert len(discovered) == 3

    rr_out = tmp_path / "rr_out"
    pols = [load_eval_policy(p, device="cpu") for p in discovered]
    rr1 = evaluate_round_robin(
        pols,
        num_envs=2,
        burnin_ticks=4,
        window_ticks=10,
        bootstrap_samples=100,
        seed=777,
        output_dir=rr_out,
    )
    assert len(rr1["pairs"]) == 3
    assert len(rr1["reused_pair_units"]) == 0
    elos = [row["elo"] for row in rr1["standings"]]
    assert np.mean(elos) == pytest.approx(1500.0, abs=0.05)

    # Re-running against verified pair artifacts reuses all 3 pairs identically
    verified_map = {
        f"pair_{i:02d}_{j:02d}": rr_out / "pairs" / f"pair_{i:02d}_{j:02d}.json"
        for i in range(3)
        for j in range(i + 1, 3)
    }
    rr2 = evaluate_round_robin(
        pols,
        num_envs=2,
        burnin_ticks=4,
        window_ticks=10,
        bootstrap_samples=100,
        seed=777,
        verified_pair_paths=verified_map,
    )
    assert sorted(rr2["reused_pair_units"]) == ["pair_00_01", "pair_00_02", "pair_01_02"]
    assert rr2["standings"] == rr1["standings"]

    # Direct Bradley-Terry check on a known win matrix (handles complete separation finitely)
    w_sep = np.array([[0.0, 10.0], [0.0, 0.0]])
    g_sep = np.array([[0.0, 10.0], [10.0, 0.0]])
    bt_elo = fit_bradley_terry_elo(w_sep, g_sep)
    assert np.all(np.isfinite(bt_elo))
    assert bt_elo[0] > bt_elo[1]
    assert np.mean(bt_elo) == pytest.approx(1500.0, abs=1e-6)


def test_prepare_eval_packet_and_import_walker(tmp_path: Path) -> None:
    """``prepare --mode eval`` builds a valid manifest and ``runtime_source_paths`` covers all imports."""
    shipped = set(runtime_source_paths("HEAD"))
    assert "research/v3_ppo/arena_eval.py" in shipped
    imported = _walk_transitive_project_imports(
        [
            "research/v3_ppo/ppo.py",
            "research/v3_ppo/ruler.py",
            "research/v3_ppo/arena_eval.py",
        ]
    )
    missing = sorted(imported - shipped)
    assert not missing, f"Entrypoint imports missing from runtime_source_paths: {missing}"

    # Stage ONLY shipped files into an isolated directory and execute arena_eval inside it
    isolated = tmp_path / "isolated_eval_tree"
    for rel in sorted(shipped):
        dst = isolated / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, dst)

    ckpt_1, _ = _make_tiny_light_checkpoint(
        tmp_path, name="iso_p1", net_size="S", seed=10, arena_radius=60.0, num_snakes=6
    )
    ckpt_2, _ = _make_tiny_light_checkpoint(
        tmp_path, name="iso_p2", net_size="S", seed=20, arena_radius=60.0, num_snakes=6
    )
    iso_out = tmp_path / "iso_eval_out"
    iso_probe = (
        "import pathlib, sys\n"
        f"repo_root = pathlib.Path({str(ROOT)!r}).resolve()\n"
        f"iso_root = pathlib.Path({str(isolated)!r}).resolve()\n"
        "sys.path[:] = [\n"
        "    str(iso_root),\n"
        "    *(p for p in sys.path if p and pathlib.Path(p).resolve() != repo_root),\n"
        "]\n"
        "import research.v3_ppo.arena_eval as ae\n"
        "assert pathlib.Path(ae.__file__).resolve().is_relative_to(iso_root)\n"
        f"rc = ae.main(['--policies', {str(ckpt_1)!r}, {str(ckpt_2)!r}, "
        f"'--num-envs', '2', '--burnin-ticks', '2', '--window-ticks', '6', "
        f"'--bootstrap-samples', '50', '--output-dir', {str(iso_out)!r}])\n"
        "assert rc == 0\n"
    )
    subprocess.run(
        [sys.executable, "-c", iso_probe],
        cwd=str(isolated),
        env={
            **os.environ,
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPATH": str(isolated),
        },
        check=True,
        capture_output=True,
        text=True,
    )
    assert (iso_out / "eval_result.json.complete").is_file()
    assert (iso_out / "summary.md.complete").is_file()

    # Verify weights-only eval bundle serialization is bit-for-bit deterministic across paths
    wo_sha_a = write_weights_only_eval_bundle(ckpt_1, tmp_path / "det_a" / "p0.eval.pt")
    wo_sha_b = write_weights_only_eval_bundle(ckpt_1, tmp_path / "det_b" / "other_name.eval.pt")
    assert wo_sha_a == wo_sha_b

    six_ckpts = [ckpt_1, ckpt_2]
    for extra_i in range(4):
        c_extra, _ = _make_tiny_light_checkpoint(
            tmp_path,
            name=f"iso_extra_{extra_i}",
            net_size="S",
            seed=30 + extra_i,
            arena_radius=60.0,
            num_snakes=6,
        )
        six_ckpts.append(c_extra)

    # Build canary receipt on L4 covering the staged eval inputs & exports, then bind 6-policy RR
    canary_l4 = prepare_v3_ppo_packet(
        tmp_path / "pkt_canary_l4",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        eval_mode="round-robin",
        eval_policies=six_ckpts,
        eval_weights_only="true",
        output_dir=tmp_path / "out_canary_l4",
        allow_unmerged_commit=True,
    )
    mf_canary_l4 = json.loads(Path(canary_l4["manifest_path"]).read_text(encoding="utf-8"))
    rcpt_l4 = _build_synthetic_canary_receipt(mf_canary_l4, "L4")

    pkt_rr = tmp_path / "pkt_eval_rr6"
    out_rr = tmp_path / "out_eval_rr6"
    summary = prepare_v3_ppo_packet(
        pkt_rr,
        commit="HEAD",
        mode="eval",
        gpu="L4",
        eval_mode="round-robin",
        eval_policies=six_ckpts,
        eval_weights_only="true",
        canary_receipt=rcpt_l4,
        output_dir=out_rr,
        allow_unmerged_commit=True,
        attempt="rr6-test",
    )
    assert summary["mode"] == "eval"
    assert summary["est_minutes"] <= 60.0
    assert summary["timeout_minutes"] >= 2.0 * summary["est_minutes"]
    assert summary["eval_plan"]["num_policies"] == 6
    assert summary["eval_plan"]["total_pairs"] == 15

    manifest = json.loads((pkt_rr / "dispatch-manifest.json").read_text(encoding="utf-8"))
    colab_schema.validate_manifest(manifest)
    colab_schema.required_export_paths(manifest)
    ops_dispatch.require_resume_policy(manifest)
    colab_supervisor.validate(manifest)
    assert manifest["env"]["V3_PPO_MODE"] == "eval"
    assert manifest["env"]["V3_PPO_EVAL_MODE"] == "round-robin"
    assert set(manifest["required_exports"]) == {"eval_result.json", "summary.md"}

    # Verify job.sh maps every V3_PPO_EVAL_* env var into valid arena_eval CLI arguments
    job_sh_path = ROOT / "research" / "v3_ppo" / "job.sh"
    job_sh_text = job_sh_path.read_text(encoding="utf-8")
    for env_key in manifest["env"]:
        if env_key != "COLAB_CLI_REQUIREMENTS":
            assert env_key in job_sh_text, f"{env_key} missing from job.sh"
    stub_dir = tmp_path / "job_sh_eval_stub"
    (stub_dir / "bin").mkdir(parents=True)
    cap_json = stub_dir / "cap.json"
    stub_py = stub_dir / "bin" / "python"
    stub_py.write_text(
        "#!/usr/bin/env bash\n"
        f"{sys.executable} -c 'import json, sys; "
        f'open({json.dumps(str(cap_json))}, "w").write(json.dumps(sys.argv[1:]))\' "$@"\n',
        encoding="utf-8",
    )
    stub_py.chmod(0o755)
    cli_root = stub_dir / "cli_root"
    (cli_root / "output").mkdir(parents=True)
    (cli_root / "work").mkdir(parents=True)
    subprocess.run(
        ["bash", str(job_sh_path)],
        env={
            "PATH": f"{stub_dir / 'bin'}:{os.environ.get('PATH', '/usr/bin:/bin')}",
            "HOME": os.environ.get("HOME", "/tmp"),
            "COLAB_CLI_ROOT": str(cli_root),
            "COLAB_INPUT_DIR": str(pkt_rr / "inputs"),
            **{k: str(v) for k, v in manifest["env"].items()},
        },
        check=True,
        capture_output=True,
        text=True,
    )
    captured_argv = json.loads(cap_json.read_text(encoding="utf-8"))
    assert captured_argv[:2] == ["-m", "research.v3_ppo.arena_eval"]
    parsed_eval_args = build_eval_arg_parser().parse_args(captured_argv[2:])
    assert parsed_eval_args.mode == "round-robin"

    # Execute CLI ledger path end-to-end as job.sh does on Colab
    pkt_h2h = tmp_path / "pkt_eval_h2h"
    out_h2h = tmp_path / "out_eval_h2h"
    rc_prep = prepare_main(
        [
            "--packet",
            str(pkt_h2h),
            "--mode",
            "eval",
            "--gpu",
            "L4",
            "--eval-policies",
            str(ckpt_1),
            str(ckpt_2),
            "--num-envs",
            "2",
            "--eval-burnin-ticks",
            "2",
            "--eval-window-ticks",
            "6",
            "--eval-weights-only",
            "true",
            "--output-dir",
            str(out_h2h),
            "--allow-missing-canary-receipt",
            "--allow-unmerged-commit",
        ]
    )
    assert rc_prep == 0
    rc_eval = arena_eval_main(
        [
            "--input-dir",
            str(pkt_h2h / "inputs"),
            "--ledger",
            str(pkt_h2h / "inputs" / "verified-ledger.json"),
            "--num-envs",
            "2",
            "--burnin-ticks",
            "2",
            "--window-ticks",
            "6",
            "--bootstrap-samples",
            "50",
            "--output-dir",
            str(out_h2h),
        ]
    )
    assert rc_eval == 0
    res_obj = json.loads((out_h2h / "eval_result.json").read_text(encoding="utf-8"))
    assert res_obj["mode"] == "head-to-head"
    assert len(res_obj["policies"]) == 2
