"""Fast hermetic tests for anchor evaluation, stronger scripted opponents, and configurable eval."""

from __future__ import annotations

import ast
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from research.arena_ppo.runtime import sha256_file
from research.colab_cli import canary as colab_canary
from research.colab_cli import schema as colab_schema
from research.colab_cli import supervisor as colab_supervisor
from research.game_v3.config import NUM_ACTIONS, GameV3Config
from research.game_v3.scripted import food_seeking_actions
from research.game_v3.sim import GameV3Sim
from research.v3_ppo.anchor_eval import (
    DEFAULT_ANCHOR_BURNIN_TICKS,
    DEFAULT_ANCHOR_LEARNER_SLOTS,
    DEFAULT_ANCHOR_NUM_ENVS,
    DEFAULT_ANCHOR_SCRIPTED_OPPONENTS,
    DEFAULT_ANCHOR_WINDOW_TICKS,
    V3_ANCHOR_EVAL_SCHEMA,
    _summarize_single_anchor_candidate,
    build_anchor_seat_assignment,
    candidate_unit_id,
    evaluate_anchor,
    format_anchor_markdown,
    parse_anchor_scripted_opponents,
)
from research.v3_ppo.arena_eval import (
    load_eval_policy,
    load_eval_policy_bundle,
)
from research.v3_ppo.arena_eval import main as arena_eval_main
from research.v3_ppo.config import (
    DEFAULT_TRAIN_EVAL_OPPONENTS,
    V3PPOConfig,
    assert_resume_config_compatible,
)
from research.v3_ppo.eval import _SimWorldSlice, evaluate_policy
from research.v3_ppo.job import build_arg_parser as build_train_arg_parser
from research.v3_ppo.job import resolve_effective_train_config
from research.v3_ppo.prepare import (
    ROOT,
    _train_eval_cost_scale_vs_default,
    prepare_v3_ppo_packet,
)
from research.v3_ppo.scripted_opponents import cut_off_actions, head_evader_actions
from tests.test_v3_ppo_arena_eval import (
    _make_tiny_light_checkpoint,
    _strip_wall_times,
    _write_synthetic_light_checkpoint_fast,
)
from tests.test_v3_ppo_packet import _build_synthetic_canary_receipt
from tools.ops import dispatch as ops_dispatch


def _strip_anchor_nondeterministic_fields(cand_entry: dict[str, Any]) -> dict[str, Any]:
    """Strip wall-clock and input-order index fields to compare candidate results bit-for-bit."""
    cleaned = dict(_strip_wall_times(cand_entry))
    cleaned.pop("ms_per_tick", None)
    cleaned.pop("candidate_index", None)
    cleaned.pop("policy", None)
    return cleaned


def test_scripted_opponents_batched_slice_parity_and_beat_food_seeking() -> None:
    """`head_evader` and `cut_off` have zero host syncs, match slices, and beat `food_seeking`."""
    src_path = ROOT / "research" / "v3_ppo" / "scripted_opponents.py"
    tree = ast.parse(src_path.read_text(encoding="utf-8"), filename=str(src_path))
    forbidden_sync_attrs = {"item", "tolist", "numpy", "cpu"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            msg = f"Forbidden sync .{node.func.attr}() at line {node.lineno}"
            assert node.func.attr not in forbidden_sync_attrs, msg

    cfg = GameV3Config(arena_radius=150.0, num_snakes=12)
    sim_a = GameV3Sim(cfg, num_envs=4, device="cpu", seeds=[12345 + i for i in range(4)])
    sim_b = GameV3Sim(cfg, num_envs=4, device="cpu", seeds=[12345 + i for i in range(4)])
    for _ in range(12):
        sim_a.step(food_seeking_actions(sim_a))
        sim_b.step(food_seeking_actions(sim_b))

    for fn in (head_evader_actions, cut_off_actions):
        act_full_1 = fn(sim_a)
        act_full_2 = fn(sim_b)
        assert act_full_1.shape == (4, 12)
        assert int(act_full_1.min().item()) >= 0
        assert int(act_full_1.max().item()) < NUM_ACTIONS
        assert torch.equal(act_full_1, act_full_2)

        sliced_parts = [
            fn(_SimWorldSlice(sim_a, 0, 2)),  # type: ignore[arg-type]
            fn(_SimWorldSlice(sim_a, 2, 4)),  # type: ignore[arg-type]
        ]
        assert torch.equal(act_full_1, torch.cat(sliced_parts, dim=0))

    # Head-to-head mixed-world test vs food_seeking in a dense arena (6v6 rotated across 4 worlds).
    h2h_cfg = GameV3Config(arena_radius=160.0, num_snakes=12)
    for opp_name, opp_fn in (("head_evader", head_evader_actions), ("cut_off", cut_off_actions)):
        sim = GameV3Sim(h2h_cfg, num_envs=4, device="cpu", seeds=[20261113 + i for i in range(4)])
        mask_strong = torch.zeros((4, 12), dtype=torch.bool)
        mask_strong[0::2, :6] = True
        mask_strong[1::2, 6:] = True
        strong_kills = 0
        food_kills = 0
        strong_deaths = 0
        food_deaths = 0
        strong_len_sum = 0
        food_len_sum = 0
        for t in range(360):
            act_s = opp_fn(sim)
            act_f = food_seeking_actions(sim)
            actions = torch.where(mask_strong, act_s, act_f)
            info = sim.step(actions)
            if t >= 60:
                died = info.died
                ks = info.killer_id
                strong_deaths += int((died & mask_strong).sum().item())
                food_deaths += int((died & (~mask_strong)).sum().item())
                valid_k = died & (ks >= 0) & (ks < 12)
                if bool(valid_k.any().item()):
                    killer_is_strong = torch.gather(mask_strong, 1, ks.clamp(0, 11).to(torch.int64))
                    strong_kills += int((valid_k & (~mask_strong) & killer_is_strong).sum().item())
                    food_kills += int((valid_k & mask_strong & (~killer_is_strong)).sum().item())
                alive = sim.alive
                lengths = sim.length.to(torch.int64)
                strong_len_sum += int(lengths[alive & mask_strong].sum().item())
                food_len_sum += int(lengths[alive & (~mask_strong)].sum().item())

        assert strong_kills > food_kills, f"{opp_name} kills {strong_kills} <= {food_kills}"
        assert strong_deaths < food_deaths, f"{opp_name} deaths {strong_deaths} >= {food_deaths}"
        assert strong_len_sum > food_len_sum, f"{opp_name} mass {strong_len_sum} <= {food_len_sum}"


def test_anchor_eval_cpu_determinism_and_candidate_independence(tmp_path: Path) -> None:
    """CPU anchor eval is deterministic and candidate X's score matches alone vs with others."""
    assert DEFAULT_ANCHOR_NUM_ENVS == 24
    assert DEFAULT_ANCHOR_BURNIN_TICKS == 1500
    assert DEFAULT_ANCHOR_WINDOW_TICKS == 6000
    assert DEFAULT_ANCHOR_LEARNER_SLOTS == 8
    assert DEFAULT_ANCHOR_SCRIPTED_OPPONENTS == ("cut_off", "head_evader")
    assert parse_anchor_scripted_opponents("none") == ()

    seat_a = build_anchor_seat_assignment(4, 6, 2, 3, seed=999, device="cpu")
    seat_b = build_anchor_seat_assignment(4, 6, 2, 3, seed=999, device="cpu")
    assert torch.equal(seat_a, seat_b)
    assert torch.all((seat_a == 0).sum(dim=1) == 2)

    ckpt_y, _ = _make_tiny_light_checkpoint(
        tmp_path, name="cand_y", net_size="S", seed=101, arena_radius=75.0, num_snakes=6
    )
    ckpt_x, _ = _make_tiny_light_checkpoint(
        tmp_path, name="cand_x", net_size="S", seed=202, arena_radius=75.0, num_snakes=6
    )
    ckpt_z, _ = _make_tiny_light_checkpoint(
        tmp_path, name="cand_z", net_size="M", seed=303, arena_radius=75.0, num_snakes=6
    )
    ckpt_ref_s, _ = _make_tiny_light_checkpoint(
        tmp_path, name="ref_s", net_size="S", seed=404, arena_radius=75.0, num_snakes=6
    )
    ckpt_ref_m, _ = _make_tiny_light_checkpoint(
        tmp_path, name="ref_m", net_size="M", seed=505, arena_radius=75.0, num_snakes=6
    )

    pol_y = load_eval_policy(ckpt_y, label="cand_y", device="cpu")
    pol_x = load_eval_policy(ckpt_x, label="cand_x", device="cpu")
    pol_z = load_eval_policy(ckpt_z, label="cand_z", device="cpu")
    ref_s = load_eval_policy(ckpt_ref_s, label="ref_s", device="cpu")
    ref_m = load_eval_policy(ckpt_ref_m, label="ref_m", device="cpu")
    y_orig_sha = sha256_file(ckpt_y)
    x_orig_sha = sha256_file(ckpt_x)
    z_orig_sha = sha256_file(ckpt_z)

    # Evaluate X alone twice with the same seed (stochastic sampling) over 50+300 ticks -> identical
    solo_1 = evaluate_anchor(
        [pol_x],
        [ref_s, ref_m],
        scripted_opponents=("cut_off", "head_evader"),
        num_envs=3,
        learner_slots=2,
        burnin_ticks=50,
        window_ticks=300,
        greedy=False,
        seed=20261113,
        bootstrap_samples=200,
    )
    solo_2 = evaluate_anchor(
        [pol_x],
        [ref_s, ref_m],
        scripted_opponents=("cut_off", "head_evader"),
        num_envs=3,
        learner_slots=2,
        burnin_ticks=50,
        window_ticks=300,
        greedy=False,
        seed=20261113,
        bootstrap_samples=200,
    )
    solo_diff_seed = evaluate_anchor(
        [pol_x],
        [ref_s, ref_m],
        scripted_opponents=("cut_off", "head_evader"),
        num_envs=3,
        learner_slots=2,
        burnin_ticks=50,
        window_ticks=300,
        greedy=False,
        seed=20261114,
        bootstrap_samples=200,
    )
    assert solo_1["schema"] == V3_ANCHOR_EVAL_SCHEMA
    assert x_orig_sha in solo_1["results_by_sha256"]
    stripped_solo_1 = _strip_anchor_nondeterministic_fields(solo_1["results_by_sha256"][x_orig_sha])
    stripped_solo_2 = _strip_anchor_nondeterministic_fields(solo_2["results_by_sha256"][x_orig_sha])
    stripped_diff_seed = _strip_anchor_nondeterministic_fields(
        solo_diff_seed["results_by_sha256"][x_orig_sha]
    )
    assert stripped_solo_1 == stripped_solo_2
    assert stripped_solo_1 != stripped_diff_seed
    assert (
        solo_1["results_by_sha256"][x_orig_sha]["metrics"]["time_averaged_length"],
        solo_1["results_by_sha256"][x_orig_sha]["metrics"]["acted_agent_steps"],
    ) != (
        solo_diff_seed["results_by_sha256"][x_orig_sha]["metrics"]["time_averaged_length"],
        solo_diff_seed["results_by_sha256"][x_orig_sha]["metrics"]["acted_agent_steps"],
    )

    # Evaluate [Z, Y, X] in reverse order -> all 3 candidates differ AND X is bit-identical to solo!
    batch_res = evaluate_anchor(
        [pol_z, pol_y, pol_x],
        [ref_s, ref_m],
        scripted_opponents=("cut_off", "head_evader"),
        num_envs=3,
        learner_slots=2,
        burnin_ticks=50,
        window_ticks=300,
        greedy=False,
        seed=20261113,
        bootstrap_samples=200,
    )
    assert set(batch_res["results_by_sha256"].keys()) == {y_orig_sha, x_orig_sha, z_orig_sha}
    stripped_x = _strip_anchor_nondeterministic_fields(batch_res["results_by_sha256"][x_orig_sha])
    stripped_y = _strip_anchor_nondeterministic_fields(batch_res["results_by_sha256"][y_orig_sha])
    stripped_z = _strip_anchor_nondeterministic_fields(batch_res["results_by_sha256"][z_orig_sha])
    assert stripped_x != stripped_y
    assert stripped_x != stripped_z
    assert stripped_y != stripped_z
    assert (
        stripped_x["metrics"]["time_averaged_length"]
        != stripped_y["metrics"]["time_averaged_length"]
    )
    assert (
        stripped_x["metrics"]["time_averaged_length"]
        != stripped_z["metrics"]["time_averaged_length"]
    )
    assert (
        stripped_y["metrics"]["time_averaged_length"]
        != stripped_z["metrics"]["time_averaged_length"]
    )
    assert stripped_x == stripped_solo_1

    def _collect_all_keys(obj: Any) -> set[str]:
        found: set[str] = set()
        if isinstance(obj, dict):
            for k, v in obj.items():
                found.add(str(k))
                found |= _collect_all_keys(v)
        elif isinstance(obj, list):
            for item in obj:
                found |= _collect_all_keys(item)
        return found

    all_keys = _collect_all_keys(batch_res)
    assert not any("binomial" in k for k in all_keys)
    assert not any("final_survivor" in k or "final_survival" in k for k in all_keys)

    # Check metric conservation, friendly-fire exclusion, (K+0.5)/(D+0.5), and CI bounds
    for cand_row in batch_res["candidates"]:
        m = cand_row["metrics"]
        opp_deaths = sum(int(o["candidate_deaths_from"]) for o in cand_row["by_opponent"])
        ff_deaths = int(cand_row["non_opponent_deaths"]["candidate_friendly_fire"]["deaths"])
        wall_deaths = int(cand_row["non_opponent_deaths"]["wall"]["deaths"])
        assert opp_deaths == int(m["deaths_by_opponent"]) == int(m["kill_death_ratio_deaths"])
        assert ff_deaths == int(m["friendly_fire_deaths_and_kills"])
        assert opp_deaths + ff_deaths + wall_deaths == int(m["deaths"])

        opp_kills = sum(int(o["candidate_kills_against"]) for o in cand_row["by_opponent"])
        ff_kills = int(cand_row["non_opponent_deaths"]["candidate_friendly_fire"]["kills"])
        assert ff_kills == ff_deaths
        assert opp_kills == int(m["kills"]) == int(m["kill_death_ratio_kills"])
        assert m["kill_death_ratio"] == round((opp_kills + 0.5) / (opp_deaths + 0.5), 4)

        for opp_row in cand_row["by_opponent"]:
            ok = int(opp_row["candidate_kills_against"])
            od = int(opp_row["candidate_deaths_from"])
            assert opp_row["kill_death_ratio_kills"] == ok
            assert opp_row["kill_death_ratio_deaths"] == od
            assert opp_row["kill_death_ratio"] == round((ok + 0.5) / (od + 0.5), 4)

        lo_len, hi_len = m["time_averaged_length_ci95"]
        assert lo_len <= m["time_averaged_length"] <= hi_len
        lo_alive, hi_alive = m["alive_at_end_rate_ci95"]
        assert 0.0 <= lo_alive <= m["alive_at_end_rate"] <= hi_alive <= 1.0
        lo_alen, hi_alen = m["alive_at_end_length_ci95"]
        assert lo_alen <= m["alive_at_end_length"] <= hi_alen

    # Direct synthetic check for (K+0.5)/(D+0.5) when D > 0 (point & bootstrap) + FF exclusion
    kill_mat_syn = np.zeros((2, 2, 3), dtype=np.int64)
    # World 0: 2 kills vs opp, 1 death from opp, 3 friendly-fire, 2 wall
    kill_mat_syn[0, 1, 0] = 2
    kill_mat_syn[0, 0, 1] = 1
    kill_mat_syn[0, 0, 0] = 3
    kill_mat_syn[0, 0, 2] = 2
    # World 1: 1 kill vs opp, 1 death from opp, 1 friendly-fire, 3 wall
    kill_mat_syn[1, 1, 0] = 1
    kill_mat_syn[1, 0, 1] = 1
    kill_mat_syn[1, 0, 0] = 1
    kill_mat_syn[1, 0, 2] = 3
    syn_summary = _summarize_single_anchor_candidate(
        candidate=pol_x,
        references=[],
        scripted_opponents=("cut_off",),
        game_cfg=GameV3Config(arena_radius=75.0, num_snakes=6),
        num_envs=2,
        learner_slots=2,
        burnin_ticks=0,
        window_ticks=100,
        greedy=True,
        seed=42,
        bootstrap_samples=500,
        wall_seconds=0.1,
        seats_per_world_np=np.array([[2, 4], [2, 4]], dtype=np.int64),
        acted_np=np.array([[200, 400], [200, 400]], dtype=np.int64),
        surv_ticks_np=np.array([[194, 398], [195, 399]], dtype=np.int64),
        surv_len_np=np.array([[2200, 4400], [2100, 4300]], dtype=np.int64),
        deaths_np=np.array([[6, 2], [5, 1]], dtype=np.int64),
        kills_np=np.array([[2, 1], [1, 1]], dtype=np.int64),
        alive_at_end_np=np.array([[2, 4], [1, 4]], dtype=np.int64),
        final_mass_np=np.array([[28, 48], [15, 44]], dtype=np.int64),
        kill_mat_np=kill_mat_syn,
    )
    sm = syn_summary["metrics"]
    assert sm["kills"] == 3
    assert sm["deaths_by_opponent"] == 2
    assert sm["friendly_fire_deaths_and_kills"] == 4
    assert sm["wall_deaths"] == 5
    assert sm["deaths"] == 11
    assert sm["kill_death_ratio_kills"] == 3
    assert sm["kill_death_ratio_deaths"] == 2
    assert sm["kill_death_ratio"] == 1.4  # (3+0.5)/(2+0.5)=1.4, NOT unsmoothed 3/2=1.5
    assert sm["kill_death_ratio_ci95"] == [1.0, 1.8]  # bootstrap (4+0.5)/(2+0.5)=1.8, NOT 4/2=2.0
    assert sm["alive_at_end"] == 3
    assert sm["alive_at_end_length"] == round(43.0 / 3.0, 4)

    md = format_anchor_markdown(batch_res)
    assert "Anchor Evaluation Report" in md
    assert "Alive at End" in md
    assert "Length of Seats Alive at End [95% CI]" in md
    assert "Final Surv" not in md
    assert "binomial" not in md.lower()
    assert "`cand_x`" in md
    assert "`ref_m`" in md


def test_anchor_bundle_dedup_hash_check_and_verified_continuation(tmp_path: Path) -> None:
    """Bundle deduplicates shared policies, checks per-policy SHA256, and skips verified units."""
    ckpt_a, _ = _make_tiny_light_checkpoint(
        tmp_path, name="pol_a", net_size="S", seed=11, arena_radius=70.0, num_snakes=6
    )
    ckpt_b, _ = _make_tiny_light_checkpoint(
        tmp_path, name="pol_b", net_size="M", seed=22, arena_radius=70.0, num_snakes=6
    )
    ckpt_shared, _ = _make_tiny_light_checkpoint(
        tmp_path, name="pol_shared", net_size="S", seed=33, arena_radius=70.0, num_snakes=6
    )
    orig_sha_a = sha256_file(ckpt_a)
    orig_sha_b = sha256_file(ckpt_b)
    orig_sha_shared = sha256_file(ckpt_shared)

    # ckpt_shared appears in BOTH candidates and references -> staged once in bundle!
    pkt_1 = tmp_path / "pkt_anchor_attempt1"
    out_1 = tmp_path / "out_anchor_attempt1"
    prep_1 = prepare_v3_ppo_packet(
        pkt_1,
        commit="HEAD",
        mode="eval",
        gpu="L4",
        eval_mode="anchor",
        eval_candidates=(ckpt_a, ckpt_b, ckpt_shared),
        eval_references=(ckpt_shared,),
        anchor_scripted="cut_off,head_evader",
        num_envs=2,
        learner_slots=2,
        eval_burnin_ticks=6,
        eval_window_ticks=18,
        assume_ms_per_tick=20.0,
        allow_missing_canary_receipt=True,
        output_dir=out_1,
        allow_unmerged_commit=True,
    )
    ledger_1 = json.loads((pkt_1 / "inputs" / "verified-ledger.json").read_text(encoding="utf-8"))
    # 3 distinct checkpoints staged once in policies_bundle.eval.pt
    assert len(ledger_1["eval_policies"]) == 3
    assert ledger_1["anchor_eval"]["candidate_bundle_indices"] == [0, 1, 2]
    assert ledger_1["anchor_eval"]["reference_bundle_indices"] == [2]
    assert prep_1["eval_plan"]["staged_inputs_file_count"] == 2

    # Tamper detection on per-policy policy_sha256 inside bundle
    bundle_path = pkt_1 / "inputs" / "policies_bundle.eval.pt"
    tampered_entries = [dict(e) for e in ledger_1["eval_policies"]]
    tampered_entries[0]["policy_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="policy_sha256 mismatch"):
        load_eval_policy_bundle(
            bundle_path,
            device="cpu",
            expected_bundle_sha256=ledger_1["eval_policy_bundle"]["sha256"],
            expected_policies=tampered_entries,
        )

    # Execute attempt 1 via CLI using --input-dir and --ledger
    rc_1 = arena_eval_main(
        [
            "--input-dir",
            str(pkt_1 / "inputs"),
            "--ledger",
            str(pkt_1 / "inputs" / "verified-ledger.json"),
            "--num-envs",
            "2",
            "--learner-slots",
            "2",
            "--burnin-ticks",
            "6",
            "--window-ticks",
            "18",
            "--bootstrap-samples",
            "100",
            "--device",
            "cpu",
            "--output-dir",
            str(out_1),
        ]
    )
    assert rc_1 == 0
    res_1 = json.loads((out_1 / "eval_result.json").read_text(encoding="utf-8"))
    assert set(res_1["results_by_sha256"].keys()) == {orig_sha_a, orig_sha_b, orig_sha_shared}
    assert res_1["eval_config"]["num_candidates_completed"] == 3
    assert res_1["eval_config"]["num_candidates_reused"] == 0

    # Simulate interrupted run where 2 of 3 candidate units completed with sidecars.
    partial_prior = tmp_path / "prior_partial_out"
    (partial_prior / "candidates").mkdir(parents=True)
    kept_uids = [candidate_unit_id(orig_sha_a), candidate_unit_id(orig_sha_b)]
    for uid in kept_uids:
        for suffix in (".json", ".json.sha256", ".json.complete"):
            shutil.copy2(
                out_1 / "candidates" / f"{uid}{suffix}",
                partial_prior / "candidates" / f"{uid}{suffix}",
            )

    # Refuses continuation if anchor settings (e.g. window_ticks or scripted opponents) changed!
    with pytest.raises(ValueError, match="reuse-key mismatch"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_anchor_bad_window",
            commit="HEAD",
            mode="eval",
            gpu="L4",
            eval_mode="anchor",
            eval_candidates=(ckpt_a, ckpt_b, ckpt_shared),
            eval_references=(ckpt_shared,),
            anchor_scripted="cut_off,head_evader",
            num_envs=2,
            learner_slots=2,
            eval_burnin_ticks=6,
            eval_window_ticks=24,
            prior_output_dir=partial_prior,
            assume_ms_per_tick=20.0,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_anchor_bad_window",
            allow_unmerged_commit=True,
        )

    # Valid continuation stages the 2 verified candidate units, skips them, and matches attempt 1!
    pkt_2 = tmp_path / "pkt_anchor_attempt2"
    out_2 = tmp_path / "out_anchor_attempt2"
    prep_2 = prepare_v3_ppo_packet(
        pkt_2,
        commit="HEAD",
        mode="eval",
        gpu="L4",
        eval_mode="anchor",
        eval_candidates=(ckpt_a, ckpt_b, ckpt_shared),
        eval_references=(ckpt_shared,),
        anchor_scripted="cut_off,head_evader",
        num_envs=2,
        learner_slots=2,
        eval_burnin_ticks=6,
        eval_window_ticks=18,
        prior_output_dir=partial_prior,
        assume_ms_per_tick=20.0,
        allow_missing_canary_receipt=True,
        attempt="attempt-02",
        output_dir=out_2,
        allow_unmerged_commit=True,
    )
    assert prep_2["eval_plan"]["reused_candidates"] == 2
    assert prep_2["eval_plan"]["remaining_candidates"] == 1

    rc_2 = arena_eval_main(
        [
            "--input-dir",
            str(pkt_2 / "inputs"),
            "--ledger",
            str(pkt_2 / "inputs" / "verified-ledger.json"),
            "--num-envs",
            "2",
            "--learner-slots",
            "2",
            "--burnin-ticks",
            "6",
            "--window-ticks",
            "18",
            "--bootstrap-samples",
            "100",
            "--device",
            "cpu",
            "--output-dir",
            str(out_2),
        ]
    )
    assert rc_2 == 0
    res_2 = json.loads((out_2 / "eval_result.json").read_text(encoding="utf-8"))
    assert res_2["eval_config"]["num_candidates_reused"] == 2
    assert res_2["eval_config"]["num_candidates_computed"] == 1
    for sha_k in (orig_sha_a, orig_sha_b, orig_sha_shared):
        assert _strip_anchor_nondeterministic_fields(
            res_2["results_by_sha256"][sha_k]
        ) == _strip_anchor_nondeterministic_fields(res_1["results_by_sha256"][sha_k])


def test_prepare_anchor_capacity_print_60min_refusal_and_canary_binding(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`prepare` prints all-S/all-M capacity, refuses >60m packets, and binds canary."""
    base_s = _write_synthetic_light_checkpoint_fast(
        tmp_path / "base_s" / "light_u0000100.pt", net_size="S", seed=4100
    )
    base_m = _write_synthetic_light_checkpoint_fast(
        tmp_path / "base_m" / "light_u0000100.pt", net_size="M", seed=4200
    )
    cands_s: list[Path] = []
    for idx in range(9):
        p = tmp_path / f"run_cand_s_{idx:02d}" / "checkpoints" / "light_u0000100.pt"
        p.parent.mkdir(parents=True)
        shutil.copyfile(base_s, p)
        # Ensure distinct SHA-256 per candidate by re-saving with distinct step counter if needed;
        # note identical copies deduplicate unless distinct files, so mutate 1 metadata int:
        payload = torch.load(p, map_location="cpu", weights_only=False)
        payload["updates_completed"] = 100 + idx
        payload["counters"]["updates_completed"] = 100 + idx
        torch.save(payload, p)
        cands_s.append(p)

    refs_m: list[Path] = []
    for idx in range(2):
        p = tmp_path / f"run_ref_m_{idx:02d}" / "checkpoints" / "light_u0000100.pt"
        p.parent.mkdir(parents=True)
        payload = torch.load(base_m, map_location="cpu", weights_only=False)
        payload["updates_completed"] = 200 + idx
        payload["counters"]["updates_completed"] = 200 + idx
        torch.save(payload, p)
        refs_m.append(p)

    # 1. 9 S candidates + 2 M references at default 7,500 ticks exceeds 60.0 min -> REFUSED!
    capsys.readouterr()
    with pytest.raises(ValueError, match=r"at most 7 candidates fit 60\.0 min"):
        prepare_v3_ppo_packet(
            tmp_path / "pkt_anchor_9s_refused",
            commit="HEAD",
            mode="eval",
            gpu="L4",
            eval_mode="anchor",
            eval_candidates=cands_s,
            eval_references=refs_m,
            allow_missing_canary_receipt=True,
            output_dir=tmp_path / "out_anchor_9s_refused",
            allow_unmerged_commit=True,
        )
    err_refused = capsys.readouterr().err
    assert "V3_PPO_ANCHOR_CAPACITY" in err_refused
    assert "all_small_at_defaults=" in err_refused
    assert "all_medium_at_defaults=" in err_refused

    # 1b. 6 S candidates + 2 M references at defaults SUCCEEDS and reports >= 6 all-S, >= 2 all-M!
    ok_6s = prepare_v3_ppo_packet(
        tmp_path / "pkt_anchor_6s_ok",
        commit="HEAD",
        mode="eval",
        gpu="L4",
        eval_mode="anchor",
        eval_candidates=cands_s[:6],
        eval_references=refs_m,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_anchor_6s_ok",
        allow_unmerged_commit=True,
    )
    assert (
        ok_6s["anchor_capacity"]["all_small_at_defaults_with_2S_references"]["fit_effective"] == 9
    )
    assert (
        ok_6s["anchor_capacity"]["all_medium_at_defaults_with_2M_references"]["fit_effective"] == 3
    )

    # 2. Full canary -> receipt -> eval job binding for anchor mode:
    tiny_c1, _ = _make_tiny_light_checkpoint(
        tmp_path, name="tc1", net_size="S", seed=61, arena_radius=60.0, num_snakes=6
    )
    tiny_c2, _ = _make_tiny_light_checkpoint(
        tmp_path, name="tc2", net_size="S", seed=62, arena_radius=60.0, num_snakes=6
    )
    tiny_rm, _ = _make_tiny_light_checkpoint(
        tmp_path, name="trm", net_size="M", seed=63, arena_radius=60.0, num_snakes=6
    )

    capsys.readouterr()
    canary_pkt = prepare_v3_ppo_packet(
        tmp_path / "pkt_canary_anchor",
        commit="HEAD",
        mode="canary",
        gpu="L4",
        eval_mode="anchor",
        eval_candidates=(tiny_c1, tiny_c2),
        eval_references=(tiny_rm,),
        learner_slots=2,
        output_dir=tmp_path / "out_canary_anchor",
        allow_unmerged_commit=True,
    )
    canary_mf = json.loads(Path(canary_pkt["manifest_path"]).read_text(encoding="utf-8"))
    colab_schema.validate_manifest(canary_mf)
    ops_dispatch.require_resume_policy(canary_mf)
    colab_supervisor.validate(canary_mf)
    colab_canary.check(canary_mf, ROOT)

    rcpt_path = _build_synthetic_canary_receipt(canary_mf, "L4")
    smoke_checks = json.loads(
        (Path(canary_mf["output_dir"]) / "exports" / "canary_checks.json").read_text(
            encoding="utf-8"
        )
    )
    assert smoke_checks["eval_canary_smoke"]["smoke_mode"] == "anchor"

    capsys.readouterr()
    eval_pkt = prepare_v3_ppo_packet(
        tmp_path / "pkt_eval_anchor_bound",
        commit="HEAD",
        mode="eval",
        gpu="L4",
        eval_mode="anchor",
        eval_candidates=(tiny_c1, tiny_c2),
        eval_references=(tiny_rm,),
        learner_slots=2,
        eval_burnin_ticks=20,
        eval_window_ticks=60,
        canary_receipt=rcpt_path,
        output_dir=tmp_path / "out_eval_anchor_bound",
        allow_unmerged_commit=True,
    )
    err_ok = capsys.readouterr().err
    assert "V3_PPO_ANCHOR_CAPACITY" in err_ok
    assert "all_small_at_defaults=" in err_ok
    assert "all_medium_at_defaults=" in err_ok

    eval_mf = json.loads(Path(eval_pkt["manifest_path"]).read_text(encoding="utf-8"))
    assert eval_mf["env"]["V3_PPO_EVAL_MODE"] == "anchor"
    assert eval_mf["env"]["V3_PPO_ANCHOR_SCRIPTED"] == "cut_off,head_evader"
    colab_schema.validate_manifest(eval_mf)
    ops_dispatch.require_resume_policy(eval_mf)
    colab_supervisor.validate(eval_mf)
    colab_canary.check(eval_mf, ROOT)


def test_train_eval_config_defaults_unchanged_and_custom_prepare(tmp_path: Path) -> None:
    """Default V3PPOConfig omits `eval_opponents`, legacy loads, and `prepare` sets harder eval."""
    default_cfg = V3PPOConfig()
    assert _train_eval_cost_scale_vs_default(default_cfg) == 1.0
    custom_base = V3PPOConfig(eval_envs=96, eval_burnin_ticks=100, eval_ticks=900)
    assert _train_eval_cost_scale_vs_default(custom_base, default_cfg=custom_base) == 1.0
    doubled_base = V3PPOConfig(eval_envs=192, eval_burnin_ticks=200, eval_ticks=1800)
    assert _train_eval_cost_scale_vs_default(
        doubled_base, default_cfg=custom_base
    ) == pytest.approx(4.0)
    assert (
        default_cfg.eval_opponents
        == DEFAULT_TRAIN_EVAL_OPPONENTS
        == (
            "food_seeking",
            "random_safe",
        )
    )
    default_dict = default_cfg.to_dict()
    assert "eval_opponents" not in default_dict

    # Legacy config dict without eval_opponents loads cleanly
    loaded_legacy = V3PPOConfig.from_dict(default_dict)
    assert loaded_legacy.eval_opponents == ("food_seeking", "random_safe")

    # Custom eval_opponents round-trips and is allowed across checkpoint resume
    hard_cfg = V3PPOConfig(
        game_config=GameV3Config(arena_radius=70.0, num_snakes=6),
        eval_envs=2,
        eval_burnin_ticks=4,
        eval_ticks=10,
        eval_learner_slots=2,
        eval_opponents=("cut_off", "head_evader"),
    )
    hard_dict = hard_cfg.to_dict()
    assert hard_dict["eval_opponents"] == ["cut_off", "head_evader"]
    assert V3PPOConfig.from_dict(hard_dict).eval_opponents == ("cut_off", "head_evader")
    assert_resume_config_compatible(
        V3PPOConfig(game_config=GameV3Config(arena_radius=70.0, num_snakes=6)),
        hard_cfg,
    )

    # CLI override in job.py + evaluate_policy execution
    parsed = build_train_arg_parser().parse_args(
        [
            "--output-dir",
            str(tmp_path / "dummy_job_out"),
            "--eval-opponents",
            "cut_off,head_evader",
            "--eval-burnin-ticks",
            "5",
            "--eval-ticks",
            "12",
        ]
    )
    eff_cfg = resolve_effective_train_config(parsed)
    assert eff_cfg.cfg.eval_opponents == ("cut_off", "head_evader")
    assert eff_cfg.cfg.eval_burnin_ticks == 5
    assert eff_cfg.cfg.eval_ticks == 12

    ckpt_t, trainer = _make_tiny_light_checkpoint(
        tmp_path, name="eval_hard_t", net_size="S", seed=77, arena_radius=70.0, num_snakes=6
    )
    assert ckpt_t.is_file()
    ev_summary = evaluate_policy(trainer.model, hard_cfg, device="cpu")
    assert set(ev_summary["matchups"].keys()) == {"vs_cut_off", "vs_head_evader"}

    # `prepare --mode train` with harder opponents & custom cadence
    pkt_train = prepare_v3_ppo_packet(
        tmp_path / "pkt_train_hard_eval",
        commit="HEAD",
        mode="train",
        gpu="A100",
        hours=8.0,
        net_size="S",
        num_envs=256,
        eval_opponents="cut_off,head_evader",
        eval_envs=256,
        eval_burnin_ticks=500,
        eval_ticks=3000,
        eval_every_minutes=120.0,
        allow_missing_canary_receipt=True,
        output_dir=tmp_path / "out_train_hard_eval",
        allow_unmerged_commit=True,
    )
    train_mf = json.loads(Path(pkt_train["manifest_path"]).read_text(encoding="utf-8"))
    staged_cfg = json.loads(train_mf["env"]["V3_PPO_CONFIG_JSON"])
    assert staged_cfg["eval_opponents"] == ["cut_off", "head_evader"]
    assert staged_cfg["eval_envs"] == 256
    assert staged_cfg["eval_burnin_ticks"] == 500
    assert staged_cfg["eval_ticks"] == 3000
    assert staged_cfg["eval_every_minutes"] == 120.0
    colab_schema.validate_manifest(train_mf)
    ops_dispatch.require_resume_policy(train_mf)
    eff_from_env = resolve_effective_train_config(
        build_train_arg_parser().parse_args(["--output-dir", str(tmp_path / "dummy_env_out")]),
        environ=train_mf["env"],
    )
    assert eff_from_env.cfg.eval_opponents == ("cut_off", "head_evader")
    assert eff_from_env.cfg.eval_envs == 256
    assert eff_from_env.cfg.eval_burnin_ticks == 500
    assert eff_from_env.cfg.eval_ticks == 3000
    assert eff_from_env.cfg.eval_every_minutes == 120.0
