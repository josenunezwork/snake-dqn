"""Absolute linear-cost anchor evaluation against a fixed reference pool for Game V3.1.

Evaluates each candidate checkpoint independently as ``M`` learner seats of ``N`` (default ``8``
of ``24``) in worlds whose remaining ``N - M`` seats are driven by a fixed, seeded reference pool
combining frozen reference policies (any mix of ``S`` and ``M`` networks) and batched scripted
opponents. Because candidates never play each other and every candidate faces identical world
seeds, spawn-slot assignments, and reference-policy RNG streams, cost scales linearly ``O(C)`` in
the number of candidates and a candidate's score is completely independent of which other
candidates are evaluated in the same packet.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.arena_ppo.runtime import (
    configure_determinism,
    phase_stamp,
    synchronize_device,
)
from research.game_v3.config import GameV3Config
from research.game_v3.obs import (
    LearnerIndex,
    ObservationBatch,
    allocate_obs_batch,
    build_obs,
)
from research.game_v3.rng import rng_u63_np
from research.game_v3.sim import GameV3Sim
from research.v3_ppo.arena_eval import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_EVAL_SEED,
    LoadedEvalPolicy,
    _make_action_generator,
    act_policy_chunked,
    resolve_code_commit,
    validate_compatible_policies,
)
from research.v3_ppo.buffer import assert_obs_batch_matches_layout
from research.v3_ppo.config import (
    V3_PPO_NAMESPACE,
    OpponentPolicy,
    obs_layout_from_game_config,
    parse_opponent_policies,
)
from research.v3_ppo.eval import (
    derive_world_seeds,
    preserve_rng_state,
    scripted_opponent_actions,
)
from research.v3_ppo.exports import (
    artifact_or_sidecar_exists,
    write_create_only_json,
)
from research.v3_ppo.trainer import StopRequestedError, seed_everything

V3_ANCHOR_EVAL_SCHEMA: str = "v3-ppo-anchor-eval/v1"
V3_ANCHOR_EVAL_CANDIDATE_SCHEMA: str = "v3-ppo-anchor-eval-candidate/v1"

DEFAULT_ANCHOR_NUM_ENVS: int = 24
DEFAULT_ANCHOR_BURNIN_TICKS: int = 1500
DEFAULT_ANCHOR_WINDOW_TICKS: int = 6000
DEFAULT_ANCHOR_LEARNER_SLOTS: int = 8
DEFAULT_ANCHOR_SCRIPTED_OPPONENTS: tuple[OpponentPolicy, ...] = ("cut_off", "head_evader")

ANCHOR_CANDIDATE_REUSE_KEY_FIELDS: tuple[str, ...] = (
    "arena_radius",
    "num_envs",
    "num_snakes",
    "learner_slots",
    "burnin_ticks",
    "window_ticks",
    "greedy",
    "seed",
    "candidate_sha256",
    "reference_sha256s",
    "scripted_opponents",
)

_ANCHOR_WORLD_SEED_STREAM: int = 3100
_ANCHOR_SEAT_SEED_STREAM_BASE: int = 3200
_ANCHOR_CAND_GEN_SALT: int = 0x517C_C1B7_2722_0A95
_ANCHOR_BOOTSTRAP_SALT: int = 0xA5A5_5A5A_F0F0_0F0F


def parse_anchor_scripted_opponents(
    value: str | Sequence[str] | None,
) -> tuple[OpponentPolicy, ...]:
    """Parse scripted opponent list for anchor evaluation (``'none'`` disables scripted seats)."""
    if value is None:
        return DEFAULT_ANCHOR_SCRIPTED_OPPONENTS
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned or cleaned.lower() == "none":
            return ()
        return parse_opponent_policies(
            cleaned,
            field_name="anchor_scripted_opponents",
            allow_empty=True,
        )
    items = [str(x).strip() for x in value if str(x).strip()]
    if not items or (len(items) == 1 and items[0].lower() == "none"):
        return ()
    return parse_opponent_policies(
        items,
        field_name="anchor_scripted_opponents",
        allow_empty=True,
    )


def candidate_unit_id(source_checkpoint_sha256: str) -> str:
    """Return deterministic, order-independent unit stem ``cand_<sha16>`` for a candidate."""
    sha = str(source_checkpoint_sha256).strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError(
            f"Expected 64-hex source_checkpoint_sha256, got {source_checkpoint_sha256!r}"
        )
    return f"cand_{sha[:16]}"


def build_anchor_candidate_reuse_key(
    *,
    arena_radius: float = 600.0,
    num_envs: int,
    num_snakes: int,
    learner_slots: int,
    burnin_ticks: int,
    window_ticks: int,
    greedy: bool,
    seed: int,
    candidate_sha256: str,
    reference_sha256s: Sequence[str],
    scripted_opponents: Sequence[str],
) -> dict[str, Any]:
    """Build immutable reuse key binding a completed ``candidates/cand_*.json`` unit."""
    return {
        "arena_radius": round(float(arena_radius), 4),
        "num_envs": int(num_envs),
        "num_snakes": int(num_snakes),
        "learner_slots": int(learner_slots),
        "burnin_ticks": int(burnin_ticks),
        "window_ticks": int(window_ticks),
        "greedy": bool(greedy),
        "seed": int(seed),
        "candidate_sha256": str(candidate_sha256),
        "reference_sha256s": [str(s) for s in reference_sha256s],
        "scripted_opponents": [str(s) for s in scripted_opponents],
    }


def extract_anchor_candidate_reuse_key(unit_payload: Mapping[str, Any]) -> dict[str, Any]:
    """Extract and validate ``reuse_key`` from a completed candidate unit JSON."""
    if not isinstance(unit_payload, Mapping):
        raise ValueError(
            f"Anchor candidate unit payload must be a dict, got {type(unit_payload)!r}"
        )
    rk = unit_payload.get("reuse_key")
    if not isinstance(rk, Mapping):
        raise ValueError("Anchor candidate unit payload missing 'reuse_key' mapping")
    missing = [k for k in ANCHOR_CANDIDATE_REUSE_KEY_FIELDS if k not in rk]
    if missing:
        raise ValueError(f"Anchor candidate unit reuse_key missing required fields: {missing}")
    if not isinstance(rk["greedy"], bool):
        raise ValueError(f"Anchor candidate reuse_key.greedy must be bool, got {rk['greedy']!r}")
    if not isinstance(rk["reference_sha256s"], Sequence) or isinstance(
        rk["reference_sha256s"], (str, bytes)
    ):
        raise ValueError("Anchor candidate reuse_key.reference_sha256s must be a list")
    if not isinstance(rk["scripted_opponents"], Sequence) or isinstance(
        rk["scripted_opponents"], (str, bytes)
    ):
        raise ValueError("Anchor candidate reuse_key.scripted_opponents must be a list")
    return build_anchor_candidate_reuse_key(
        arena_radius=float(rk["arena_radius"]),
        num_envs=int(rk["num_envs"]),
        num_snakes=int(rk["num_snakes"]),
        learner_slots=int(rk["learner_slots"]),
        burnin_ticks=int(rk["burnin_ticks"]),
        window_ticks=int(rk["window_ticks"]),
        greedy=bool(rk["greedy"]),
        seed=int(rk["seed"]),
        candidate_sha256=str(rk["candidate_sha256"]),
        reference_sha256s=[str(x) for x in rk["reference_sha256s"]],
        scripted_opponents=[str(x) for x in rk["scripted_opponents"]],
    )


def validate_anchor_candidate_unit_reuse(
    unit_id: str,
    unit_payload: Mapping[str, Any],
    expected_key: Mapping[str, Any],
) -> None:
    """Raise ``ValueError`` if ``unit_payload`` disagrees with ``expected_key`` on any field."""
    if unit_payload.get("namespace") not in (None, V3_PPO_NAMESPACE):
        raise ValueError(
            f"Foreign anchor candidate unit namespace {unit_payload.get('namespace')!r} "
            f"!= {V3_PPO_NAMESPACE!r}"
        )
    actual = extract_anchor_candidate_reuse_key(unit_payload)
    mismatches = {
        k: (actual.get(k), expected_key.get(k))
        for k in ANCHOR_CANDIDATE_REUSE_KEY_FIELDS
        if actual.get(k) != expected_key.get(k)
    }
    if mismatches:
        detail = ", ".join(
            f"{k}={got!r} != expected {exp!r}" for k, (got, exp) in mismatches.items()
        )
        raise ValueError(f"Staged/existing candidate unit {unit_id} reuse-key mismatch: {detail}")


def build_anchor_seat_assignment(
    num_envs: int,
    num_snakes: int,
    learner_slots: int,
    num_opponents: int,
    *,
    seed: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Build fixed SplitMix64-seeded ``(E, N)`` role assignment identical for every candidate.

    Role ``0`` denotes the candidate's ``learner_slots`` seats in each world; roles
    ``1 .. num_opponents`` denote the reference policies followed by the scripted opponents.
    Remainder opponent seats when ``(num_snakes - learner_slots) % num_opponents != 0`` rotate
    across worlds so every opponent role gets balanced representation, and all ``N`` seats in each
    world ``e`` are permuted across spawn slots using ``rng_u63_np(seed, 3200 + e, 77, slots)``.
    """
    e_cnt = int(num_envs)
    n_cnt = int(num_snakes)
    m_cand = int(learner_slots)
    k_opp = int(num_opponents)
    if e_cnt < 1:
        raise ValueError(f"num_envs must be >= 1, got {e_cnt}")
    if n_cnt < 2:
        raise ValueError(f"num_snakes must be >= 2, got {n_cnt}")
    if not (1 <= m_cand <= n_cnt):
        raise ValueError(f"learner_slots must be in 1..{n_cnt}, got {m_cand}")
    opp_seats = n_cnt - m_cand
    if opp_seats > 0 and k_opp < 1:
        raise ValueError(
            f"Anchor evaluation with learner_slots={m_cand} < num_snakes={n_cnt} requires at "
            "least 1 reference policy or scripted opponent."
        )
    if opp_seats == 0 and k_opp > 0:
        raise ValueError(
            f"learner_slots={m_cand} occupies all num_snakes={n_cnt} seats; opponent pool cannot "
            f"be assigned ({k_opp} opponents supplied)."
        )

    role_grid = np.zeros((e_cnt, n_cnt), dtype=np.int64)
    slots = np.arange(n_cnt, dtype=np.int64)
    for e_idx in range(e_cnt):
        if opp_seats == 0:
            row = np.zeros((n_cnt,), dtype=np.int64)
        else:
            base_cnt = opp_seats // k_opp
            rem_cnt = opp_seats % k_opp
            roles_list: list[int] = [0] * m_cand
            for opp_idx in range(k_opp):
                extra = 1 if ((opp_idx - e_idx) % k_opp) < rem_cnt else 0
                roles_list.extend([opp_idx + 1] * (base_cnt + extra))
            row = np.asarray(roles_list, dtype=np.int64)
        keys = rng_u63_np(int(seed), _ANCHOR_SEAT_SEED_STREAM_BASE + e_idx, 77, slots)
        order = np.argsort(keys, kind="mergesort")
        role_grid[e_idx] = row[order]

    return torch.as_tensor(role_grid, dtype=torch.int64, device=device)


def wilson_binomial_ci95(
    successes: int,
    trials: int,
    *,
    scale: float = 1.0,
    decimals: int = 6,
) -> list[float]:
    """Compute exact 95% Wilson score binomial confidence interval ``[low, high] * scale``."""
    n = int(trials)
    k = int(successes)
    if n <= 0:
        return [0.0, 0.0]
    k = max(0, min(n, k))
    z = 1.959963984540054
    z2 = z * z
    p = float(k) / float(n)
    denom = 1.0 + z2 / float(n)
    center = (p + z2 / (2.0 * float(n))) / denom
    half = (z / denom) * math.sqrt((p * (1.0 - p) / float(n)) + (z2 / (4.0 * float(n) * float(n))))
    lo = max(0.0, center - half) * float(scale)
    hi = min(1.0, center + half) * float(scale)
    return [round(lo, decimals), round(hi, decimals)]


def _bootstrap_world_ratio_ci(
    num_w: np.ndarray,
    den_w: np.ndarray,
    *,
    bs_idx: np.ndarray | None,
    scale: float = 1.0,
    zero_den_default: float = 0.0,
    bs_smooth_num: float = 0.0,
    bs_smooth_den: float = 0.0,
    decimals: int = 6,
) -> tuple[float, list[float]]:
    """Compute pooled ratio ``scale * sum(num_w) / sum(den_w)`` and 95% world-bootstrap CI."""
    tot_num = float(num_w.sum())
    tot_den = float(den_w.sum())
    if tot_den > 0.0:
        pt = float(scale) * (tot_num / tot_den)
    elif bs_smooth_den > 0.0:
        pt = float(scale) * ((tot_num + bs_smooth_num) / bs_smooth_den)
    else:
        pt = float(zero_den_default)

    pt_r = round(pt, decimals)
    if bs_idx is None or int(num_w.shape[0]) <= 1:
        return pt_r, [pt_r, pt_r]

    bs_n = num_w[bs_idx].sum(axis=1).astype(np.float64)
    bs_d = den_w[bs_idx].sum(axis=1).astype(np.float64)
    if bs_smooth_den > 0.0:
        bs_val = float(scale) * np.where(
            bs_d > 0.0,
            bs_n / np.maximum(bs_d, 1e-12),
            (bs_n + bs_smooth_num) / (bs_d + bs_smooth_den),
        )
    else:
        bs_val = np.where(
            bs_d > 0.0,
            float(scale) * (bs_n / np.maximum(bs_d, 1e-12)),
            float(zero_den_default),
        )
    lo, hi = np.percentile(bs_val, [2.5, 97.5])
    return pt_r, [round(float(lo), decimals), round(float(hi), decimals)]


def _build_opponent_descriptors(
    references: Sequence[LoadedEvalPolicy],
    scripted_opponents: Sequence[OpponentPolicy],
) -> list[dict[str, Any]]:
    """Build ordered metadata dicts for roles ``1 .. len(references) + len(scripted_opponents)``."""
    out: list[dict[str, Any]] = []
    for r_idx, ref_pol in enumerate(references):
        out.append(
            {
                "role_id": r_idx + 1,
                "opponent_key": f"ref:{ref_pol.label}",
                "kind": "reference_policy",
                "label": ref_pol.label,
                "net_size": ref_pol.net_size,
                "checkpoint_sha256": ref_pol.source_checkpoint_sha256,
                "checkpoint_path": ref_pol.source_checkpoint_path,
                "updates_completed": int(ref_pol.updates_completed),
                "agent_steps_collected": int(ref_pol.agent_steps_collected),
            }
        )
    base_role = len(references) + 1
    for s_idx, opp_name in enumerate(scripted_opponents):
        out.append(
            {
                "role_id": base_role + s_idx,
                "opponent_key": f"scripted:{opp_name}",
                "kind": "scripted",
                "label": str(opp_name),
                "policy": str(opp_name),
                "net_size": None,
                "checkpoint_sha256": None,
                "checkpoint_path": None,
            }
        )
    return out


@torch.no_grad()
def evaluate_single_anchor_candidate(
    candidate: LoadedEvalPolicy,
    references: Sequence[LoadedEvalPolicy],
    *,
    scripted_opponents: Sequence[OpponentPolicy] = DEFAULT_ANCHOR_SCRIPTED_OPPONENTS,
    num_envs: int = DEFAULT_ANCHOR_NUM_ENVS,
    learner_slots: int = DEFAULT_ANCHOR_LEARNER_SLOTS,
    burnin_ticks: int = DEFAULT_ANCHOR_BURNIN_TICKS,
    window_ticks: int = DEFAULT_ANCHOR_WINDOW_TICKS,
    greedy: bool = False,
    seed: int = DEFAULT_EVAL_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    deterministic: bool = True,
    arena_radius: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate one candidate policy against the fixed reference + scripted pool.

    Resets all RNG state and world/generator seeds strictly from ``seed`` inside
    :func:`preserve_rng_state` so the result for ``candidate`` is bit-identical on CPU regardless
    of whether ``candidate`` is evaluated alone or alongside any other candidates.
    """
    e_cnt = int(num_envs)
    m_cand = int(learner_slots)
    burnin = int(burnin_ticks)
    window = int(window_ticks)
    if e_cnt < 1:
        raise ValueError(f"num_envs must be >= 1, got {e_cnt}")
    if burnin < 0:
        raise ValueError(f"burnin_ticks must be >= 0, got {burnin}")
    if window < 1:
        raise ValueError(f"window_ticks must be >= 1, got {window}")

    scripted_tuple = parse_anchor_scripted_opponents(scripted_opponents)
    all_neural: list[LoadedEvalPolicy] = [candidate, *references]
    game_cfg: GameV3Config = validate_compatible_policies(all_neural)
    if arena_radius is not None:
        if float(arena_radius) <= 0.0:
            raise ValueError(f"arena_radius must be > 0, got {arena_radius}")
        game_cfg = replace(game_cfg, arena_radius=float(arena_radius))
    n_cnt = int(game_cfg.num_snakes)
    num_refs = len(references)
    num_scripted = len(scripted_tuple)
    num_opponents = num_refs + num_scripted
    num_roles = 1 + num_opponents

    dev = next(candidate.model.parameters()).device
    for pol in all_neural:
        pol.model.eval()

    total_ticks = burnin + window
    t_wall0 = time.perf_counter()

    with preserve_rng_state():
        if dev.type == "cpu" and deterministic:
            configure_determinism(True)
        seed_everything(int(seed))

        world_seeds = derive_world_seeds(int(seed), e_cnt, stream_id=_ANCHOR_WORLD_SEED_STREAM)
        sim = GameV3Sim(game_cfg, num_envs=e_cnt, device=dev, seeds=world_seeds)

        seat_role = build_anchor_seat_assignment(
            e_cnt,
            n_cnt,
            m_cand,
            num_opponents,
            seed=int(seed),
            device=dev,
        )
        seats_per_world_np = (
            torch.stack([(seat_role == r_idx).sum(dim=1) for r_idx in range(num_roles)], dim=1)
            .cpu()
            .numpy()
        )

        # Single fused observation batch over all neural seats (candidate + references).
        neural_mask = seat_role <= num_refs
        neural_idx = LearnerIndex(sim, neural_mask)
        neural_obs: ObservationBatch = allocate_obs_batch(sim, neural_idx)
        assert_obs_batch_matches_layout(
            neural_obs,
            candidate.obs_layout,
            batch_rows=int(neural_idx.count),
            context="anchor_eval.neural_obs",
        )

        seat_role_flat = seat_role.reshape(-1)
        neural_roles = seat_role_flat[neural_mask.reshape(-1)]
        sub_idx_by_neural: list[torch.Tensor] = [
            torch.nonzero(neural_roles == r_idx, as_tuple=False).squeeze(1)
            for r_idx in range(num_refs + 1)
        ]
        flat_idx_by_neural: list[torch.Tensor] = [
            torch.nonzero(seat_role_flat == r_idx, as_tuple=False).squeeze(1)
            for r_idx in range(num_refs + 1)
        ]

        scripted_role_masks: dict[OpponentPolicy, torch.Tensor] = {}
        for s_idx, opp_kind in enumerate(scripted_tuple):
            role_id = num_refs + 1 + s_idx
            role_mask = seat_role == role_id
            if opp_kind in scripted_role_masks:
                scripted_role_masks[opp_kind] = scripted_role_masks[opp_kind] | role_mask
            else:
                scripted_role_masks[opp_kind] = role_mask

        gens: list[torch.Generator | None] = [
            _make_action_generator(dev, int(seed) ^ _ANCHOR_CAND_GEN_SALT),
            *[
                _make_action_generator(dev, int(seed) ^ (0x9E37_79B9 * (r_idx + 101)))
                for r_idx in range(num_refs)
            ],
        ]

        # Zero-host-sync per-world accumulators over the post-burn-in window.
        acted_steps_wr = torch.zeros((e_cnt, num_roles), dtype=torch.int64, device=dev)
        survived_ticks_wr = torch.zeros((e_cnt, num_roles), dtype=torch.int64, device=dev)
        survived_len_sum_wr = torch.zeros((e_cnt, num_roles), dtype=torch.int64, device=dev)
        deaths_wr = torch.zeros((e_cnt, num_roles), dtype=torch.int64, device=dev)
        kills_wr = torch.zeros((e_cnt, num_roles), dtype=torch.int64, device=dev)

        kill_cols = num_roles + 1  # killer_role 0..R-1, plus R = wall
        kill_mat_1d = torch.zeros((e_cnt * num_roles * kill_cols,), dtype=torch.int64, device=dev)
        world_ids_en = torch.arange(e_cnt, dtype=torch.int64, device=dev)[:, None].expand(
            e_cnt, n_cnt
        )
        zeros_en_i64 = torch.zeros((e_cnt, n_cnt), dtype=torch.int64, device=dev)

        world_actions_flat = torch.full((e_cnt * n_cnt,), 2, dtype=torch.long, device=dev)
        last_progress_mono = time.perf_counter()

        for tick in range(total_ticks):
            if should_stop is not None and (tick % 16 == 0) and should_stop():
                raise StopRequestedError(
                    f"Stop requested during anchor evaluation of {candidate.label!r} at tick {tick}"
                )
            now_mono = time.perf_counter()
            if (now_mono - last_progress_mono) >= 90.0:
                phase_stamp(
                    "anchor_eval",
                    "PROGRESS",
                    candidate=candidate.label,
                    tick=tick,
                    total_ticks=total_ticks,
                    elapsed_sec=round(now_mono - t_wall0, 2),
                )
                last_progress_mono = now_mono

            acted_es = sim.alive & (~sim.truncated_pending)
            world_actions_flat.fill_(2)

            build_obs(sim, learner_index=neural_idx, out=neural_obs)
            for r_idx, pol in enumerate(all_neural):
                sub_idx = sub_idx_by_neural[r_idx]
                if int(sub_idx.numel()) == 0:
                    continue
                acts_r = act_policy_chunked(
                    pol,
                    neural_obs.raster_s1.index_select(0, sub_idx),
                    neural_obs.raster_s4.index_select(0, sub_idx),
                    neural_obs.raster_s16.index_select(0, sub_idx),
                    neural_obs.scalars.index_select(0, sub_idx),
                    neural_obs.action_mask.index_select(0, sub_idx),
                    greedy=greedy,
                    generator=gens[r_idx],
                )
                world_actions_flat.index_copy_(0, flat_idx_by_neural[r_idx], acts_r)

            world_actions = world_actions_flat.view(e_cnt, n_cnt)
            for opp_kind, role_mask in scripted_role_masks.items():
                world_actions = torch.where(
                    role_mask,
                    scripted_opponent_actions(sim, opp_kind),
                    world_actions,
                )

            info = sim.step(world_actions, truncate=False)

            if tick >= burnin:
                died_es = acted_es & info.died
                survived_es = acted_es & (~info.died)
                acted_steps_wr.scatter_add_(1, seat_role, acted_es.to(torch.int64))
                survived_ticks_wr.scatter_add_(1, seat_role, survived_es.to(torch.int64))
                survived_len_sum_wr.scatter_add_(
                    1,
                    seat_role,
                    torch.where(survived_es, info.length.to(torch.int64), zeros_en_i64),
                )
                deaths_wr.scatter_add_(1, seat_role, died_es.to(torch.int64))
                kills_wr.scatter_add_(
                    1,
                    seat_role,
                    torch.where(acted_es, info.kill_credit.to(torch.int64), zeros_en_i64),
                )

                ks = info.killer_id
                killer_role = seat_role.gather(1, ks.clamp(0, n_cnt - 1).to(torch.int64))
                killer_col = torch.where(
                    ks >= 0,
                    killer_role,
                    torch.full_like(seat_role, num_roles),
                )
                flat_km_key = (
                    world_ids_en * (num_roles * kill_cols) + seat_role * kill_cols + killer_col
                ).reshape(-1)
                kill_mat_1d.index_put_(
                    (flat_km_key,),
                    died_es.reshape(-1).to(torch.int64),
                    accumulate=True,
                )

        end_alive = sim.alive & (~sim.truncated_pending)
        final_survivors_wr = torch.zeros((e_cnt, num_roles), dtype=torch.int64, device=dev)
        final_mass_wr = torch.zeros((e_cnt, num_roles), dtype=torch.int64, device=dev)
        final_survivors_wr.scatter_add_(1, seat_role, end_alive.to(torch.int64))
        final_mass_wr.scatter_add_(
            1,
            seat_role,
            torch.where(end_alive, sim.length.to(torch.int64), zeros_en_i64),
        )

        synchronize_device(dev)
        wall_seconds = time.perf_counter() - t_wall0

    acted_np = acted_steps_wr.cpu().numpy()
    surv_ticks_np = survived_ticks_wr.cpu().numpy()
    surv_len_np = survived_len_sum_wr.cpu().numpy()
    deaths_np = deaths_wr.cpu().numpy()
    kills_np = kills_wr.cpu().numpy()
    final_surv_np = final_survivors_wr.cpu().numpy()
    final_mass_np = final_mass_wr.cpu().numpy()
    kill_mat_np = kill_mat_1d.view(e_cnt, num_roles, kill_cols).cpu().numpy()

    return _summarize_single_anchor_candidate(
        candidate=candidate,
        references=references,
        scripted_opponents=scripted_tuple,
        game_cfg=game_cfg,
        num_envs=e_cnt,
        learner_slots=m_cand,
        burnin_ticks=burnin,
        window_ticks=window,
        greedy=bool(greedy),
        seed=int(seed),
        bootstrap_samples=int(bootstrap_samples),
        wall_seconds=wall_seconds,
        seats_per_world_np=seats_per_world_np,
        acted_np=acted_np,
        surv_ticks_np=surv_ticks_np,
        surv_len_np=surv_len_np,
        deaths_np=deaths_np,
        kills_np=kills_np,
        final_surv_np=final_surv_np,
        final_mass_np=final_mass_np,
        kill_mat_np=kill_mat_np,
    )


def _summarize_single_anchor_candidate(
    *,
    candidate: LoadedEvalPolicy,
    references: Sequence[LoadedEvalPolicy],
    scripted_opponents: Sequence[OpponentPolicy],
    game_cfg: GameV3Config,
    num_envs: int,
    learner_slots: int,
    burnin_ticks: int,
    window_ticks: int,
    greedy: bool,
    seed: int,
    bootstrap_samples: int,
    wall_seconds: float,
    seats_per_world_np: np.ndarray,
    acted_np: np.ndarray,
    surv_ticks_np: np.ndarray,
    surv_len_np: np.ndarray,
    deaths_np: np.ndarray,
    kills_np: np.ndarray,
    final_surv_np: np.ndarray,
    final_mass_np: np.ndarray,
    kill_mat_np: np.ndarray,
) -> dict[str, Any]:
    """Compute world-bootstrap and Wilson binomial CIs plus per-opponent kill/death attribution."""
    e_cnt = int(num_envs)
    n_bs = max(0, int(bootstrap_samples))
    if e_cnt > 1 and n_bs > 0:
        rng = np.random.Generator(
            np.random.PCG64((int(seed) ^ _ANCHOR_BOOTSTRAP_SALT) & 0xFFFF_FFFF_FFFF_FFFF)
        )
        bs_idx: np.ndarray | None = rng.integers(0, e_cnt, size=(n_bs, e_cnt))
    else:
        bs_idx = None

    c_acted_w = acted_np[:, 0]
    c_surv_ticks_w = surv_ticks_np[:, 0]
    c_surv_len_w = surv_len_np[:, 0]
    c_deaths_w = deaths_np[:, 0]
    c_kills_w = kills_np[:, 0]
    c_final_surv_w = final_surv_np[:, 0]
    c_final_mass_w = final_mass_np[:, 0]
    c_seats_w = seats_per_world_np[:, 0]

    final_arena_mass_w = final_mass_np.sum(axis=1)
    window_arena_mass_w = surv_len_np.sum(axis=1)

    tot_acted = int(c_acted_w.sum())
    tot_surv_ticks = int(c_surv_ticks_w.sum())
    tot_deaths = int(c_deaths_w.sum())
    tot_kills = int(c_kills_w.sum())
    tot_final_surv = int(c_final_surv_w.sum())
    tot_cand_seats = int(c_seats_w.sum())

    time_avg_len, time_avg_len_ci = _bootstrap_world_ratio_ci(
        c_surv_len_w,
        c_surv_ticks_w,
        bs_idx=bs_idx,
        zero_den_default=float(game_cfg.start_length),
        decimals=4,
    )
    final_surv_len, final_surv_len_ci = _bootstrap_world_ratio_ci(
        c_final_mass_w,
        c_final_surv_w,
        bs_idx=bs_idx,
        zero_den_default=0.0,
        decimals=4,
    )
    final_surv_rate, final_surv_rate_ci = _bootstrap_world_ratio_ci(
        c_final_surv_w,
        c_seats_w,
        bs_idx=bs_idx,
        decimals=6,
    )
    final_surv_rate_binom_ci = wilson_binomial_ci95(tot_final_surv, tot_cand_seats, decimals=6)

    deaths_per_1k, deaths_per_1k_ci = _bootstrap_world_ratio_ci(
        c_deaths_w,
        c_acted_w,
        bs_idx=bs_idx,
        scale=1000.0,
        decimals=4,
    )
    deaths_per_1k_binom_ci = wilson_binomial_ci95(tot_deaths, tot_acted, scale=1000.0, decimals=4)

    kills_per_1k, kills_per_1k_ci = _bootstrap_world_ratio_ci(
        c_kills_w,
        c_acted_w,
        bs_idx=bs_idx,
        scale=1000.0,
        decimals=4,
    )
    kills_per_1k_binom_ci = wilson_binomial_ci95(tot_kills, tot_acted, scale=1000.0, decimals=4)

    kd_ratio, kd_ratio_ci = _bootstrap_world_ratio_ci(
        c_kills_w,
        c_deaths_w,
        bs_idx=bs_idx,
        bs_smooth_num=0.5,
        bs_smooth_den=0.5,
        decimals=4,
    )

    final_mass_share, final_mass_share_ci = _bootstrap_world_ratio_ci(
        c_final_mass_w,
        final_arena_mass_w,
        bs_idx=bs_idx,
        zero_den_default=float(learner_slots) / float(game_cfg.num_snakes),
        decimals=6,
    )
    window_mass_share, window_mass_share_ci = _bootstrap_world_ratio_ci(
        c_surv_len_w,
        window_arena_mass_w,
        bs_idx=bs_idx,
        zero_den_default=float(learner_slots) / float(game_cfg.num_snakes),
        decimals=6,
    )

    opp_descs = _build_opponent_descriptors(references, scripted_opponents)
    by_opponent_list: list[dict[str, Any]] = []
    by_opponent_map: dict[str, dict[str, Any]] = {}

    for desc in opp_descs:
        r_id = int(desc["role_id"])
        kills_vs_w = kill_mat_np[:, r_id, 0]
        deaths_from_w = kill_mat_np[:, 0, r_id]
        combat_tot_w = kills_vs_w + deaths_from_w

        k_vs = int(kills_vs_w.sum())
        d_from = int(deaths_from_w.sum())

        k_vs_1k, k_vs_1k_ci = _bootstrap_world_ratio_ci(
            kills_vs_w,
            c_acted_w,
            bs_idx=bs_idx,
            scale=1000.0,
            decimals=4,
        )
        d_from_1k, d_from_1k_ci = _bootstrap_world_ratio_ci(
            deaths_from_w,
            c_acted_w,
            bs_idx=bs_idx,
            scale=1000.0,
            decimals=4,
        )
        kd_vs, kd_vs_ci = _bootstrap_world_ratio_ci(
            kills_vs_w,
            deaths_from_w,
            bs_idx=bs_idx,
            bs_smooth_num=0.5,
            bs_smooth_den=0.5,
            decimals=4,
        )
        kill_share_vs, kill_share_vs_ci = _bootstrap_world_ratio_ci(
            kills_vs_w,
            combat_tot_w,
            bs_idx=bs_idx,
            zero_den_default=0.5,
            decimals=6,
        )
        kill_share_vs_binom_ci = (
            wilson_binomial_ci95(k_vs, k_vs + d_from, decimals=6)
            if (k_vs + d_from) > 0
            else [0.5, 0.5]
        )

        opp_len, opp_len_ci = _bootstrap_world_ratio_ci(
            surv_len_np[:, r_id],
            surv_ticks_np[:, r_id],
            bs_idx=bs_idx,
            zero_den_default=float(game_cfg.start_length),
            decimals=4,
        )
        opp_final_surv_len, _ = _bootstrap_world_ratio_ci(
            final_mass_np[:, r_id],
            final_surv_np[:, r_id],
            bs_idx=bs_idx,
            zero_den_default=0.0,
            decimals=4,
        )
        opp_end_share, opp_end_share_ci = _bootstrap_world_ratio_ci(
            final_mass_np[:, r_id],
            final_arena_mass_w,
            bs_idx=bs_idx,
            zero_den_default=0.0,
            decimals=6,
        )
        opp_d_1k, _ = _bootstrap_world_ratio_ci(
            deaths_np[:, r_id],
            acted_np[:, r_id],
            bs_idx=bs_idx,
            scale=1000.0,
            decimals=4,
        )
        opp_k_1k, _ = _bootstrap_world_ratio_ci(
            kills_np[:, r_id],
            acted_np[:, r_id],
            bs_idx=bs_idx,
            scale=1000.0,
            decimals=4,
        )

        opp_entry: dict[str, Any] = {
            **desc,
            "seats_per_world": round(float(seats_per_world_np[:, r_id].mean()), 4),
            "candidate_kills_against": k_vs,
            "candidate_kills_per_1k_agent_steps": k_vs_1k,
            "candidate_kills_per_1k_ci95": k_vs_1k_ci,
            "candidate_deaths_from": d_from,
            "candidate_deaths_per_1k_agent_steps": d_from_1k,
            "candidate_deaths_per_1k_ci95": d_from_1k_ci,
            "kill_death_ratio": kd_vs,
            "kill_death_ratio_ci95": kd_vs_ci,
            "combat_kill_share": kill_share_vs,
            "combat_kill_share_ci95": kill_share_vs_ci,
            "combat_kill_share_binomial_ci95": kill_share_vs_binom_ci,
            "opponent_time_averaged_length": opp_len,
            "opponent_time_averaged_length_ci95": opp_len_ci,
            "opponent_final_survivor_length": opp_final_surv_len,
            "opponent_final_arena_mass_share": opp_end_share,
            "opponent_final_arena_mass_share_ci95": opp_end_share_ci,
            "opponent_deaths_per_1k_agent_steps": opp_d_1k,
            "opponent_kills_per_1k_agent_steps": opp_k_1k,
        }
        by_opponent_list.append(opp_entry)
        by_opponent_map[str(desc["opponent_key"])] = opp_entry

    num_roles = 1 + len(opp_descs)
    friendly_deaths_w = kill_mat_np[:, 0, 0]
    wall_deaths_w = kill_mat_np[:, 0, num_roles]
    friendly_d = int(friendly_deaths_w.sum())
    wall_d = int(wall_deaths_w.sum())

    friendly_d_1k, friendly_d_1k_ci = _bootstrap_world_ratio_ci(
        friendly_deaths_w,
        c_acted_w,
        bs_idx=bs_idx,
        scale=1000.0,
        decimals=4,
    )
    wall_d_1k, wall_d_1k_ci = _bootstrap_world_ratio_ci(
        wall_deaths_w,
        c_acted_w,
        bs_idx=bs_idx,
        scale=1000.0,
        decimals=4,
    )

    metrics = {
        "acted_agent_steps": tot_acted,
        "survived_agent_steps": tot_surv_ticks,
        "time_averaged_length": time_avg_len,
        "time_averaged_length_ci95": time_avg_len_ci,
        "final_survivor_length": final_surv_len,
        "final_survivor_length_ci95": final_surv_len_ci,
        "final_survivors": tot_final_surv,
        "candidate_seats_total": tot_cand_seats,
        "final_survival_rate": final_surv_rate,
        "final_survival_rate_ci95": final_surv_rate_ci,
        "final_survival_rate_binomial_ci95": final_surv_rate_binom_ci,
        "deaths": tot_deaths,
        "deaths_per_1k_agent_steps": deaths_per_1k,
        "deaths_per_1k_ci95": deaths_per_1k_ci,
        "deaths_per_1k_binomial_ci95": deaths_per_1k_binom_ci,
        "kills": tot_kills,
        "kills_per_1k_agent_steps": kills_per_1k,
        "kills_per_1k_ci95": kills_per_1k_ci,
        "kills_per_1k_binomial_ci95": kills_per_1k_binom_ci,
        "kill_death_ratio": kd_ratio,
        "kill_death_ratio_ci95": kd_ratio_ci,
        "final_arena_mass_share": final_mass_share,
        "final_arena_mass_share_ci95": final_mass_share_ci,
        "window_arena_mass_share": window_mass_share,
        "window_arena_mass_share_ci95": window_mass_share_ci,
        "friendly_fire_deaths_and_kills": friendly_d,
        "friendly_fire_deaths_per_1k_agent_steps": friendly_d_1k,
        "friendly_fire_deaths_per_1k_ci95": friendly_d_1k_ci,
        "wall_deaths": wall_d,
        "wall_deaths_per_1k_agent_steps": wall_d_1k,
        "wall_deaths_per_1k_ci95": wall_d_1k_ci,
    }

    total_ticks = int(burnin_ticks) + int(window_ticks)
    return {
        "unit_id": candidate_unit_id(candidate.source_checkpoint_sha256),
        "checkpoint_sha256": candidate.source_checkpoint_sha256,
        "label": candidate.label,
        "net_size": candidate.net_size,
        "updates_completed": int(candidate.updates_completed),
        "agent_steps_collected": int(candidate.agent_steps_collected),
        "policy": candidate.provenance_dict(policy_index=0),
        "wall_seconds": round(float(wall_seconds), 4),
        "ms_per_tick": round(1000.0 * float(wall_seconds) / float(max(1, total_ticks)), 4),
        "metrics": metrics,
        "by_opponent": by_opponent_list,
        "by_opponent_map": by_opponent_map,
        "non_opponent_deaths": {
            "candidate_friendly_fire": {
                "deaths": friendly_d,
                "kills": friendly_d,
                "deaths_per_1k_agent_steps": friendly_d_1k,
                "deaths_per_1k_ci95": friendly_d_1k_ci,
            },
            "wall": {
                "deaths": wall_d,
                "kills": 0,
                "deaths_per_1k_agent_steps": wall_d_1k,
                "deaths_per_1k_ci95": wall_d_1k_ci,
            },
        },
        "per_world": {
            "acted_agent_steps": [int(x) for x in c_acted_w.tolist()],
            "survived_agent_steps": [int(x) for x in c_surv_ticks_w.tolist()],
            "survived_length_sum": [int(x) for x in c_surv_len_w.tolist()],
            "deaths": [int(x) for x in c_deaths_w.tolist()],
            "kills": [int(x) for x in c_kills_w.tolist()],
            "final_survivors": [int(x) for x in c_final_surv_w.tolist()],
            "final_candidate_mass": [int(x) for x in c_final_mass_w.tolist()],
            "final_arena_mass": [int(x) for x in final_arena_mass_w.tolist()],
        },
    }


def evaluate_anchor(
    candidates: Sequence[LoadedEvalPolicy],
    references: Sequence[LoadedEvalPolicy],
    *,
    scripted_opponents: Sequence[OpponentPolicy] = DEFAULT_ANCHOR_SCRIPTED_OPPONENTS,
    num_envs: int = DEFAULT_ANCHOR_NUM_ENVS,
    learner_slots: int = DEFAULT_ANCHOR_LEARNER_SLOTS,
    burnin_ticks: int = DEFAULT_ANCHOR_BURNIN_TICKS,
    window_ticks: int = DEFAULT_ANCHOR_WINDOW_TICKS,
    greedy: bool = False,
    seed: int = DEFAULT_EVAL_SEED,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    deterministic: bool = True,
    arena_radius: float | None = None,
    output_dir: Path | str | None = None,
    verified_candidate_paths: Mapping[str, Path] | None = None,
    max_wall_seconds: float | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Evaluate all ``candidates`` linearly against the fixed reference + scripted pool.

    Results are keyed by each candidate's ORIGINAL ``source_checkpoint_sha256`` in
    ``results_by_sha256`` (and listed in input order in ``candidates``). When ``output_dir`` is
    provided, each completed candidate is atomically exported as
    ``candidates/cand_<sha16>.json`` with ``.sha256`` and ``.complete`` sidecars for verified
    ledger resume skip.
    """
    if not candidates:
        raise ValueError("Anchor evaluation requires at least 1 candidate policy")

    scripted_tuple = parse_anchor_scripted_opponents(scripted_opponents)
    dedup_candidates: list[LoadedEvalPolicy] = []
    seen_cand_shas: set[str] = set()
    for cand in candidates:
        c_sha = str(cand.source_checkpoint_sha256)
        if c_sha not in seen_cand_shas:
            seen_cand_shas.add(c_sha)
            dedup_candidates.append(cand)

    game_cfg = validate_compatible_policies([*dedup_candidates, *references])
    if arena_radius is not None:
        if float(arena_radius) <= 0.0:
            raise ValueError(f"arena_radius must be > 0, got {arena_radius}")
        game_cfg = replace(game_cfg, arena_radius=float(arena_radius))
    n_cnt = int(game_cfg.num_snakes)
    e_cnt = int(num_envs)
    m_cand = int(learner_slots)
    burnin = int(burnin_ticks)
    window = int(window_ticks)

    # Validate seat assignment up front before running any candidate.
    seat_preview = build_anchor_seat_assignment(
        e_cnt,
        n_cnt,
        m_cand,
        len(references) + len(scripted_tuple),
        seed=int(seed),
        device="cpu",
    )
    num_roles = 1 + len(references) + len(scripted_tuple)
    mean_seats_by_role = [
        round(float((seat_preview == r_idx).sum(dim=1).to(torch.float64).mean().item()), 4)
        for r_idx in range(num_roles)
    ]

    ref_shas = [str(r.source_checkpoint_sha256) for r in references]
    scripted_names = [str(s) for s in scripted_tuple]
    verified_map = dict(verified_candidate_paths or {})

    # Pre-validate any staged/verified candidate units before running compute.
    expected_reuse_by_uid: dict[str, dict[str, Any]] = {}
    for cand in dedup_candidates:
        uid = candidate_unit_id(cand.source_checkpoint_sha256)
        expected_reuse_by_uid[uid] = build_anchor_candidate_reuse_key(
            arena_radius=float(game_cfg.arena_radius),
            num_envs=e_cnt,
            num_snakes=n_cnt,
            learner_slots=m_cand,
            burnin_ticks=burnin,
            window_ticks=window,
            greedy=bool(greedy),
            seed=int(seed),
            candidate_sha256=cand.source_checkpoint_sha256,
            reference_sha256s=ref_shas,
            scripted_opponents=scripted_names,
        )
        if uid in verified_map:
            v_file = Path(verified_map[uid])
            v_payload = json.loads(v_file.read_text(encoding="utf-8"))
            validate_anchor_candidate_unit_reuse(uid, v_payload, expected_reuse_by_uid[uid])

    cand_out_dir: Path | None = None
    if output_dir is not None:
        cand_out_dir = Path(output_dir) / "candidates"
        cand_out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.perf_counter()
    cand_results: list[dict[str, Any]] = []
    results_by_sha256: dict[str, dict[str, Any]] = {}
    reused_count = 0
    computed_count = 0
    aborted = False

    for c_idx, cand in enumerate(dedup_candidates):
        uid = candidate_unit_id(cand.source_checkpoint_sha256)
        exp_rk = expected_reuse_by_uid[uid]

        if uid in verified_map:
            v_payload = json.loads(Path(verified_map[uid]).read_text(encoding="utf-8"))
            validate_anchor_candidate_unit_reuse(uid, v_payload, exp_rk)
            cand_entry = dict(v_payload["candidate_result"])
            # Update display label if explicit labels were passed while keeping sha256 exact.
            cand_entry["label"] = cand.label
            cand_results.append(cand_entry)
            results_by_sha256[cand.source_checkpoint_sha256] = cand_entry
            reused_count += 1
            continue

        elapsed = time.perf_counter() - t_start
        if (should_stop is not None and should_stop()) or (
            max_wall_seconds is not None and elapsed >= float(max_wall_seconds)
        ):
            aborted = True
            break

        def _cand_stop() -> bool:
            if should_stop is not None and should_stop():
                return True
            if max_wall_seconds is not None:
                return (time.perf_counter() - t_start) >= float(max_wall_seconds)
            return False

        try:
            cand_entry = evaluate_single_anchor_candidate(
                cand,
                references,
                scripted_opponents=scripted_tuple,
                num_envs=e_cnt,
                learner_slots=m_cand,
                burnin_ticks=burnin,
                window_ticks=window,
                greedy=greedy,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
                deterministic=deterministic,
                arena_radius=float(game_cfg.arena_radius),
                should_stop=_cand_stop,
            )
        except StopRequestedError:
            aborted = True
            break

        cand_entry["candidate_index"] = c_idx
        if cand_out_dir is not None:
            unit_file = cand_out_dir / f"{uid}.json"
            if not artifact_or_sidecar_exists(unit_file):
                write_create_only_json(
                    unit_file,
                    {
                        "schema": V3_ANCHOR_EVAL_CANDIDATE_SCHEMA,
                        "namespace": V3_PPO_NAMESPACE,
                        "unit_id": uid,
                        "seed": int(seed),
                        "reuse_key": exp_rk,
                        "candidate_sha256": cand.source_checkpoint_sha256,
                        "candidate_result": cand_entry,
                    },
                )

        cand_results.append(cand_entry)
        results_by_sha256[cand.source_checkpoint_sha256] = cand_entry
        computed_count += 1

    total_wall = time.perf_counter() - t_start
    standings = sorted(
        cand_results,
        key=lambda c: (
            float(c["metrics"]["final_arena_mass_share"]),
            float(c["metrics"]["time_averaged_length"]),
            float(c["metrics"]["kill_death_ratio"]),
        ),
        reverse=True,
    )
    ranked_standings = [
        {
            "rank": rank_idx + 1,
            "label": row["label"],
            "checkpoint_sha256": row["checkpoint_sha256"],
            "net_size": row["net_size"],
            "updates_completed": row["updates_completed"],
            "agent_steps_collected": row["agent_steps_collected"],
            "time_averaged_length": row["metrics"]["time_averaged_length"],
            "time_averaged_length_ci95": row["metrics"]["time_averaged_length_ci95"],
            "final_survivor_length": row["metrics"]["final_survivor_length"],
            "final_survivor_length_ci95": row["metrics"]["final_survivor_length_ci95"],
            "final_survivors": row["metrics"]["final_survivors"],
            "candidate_seats_total": row["metrics"]["candidate_seats_total"],
            "deaths": row["metrics"]["deaths"],
            "deaths_per_1k_agent_steps": row["metrics"]["deaths_per_1k_agent_steps"],
            "deaths_per_1k_ci95": row["metrics"]["deaths_per_1k_ci95"],
            "kills": row["metrics"]["kills"],
            "kills_per_1k_agent_steps": row["metrics"]["kills_per_1k_agent_steps"],
            "kills_per_1k_ci95": row["metrics"]["kills_per_1k_ci95"],
            "kill_death_ratio": row["metrics"]["kill_death_ratio"],
            "kill_death_ratio_ci95": row["metrics"]["kill_death_ratio_ci95"],
            "final_arena_mass_share": row["metrics"]["final_arena_mass_share"],
            "final_arena_mass_share_ci95": row["metrics"]["final_arena_mass_share_ci95"],
        }
        for rank_idx, row in enumerate(standings)
    ]

    opp_descs = _build_opponent_descriptors(references, scripted_tuple)
    for idx_o, o_desc in enumerate(opp_descs):
        o_desc["seats_per_world"] = mean_seats_by_role[idx_o + 1]

    dev = next(dedup_candidates[0].model.parameters()).device
    return {
        "schema": V3_ANCHOR_EVAL_SCHEMA,
        "namespace": V3_PPO_NAMESPACE,
        "mode": "anchor",
        "code_commit": resolve_code_commit(),
        "device": str(dev),
        "greedy": bool(greedy),
        "seed": int(seed),
        "bootstrap_samples": int(bootstrap_samples),
        "aborted": bool(aborted),
        "wall_seconds": round(float(total_wall), 4),
        "eval_config": {
            "num_candidates_requested": len(dedup_candidates),
            "num_candidates_completed": len(cand_results),
            "num_candidates_reused": reused_count,
            "num_candidates_computed": computed_count,
            "num_references": len(references),
            "scripted_opponents": scripted_names,
            "num_envs": e_cnt,
            "num_snakes": n_cnt,
            "learner_slots": m_cand,
            "candidate_seats_per_world": mean_seats_by_role[0],
            "burnin_ticks": burnin,
            "window_ticks": window,
            "total_ticks_per_candidate": burnin + window,
            "arena_radius": float(game_cfg.arena_radius),
            "start_length": int(game_cfg.start_length),
            "obs_layout": obs_layout_from_game_config(game_cfg).to_dict(),
        },
        "reference_pool": {
            "references": [
                ref_pol.provenance_dict(policy_index=idx) for idx, ref_pol in enumerate(references)
            ],
            "scripted_opponents": scripted_names,
            "opponents": opp_descs,
        },
        "policies": [
            cand.provenance_dict(policy_index=idx) for idx, cand in enumerate(dedup_candidates)
        ],
        "candidates": cand_results,
        "results_by_sha256": results_by_sha256,
        "standings": ranked_standings,
    }


def format_anchor_markdown(result: Mapping[str, Any]) -> str:
    """Format an ``anchor`` evaluation result payload as GitHub-flavored Markdown."""
    cfg = result.get("eval_config", {})
    ref_pool = result.get("reference_pool", {})
    opponents = ref_pool.get("opponents", [])
    opp_summary = (
        ", ".join(f"{o['label']} ({o['seats_per_world']:g} seats)" for o in opponents) or "none"
    )

    act_mode = "greedy" if result.get("greedy") else "stochastic"
    lines: list[str] = [
        "# Game V3.1 Anchor Evaluation Report",
        "",
        f"- **Mode:** `anchor` | **Commit:** `{result.get('code_commit', 'unknown')}` | "
        f"**Device:** `{result.get('device')}` | **Action Selection:** "
        f"`{act_mode}` | **Seed:** `{result.get('seed')}`",
        f"- **Horizon per Candidate:** `{cfg.get('num_envs')}` worlds × "
        f"(`{cfg.get('burnin_ticks')}` burn-in + `{cfg.get('window_ticks')}` window = "
        f"`{cfg.get('total_ticks_per_candidate')}` ticks), "
        f"`{cfg.get('learner_slots')}/{cfg.get('num_snakes')}` candidate seats/world",
        f"- **Fixed Reference Pool:** {opp_summary}",
        f"- **Candidates Completed:** `{cfg.get('num_candidates_completed')}/"
        f"{cfg.get('num_candidates_requested')}` "
        f"(reused `{cfg.get('num_candidates_reused', 0)}`, "
        f"wall `{float(result.get('wall_seconds', 0.0)):.2f} s`)",
        "",
        "## Candidate Scores vs Fixed Reference Pool (95% World-Bootstrap CIs)",
        "",
        (
            "| Candidate | Orig SHA256 | Net | Time-Avg Len [95% CI] | "
            "Final Surv Len [95% CI] (Surv) | Deaths/1k [95% CI] | Kills/1k [95% CI] | "
            "K/D [95% CI] | End Mass Share [95% CI] |"
        ),
        "| :--- | :--- | :---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for c in result.get("candidates", []):
        m = c["metrics"]
        sha12 = str(c["checkpoint_sha256"])[:12]
        tl_ci = m["time_averaged_length_ci95"]
        fl_ci = m["final_survivor_length_ci95"]
        d_ci = m["deaths_per_1k_ci95"]
        k_ci = m["kills_per_1k_ci95"]
        kd_ci = m["kill_death_ratio_ci95"]
        ms_ci = m["final_arena_mass_share_ci95"]
        d_rate = m["deaths_per_1k_agent_steps"]
        k_rate = m["kills_per_1k_agent_steps"]
        lines.append(
            f"| `{c['label']}` | `{sha12}` | {c['net_size']} | "
            f"{m['time_averaged_length']:.2f} [{tl_ci[0]:.2f}, {tl_ci[1]:.2f}] | "
            f"{m['final_survivor_length']:.2f} [{fl_ci[0]:.2f}, {fl_ci[1]:.2f}] "
            f"({m['final_survivors']}/{m['candidate_seats_total']}) | "
            f"{d_rate:.3f} [{d_ci[0]:.3f}, {d_ci[1]:.3f}] ({m['deaths']}) | "
            f"{k_rate:.3f} [{k_ci[0]:.3f}, {k_ci[1]:.3f}] ({m['kills']}) | "
            f"{m['kill_death_ratio']:.2f} [{kd_ci[0]:.2f}, {kd_ci[1]:.2f}] | "
            f"{100.0 * m['final_arena_mass_share']:.2f}% "
            f"[{100.0 * ms_ci[0]:.2f}%, {100.0 * ms_ci[1]:.2f}%] |"
        )

    lines.extend(
        [
            "",
            "## Per-Opponent Kill / Death Split (95% World-Bootstrap CIs)",
            "",
            (
                "| Candidate | Opponent / Source | Kind | Kills Against (/1k [95% CI]) | "
                "Deaths From (/1k [95% CI]) | K/D vs Opp [95% CI] | Opp End Mass % |"
            ),
            "| :--- | :--- | :---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for c in result.get("candidates", []):
        for opp in c.get("by_opponent", []):
            k_ci = opp["candidate_kills_per_1k_ci95"]
            d_ci = opp["candidate_deaths_per_1k_ci95"]
            kd_ci = opp["kill_death_ratio_ci95"]
            ok_cnt = opp["candidate_kills_against"]
            ok_1k = opp["candidate_kills_per_1k_agent_steps"]
            od_cnt = opp["candidate_deaths_from"]
            od_1k = opp["candidate_deaths_per_1k_agent_steps"]
            lines.append(
                f"| `{c['label']}` | `{opp['label']}` | {opp['kind']} | "
                f"{ok_cnt} ({ok_1k:.3f} [{k_ci[0]:.3f}, {k_ci[1]:.3f}]) | "
                f"{od_cnt} ({od_1k:.3f} [{d_ci[0]:.3f}, {d_ci[1]:.3f}]) | "
                f"{opp['kill_death_ratio']:.2f} [{kd_ci[0]:.2f}, {kd_ci[1]:.2f}] | "
                f"{100.0 * opp['opponent_final_arena_mass_share']:.2f}% |"
            )
        non_opp = c.get("non_opponent_deaths", {})
        ff = non_opp.get("candidate_friendly_fire", {})
        wl = non_opp.get("wall", {})
        if ff:
            ff_ci = ff["deaths_per_1k_ci95"]
            ff_1k = ff["deaths_per_1k_agent_steps"]
            lines.append(
                f"| `{c['label']}` | `candidate_friendly_fire` | friendly | "
                f"{ff['kills']} ({ff_1k:.3f} [{ff_ci[0]:.3f}, {ff_ci[1]:.3f}]) | "
                f"{ff['deaths']} ({ff_1k:.3f} [{ff_ci[0]:.3f}, {ff_ci[1]:.3f}]) | "
                f"1.00 [1.00, 1.00] | — |"
            )
        if wl:
            wl_ci = wl["deaths_per_1k_ci95"]
            wl_1k = wl["deaths_per_1k_agent_steps"]
            lines.append(
                f"| `{c['label']}` | `wall` | boundary | "
                f"0 (0.000 [0.000, 0.000]) | "
                f"{wl['deaths']} ({wl_1k:.3f} [{wl_ci[0]:.3f}, {wl_ci[1]:.3f}]) | "
                f"0.00 [0.00, 0.00] | — |"
            )

    return "\n".join(lines) + "\n"
