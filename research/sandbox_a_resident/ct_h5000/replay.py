"""Successor-only replay: recorded batched Q/action is authoritative at a near tie.

The reconstruction body follows the frozen predecessor's action-tape replay.
A discrepancy outside the prospective four-ULP rule remains an integrity failure.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
from torch import nn

from research.sandbox_a_resident.ct_mechanics import CTDeviceSim, CTOpponentRuntime
from research.sandbox_a_resident.ct_snapshots import (
    CompactWorldSnapshot,
)
from research.sandbox_a_resident.ct_snapshots import LookbackRing as OriginalLookbackRing
from research.sandbox_a_resident.ct_snapshots import (
    capture_world_snapshot,
    observe_hero_bv8,
    restore_world_snapshots_into,
)
from research.sandbox_a_resident.ct_topology import compute_topology_features
from research.sandbox_a_resident.training_rosters import TrainingRoster


def qualify_tie(
    recorded: np.ndarray,
    replayed: np.ndarray,
    recorded_action: int,
    replayed_action: int,
    legal: np.ndarray,
) -> dict[str, Any]:
    """Accept only two legal greedy actions separated by <=4 float32 ULP in both forwards."""
    a, b = recorded_action, replayed_action
    r, q = np.asarray(recorded, dtype=np.float32), np.asarray(replayed, dtype=np.float32)
    r_gap, q_gap = abs(float(r[a]) - float(r[b])), abs(float(q[a]) - float(q[b]))
    r_tol = 4 * max(abs(float(np.spacing(r[a]))), abs(float(np.spacing(r[b]))))
    q_tol = 4 * max(abs(float(np.spacing(q[a]))), abs(float(np.spacing(q[b]))))
    explained = bool(
        np.isfinite(r).all()
        and np.isfinite(q).all()
        and legal[a]
        and legal[b]
        and a == int(np.argmax(np.where(legal, r, -np.inf)))
        and b == int(np.argmax(np.where(legal, q, -np.inf)))
        and r_gap <= r_tol
        and q_gap <= q_tol
    )
    return {
        "recorded_action": a,
        "replayed_action": b,
        "recorded_q": r.tolist(),
        "replayed_q": q.tolist(),
        "legal": legal.tolist(),
        "recorded_gap": r_gap,
        "replayed_gap": q_gap,
        "recorded_tolerance": r_tol,
        "replayed_tolerance": q_tol,
        "tie_explained": explained,
    }


class LookbackRing(OriginalLookbackRing):
    """Retain Q beside the existing ring action tape, without changing collection actions."""

    def __init__(
        self, *, diagnostic: Callable[[str, Mapping[str, Any]], None] | None = None, **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self.diagnostic = diagnostic
        self._q_tape = np.full((self.num_worlds, self.tape_len, 6), np.nan, dtype=np.float32)

    def record_tick_actions(
        self, *, tick: int, hero_actions: np.ndarray, life_ids: Sequence[int], q_b8: torch.Tensor
    ) -> None:
        super().record_tick_actions(tick=tick, hero_actions=hero_actions, life_ids=life_ids)
        self._q_tape[:, tick % self.tape_len] = q_b8.detach().cpu().numpy()

    @torch.inference_mode()
    def reconstruct_lookback_root(
        self,
        *,
        world_id: int,
        target_tick: int,
        expected_life_id: int,
        roster: TrainingRoster,
        cfg: Any,
        b8_network: nn.Module,
        checkpoint_networks: Mapping[str, nn.Module],
        device: torch.device | str,
        replay_cache: dict[tuple[Any, ...], tuple[CTDeviceSim, CTOpponentRuntime]] | None = None,
    ) -> tuple[
        CompactWorldSnapshot,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        int,
        np.ndarray,
        np.ndarray,
    ]:
        """Deterministically replay ``0..stride-1`` ticks to recover prepared ``target_tick``."""
        w = int(world_id)
        t_target = int(target_tick)
        t_base = (t_target // self.stride) * self.stride
        delta = t_target - t_base
        if not 0 <= delta < self.stride:
            raise RuntimeError(f"invalid replay delta={delta}")
        slot = (t_base // self.stride) % self.num_slots
        base_snap = self._slots[w][slot]
        if base_snap is None or base_snap.tick != t_base:
            raise RuntimeError(
                f"missing base checkpoint for world={w} target={t_target} base={t_base}"
            )
        if base_snap.life_id != int(expected_life_id):
            raise RuntimeError(
                f"life boundary crossed in lookback replay world={w} "
                f"base_life={base_snap.life_id} expected={expected_life_id}"
            )

        dev = torch.device(device)
        cache_key = (roster.identity, str(dev))
        if replay_cache is not None and cache_key in replay_cache:
            r_sim, r_opp = replay_cache[cache_key]
        else:
            from dataclasses import replace as dc_replace

            cfg1 = dc_replace(cfg, num_envs=1)
            r_sim = CTDeviceSim(cfg1, device=dev, train_mode=False)
            r_sim.terminal_slots[:, 0] = True
            r_opp = CTOpponentRuntime(
                r_sim,
                [roster],
                b8_network=b8_network,
                checkpoint_networks=checkpoint_networks,
            )
            if replay_cache is not None:
                replay_cache[cache_key] = (r_sim, r_opp)

        restore_world_snapshots_into([base_snap], r_sim, r_opp)

        for step_offset in range(delta):
            tau = t_base + step_offset
            pos = tau % self.tape_len
            if int(self._tape_tick[w, pos]) != tau or int(self._tape_life[w, pos]) != int(
                expected_life_id
            ):
                raise RuntimeError(f"missing action tape for world={w} tick={tau}")
            r_sim.prepare()
            recorded_act = int(self._action_tape[w, pos])
            hero_act_t = torch.tensor([recorded_act], dtype=torch.long, device=dev)
            full_actions = r_opp.actions(r_sim, hero_act_t)
            r_sim.step(full_actions)
            r_opp.after_step(r_sim)
            if bool(r_sim._last_done[0, 0].item()):
                raise RuntimeError(
                    f"unexpected hero death during lookback replay at tau={tau} < {t_target}"
                )

        r_sim.prepare()
        exact_snap = capture_world_snapshot(
            r_sim,
            r_opp,
            0,
            world_id=w,
            life_id=int(expected_life_id),
            tick=t_target,
            max_body_capacity=self.max_body_capacity,
        )
        local_obs, glob_obs, scalars_obs, legal_mask, q_b8, a_b8_t = observe_hero_bv8(
            r_sim, b8_network
        )
        b8_act = int(a_b8_t[0].item())

        pos_target = t_target % self.tape_len
        if int(self._tape_tick[w, pos_target]) != t_target or int(
            self._tape_life[w, pos_target]
        ) != int(expected_life_id):
            raise RuntimeError("missing target action/Q tape")
        recorded_target_act = int(self._action_tape[w, pos_target])
        recorded_q = self._q_tape[w, pos_target].copy()
        if b8_act != recorded_target_act:
            report = qualify_tie(
                recorded_q,
                q_b8[0].cpu().numpy(),
                recorded_target_act,
                b8_act,
                legal_mask[0].cpu().numpy(),
            )
            report.update(world=w, life=int(expected_life_id), tick=t_target)
            if self.diagnostic is not None:
                self.diagnostic("replay-ties", report)
            if not report["tie_explained"]:
                raise RuntimeError(f"unexplained B8 replay action mismatch: {report}")
        # The bank records the actual batched collection decision and Q vector;
        # no altered action is fed into the environment or substituted for B8.
        b8_act = recorded_target_act
        q_b8 = torch.from_numpy(recorded_q).to(dev).unsqueeze(0)

        planes_u16, sums_f32 = compute_topology_features(r_sim, [0])
        return (
            exact_snap,
            local_obs[0],
            glob_obs[0],
            scalars_obs[0],
            legal_mask[0],
            q_b8[0],
            b8_act,
            planes_u16[0],
            sums_f32[0],
        )
