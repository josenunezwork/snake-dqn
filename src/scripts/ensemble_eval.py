#!/usr/bin/env python3
"""Evaluate a Q-averaging ENSEMBLE hero vs a single baseline, paired seeds.

Rationale: two challengers (extended-arena, 12-opp) had higher MEAN mass than the
champion but lost on significance due to high variance. Ensembling independent
checkpoints averages out per-seed variance — so the ensemble may cross into a
paired-significant win where each member alone could not.

Hero (snake 0) runs either the ensemble (mean Q over member nets) or the single
baseline; opponents (snakes 1..N-1) share one frozen policy. Paired per-seed
Δmean_mass (ensemble - baseline) with 95% CI decides promotion.

Usage:
  ./venv/bin/python src/scripts/ensemble_eval.py \
    --members saved_snakes/best_apex.pth saved_snakes/exp_arena08_long/apex_final.pth \
              saved_snakes/exp_snakes12/apex_final.pth \
    --baseline saved_snakes/best_apex.pth \
    --opponent saved_snakes/best_apex_stage1_20260615.bak.pth \
    --frames 2500 --seeds 0-19
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.core.config_loader import apply_config_to_game_config, load_config  # noqa: E402
from src.core.game_config import initialize_config  # noqa: E402
from src.main import configure_eval_game_state, create_training_game_state  # noqa: E402
from src.scripts.tournament_eval import build_policy_from_checkpoint, set_seed  # noqa: E402
from src.training.action_mask import (  # noqa: E402
    coerce_action_mask,
    mask_invalid_q_values,
    resolve_action_mask,
)


class EnsemblePolicy:
    """Greedy Q-averaging ensemble; mirrors ApexPolicy.select_action's greedy path."""

    def __init__(self, member_nets, output_size, device):
        self.members = member_nets
        for m in self.members:
            m.eval()
        self.output_size = output_size
        self.device = device
        self.use_gru = False
        self.epsilon = 0.0
        self.training = False
        self.memory = None
        self.dqn = member_nets[0]
        self.target_dqn = member_nets[0]

    def select_action(self, state, snake_id=None, action_mask=None):
        with torch.no_grad():
            if state.dim() == 1:
                state = state.unsqueeze(0)
            state = state.to(self.device)
            sel_mask = None
            if action_mask is not None:
                probe = torch.empty((state.shape[0], self.output_size), dtype=torch.float32, device=state.device)
                sel_mask = coerce_action_mask(action_mask, probe)
                if sel_mask.dim() == 1:
                    sel_mask = sel_mask.unsqueeze(0)
            q = torch.stack([m(state) for m in self.members], dim=0).mean(dim=0)
            resolved = resolve_action_mask(q, state, action_masks=sel_mask)
            if resolved.shape == q.shape and not bool(resolved.any()):
                fb = torch.zeros_like(resolved)
                fb[..., : min(3, self.output_size)] = True
                resolved = fb
            masked = mask_invalid_q_values(q, state, action_masks=resolved)
            return int(masked.argmax(dim=-1).item())

    # no-ops the game/eval may call
    def reset_hidden(self, *a, **k):
        pass


def parse_seeds(v):
    if "-" in v and "," not in v:
        a, b = v.split("-")
        return list(range(int(a), int(b) + 1))
    return [int(x) for x in v.split(",") if x.strip()]


def hero_rollout(make_hero_policy, opponent_path, frames, seed):
    set_seed(seed)
    gs = create_training_game_state(eval_mode=False)
    gs.snakes[0].policy = make_hero_policy()
    opp = build_policy_from_checkpoint(opponent_path)
    for s in gs.snakes[1:]:
        s.policy = opp
    gs._shared_policy = None
    configure_eval_game_state(gs)
    hero = gs.snakes[0]
    max_mass = len(hero.segments); mass_sum = 0.0; alive = 0; deaths = 0; kills = 0
    prev = hero.is_alive
    for _ in range(frames):
        gs.update(train_mode=True, learn=False)
        if hero.is_alive:
            m = len(hero.segments); max_mass = max(max_mass, m); mass_sum += m; alive += 1
        if prev and not hero.is_alive:
            deaths += 1
        prev = hero.is_alive
        kills += len(gs.frame_kills.get(hero.id, []))
    gs.full_cleanup()
    return {"mean_mass": mass_sum / alive if alive else 0.0, "max_mass": float(max_mass),
            "kills": float(kills), "deaths": float(deaths)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", required=True)
    p.add_argument("--baseline", required=True)
    p.add_argument("--opponent", required=True)
    p.add_argument("--config", default="configs/goal_slither_20260615.yaml")
    p.add_argument("--frames", type=int, default=2500)
    p.add_argument("--seeds", type=parse_seeds, default=list(range(20)))
    p.add_argument("--json-output", default=None)
    args = p.parse_args()

    cfg = load_config(args.config); initialize_config(cfg); apply_config_to_game_config(cfg)
    device = torch.device("cpu")
    out = None

    def make_ensemble():
        nets = [build_policy_from_checkpoint(m).dqn for m in args.members]
        return EnsemblePolicy(nets, nets[0].state_dict()["advantage_stream.2.weight"].shape[0], device)

    def make_baseline():
        return build_policy_from_checkpoint(args.baseline)

    print(f"Ensemble ({len(args.members)} members) vs baseline {Path(args.baseline).name} | opp {Path(args.opponent).name}")
    ens, base = [], []
    for sd in args.seeds:
        e = hero_rollout(make_ensemble, args.opponent, args.frames, sd)
        b = hero_rollout(make_baseline, args.opponent, args.frames, sd)
        ens.append(e); base.append(b)
        print(f"  seed {sd}: ensemble {e['mean_mass']:.1f} (k{e['kills']:.0f}) | baseline {b['mean_mass']:.1f} (k{b['kills']:.0f})")

    diffs = [e["mean_mass"] - b["mean_mass"] for e, b in zip(ens, base)]
    n = len(diffs); m = sum(diffs) / n
    sd_ = (sum((x - m) ** 2 for x in diffs) / (n - 1)) ** 0.5
    ci = 1.96 * sd_ / math.sqrt(n)
    em = float(np.mean([e["mean_mass"] for e in ens])); bm = float(np.mean([b["mean_mass"] for b in base]))
    ek = float(np.mean([e["kills"] for e in ens])); bk = float(np.mean([b["kills"] for b in base]))
    sig = "SIGNIFICANT" if (m - ci) > 0 else "ns"
    print(f"\nENSEMBLE mean_mass {em:.1f} (kills {ek:.2f}) vs BASELINE {bm:.1f} (kills {bk:.2f})")
    print(f"paired Δ = {m:+.2f} ± {ci:.2f}  [{m-ci:+.2f}, {m+ci:+.2f}]  wins {sum(1 for x in diffs if x>0)}/{n}  -> {sig}")
    if args.json_output:
        Path(args.json_output).write_text(json.dumps({"ens": ens, "base": base, "delta": m, "ci": ci}, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
