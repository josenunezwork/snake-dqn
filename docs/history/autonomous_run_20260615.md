# Autonomous RL improvement run — started 2026-06-15 ~23:30

Mandate: ~8 hours autonomous, try diverse techniques, continually improve the
snake. All decisions self-made. Gate every promotion on `tournament_eval.py`
(absolute skill vs ≥2 frozen opponents, paired seeds, 95% CI) — promote only on a
CI-separated win. Champion protected (read-only `champion_*.pth` backups).

Baseline champion at start: **S4** (`best_apex.pth`), feedforward, boost-fix
(boost_segment=3.0), geom 8×0.6. Absolute mass ~36 (vs frozen S1) / ~52 (vs pre).

Frozen opponent pool for eval: `best_apex_stage1_20260615.bak.pth` (S1),
`best_apex_pre_slither_20260615.bak.pth` (pre-session).

## Experiment ledger

| # | technique | result | promoted? |
|---|-----------|--------|-----------|
| infra | DRQN wiring (factory use_gru + worker config propagation) | fixed + verified | n/a |
| 1 | arena-align board_scale 0.6→0.8 (warm-start S4) | mass 36→**60** vs S1, 52→**64** vs pre (CI-sep vs S1, rank1 vs both) | ✅ **promoted (champion: arena-0.8)** |
| 2 | arena-align board_scale 0.8→1.0 | REGRESSED (38/30 vs champ 60/64); board 1.0 too sparse, 0.8 is the sweet spot | ❌ rejected |
| 3 | DRQN full run (recurrent, local headless) | NOT COMPETITIVE on CPU budget: mass ~5-6 vs champ ~60 after 40min, flat. From-scratch recurrent too slow vs warm-started feedforward. | ❌ killed |
| 4 | reward: mass-proportional death penalty (scale 1.0) | worse (47/51 vs champ 60/65); no survival gain | ❌ rejected |
| 5 | longer arena-0.8 train (230k→350k) | marginal (final 63.6/70.9 vs champ 59.5/64.3) but NOT CI-separated, higher variance | ❌ not promoted (noise) |
| 6 | boost_segment 3.0→4.0 sweep | worse/equal (55/63 vs champ 60/64); 3.0 is the sweet spot | ❌ not promoted |
| 7 | bigger network hidden=1024 from scratch | BLOCKED: apex distributed path doesn't propagate network.hidden_size to all workers (learner 1024 vs actor 512 mismatch). Not worth debugging on budget. | ⏭ skipped |
| 8 | from-scratch 512, clean contract (board 0.8 from step 0) | mass 19/22 vs champ 58/64 at 100k; far behind. INSIGHT: progressive arena curriculum (0.35→0.6→0.8) is essential — from-scratch on sparse board 0.8 underperforms (same reason arena-1.0 failed). | ❌ killed |
| 9 | more opponents 8→12 (denser crowding) | directional gain (Δmass +8.5/+6.3, 20-seed) + EMERGENT KILLS (~0.2-0.4) but paired CI includes 0 (t≈1.1/0.7, wins 14/20,13/20) | ❌ not promoted (marginal) |
| 10 | extend 12-opponent crowding (330k→480k) | REGRESSED (48/52 vs champ 56/63); 12-opp peaked ~330k then overtrained, like other continuations | ❌ not promoted |
| 11 | 12-opp crowding at lower LR (1e-4) | much WORSE (40/45 vs champ 56/63, Δ-16/-18, wins 3-5/20); low LR underfits new dist | ❌ rejected |
| 12 | DRQN-dense (recurrent on dense 820x520 arena) | hopeless again: mass ~4 vs champ ~58 after 46min. From-scratch recurrent NOT viable on CPU budget (confirmed 2x). | ❌ killed |
| 13 | kill-reward emphasis kill_base 1→4 + 12-opp crowding | WORSE (mass 36/43 vs champ 56/63, Δ-20, wins 4-5/20) and kills did NOT emerge (0.0/0.1 = same as champ). Explicit kill reward insufficient — killing needs planning, not incentive. | ❌ rejected |

## FINAL SUMMARY (autonomous run, 2026-06-15 23:30 → 2026-06-16 ~17:35)

**Champion: arena-0.8** (`saved_snakes/best_apex.pth`) — feedforward Apex DQN, board_scale
0.8, 8-snake actor env, boost-fix reward (boost_segment 3.0), gamma 0.99, n_step 3,
hidden 512, step 230000. Absolute skill (tournament vs frozen opponents, 20-seed):
**mass ~56–60 vs S1, ~63 vs pre** — up from the pre-session model's ~6 (≈**8–9× absolute gain**).
Protected read-only copies: `champion_arena08.pth`, `best_apex_s4_20260615.bak.pth`.

### What worked (the only significant wins)
1. **Progressive arena curriculum** (board 0.35 → 0.6 → 0.8): the single biggest lever.
   Each step warm-started the previous via checkpoint metadata rebase. 0.8 is the sweet spot.
2. **Boost-abuse reward fix** (penalize boost-burned segments): broke the 89%-boost exploit;
   the model went from never growing to mass ~36.
3. **Disciplined eval gate**: absolute tournament vs FROZEN opponents + paired-seed significance.
   This correctly rejected ~12 noise/regression "improvements" that self-play reward would have promoted.

### What did NOT work (13 experiments, 1 promotion)
- arena-1.0 (too sparse), continuation/over-training ×3 (peak-then-decline), from-scratch-512
  on sparse board (curriculum is essential), lower-LR (underfits), reward tweaks
  (death-scale, boost-4, kill_base-4 all neutral/worse), bigger-net-1024 (infra-blocked).
- **DRQN (recurrent) ×2 — not viable on CPU from scratch** (mass ~4–6 vs ~58 after 40-46min,
  both sparse and dense arenas). The recurrent frontier needs a GPU or a warm-started feature layer.
- **Kills never emerged robustly** — even explicit kill_base=4 produced ~0 kills. Killing is
  structural (victim must hit your body; trained opponents avoid danger) → needs planning/
  opponent-modeling, not just incentive. 12-opponent crowding gave a faint kill signal (~0.2-0.4)
  but not a paired-significant mass gain.

### Why the plateau (honest read)
The champion is a robust local optimum for this configuration: **feedforward DQN + self-play +
58-D hand-crafted state + CPU budget**. The remaining gains require capabilities this setup can't
cheaply provide (memory/planning for kills, more compute for from-scratch recurrence/capacity).

### Recommended pivots (highest-EV next, in order)
1. **GPU** — the real unlock; DRQN, hidden=1024, and from-scratch-curriculum all become viable.
2. **Warm-start DRQN from the champion's feature_layer** (reuse 58→512→256, learn only GRU+heads) —
   makes recurrence affordable even on CPU; the most promising concrete code experiment.
3. **Fix apex distributed `hidden_size` propagation** (actors built 512 while learner built 1024) —
   unblocks fast bigger-net + would let DRQN train on the fast distributed path.
4. **State enrichment for kills** — add encirclement/trajectory-prediction features so kills are
   learnable without recurrence.
5. **League / PFSP** — train against the frozen champion pool (S1/S3/S4), not just self-play, for a
   robust non-exploitable policy (needs heterogeneous opponents wired into the training env).

### Infra fixed this run (committable improvements)
- `SnakeFactory` now passes `use_gru` from config (DRQN was silently feedforward).
- Headless workers re-init config from `SNAKE_DQN_CONFIG` env (overrides reached spawned workers).
- `ApexPolicy.load_state_dict` skips the TD/reward contract when `training=False` (inference/eval/GUI
  can load any checkpoint regardless of reward contract) — fixed a backward-compat break.
- GUI `--eval` is now a true read-only watch mode (no train, no auto-save over the champion).
- `ai_snake.cleanup` tolerates inference policies (memory=None).
- New reward fields `boost_segment`, `death_length_scale`; `tournament_eval.py` (architecture-aware,
  paired-seed, CI); `rebase_checkpoint.py` (curriculum warm-start helper).

| 14 | WARM-STARTED DRQN (copy champion feature_layer+streams, only GRU fresh) | STILL hopeless: mass 5.8 vs champ 58 after ~1.5hr, best reward flat 50.9. Inserting a fresh GRU between warm-started features & heads scrambles them; relearning the GRU→head alignment is as slow as from-scratch on CPU. DEFINITIVE: recurrence needs a GPU here (3rd DRQN confirmation). | ❌ killed |

| 15 | Q-averaging ensemble (champion + extended-arena + 12-opp) | NO EFFECT: byte-identical to champion (Δ=0.00, 0/10 wins, 2500f×10seeds). Members are correlated arena-0.8 descendants — they always agree on in-distribution greedy actions, so averaging changes nothing. Ensembling needs DIVERSE (different-seed/init) models, which don't exist in this lineage. | ❌ no gain |

## STATE OF THE ART (final)
Champion = **arena-0.8** (best_apex.pth): mass ~56-60 vs frozen S1 / ~63 vs pre, ≈8-9x the
pre-session model. 15 experiments, 1 promotion. The disciplined paired-significance gate held
the champion stable against every challenger. Non-GPU levers are EXHAUSTED:
- recurrence (DRQN) needs a GPU (3x confirmed, incl. warm-start);
- capacity (1024) blocked by an apex hidden_size propagation bug + low EV;
- arena/reward/continuation/crowding all plateaued; ensembling needs diverse models we don't have.
Genuine remaining frontiers: (1) GPU -> DRQN + bigger nets + faster from-scratch; (2) STATE
ENRICHMENT for kills (add encirclement / enemy-next-position features so a feedforward net can
learn to trap opponents — the only non-GPU path to a qualitative gain, but a big from-scratch effort).

