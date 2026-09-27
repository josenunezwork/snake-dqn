# Apex serving-time free-space veto: H5000 dev screen, 2026-09-27

Status: **pre-registered decision INCOMPLETE.** The primary paired comparison
finished. The determinism control did not: the deadline stopped it after 1 of
24 games. This is a Tier-1, non-authoritative screen. Nothing is promoted, and
Apex/vector61 remains the incumbent.

Source: branch `apex-safety-and-eval-tooling` at `4eec2d7`, local only.

- Code: `research/apex_safety_20260926/`
- Veto: `src/evaluation/safety_veto.py` (opt-in, default off)
- Run root: `snake-dqn-artifacts/apex-safety-screen-20260926/run-v1`

## Why

At H5000, 85 of the champion's 96 recorded deaths in the strict pilot were
self-collisions, and Apex was alive for only 29–37% of the horizon. Apex already
computes per-action flood-fill free space as an input but does not act on it.
The scripted opponents already use a free-space veto.

## What ran

Two arms were compared:

- **A:** the champion `champion_a5_freespace_20260621.pth` (`43d4e2c5…`).
- **B:** the same champion plus a serving-time veto (`free-space-veto/v2-speed-preserving`).

When the chosen move leads to reachable space smaller than
`min(logical length, cap)` and a masked-legal move with enough space exists,
the veto swaps it for the highest-Q such move in the same speed mode. It uses
the same threshold as the scripted-snake veto.

Setup:

- **Profile:** `promotion-v2-watch-rect`, H5000.
- **Mixes:** frozen, scripted and mixed, with the strict pilot's roster construction.
- **Worlds:** 40 per mix, from the new namespace `apex-safety-screen-v1`,
  checked disjoint from earlier namespaces.
- **Pairing:** by world.

All 240 A/B games completed:

- A averaged 29.8 s per game; B averaged 33.5 s.
- Host load was high, so games took far longer than the ~18 s estimate.

## Primary result

Mass-integral difference, B − A (one-sided Holm over the three mixes):

| Mix | Mean A | Mean B | Δ (95% CI, descriptive) | Holm-adj p | Worlds B better / equal / worse |
|---|---:|---:|---|---:|---|
| frozen | 66.1 | 109.5 | +43.3 [15.6, 71.1] | 0.0023 | 26 / 9 / 5 |
| scripted | 35.6 | 90.7 | +55.1 [23.1, 87.2] | 0.0019 | 22 / 13 / 5 |
| mixed | 64.9 | 112.5 | +47.7 [18.1, 77.2] | 0.0023 | 21 / 14 / 5 |

All three mixes pass the pre-registered Holm superiority test. Only two had to
pass. The pooled random-effects estimate is +48.2, with a 90% CI of
[23.5, 72.8] and I² = 0. Survival fraction rose in step with mass.

Veto mechanics:

- The veto changed an action in 26–32 of 40 games per mix, with 69–102 vetoes
  per mix. That is about 0.1% of decisions.
- There were 747–942 decisions per mix where no move had enough space; the
  veto left those unchanged.
- Most B deaths are still self-collisions (33–37 of 40).

So the one-step veto delays self-traps rather than eliminating them. A
multi-step or tail-aware check may do more.

## Why the decision is INCOMPLETE

The pre-registered rule checks completeness first. The determinism control
(arm C, which repeats A on 8 worlds per mix) ran 1 of 24 games before the
deadline. That one game was byte-identical to its A game. The rule therefore
yields INCOMPLETE rather than RECOMMEND_STRICT_GATE, and it is not relabeled.

## Limits

- Tier-1 dev screen, unaudited, one run, fixed rosters.
- It is not a strict-gate result.
- The effect is on the champion checkpoint only.
- The informational strict sizing at the fixed 10%-of-incumbent MDE still asks
  for 1,610–7,426 worlds per mix, because per-world SD is about 87–100.
- The observed effect is 7–15× that MDE. A strict run needs a pre-declared MDE
  justified independently of this result, and ideally the SIMD engine
  (vector61 featurizer at bit parity, not yet wired).
