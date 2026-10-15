# Panel review summary: B8's scripted-survival gap (2026-10-14/15)

One-page summary of the two-expert panel (game mechanics + deep RL, chaired, read-only, run from
the Oct-8 repo on another machine) and of the Mac verification of its claims. Full texts:
`panel_2026-10-15/` (decision memo, round-1/2 memos, librarian answers); Mac verification with line
numbers and per-world numbers: `panel_mac_answers_2026-10-14.md`. Not gate evidence.

## Where we stand

B8 (ego2s-bv8 student, no veto) beats the served agent frp3-s12 + v8 on H5000 mass by +128.7
(LB +101.5, 5/5 seeds) and is non-inferior on survival against it on all three mixes. Phase R
returned PARTIAL because one clause failed: alive-frame scripted survival vs champion_a5 + v8 on
seed 0's 32-world bank, -0.156 at a -0.05 threshold.

## What the panel concluded (and what held up)

| Claim | Status on the Mac |
|---|---|
| Deaths are late neck-sealing enclosures at L >= 800 in crowded scripted worlds; the census trap flag marks the seal, not the commitment | Holds. Pocket at the seal is ~130 cells (IQR 32-159); a quarter of seals drop the region from >1800 to <160 in one frame. |
| The learner cannot value them: Huber beta 1 at a death delta of ~Phi(L)+3 (100-130 Q units) fits a median target, so sub-50% death risk is nearly invisible | Holds. `smooth_l1_loss` default beta, PBRS kept plus mass term, death target -Phi(L)-3; q_fatal -92..-99 at L 1284. |
| Replay content cycles with the 5000-frame segment clock (1.28M period); `death_len_mean` 200-235 was phase-confounded | Holds. CTRL_s10 qmax dips every 1.3M; death_len_mean cycles 86-1230, cycle mean ~600-700. |
| M6 fine-tunes ran the v8-margin anchor at lambda 1.0, a "keep your argmax" trust region | **Wrong.** lambda 0.1 for all 8M; anchor ~0.5% of the loss. Anchor-off is a cheap ablation only. |
| A growth governor can pay if hazard rises with length at fixed frame (rule R6) | **Does not fire.** Hazard is flat in L at fixed frame and rises 3x with frame at fixed L. No governor, ever. |
| "Tried and null" (M6) overstates: 5 x 32 had 63% power for +0.11 | Holds. |
| The 0.64 target mixed episode-level census numbers with the clause's alive-frame metric | Holds. Clause-metric gap vs champion: -0.039 pooled on the census bank (B8_s0 alone +0.036); look-2's -0.156 (paired SE 0.067) came from one bank where B8_s0 died 19/32. |
| Perception is last; most tail-topology quantities are already inputs; no boundary-TTL input exists | Holds (bv8 = area_g, head_risky, landing_ok, v5_eligible, spacious x 6; area cap 2L). |
| Exact counterfactual rollouts are feasible in all three mixes | Holds in principle; GridBatchSim deep-copy parity over a long replay not yet tested. |

## Recommended plan (panel + Mac)

1. **T1, measurement, no training.** Zero-compute half done (per-world look-2 records, census
   alive-frame survival, hazard crosstab, trap-onset region). Remaining: deep-copy parity and
   timing; exact-replay commitment/contrast study on the 75 census deaths (last avoidable decision,
   Q-gap, surviving action, tail-in-region, sealer owner; routing rules R1-R5 fixed in the memo);
   logistic probe on B8's 256-d fuse layer vs the M7 scalar baselines (AUC 0.91/0.81/0.70).
2. **T2, learner-regime screen.** HUBER (new TD-loss beta flag, ~100 at the current Q scale) x
   ANCHOR0 (`--anchor-floor 0` on the M6 recipe), 5 paired 8M fine-tunes from the B8 finals, 96
   fresh scripted worlds per seed, both survival metrics, mass vs the incumbent. GO +0.06 with
   mass >= -30; KILL <= 0. CTRL reuses M6 CTRL_s10-14. Mac only (~9 h).
3. **T3, rollout-labelled late-game training (training-time only)**, gated on T1 showing a
   surviving alternative within ~150 decisions for most deaths. Sparse decisive labels through the
   LabelBuffer, snapshot restarts; serving stays a bare network. Disclose in the result doc.
4. Follow-on if GO: 20M retrain with the winning recipe, fresh-bank Phase R under the rule fixed
   beforehand.

## Decisions for the owner

- **Guard clause.** Panel and Mac recommend option B: drop the champion comparator (c2 already
  protects scripted alive-frame survival vs the served agent at -0.03 pooled over 5 seeds), add a
  per-seed floor (-0.15 vs the served agent) and a pooled episode-survival floor (0.50 per mix).
  Amend once, prospectively, fresh banks, never re-score look 2, publish both metrics. Options A/C
  (keep the champion comparator, with power fixed or on episode-level) only if the product value is
  "the Watch snake rarely dies visibly"; B8 as trained then probably fails.
- **Planner.** Training-time rollout labels count as "no veto" with disclosure; no serving-time
  planner. Panel and Mac agree.
- **Pod spend.** Mac for T1 and T2; cap this round at $10.
- **Eligibility.** Prefer a 20M retrain over promoting an 8M fine-tune.

## State at hand-off

All work is on `main` (this doc, the panel texts, the Mac answers, the M6/M4 docs). Census README
reworded as the verifier asked. The T1 Part A and Part C agents were started and stopped before
producing code (token budget); resume from the specs in `panel_mac_answers_2026-10-14.md`
section 6 and the memo's section 4.2. Paused workflows (governor / ensemble / planner) are
unchanged from `m7-paused-state`.
