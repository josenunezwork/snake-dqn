# B8 Late-Game Death Census, Two-Expert Panel and T1–T11 Probe Series (2026-10-14 to 2026-10-17)

## Question and Goal

At M4 Phase R look 2, `B8` (`ego2s-bv8`, no veto) beat the served `frp3-s12 + v8` by **+128.7** H5000 mass (SE 16.4, LB **+101.5**) and passed the pooled scripted survival clause (`-0.003 >= -0.03`), but failed the single-seed guard `c6` (scripted surv5 vs `champion_a5 + v8`, seed 0: **-0.156** vs `>= -0.05`) (`docs/research/panel_2026-10-15/PANEL_DECISION_MEMO.md` §2.3). This lineage asked why B8 dies late in scripted games, whether that headroom can be taught to a bare network (labels, losses, inputs), and how the guard clause should be stated before the confirmation gate (T7).

## Timeline of Studies and Steps

| Date (PT) | Study / Step | What Was Run | Verdict |
|---|---|---|---|
| 2026-10-14 | `b8_death_census_20261014` | Scripted death census of M4 B8 (5 seeds × 32 round-52 worlds) against `frp3-s12+v8` and `champion_a5+v8`. | **Self-enclosure at `L >= 800`**: `49/75` B8 deaths at `L >= 800`; median trap formed ~31 decisions before death. |
| 2026-10-15 | Two-expert panel (`panel_2026-10-15/`) | Mechanics + deep-RL panel, two rounds plus librarian answers. | Shared diagnosis: late deaths are sealed-neck enclosures; Huber loss at beta 1 and the keep-your-argmax anchor limit learning; recommended measurement, then a learner-regime screen, then rollout labels. Guard clause sent to the owner (advice favored option B). |
| 2026-10-15 | T1 Part A commitment (`t1_commitment_20261015`) | Exact backward replay of all 75 census deaths (49 late, 26 early), all advisory-legal actions followed by B8 greedy for 300 transitions. | Late: survivor within 300 decisions `46/49` (93.9%), within 150 `25/49`; R1 (T3 eligible) **does not fire**; R2 value learning is the limit; R3 tail-topology inputs **NO_GO**; R4 include boost labels. |
| 2026-10-15 | T1 Part C fuse probe (`t1_fuse_probe_20261015`) | World-held-out LR on B8's 256-D post-ReLU fuse layer. | Routing: **value learning is the limit** (pooled nested LR H100 AUC `0.837 [0.773, 0.897]`). |
| 2026-10-15 | T2 learner-regime screen (`t2_screen_20261015`) | 5 paired seeds × 8M from B8_s0–4: `HUBER` (delta 100), `ANCHOR0` (floor 0), `HUBER_ANCHOR0`, vs M6 CTRL. | **INCOMPLETE** (HUBER seed 12 hit the action-collapse tripwire); ANCHOR0 EXTEND exhausted (`+0.003` episode delta at 192 worlds); HUBER_ANCHOR0 **KILL** (mass collapse). |
| 2026-10-15 | T3 rollout ceiling (`docs/research/t3_ceiling_test_2026-10-15.md`) | Sparse Q-threshold trigger: **NO_GO** (`41/64` vs `0.65` bar). Dense addendum (every fifth decision at `L >= 600`, all actions, D=300, B8 continuation). | **Dense GO**: pooled scripted survival `54/64` vs B8 `38/64`; mass `700.0` vs `608.1`; 75 changed actions in 301,589 decisions. |
| 2026-10-16 | T3 label screen (`docs/research/t3_screen_result_2026-10-15.md`) | `LABELS`, `LABELS_KEEPV8`, `LABELS_KEEPV8_LO` students vs CTRL. | **No GO**: scripted episode delta `+0.008 / +0.017 / -0.010`; scripted mass vs CTRL `-88.0 / -87.9 / -81.2`. |
| 2026-10-16 | T4a two-sided margin (`docs/research/t4a_screen_result_2026-10-16.md`) | `TWO_SIDED` counter-row screen. | **KILL**: scripted episode delta `-0.027`, mass `-24.1` vs CTRL. |
| 2026-10-16 | T4e/T4f `RETURN` / `RETURN_C` (advice `2026-10-16_return_halt.md`, `2026-10-16_return_c_review.md`) | Discounted H300 return regression from one-step-deviation branches. | **Halted** at 0.8M with greedy boost rate 34% and return loss ~200; advice: one-step-deviation targets flatten the action landscape under telescoping PBRS, and one boost frame costs no segment. |
| 2026-10-16 | T5 separability (`t5_separability_20261016`) | 24 per-action geometry features on 19,976 restored T4e states / 342 worlds, 5-fold world-held-out. | **FAIL**: death-AUC gain `+0.0088` (bar `+0.05`), 24-alone AUC `0.748` (bar `0.80`), T1 share `17/46` (bar 70%). |
| 2026-10-16 | T5b risk and fuse (`t5b_risk_and_fuse_20261016`) | Guard-failure risk model on reused H5000 records; fuse-layer probes. | Modeled survival-guard failure risk **38.81%** (> 25% flag); all fuse probes `< 0.85` AUC (GBM `0.7630–0.7693`). |
| 2026-10-16 | **T7 B8 fresh-bank Phase R** (`docs/research/t7_phase_r_result_2026-10-16.md`) | One look over 5,760 H10000 episodes + 45 H5000 controls, Mac CPU. | **STRICT_PASS**, seed 3: pooled mass `+138.67` (LB90 `+129.48`), scripted alive `-0.0069` (`>= -0.03`), Tier 2 scripted episode survival vs champion `-0.0156` (`>= -0.05`). |
| 2026-10-17 | T6 snapshot restart (`docs/research/t6_screen_result_2026-10-17.md`) | 5-seed RESTART screen. | **KILL**: scripted survival delta `0.0000` (primary `<= 0`), scripted mass `-2.20` vs CTRL. |
| 2026-10-17 | T8 probes (`t8_probes_20261017`) | Arrival-race features and a raw-raster D300 CNN on the T5 bank. | **FAIL vs 0.85 bar**: CNN AUC `0.742`; arrival features `+0.014` (GBM `0.735 → 0.749`). |
| 2026-10-17 | T9 divergence (`t9_divergence_20261017`) | Fatal vs surviving branch replay after each of the 46 late LADs. | No clean early separator; first 75% directional separation at offset **49** (race-128/256 smaller in `20/26` pairs), sustained at offsets 65–83. |
| 2026-10-17 | T10 certificate/horizon (`t10_certificate_horizon_20261017`) | Tail-by-L certificate audit; horizon-stratified AUC. | Certificate held on `99.970%` (dense) vs `99.915%` (B8) of eligible decisions — not separating; AUC decays D30→D300 (e.g. 60 scalars `0.912 → 0.726`), not monotonically. |
| 2026-10-17 | T11 LOCAL30 (`t11_local30_20261017`, `t11_screen_20261017`) | 5-seed fine-tune screen with LOCAL30 inputs. | **KILL before gameplay**: seed 13 halted at 300,160 transitions with `L < 700` greedy boost `1.073%` > `1%` limit. |

## Final Verdict and Key Numbers

1. **Census (`PANEL_DECISION_MEMO.md` §2.1, round 52, 32 scripted worlds per arm):**

| Agent | Survival | Deaths | Mass | Median L at death |
|---|---|---|---|---|
| B8 pooled (5 seeds) | 0.531 | 75 | 570.7 | 974 |
| frp3-s12 + v8 | 0.656 | 11 | 454.6 | 643 |
| champion_a5 + v8 | 0.688 | 10 | 368.7 | 520 |

   - B8 hazard per 1,000 decisions: `0.068` at `L < 800` (26 deaths, 58% of decisions) vs `0.177` at `L >= 800` (49 deaths, 42% of 659,595 decisions). The champion spends only 5.6% of decisions at `L >= 800` (0 deaths, 1.2 expected at B8's rate).
   - 85% of the 75 deaths are enclosures; median trap 31 decisions; 41 of 64 trap onsets had no one-step escape.
   - A 32-vs-32 alive-frame survival difference has SE about `0.075` (MDE about `0.19`), so `c6` on one seed is a weak test both ways.

2. **Headroom vs distillation:** exact search has large headroom (T1: `46/49` late deaths escapable within 300 decisions, LAD Q-gap median `0.45`; `17/46` escapable only by boost, B8 chose boost at 2 of 46; T3 dense: `54/64` vs `38/64` survival), but every training route tried (T2, T3 labels, T4a, T4e/f, T6, T11) failed its screen.

3. **Inputs/representation:** static features (T5), fuse probes (T1c, T5b), arrival features and a raw-raster CNN (T8) all stayed near `0.73–0.77` D300 AUC; T9/T10 found no early separator or certificate that dense search holds and B8 does not.

4. **Gate:** under the restated guard, **B8 seed 3 passed T7 STRICT_PASS** (2026-10-16) and was released on 2026-10-17 (see [`b8-release-serving.md`](b8-release-serving.md)).

## What Was Learned

1. B8's scripted gap is concentrated at `L >= 800`, where it spends far more time than the champion; below 800 it is about as safe per decision.
2. The deaths are multi-step enclosures that greedy one-step features and the frozen representation do not predict well before the seal; separation between fatal and surviving branches appears only ~49–65 decisions after the LAD.
3. Exact rollout search fixes many of them with very sparse interventions (75 changed actions), but the student routes tried over-perturbed ordinary states or drifted into boost (T3 labels, T4e/f, T11).

## Why It Closed

The T1–T11 probe and fine-tune line closed on 2026-10-17 after T11's KILL; meanwhile B8 seed 3 passed T7 and was released. Work moved to the post-release plan (T12–T13 serving/roster, T21–T23, Q0/Q1) and later the T34 search-teacher line.

## Kept Lineage Documents

- [PANEL_DECISION_MEMO.md](../panel_2026-10-15/PANEL_DECISION_MEMO.md) — panel synthesis, census tables and guard analysis.
- [research/b8_death_census_20261014/README.md](../../../research/b8_death_census_20261014/README.md) — census package.
- [research/t1_commitment_20261015/README.md](../../../research/t1_commitment_20261015/README.md) and [docs/research/t1_commitment_study_2026-10-15.md](../t1_commitment_study_2026-10-15.md) — T1 Part A.
- [research/t1_fuse_probe_20261015/README.md](../../../research/t1_fuse_probe_20261015/README.md) and [docs/research/t1_fuse_probe_2026-10-15.md](../t1_fuse_probe_2026-10-15.md) — T1 Part C.
- [docs/research/t3_ceiling_test_2026-10-15.md](../t3_ceiling_test_2026-10-15.md), [docs/research/t3_screen_result_2026-10-15.md](../t3_screen_result_2026-10-15.md) — T3 ceiling and label screen.
- [docs/research/t7_phase_r_result_2026-10-16.md](../t7_phase_r_result_2026-10-16.md) — T7 STRICT_PASS.
- [2026-10-15_t3_launch_review.md](../advice/2026-10-15_t3_launch_review.md), [2026-10-16_return_arm_review.md](../advice/2026-10-16_return_arm_review.md), [2026-10-16_inputs_design.md](../advice/2026-10-16_inputs_design.md), [2026-10-16_strategy.md](../advice/2026-10-16_strategy.md), [2026-10-17_clever_ideas.md](../advice/2026-10-17_clever_ideas.md) — advice.
- [research/t10_certificate_horizon_20261017/README.md](../../../research/t10_certificate_horizon_20261017/README.md), [docs/research/t10_certificate_horizon_2026-10-17.md](../t10_certificate_horizon_2026-10-17.md) — T10.
- [research/t11_local30_20261017/README.md](../../../research/t11_local30_20261017/README.md), [research/t11_screen_20261017/README.md](../../../research/t11_screen_20261017/README.md), [docs/research/t11_screen_result_2026-10-17.md](../t11_screen_result_2026-10-17.md) — T11.
- Removed in consolidation (see git c4c4ab41): `research/t2_screen_20261015/README.md`, `research/t5_separability_20261016/README.md`, `research/t5b_risk_and_fuse_20261016/README.md`, `research/t8_probes_20261017/README.md`, `research/t9_divergence_20261017/README.md`, `docs/research/t9_divergence_2026-10-17.md`, the round-1/2 panel files and the other 2026-10-15..17 advice notes.
