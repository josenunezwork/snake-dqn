# FRP-v3 seed-12 M3@60000 + v8: strict gate result (2026-10-06)

**Outcome: `STRICT_PASS`** (sequential decision `STOP_PASS` at look index 2, n = 207 worlds per
mix of N_max 275). Tier-2 strict gate of the champion change *FRP-v3 arm M3, seed 12, 60000
learner updates + released v8 veto* against the incumbent *champion_a5 + v8*. No promotion was
performed by the gate; release still requires a Mac web serving qualification
(`research/frp3_checkpoint_serving_20261005/`).

## Identities

| Item | Value |
|---|---|
| Candidate checkpoint | `frp-v3-20261005/train/arm-M3/seed-12/checkpoints/apex_mark_u60000.pth`, sha256 `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723` |
| Incumbent checkpoint | `champion_a5_freespace_20260621.pth`, sha256 `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93` |
| Wrapper (both arms) | `free-space-veto/v8-space-and-head(lambda=8.0)`, v8 source sha `faf3695f…ac05` |
| Study root | `snake-dqn-artifacts/frp3-m3s12-strict-20261005/run-v1` |
| Intent | `run-v1/intent.json`, sha256 `5deef5a9d415ec135b4c16d59757b486e97b9b5af2068564cfe003a616520485`, created 2026-10-06T01:32:37Z |
| Code | branch `band-v2`, commit `865ee9e` (clean tree, `allow_dirty` false) |
| Receipt | `run-v1/output/receipt.json`, sha256 `5107f3fc540af41f2d81f907d2c9e40302ea4140141833144460f562e5acc5b3` |
| In-run audit | `run-v1/output/audit/audit.json`, sha256 `fa015c2f…2911`, PASS |
| Post-hoc audit | `run-v1/audit-posthoc/audit.json`, sha256 `84609292…421a`, status PASS (44 checks, 0 failures) |
| Closeout | `run-v1/output/closeout.json`, outcome `STRICT_PASS`, 2026-10-06T07:19:37Z |
| Platform | Mac, 2 shared CPU slots (`cpu-slot-1`, `cpu-slot-2`); RunPod not used (worst case exceeded the balance) |

## Primary endpoint: mass (H5000, paired, per mix)

Method `strict-sequential-obf-bonferroni-v1`; per-mix nominal p at look 2 = 0.00556.

| Mix | Mean delta (candidate - incumbent) | SD | t | p | Crossed at look |
|---|---|---|---|---|---|
| frozen | **+91.0** | 186.6 | 7.02 | 1.6e-11 | 1 |
| mixed | **+59.8** | 190.5 | 4.52 | 5.3e-06 | 1 |
| scripted | **+61.7** | 276.6 | 3.21 | 7.7e-04 | 2 |

Scripted non-inferiority (absolute NI margin 9.81): lower bound **+22.8** (passes).
Mass superiority was established at look 1 for frozen and mixed; the gate continued because
the survival band failed at look 1, and stopped at look 2 when all three mixes had crossed and
the band passed.

## Survival band v2 (`pooled_ni_continue`, option 1, RCI from the qualifying look)

Pooled margin 0.05, per-mix margin 0.075, per-mix floor 0.30, alpha 0.05, `rci_obf` bound.

| Look | n/mix | Pooled delta (LB) | frozen delta (LB) | mixed delta (LB) | scripted delta (LB) | Band |
|---|---|---|---|---|---|---|
| 0 | 69 | +0.020 (-0.075) | +0.033 (-0.100) | +0.031 (-0.110) | -0.005 (-0.190) | fail (not judged) |
| 1 | 138 | +0.006 (-0.037) | +0.031 (-0.034) | -0.001 (-0.062) | -0.011 (-0.102) | fail (scripted per-mix) |
| 2 | 207 | +0.016 (-0.011) | **+0.045** (+0.003) | **+0.006** (-0.035) | **-0.002** (-0.060) | **pass** |

Survival fractions at look 2 (candidate / incumbent): frozen 0.912 / 0.867, mixed 0.911 /
0.905, scripted 0.809 / 0.811. All mixes are far above the 0.30 floor.

## Disclosure: the old band would have failed this gate

Under the previously ratified band (`paired_ni_at_stop`: each mix's paired survival lower
bound must clear **-0.05**, judged once at the qualifying look, failure ends the gate), the
scripted lower bound at the stopping look, **-0.060**, is below -0.05, so the old rule would
have produced a fail, not a pass. The result above depends on survival band v2.

Band v2 (`docs/research/governance_amendment_survival_band_v2_2026-10-06.md`) was proposed,
reviewed, implemented and its operating characteristics simulated on independent pre-FRP-v3
pools, then **ratified (option 1, per-mix margin 0.075) in commit `865ee9e` at
2026-10-06T01:28:04Z, before the strict intent was written (01:32:37Z) and before any gate
world was played**. The amendment's ratification answers accept explicitly that a survival
loss confined to one mix is tolerated below the per-mix margin. The observed scripted
survival point estimate is essentially zero (-0.002); its bound is wide because scripted
survival is the noisiest mix (paired SD 0.41).

## What this does and does not claim

- Claims: the candidate + v8 beats the champion + v8 on H5000 mass in all three opponent mixes
  under the pre-registered sequential strict method, with pooled survival non-inferior at 0.05
  and per-mix survival non-inferior at 0.075.
- Does not claim: per-mix survival non-inferiority at 0.05 (scripted bound -0.060); any
  long-horizon or serving property; the result for any other FRP-v3 seed or update count.
- Next: Mac web serving qualification of the checkpoint swap (`frp3-serving` lane). Release
  only after `SERVING_PASS` with audit PASS.
