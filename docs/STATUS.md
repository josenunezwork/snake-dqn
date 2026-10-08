# Current state — 2026-10-08

Start: [documentation map](README.md) · [study index](research/INDEX.md) · [glossary](research/GLOSSARY.md).
This is a dated documentary snapshot, not a live job or deployment check. Newest Claude
memory/decision records take precedence over older launch notes; source pointers are below.

## Released agent

**frp3-s12 + v8 veto** (2026-10-06): `STRICT_PASS` under survival band v2, then Mac
`SERVING_PASS`; release recorded on local main, not pushed. Checkpoint: FRP-v3 M3 seed 12,
60,000 updates, SHA prefix `eec144bf`. [Release and rollback record](research/frp3_serving_result_and_release_2026-10-06.md).
`SNAKE_SERVE_CHECKPOINT=champion` restores champion+v8. With no checkpoint named,
`SNAKE_SERVE_VETO_VARIANT=v7` (also v5/v2) restores champion+that variant. Explicit
`SNAKE_SERVE_CHECKPOINT=frp3-s12` plus v7 serves frp3 **unwrapped**, not frp3+v7.

## Standing decisions (2026-10-08)

- **vector61 FROZEN:** no new governed FRP studies on the 61-D vector line. If the line is reopened, S2's 67-D sight inputs are the default.
- **Redesign is the sole main line.** The goal is to beat **frp3-s12+v8 WITHOUT a veto**, using GridBatchSim, ego2s-b and distilled DQN.
- **M3-C next:** code and pre-registration only until spend is approved. It must address Q drift (97→113–119), the peak at 10M followed by decline, and under-boosting, and it must include a no-veto arm. Size it at 10–15 seeds or multiple actors per learner, based on M3-B measurements. Target >70% utilization of the GPU-pod bottleneck, with incremental checkpoint pulls and live telemetry.
- **Process:** batch code changes before ratifying. Sequential Phase R is the default for new studies. **No new pod work until the runner is stable.** Existing fixed-N studies keep their registered rules.

## Operating rules (current)

These rules replace any older text that conflicts with them (for example the 5x rule,
"always serverless", or champion+v8 as the served agent).

- **Released agent:** frp3-s12 + v8. The rollbacks are listed under "Released agent" above.
- **RunPod per-step rule:** use RunPod for a step only if its projected wall-clock time is
  **≥4× faster than the same step on the Mac**. Base the projection on measured timings,
  including setup and transfers. The threshold was 5× until the user lowered it on 2026-10-05.
  Waivers below 4× are granted by the user case by case and are recorded per step. So far:
  - FRP-v5-H Phase R on pods at about 3.3–3.8× (2026-10-07);
  - FRP-v5-S2 Phase R on pods (2026-10-07).
  FRP-v3 Phase R at about 4.7× was granted as an exception under the old 5× bar. Under 4× it
  qualifies without a waiver. A waiver never carries over to another step.
- **Cheap pods first:** when RunPod is used, poll for cheap CPU pods first. Serverless is only
  a fallback when cheap pods stay unavailable and the job cannot wait. Its real cost has run
  at about 6–10× the catalog rate. The older "always serverless" default is superseded. The
  Mac is available whenever it is on AC.
- **Strict gates and serving qualification run on the Mac.** The
  [strict-on-RunPod amendment](research/governance_amendment_strict_on_runpod_2026-10-05.md)
  is **NOT RATIFIED and was superseded on 2026-10-08**; see its
  [status note](research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md).
  It was never used. Mac↔x86 bitwise identity fails at random on exact float32 ties. Any future
  strict-on-pods rule needs a new amendment with a tie-explaining identity check and a
  per-world single CPU model.

## Latest results

Deltas below are H5000 mass integral unless stated; comparisons differ by row. Screening,
learning-slope and distillation outcomes do not confer promotion.

| Date | Study | Recorded outcome | Result / comparison |
|---|---|---|---|
| 10-05 | FRP-v2 | `GO_L` | +41.3 (LB90 +26.3) vs champion+v8, 5/5 seeds; scripted survival −0.057 withheld `GO_H5000`. |
| 10-05–06 | FRP-v3 | `GO_R` → LH-1 `CLEAR` → `STRICT_PASS` → `SERVING_PASS` → released | Phase R +65.1 (HK LB90 +43.0) vs champion+v8, 5/5 seeds. |
| 10-07 | FRP-v4 | `PARTIAL` | +6.8 [LB90 −15.6, UB90 +29.1] vs frp3-s12+v8, 3/5 seeds; no replication. B4 platform check failed; not evidence. |
| 10-07 | FRP-v5-S | `FIDELITY_NOT_PASSED` | Hazard 0.1533/1k > 0.15; no training or Phase R; replaced by fresh-world S2. |
| 10-08 | FRP-v5-H | `KILL_H` | H5−control −41.5 (UB −24.8), 1/8 seeds positive; H5−incumbent −26.6; control +14.3. |
| 10-08 | FRP-v5-S2 | `GO_S_RECIPE` | +21.5 (LB90 +8.4) vs incumbent; S−control +8.3 (LB −9.6). Gate infeasible: N_max 5551 > 600; no promotion. |
| 10-08 | Redesign M1 | Done; G1 met | Grid/BatchSim 72/72 H5000; live/sim raster 45k frames (excluding unused scripted-snake hunger field); 21–28k hero steps/s/process with scripted opponents. |
| 10-08 | Redesign M2 | `FAIL` | No-veto student vs no-veto frp3-s12: −29.5 (LB −43.7; NI margin −15). |
| 10-08 | Redesign M2b | `PASS` | No-veto student vs no-veto frp3-s12: +52.4 (LB +34.8); both with v8: +11.1 (LB −16.4), descriptive. |
| 10-08 | Redesign M3-B | Learning-slope gate `PASS` | 5×20M RL; slope LB90 +0.46 > 0; ~5.9k transitions/s combined, 4.25× Mac; $3.92. |
| 10-08 | Redesign M3 Phase R | `NO_GO_EARLY` | Look 1, n=22/seed×mix; candidate+v8 vs frp3-s12+v8: +11.4 (SE 13.7), UB +40.5 < G=50; MI10 −18 (SE 37). Verify/audit PASS. |

## Open questions and compute

Can M3-C fix drift, late decline and boosting, and beat the served agent without a veto?
Can the runner provide reliable incremental pulls and a pre-registered, tie-explaining
Mac↔pod identity check? Exact float32 Q ties caused S2's 1/54 identity failure; this is
not permission to waive identity. M3-C remains the next decision, not a reported result here.

Mac: AC + open lid, thermal guard, 3×2-thread Tier-1 slots; strict gates 2 slots.
RunPod: see "Operating rules" above. That covers the ≥4× per-step rule, waivers, cheap pods
first with serverless as fallback, approved spend and uploads, caps, cleanup, and maximizing
useful utilization. Strict gates and serving qualification stay on the Mac. The strict-on-RunPod
amendment is not ratified and was superseded on 2026-10-08. See [compute policy](research/compute_policy_2026-10-02.md) (historical),
[sequential Phase R amendment](research/governance_amendment_sequential_phase_r_2026-10-07.md),
and [external policy sources / discrepancies](README.md).

Sources: Claude memory `standing-decisions-2026-10-08.md`, `redesign-scope.md`,
`frp3-s12-release.md`, `frp-v2-phase2-go-l.md`, `frp-v3-go-r.md`, `frp-v4-partial.md`,
`frp-v5h-kill.md`, `frp-v5s2-recipe.md`, `mac-pod-float-ties.md`, `runpod-5x-rule.md`,
`runpod-cheap-pods-first.md`. Campaign keys: `standing_decisions_2026-10-08`, plus the
`final_outcome` fields added on 2026-10-08 to `training_bet_frp_v4`, `training_bet_frp_v5h`,
`training_bet_frp_v5s`, `training_bet_frp_v5s2` and `governance_strict_on_runpod`. Locations
and branch documents are in the [map](README.md) and [index](research/INDEX.md). The S2
artifact namespace contains `20261009`. That suffix is a namespace label; the recorded
merge/result date is **2026-10-08**.
