# Current state — snapshot updated 2026-10-27; release updated 2026-10-17

Start: [documentation map](README.md) · [study index](research/INDEX.md) · [glossary](research/GLOSSARY.md).
This is a dated documentary snapshot, not a live job or deployment check. The owner-ratified campaign
documents named below take precedence over older launch notes and over memory; this file is updated
when they change.

## Campaign authority (owner-ratified 2026-10-19) — overrides older text

Active campaign: **T34 ExIt-B8**. Binding documents on main, in order:
[SUCCESS_CONTRACT.md](research/final_run_20261019/SUCCESS_CONTRACT.md) ·
[DESIGN_V2_SHADOW.md](research/final_run_20261019/DESIGN_V2_SHADOW.md) (+ Amendment 1) ·
[P7_STATISTICS.md](research/final_run_20261019/P7_STATISTICS.md) · unsuperseded parts of
[FINAL_PLAN.md](research/final_run_20261019/FINAL_PLAN.md).
Superseded by them: the "single bare network only" goal (now **Claim A**; **Claim B** = a visibly labelled
assisted hero may also succeed; bare wins unresolved ties); "beat frp3-s12+v8 without a veto" (met by B8 s3,
closed); "M3-C next"; five-training-seed inference for fixed controllers (unit = paired world); historical
FRP3/champion "Tier 1 + Tier 2" guards as automatic T34 obligations (comparator is B8 s3 for every bar);
RunPod for T34 (none); the Q0/Q1 budget envelope. Watch product metrics are binding per contract §3.
Kept: Mac-only strict gate and serving qualification, fresh paired banks, survival band v2, pre-data freezing.

**Path B v2 retention (re-ratified 2026-10-20 ~20:05 PT):** the registration ratified at 97232d51 (study `t34-retention-v2-operator-01`) is SUPERSEDED pre-data by the bare-unpaced amendment (`docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md`) and is VOID for execution. Execute only from the frozen worktree `/Users/josenunez/Projects/ml/snake-dqn-retention-v2-frozen-02` (detached at `81c54731a06f5a630bc1b95edca876832906a812`), study `t34-retention-v2-operator-02`, with the current `research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json` (authoring checkout / `history_ledger.repo`: `/Users/josenunez/Projects/ml/snake-dqn-retention-v2-author`; ledger `research/t34_pathb/ledger/t34-retention-v2-operator-02.jsonl`). Operator-01 is VOID. The Colab unpaced arm is dispatchable only at `81c54731`. Path B now uses the relative boost floor (0.5 × paired bare share ≤ paced ≤ 0.10); Round 0 has its own pre-look boost-floor amendment (merge 30b2739a), outside this registration. Timing gate only after Phase 0 finishes and the Round 0 Mac replay.

**Colab credits (owner, 2026-10-20):** use the owner's Colab Pro+ credits whenever they speed up our work or add a parallel research lane — do not wait for the Mac or ask first. Go through the qualified CLI supervisor (`research/colab_cli`, QUALIFICATION.md), L4 by default, price per useful result, stay within each study's registered caps. Mac-only work stays on the Mac: Path B timing/paced tests, the strict gate, serving qualification. **Goal (owner, 2026-10-20): spend all ~1,160 Colab units by ~2026-11-20 (~39 u/day) on lanes that produce decisions** — see the compute portfolio in docs/research/final_run_20261019/CAMPAIGN_LOG.md. **Units never block (owner, 2026-10-24, STANDING):** Colab units are never by themselves a reason to decline, shrink or delay work that could speed up the campaign. Only science/pre-registration, integrity, safety and the owner's release rules may block; budgets/caps are generous runaway guards (timeouts, per-job caps), not rationing — no judge rulings over unit shortfalls, the operator raises the guard and logs it. Caps written into a pre-registration before data still need a pre-data judge amendment. **No Colab budgets (owner, 2026-10-24 later, STANDING):** 'remove budget — fine using all Colab credits in 1 week if even possible.' There are no unit budgets or ceilings; per-job timeouts remain as runaway guards; pre-registered science caps remain unless a pre-data judge amends them; actively seek high-value uses for the credits. **Approvals (owner, 2026-10-20):** all Colab use is pre-approved; any other decision needing approval (ratifying a registration, funding gates) goes to one independent Fable judge instead of the owner (**2026-10-20 ~19:05 PT, owner:** while Fable is at its usage limit, independent Opus stand-ins serve as panel members and approval judges until Fable returns), is recorded in CAMPAIGN_LOG.md, and is reported to the owner. **Release/serving changes still need the owner's OK** (SUCCESS_CONTRACT §4; corrected 2026-10-20 per RULING_label_ordering.md §4 — a Fable judge may only recommend a release).


## Development snapshot — 2026-10-27 PT

Snapshot at 23:56 PT, main `aa5d6dfc`. Operational progress below is development evidence;
no new strict pass, serving qualification or release is established. Artifact paths are
relative to `/Users/josenunez/Projects/ml/snake-dqn-artifacts/`.

**Direction:** F5 closed before any fit by owner decision (`81863023`; campaign-log
closure row dated 2026-10-26). Work redirected to GPU-resident training and Mac served
search. The A-3 throughput result is already recorded in the campaign log (`88ba3f56`);
it measures collection plus PPO throughput, not skill or live-game identity.

**GPU training — TRAIN-01 packet 15 is running; packet 16 is being built.** Packet 15
runtime/seal/verification are `5c0a101a` / `a51380c1` / `6e54ed4c`.
`ops/logs/sandbox-a-train01-20261027-15.log` shows ongoing running/nonterminal telemetry,
finite learner completion through update 16 and the diagnostic-update-16 proxy in progress.
The terminal `sandbox-a-resident/train01-attempt-15/output/receipt.json` is absent at this
snapshot. Packet 16's build state is operator-reported; acceptance or dispatch is not
established. These are simulator-D development runs, with no completed 384-update endpoint,
B8 efficacy claim or gate result. The prospective recipe/diagnostic ruling is
`t34-sol/astra-rule-argon-1027/RULING.md`; update 16 does not expand the nomination set.

Earlier TRAIN-01 receipts retain their original dispositions: 05/07/10 INCOMPLETE after
supervisor interruption; 08/09 INCOMPLETE / RPC_EXEC_POLL_TIMEOUT; 11 INCOMPLETE /
CHILD_NONZERO_EXIT after a non-finite PPO actor loss; 13 INCOMPLETE / UPLOAD_CAP before
resume staging completed. Each extant receipt in
`sandbox-a-resident/train01-attempt-{05,07,08,09,10,11,13}/output/receipt.json` records
verified stop and `cleanup_error=null`. Packet numbers do not imply executed runs: receipts
for 06/12/14 are absent at the requested paths. Packet 11 preserved iteration-0000 full
state; compact resume in 14/15 passed independent integrity and CPU smoke checks.
Astra's `t34-sol/astra-review-train01-14/REVIEW.md` rejected a MAJOR finite-input KL
overflow; `astra-review-train01-15/REVIEW.md` is ACCEPT-WITH-FIXES after its repair,
with a remaining MINOR saturated KL/entropy value/helper-gradient exactness issue.
The review does not establish material learning harm or a completed CUDA continuation.

**Mac served search — the fresh scored served panel has not yet run.** Historical host
attempts remain INVALID under `served-tick-20261026/`; desktop-13 refused its observation
binding before panel execution. The supervision and desktop rulings at
`t34-sol/astra-guard-supervisor/RULING.md` and `astra-desktop-apps/RULING.md` are prospective
and do not salvage earlier timing cells. Desktop fixes `a3799098`, merge `2c748fcb` and
protected-service precedence `6fa8e658` have ACCEPT reviews in
`t34-sol/astra-review-argon-desktop4/REVIEW.md`, `astra-review-desktop-merge/REVIEW.md`
and `astra-review-desktop-live2/REVIEW.md`. Their short integration/replay evidence is
not scored timing or concurrent-supervisor qualification for the current TRAIN-01 job.

The opt-in authoritative game/DTO split (`8ffc34d0`, `59217589`) and W1/K1 contention arm
(`892bcd1e`, `709ccdf3`) have accepted source repairs: Argon's
`big-lever/exp/argon-20261026/review-astra-served-split-fix/answer.md`, Astra's
`t34-sol/astra-review-argon-w1b/REVIEW.md` and `astra-review-w1-merge/REVIEW.md`.
No measured timing-panel verdict, client-cadence qualification or delivered-value result
is established. The documented release remains **bare B8 s3**; releases still need the
owner's approval.

## Released agent

**b8-s3 hero without a veto** (2026-10-17), checkpoint SHA prefix `3e649783`.
Watch now defaults to **mixed**: the bare B8 hero at slot 0 faces **frp3-s12 + v8
opponents** at slots 1 through N−1. The owner approved this roster switch after
T15 `SERVING_PASS`. [Mixed Watch release erratum](research/b8_s3_release_erratum_mixed_watch_2026-10-17.md).
Rollback: set `SNAKE_SERVE_WATCH_ROSTER=all_b8` and restart; unsetting it selects mixed.
T7 fresh-bank Phase R `STRICT_PASS` and Tier 2 PASS, followed by all eight T12 serving
checks (check 4 cleared by T12B). Claim: **more mass at equal survival against the served
agent, with no veto**, not safer. [Release, startup verification and rollback record](research/b8_s3_release_2026-10-17.md).

The default is pin-gated: B8 refusal falls back to verified frp3-s12 + v8, then champion,
with a reason. Restart with `SNAKE_SERVE_CHECKPOINT=frp3-s12` or `=champion` to roll back
with the remaining serving environment at defaults. With no checkpoint named,
`SNAKE_SERVE_VETO_VARIANT=v7` (also v5/v2) retains champion + that variant. Explicit
frp3-s12 with v7 remains unwrapped.

B8 was trained with v8 labels and is served without a veto. Tier 2 margin was 0.034 at
SE 0.023; self-play rosters are ungated; pre/post-release leaderboard rows are not
comparable. Existing ego2s ceilings are 24 fps through six driven AI and 12 fps through
twelve. Train is refused and unqualified. Monitor for at least seven days and 100,000
Watch hero decisions; roll back on fallback/ego2s exceptions, roster-specific hero
deaths/1k >0.544 at mixed Watch6; all-B8 thresholds remain >0.848 at Watch6,
>0.992 at Watch8, and >1.536 at Watch12; or matched
default-speed p95 >1.5× qualification. See the
[monitoring erratum](research/b8_s3_monitoring_erratum_2026-10-17.md); Watch8's death
baseline does not qualify its all-B8 latency. Mixed Watch6/8/12 have T15 latency
baselines, but mixed Watch8/12 have no registered death baseline. Observation windows
are separate by roster mode and size. Owner check-in: 2026-10-24.

T13b's fixed six-snake rule selected the mixed roster: the death gap 0.096/1k
is within 0.10, but the p95-length gap 127 cells exceeds 100. T15 subsequently
qualified the mixed serving path, and the owner approved the default switch in T15B.
See the [T13b result](research/t13b_six_snake_roster_2026-10-17_results.md),
[T15 qualification](research/t15_mixed_roster_qualification_2026-10-17.md), and
[T15B release erratum](research/b8_s3_release_erratum_mixed_watch_2026-10-17.md).

## Standing decisions (current, updated 2026-10-17)

- **Campaign goal met and released.** The October redesign goal was to beat **frp3-s12+v8 WITHOUT a veto**, using GridBatchSim, ego2s-b and distilled DQN; B8 seed 3 met it and is the served default. Its claim is more mass at equal survival against the served agent, with no veto. The release and rollback boundary are above.
- **vector61 FROZEN:** no new governed FRP studies on the 61-D vector line. If the line is reopened, S2's 67-D sight inputs are the default.
- **Late-death bare-network arms are closed.** Reopen only with new evidence of a different mechanism or a new observation/architecture. The [campaign retrospective](research/redesign_campaign_retrospective_2026-10-17.md) links the closed-arm and diagnostic record.
- **M3-C next is superseded.** The pre-release 2026-10-08 plan to address Q drift, late decline, and under-boosting is preserved in the historical record; the B8 release and late-death closure do not authorize it as current work.
- **Process:** batch code changes before ratifying. Sequential Phase R is the default for new studies. **No new pod work until the runner is stable.** Existing fixed-N studies keep their registered rules.

## Operating rules (current)

These rules replace any older text that conflicts with them (for example the 5x rule,
"always serverless", or champion+v8 as the served agent).

- **Released agent:** b8-s3 without a veto (2026-10-17). Rollbacks and monitoring are above.
- **Implementers (owner, 2026-10-27):** project code is implemented and reviewed only by
  Argon (Gemini 4 Argon via Antigravity CLI, own worktree/local commits) and Astra (Codex).
  The reviewer is always the other one; authors never review their own builds. Do not
  use the Sol implementer persona. Source: `AGENTS.md`, commit `1ad5dc95`.
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

## Historical pre-release questions and compute

Before the B8 release, the recorded questions asked whether M3-C could fix drift,
late decline, and boosting, and whether the runner could provide reliable incremental
pulls and a pre-registered, tie-explaining Mac↔pod identity check. They are
historical questions, not a current authorization. Exact float32 Q ties caused S2's
1/54 identity failure; this is not permission to waive identity.

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

## B8 serving qualification HOLD (2026-10-17)

T12 completed with `SERVING_FAIL`: checks 1–3 and 5–8 passed, while check 4 failed its
paced Watch12 high-speed retry. This is not a release: `CHECKPOINT_RELEASED_DEFAULT` remains
`frp3-s12`, and T12 selected B8 only with `SNAKE_SERVE_CHECKPOINT=b8-s3`. No default flip or
extra timing attempt was made. See the [T12 B8 serving qualification](research/t12_b8_serving_qualification_2026-10-17.md)
for the audit-bound evidence and the required fresh-qualification boundary for any repair.

### T12B check-4 repair passed (2026-10-17)

The authorized check-4-only rerun passed its pinned sibling amendment: Watch6/Play6
are server-clamped to 24 fps, Watch12/Play12 to 12 fps, and all default/ceiling paced
probes had zero overruns. Checks 1–3 and 5–8 carry forward from T12; its original failed
attempt remains preserved. The repair changes ego2s serving speed controls only, with
policy math and forward paths unchanged. **This is not a release:**
`CHECKPOINT_RELEASED_DEFAULT` remains `frp3-s12`. See the
[T12B result and evidence](research/t12_b8_serving_qualification_2026-10-17.md#t12b-check-4-result-and-evidence).

## T12C release (2026-10-17)

B8 seed 3 is now the released default. This supersedes the historical T12/T12B HOLD and
not-yet-released notes above, without changing their evidence. The default and both
explicit rollback configurations were started live, their stderr identity/veto lines
recorded, and every server started for verification stopped. See the
[release record](research/b8_s3_release_2026-10-17.md) for exact lines, validation,
monitoring command and disclosures. Source changes are published to `main` and
`t12-20261017`; no persistent server or monitoring job is left running by this task.
