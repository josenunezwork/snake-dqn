# T34 Round 0-D: dense-teacher premise test (pre-registration)

**DRAFT_PENDING_INDEPENDENT_REVIEW_AND_FABLE_APPROVAL. 2026-10-20 ~20:00 PT. Written before any Round 0 outcome existed.**
When this was committed, `t34-round0-study-01/analysis/` did not exist, and no Round 0 chunk, pair, or outcome
file had been opened. Round 0-D is **dispatched only on a Round 0 NO-GO** (branch B or C of the
[NO-GO branch plan](../../docs/research/specs/2026-10-20_t34_nogo_branch_plan.md) §3). A Round 0 GO shelves it
unrun. Approval of this document authorizes A5-D (excluded measurement), the operator's mechanical
ratification command, and nothing else. It never authorizes a strict pass, serving, or release.
After ratification these bytes are frozen. Any change needs a prospective sibling amendment written before
the affected data.

## Authority and question

Binding, in order: [SUCCESS_CONTRACT](../../docs/research/final_run_20261019/SUCCESS_CONTRACT.md),
[DESIGN_V2_SHADOW + Amendment 2](../../docs/research/final_run_20261019/DESIGN_V2_SHADOW.md),
[P7_STATISTICS](../../docs/research/final_run_20261019/P7_STATISTICS.md), the unsuperseded parts of
[FINAL_PLAN](../../docs/research/final_run_20261019/FINAL_PLAN.md), and the NO-GO branch plan §4 (`59e932c2`).
The Round 0 rules this study copies are those of
[PREREGISTRATION_ROUND0.md](../t34_round0/PREREGISTRATION_ROUND0.md), as superseded by
[PREREGISTRATION_ROUND0_FIXES1.md](../t34_round0/PREREGISTRATION_ROUND0_FIXES1.md),
[ERRATUM_FIXES1.md](../t34_round0/ERRATUM_FIXES1.md), and
[ERRATUM_INTEGRITY_SIZING.md](../t34_round0/ERRATUM_INTEGRITY_SIZING.md). All of these are bound read-only in
the Round 0-D source closure.

**Question.** Does the exact T3-dense acting rule, applied over released bare B8 s3 (SHA-256 `3e649783…4eb4f8`),
clear the Round 0 premise bar on fresh worlds across all three mixes?

This is Tier 1 development evidence, conditional on this one fixed checkpoint. The unit is the paired world
(P7 §1). A failure is reported as "this teacher failed its registered bar". If Round 0-D returns NO-GO,
search-teacher Claim A closes for this campaign (plan §4).

## Controller (frozen)

The T3-dense rule is exactly the one in `research/t3_rollout_labels_20261015/ceiling.py:103-140`, dense mode,
as implemented in `research/t34_round0d/controller.py`:

- **Trigger.** T3's own `DenseTrigger(min_length=600, interval=5)` class is imported unchanged. The clock starts
  at the first live decision with L ≥ 600. It fires every 5th episode-local decision after that and keeps
  advancing through later dips in length. There is no Q threshold, and at most one batch runs per 5 decisions.
- **Candidates.** Every advisory-legal action is a candidate, boosts included. If the advisory mask is empty,
  no batch runs and the arm uses bare B8's normal-only fallback.
- **Rollouts.** Each candidate is rolled forward D = min(300, 5000 − decision) frames. The continuation is B8
  greedy with ε = 0 and no hero veto, on a cloned world: cloned RNG, and real opponents carrying their v8 state.
- **Selection.** The arm takes the candidate with the most alive transitions. Ties go to B8's argmax, then the
  highest Q, then the lowest index. **All-die delays execute** (the T3 rule). Unit tests check this selection
  against T3's own expression.
- **Off-trigger decisions** are identical to the bare arm. No rewinds, no stochastic rollouts, no tower, no
  training.

**Engine substitution (deliberate).** T3's `batched_rollout` supports only stateless scripted anchors, so it
cannot run frozen or mixed rosters. Rollouts therefore use the audited T34 merged-batch continuation
`ExitTeacher._rollout_merged` (`research/t34_exit_b8/engine.py`), with its horizon set to D for each batch. The
world harness is Round 0's: `research.t34_exit_b8.trajectory.build_world`, prepared Watch boundary, six-snake
dispatcher.

**Pre-registration Mac check (development only, not evidence).** On T3 census worlds b8_s3_0, _1 and _6, this
harness reproduced T3's lengths, greedy actions and advisory sets at the first trigger decisions. It also
reproduced T3's per-action frames-survived exactly at the first batch (env 0) and at the first changed
decisions (env 1 at decision 2592: greedy 1 → chosen 4; env 6 at decision 3751: greedy 1 → chosen 0). Details
are in `VERIFICATION_ROUND0D.md`.

## Design (copied from Round 0)

Exactly **240 paired terminal-hero H5000 worlds**: 96 scripted, 72 frozen and 72 mixed. Each world uses slot 0,
six snakes and mechanics v1. Both arms run sequentially in the **same spawned single-thread worker, on the same
runtime**. The bare arm is Round 0's `_bare_trace`, imported unchanged. Production is Colab L4 with K = 12
single-thread CPU inference processes, W1, fp32. There is no GPU substitution.

**Bank.** `BANK_ROUND0D.json` is the materialized authority. The namespace is `t34-round0d-dense-20261021/v1`,
with a per-mix suffix. Seeds are the first four SHA-256 bytes of `{namespace}/{mix}/world/{index}`, big-endian.
Rows come from Round 0's own builder, `research.t34_round0.bank._rows`, imported unchanged, so rosters, the
FRP3-s12 / champion_a5 frozen-slot rotation, v8 λ = 8 opponents and profiles match Round 0. The execution order
is Round 0's 4:3:3 strata (`ordered_rows`). Excluded worlds use separate namespaces and never enter any
estimate, fit or bank:

- 2 A6 Mac fixtures (`t34-round0d-preflight-20261021/v1`, H500);
- 30 A5-D throughput worlds (`t34-round0d-a5d-20261021/v1`, 12/9/9).

**Seed audit.** `SEED_AUDIT_ROUND0D.json` covers all 272 seeds in one pass, rejecting the whole allocation on any
hit:

- explicit set-disjointness from the Round 0 bank (rows and fixtures), the Path B retention-v2 bank, the
  Phase 0 bank, and T3's 32 dense census seeds;
- a literal text scan (JSON/JSONL/MD/PY/YAML/SH/CSV/TSV, hidden and ignored files included) of docs/,
  research/, src/, scripts/ and the sibling artifact tree, which covers strict-gate, T3 and every historical
  bank in text;
- the 7 historical FRP training-exclusion inventories.

The scan's positive controls (one known seed from each explicit bank) were all found. The result is
**CLEAR, zero collisions**. Coverage limits: opaque binaries, unrecorded external runs, and the live Round 0
study root, which was excluded so that no Round 0 outcome byte was read. That root's identities are checked
through `BANK_ROUND0.json`. There is no replacement, reroll, or post-data namespace change.

## Endpoints, inference, bars, and one look (verbatim from Round 0)

These rules are executed by **importing Round 0's `research.t34_round0.analyze.summarize`**. A test asserts that
it is the same function object, so the clauses, thresholds and decision order cannot drift. The Round 0 text is
reproduced here unchanged:

> Primary efficacy: paired **full-horizon logical-mass integral**, dead frames zero; each arm's mass sum divided
> by 5000. Alive-frame fraction and no-death survival are distinct endpoints. Terminal death ends credit; the
> hero never earns respawn credit. Boost share is boosted action decisions / all delivered hero decisions,
> summed over all mixes, rather than the mean of per-world ratios.
>
> Independent unit = paired world. Means are equal-weight across mixes even though N differs. For paired deltas
> d, pooled G=(mean(d_scripted)+mean(d_frozen)+mean(d_mixed))/3; variance=sum(s_mix²/n_mix)/9.
> Welch-Satterthwaite df uses the three independent strata. LB90/UB90 are one-sided 90% t bounds (two-sided 80%
> interval endpoints). Per-mix bounds use paired t statistics with n_mix−1 df. Survival guard A uses one-sided
> 95% lower bounds, per the survival-band amendment. Guard B is the scripted no-death point gain. Exactly one
> active controller and one look: Holm over this singleton changes nothing, family alpha .10 for efficacy; the
> guard intersection has its own .05 bounds. No interim outcome peeks, optional stopping, extension, extra arm,
> replacement, or checkpoint search.
>
> | Clause | Required value |
> |---|---|
> | Teacher scripted no-death survival | ≥ 0.66 |
> | Paired scripted no-death gain vs bare B8 s3 | ≥ +0.10 |
> | Equal-mix pooled mass gain | ≥ +40 |
> | Efficacy LB90 | > 0 |
> | Delivered teacher boost share | [0.005, 0.10], inclusive |
> | Survival guard A pooled alive-frame delta LB95 | > −0.05 |
> | Survival guard A each mix alive-frame delta LB95 | > −0.075 |
> | Survival guard A absolute alive-frame fraction each mix | ≥ 0.30 |
> | Survival guard B scripted no-death point gain | ≥ +0.03 (subsumed by +0.10) |
>
> `STOP` if a complete-data point bar fails (scripted absolute/gain, pooled mass, boost interval, absolute alive
> fraction, guard B), or a survival guard's upper 95% bound is at/below its NI boundary. Otherwise a failed mass
> or survival lower bound with compatible point bars is `INCONCLUSIVE`. Exact floating-point thresholds are
> applied without display rounding. No second amendment to rescue a failed survival bar. STOP closes this
> teacher as tested; INCONCLUSIVE records insufficient evidence and authorizes no extension; GO buys only the
> next planning decision.

**Faults (Round 0 rule).** An unconditional controller fault yields `STOP`, and it takes precedence over
missing rows. The faults are: non-finite Q, an illegal action, or a > 90% one-action share in any rolling
100,000-decision window, including windows that cross chunk boundaries in canonical bank order. These faults
have dedicated exception types. Export and serialization errors are operational INCOMPLETE, not STOP.

**OC.** `OC_ROUND0.json` is reused unchanged (plan §4). It is not re-sized. The T3-like power is reported from
it, not recomputed.

**Descriptive decomposition (plan §3.1; reported in `decision.json` under `descriptive`, never a gate).** The
report covers:

- per-mix paired mass, alive-frame and no-death deltas;
- scripted no-death gain next to T3's +0.28 for B8 s3;
- interventions, batches, and rescue and delay changes per world;
- teacher-arm deaths split into those preceded by an executed horizon rescue and those with none;
- boost share per mix.

## Integrity (Round 0's, imported or ported unchanged in substance)

- **Ratification.** Ratification fixes one canonical absolute `study_root`
  (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0d-study-01`), an authoring-checkout ledger
  `research/t34_round0d/ledger/<study>.jsonl`, the source commit and closure, the exact chunk sizing, and the
  **48-hour dispatch deadline** from `approved_at` (Round 0's `dispatch_deadline`, imported). The canonical
  `REGISTRATION_ROUND0D.json` must be committed and pushed to main before the first dispatch.
- **Create-only history.** Every phase first publishes a create-only remote tag,
  `refs/tags/t34-round0d/<study>/<phase>`. This prefix never matches Round 0's `t34-round0/*` admission glob, and
  a test checks that against a real git remote. Fixed phases are `chunks/chunk-NNN/attempt-0[12]`, `replay`, and
  `analysis`. Terminal seals bind every retained file, and their hashes append to the ledger. Each append must be
  committed and pushed before the next dispatch, replay or analysis. History is verified against origin/main,
  including deletions.
- **Settlement.** The rules are Round 0's:
  - Operator abort, source drift, unexplained interruption, an unfinished durable claim, a missed dispatch
    deadline, or a PASS receipt over its timeout: **KILL**.
  - A supervisor wall-cap or unit-cap expiry: **INCONCLUSIVE**, with no extension.
  - An explained typed controller fault: **STOP**.
  - At most **one outcome-independent mechanical rerun** per chunk (attempt-02), only after a verified-release
    export or transfer failure, with an outcomes-unseen attestation. No automatic retry, no world replacement,
    no bank extension, no second look.
- **Sole look.** `analysis/decision.json` is the sole verdict. Replay and analysis refuse a pending in-time
  schedule before reserving a phase, and a reserved analysis that is interrupted seals KILL. Chunk outcome
  exports stay unopened until every chunk is terminal.
- **Two lanes (needed for the plan's wall time).** Round 0 dispatches serially. Round 0-D allows two concurrent
  lanes. Each attempt holds an exclusive liveness lock (`inflight.lock`) from reservation until its seal.
  History checks skip an attempt whose lock is held (in flight). An unsealed attempt whose lock is free is
  Round 0's unfinished claim, which is **KILL**. Replay and analysis refuse to reserve while any attempt is in
  flight.

## Two deliberate differences from Round 0 (frozen now, plan §4)

1. **Replay on a second L4 runtime, not the Mac** (`replay.py`, `replay.sh`). The sample rule is Round 0's
   `selected_rows`, imported: ceil(10%) per chunk, mix-stratified, in bank order, which for 240 worlds is
   24 worlds and 48 traces. The sample is chosen from chunk slices before any outcome is read. Inside the reserved
   `replay` phase, one supervised L4 job re-simulates both arms of every sampled world. It runs with K = 12, has
   a 60-minute timeout and a **2.0 u** cap. The job's runtime must differ from every producing runtime. Any
   hash or outcome divergence fails the replay, which makes the analysis INCONCLUSIVE. The Mac strict gate and
   serving qualification stay Mac-only.
2. **Teacher-arm labels are exported, sealed, and conditionally admissible.**
   - Each teacher world writes a public trace with its labels stripped; this trace is scored and replayed
     exactly like Round 0's. It also writes a second, sealed copy in the `research/t34_exit_b8/exports.py`
     schema. The sealed copy carries the per-action survive, alive_steps and mean_mass fields, a horizon field,
     the forbidden set, and Amendment 2 roles: `decisive` for horizon rescues, `delay` for executed all-die
     delays (`label_weight` 0.0), and `confirm` and `no_rescue` for KL-anchor and no-rescue rows.
   - Sealed files live under `exports/round0d/sealed/`. They are hash-bound in each chunk receipt and are
     never opened by replay, analysis or any operator before the verdict.
   - The only sanctioned reader is `labels.py`. It admits the 240 sealed worlds **only to dense R1 fitting, and
     only if the published sole decision is GO**. It refuses every screen, probe or gate purpose.

## A5-D and A6 (before ratification; excluded worlds only)

**A5-D** (`a5d.py`, `a5d.sh`) is one L4 lane with K = 12, a 90-minute timeout, an estimate of 45 minutes, and a
**3.0 u cap** (+20% held reserve). The plan said ≤ 60 minutes; see the reviewer questions below. The job has two
parts:

- **Fidelity.** The first 8 work items are the T3 dense census worlds (b8_s3, envs 0–7, in bank order),
  frozen in a Mac-prepared portable fixture `FIDELITY_T3.json` that is hash-bound as an upload input. Each world
  must reproduce exactly:
  - its historical per-world endpoints: survival, death decision, alive fraction, mass integral, decisions;
  - the exact batch schedule;
  - every changed action (decision, greedy, chosen).

  Any difference is **HOLD**. Sizing refuses until the operator explains each difference with the revalidated
  M2 queries. There is no automatic pass, and an incomplete fidelity world is also HOLD.
- **Throughput.** Dense worlds completed in the 1,200 s steady window after a 1,200 s cold fill, multiplied by
  three, give the lane rate r. **KILL if r < 8 worlds/h/lane**: report the result and re-plan on Mac capacity
  after Path B. The benchmark also records the measured export bytes per pair, which must stay ≤ 9 MiB.

**A6** is the excluded H500 Mac preflight of the real entrypoint. It runs one thread, nice 10, while Phase 0
runs. It must be source-current at ratification. Its bare and metric costs are scaled 10× (H500 → H5000) and by
3× for the L4 CPU, divided by K = 12, exactly as in Round 0.

## Sizing and caps

**Chunk sizing** uses Round 0's formula, imported unchanged (`research.t34_round0.make_manifest.size_round`):

- n is the largest multiple of 10 such that setup + n·(3600/r + 3·(bare + metric)/12)/60 ≤ 180 minutes, and
  the sized exports stay < 2 GiB at 9 MiB × 1.25 per pair;
- the timeout is at least max(estimate + 25, 2 × estimate);
- the price is 2.0 u/h with a 20% reserve held;
- the setup allowance is ≥ the measured A5-D setup, export and cleanup time.

**Spend ceilings.** These correct the plan's arithmetic; see the reviewer questions.

- **Ratification gate.** The forecast expected spend must be ≤ **40 u**, the plan's cap. Otherwise re-plan.
- **Study ceiling.** No chunk is admitted if (charged units of all prior attempts + this chunk's expected
  units) > **48 u**, which is 40 u plus the plan's held 20% reserve.
  - A settled attempt is charged its whole-session elapsed time at 2.0 u/h.
  - An in-flight attempt, or a killed attempt without a receipt, is charged its full unit cap.
  - A refusal publishes no claim. If settled charges alone leave the schedule unable to finish, the study
    settles INCONCLUSIVE under the first-cap-wins rule.
- **Lanes.** Two lanes; the third lane stays free for the unpaced Path B arm. Other account assignments count
  against the cap of three.

**ESTIMATED forecast** (not dispatchable; uses Round 0's measured A6 bare/metric costs as placeholders and 15 min
setup per chunk):

| r (worlds/h/lane) | chunks × pairs | expected u | lane-h | wall-h on 2 lanes | sum of priced chunk caps (+reserve) |
|---|---|---|---|---|---|
| 15 | 8 × 30 | 39.3 | 19.7 | ~9.8 | 78.9 (+15.8) |
| 19 | 6 × 40 | 31.6 | 15.8 | ~7.9 | 63.2 (+12.6) |
| 23 | 5 × 50/40 | 26.7 | 13.4 | ~6.7 | 53.7 (+10.7) |

The replay adds ≤ 2 u and A5-D adds ≤ 3 u. At r < ~15 the 40 u ratification gate refuses, and the plan is
redone.

## Conditional dense R1 sizing (only on a published Round 0-D GO)

- **Worlds.** Dense R1 uses the 240 sealed teacher-arm worlds plus **560 fresh dense worlds**, 800 in total.
- **Namespace.** The fresh worlds use `t34-round0d-dense-r1-20261021/v1` with 224/168/168 by mix. They are
  allocated and audited with this document's rule (whole-allocation reject) before R1 dispatch, and are never
  admitted to any screen, probe or gate bank.
- **Runs.** The teacher arm only, with the same controller, engine, layout and chunk formula. Three lanes.
- **Fit.** Uses FINAL_PLAN §3/§8 rules and P7 unchanged: decisive rows weighted per FINAL_PLAN §3, delay rows
  weighted ~0 (Amendment 2), the Claim A pipeline, the §8 anchor (τ = 0.05, margin 0.1), and its tripwires
  (all-length disagreement ≤ 5%, boost ≤ 1.5×). R1 is non-terminal.
- **Sizing formula** (`sizing.dense_r1_plan`): 560/r lane-hours, divided by 3 for wall time, at 2.0 u/h.
- **Plan cap ≤ 30 u.** At r = 15–23 the formula gives **24–37 lane-h, 8–12 h wall, 49–75 u**. The plan's
  "15–25 u" appears to omit the factor of three lanes, so the plan's 30 u cap is **not met at any plausible r**.
  As registered, R1 is **fundable only if its expected units are ≤ 30 u**. A larger R1 needs a new Fable
  approval. This sizing removes no rule.
- **R1 entrypoint.** The labels-only entrypoint is not yet built. It is a follow-up build item reviewed under
  this registration.

## Restrictions

Preparation, tests and A6 run on the Mac with one thread under nice 10, with pytest using at most 2 workers,
while Phase 0 runs. This task performed no Colab, RunPod or tmux operation and ratified nothing. Path B is
never touched. It is never aborted, because an operator abort is KILL there. The commands are in
[README.md](README.md).
