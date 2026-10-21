# T34 Round 0-D: dense-teacher premise test (pre-registration, RE-SCOPED)

**DRAFT_PENDING_INDEPENDENT_REVIEW_AND_JUDGE_APPROVAL. First committed 2026-10-20 18:50 PT (`8729a1db`); re-scoped 2026-10-20 ~21:30 PT and revised after the FIX-FIRST review (`REVIEW_RESCOPE.md`) the same night (branch `t34-round0d-rescope-20261021`). Both versions were written before any Round 0 outcome existed.**
At each commit `t34-round0-study-01/analysis/` did not exist, and no Round 0 chunk, pair or outcome file had
been opened. The re-scope implements the Opus stand-in panels (`snake-dqn-artifacts/fable-panel-round0d/`
integrity, science and ops), JUDGMENT §5 action 3 and §7e/§7f (`docs/research/advice/2026-10-20_colab_lanes/`),
and RULING_round0_go_file.md (same folder).

## Approval and activation (two stages)

1. **Approval, before the Round 0 sole look.** An independent judge approves these exact bytes: rules, code,
   and the import pins in `pins.py`. This makes the branch choice mechanical (JUDGMENT §3). Approval authorizes
   only A6 and A5-D, plus the mechanical activation below. A6 and A5-D use excluded worlds only and **may run
   before the Round 0 verdict**.
2. **Activation (`registration.py ratify`), mechanical, no second judgment.** It runs only when all of the
   following hold:
   - the **governing** Round 0 verdict is published: `<round0 root>/amendment-boost-floor/decision.json` with
     `governing: true`, a sealed COMPLETE `phase_terminal.json`, and its ledger line on origin/main;
   - that verdict routes to Round 0-D under RULING §3, which means a **scientific** STOP or INCONCLUSIVE;
   - a source-current A6 exists;
   - an A5-D receipt exists, and the frozen rate-to-action table (below) gives RATIFY for it.

   Activation records the decision.json SHA-256, the phase_terminal SHA-256, and the amendment-ledger commit.
   It reads only the verdict code fields. It freezes sizing and starts the 48 h dispatch clock. Every Mac-side
   consumer re-hashes the bound verdict.

   Any other governing outcome shelves Round 0-D: GO, a fault STOP, an integrity INCONCLUSIVE, KILL, or a
   verdict that has not been published. One fallback exists, the registered-file route under RULING §2.2. It
   requires the preserved `decide` refusal (`--fallback-refusal`).

Activation never authorizes a strict pass, serving, or release. Once activated, these bytes are frozen. Any
change needs a prospective sibling amendment.

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
(P7 §1). A failure is reported as "this teacher failed its registered bar". The readings of every
outcome are frozen in the Round 0 × Round 0-D table below.

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

## Endpoints, inference, bars, and one look (verbatim from Round 0, boost row amended)

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


**Boost clause: the owner-approved relative floor replaces the registered row above.**

The rule lives in `research/t34_round0_amend/rule.py`, which is imported and pinned. Its sources are
`AMENDMENT_BOOST_FLOOR.json` and `docs/research/specs/2026-10-20_t34_round0_boost_floor_amendment.md`:

- **0.5 × paired bare pooled share ≤ teacher pooled share ≤ 0.10**, both bounds inclusive;
- the floor is compared exactly in integers, and a bare share of 0 gives a floor of 0;
- `analyze.py` calls `rule.amend` immediately after the imported `summarize`;
- only the boost clause is re-evaluated. Every other clause and threshold, fault precedence, and Round 0's
  decision order are unchanged. Tests use B8-like boost rates of about 0.1%.

Reason: bare B8 s3 delivers about 0.12% boosts (`boost-bar-check-20261020/REPORT.md`). Under the absolute 0.5%
floor, Round 0-D would have been a mechanical STOP whatever the premise showed.

The registered Round 0 clause verdict is kept in `decision.json` as `registered_round0_clause_decision`, for the
record only.

**Faults (Round 0 rule).** An unconditional controller fault yields `STOP`, and it takes precedence over missing
rows. The faults are:

- non-finite Q;
- an illegal action;
- a > 90% one-action share in any rolling 100,000-decision window, including windows that cross chunk boundaries
  in canonical bank order.

These faults have dedicated exception types. Export and serialization errors are operational INCOMPLETE, not
STOP.

**Power (pre-data).** This replaces the earlier sentence about OC, which `OC_ROUND0.json` cannot support.
OC_ROUND0 covers only the mass and survival bars and is reused unchanged for those. The scripted bars were
checked by the science panel's pre-data binomial simulation (20k reps; bare scripted no-death 0.53 from the T3
census; n = 96):

| True gain | P(teacher ≥ 0.66 and gain ≥ +0.10) |
|---|---|
| +0.28 (T3) | 1.00 |
| +0.20 | 0.93 |
| +0.14 | 0.53 |
| +0.10 | 0.16 |

- Dense mass deltas are zero-inflated, so the OC's FRP-pool SD is conservative for the mass LB90.
- If only the scripted mix gains about T3's +123, pooled mass is about +41, right at the +40 point bar. A mass
  STOP with a strong scripted gain therefore reads "does not generalize across mixes", not "T3 was wrong".

**Descriptive decomposition (plan §3.1).** Reported in `decision.json` under `descriptive`; never a gate:

- per-mix paired mass, alive-frame and no-death deltas;
- scripted no-death gain, next to T3's +0.28 for B8 s3;
- interventions, batches, and rescue and delay changes per world;
- teacher-arm deaths, split into those preceded by an executed horizon rescue and those with none;
- boost share per mix.

Round 0 and Round 0-D use **different banks**. Any cross-study contrast is unpaired and carries bank-sampling
noise on top of the rule difference.

## Round 0 × Round 0-D outcome table (frozen pre-data)

Round 0-D runs only after a governing scientific Round 0 NO-GO. Its sole decision then reads as follows.

| Round 0-D sole decision | Reading | Consequence |
|---|---|---|
| GO | **Rule loss (ii):** the dense rule works on fresh 3-mix worlds where teacher_v2 did not | Sealed labels become admissible to dense R1 *fitting*. Dense R1 runs only under the yield rule below. It fits only after the ratified Claim A loss amendment and the PB-disposition ordering clause |
| STOP on scripted or mass point bars | **Premise shrinkage (i):** T3's effect does not survive fresh worlds under either rule. A mass-only STOP with scripted gain ≥ +0.10 reads "no generalization across mixes" | Search-teacher Claim A family closes (plan §4; JUDGMENT L7) |
| STOP on the survival guards (absolute alive fraction < 0.30 in a mix, guard B < +0.03, or a guard's upper bound at or below its NI boundary), with no scripted/mass point-bar failure | **The teacher harms survival** | Family closes |
| STOP on the relative boost clause only | Teacher boost use falls below half of bare's, or exceeds 10% | Recorded as this teacher failing its registered bar; family closes |
| STOP by unconditional fault | Controller bug, not science | Fix and re-register on a fresh namespace (Round 0's rule), with its own approval. A bug does not close the family |
| Scientific INCONCLUSIVE | Unresolved; authorizes no extension | Family closes for this campaign (JUDGMENT: "scientific non-GO → close") |
| Integrity INCONCLUSIVE (cap, ceiling, divergence, deadline) or KILL | Mechanical; not an answer about the premise | No extension. A fresh-namespace re-registration needs new judge approval and reserve units (JUDGMENT reserve). Without that, the family closes as "not tested to completion" |
| Not activated (A5-D KILL or HOLD, or table KILL) | Premise not tested | Reported as a capacity or fidelity failure, never as evidence about the premise |

**Precedence for a STOP that fails several clauses.** The first matching reading applies, in this order:

1. unconditional fault, which is the bug branch;
2. any scripted or mass point-bar failure, which is reading (i): scripted no-death ≥ 0.66 or its gain
   ≥ +0.10, or pooled mass ≥ +40;
3. survival-guard failure, which reads as survival harm;
4. boost-clause-only failure.

The reading never changes the decision code. It only selects the row's wording, and it is computed
mechanically from the `clauses` and survival bounds in `decision.json`.

## Integrity (Round 0's, imported or ported unchanged in substance)

**What activation fixes:**

- one canonical absolute `study_root`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0d-study-01`;
- an authoring-checkout ledger at `research/t34_round0d/ledger/<study>.jsonl`;
- the source commit and closure, and the **import pins**;
- the bound governing Round 0 verdict;
- the exact chunk sizing;
- the **48-hour dispatch deadline**, counted from `approved_at` (Round 0's `dispatch_deadline`, imported).

The canonical `REGISTRATION_ROUND0D.json` must be on main before the first dispatch.

**Import pins.** `pins.py` records the SHA-256 of:

- every imported Round 0 module and rule document;
- the boost-floor `rule.py` and `AMENDMENT_BOOST_FLOOR.json`;
- the T34 exit_b8 and Path B helpers;
- T3's trigger;
- `eval_stats.py`.

`check_pins` refuses any drift in the tests, at activation, and inside every `require_registration` call. A later
Round 0 erratum is therefore never inherited silently.

**Create-only history.**

- Every phase first publishes a create-only remote tag, `refs/tags/t34-round0d/<study>/<phase>`. This prefix
  never matches `t34-round0/*` or `t34-round0-amend/*`; a test confirms this against a real git remote.
- The fixed phases are `chunks/chunk-NNN/attempt-0[12]`, `ceiling`, `replay` and `analysis`.
- Seals bind every retained file, and each seal's hash is appended to the ledger. Every append is committed and
  pushed before the next admission.

**Settlement (Round 0's rules).**

- **KILL:** operator abort, source drift, an unexplained interruption, an unfinished durable claim, a missed
  dispatch deadline, or a PASS receipt that ran past its timeout.
- **INCONCLUSIVE, no extension:** a supervisor wall-cap or unit-cap expiry.
- **STOP:** an explained, typed controller fault.
- At most one outcome-independent mechanical rerun per chunk.

**Study ceiling.** The operative ceiling is exactly **120 u**, the JUDGMENT lane L7 study total. The
forecast-derived bound must be ≤ 120 at activation (see Sizing), so whole-wave chunks and slow worlds do not
strand the last chunk. A5-D's 10 u is pre-study qualification and is reported separately. Before each chunk is
admitted:

- settled attempts are charged their recorded elapsed time at 2.0 u/h;
- in-flight attempts, and attempts without a receipt, are charged their full cap.

If the next chunk would cross the ceiling only because an attempt is in flight, the dispatcher refuses without
making a claim; the operator retries once that attempt settles. If **settled charges alone** would cross it, the
dispatcher claims the create-only `ceiling` phase and seals it **INCONCLUSIVE** (first cap wins). Analysis then
settles INCONCLUSIVE through the same path as a cap expiry. This is enforced in code and covered by a test.

**Two lanes.**

- Each attempt holds an exclusive liveness lock (`inflight.lock`) from reservation until its seal.
- History checks probe that lock with a *shared* non-blocking lock, so concurrent probes cannot make a dead
  attempt look alive.
- Admission is **refused while 2 attempts are in flight**. This keeps the third account assignment free for the
  Path B unpaced arm.
- An unsealed attempt whose lock is free is Round 0's unfinished claim, and it is KILL.
- Replay and analysis refuse to reserve while any attempt is in flight.
- A hung dispatcher holds its lock indefinitely, and killing it as the operator is KILL.
- Both dispatchers run in tmux under `caffeinate`, on AC power.
- A Mac crash, or a full disk during a seal, is KILL. The data volume is about 96% full, so check free space
  before each dispatch.

**Sole look.** `analysis/decision.json` is the sole verdict. Chunk outcome exports stay unopened until every chunk
is terminal.

**Sealed labels.** Confidentiality is procedural, as with Round 0's "exports unopened" rule. The labels are
hash-bound three ways, and `labels.py` is the only sanctioned reader. A test asserts that no other module reads
`round0d` `sealed/` paths.

## Deliberate differences from Round 0 (frozen)

1. **The replay runs on a second L4 runtime, not on the Mac** (`replay.py`, `replay.sh`).
   - The sample rule is Round 0's `selected_rows`, imported: ceil(10%) per chunk, mix-stratified, in bank order.
     With 12-pair chunks this is 2 pairs per chunk: 40 worlds and 80 traces when chunks are 12 pairs, and
     fewer when chunks are larger.
   - As in Round 0, the replay **re-scores the exported action traces**: hashes, metrics and the final hash, for
     both arms. It does not re-run the dense search.
   - It is sized from A5-D's measured L4 metric cost (`sizing.replay_sizing`): timeout ≥ max(estimate + 25,
     2 × estimate, 60 min), priced at 2.0 u/h.
   - The replay runtime must differ from every producer runtime. Any divergence makes the analysis INCONCLUSIVE.
2. **Teacher-arm labels are exported, sealed, and admissible only under conditions.**
   - Each world writes a public, label-free trace, which is the one scored and replayed.
   - It also writes a sealed copy in the `exports.py` schema: per-action survive, alive_steps and mean_mass;
     horizon; the forbidden set; and the Amendment 2 roles (`decisive`, `delay` with weight 0, `confirm`,
     `no_rescue`).
   - The sealed copy is admissible only to dense R1 **fitting**, and only after a published GO.
   - Delay rows still shape the teacher-arm state distribution.
   - The FINAL_PLAN §8 held-out split may draw on sealed worlds, which count as fit-internal.
3. **The relative boost floor**, as described above.
4. **Wave-based chunking replaces Round 0's perfect-packing chunk formula** (`sizing.chunk_record`). Round 0's
   pinned `size_round` is not called, and Round 0 files are unchanged.
   - The runner gives each pair to one of 12 processes, so a chunk of n pairs runs ceil(n/12) whole waves.
   - Chunks are multiples of **12 pairs**. A chunk therefore no longer holds an exact 4:3:3 mix ratio;
     composition follows Round 0's interleaved `ordered_rows`. This is harmless, because no chunk outcome is
     opened before all chunks are terminal.
   - The replay sample stays Round 0's ceil(10%) per chunk: 2 pairs per 12-pair chunk.

## A6 and A5-D (excluded worlds only; may run before the Round 0 verdict)

**Order:** final commit, then a source-current A6, then A5-D, then activation if the verdict routes here.

- A6 runs on the Mac at H500, one thread, nice 10.
- At H500 the dense path never fires (L < 600). Dense coverage on the real engine therefore comes from A5-D.

**A5-D** (`a5d.py`, `a5d.sh`) is one supervised L4 job:

- K = 12 processes;
- estimate 150 min, **timeout 300 min**;
- **cap 10 u**, plus a 20% held reserve;
- 4.5 h of work, which covers the slowest fidelity world (the ops panel's 1–4 h on L4).

**Fidelity.**

- The 8 T3 dense census worlds (b8_s3, envs 0–7) come from a Mac-prepared, hash-bound fixture. They start at
  t = 0 on 8 of the 12 processes.
- Each must reproduce its endpoints, its batch schedule, and every changed action.
- **EXPLAINED_TIE** (pre-registered, narrow): the first divergence falls at a decision where this run batched,
  and at that decision either the B8 top-2 advisory Q gap is ≤ 1e-4, or the Q gap among the tied-longest
  candidates is ≤ 1e-4. This is the known Mac↔x86 float32 BLAS-order flip; T3's receipts were produced on the
  Mac.
- Any other difference, or an incomplete world, is **HOLD**.
- **Pre-registered HOLD resolution: a Mac re-run** (`a5d.py resolve-hold`). It is needed because the tie rule
  covers only batched decisions, and most decisions are unbatched.
  - Each divergent fidelity world is re-run on the Mac with this exact harness: one thread, nice 10, outside
    the Path B quiet and timing windows.
  - **EXPLAINED_PLATFORM** iff (1) the Mac run reproduces T3 exactly, and (2) at the first decision where the
    Mac and L4 hero action traces differ, the Mac's advisory top-2 Q gap is ≤ 1e-4, or, at a batched
    decision, the Q gap among the tied-longest candidates is ≤ 1e-4. The L4 fidelity export carries the full
    actions.
  - Anything else stays HOLD.
  - The resolution records are hash-bound into the A5-D receipt (`sizing --fidelity-resolution`).
- Scope: fidelity tests the T3 rule only on scripted worlds. For frozen and mixed worlds, "the exact T3 rule" is
  a definitional extension. Its correctness rests on the merged-vs-serial oracle
  (`tests/test_t34_shadow_batch_oracle.py`, all mixes), not on reproducing T3.

**Throughput.**

- The other 4 processes start at t = 0 on the 30 excluded A5-D worlds. Every process that frees up continues with
  those worlds.
- Items still running at stop are recorded with their elapsed time. That is a censored lower bound, and it
  still counts toward the slowest-world bound.
- Each throughput item runs the full paired recipe (bare, dense, metric replay). Bare, metric and export costs
  are therefore measured on L4.
- The rate estimate is frozen as follows:
  - ρ = teacher process-seconds per (rollout row-step + decision), over every completed world;
  - R = max(T3's 859,420 rows per world, the 4:3:3-weighted mean over completed throughput worlds);
  - r_rows = 12·3600 / (ρ·(R + 5000));
  - r_completed = 12·3600 / (mean teacher process-seconds per completed throughput world);
  - **r = min(r_rows, r_completed)**.
- r is never a count of completions in a time window. If nothing completes, the result is HOLD.

## Sizing: frozen rate-to-action table

All inputs come from the A5-D receipt.

**Inputs.**

- **Setup minutes are frozen** at max(15, ceil(observed A5-D setup, export and cleanup) + 5). This is not an
  operator choice.
- **One wave** = 12·60/r + (bare + metric)/60 minutes. Bare and metric are L4-measured means.
- **slowest** = A5-D's slowest world in minutes, counting completed pairs, fidelity worlds and censored in-progress
  items.

**Each chunk.**

- Size: the largest multiple of 12 pairs such that setup + waves·wave ≤ 180 minutes. This is Round 0's
  estimate bound, and it is pinned.
- Forecast: setup + ceil(n/12)·wave.
- Timeout ≥ max(forecast + 25, 2 × forecast, setup + ceil(n/12)·max(slowest, wave) + 25).
- Price: 2.0 u/h on the timeout, plus a 20% reserve held.

**Totals.**

- Expected units = Σ chunk forecasts + the replay.
- Wall = the longer lane's chunk forecasts + 10 min dispatch and ledger gap per chunk + the replay.
- Derived ceiling = 1.2 × expected + (lanes − 1) × max over chunks of (cap − expected).

| Condition (evaluated in order) | Action |
|---|---|
| fidelity not PASS / EXPLAINED_TIE / EXPLAINED_PLATFORM, or nothing measured | **HOLD** |
| r < 5 dense worlds per lane-hour | **KILL** |
| one 12-pair wave exceeds the 180-minute bound | **KILL** |
| expected > 120 u, or derived ceiling > 120 u | **KILL** |
| forecast wall > 24 h on the 2 registered lanes | **KILL** |
| otherwise | **RATIFY**, with an operative study ceiling of **120 u** |

**ESTIMATED rows** (`registration draft`, placeholder L4 costs: bare 190 s/world, metric 50 s/pair, slowest
world = 1.3 × wave, setup 15 min). A test checks that no chunk is stranded at r ∈ {6.5, 7, 8, 10, 15, 23}. The
check runs every chunk as whole waves at the slowest-world time. It charges the other lane in flight at its full
cap, and every admission must fit 120 u.

| r | Action | Chunks × pairs | Expected u | Derived ceiling u | Wall h (2 lanes) |
|---|---|---|---|---|---|
| 5 | KILL (derived ceiling) | 20 × 12 | 109.3 | 136.6 | 29.3 |
| 6 | KILL (wall) | 20 × 12 | 93.3 | 116.6 | 25.3 |
| 6.5 | RATIFY | 20 × 12 | 87.1 | 108.9 | 23.8 |
| 7 | RATIFY | 20 × 12 | 81.8 | 102.3 | 22.5 |
| 8 | RATIFY | 20 × 12 | 73.3 | 91.6 | 20.3 |
| 10 | RATIFY | 10 × 24 | 56.2 | 73.1 | 15.2 |
| 15 | RATIFY | 7 × 36 (last 24) | 38.8 | 52.2 | 11.7 |
| 23 | RATIFY | 5 × 48 | 26.6 | 37.2 | 8.8 |

**Honest 2-lane threshold: about r = 6.5.** With 10-pair chunks and honest wave forecasting it would be about 8.
This registration uses 2 lanes; a third lane is not pre-registered.

- The 120 u cap binds at almost the same r as the 24 h wall.
- A third lane would take the assignment reserved for the Path B unpaced arm.

Below the threshold, the outcome is "not activated: capacity". The only lever that moves the threshold is
per-world cost, through the disclosed greedy-survives-skip amendment below.

## Dense R1: label-yield sizing (only after a published Round 0-D GO)

This replaces the earlier "≤ 30 u" rule, which no plausible r could meet (JUDGMENT §7f).

1. After GO, `labels.measure_yield` reads the 240 sealed teacher worlds. It measures y, the decisive rows per
   world, and p, the fraction of worlds that bear a decisive row.
2. **N_fresh = min(560, max(ceil(300 / y), ceil(60 / p)))**. This is the smallest whole-world count whose
   expectation meets the L2 held-out minimum: ≥ 300 decisive rows and ≥ 60 decisive-bearing worlds.
3. Cost is N_fresh / r lane-hours at 2.0 u/h, on 3 lanes.
4. R1 is **funded only at ≤ 200 u and ≤ 24 h wall**. Otherwise the result is PRECONDITION_FAILED, which is
   reported, with no top-up.

The fresh worlds use namespace `t34-round0d-dense-r1-20261021/v1` at 4:3:3. They are allocated and audited under
this document's rule before dispatch, and are never admitted to any screen, probe, or gate bank.

Examples, at the science panel's yield of about 0.66 decisive rows per world: N_fresh = 455. At r = 8 that costs
about 114 u and 19 h. At r = 15 it costs about 61 u and 10 h. At r = 5 it is not fundable (30 h).

**Dense R1 labels authorize no fit.** A fit also needs:

- (a) the ratified Claim A loss amendment (JUDGMENT §7a), shown on existing rows to move decisive argmaxes;
- (b) its own ordering clause: labels only, until the Path B disposition is published.

The labels-only R1 entrypoint is a follow-up build item, reviewed under its own registration.

**Speed lever, not adopted.** If B8's greedy action survives the full D, no candidate can beat it, so the other
rollouts could be skipped. The ops panel estimates a 3–5× cut in rollout rows. This lever is **not** part of the
registration: it drops the per-action labels on confirm batches, and fidelity would have to be re-run. It can be
adopted only as a disclosed sibling amendment, judged before any A5-D data that would use it.

## Seed-audit coverage limits

**Scanned:** this build's worktree (`docs/`, `research/`, `src/`, `scripts/`) and the sibling artifact tree.

**Not scanned:**

- the main checkout's untracked and ignored files;
- `tests/`, `configs/`, `runs/` and `logs/`;
- opaque binaries.

Matching uses word boundaries, so a seed glued to `_` (for example `seed_123`) is not found. The T3 census check
now fails closed; the census was available for the frozen audit.

## Restrictions

- Preparation, tests and A6 run on the Mac with one thread, at nice 10; pytest uses at most 2 workers.
- This build performed no Colab, RunPod or tmux operation, and activated nothing.
- Path B is never touched and never aborted, because an abort is KILL there.
- During the Path B quiet window the Mac runs only light operator commands. During its timing gate there is no
  Round 0-D settle, no download, and no hashing.

Commands are in [README.md](README.md).
