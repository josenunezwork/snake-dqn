# T35 two-deviation rescue probe — pre-registration (2026-10-20 PT)

**Status: REGISTERED, NOT RUN.** Written and committed before any registered-shape run. Research
probe only: it grants no gate, training, serving or release authority.
**Amended pre-data on 2026-10-20 (Amendment 1, at the end of this file): it supersedes the event
set, beam, verdict, timeout and Commands sections below; the original text is kept as history.**
Source: GPU expert 2,
`docs/research/advice/2026-10-20_a100/gpu_expert_2.md` §1 (last bullet) and §5.

## Question

T3 dense (one deviation from B8 greedy, then B8 greedy, 300-step survival) is the ceiling
mechanism behind 54/64. In the T34 shadow M2 truth scan (`research/t34_shadow/RESULTS.md`),
**33 of 75** historical T3 action changes have *no* single deviation anywhere in their
registered D=300 window that survives 300 steps ("all-die events"). **Are some all-die events
rescuable with two deviations?** If yes, a denser/deeper search oracle has a customer (and a
device port a reason); if no, 1-deviation search is the ceiling and GPU search is dead here.

## Inputs (immutable, Mac)

- T3 dense ceiling receipts `snake-dqn-artifacts/t3-rollout-labels-20261015/dense/ceiling/<world>.json`.
- T34 shadow M2 per-world receipts `snake-dqn-artifacts/t34-shadow-20261019/bank/<world>.json`
  (they define the all-die population, the windows and the layer-1 identity control).
- B8 checkpoints named by the T3 receipts: B8_s0 `bf6a42b3…865f4f`, B8_s3 `3e649783…eb4f8`
  (sha256 rechecked at every event start).

| World | T3 ceiling receipt sha256 | M2 receipt sha256 |
|---|---|---|
| b8_s0_21 | `c01d6032aa1bce47cbba88753dbfda274dfc37bb2d18671f86fb2308e38c1904` | `a89b054ed124005f92ff6c7caa05c0447fc7bd163f155ffa96c863431766c514` |
| b8_s0_8 | `082886120ea15d93b7a54d589ebedfaa83de34f9beaea3fb83ff1ab4bdce87fb` | `78bf7a37338394da8ac0924d0a03b8fe1e8eab98063eafeae0a38ef1f4566fe1` |
| b8_s3_19 | `c39856f9534dfbcd647974a2c3b16cf7ab240b55a7a2958de87dffa4e0707946` | `ead5859a0f336dd0223bf0bf6a091a41f4e761941ae8f09c685902f1f91f9210` |
| b8_s3_26 | `7a09499d474783d43a9b3058f8b44a311cfe8e17a3a1b031b8522190310ce6f6` | `273a8bc75b762b9e7279140000c63f435501589147eeb437820cfb1d515d4f67` |
| b8_s3_6 | `ae5cfa6ed397e33ef27fd62542a74a28cbd6bdf8d29983229be8487583f9c6f4` | `1f0e1a47b4a52cea4976e3bddbc1d1a05c769112af8df4e64913de7e8d46e789` |
| b8_s3_8 | `3981514c4bb0c4443b712d6ff2335b95a55dbf784fc25c85de4e46215ca2af8b` | `40c867e746f06f999687db5b267a9f9eb9667bbe7e60440de7787dcf088501a1` |
| b8_s3_9 | `4cc92367cee51153ba0b4dec1cb1ba19d35c5a3a18eb31542b092f4736b41140` | `4e84f495d56abd7d46e62642af307b9ad42628f00f36a80c5b48cac79b202e41` |

## Events (fixed before any T35 outcome)

An event is one M2 case `world@start`: `start` = M2's registered window start
(`max(previous T3 change + 1, doom − 299)`), `doom` = the reproduced greedy fatal decision.
All-die = M2 stride-one truth `t_star is None`. Denominator check: exactly 33 (code refuses drift).

**Selection rule (outcome-blind):** order worlds by name (the M2 `world_names` sort); take the
earliest all-die event of every world, then the second-earliest of every world that has one,
and so on (round robin). The first 10 are the primary set; the remaining 23, in the same order,
are the EXTEND set. Uses only world names and window starts (`events.round_robin_order`).

*Deviation from the advice text, on purpose:* the expert said "first 10 by world order"; taken
literally that draws 5/10 from one world's chain of successive T3 delays (b8_s0_21) and only
3 worlds. Round robin covers all 7 all-die worlds. Neither rule looks at any outcome.

Primary (10): `b8_s0_21@3707`, `b8_s0_8@4318`, `b8_s3_19@4693`, `b8_s3_26@3647`, `b8_s3_6@4562`,
`b8_s3_8@4246`, `b8_s3_9@4188`, `b8_s0_21@3737`, `b8_s0_8@4353`, `b8_s3_26@3675`.

EXTEND (23): `b8_s3_6@4592`, `b8_s3_8@4256`, `b8_s3_9@4259`, `b8_s0_21@3782`, `b8_s0_8@4378`,
`b8_s3_26@3850`, `b8_s3_6@4602`, `b8_s3_8@4361`, `b8_s3_9@4304`, `b8_s0_21@3787`, `b8_s3_26@3855`,
`b8_s3_6@4702`, `b8_s3_8@4381`, `b8_s3_9@4424`, `b8_s0_21@3882`, `b8_s3_6@4722`, `b8_s3_8@4421`,
`b8_s3_9@4434`, `b8_s3_6@4757`, `b8_s3_8@4426`, `b8_s3_9@4464`, `b8_s3_8@4456`, `b8_s3_8@4496`.

Events are not independent (several share a world); per-world results are reported alongside.

## Engine and semantics

Exact CPU path of T34 shadow M2 (imported, unmodified): `bank_measure.reconstruct_sources`
rebuilds T3's altered timeline to `start` (every saved trigger's greedy action and length
checked), `greedy_shadow` reproduces the doom, `expand_snapshot` → T3 `engine.batched_rollout`
rolls alternatives (B8 rowwise greedy continuation, advisory mask, eps 0, no veto, the world's
real scripted opponents, cloned RNG). Deviation = one non-greedy advisory-legal action.
Survival to horizon H means alive after `min(H, 5000 − t)` executed steps (H5000 episode end).

1. **Layer 1:** expand every decision `t` in `[start, doom]` (≤ 300), every alternative, H = 300.
2. **Beam:** rank all layer-1 first deviations `(t1, a1)` by latest death decision, then earlier
   `t1`, then higher B8 Q, then lower action index; keep **k = 8**.
3. **Layer 2:** continue each beam branch greedily to its own fatal decision `d1`; expand every
   decision `t2` in `[max(t1 + 1, d1 − 299), d1]` (≤ 300), backward from `d1`, every
   alternative, H = 300. Per branch the scan stops at its first confirmed rescue; all 8
   branches are always run.
4. **Survival / rescue:** a layer-2 alternative alive at H = 300 is a candidate; it is a
   **rescue** only if the same greedy continuation is alive at `min(t2 + 600, 5000)`
   (alive 300 further steps, to exclude delays). Candidates in B8-Q order.

**Event outcome:** `RESCUED_DEPTH2` (≥ 1 rescue), `NOT_RESCUED` (full registered search, none),
or INCOMPLETE. Controls, each fatal for the event: input/checkpoint hashes; every
reconstruction trigger check; greedy doom reproduction; **layer-1 identity** (every decision,
every action: survives / death decision / frames executed / horizon equal to the M2 truth
scan); each beam branch's re-simulated death equals its layer-1 death. A confirmed rescue
found before a wall-cap expiry still counts (constructive proof); a timed-out search without
one is INCOMPLETE. No retries, no replacement events.

## Verdict (primary 10)

R = rescued, U = INCOMPLETE/missing. Rule f(r): r ≥ 3 → **REAL** (spec the §3 device kernel as
a successor-campaign item, first round sized ≥ 2,000 dense worlds); r ≤ 1 → **KILL** GPU search
for this project (1-deviation dense CPU is the ceiling mechanism); r = 2 → **EXTEND** (run the
23 EXTEND events, same shape, separate cap; decide on the pooled 33 with the owner/Fable).
If f(R) ≠ f(R + U) the verdict is **INCONCLUSIVE** (report causes; no automatic rerun).

Also reported: rescue rank(s) per event (rescues at beam rank > 4 tell whether k must grow),
row-steps per rescue, unconfirmed (delay-only) candidates, latest layer-2 death per branch
(frames of delay bought by the second deviation), per-world table.

## Layout, caps and preflight

**Layout: Mac, 3 concurrent single-thread processes, `nice 10`, longest-first by the
outcome-blind M2 layer-1 cost, cap 3 h wall for the primary set (0 Colab units).** Rationale:

- Measured Mac preflight (below): ~1,000 row-steps/s per process at nice 10 with Phase 0's six
  workers and the test-hygiene job running (load ≈ 12–13 / 18 cores). Projected primary work
  ≈ 9 × 824,428 (M2 layer-1 row-steps of the 10 events) ≈ **7.4 M row-steps ≈ 2.1 worker-h**;
  largest event ≈ 0.93 M ≈ 16 min; 3 processes → **≈ 45 min wall**. The 3 h cap is ≈ 4× that.
  Three nice-10 single-thread processes keep total load under the 18 cores; Phase 0 runs at
  nice 0 and its endpoints are mass, not timing (compute policy: ≤ 3 research slots).
- One Colab L4 lane (10–12 procs at the 2.4–3.7× per-core slowdown) projects ≈ 40–60 min of
  compute plus setup, ≈ 2–3 u: not faster. It also needs a new hash-bound manifest builder,
  B8_s0 (not in the public release) and census/T1 path redirection, and it would run the
  layer-1 identity control across Mac→x86 float32 ties (known ~1/50-episode tie flips), which
  could turn an all-die event INCOMPLETE or non-all-die. The all-die population was defined
  on this Mac, so the Mac is the faithful platform.
- If the Mac is unavailable (no AC, thermal trip), the run waits; moving to Colab requires a
  layout amendment before data, capped at ≤ 6 u, 3 h.

EXTEND (only on an EXTEND verdict): same layout, cap 2 h wall on the Mac (if moved to Colab by
amendment: ≤ 14 u, separately capped).

**Preflight (Mac, real entrypoint, threads = 1, nice 10; outputs in the session scratchpad, not
counted, not evidence):** disclosed honestly — the preflight ran *before* this file was
committed, after the selection code was written. It used restricted shapes only:
`b8_s0_21@3707` window1-last 20 / beam 2 / window2 20 (9.2 s wall; reconstruction 5.8 s for
3,712 decisions, 357 trigger checks; 352 row-steps; layer-1 identity 0 mismatches) and
window1-last 120 / beam 1 / window2 120 (39.7 s wall; 32,727 row-steps in 32.6 s =
**1,004 row-steps/s**; 0 identity mismatches), plus a launch-plumbing check on EXTEND event
`b8_s3_8@4496` (window1-last 10 / beam 1). Restricted-shape outcomes cannot answer the
question (they omit most of both windows) and do not enter any count.

## Commands

```bash
cd /Users/josenunez/Projects/ml/snake-dqn   # main, at the registration commit
SNAKE_DQN_DEVICE=cpu ./venv/bin/python research/t35_two_deviation/run.py launch \
  --events primary --procs 3 --nice 10 --cap-seconds 10800 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t35-two-deviation-20261020/primary
./venv/bin/python research/t35_two_deviation/run.py summarize --events primary --write \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t35-two-deviation-20261020/primary
```

`launch` sets all numerical thread variables to 1, writes a create-only plan, and every event
writes a create-only receipt with the pinned source-closure hashes; progress lines every 60 s
in `progress/<event>.jsonl`. Code: `search.py` (engine-agnostic search), `backend.py` (exact
CPU backend), `events.py` (selection), `run.py`; tests `tests/test_t35_two_deviation.py`
(toy-world semantics incl. a known depth-1 rescue, beam limits, delay exclusion, episode-end
cap, control failure, and a real-engine check that M2's `b8_s3_21` LAD 1978 / a* = 0
single-deviation rescue is found at depth 1).

---

## Amendment 1 — 2026-10-20 ~19:00 PT, pre-data (no registered-shape run has happened)

Trigger: a 3-persona Fable panel (design / engine / ops) reviewed `a4cb8d5d`; reports in
`snake-dqn-artifacts/fable-panel-2dev/{design,engine,ops}.md`. No blocker; the orchestrator ruled
the fixes below before the approval judge. Everything above that conflicts is superseded.

**A1.1 Decision unit = distinct world; primary = 7 events.** Only 7 worlds have all-die events,
so slots 8–10 of the original list were chained twins of the same trap (dooms 2–17 decisions
apart). Primary = the earliest all-die event of each all-die world (the first round of the
round-robin rule; still outcome-blind): `b8_s0_21@3707`, `b8_s0_8@4318`, `b8_s3_19@4693`,
`b8_s3_26@3647`, `b8_s3_6@4562`, `b8_s3_8@4246`, `b8_s3_9@4188`. EXTEND = the other 26 in
round-robin order: `b8_s0_21@3737`, `b8_s0_8@4353`, `b8_s3_26@3675`, `b8_s3_6@4592`, `b8_s3_8@4256`,
`b8_s3_9@4259`, `b8_s0_21@3782`, `b8_s0_8@4378`, `b8_s3_26@3850`, `b8_s3_6@4602`, `b8_s3_8@4361`,
`b8_s3_9@4304`, `b8_s0_21@3787`, `b8_s3_26@3855`, `b8_s3_6@4702`, `b8_s3_8@4381`, `b8_s3_9@4424`,
`b8_s0_21@3882`, `b8_s3_6@4722`, `b8_s3_8@4421`, `b8_s3_9@4434`, `b8_s3_6@4757`, `b8_s3_8@4426`,
`b8_s3_9@4464`, `b8_s3_8@4456`, `b8_s3_8@4496`.

**A1.2 Beam rule (replaces step 2).** (a) *Exclusion:* drop a first deviation whose branch M2
already proved all-die. Exactly: T3's historical change `(c, chosen)` of the event is excluded
when the M2 case of the same world starting at `c + 1` is all-die (that branch's greedy
continuation is that case's greedy shadow and its layer-2 window equals that case's window).
Primary exclusions: `b8_s0_21@3707` (3736, 2); `b8_s0_8@4318` (4352, 2); `b8_s3_26@3647`
(3674, 4); `b8_s3_6@4562` (4591, 0); `b8_s3_8@4246` (4255, 5); none for `b8_s3_19@4693`,
`b8_s3_9@4188` (their next case is truth-capable). (b) *Stratification:* split the layer-1
window (n decisions, first decision s) into k = 8 index bins, bin(t) = ⌊8 (t − s) / n⌋; keep the
best remaining first deviation of each bin by the original key (latest death decision, then
earlier t1, higher B8 Q, lower action). If fewer than 8 bins have a branch, fill with the best
remaining branches at a t1 not yet used, then with any remaining branch. Ranks = the 8 kept
branches in key order (rank 1 = latest death). Result: 8 distinct t1 spread over the window,
early bins giving long (~300-decision) layer-2 windows. Code: `search.select_beam`.

**A1.3 Verdict on 7 worlds; KILL wording; secondary read-out.** W = rescued worlds,
U = INCOMPLETE/missing: W ≥ 3/7 → REAL; W ≤ 1/7 → KILL; W = 2/7 → EXTEND (the 26 remaining
all-die events, same shape, separate cap ≤ 2 h on the Mac); INCONCLUSIVE when f(W) ≠ f(W + U).
**KILL means: "the registered beam heuristic found ≤ 1 two-deviation rescue → do not fund GPU
search now."** It does *not* show that one deviation is the ceiling (the beam covers ~1% of
first deviations; the advisory mask also limits deviations). The t2 + 600 bar stays the
primary rescue definition. **Secondary read-out (not in the verdict):** number of worlds with
≥ 1 layer-2 candidate alive at t2 + 300 (`layer2_candidates_300`; confirmed + delay-only).

**A1.4 Cap expiry (code now matches the text).** A wall-cap `TimeoutError` during layer 2 keeps
all completed branches: a confirmed rescue already found stays `RESCUED_DEPTH2`
(`cap_expired: true`, counts); otherwise `INCOMPLETE_TIMEOUT` (counts in U). A timeout during
reconstruction or layer 1 is `INCOMPLETE`.

**A1.5 Confirmation control.** The depth-600 confirmation must reproduce the expansion's
300-step survival: a confirmation death before `t2 + min(300, 5000 − t2)` raises
`ControlMismatch` (event INCOMPLETE). The layer-1 identity check now runs right after layer 1
(`INCOMPLETE_IDENTITY` stops before layer 2). Wording correction: `backend.trimmed_source` is new
T35 code (keeps saved triggers through the first one at/after `start`); everything else in
reconstruction is imported unmodified, and the identity control would expose any difference.

**A1.6 Preflight disclosure (outcomes).** Before `a4cb8d5d` was committed: primary event
`b8_s0_21@3707` with window1-last 20 / beam 2 / window2 20 → NOT_RESCUED; with window1-last 120 /
beam 1 / window2 120 → **NOT_RESCUED**, 0 candidates, rank-1 branch (3888, 5) d1 4017, best
layer-2 death 4026; EXTEND `b8_s3_8@4496` window1-last 10 / beam 1 → NOT_RESCUED. After the
amendment code: EXTEND `b8_s3_8@4456` window1-last 15 / beam 2 / window2 15 → NOT_RESCUED
(plumbing check of exclusion (4495, 0) and the early identity check, 0 mismatches). These are
restricted shapes and enter no count. Rule: any further preflight runs only on EXTEND events.

**A1.7 Pre-data facts from the immutable inputs (disclosed; not counted).** T3 itself took
multi-deviation paths in two primary worlds. `b8_s3_19@4693`: T3 changed (4697, 2), then
(4727, 4), and the T3 world survived to 5000 — a two-deviation sequence that meets the 600 bar
(episode end) exists for this event. `b8_s3_9@4188`: T3 changed (4198, 0); the M2 case starting
at 4199 has a single-deviation 300-step survivor at 4204, i.e. a known two-deviation *candidate*
(600 bar unknown). The verdict counts only rescues found by the registered beam; the summary
reports these two known paths separately and whether the beam found them.

**A1.8 Layout and Commands (replaces "Commands").** Unchanged layout: Mac, 3 single-thread
processes, nice 10, cap 3 h wall (10,800 s), 0 Colab units. Projection: M2 layer-1 cost of the
7 events = 578,932 row-steps, × 9 ≈ 5.2 M ≈ 1.45 worker-h at 1,004 row-steps/s → ≈ 30–40 min
on 3 processes (largest event ≈ 15 min). Launch from a **detached frozen worktree** at the commit
that adds this amendment (main fast-forwards every 10 min), with the main venv's absolute
interpreter; the plan file records root, commit and interpreter, and `summarize` refuses
receipts bound to more than one commit:

```bash
git -C /Users/josenunez/Projects/ml/snake-dqn worktree add --detach \
  /Users/josenunez/Projects/ml/snake-dqn-t35-2dev-frozen <AMENDMENT-1 COMMIT>
cd /Users/josenunez/Projects/ml/snake-dqn-t35-2dev-frozen
SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  research/t35_two_deviation/run.py launch --events primary --procs 3 --nice 10 \
  --cap-seconds 10800 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t35-two-deviation-20261020/primary
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python research/t35_two_deviation/run.py \
  summarize --events primary --write \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t35-two-deviation-20261020/primary
```

`<AMENDMENT-1 COMMIT>` is the commit that introduces this section (recorded in CAMPAIGN_LOG /
the handoff report). Tests: `tests/test_t35_two_deviation.py`, 22 (21 fast + 1 slow real-engine).
