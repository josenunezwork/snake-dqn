# Multi-session paced execution on the Mac (speed program S4) — design memo

Date: 2026-10-21 (written 2026-10-20 ~21:45 PT). Author: Opus agent (design only). Status: **DESIGN — not
registered, not built, not ratified.** Governs nothing until a future study's pre-data registration binds it.

Scope: the **next** paced Mac study (strict gate or the L8 successor; JUDGMENT §3 S4, §5 D5/D7). It does **not**
apply to the ratified Path B retention study `t34-retention-v2-operator-02` at `81c54731`, which runs one session
at a time exactly as registered (JUDGMENT §2 row 5 and §4).

Sources read (main at `d62070e5`): `docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md`,
`research/t34_pathb/README_V2.md`, `research/t34_pathb/{retention_v2,host_guard_v2,analyze_v2,timing_v2}.py`,
`research/t34_pathb/controller_v2.py` (counters), `REGISTRATION_RETENTION_V2_RATIFIED.json` (`load_profile`), the
retention FIX-FIRST rulings (`2026-10-20_t34_retention_prereg_fixes.md`, `_fixes2.md` … `_fixes6.md`) and
`2026-10-20_t34_retention_amendment_bare_unpaced.md`.

---

## 0. Summary

1. Run **N ≤ 4 session processes** at once under one study supervisor. Each session is a separate spawned OS
   process with all numerical pools capped at 1, running whole registered rows (bare then paced, as now).
   Peak demand is 2 busy threads per session, so at most 8 of 18 cores.
2. **The contention ledger counts session-seconds, not host seconds.** A contended 5-second host sample
   costs 5 s × (number of sessions inside a row at that moment). The budget stays at the same absolute
   operator-02 numbers: 2,880 session-s for the study and 576 for the pre-check. So the ledger allows no more
   contended row exposure than a single session does.
3. **Interleaved 1-vs-N timing probe.** It runs after the unchanged single-session v2 timing gate and before any
   gameplay. Blocks: A = 1 session; B = N sessions; C = N sessions + 3 cores of registered synthetic foreign
   load. All blocks are real Watch6 paced probes, about 0.67 h in total.
   - The JUDGMENT D7 bars apply: update p95 ≤ 1.10× single, zero overruns, cadence loss ≤ 5%.
   - New worker-side equivalence bars apply: late/stale/deadline-miss rates and first-result latency. They are
     needed because efficacy depends on the search worker, not only on the tick thread.
   - If a level fails, the study steps down N = 4 → 3 → 2 → 1 inside the same timing phase. N = 1 is today's
     design.
4. **Quiet-host rules generalize as follows.**
   - Sibling sessions are "own" (study process group).
   - The load threshold gains the already-declared but currently **unused** `own_cpu_worker_allowance`
     (2 per active session).
   - A new foreign-CPU-sum rule (> 300% for > 60 s) matches exactly what probe block C validated.
   - The remaining rules are unchanged: the 120 s admission window, power/lid/thermal, and the 2% budget.
5. **Calendar.** About 16.6 h central (19.4 h upper) of single-session chain drops to about **6.6 h (7.3 h)** at
   N = 4. That is **2.5–2.7×** calendar and about 3.0× on compute alone. The JUDGMENT's "3–4×" is optimistic
   because the timing phase and per-phase fixed costs do not parallelize (§5).
6. **Main validity risks.**
   - Core-tier heterogeneity: 6 Super + 12 Performance cores.
   - Contention is *not* guaranteed conservative for the paced arm.
   - Correlated host episodes across concurrent rows.
   - N× exposure to an unmeasured process crash, which counts as a Claim B KILL under the FIX-FIRST #6 abort rule.
   - All are bounded by the probe bars, the build acceptance soak and the ledger (§6).

---

## 1. What the current single-session machinery assumes (and why N > 1 breaks it as-is)

| Mechanism (file) | Single-session behaviour | What breaks with N concurrent sessions |
|---|---|---|
| `HostGuard._own_pids` (`host_guard_v2.py:85-104`) | Excludes only the calling process's descendants and its ancestor chain | Sibling sessions are neither, so each one sees the others as foreign: > 50% CPU for > 60 s gives `other_cpu_sustained`, and `research.` matches `known_heavy_process_patterns` (registered list contains `"research/"`, `"research."`). **Every sample would be contended.** |
| Load rule (`host_guard_v2.py:188-213`) | `load_1m − idle > 3.0` sustained > 60 s; `idle = min(first load_1m, cpu_count/4)` | Own load at N = 4 is ~4.7 cores on average and 8 at peak, so it trips 3.0 continuously. `own_cpu_worker_allowance: 2` is validated (`:52, :68`) but **never used** in `_measure`. Single-session headroom only works because 1–2 own threads fit under 3.0 (bare-unpaced amendment §4). |
| `admit()` quiet window (`:268-293`) | Refuses on any non-quiet sample (`quiet` also requires no process > 50%) | A second session admitted after the first starts can never be admitted. |
| Contention budget (`_summary_locked`, `:331-362`) | `contended_seconds / phase_cap_seconds` against 2%; denominator is the **registered cap** (40 h study), not elapsed | At ~3 h of wall time, 2% of 40 h is 2,880 s, which is 27% of actual wall time. A host episode that hits 4 rows at once would be charged once. The budget must be re-based (§3.3). |
| Caps (`retention_v2._cap`, `:1213-1220`) | `time.process_time()` of the one process vs 60 CPU-h | Per-process clocks under-count by N×. They must be aggregated. |
| Timing gate (`timing_v2.run_probes`; `analyze_v2.timing_summary`, `:326-374`) | Four single-session Watch6 probes (bare/paced × 12/24 fps), 5,000 scored ticks each | Says nothing about the update p95, overruns, cadence or worker latency of a session with N−1 busy siblings. |
| Global RNG / module state (`open_watch` restores the `random` state; bare-unpaced amendment §3 item 4) | One process, one session at a time | Concurrent sessions in **one** process would share `random`, the GIL and module globals. **Sessions must be separate processes.** |

Two incidental observations from reading the code, both outside this design's scope and recorded for the reviewer:

- `own_cpu_worker_allowance` is a dead policy key in v2.
- `timing_v2.probe` writes `"parent_cpu_ms": cpu_ms`, which is the **last tick's** CPU, next to the correct
  `parent_cpu_total_ms`. It is diagnostic only.

---

## 2. Execution architecture

### 2.1 Processes and threads

- **Supervisor** (one process; new module, proposed `research/t34_pathb/multisession_v3.py` or the successor
  study's equivalent) owns the following:
  - registration validation;
  - the remote phase tag;
  - the single 120 s admission window;
  - the **only** host sampler;
  - the row queue;
  - the contention ledger;
  - the aggregated caps;
  - the phase-terminal records and the history ledger append.

  It does no gameplay.
- **Sessions**: `N` worker processes created with `multiprocessing.get_context("spawn")`, never `fork`. Each has
  `SNAKE_DQN_DEVICE=cpu` and `OMP/OPENBLAS/MKL/NUMBA/VECLIB/NUMEXPR_NUM_THREADS=1` in its environment before
  imports, nice 0 and default QoS. **Never** `taskpolicy -b` or background QoS: a trivial probe measured
  background clamping at **2.8× slower** single-thread (0.35 s vs 0.123 s per loop; §4.3).

  Each session runs unchanged v2 row code: `opener(row, ARM_FPS[arm], arm)` → `_collect` → `close(wait=True)`,
  bare first, then paced, both arms in the same process. The process makes one "tick" thread plus, for the paced
  arm, the controller's one worker thread (PROTOCOL_V2 "one extra CPU thread, at most one outstanding request").
- **Per-tick host check** stays a non-blocking cached read. Sessions read the supervisor's latest sample from a
  `multiprocessing.shared_memory` record guarded by a seqlock. The read never waits; a torn read retries the copy,
  not the sample.
- Each session publishes its state word to the supervisor: `idle | bare | paced`, the row id, and its cumulative
  `process_time`. The supervisor attributes samples to active sessions and enforces the CPU cap on the sum.

### 2.2 Row dispatch

- **FIFO in registered order:** the next unstarted row goes to the first free session. Both arms of a row always
  run in one session, bare then paced, as in v2.
- Every row runs exactly once. There is no early stop, retry, exclusion or replacement, as in v2.
- Assignment is recorded in the ledger: `session_id`, `pid`, start/end monotonic and wall time.
- Outcomes do not depend on which session runs a row:
  - the bare arm is clock-free (bare-unpaced amendment §3);
  - the paced arm depends on the session only through timing, which §3 bounds.
- Dynamic dispatch therefore introduces no selection. Static round-robin would add ~5–10% makespan imbalance from
  unequal terminal lifetimes. Both are acceptable; FIFO is recommended.
- The registered row order already interleaves mixes. FIFO therefore keeps concurrent rows mixed, so a host
  episode does not land on one mix.

### 2.3 Failure semantics (unchanged rules, made N-aware)

- **Unconditional fault in any session** (non-finite Q, illegal action, worker error): the supervisor stops all
  sessions at their next tick boundary and records **KILL**, as now. A fault is a property of the controller.
- **Power/lid/thermal state:** non-acceptable state pauses **every** session at its own next row boundary,
  dropping pending assistance as counted fallbacks (v2 `between_rows`). Persisting 120 s gives INCONCLUSIVE
  (unchanged).
- **A session process that dies without a measured cause** (OOM, segfault, signal) counts as an unmeasured
  interruption, which is a Claim B **KILL** under the FIX-FIRST #6 abort rule. N processes multiply this
  exposure; §6 R5 mitigates it.
- **Records:** `records/{kind}-{index:03d}.json` names and the create-only discipline are unchanged; the record
  adds `session_id` and `concurrency`. One remote tag and one `phase_terminal.json` per phase are written, by the
  supervisor only.

---

## 3. Contention accounting generalized

### 3.1 Study process group ("own")

`own` = supervisor PID ∪ all descendants of the supervisor (sessions, their `ps`/`pmset` children) ∪ the
supervisor's ancestor chain. During probe block C it also includes the registered synthetic load processes (§3.4).

Foreign processes keep every v2 rule:

- \> 50% CPU for > 60 s gives `other_cpu_sustained`;
- a pattern match on a hot process gives `known_heavy_process`;
- the patterns themselves are unchanged and still catch a second, unrelated study.

### 3.2 Load rules

| Rule | v2 (N = 1) | N-mode |
|---|---|---|
| Idle baseline | `min(first admission load_1m, cpu_count/4)` = min(·, 4.5) | unchanged; measured once at the supervisor's admission, before any session starts |
| Load threshold | idle + 3.0 | idle + 3.0 + `own_cpu_worker_allowance` × `active_sessions(t)` = idle + 3.0 + 2·k |
| Foreign CPU sum (**new**) | — | Σ foreign `%CPU` > 300 for > 60 s continuously gives `foreign_cpu_sum` |
| Per-process foreign > 50% for > 60 s | yes | unchanged |
| AC / lid / thermal | yes | unchanged; also record `pmset -g therm` `CPU_Speed_Limit` in every sample |
| Memory (**new, diagnostic**) | — | record per-session RSS and `vm_stat` pageouts per sample; a pageout delta > 0 for > 60 s gives `memory_pressure` |

The foreign-CPU-sum rule is the exact boundary that probe block C validates. Without it, many small foreign
processes could add arbitrary load without tripping any v2 rule except `load_1m`, and `load_1m` now carries an
8-thread own allowance.

### 3.3 The ledger: session-seconds

For each host sample `s` (every 5 s): `contended_session_seconds += 5 × k(s) × contended(s)`, where `k(s)` is the
number of sessions inside a row (state `bare` or `paced`).

| Phase | v2 budget (host-s) | N-mode budget (session-s) | Equivalence |
|---|---|---|---|
| Study | 2% × 144,000 = 2,880 | **2,880** | same absolute contended row exposure as operator-02 |
| Pre-check | 2% × 28,800 = 576 | **576** | same |
| Single-session timing gate | 72 (scored ticks: zero) | unchanged (it runs with N = 1) | identical |
| 1-vs-N probe | — | **zero** contended samples in any scored window of any session; else INCONCLUSIVE-retry | same rule as v2 scored timing ticks |

Why session-seconds:

- operator-02 tolerates at most 2,880 s of contended *row exposure*.
- A host-seconds ledger at N = 4 would let one contended host minute touch four rows for the price of one.
- A wall-elapsed denominator would loosen the rule as runs get faster.
- Session-seconds is never laxer than v2: equality holds when N = 1.

Per-tick `host_contended_ticks` flags stay in every row record (v2 `_collect`), so contaminated rows are
identifiable.

**Per-session ledger file** (`contention_ledger.json`, create-only at phase end; JSONL while running):

```json
{
  "schema": "t34-multisession-ledger/v1",
  "N": 4,
  "budget_session_seconds": 2880,
  "samples": [
    {"t": 0.0, "wall": "…", "contended": false, "reasons": [],
     "active": [{"session": 0, "row": "terminal-012", "arm": "paced"}],
     "load_1m": 0.0, "threshold": 0.0, "foreign_cpu_sum": 0.0,
     "cpu_speed_limit": 100, "rss_mb": {"0": 0}, "pageouts_delta": 0}
  ],
  "sessions": [
    {"session": 0, "pid": 0, "cpu_seconds": 0.0,
     "rows": [{"row": "terminal-012", "arm": "bare", "start": 0.0, "end": 0.0,
               "ticks": 0, "contended_ticks": 0, "contended_session_seconds": 0}]}
  ],
  "totals": {"contended_session_seconds": 0, "fraction_of_budget": 0.0,
             "by_reason": {}, "by_session": {}},
  "status": "PASS"
}
```

The status rule is: **INCONCLUSIVE iff `contended_session_seconds > budget` at phase end.** As in v2, the check is
made only at phase end, and interim excess does not stop a phase.

### 3.4 Caps

- **CPU cap:** 60 CPU-h, unchanged, on Σ session `process_time` plus the supervisor's. The work is the same, so it
  is not rescaled. Contention may raise CPU per tick a little, and this shows up in the ledger.
- **Elapsed caps:** keep the study cap at 40 h and the pre-check cap at 8 h. They no longer feed the contention
  denominator, so they are harmless.
- **Timing-phase cap:** rises from 1 h to **2.5 h** to fit the single gate (0.36 h), up to three probe levels
  (0.67 h each) and transitions.

---

## 4. The 1-vs-N timing probe

### 4.1 Placement and identity

The timing phase runs as follows:

1. Supervisor admission (120 s quiet).
2. The **unchanged v2 single-session gate**: four probes, `timing_row`, 5,000 scored and 200 warm-up ticks. All its
   v2 bars apply (update p95 ≤ 11.688 ms at 24 fps, zero overruns at 12/24, cadence ≥ 0.95 × bare).
3. If it passes, the **1-vs-N probe** at `N_max = 4`.
4. Step-down if needed.

The probe runs real offline Watch6 sessions on registered timing seeds `timing_row_s` (s = 0..N−1). They are
disjoint from every study seed and allocated with the study's banks, so each session plays a distinct world, as it
will in the study. No efficacy outcome is produced or read. The selected `N*` is written into the create-only
`timing_gate.json` and binds pre-check and efficacy. It cannot be changed afterwards.

### 4.2 Blocks (palindromic, to cancel linear thermal/clock drift)

| Order | Block | Sessions playing | fps | Scored + warm-up ticks per session | Wall |
|---|---|---|---|---|---|
| 1 | A24 | 1 paced (others spawned, idle) | 24 | 5,000 + 200 | 216.7 s |
| 2 | B24 | N paced | 24 | 5,000 + 200 | 216.7 s |
| 3 | C24 | N paced + synthetic foreign load | 24 | 5,000 + 200 | 216.7 s |
| 4 | A12 | 1 paced | 12 | 5,000 + 200 | 433.3 s |
| 5 | B12 | N paced | 12 | 5,000 + 200 | 433.3 s |
| 6 | C24 | as 3 | 24 | 5,000 + 200 | 216.7 s |
| 7 | B24 | as 2 | 24 | 5,000 + 200 | 216.7 s |
| 8 | A24 | as 1 | 24 | 5,000 + 200 | 216.7 s |

Total = 6 × 216.7 + 2 × 433.3 + 8 × 30 s transitions = **2,407 s ≈ 0.67 h** per level.

Notes on the blocks:

- **Paced is the worst case.** All N sessions play the **paced** arm (tick + saturated worker = 2 threads each).
  This is the worst sustained per-session demand. In the study, siblings also spend time in bare bursts, which use
  one thread.
- **Synthetic foreign load (block C)** is the largest foreign load the generalized guard tolerates *without
  counting it*. Six registered duty-cycled pure-Python processes each run 45 ms busy / 55 ms idle (~45% < 50%
  per-process rule) at nice 0, about 2.7 cores in total. They are members of the probe group, so the guard does not
  flag them, and their 3.0 is added to the own-load allowance during C blocks only.

  Block C answers this question: if the host carries everything the guard lets through, do the timing bars still
  hold?

### 4.3 Bars at level N (all must hold, in every session, in every B and C block)

| # | Bar | Source |
|---|---|---|
| T1 | update p95 ≤ 11.688 ms at 24 fps | contract §4, unchanged absolute bar |
| T2 | max over sessions and B24/C24 blocks of update p95 ≤ **1.10 ×** pooled A24 update p95 | JUDGMENT D7 |
| T3 | zero overruns (full tick ≥ 1/fps) at 24 fps (B, C) and 12 fps (B) | JUDGMENT D7; v2 zero-overrun bar |
| T4 | achieved fps ≥ **0.95 ×** A-block achieved fps at the same rate | JUDGMENT D7 cadence loss ≤ 5% |
| W1 | (late_discarded + stale_discarded + deadline_miss) per 1,000 decisions ≤ max(1.25 × A, A + 2) | **new**: worker-side equivalence |
| W2 | `maintenance_dropped_busy` per 1,000 ticks ≤ max(1.25 × A, A + 2) | **new** |
| W3 | first-result latency p95 ≤ 1.25 × A when ≥ 20 samples exist in both; otherwise `NOT_EVALUABLE` (reported, not a bar) | **new**; PROTOCOL_V2 keeps the 2 s target as engineering-only |
| H1 | zero contended host samples inside any scored window | v2 timing rule → INCONCLUSIVE-retry |

**Why W1–W3.** The tick-thread bars (T1–T4) protect the *visible* cadence. The efficacy signal comes from the
search worker: results scheduled "at frame t*" are discarded when late (`controller_v2._finish` counts
`late_discarded`, `stale_discarded`, `deadline_miss`). Frames advance with wall time at 24 fps, so a slower worker
means fewer rescues even when every tick meets its deadline. D7 alone would not detect this.

All metrics come from existing counters (`controller_state()["counts"]`, `first_result_latency_records`,
`worker_cpu_seconds`). No new instrumentation is needed in the controller.

### 4.4 Step-down and outcomes

- Probe at N = 4.
  - If H1 fails: **INCONCLUSIVE-retry**. This consumes a timing attempt; the maximum of 3 is unchanged.
  - If any T or W bar fails: re-run the full block set at N = 3, then N = 2.
  - `N*` = the largest passing level. If none passes: `N* = 1`, which is the v2 single-session design, already
    qualified by step 2.
- **The step-down never KILLs the study.** D7's "kill" kills *multi-session mode*, not Claim B.
- **No inflation of false timing PASS:** every level must still satisfy the absolute v2 bars (T1, T3), and the
  single-session gate must pass first. Stepping down only removes throughput.
- **Re-roll protection** is v2's: one fixed `timing/attempt-NN` parent, the ledger and the remote tag.

### 4.5 Probe arithmetic

- 5,000 scored ticks × N sessions give p95 order statistics with ~250 exceedances each. The sampling noise of p95
  at this n is ~2–4%, well inside the 10% margin.
- Each condition appears twice in palindromic order. Reporting T2 per repetition (max over both) guards against
  one lucky block.

---

## 5. CPU / thermal budget (M5 Pro) and calendar arithmetic

### 5.1 Host facts (measured 2026-10-20 21:32 PT, `sysctl`, `pmset`, `ps`; read-only)

| Item | Value |
|---|---|
| CPU | Apple M5 Pro, 18 cores: **perflevel0 "Super" 6 cores** (one cluster, 16 MB L2), **perflevel1 "Performance" 12 cores** (two 6-core clusters, 8 MB L2 each) |
| RAM | 64 GB |
| Load at measurement | 12.12 / 10.95 / 9.75: Phase 0's six workers at ~92–94% CPU each, the Claim A harness at ~112%, a 165% `ugrep` |
| Thermal | `pmset -g therm`: no thermal warning, performance warning or CPU power status recorded, after ~3 days of Phase 0 load |
| Power | AC attached, battery 80% not charging |
| Session RSS reference | Phase 0 workers 0.56–0.66 GB; Claim A harness 1.31 GB |

A trivial probe (1 thread, nice 10, < 3 s, contended host) timed a pure-Python loop at 0.123 s with default QoS,
0.125–0.143 s under `utility`, and 0.344–0.364 s under `background`. Background QoS must never be used for
sessions. The probe does not separate Super from Performance cores; the B blocks measure that directly.

### 5.2 Thread demand

| Per session | Busy threads |
|---|---|
| paced arm | tick ≈ 8–10 ms / 41.7 ms ≈ 0.2–0.25 core + worker ≤ 1.0 core (worst case saturated) → **≤ 1.25 avg, 2 peak** |
| bare arm (unpaced) | 1.0 core while free-running, no worker |
| efficacy mix (paced 8.55 h : bare 2.95 h of work) | 0.74 × 1.24 + 0.26 × 1.0 ≈ **1.18 cores average** |

**N = 4.**

- Demand is **~4.7 cores on average and 8 threads at peak**, plus the supervisor sampler (~0.01) and the OS
  (WindowServer etc., ~0.3–0.5). That leaves ≥ 9 of 18 cores idle.
- The average fits within the 6 Super cores. Peaks spill ~2 threads onto Performance cores, which is the main
  timing risk (§6 R1) and exactly what B24/C24 measure.
- Thermal: Phase 0 has run 6 threads at ~93% each, plus other research load, for ~3 days with no thermal record.
  The planned average of 4.7 cores is below that regime and the peak of 8 is near it.
- Memory: ≤ 1.5 GB per session gives ≤ 6 GB of 64 GB.

**N = 3** keeps the peak at 6 threads, matching the Super-core count. That is why N = 3 is the first step-down
level.

**Policy conflict to resolve at registration.** `docs/STATUS.md:125` says "strict gates 2 slots", and the
compute-use policy says strict gates stay on 2 slots. JUDGMENT §2 row 5 relaxes "one paced session at a time" for
later paced runs at "N≈4 … each ≤ 2 threads". The registering study must cite that row and record the
STATUS/compute-policy supersession in CAMPAIGN_LOG. A strict-gate run in N-mode also needs the Fable judge's
ratification (AGENTS.md approvals).

### 5.3 Calendar arithmetic

Inputs:

- Single-session planning numbers from the bare-unpaced amendment §5:
  - timing 0.361 h (= 2 × 5,200/12 + 2 × 5,200/24 s);
  - pre-check: paced 3.19 h + bare ≤ 0.77 h;
  - efficacy: paced terminal 6.7 h + paced Watch6 1.85 h + bare ≤ 2.27 h;
  - upper bounds: pre-check 4.6 h, study 12.9 h.
- Fixed overhead O ≈ 1.5 h: three admissions, three ledger commit/push gates, analysis, operator latency between
  phases. It does not parallelize.
- Bare work under concurrency costs **s_b = 1.3×**, an assumption for Performance-core spill. Paced work is
  wall-clock fixed.
- Per-phase tail = one longest row (~0.07–0.08 h).
- N-mode timing = 0.361 + 0.669 = **1.03 h** (one probe level).

| Scenario | N | Timing | Pre-check | Efficacy | Compute | Chain (+O) | Calendar speed-up | Compute-only |
|---|---|---|---|---|---|---|---|---|
| central | 1 | 0.36 | 3.96 | 10.82 | 15.14 | **16.6 h** | 1.0× | 1.0× |
| central | 2 | 1.03 | 2.17 | 5.83 | 9.03 | 10.5 h | 1.58× | 1.68× |
| central | 3 | 1.03 | 1.47 | 3.91 | 6.41 | 7.9 h | 2.10× | 2.36× |
| central | **4** | 1.03 | 1.12 | 2.96 | 5.10 | **6.6 h** | **2.52×** | 2.97× |
| central | 6 (not proposed) | 1.03 | 0.77 | 2.00 | 3.80 | 5.3 h | 3.14× | 3.99× |
| upper | 1 | 0.36 | 4.60 | 12.90 | 17.86 | **19.4 h** | 1.0× | 1.0× |
| upper | **4** | 1.03 | 1.28 | 3.51 | 5.82 | **7.3 h** | **2.65×** | 3.07× |

Example row (central, N = 4):

- pre-check = (3.19 + 0.77 × 1.3)/4 + 0.07 = 1.12 h;
- efficacy = (8.55 + 2.27 × 1.3)/4 + 0.08 = 2.96 h.

Each probe step-down level adds 0.67 h, so stepping to N = 3 gives ≈ 8.6 h central (1.9×). The operator-02
forecast of 18–19 h sits between our central and upper single-session rows, so the honest expectation is
**18–19 h → 6.5–7.5 h**.

JUDGMENT §3 S4 estimated "~3–4× (18–19 h → ~5–6 h)". Only compute-only at N ≥ 4 reaches that range. A one-time
1-vs-N probe and the fixed per-phase steps are Amdahl terms. To reach 3×, either cut O (scripted ledger pushes)
or reuse a passed probe across studies with identical source and host. Reuse needs its own ruling and is **not**
proposed here.

Cores freed: single-session operator-02 idles ~16 of 18 cores for ~17 h. N = 4 uses ~4.7 on average for ~5 h.
Total Mac CPU-hours are unchanged (same work), so the 60 CPU-h cap is unaffected.

---

## 6. Risks to validity and their controls

| # | Risk | Mechanism | Control |
|---|---|---|---|
| R1 | **Core-tier heterogeneity** | The single-session gate runs on a Super core. At N = 4 the peak of 8 threads exceeds the 6 Super cores, so a tick or worker thread may run on a slower Performance core, raising update ms and worker latency. | B24/C24 measure this directly (T2, W1–W3). The step-down to N = 3 matches the Super-core count. QoS stays default for all sessions, with no per-thread QoS tweaks: those would be a new controller identity, and v2 has none. |
| R2 | **Contention is not guaranteed conservative** | The bare-unpaced amendment §4 argues load only costs the candidate deadlines. Fewer *applied* interventions move the paced arm toward bare, which makes survival non-inferiority guards **easier** to pass. If late results are disproportionately the harmful interventions, a mass gain could survive while harm is masked. | W1 bounds the discard rates at ≤ 1.25× single. The analysis pre-registers a **descriptive** sensitivity table: per paced row, applied/late/stale/deadline-miss rates versus mean concurrency `k̄` during the row, and the paired mass/death deltas by `k̄` tertile. It is never verdict-changing, consistent with prereg-fixes #1 (efficacy overruns are diagnostic). |
| R3 | **Estimand drift** | Deployment serves one controller per Watch host. N-mode measures that controller with ≤ N−1 sibling processes. | The probe proves equivalence of the visible timing and worker-side metrics within registered margins. The registration states the estimand as "paced controller under probe-qualified concurrency N*", not a different controller. |
| R4 | **Correlated host episodes** | Concurrent rows share host state, so a single foreign burst touches up to N rows. Paired-world independence (P7 unit) weakens slightly. | The session-second ledger charges each episode N-fold. FIFO keeps concurrent rows on different mixes and seeds. The 2,880 session-s budget equals operator-02's. |
| R5 | **N× crash exposure** | Any unmeasured session death counts as a Claim B KILL (FIX-FIRST #6). | Build acceptance includes a **4-session development soak**: ≥ 30 min, development rows only, no registered seeds, including the longest Watch6 rows and death/respawn. It records peak RSS and requires 0 crashes. RSS ≤ 1.5 GB per session is asserted. Sessions use spawn, so no fork-copied state. |
| R6 | **Process isolation of bare outcomes** | The bare arm must be identical regardless of concurrency. Hidden shared state (files, global RNG) would break pairing. | Build acceptance test: reuse `tests/test_t34_pathb_bare_unpaced_v2.py`'s canonical-sha comparison for 3 terminal mixes + 1 Watch6 development row, run (a) alone and (b) as one of 4 concurrent sessions. Every `_collect` outcome field must be byte-identical. This joins the trace-receipt glob. |
| R7 | **Thermal or clock drift over hours** | The probe is ~40 min; the study runs ~5 h at sustained load. | Palindromic blocks; C blocks run hot. `CPU_Speed_Limit` is recorded each sample, and `thermal_not_ok` for 120 s gives INCONCLUSIVE (unchanged). There is a descriptive efficacy-phase sentinel: paced-row overrun and discard rates by hour versus the probe's B level. |
| R8 | **Sampler/ledger bugs** create false PASS on contention | New code path. | Executable acceptance contract (§7). Synthetic-sample tests cover: sibling exclusion; that a foreign process still trips; own allowance scaling with k(t); foreign-sum boundary at 300; session-second arithmetic; phase-end-only evaluation. |
| R9 | **Re-roll via N selection** | Repeated probes could be used to fish. | N* is chosen only from timing data, before any gameplay, by a fixed rule. Max 3 attempts, the fixed parent, the history ledger and the remote tag are unchanged. |
| R10 | **Pooling with operator-02** | Different execution identity. | Not pooled. The N-mode study stands alone (D5: a timing-changing controller identity gets a fresh paced test). |

---

## 7. Build plan (Sol, 1–2 days, after operator-02 starts; JUDGMENT S4)

New or changed modules:

- `multisession_v3.py`: supervisor, FIFO dispatch, shared-memory sample, aggregated caps, ledger.
- `host_guard_v3.py`: group-own set, k-scaled allowance, foreign-sum rule, session-second ledger, memory diagnostic.
  The `validate_policy` keys gain `foreign_cpu_sum_percent: 300`, `foreign_cpu_sum_sustained_seconds: 60` and
  `contention_unit: "session_seconds"`.
- `timing_multi_v3.py`: block schedule, synthetic load processes, bars T1–T4, W1–W3, H1, step-down.
- `analyze` additions: probe summary, sensitivity table.

v2 files stay untouched for operator-02.

One executable acceptance contract (JUDGMENT §2 row 7):

1. All v2 retention tests are still green.
2. New unit tests for R8.
3. The R6 identity test.
4. The R5 soak.
5. A dry probe on development seeds at N = 2 with 500 ticks per block, labelled development: **no registered
   seeds, outside any quiet window, ≤ 2 threads**. Only the quiet-host registered probe can qualify.

Then one independent reviewer, delta-only re-review, and one judge.

---

## 8. Draft registration text (for the next paced Mac study's preregistration)

> **Multi-session paced execution (pre-data).** The study runs under `execution_mode: "multisession/v1"`
> (design: `docs/research/specs/2026-10-21_paced_multisession_design.md`, bound by sha256 in `source_hashes` and
> `amendments`).
>
> **1. Processes.** One supervisor process and `N*` spawned session processes. `N*` ∈ {1, 2, 3, 4} is selected in
> the timing phase by §4. Every session sets all numerical thread pools to 1, runs at nice 0 with default QoS, and
> executes whole registered rows, bare (free-running) then paced (24 fps), in one process. Rows are dispatched FIFO
> in registered order to the first free session. The assignment is recorded and is not an analysis variable. No
> row is retried, excluded, replaced or resumed.
>
> **2. Host guard.** Admission is one 120-second quiet window before any session starts. "Own" processes are the
> supervisor, its descendants and its ancestors. A sample is contended if any of the following holds:
>
> - (a) `load_1m` exceeds `min(idle, cpu_count/4) + 3.0 + 2·k(t)` for > 60 s, where k(t) is the number of
>   sessions inside a row;
> - (b) any foreign process exceeds 50% CPU for > 60 s;
> - (c) a hot foreign process matches a registered heavy-process pattern;
> - (d) the summed foreign CPU exceeds 300% for > 60 s;
> - (e) AC, lid or thermal state is not acceptable.
>
> Sampling is every 5 s. Per-tick checks read a cached sample and never wait. Pauses happen only between rows and
> drop pending assistance as counted fallbacks. A power/lid/thermal state persisting 120 s is INCONCLUSIVE.
>
> **3. Contention ledger.** Each contended sample adds 5 s × k(t) contended session-seconds. Budgets are evaluated
> only at phase end: pre-check 576, study 2,880 session-seconds. Exceeding a budget makes that phase INCONCLUSIVE.
> The single-session timing gate keeps its v2 rules. The 1-vs-N probe allows zero contended samples in any scored
> window (INCONCLUSIVE-retry).
>
> **4. Timing phase** (cap 2.5 h, counted toward the study's elapsed and CPU caps; at most 3 attempts in
> `<study_root>/timing/attempt-NN`):
>
> - (i) The v2 single-session Watch6 gate, unchanged.
> - (ii) On PASS, the interleaved probe at N = 4. Blocks A24, B24, C24, A12, B12, C24, B24, A24, each 200 warm-up
>   + 5,000 scored ticks per playing session, on registered timing seeds disjoint from all study seeds.
>   - A = 1 session.
>   - B = N concurrent paced sessions.
>   - C = B plus six registered 45%-duty synthetic load processes.
> - The level passes iff, in every session and every B/C block:
>   - update p95 at 24 fps ≤ 11.688 ms and ≤ 1.10 × the pooled A24 p95;
>   - zero overruns at 12 and 24 fps;
>   - achieved fps ≥ 0.95 × A at the same rate;
>   - (late + stale + deadline-miss) per 1,000 decisions ≤ max(1.25 × A, A + 2);
>   - busy-dropped maintenance per 1,000 ticks ≤ max(1.25 × A, A + 2);
>   - first-result p95 ≤ 1.25 × A where both have ≥ 20 samples.
> - A failing level is re-probed at N − 1. `N*` is the largest passing level, or 1 if none passes.
> - `N*` is written to `timing_gate.json` and binds all later phases.
> - Multi-session failure never by itself KILLs the study.
>
> **5. Caps.** Elapsed caps are as registered by the study (pre-check 8 h, study 40 h). The CPU cap is 60 CPU-h on
> the sum over supervisor and sessions.
>
> **6. Faults and interruptions.** Any unconditional fault in any session is KILL for the whole study. Any session
> exit without a HostGuard-recorded cause is an unmeasured interruption, which is Claim B KILL (FIX-FIRST #6 rule).
>
> **7. Pre-registered descriptive analysis** (never verdict-changing):
>
> - per paced row: applied, late, stale and deadline-miss rates, efficacy overruns and mean concurrency k̄;
> - paired mass and death deltas by k̄ tertile;
> - an hourly sentinel versus the probe's B level.
>
> **8. Governance.** This relaxes "one paced session at a time" and the "strict gates 2 slots" line
> (`docs/STATUS.md:125`) for this study only, per JUDGMENT 2026-10-20 §2 row 5 and D7. Recorded in CAMPAIGN_LOG
> with the Fable judge's ratification. Not applicable to `t34-retention-v2-operator-02`.

---

## 9. Open questions for the reviewer and judge

1. **s_b = 1.3 bare slowdown.** This is an assumption. The first B-block run measures bare-free ticks only if a
   bare probe is added. Should the probe add a "B-bare" block (N−1 paced + 1 bare free-running) to price it? It
   affects only the calendar, not validity.
2. **W-bar margins.** max(1.25×, +2 per 1,000) is a judgment call. Development probes (§7 item 5) should report the
   observed A-vs-A spread first, and the margins should be frozen pre-data at ≥ 2× that spread.
3. **Probe reuse.** Should a passed probe at an identical source commit and host be reusable by a later study
   within 7 days? This would save 0.67 h. Not proposed.
