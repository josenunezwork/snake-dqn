RULING: APPROVE

# L2′ Mac-placement amendment — independent Fable judgment

Judge: Fable (independent, one judge), 2026-10-22 ~08:15 PT. Read-only; nothing run, nothing committed; the sealed
`t34-round0-study-01` directory was not opened. This file is the ruling the round-12 re-sequencing table scheduled for
"Sat 12:00 — L2′ placement ruling (b)"; it is issued early because every input it was waiting for has landed.

Inputs verified (bytes read, not summaries):
- `exp/bl-k1/timing.json` — `K1_PASS`, `opponent_v8` speedup **20.129×** (bar 5.0, ratio of warm-29 medians, frozen-1500,
  kernel mode, 810 batches, threads 1, workers 1, `foreign_load_observed: true`, `serving_or_release: NOT_GRANTED`).
- `exp/bl-l2/y1-native-mac/ruler-receipt.json` — `bl-native-paired-ruler/v1`, `PAIRED_PASS`, `native_mode: kernel`,
  lease `6825d443…` (owner `claude-sol-s-friday`), `decision_identity {PASS, 270 compared, [] mismatches}`;
  cell `all`: 270/270 batches, 0.6111 batches/s, W1; cell `exact_stop`: 270/270, 1.7930 batches/s, W1; every batch record
  carries one-thread attestation (`OMP/OPENBLAS/MKL/NUMBA/VECLIB=1`, torch 1, numba 1) and `native_v8` counters
  (`mode: kernel`, `stock_rows 0`, `fallback_rows 0`, `noncandidate_rows 0`; `kernel_rows > 0` on frozen/mixed).
  Exact-stop branch counts: 30 evaluated / 90 skipped on eight fixtures; frozen-800 (the doom fixture) 60 / 60.
- `exp/bl-l2/placement-stop-mac.json` — `MAC_AMENDMENT_ELIGIBLE`; per-mix per-worker rates (exact_stop) frozen 1.2221,
  mixed 1.8539, scripted 3.2482 batches/s; (all) 0.4267 / 0.5224 / 1.5683; L4 receipt absent; `timing_transfer: FORBIDDEN`.
- `exp/bl-l2/y1-native-mac/idle.json` — `QUIET`, AC, lid open, no thermal note, Phase 0 absent, `heavy_processes: []`.
- Governing texts: `PLAN_v2.md` §0–§8; `round-11/JUDGMENT.md` (KEEP_B8 §6 text); `round-12/JUDGMENT.md` (Sat 12:00 clause (b):
  "prospective Mac-placement amendment (same bank, analyzer, one look; two workers) only if measured native forecast ≤ 60
  wall-h incl. 20 % reserve and no paced work is displaced"; §4.7 Mac lease); `round-13/JUDGMENT.md` §0, §1.6 (exact stop
  is decision-identical under `_choose`; "reviewed by Codex-V before any bank is touched"); `AMENDMENT_STOP.md` rule 5;
  `SCHEDULING_STOP_STATUS_20261022.md` (any single run once K1_PASS; lease 2 h from launch); `PREREGISTRATION_L2.md`
  (bank 64/64/32, H5000, one look, label rules); `LIVE_FACTS.md` through 07:50 PT.
- Code on main: `ceiling.py` (`_step` has `search_mode`; `run_arm` does NOT pass it — production path is `all`; `_worker`
  never calls `bl_k1.native.configure` — production path is the stock hook); `ruler_operator.py` (reuses `ceiling._step`
  under `configure(native_mode)`, so the ruler measured the same step function the production path uses);
  `bank_engine.rollout_all` (K-1(a) `branch_scope(world.mix)` present, mode from thread-local settings, default stock);
  `analyze.py` (pins `identity.source_manifest == ceiling.source_manifest()` at look time; `require_published_ledger`;
  INCONCLUSIVE with no metrics on any chunk error; consumes no per-branch outcomes); `parity_v2.py` + the preflight fixture
  rows (3 excluded worlds, **H500, 0 dense batches** — the runtime-parity fixture exercises no search at all);
  `mac_lease.py` (create-only, no stale steal); `chunks.py` (8 × 20 = 8/8/4, deterministic order).
- Mac now (07:45 PT): no `exp/mac-lease.json`; tmux `git-sync`, `o2-v3-00/01` (Colab CLI supervisors, 0 % CPU); AC; lid
  open; `pmset -g therm` clean; load avg 5.8/3.5/2.7 from desktop apps (claude.app, fileproviderd), no research compute.

## 1. Ruling and why

**APPROVE** a prospective Mac-placement amendment for L2′, gated on the preconditions in §3. The clause's three tests hold:

1. **Measured native forecast ≤ 60 wall-h incl. 20 % reserve.** Exact-stop: **14.15 h** (base 11.79 h); all-branches:
   **33.06 h** (base 27.55 h). Both from actual per-worker wall seconds on this Mac, same source, same fixture bytes, W1 one
   thread, with the tool's deliberately Colab-shaped 30 min/chunk setup kept in (§2). No Colab number is used.
2. **Decision identity.** 270/270 chosen actions identical between `all` and `exact_stop` on the nine burned Y1 fixtures
   (incl. the doom fixture frozen-800, where the stop must walk alternatives), on top of K-1's 294/294 kernel identity in
   `rollout_all` and the 15,552-case exhaustive unit oracle against unchanged `_choose`. The estimand (paired H5000 mass of
   the dense T3 overlay vs bare B8 s3 on the registered 64/64/32 bank) is unchanged; only diagnostics footprints change
   (round-13 §0, verified: `analyze.py` reads no per-branch outcome).
3. **No paced work is displaced.** PLAN_v2 §0: no paced qualification this week; WP-4 fixture measurement is suspended
   (round-12); E14 is stopped (E3 KILL); Phase 0 finished; nothing paced is registered on this Mac through Tue 23:00. The
   remaining Mac items are unpaced timing receipts (X5 ≈ 2 h, V0 ≤ 1 h, P1 actor probe ≤ 1 h) — they are *sequenced*, not
   displaced (§3 C9). I confirm the "no paced-work displacement" condition the placement tool asked a judge to confirm.

Why not REFUSE: L4 placement is unlawful (stock R = 0.166 vs 2.18; no native L4 receipt), the Colab slots are the campaign's
scarce resource (3 measured, two held by O2 v3 today), and the only cost of the Mac path is ~9–14 wall-hours of an otherwise
idle 18-core machine for the single measurement PLAN_v2 §0 calls the week's M1 decision. Deferring intact to week 2 would
leave the Fri 23:59 / Tue 23:00 KEEP_B8 machinery with nothing to read for the search family.

Why the approval is gated rather than immediate: **the production entrypoint on main cannot yet run the mode that was
measured.** `run_arm` hard-codes `search_mode="all"` and no worker enters kernel mode, so `ceiling --allow-production` today
would run the stock Python hook with all branches — the old Mac rates (frozen 0.055 batches/s ⇒ ≈ 190 Mac-h for the frozen
mix alone), far outside the clause. `AMENDMENT_STOP.md` says so itself ("does not dispatch or change L2 execution"). The
measured path exists only in `ruler_operator.ExactT3Operator`. Wiring it into the production path is a small, pre-data,
Codex-V-reviewed change (§3 P1) — exactly what round-13 §1.6 anticipated — and the chunk receipts must attest execution by
counters, never by a flag.

## 2. The arithmetic (measured rates → wall forecast; sizing, not evidence)

Registered workload: 160 worlds (64 frozen / 64 mixed / 32 scripted), H5000, dense trigger L ≥ 600 every 5th decision;
assumption 578 batches/world (= T3-dense 36,996 batches / 64 worlds; theoretical ceiling 5000/5 = 1,000 ⇒ ≤ 1.73× the
assumption). Rates are per worker-second at W1, one thread; the Mac clause fixes two workers.

**Exact-stop (admitted mode A):**
- World seconds: frozen 578 / 1.2221 = **472.9 s**; mixed 578 / 1.8539 = **311.8 s**; scripted 578 / 3.2482 = **177.9 s**.
- Bank worker-seconds: 64 × 472.9 + 64 × 311.8 + 32 × 177.9 = 30,268 + 19,954 + 5,694 = **55,916 s = 15.53 worker-h**.
- Two lanes (deterministic interleave, longest lane): 28,035 s = **7.79 h**.
- Tool setup assumption 8 chunks × 30 min = 4.00 h ⇒ base **11.79 h**; × 1.2 reserve ⇒ **14.15 h ≤ 60 h** (eligible).
- Per chunk (8/8/4): 8 × 472.9 + 8 × 311.8 + 4 × 177.9 = 6,989 worker-s ⇒ two lanes 3,495 s = 58 min; lane imbalance ≤ one
  frozen world (7.9 min) ⇒ ≤ 66 min; + 5 min Mac-local export/parity ⇒ **chunk estimate 71 min; chunk timeout = 2× = 142 min**.
- Realistic Mac wall (no transport): 8 × ~66 min ≈ **8.8 h**; worst case under timeouts 8 × 142 min = **18.9 h**.
- Sensitivity: at the 1,000-batch ceiling, lanes 13.5 h + 4 h = 17.5 h; × 1.2 = **21.0 h ≤ 60 h** — still eligible.
- Whole-run hard stop: **2 × 14.15 h = 28.3 h from launch** (lease `until`).

**All-branches kernel (fallback mode B, only if P1's exact-stop wiring is not reviewable in time):**
- World seconds: 1,354.7 / 1,106.5 / 368.6 s ⇒ bank 169,312 worker-s = 47.0 worker-h ⇒ two lanes 23.5 h + 4 h = 27.5 h;
  × 1.2 = **33.06 h ≤ 60 h**. Per chunk 21,164 worker-s ⇒ ~3.3 h estimate, **timeout 6.6 h**; worst 8 × 6.6 = 53 h.
  Sensitivity at 1,000 batches/world: 23.5 × 1.73 + 4 = 44.7 h; × 1.2 = 53.6 h — eligible but with little margin; hard stop
  = the 60-h cap from launch.

Start-to-print calendar (mode A): preconditions Thu PM; earliest launch **Thu ~20:00 PT**; terminal **Fri 05:00–11:00**;
one look Fri afternoon ⇒ a print can exist **before Fri 23:59**. If launch slips to Fri, terminal Sat, look Sat/Sun.

## 3. Conditions (all binding; any miss ⇒ the chunk or run is INCONCLUSIVE/REFUSED, never relabelled)

**Preconditions — receipts that must exist before the first bank world is touched**

- **P1 Production wiring (Sol-S → Codex-V review receipt, pre-data, committed + pushed to main before launch).**
  (a) `ceiling.run_arm` passes `search_mode`; `_worker` (inside the spawned process, not the parent) enters
  `bl_k1.native.configure(native_mode, RowStats)` so `rollout_all`'s `branch_scope` runs the kernel; both selected by explicit
  CLI flags (`--search-mode exact_stop --native-mode kernel`), default unchanged (`all`/`stock`) so nothing else moves.
  (b) Each world row exports `search_mode`, `native_mode`, per-world `native_v8` counters (as the ruler's batch records do),
  `branches_evaluated/skipped` totals and `evaluation_order` per batch, observed `worker_threads`; the chunk `identity` records
  `search_mode`, `native_mode`, `workers`, the lease id, and keeps `dense_rule` truthful (the stop evaluates ≤ 6 actions × H300;
  say so). (c) `analyze.py` **refuses** a frozen/mixed row whose counters show `stock_rows > 0`, `fallback_rows > 0`,
  `noncandidate_rows > 0`, or `kernel_rows == 0` while `batches > 0`; a scripted row with any kernel call; a row whose
  `search_mode`/`native_mode` differs from the amendment's registered pair; or a survivor branch with frames ≠ 300. Label map,
  bars, N, H5000, comparator, one-look rule, `_choose`, `DenseTrigger(600, 5)`, `BANK_L2.json`
  (`0a8d3059e1feb7f9c954171093e0c21258b6cf89ea43e1c20f8e035776fcb8d3`) and `B8_SHA256` are byte-unchanged.
  (d) The Mac runtime-parity path works without a Colab root (today `validate_runtime_parity(require_local_runtime=True)` and
  `parity_v2.verify` read `COLAB_CLI_ROOT`); the Mac reference and verification fixture runs are produced on this host from
  the amended source (the old reference's `source_manifest` will no longer match — that is correct, regenerate it; do not
  weaken the check). (e) A Mac launcher in the `run_stop_mac.sh` pattern: `--check` read-only; `--run` acquires the lease,
  runs chunk 0..7 sequentially with `--workers 2 --shard-count 8`, create-only outputs under
  `exp/bl-l2/mac-amendment-01/chunk-0k/`, per-chunk timeout from §2, host samples every 5 min (power, lid, thermal, load,
  top CPU processes), no auto-retry, releases only its own lease on exit. Codex-V's review receipt names the commit, the
  diff scope (only the files above), and the targeted tests run.
- **P2 Amendment text** `research/bl_served_search/AMENDMENT_MAC_L2.md` (sibling of `AMENDMENT_STOP.md`; historical files
  untouched), published before data, stating: platform = this Mac; mode A (or B) fixed for all eight chunks; W = 2; one thread;
  the §2 forecast and timeouts; the lease; the kill rules below; that it changes execution only, never estimand, bank,
  analyzer labels or bars; and that Mac timings never price L4.
- **P3 Production-path identity receipt (0 u; ≤ 2 leased Mac-h; Claude runs; Codex-V checks).** Because the registered
  excluded preflight worlds (H500) never fire the dense trigger, they cannot attest the new mode. Run `run_world` end-to-end on
  **five fresh non-bank development worlds** (2 frozen / 2 mixed / 1 scripted, a new seed namespace audited disjoint from
  every visible bank incl. BANK_L2 and the strict banks, H5000, same B8 bytes) in **both** kernel modes (`all` and
  `exact_stop`) and require: identical chosen action at every fired batch, identical `mass_trace`/`alive_trace`/`mass_ledger`/
  `world_ledger`/`paired_delta` after removing timing and branch-diagnostic fields, counters as in P1(c), and
  `analyze._row_error(row) is None` for every row of both modes. Any difference ⇒ mode A INVALID (fall back to mode B only if
  its own rows pass the same checks against the stock hook on those five worlds, else NOT_READY). This receipt is development
  identity; it is never a speed claim and never touches the bank.
- **P4 Host gate at launch:** `exp/mac-lease.json` absent; idle receipt `QUIET` by the same checker the ruler used (AC, lid open,
  no thermal/performance note, no heavy foreign process, Phase 0 absent); `git status` clean on the launching checkout at the
  P1 commit; published-ledger check would pass (local == origin/main).

**Run conditions**

- **C1 Placement:** this Mac only (M5 Pro, 18 cores, CPU; `SNAKE_DQN_DEVICE=cpu`; MPS never).
- **C2 Processes/threads:** exactly **2 worker processes** (`ProcessPoolExecutor(2)`, spawn) + the heartbeat parent;
  `OMP/OPENBLAS/MKL/NUMBA/VECLIB_MAXIMUM_THREADS=1`, torch 1, numba 1, attested in every row; launched under
  `/usr/bin/nice -n 10 /usr/bin/caffeinate -is` exactly as the ruler was (the measured rates include that scheduling class —
  do not "improve" it mid-run). No third worker, no chunk-level parallelism, no thread raise, even if the Mac looks idle.
- **C3 Lease:** one create-only `exp/mac-lease.json` (owner e.g. `claude-l2prime-mac`, purpose "L2′ Mac amendment chunks
  0–7", `until` = launch + 28.3 h for mode A / launch + 60 h for mode B). Every chunk launch calls `require`; every chunk
  receipt binds the lease id; an expired or foreign lease refuses the next chunk and is never auto-stolen; the lease is
  released only by the launcher on terminal exit (or by the operator after a recorded INCONCLUSIVE).
- **C4 Chunking and timeouts:** the registered eight immutable 8/8/4 chunks in `chunks.py` order, sequential, each
  `--shard-index k --shard-count 8 --workers 2`, create-only outputs + `.sha256`/`.complete`; chunk timeout **142 min** (mode A)
  / **6.6 h** (mode B). A chunk that times out, faults, exits unclean, or fails its counter/identity validation is terminal
  INCONCLUSIVE for the whole bank (PLAN_v2 §3: "chunk fault ⇒ INCONCLUSIVE, never partial analysis"); no retry of a bank
  chunk under this amendment; a fresh prospective recovery needs its own judge ruling.
- **C5 Whole-run stop:** not terminal by lease expiry ⇒ INCONCLUSIVE; stopped attempts are retained, never deleted.
- **C6 Idle / Spotlight / thermal:** the start gate is P4. During the run, host samples are **logged, not acted on** — foreign
  load cannot change an exact-sim outcome, only wall, and wall is covered by C4/C5. Exceptions: (i) a recorded thermal or
  performance warning (`pmset -g therm`) or AC loss ⇒ do not admit the next chunk until cleared within the lease; never kill
  a running chunk; (ii) never kill or renice daemons (`mds`, `mds_stores`, `analyticsd`, `fileproviderd`), never change
  Spotlight/System settings; writing the run directory with a `.noindex` suffix is permitted if the receipt paths are declared
  in P2 before launch. `caffeinate -is` keeps the Mac awake; lid open, AC attached throughout.
- **C7 What may not change (byte-pinned):** `BANK_L2.json` and its audit; `chunks.py` allocation; `analyze.py` label map
  (`FUND_MAC_SYSTEMS→FUND_SYSTEMS`, `KILL_SERVED_SEARCH_PUSH→PACKAGE_KILL`), bars (pooled UB90 < 40 ⇒ PACKAGE_KILL;
  frozen ∧ mixed LB90 > 0 ∧ pooled point ≥ 40 ⇒ FUND_SYSTEMS; else INCONCLUSIVE), the one look; H5000; `DenseTrigger(600, 5)`;
  H300 branches; `_choose` ranking and tie rule; exact-stop semantics (bare first; Q descending, lowest index on ties; first
  H300 survivor stops; all-die ⇒ evaluate all, unchanged ranking; skipped branches exactly `{"status":"NOT_EVALUATED"}`);
  B8 s3 bytes; the served hero (no serving change, no label, no rollback). No source edit in any file of
  `ceiling.source_manifest()` after launch — the analyzer pins it.
- **C8 No timing transfer:** nothing measured here prices L4, a replication bank, or a served claim; chunk wall times are
  recorded as diagnostics only.
- **C9 Sequencing with the other Mac receipts (no displacement):** X5 (2 h), V0 (≤ 1 h) and the P1 actor probe (≤ 1 h) are
  unpaced timing receipts that need the exclusive lease; they run **before** L2′ if ready when P1–P4 land, otherwise **after**
  L2′ is terminal — never concurrently, never by pausing L2′. Colab lanes (P1 100M, O2 v3, density) are unaffected; the
  Mac-light CLI supervisors in tmux may keep running (they are not "research compute" for the idle gate; record them).

**The one look (C10)**

After all eight chunk receipts exist with `.sha256`/`.complete` and the lease is released: Claude runs `analyze.py` **once**,
from a clean checkout at the P1 commit with local == `origin/main` (the built-in published-ledger check refuses otherwise),
`--bank BANK_L2.json --shard chunk-00..07 --out exp/bl-l2/mac-amendment-01/analysis.json`. That invocation is the look; there is
no separate "validation pass", no reading of `worlds/*.json` or shard rows by any person or script before it, and the
progress files carry counts only. A mechanical refusal (missing chunk, digest, counter rule) prints `INCONCLUSIVE` with no
metrics and is not a look. The label goes to `LIVE_FACTS.md`, `CAMPAIGN_LOG.md` and the Sat 16:00 handoff with the PLAN_v2 §3
D3 consequences; the judge who reads it is not the operator who ran it.

## 4. How KEEP_B8 reads with L2′ on the Mac

- **If the print exists by Fri Oct 23 23:59 PT:** clause 1 (PLAN_v2 §6) applies as written — PACKAGE_KILL (pooled UB90 < 40),
  or INCONCLUSIVE with pooled point < 40, ⇒ KEEP_B8 for the search family now (serving/engineering scope; science preserved).
  FUND_SYSTEMS ⇒ §5 route; INCONCLUSIVE with point ≥ 40 ⇒ replication bank registered before any engineering.
- **If L2′ is mid-run at Fri 23:59 (or the look has not been taken):** no print exists, so clause 1 **cannot fire** — the
  literal reading round-12 and round-13 fixed ("missing L2′ is NOT_READY(L2′), not PACKAGE_KILL"). The run is **not**
  stopped, shrunk, peeked at, or hurried for the date; the day-3 D3 decision is taken at the look, whenever the eight chunks
  are terminal, and its consequences apply from that moment. Sat 16:00 handoff line: "L2′ RUNNING on the Mac under the
  amendment; k/8 chunks terminal; look expected <time>".
- **Tue Oct 27 23:00:** clause 2 needs *measured* K-1 < 5×; K-1 measured **20.13×**, so clause 2 cannot fire this week. With
  a print, the §7 reducer uses it; without one (run faulted ⇒ INCONCLUSIVE, or not terminal), the Tuesday disposition is
  NOT_READY(L2′) — nothing funded for serving/engineering, science preserved, bank deferred intact.
- A faulted run (INCONCLUSIVE by C4/C5) is not PACKAGE_KILL and must not be reported as one.

## 5. Is the L4 native ruler still needed?

**Not for this ruling, and not this week.** Mac placement needs no L4 price, and no Colab timing may be copied onto the Mac
either. The ≤ 4-u native L4 ruler regains a purpose only if (i) L2′ prints point ≥ 40 and a *replication bank* (another
64/64/32, PLAN_v2 §3) is registered on Colab rather than on the Mac, or (ii) week 2 wants L2-class search evaluation on L4 —
then it is the lawful price for that family and runs first (same canary/transport/requalification gates as
`HANDOFF_STOP.md`). Do not spend it speculatively; the stale "L4 lawful ⇒ dispatch Sat on B+C" branch of the Sat 12:00
clause is closed by this ruling. The L4 slots are better used by P1 and O2 v3 today.

## 6. Honest limits

- The forecast's per-worker rate was measured at W1; W2 on an 18-core machine with one math thread each is a safe
  assumption, but it is an assumption — C4's 2× chunk timeouts and C5 absorb a 2× miss; a larger miss is INCONCLUSIVE.
- 578 batches/world is the scripted T3-dense mean; frozen/mixed banks may differ. §2's 1,000-batch bound shows mode A stays
  eligible at any count; mode B is eligible with little margin.
- The ruler fixtures are long-body roots (L800–1500) — the dearest decisions — so the rates are conservative for the
  average bank batch; the setup term (4 h) is Colab-shaped and does not exist on the Mac. Expected wall is nearer 9 h.
- Nothing here is efficacy, a served claim, or a release; B8 s3 stays the served hero; no release question exists this week.
