# Round 12 judgment — RETROSPECTIVE: why the week-1 levers went badly, and whether PLAN_v2 is still the plan

Judge: Fable (round-12 judge), 2026-10-21 **~20:50 PT** (wall clock `date` = 20:45 PDT at start). Inputs: `../CHARTER.md`,
`../round-10/JUDGMENT.md`, `../round-11/JUDGMENT.md`, `../PLAN.md`, `../PLAN_v2.md`, `../LIVE_FACTS.md` (newest 17:30 PT),
`../QUEUE.md`, `QUESTION.md`, `personas.txt`, `astra_01..10.md` (all read in full), `CAMPAIGN_LOG.md:78-109`. Repo read-only at
`1a000f3e` (main HEAD); receipts under `../exp/` and `../../t34-*/` opened with light commands. Sealed study not opened. No jobs,
Colab or tmux. Owner's override for this round: retrospective only — **no round-13 inputs are written.**

Facts newer than LIVE_FACTS / QUESTION, verified in receipts tonight and folded in as authoritative:
- **P1 Y3 source-02 was dispatched (17:24) and FAILED at 18:12 PT** — `exp/bl-p1/y3/source-02/run.log` (EXIT=1, elapsed 2,915 s ≈ 49
  min ≈ 1.6 u, `stop_verified: true`), `colab-output/terminal.json` `returncode: 1`, job log tail
  `ValueError: P1 phase did not stop at its registered pause boundary` raised by `research/bl_ppo/p1_runtime.py:357` on the **origin**
  snapshot (`_require_pause(origin, pause=0)` at `:490`). The supervisor reported it as `Refused('remote terminal failed or foreign
  generation')` (`research/colab_cli/supervisor.py:1457-1458` conflates nonzero child exit with generation mismatch — 03 right).
  `exp/bl-p1/y3/receipt.json` still says `PREPARED_NOT_DISPATCHED` (preparation receipt, never superseded). Sol's fix lane
  `t34-sol/p1-fix/` opened 20:45, empty. **D1′ (Thu 14:00) cannot be met as written.** This is the **ninth** mechanical Colab failure.
- **X5 has no rerun receipt.** `CAMPAIGN_LOG.md:109` says "X5 rerun launched in an exclusive Mac window"; the newest X5 artifact is
  `exp/week1-20261021/mac/x5-independent-review.json` (15:01) and `x5-disposition.json` says `NOT_READY`, `timing_measurement:
  UNAVAILABLE`. No X5 process is running. X5 is still NOT_READY (08 right that the log overstates receipts).
- The QUESTION's E4 pointer (`exp/bl-e2/e8-preflight-03/e4-ruler.json`) is the **tiny Mac preflight** (2 worlds, 1 process, 254/s);
  the real L4 ruler is `exp/bl-e2/z9-e4-recovery-01/e4-output/exports/e4-ruler.json` (1,403 / 1,444 active, 821 / 867 allocated
  hero transitions/s at 6 / 12 procs) — 01/06 right. Numbers in S4 are unchanged.
- P1's registered invoice is **stricter than PLAN_v2's 46-h gate**: 120 u incl. 2× liability + 20 % reserve ⇒ nominal complete-job
  forecast **≤ 24 h** at 2 u/h (`research/bl_ppo/REGISTRATION_P1.md:106-111`) — 05/06/08 right. E4's 19.3 h *collection-only*
  figure leaves ~5 h for updates, probes, checkpoints and transfer; a 32-h allocated-rate forecast fails the invoice.
- PLAN_v2 §2 starts Y3 at Thu 12:00 on slot A while §3 D1′ at Thu 14:00 requires Y3 PASS **and** the 1M qualification, which WP-3
  schedules "Thu PM" (`PLAN_v2.md:29,46,71`) — an internal inconsistency (08 right), now moot because Y3 failed.
- Verified code facts underlying F7: `ruler.py:134-135,214` validates/sets a Boolean `s2_native_v8` only; `native_v8.py:116-141`
  runs natively only inside `native_scope`; the newly integrated `ruler_operator.py:115-116` defaults `native_mode="stock"`
  (10 right). K-1 (`research/bl_k1/native.py:93-113`) attests kernel inputs row-by-row against the stock hook — that is the
  execution-attestation pattern every timing receipt should have had.
- Mechanical-failure receipts re-read: `exp/bl-l2/run-y1.log:1` (AccountingError list-vs-map, `accounting.py:93`), `run-y1b.log:1`
  (upload cap on `ruler-fixtures/*.sha256` — cumulative 300-s staging exhaustion, not an oversized sidecar), `exp/bl-e3/run-01.log:1`
  (57.8 MB archive), `run-02.log:21` (exec poll 51 s), `run-03.log:88-89` (decision-2916 identity mismatch, masked as foreign
  generation), `exp/bl-e2/run-e4.log:1` (checkpoint upload), `exp/bl-e1b/l4-output/poll-000010.json` (`scatter_reduce_` on a bool),
  `CAMPAIGN_LOG.md:96` (D0 sidecars), `:106` (2.5-h concurrency-test gap), `x5-disposition.json` (K-1 audit in the "exclusive" window).
- Git census since 10-20 17:00: 338 commits, 90 merges (matches the question). Builder/review **hours are unmeasured**; PLAN budgeted
  122–146 h, PLAN_v2 54–72 h (`PLAN.md:70-71`, `PLAN_v2.md:93-95`) — forecasts, not a timesheet (all ten advisors agree).

## 1. Bottom line (for the owner)

1. **The levers were mostly not disproven this week; our premises were.** Of the ten bad outcomes, three are clean scientific
   negatives (Path B deliverability, Claim A distillation, E3 representation), five are planning errors (we committed dates, bars
   and builds on numbers nobody had measured on the target platform or code path), and two are process failures (one operator
   sequencing slip that voided a sealed 240-pair study; nine mechanical Colab failures from an unqualified transport/CUDA pipeline).
2. **The one mechanism that has ever beaten B8 (exact search, +92 / +122.8 offline, scripted) is still unmeasured on frozen/mixed
   worlds and still unserved.** Its served form is 15–25× too slow on this Mac *as timed* — but that timing ran the Python opponent
   hook while the compiled, action-identical replacement sat merged and switched off. Even with it on, arithmetic says ~4× short.
3. **Colab L4 vCPUs are 7–10× slower per worker than this Mac on our search workload.** That single number, obtainable for < 1 u on
   day one, invalidated L2′'s sizing, chunking, bar and both plans' calendars. A 10-17 memo had already warned of 3.7×.
4. **PLAN_v2 is off its own dates and its central measurement (L2′) has no lawful execution path on Colab.** Its three questions
   are still the right ones; its calendar and its promise of three prints by Sat are not.
5. **B8 s3 stays served. No release question exists this week** (unchanged). The pre-registration machinery worked as a brake —
   every negative is clean, nothing was laundered — and failed as a steering wheel: it did not make a single instrument executable
   before dependents were built.
6. **What changes now:** K-1 timing Thu night becomes the critical receipt; L2′ is placed only after a *whole-workload* native ruler
   on the host that would run it; P1 keeps one slot but must first fix and pass Y3 and meet its own ≤ 24-h invoice; WP-4 and E14
   stop; eight process gates (target-platform canary, measure-before-plan, execution attestation, shared schema validation, error
   classes, power sheet before build, Mac lease, pre-claim ledger check) are adopted with owners and enforcement.
7. **The KEEP_B8 criterion stands unchanged**; it is applied literally (missing timing ≠ "< 5×", missing L2′ ≠ PACKAGE_KILL).
8. Honest Saturday: K-1 verdict, L2′ placement status (possibly UNKNOWN), P1 readiness, X5 rate, E3 closure — not three prints.

## 2. Failure classification (one primary class each; verified against receipts)

| # | Outcome | Primary class | Root cause (receipt) | Knowable earlier? |
|---|---|---|---|---|
| F1 | Path B v2 smoke STOP (0/34 rescues; N=6 could not hold 24 fps) | **World/science** (for the tested dense served form) | Full evaluations arrive 7–30 s after a 9–18 s doom lead; the dense wrapper cannot deliver inside the lead with this engine (`t34-pathb-smoke-20261021/RESULT.md:10-39`; stopping rule 7622d62e pre-data). | **No** — deliverability needed the run; running the cheap smoke *before* the 27-h operator-02 chain was the process working. The planning error avoided here (release scope on an unmeasured served latency) recurs as F7. |
| F2 | Claim A D0 = KILL | **World/science** | No corrected objective keeps anchor guards ≥ 512 updates; held-out agreement collapses (`d0-l4-02/.../D0_DECISION.json`, `registered_run: true`). The lost first run is F10(a). | **No** (needed the fit). |
| F3 | Round 0 integrity KILL, 240 sealed pairs never read | **Process/ops** | Replay ledger not published before the sole analysis claim; `require_published_history` fired first; HistoryError → KILL (`2026-10-21_round0_integrity/JUDGMENT.md:15-24`; CAMPAIGN_LOG 00:20/00:40). Operator error on a known ordering requirement; the system enforced its rule and lost the science. | **Yes** — the README ordering existed; a read-only pre-claim ledger check would have refused before consumption. |
| F4 | E1b GPU engine NO_CONTINUE (0.217 world-steps/s vs ≥ 20,000 bar; 1/18 decisions in 10,820 s) | **Planning error** (dtype miss = process) | The engine thesis was funded as a week-1 lever with no target-platform throughput measurement; Mac CPU preflight printed 139/s "not predictive of CUDA"; first L4 run died on `scatter_reduce_` over a bool that CPU cannot see (`benchmark.json`; `poll-000010.json`; fix 88adb3e2). A ~92,000× miss is a design fact about a per-decision serial device loop, obtainable from a one-decision CUDA canary before committing a 3-h budget. | **Direction yes** (10-20 A100 memo: host-bound; E1 spec asked for throughput first), **magnitude no**. |
| F5 | E9/E9′ screen underpowered → E2 not funded (.27–.36 < .60) | **Planning error** | Screen family and bars designed and built before the funding event was simulated; T23 SDs 365/263/212 existed on 10-20 (`E9_REPLACEMENT.md:132-133`; `e9-replacement/report.json`). Not a measured small PPO effect (05/06 right: .362 is a design number). | **Yes.** |
| F6 | O2 needs 5,097 worlds (fallback 1,260 at .34) | **Planning error** | Built at N96 before pricing +10 incremental precision; its own prereg disclosed inadequate power (`REGISTRATION_O2_V2.md:46-56,80-82`; 02 cites `PREREGISTRATION_O2.md:114-134`). The 5,097 is a bound under proxy variances, not a harvest fact. | **Yes.** |
| F7 | L1′ served exact search 15–25× too slow (stock hook) | **Planning error** | Timed with the Python opponent veto (78–90 % of every frozen/mixed batch) while S2 native (merged 23:58, 1.90× whole-workload, D3 PASS) was opt-in and never entered; `ruler.py:134,214` Boolean asserted "S2 on"; production consumption path never built; host not quiet (`RESULT_L1.md:29-44`; `y1-mac-01/mac-ruler.json`; `native_v8.py:116-141`). The world-fact residue: even at a 5× hook gain the frozen-1500 batch is ≈ 5.9 s vs a 1.458-s tail cap (Amdahl at p = .78–.90 ⇒ 2.7–3.6× whole-batch; 04/07 arithmetic verified). | **Yes** — opt-in was in the docstring and D3 receipt (`RESULT_D3_MAC_CONTENDED_01.md:75-76`); a 5-minute wiring review before timing would have caught it. |
| F8 | Y1 L4 R = 0.166 vs bar 2.18 → L2′ NOT_DISPATCHABLE; Mac×2 forecast 566 min/chunk, 342 u | **Planning error** | L2′ sized, chunked and barred on Mac rates with no L4 batch; PLAN assumed R ≥ 2.18, PLAN_v2 assumed 1.5–2× slower cores; measured 7.4/7.7/10.2× (`ruler-receipt.json` per_mix; `bl-l2/v2/plan.json` `basis: MAC_Y1_X2_SIZING_ONLY`, `timeout_minutes 1131.9`, `liability_plus_reserve_units 341.9`). Concurrency (12 procs on 12 vCPU) confounds the per-core ratio (04), but the admission number is what it is. | **Direction yes** (`2026-10-17_colab_slower.md:17-19`: 3.7× actor slowdown), **magnitude no**; a one-batch L4 ruler cost ≈ 0.4–1 u incl. setup (04's counterfactual). |
| F9 | E3 = KILL (geometry LB90 −5.41, 56 % unnecessary switches; history −18.2 %) | **World/science** | Clean registered negative on equal-capacity ranking students (`fit_report.json`; one world, 239, carries +398.7 of B's regret gain — 06's fragility arithmetic verified). Three failed recoveries before it are F10. | **No.** |
| F10 | Nine mechanical Colab failures + lost capacity: D0 sidecars; E3 archive upload; E4 checkpoint upload; Y1 fixture-dir upload after "Z-9 fixed"; E1b CUDA bool; E3 exec-poll 51 s; E3 Mac↔x86 drift (known since 10-08); Y1 list-vs-map manifest; **Y3 origin pause-boundary (new, 18:12)**; concurrency script crash (2.5 h idle); X5 blocked by K-1 audit in the "exclusive" window | **Process/ops** | No new manifest family ever went through a real-transport, real-CUDA, real-export dry run before science; schema validation did not run the accounting validator (`accounting.py:88-100` vs `supervisor.validate`); supervisor masks child exit as "foreign generation" (`supervisor.py:1457-1458`); Z-9 fixed checkpoints only; no Mac exclusion lock; ad-hoc shell on a critical-path slot (`colab-limit-test/test2.sh`). | **Yes** for schema, sidecars, upload bytes×count at 44 kB/s, drift, lease; **partly** for CUDA dtype and poll timeout (needed one real run — which is exactly what a canary is). |

Tally: world/science 3 (F1, F2, F9); planning 5 (F4–F8); process/ops 2 (F3, F10 — F10 is nine incidents). Measurable cost of
F10 (estimate, not a timesheet): ≈ 8–12 u of dead Colab time and ≈ 20–30 critical-path hours (five uploads, three reruns of E3, two
of Y1, one of D0, one of Y3, the 2.5-h gap). The governance machinery's cost in builder/review hours is **unmeasured**; its value
is five clean negatives and zero laundered claims.

## 3. Verdict on PLAN_v2 — on plan? right plan? what changes Thu–Tue

**On plan: no.** By its own dates: D1 (Thu 12:00, L2′ dispatchable) cannot be met — L2′ is NOT_DISPATCHABLE on L4 at any lawful
envelope with the stock hook, and the Mac×2 forecast violates both the 12-h chunk cap and the 200-u envelope. D1′ (Thu 14:00, P1
funded) cannot be met — Y3 failed mechanically at 18:12 and the 1M qualification has not started. D3 (Fri 23:59, L2′ print) cannot
happen. X5 is NOT_READY. E3's KILL removes E14 and E16-G. Only D2 (K-1 timing, Thu 20:00–22:00) is still on schedule.

**Right plan: the questions yes, the calendar and the build commitments no.** M1 (offline frozen/mixed ceiling of exact search),
M2 (can the opponent-veto cost be removed exactly on this Mac) and P1 (one honest 100M PPO endpoint) are still the three facts
that make week 2 a decision rather than another NOT_READY. What was wrong: (i) promising three prints by Sat on unmeasured rates;
(ii) ordering L2′ before K-1 when K-1 decides where L2′ can run; (iii) funding WP-4 (12–16 builder-h) on K-1 alone when L2′'s value
is unknown; (iv) a P1 gate (46 h) looser than P1's own invoice (24 h).

**Re-sequencing Thu Oct 22 → Tue Oct 27 (bars, bank, one look, KEEP_B8 text unchanged):**

| When (PT) | Action | Rule / receipt |
|---|---|---|
| Thu 09:00–12:00 | **Y3 fix** (Sol-R → Codex-V): origin snapshot must satisfy `_require_pause(origin, pause=0)`; Mac preflight executes the *same* function on the real origin artifact; new prepared attempt `y3/source-03` inside the remaining Y3 envelope (8 − 1.6 u). Write D1 slip file for L2′ (`stage0/slips/l2-D1.json`): NOT_DISPATCHABLE(rate). Release slots B/C; no placeholder. | `exp/bl-p1/y3/source-03/receipt.json`; dispatch only through the canary gate (§4.1) |
| Thu 14:00 | D1′ **slips to Fri 12:00**: Y3 PASS ∧ 1M qualification complete-job forecast **≤ 24 h nominal at 2 u/h** (registration invoice; the 46-h clause is subsumed) ∧ critic gate ⇒ tranche Fri night. Any fail ⇒ P1 NOT_READY; no shrink, no seed/LR change. | `REGISTRATION_P1.md:106-111` governs |
| Thu 20:00–22:00 | **K-1 timing** under a Mac lease (nothing else on the Mac; the identity audit is done). Report hook time, whole-batch time, native/fallback row counts, load log. D2 ruling by Fable Thu 22:30. | `exp/bl-k1/timing.json`; < 5× on `opponent_v8` frozen-1500 or any mismatch ⇒ exact served search CLOSED for week 2 (unchanged) |
| Fri 09:00–11:00 | **X5** bare strict timing under the lease (2 unpaced Mac-h). | `exp/week1-20261021/mac/x5.json` |
| Fri 12:00–20:00 (only if D2 PASS) | Sol-S enters K-1(a) into `ruler_operator` (`native_mode` ≠ stock in the timed path, counters attested) → **Mac whole-workload Y1 rerun** on the 9 fixtures (1–2 Mac-h, lease) and **one L4 native ruler ≤ 4 u** through the canary gate. Never multiply the old aggregate by hook speed. | `exp/bl-l2/y1-native-{mac,l4}/ruler-receipt.json` |
| Sat 12:00 | **L2′ placement ruling** (Fable): (a) L4 lawful — native R ≥ 2.18 ∧ 8 chunks each `2×est ≤ 12 h` ∧ ≤ 200 u ⇒ dispatch Sat on B+C, print Mon/Tue; (b) else a **prospective Mac-placement amendment** (same bank, analyzer, one look; two workers) only if measured native forecast ≤ 60 wall-h incl. 20 % reserve and no paced work is displaced ⇒ start Sat night, print Mon/Tue; (c) else **L2′ NOT_READY, deferred intact to week 2** — never a smaller bank, partial look or stock-hook L4 dispatch. | `exp/bl-l2/placement-ruling.json` |
| Sat 16:00 | Handoff `INDEX_v2.json`: K-1 verdict, L2′ placement status (possibly UNKNOWN), P1 readiness/progress, X5 rate, E3 closure, actual units; text B only; the one owner question (PLAN_v2 §7) asked with those facts. | — |
| Sun–Tue | Finish whatever was admitted (L2′ chunks, P1 100M + 128/mix eval). No new engineering. | — |
| Tue 23:00 | PLAN §7 reducer; **KEEP_B8 §6 applied literally**: clause 2 needs *measured* K-1 < 5×; missing timing is NOT_READY, missing L2′ is NOT_READY(L2′) not PACKAGE_KILL. Under NOT_READY nothing is funded for serving/engineering either (PLAN_v2 §5 funds only on FUND ∧ PASS) — science preserved, spend stopped. | — |

**Stopped now:** E14 (E3 KILL), E16-G, WP-4 build (suspended until L2′ has a print, not merely K-1 PASS), O2, Round 0
re-registration, E1b, Claim A descendants, second P1 seed, A100/H100 (nothing is GPU-bound — unanimous for the 17th time).

**Why not stop the campaign tonight (09's case):** the two receipts that decide week 2 cost 0 u + 2 Mac-h (K-1) and ≤ 12 u (Y3 +
1M), and P1's eval is the one receipt 09 itself says would change its view. What 09 wins: no promises of prints, no new build
packages, and the Sat/Tue reports say "unmeasured" where that is the truth.

## 4. Process changes to adopt now (owner → enforcement)

1. **Target-platform canary gate before any science dispatch of a new manifest family** — real transport at real bytes *and file
   count* (≤ 50 % of the 300-s cap), one CUDA decision on L4 through the real entrypoint (all dtype branches incl. `_paint`),
   export sidecars written by the real producer and recovered through the supervisor, verified release. Owner: **Sol** builds a
   `colab_cli` canary mode; **Codex-V** reviews; **Claude** runs it. Enforcement: `supervisor.validate` refuses a dispatch whose
   manifest family lacks a canary receipt sha-bound to (family, source commit); shared-supervisor edits trigger requalification
   (`QUALIFICATION.md:23-32`, ≈ 1.5 u in a 4-u envelope). Cost ≈ 0.5–1 L4-h / 1–2 u per family. Would have intercepted F4 (dtype),
   F10 a–e, h, i and the Y3 origin defect.
2. **Measure-before-plan (premise register).** Any plan date, bar or envelope that rests on a rate, latency, power or transport
   number must cite a receipt measured on the target platform with the exact code path, or be written as NOT_READY(name); no
   dependent build is authorized on an unmeasured premise. Owner: **Fable judge** (plan author) and **Claude** (operator).
   Enforcement: every PLAN/registration carries a premise table (claim | receipt path | platform | code path | falsifier ≤ 1 u);
   the judge refuses to date a milestone whose receipt cell is empty; QUEUE marks descendants of empty cells "blocked". Would have
   intercepted F5, F6, F7, F8 and both plans' calendars.
3. **Execution attestation, never metadata.** A speed or identity claim is valid only from a receipt that counts the executed path
   (native vs fallback rows, device kernels run, workers actually active). Owner: **Sol** (operators/rulers export counters),
   **Codex-V** (review). Enforcement: ruler/timing receipts without `native_v8.calls > 0 ∧ fallback == 0` (or the equivalent) are
   INVALID for native claims; `ruler_operator` default `native_mode="stock"` must be an explicit, attested choice. Intercepts F7.
4. **One schema validator, run before allocation.** The Mac preflight invokes the same accounting validator the supervisor uses
   (`_manifest_rates`, nested budget fields) plus a required-export inventory check (`.complete` + `.sha256` producer contract,
   `runtime.py:284-306`). Owner: **Sol**; enforcement: `colab prepare` refuses without a validation sidecar; CI fixtures for
   list-vs-map and missing-sidecar cases. Intercepts F10 a, h.
5. **Distinct error classes in the supervisor.** Separate "child nonzero exit" (carry the last traceback line into the receipt)
   from "foreign generation" and "RPC timeout"; recovery policy keyed on class. Owner: **Sol**; enforcement: receipt schema field
   `failure_class` required; ambiguous class ⇒ receipt refused. Intercepts the misdiagnosis in F10 g and Y3 (18:12).
6. **Power/size sheet before builder assignment** for any screen, bank or evaluation: estimand, SDs (receipt), N per mix,
   power at the planning alternative, the *complete* pass event incl. guards, expected runtime-hours. Owner: **Claude** drafts,
   **Fable judge** signs; enforcement: registration must link the sheet; Sol gets no build ticket without it. Intercepts F5, F6.
7. **Mac exclusive lease.** One lock file (`exp/mac-lease.json`: owner, purpose, until); every timing, identity-audit or paced job
   acquires it and refuses to start while another holds it; receipts record the lease id. Owner: **Claude** (operator);
   **Sol** writes the 20-line helper. Enforcement: launch scripts and operators check the lease. Intercepts X5/K-1 collision and
   the L1′ quiet-host ambiguity.
8. **Pre-claim published-ledger check.** Every registered analysis entrypoint runs a read-only "local ledger == pushed ledger"
   check and refuses *before* consuming the look; the integrity ruling asked for this on 10-21 00:40 — implement it. Owner:
   **Sol**; enforcement: in the analyze CLI; no analysis receipt without the check's hash. Intercepts F3.

Also binding: no ad-hoc shell on an account slot or the Mac critical path — probes go through the supervisor with a cap and a
receipt (the 2.5-h gap); and a mechanical failure's recovery attempt counts against the same envelope (already PLAN_v2 §8.2).

## 5. Next five actions (owners)

1. **Claude (tonight):** append to `LIVE_FACTS.md` and correct `CAMPAIGN_LOG.md:109`: Y3 source-02 FAILED 18:12 PT (origin pause
   boundary, ≈ 1.6 u, masked as foreign generation); X5 still NOT_READY (no rerun receipt); L2′ D1 slip (NOT_DISPATCHABLE: rate);
   E14/E16-G/WP-4 stopped. Write the Thu–Tue table above into `PLAN_v2.md` as a §9 "round-12 re-sequencing" (text, not a new plan).
2. **Sol-R → Codex-V (Thu 12:00):** fix the P1 origin snapshot / `_require_pause` defect; Mac preflight must execute the same
   check on the real origin; prepare `y3/source-03` through the canary gate; then 1M qualification with the ≤ 24-h nominal invoice.
3. **Sol-K + Claude (Thu 20:00–22:00), Fable (22:30):** K-1 timing under the Mac lease with hook/whole-batch times and attested
   native/fallback counts; D2 ruling.
4. **Sol-S → Codex-V (Fri, only on D2 PASS):** native-enabled ruler operator in the timed path; Mac whole-workload Y1 rerun; one
   L4 native ruler ≤ 4 u via the canary; **Fable** L2′ placement ruling Sat 12:00 (L4 lawful / Mac amendment / NOT_READY).
5. **Sol (colab_cli) → Codex-V (Fri 12:00):** canary mode, shared schema validator, failure classes, Mac lease helper, pre-claim
   ledger check; supervisor requalification (≈ 1.5 u) before the next science dispatch. **Claude:** X5 Fri 09:00–11:00 under the lease.

## 6. Per-advisor verdicts

| ID | Persona | Verdict | Why (verified facts in **bold**) |
|---|---|---|---|
| 01 | Blameless post-mortem | **keep** | Three recurring latent conditions (boundary coverage, unmeasured feasibility dependencies, manual state/resource coordination) adopted as the frame; **caught the wrong E4 pointer**; "F10 is not an independent tenth replicate" — right. |
| 02 | Research-methods critic | **keep** | Premise ledger is the best artifact of the round; **O2's own prereg disclosed inadequate power (`PREREGISTRATION_O2.md:114-134`)**; "Y1 R ≥ 2.18 was a required rate scheduled as if it would pass" — exactly right. Premise register adopted (§4.2). |
| 03 | Colab ops/SRE | **keep — the round's catch** | **"Foreign generation" masks a nonzero child exit (`supervisor.py:1457-1458`)** — confirmed again tonight by Y3; **Y1's tiny `.sha256` death = cumulative 300-s exhaustion**; costed gates (0.75–1 L4-h) adopted (§4.1, 4.4, 4.5). |
| 04 | Performance engineer | **keep** | **Amdahl at p = .78–.90: 5× hook ⇒ 2.66–3.57× batch; frozen-1500 ⇒ ≈ 5.9 s, residual floor 2.09 s**; 12-proc concurrency confounds the "per-core" ratio; day-one one-batch counterfactual ≈ 0.4–1 u; **`ruler_operator` default stock**. |
| 05 | Statistician | **keep** | **P1 invoice ⇒ ≤ 24-h nominal (`REGISTRATION_P1.md:106-111`)**; L2′ FUND ≈ .20 / .52 / .80 at +40/+60/+80 corroborated; "+40 passes point ≥ 40 only ~50 %; nested, not multiplied" — right. Its 92.6 % vs round-11's 86 % is an LB90 definition difference (1.28 vs 1.645 SE), not load-bearing. |
| 06 | RL researcher | **partly** | **E3's mean gain rides on world 239 (+398.7; other 31 average −0.02)**; D0/E3 are intervention-specific negatives; "P1's 10 % is a judgment, not evidence" — accepted. Its claim that P1's collection estimate was "mistaken for complete training" overstates: PLAN_v2 §3 already separates them; the real defect is the 46-h vs 24-h mismatch. |
| 07 | Search researcher | **keep (conditional)** | **`rollout_all` serial; batched engine refuses stateful opponents** (unchanged); "L2′ is an operational ceiling of one dense controller, not a bound on every sparse controller" — right and recorded; Mac-placement ≤ 72 h proposal narrowed to ≤ 60 wall-h incl. reserve and made conditional on a measured native forecast (§3). |
| 08 | Project manager | **keep** | **PLAN_v2 §2/§3 Y3–1M inconsistency; CAMPAIGN_LOG:109 overstates receipts** — both verified (and worse than 08 knew: Y3 then failed). Its Thu–Tue table is the skeleton of §3 with dates shifted for the Y3 failure. |
| 09 | Red team: stop now | **partly** | Right that Tuesday's *serving* disposition is invariant and that no print should be promised; right on the one receipt that would change its view (P1 eval at bar). Overruled on closing tonight: K-1 (0 u) and Y3/1M (≤ 12 u) are the cheapest decision-bearing receipts left and the owner asked for the credits to be used. |
| 10 | Builder-quality reviewer | **keep** | **139/248 non-merge commits touched no runtime source; narrow scaffolding proxy ≈ 14 % of added Python** — honest refusal to overstate the "mostly scaffolding" claim; **Codex-V caught the P1 sidecar rename pre-launch**; "review must witness the execution boundary" is §4.1's rationale; `ruler_operator` stock default. |

Factual errors that change a conclusion: none among the ten. Shared blind spot: none opened `exp/bl-p1/y3/source-02/run.log`
(modified 18:12, after their snapshot) — the ninth mechanical failure was invisible to the round.

## 7. Disagreements ruled

1. **Stop now (09) vs continue with K-1 + conditional P1 + L2′ feasibility (01–08, 10).** Continue, with 09's constraints: no
   promised prints, no new build packages, "unmeasured" reported as such.
2. **L2′ on the Mac (07) vs conditional/deferred (02, 05, 06, 08).** Conditional: only on a measured native whole-workload forecast
   ≤ 60 wall-h at two workers, by prospective amendment, after X5 and K-1 have their Mac windows; otherwise deferred intact.
3. **WP-4 build on K-1 PASS (PLAN_v2) vs suspend until L2′ (01, 06, 07, 10).** Suspend. A 12–16-h build on a mechanism whose
   frozen/mixed value is unknown repeats the week's defining error.
4. **E4 receipt and P1 sizing.** The `z9-e4-recovery-01` receipt governs; the ≤ 24-h nominal invoice governs over the 46-h gate.
5. **"Missing L2′ = KEEP_B8" (implicit in PLAN_v2 §6 clause 1 wording) vs literal reading (05, 07, 08, 09, 10).** Literal: clause 1
   requires a *print*; clause 2 requires *measured* K-1 < 5×. NOT_READY is a third state that funds nothing.

## 8. What is still unresolved

K-1 timing (Thu night); native whole-workload rate on the Mac and on L4 (Fri); L2′ placement (Sat 12:00); P1 Y3 fix → 1M complete-job
forecast vs the 24-h invoice (Fri 12:00); X5 episodes/h (Fri AM); actual builder/review hours (unmeasured — start a per-WP time
log Thu); whether the canary gate's requalification lands before the next science dispatch (Fri 12:00).
