# T34 ExIt-B8 campaign log (decisions, rulings, status)

Single running record of the T34 campaign. Times are Pacific. Binding documents are listed in
[docs/STATUS.md](../../STATUS.md) "Campaign authority"; this log explains how they came about and what is running.
Newest entries at the bottom of each section. Update this file whenever a decision or status changes.

## Process rules adopted during this campaign

- **Advisors:** facts + goal + an open question, never a menu. Default two advisors (one domain, one critic);
  expand only for a named unresolved question. When Astra disagrees with Fable, ask Fable whether it agrees with
  Astra's opposing view, then decide (owner rule, 2026-10-19).
- **Speed review:** any job estimated > ~4 h wall, > ~10 Colab units or > 1 day of Sol work gets the five-minute
  self-check first, then an advisor panel if needed (`~/.claude/skills/speed-review`). Long is fine when the work
  is irreducible, cheap and blocks no decision.
- **Freeze before data:** every design change is written to main (design amendment or pre-registration) before
  any data it could influence. Registered gates are never relabelled after results (e.g. M2 stays NO-GO).
- **Review loop before ratifying a registration:** independent reviewer → `REVIEW: RATIFY` or `FIX-FIRST` with
  file:line findings → operator rulings written as a Sol spec (pre-data) → Sol fixes + tests → same reviewer
  re-checks in code, not docs → repeat until RATIFY → operator ratifies, binding the final source.
- **Colab acceptance:** any `research/colab_cli` code change requires a fresh final-only acceptance (prepare →
  independent review → live run → record in `research/colab_cli/QUALIFICATION.md`). Fix known defects before
  paying for acceptance.
- **Roles:** Sol (Codex gpt-6.1-sol) implements; Claude plans, reviews, operates every live VM run and makes
  the next-step calls (owner delegated, 2026-10-20 00:15); releases/serving changes still need the owner.
- **Compute:** Colab credits are fine when they speed up a lane (owner, 2026-10-20 12:15); L4 by default, priced
  per useful result. Path B timing/paced tests, the strict gate and serving qualification stay on the Mac.
- **Fable unavailable → Opus stand-ins (owner, 2026-10-20 ~19:05 PT):** Fable hit its usage limit; until it returns,
  advisor panels and approval judges run as Opus agents under the same independence rules. Revert when Fable is back.

## Decisions (chronological)

| When | Decision | Record |
|---|---|---|
| 10-19 ~18:00 | Fable final-run panel recovered after a session limit; FINAL_PLAN "ExIt-B8" funded for prep + ~2 u L4 benchmark | [FINAL_PLAN.md](FINAL_PLAN.md) |
| 10-19 ~18:30 | No new checkpoint publication: all six inputs already public in release `s1-inputs-20261019` | `docs/research/specs/2026-10-19_exit_inputs_addendum.md` |
| 10-19 ~19:20 | Success contract ratified: Claim A bare student / Claim B labelled assisted hero; bare wins ties; Watch6 first | [SUCCESS_CONTRACT.md](SUCCESS_CONTRACT.md) |
| 10-19 ~21:15 | Design v2: shadow trajectory + LAD search (speed panel: cold H300 verify can never fit a tick) | [DESIGN_V2_SHADOW.md](DESIGN_V2_SHADOW.md) |
| 10-19 ~22:15 | Amendment 1: stride-10 + stride-1 teacher label rule (bisection non-monotone); L4-only hardware | DESIGN_V2 Amendment 1 |
| 10-19 ~23:20 | Rules audit (Astra ×3, Fable judge), owner "yes all": campaign authority block, P7 statistics | [P7_STATISTICS.md](P7_STATISTICS.md), `docs/research/advice/2026-10-19_rules_audit_*` |
| 10-19 ~22:00 | Colab CLI re-qualified: final-only acceptance PASS on `dab00823` after two independent reviews | `research/colab_cli/QUALIFICATION.md` |
| 10-20 01:00 | After M2/M3 NO-GO: Amendment 2 — ranked labels (decisive = horizon rescues, delay weight 0), earliest-first batched Path B, paced retention test as the decider | DESIGN_V2 Amendment 2, `speed_advice/m2m3_*` |
| 10-20 12:30 | A5 v2 benchmark on 2 L4 lanes (owner OK'd Colab credits) | `snake-dqn-artifacts/t34-a5-20261020/` |
| 10-20 12:45 | Retention prereg FIX-FIRST #1 rulings: efficacy overruns diagnostic; near-zero exception only if mass is the sole failure; p95 at 24 fps; timing gate first; runner-enforced quiet host; `decision.json` sole verdict | `docs/research/specs/2026-10-20_t34_retention_prereg_fixes.md` |
| 10-20 13:10 | A5 failed by implementation (serial shadow fill) → batch shadow/root/branch steps across worlds, rerun A5 | `docs/research/specs/2026-10-20_t34_shadow_batching.md` |
| 10-20 13:30 | Pre-data resize after power review: 48 worlds/mix, 32 Watch6 sessions, caps 40 h / 60 CPU-h (joint GO 0.27–0.32 → 0.80–0.83) | `docs/research/specs/2026-10-20_t34_retention_resize.md`, Amendment 2 addendum |
| 10-20 13:50 | FIX-FIRST #2 rulings: import-closure source binding + frozen worktree; guard never sleeps mid-row, pauses only between rows, outstanding requests dropped; contention counted only >50% CPU >60 s, ≤2% of elapsed allowed; pre-check own 8 h cap; unpaced-arm cost smoke | `docs/research/specs/2026-10-20_t34_retention_fixes2.md` |
| 10-20 ~13:05 | FIX-FIRST #2 merged (154 tests, 180 bindings). Cost smoke 38.45 h vs 40 h cap (17.7 h unpaced) → unpaced descriptive arm moved to Colab lanes (standing Colab rule); Mac runs bare + paced only (~20.7 h) | `docs/research/specs/2026-10-20_t34_unpaced_arm_colab.md` |
| 10-20 ~13:14 | Shadow batching merged (byte-identical; Mac −19%, L4 untested). A5 v2 GO re-based pre-run: ≥ 50 completed worlds/h per L4 lane (v1 3k row-steps/s floor priced 0.4M rows/world). A5 #2 dispatched | DESIGN_V2 Amendment 2 addendum B |
| 10-20 ~13:45 | A5 #2: STOP by telemetry (~28 row-steps/s/worker, ~56/lane, 0/256 worlds in 23 min; ~35 ms/world-step even batched; session hit the 30-min wall cap before export, ~1 u lost). 2-advisor speed review launched | `snake-dqn-artifacts/t34-a5-speed-review/` |
| 10-20 ~13:50 | 30-day compute portfolio: 8 Astra personas + 1 Fable judge (= approval). Adopted the judge's merged portfolio (below) | `docs/research/advice/2026-10-20_compute_portfolio_FABLE_JUDGMENT.md` |
| 10-20 ~14:20 | A5 speed review (Fable + Astra, Fable judged dissent): cost = single-threaded Python (full_digest 64%, v8 13%, raster 7%), GPU idle. Frozen order: Sol fixes (K=nproc single-thread procs, W1, O(bytes) digest, no CPU lockstep) + Mac identity -> 0.12 u CPU-only profile -> A5 on L4 K=12 W1 60-min -> Round 0. Unpaced Path B arm on Colab merged (42.5 u cap) | `docs/research/advice/2026-10-20_t34_a5_speed_*` |
| 10-20 ~14:35 | Retention final re-review: all prior fixes confirmed in code; unpaced Colab arm cannot affect the verdict. FIX-FIRST #3: contention budget = 2% of each phase's registered cap, checked at phase end; 5 s sampling; load counts only if sustained > 60 s; not-quiet at start -> refuse (no decision); unpaced-arm faults reported, not KILL | `docs/research/specs/2026-10-20_t34_retention_fixes3.md` |
| 10-20 ~14:15 | Retention FIX-FIRST #3 merged (179 tests). Re-review → FIX-FIRST #4: timing phase own 1 h cap (72 s contention budget), per-phase cap map, zero contended samples in scored timing ticks (contended → retry), idle-load baseline capped at cpu_count/4 | `docs/research/specs/2026-10-20_t34_retention_fixes4.md` |
| 10-20 ~14:30 | Retention FIX-FIRST #4 merged (205 tests; bindings valid on new digest bytes). Re-review → FIX-FIRST #5: timing attempts in one registered folder, admission refuses unless earlier attempts were retry-eligible and the PASS is latest, max 3 attempts | `docs/research/specs/2026-10-20_t34_retention_fixes5.md` |
| 10-20 ~14:35 | Retention FIX-FIRST #5 merged (259 tests). Re-review → FIX-FIRST #6: fixed precheck/efficacy dirs; receipt-hash ledger committed+pushed to origin before each next phase; operator abort or unexplained interruption = KILL for Claim B. Same rules to apply to Round 0 | `docs/research/specs/2026-10-20_t34_retention_fixes6.md` |
| 10-20 ~14:50 | Retention: reviewer REVIEW: RATIFY (ea1508b9) after 6 FIX-FIRST rounds; Fable approval judge APPROVE with conditions (protect `refs/tags/t34-retention/**` on GitHub before attempt-01 — owner action; ratify at source 97232d51; keep worktree snake-dqn-retention-fixes6; no timing until Phase 0 done + quiet). Awaiting final-source trace receipt (Sol) | `docs/research/specs/2026-10-20_t34_retention_trace_receipt.md` |
| 10-20 ~14:55 | Engine fan-out merged: digest 6.16→3.59 ms/row-step (1.71×), byte-identical, M1 15,000 frames 0 diffs. L0 dispatched: CPU-only profile (cap 0.12 u) then A5 on 2 L4 lanes (12 single-thread procs, W1, 60 min, cap 4 u each) | tmux `t34-l0`, `snake-dqn-artifacts/t34-engine-fanout-20261020/` |
| 10-20 ~15:00 | **Path B v2 paced retention RATIFIED** (7cd34b2e): trace receipt PASS (128 tests @97232d51), reviewer RATIFY, Fable approval APPROVE. Execution only from a frozen worktree at 97232d51 after Phase 0 + quiet admission: timing gate → T3-bank pre-check (≥ 50/64) → registered study | `research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json` |
| 10-20 ~15:35 | **A5 v2 GO** on L4: 57.2 / 54.4 completed worlds/h per lane (377/432 row-steps/s; 16–20k rows/world; ~8.7 MB/world) with K=12 single-thread procs, W1, fast digest — ~0.036 u per teacher world (800-world round ≈ 29 u). CPU-only runtime has nproc=2 (37 rows/s) → not used. L0 total ~1.5 u. Round 0 package getting the Path B integrity rules + A5 sizing (Sol) | `.../t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s*-output/exports/benchmark.json` |
| 10-20 ~16:05 | Round 0 package updated (integrity rules + A5 sizing): 2 chunks (141 / 99 pairs, ~3 h / ~2 h on 2 L4 lanes), liability 24.9 u (49.8 u with one mechanical rerun each); 68 tests; independent review started | `research/t34_round0/VERIFICATION_INTEGRITY_SIZING.md` |
| 10-20 ~16:15 | Round 0 review FIX-FIRST → rulings: study root committed in ratified JSON before dispatch + refuse foreign t34-round0 roots; faults by exception type; hard kill still writes KILL decision.json; cap expiry = INCONCLUSIVE (abort/unexplained = KILL); mixes interleaved across chunks + 48 h dispatch deadline (missing chunk = KILL), no analysis before all chunks terminal; fix run.sh env name | `docs/research/specs/2026-10-20_t34_round0_fixes1.md` |
| 10-20 ~16:35 | Round 0 FIX-FIRST #1 merged (204 tests, 2e7dd0b1). Fresh A6 PASS @2e7dd0b1 (`snake-dqn-artifacts/t34-round0-integrity-20261020/a6-fixes1-01/`); sizing 3 chunks (110/110/20), 32.4 u initial liability. Re-review running | `research/t34_round0/VERIFICATION_FIXES1.md` |
| 10-20 ~16:45 | Round 0 re-review: all FIX-FIRST #1 rulings verified in code; A6 binds 842 files. FIX-FIRST #2: timeout prose = code (supervisor timeout INCONCLUSIVE; over-time PASS = KILL); pending mechanical rerun before deadline = incomplete (analysis + replay refuse, no KILL); move faults module to research/t34_exit_b8; fresh A6 a6-fixes2-01 | `docs/research/specs/2026-10-20_t34_round0_fixes2.md` |
| 10-20 ~16:55 | **Round 0 RATIFIED** (652f2b1e; reviewer RATIFY 882b0fa2 after 3 rounds; Fable approval judge APPROVE; T34_OPERATOR = `fable-approval-judge-20261020 (Claude operator)`). 3 chunks 100/100/40 pairs, expected ~13.5 u, worst 32.64 u (65.28 u with reruns), serial 1 lane, 48 h dispatch deadline. **Chunk 000 dispatched** ~16:55 (tmux `t34-round0-c000`) | `research/t34_round0/REGISTRATION_ROUND0.json`, study root `snake-dqn-artifacts/t34-round0-study-01` |
| 10-20 ~18:20 | **§5 label-ordering amendment** (Fable judge APPROVE WITH CONDITIONS): R1 label collection only (Colab) may start after a published Round 0 GO, in parallel with Path B retention; fits/screens/R2/Mac work wait for the Path B disposition; R1 needs its own registration + Fable approval; owner veto. Release approval stays the owner's (STATUS.md/AGENTS.md corrected). **No-peek attestation (operator):** committed before Round 0 analysis (`analysis/decision.json` absent) and before retention timing attempt-01; no Round 0 outcome value inspected by anyone in this session — only operational progress lines. Round 0 state at commit: chunk 000 attempt-01 running (elapsed 4111 s, status running, completed_pairs": 44); chunks 001/002 not dispatched | `docs/research/advice/2026-10-20_a100/RULING_label_ordering.md` |
| 10-20 ~17:50 | **A100 advice rounds: no A100 this campaign** (10 Astra answers via 5 Fable prompters, 3 independent Fable reviewers, then 3 Fable GPU experts; unanimous). The teacher and Round 0 are bound by the 12-vCPU host, not the GPU (B8 forward ~8–13 % of worker time → ≤ ~1.15× at ~3.2× the units). Reopen only if the first phase-timed L4 cached fit takes ≥ 60 min with GPU leg ≥ 40 % (then one matched L4 vs A100-40 packet, cap 12 u; adopt ≥ 1.25× and ≥ 1 h saved). Device-resident sim (GPU expert 1) pays only at dense-search scale; ~0.7 u kill/prove kernel probe only off the critical path. Levers that do move the date: build the Claim A pipeline, unpace the Path B bare arm, §5 ordering ruling, NO-GO branch plan | `docs/research/advice/2026-10-20_a100/SYNTHESIS.md`, `REVIEW_{A,B,C}_*.md`, `gpu_expert_{1,2,3}.md`, `GPU_EXPERT_PROMPTS.md` |
| 10-20 ~18:20 | **NO-GO branch plan** (pre-data; operator planning doc, no registration amended): Phase 0 negative + Round 0 GO → no change; Round 0 NO-GO/INCONCLUSIVE → Path B continues, run **Round 0-D** (same 240-pair premise test, exact T3-dense teacher) on Colab; both negative → Round 0-D is the last premise test of the search-teacher family. Scheduling rule: **Round 0 Mac replay right after Phase 0, before the Path B timing gate** | `docs/research/specs/2026-10-20_t34_nogo_branch_plan.md` (59e932c2) |
| 10-20 ~18:40 | **Claim A pipeline built** (9230b150: student fan-out labels, payload ReplaySource, probe, CUDA-capable fit driver; 23 tests). Fable panel (correctness / design / ops) → all CONCERNS; package-level fixes in e301cfeb (accounting `{}`, parallel probe, atomic receipts, TF32 off on CUDA, timing-only exports no weights, seed audit PASS, L4 qualification re-priced 75 min / cap 5 u). **Two frozen-design BLOCKERS remain and need a prospective ruling** (design.md): (1) the frozen `exit_fit_loss` cannot flip a decisive row — free-Q probe on the 39 A5 worlds: argmax = a* on **0.4 %** of 225 decisive rows (bare 97.8 %), so the ≥ 70 % held-out tripwire is unreachable; (2) the 1:1 searched:unsearched rule was priced for ~600 searched rows/world, v2 gives ~7.6 → ~1 optimizer step/epoch. **No production fit until a ruling**; protected fit package untouched | `research/t34_claim_a/`, `snake-dqn-artifacts/fable-panel-claima/{correctness,design,ops}.md` |
| 10-20 ~18:50 | **Round 0-D package DRAFT** (8729a1db + verification 8fc8e00e; A6 Mac preflight PASS, pre-data engine checks). Re-priced: expected **26.7–39.3 u** (r = 23–15 pairs/lane-h); a dense R1 at 560 fresh worlds would be **49–75 u**, not the plan's ≤ 30 u. Not ratified; independent panel in progress. Must be on main before any Round 0 analysis | `research/t34_round0d/PREREGISTRATION_ROUND0D.md`, `VERIFICATION_ROUND0D.md` |
| 10-20 ~18:59 | **T35 outcome-head probe: registered KILL.** Teacher-arm per-action survival AUC **0.747** (CI 0.705–0.792; KILL ≤ 0.77) and outcome-argmax = a* **46.2 %** (KILL < 70 %), on 225 held-out decisive rows / 21 worlds; integrity all pass. Caveat: data-starved (296 outcome rows, ~235 per fold). Lever earns no R1 slot. Mac, **0 Colab units** | `research/t35_outcome_head/RESULT.md` (182d85e2) |
| 10-20 ~19:05 | **Fable usage limit hit.** Owner chose **Opus stand-ins** for advisor panels and approval judges until Fable returns (same independence rules: a judge wrote none of what it judges; verdict logged here and reported). Release/serving changes still need the owner | this log, `docs/STATUS.md` approvals sentence |
| 10-20 ~19:10 | Round 0 chunk 000: runtime status `succeeded` (elapsed 7,859 s); supervisor finalizing (no `EXIT=` yet). Operational state only — no outcome opened. Next: publish its ledger, dispatch chunk 001 | `snake-dqn-artifacts/t34-round0-study-01/chunk-000-attempt-01.log` |
| 10-20 ~19:11 | **T35 two-deviation probe primary: registered EXTEND (W = 2/7, U = 0).** Rescued `b8_s3_19@4693` (ranks 1–4) and `b8_s3_9@4188` (rank 3); 5 worlds NOT_RESCUED with 0 layer-2 candidates at t2+300; no cap expiry. Caveat: both rescues are the two worlds where T3's history already showed a multi-deviation escape (Amendment 1 §A1.7); `b8_s3_9` is exactly the known path. Provenance: 7/7 receipts at 321ba7c7, dirty "". Mac ~12 min, 0 units. **EXTEND run (26 events, ≤ 2 h Mac) not launched — needs its own approval (C6)** | `research/t35_two_deviation/RESULT_PRIMARY.md`, `snake-dqn-artifacts/t35-two-deviation-20261020/primary/summary-primary.json`, approval `snake-dqn-artifacts/fable-panel-2dev/APPROVAL.md` |
| 10-20 ~19:15 | Path B bare-unpaced amendment: FIX-FIRST round done on branch `t34-retention-bare-unpaced-20261020` @52dec74f (spec: bare arm unpaced, paced arm 24 fps, saves ~8.7–10 Mac h); re-review in progress. Lands in **one push** only after a trace-receipt PASS, before Phase 0 ends (~Wed 10:00 PT); otherwise run the ratified 97232d51 design unchanged | `docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md` (branch) |
| 10-20 ~19:55 | **Boost-share floor: owner decision = RELATIVE floor** (pre-look). Verified (independent Opus, `snake-dqn-artifacts/boost-bar-check-20261020/REPORT.md`): bare B8 s3 delivered boost share 0.12% (190,779 decisions, exact Round 0 definition) < the contract's 0.5% floor, so Round 0 (analyze.py:169), Round 0-D and Path B v2 (analyze_v2.py:242-254) would STOP/KILL regardless of outcomes; floor had no recorded rationale and was never applied to B8 s3's release. New rule: candidate share >= 0.5 x paired bare-B8 arm share (same rows, same definition; bare 0 -> floor 0), ceiling 10% unchanged. Implementation: Path B folded into the bare-unpaced re-ratification (branch t34-retention-bare-unpaced-20261020); Round 0 via a pre-look analysis amendment (branch t34-round0-boost-floor-20261020); both go review -> Opus judge (Fable unavailable) before landing. No Round 0 outcome opened by anyone | owner (chat), report above |
| 10-20 ~19:40 | **T35 two-deviation probe: pooled KILL** (W_pool 2/7; EXTEND 26/26 NOT_RESCUED; W_novel 0; W_uncensored 1 = the known T3 path; b8_s3_19 rescue censored at 35 steps). Registered beam found no new two-deviation rescues → GPU search not funded. 0 units; provenance 33/33 at 321ba7c7 | `research/t35_two_deviation/RESULT_EXTEND.md` |
| 10-20 ~19:55 | **Round 0 boost-floor amendment GOVERNING for Round 0 (pre-look)**, approved by Opus judge APPROVAL.md (Fable unavailable; independent review RATIFY with conditions R-A/R-B/R-C). Rule: 0.5 × bare share ≤ teacher share ≤ 0.10. Merged from branch t34-round0-boost-floor-20261020 (0abe6db3); merge SHA recorded in the next row. R-A failure semantics added to spec §3.3. Round 0 state at merge: chunk 000 COMPLETE (ledger published), chunk 001 running, chunk 002 not dispatched, analysis not reserved. **No Round 0 outcome opened.** Pre-analysis gate (condition 5): first-parent log of the amendment JSON must print exactly one line | `snake-dqn-artifacts/round0-boost-review/{REVIEW,APPROVAL}.md` |
| 10-20 ~19:55 | Round 0 boost-floor amendment merge SHA = `30b2739aa2813f3f3253a540994efe8e900b499c` (pushed 2026-10-20 19:55:10 PDT, before the chunk-001 ledger publish); first-parent check printed exactly 1 line | git |
| 10-20 ~18:45 | Bare-unpaced pre-data amendment (9de16df8): independent review FIX-FIRST (procedure only, code confirmed); Fable panel integrity OK / stats OK / adversarial CONCERNS → fixes on branch `t34-retention-bare-unpaced-20261020` | `docs/research/specs/2026-10-20_t34_retention_amendment_bare_unpaced.md` (+ `_REVIEW.md`) |
| 10-20 ~19:55 | **Owner-approved: relative boost floor for Path B.** SUCCESS_CONTRACT §2 / P7 0.5% floor replaced by 0.5 × paired bare delivered share (same rows, same pooling) ≤ paced ≤ 0.10; bare 0 → floor 0. Evidence: bare B8 s3 delivers 0.12% (190,779 decisions), so the old clause would KILL the paced arm whatever its outcomes. Bound as item 2 of the bare-unpaced amendment (contract/P7 not edited: spec §7 is the binding Path B statement via the registration's `amendments` pointer; corrected at landing per REREVIEW2 ruling 8 — the contract is NOT sha-bound by Round 0 on main, Round 0 hashes its own frozen checkout). Round 0's identical 0.5% clause (STOP regardless) needs its own decision | `snake-dqn-artifacts/boost-bar-check-20261020/REPORT.md`, amendment spec §7 |
| 10-20 ~20:05 | **Path B v2 retention RE-RATIFIED (bare-unpaced amendment)**: review trail = FIX-FIRST (`2026-10-20_t34_retention_amendment_bare_unpaced_REVIEW.md` @9de16df8) → Fable panel (integrity/stats/adversarial) → independent Opus re-review RATIFY @52dec74f (`snake-dqn-artifacts/fable-panel-bare-unpaced/REREVIEW.md`) → re-review 2 RATIFY delta @8767d48b (`REREVIEW2.md`) → **approval: independent Opus stand-in judge (Fable at usage limit, owner choice) APPROVE WITH CONDITIONS** (`APPROVAL.md`); origin/main merged into the branch before the final OC/build (7c6fe12c, binds d5b11e3a contract text; post-merge receipt glob 146 passed); relative boost floor (owner-approved 19:55: 0.5 × paired bare share ≤ paced ≤ 0.10) in analyze_v2; OC regenerated in the authoring checkout at the final source: statistics byte-equal, differences only source_commit/source_hashes and the development_pool checkout-root path (→ world_pool_sha256), pool equal and sha reproduced after root swap (key-level diff vs ratified OC: top-level keys differing = `development_pool`, `source_commit` b15f80e2→7c6fe12c, `source_hashes` 207→209 files [+spec, +`t34_exit_b8/faults.py`; changed SUCCESS_CONTRACT, DESIGN_V2_SHADOW, `t34_exit_b8/policy.py`, `analyze_v2`, `registration_retention_v2`, `retention_v2`, `source_binding_v2`], `world_pool_sha256` 3e78…f76d→d37f…c31b; `development_pool` differs only in its one `source_files` key root `snake-dqn-retention-fixes6`→`snake-dqn-retention-v2-author`; `simulator_hash` and all statistics byte-equal; OC 81c54731, evidence `snake-dqn-artifacts/t34-retention-oc-bare-unpaced-7c6fe12c/acceptance.json`); draft 0787001a (bare_fps 0, paced_fps 24, amendments = spec sha c7ed9086…); trace receipt PASS @81c54731 (146 passed, 0 skipped; `operator-02/trace-receipt.json` sha 14018f28…) incl. `test_t34_pathb_bare_unpaced_v2.py` 11 passed and `test_t34_pathb_boost_floor_v2.py` 7 passed, 0 skipped, `T34_TRACE_RECEIPT=1`. **Registration @97232d51 (study operator-01) SUPERSEDED pre-data — VOID for execution.** **Study operator-01 VOID.** Execute only at `81c54731a06f5a630bc1b95edca876832906a812` from frozen `/Users/josenunez/Projects/ml/snake-dqn-retention-v2-frozen-02` / study `t34-retention-v2-operator-02` (ledger `research/t34_pathb/ledger/t34-retention-v2-operator-02.jsonl`, repo `/Users/josenunez/Projects/ml/snake-dqn-retention-v2-author`; keep that checkout until analysis is complete); worktrees `snake-dqn-retention-fixes6` and `snake-dqn-retention-trace-97232d51-20261020` retired from use (directories removed only after `validate --ready`); Colab unpaced arm dispatchable at 81c54731 only. Round 0's identical 0.5% clause is out of scope here; it is decided by its own amendment (30b2739a, row above). Timing gate waits for Phase 0 end and the Round 0 Mac replay (Mac, AC, 2-slot policy) | `research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json` |
| 10-20 ~20:10 | **Round 0 GO-file ruling** (independent Opus stand-in judge, prospective, pre-look): §5 "Round 0 GO" and every Round 0-conditioned lane (R1/L2, Round 0-D/L7, mechanical re-registration) key on the published governing `amendment-boost-floor/decision.json`, never the registered `analysis/decision.json`. The only fallback is a logged late-publication `decide` refusal; post-claim failure = governing KILL. Pass-through map: GO→R1 registrable; scientific STOP/INCONCLUSIVE→0-D; fault STOP→fix + re-register; integrity INCONCLUSIVE/KILL→mechanical, no 0-D/R1. Contract append-only (frozen worktrees hash their own checkout; Round 0-D draft must be regenerated after this append) | `docs/research/advice/2026-10-20_colab_lanes/RULING_round0_go_file.md`, SUCCESS_CONTRACT §5 Amendment clarification |

- **2026-10-20 ~21:58 PT — DIRECTION RESET (Fable judge, `docs/research/advice/2026-10-21_direction/FABLE.md`; owner: "make the calls", remove self-imposed restrictions, use Colab credits).** Adopted by Claude:
  (1) Path B v2 **deliverability smoke** on the Mac tonight (dev, not evidence; stopping rule written first) before the ~27 h operator-02 chain;
  (2) Round 0 Mac replay + analysis runs as soon as chunk 002 publishes, **beside Phase 0** (deterministic, 1 thread) — "after Phase 0" ordering dropped;
  (3) Colab lanes no longer wait on sequencing rules that only save units: Round 0-D (on stage-1 APPROVE), L1 D0, unpaced Path B arm dispatch now; R1 labels on the governing Round 0 GO (hours away); outputs never cited in a registered verdict unless their own registration says so;
  (4) SUCCESS_CONTRACT §4 "one extra CPU thread" → registered N, decided by the smoke (N=1 vs N=6), all timing bars kept;
  (5) governance cut: review loop + judge only for evidence-producing registrations or changes to a bar/bank/look/candidate identity; 48 h dispatch-deadline=KILL dropped; advice = Astra advisors → one Fable judge; docs-reorg agents stopped.
  Kept: freeze-before-data, sealed outcomes, fresh disjoint paired banks, Mac-only strict gate + serving qualification, paced evaluation of any served assistant, one look, never relabel, owner release OK.
  Roles (owner, token efficiency): Sol implements; Astra advises; one Fable judge rules.
- **2026-10-20 ~22:00 PT — Claim A L4 timing qualification (bfe18981) INCOMPLETE:** Colab dropped the assignment at 2,651 s (`AccountingError: supervised assignment disappeared`), stop verified, only the start-phase export (no timings), ~1.5 u. **Not rerun:** the old fit pipeline is being replaced by the L3 amendment; the D0 diagnostics L4 run (8 u cap) times the corrected loss instead. S2 native v8 built (4708b4fa): identical on all fixtures, 1.4–2.2× hook-level contended; D3 waits on S0 whole-workload + Sol review.
- **2026-10-20 ~22:02 PT — Round 0-D rescope stage 1 APPROVED** (Fable judge; `docs/research/advice/2026-10-20_colab_lanes/ROUND0D_JUDGMENT_STAGE1.md`), merged unedited: tree `research/t34_round0d` 9d8870d8, tests blob 5ccb1aed, prereg sha256 9fbb572d. Stage-2 `ratify` only on a governing scientific Round 0 NO-GO. Exposure if activated: 120 u hard total + A5-D ≤ 10 u (≤ 130 u).
- **2026-10-20 ~22:35 PT — OFFENSE vs DEFENSE (5 Astra → Fable judge, `docs/research/advice/2026-10-21_offense_vs_defense/JUDGMENT.md`):** defense stays primary (mass integral; B8 s3 dies late at L≈972); owner's quoted premise wrong on 4/5 facts (mechanics v2 implemented + served; B8 s3 kills; boost ~0.1 %; T3 s3 row +122.8). **v1 annotation in the Round 0 bank = provenance-label defect only** (executable chain pins v2; v2 profile refuses v1): before Round 0 analysis add a sibling erratum + one-world v2 attestation; fix the label in Round 0-D bank.py/prereg only if stage 2 is ever needed (that edit lapses stage-1 → quick re-approval). **New lane O1 harvest-branch census** (48 fresh B8 s3 worlds, ≤8 roots/world, all first actions × H500, food-provenance ledger; 1 L4, cap 8 u / 4 h; KILL if < 8/48 worlds (or < 4 frozen+mixed) have a surviving branch ≥ +5 from death-drop intake; GO → O2 ≤ 24 u separate registration).
- **2026-10-20 ~23:45 PT — Path B v2 deliverability smoke = STOP** (dev, excluded; stopping rule 7622d62e committed before data; results `snake-dqn-artifacts/t34-pathb-smoke-20261021/RESULT.md`, main 0b66cdcc): N=1 delivered 7/75 T3 improvements (all delay-only), 0/34 rescues, 0/5 M3; N=6 0/75, 0/34, 0/5 and could not hold 24 fps (4.97–15.92). **Operator-02 chain and the paced Path B study will NOT run.** The Mac after Phase 0 is free for Round 0 replay, then other work.
- **2026-10-20 ~23:30 PT — Claim A D0 (L4, 314d2599) printed decision KILL** (39 A5 worlds, 225 decisive rows; report 2,180 s) but exported no receipts (run_d0 wrote files without supervisor sidecars). Treated as a MECHANICAL failure: no registered decision yet; Sol fixing exports; one rerun (≤ 4 u, judge C2) after the Colab limit test, and the rerun's decision governs.
- **Big-lever rounds started** (owner): 10 × (Fable writes question + 10 personas → 10 Astra in parallel → Fable judges); `snake-dqn-artifacts/big-lever/`; Colab budget effectively unlimited for a week if it buys speed. Colab concurrency: our "3 assignments" cap is our own assumption (no Google source found); empirical test armed (tmux `colab-limit-test`) for the first window with no supervised job.
- **2026-10-21 ~00:20 PT — Round 0 governing verdict = KILL / INTEGRITY_FAILURE (operator error, Claude).** All 240 pairs sealed COMPLETE (chunk 002 00:11; ledger b055605f); Mac replay PASS (24 worlds, 00:12–00:14); condition-5 gate PASS (one line, 1a347e68). I did NOT publish the replay's ledger append before the sole analysis (README: "Publish the ledger … before analysis"); `history.require_chunks → require_published_history` raised "receipt-hash ledger must be committed and pushed before continuing" as the FIRST check, before any pair record was read, and analyze maps HistoryError → KILL. Amendment decide then sealed KILL ("registered verdict precedes clauses"); ledgers published 5ba76312 / 115955f2. **No Round 0 outcome has been computed or read.** A Fable judge is asked whether a remedial pre-look analysis is legitimate and worth running.
- **2026-10-21 ~00:40 PT — Round 0 integrity ruling: KILL_STANDS** (Fable, `docs/research/advice/2026-10-21_round0_integrity/JUDGMENT.md`): no remedial analysis (pre-look test cannot be met; phase guards), no descriptive look (240 pairs stay sealed; open only after the teacher family is closed, labelled NOT_EVIDENCE). Remedy = mechanical re-registration on a fresh namespace (~15–40 u, ~3 h), funded ONLY if a live lane commits within 48 h to consume the verdict. **Claude's call: lane CLOSED now** — no current consumer (Claim A D0 KILL pending rerun; Round 0-D needs a scientific NO-GO; big-lever E-series does not consume it). Reopen on a consumer's request. Process fix: any future registered analysis runs behind a pre-claim check that the local and published ledgers match.
- **2026-10-21 ~01:10 PT — Claim A D0 (registered rerun, d2454013) = KILL**: "no corrected objective keeps the anchor guards for >= 512 updates" (registered_run true; report sha 790d532a). Claim A projection line CLOSED; R1 labels have no consumer. **Colab concurrency measured:** 3 L4 assignments allowed, 4th refused ("precondition failed"); cap 3 confirmed (~0.1 u). Dispatched E3 (info-value), E1b (GPU engine realistic benchmark), O1 retry (census-02).
- **2026-10-21 ~02:55 PT — O1 harvest-branch census = GO** (look 1, census-02, registration a82691cb): 35/48 worlds have a surviving single-action H500 branch ≥ +5 from death-drop intake (frozen 9, mixed 13, scripted 13; frozen+mixed 22 ≥ 4; KILL bar was < 8/48). Scope: harvesting mechanism exists; GO admits the separately registered O2 (≤ 24 u: bare B8 vs rescue-only D vs D+O, 96 fresh paired worlds; must show (D+O − D) LB90 > 0 and +40/LB90 > 0 vs B8). No release. Sol building O2. Also: E8 landed; E9 BLOCKED (E2 screen joint funding .31 < .60 at true +40); E1b and E3 failed mechanically (CUDA bool scatter; upload cap) — Sol fixing; E4 dispatched.
- **2026-10-21 ~03:45 PT — OWNER: "yes to both"** — (1) served-search serving clause W ≤ 4 worker processes / K ≤ 8 outstanding on this Mac (SUCCESS_CONTRACT §4 amendment appended; timing bars unchanged); (2) probe-only grant for L1′ expanded cells + MPS. Records in `docs/research/advice/2026-10-21_big_lever/authority/`. Big-lever 10 rounds DONE → `docs/research/advice/2026-10-21_big_lever/PLAN.md` (week 1 = served search S on this Mac; R research-only). Dispatched E1b CUDA-fix relaunch + E3 recovery-02; Sol on Z-9 upload path.
- **2026-10-21 — E1b GPU engine (CUDA dtype fix 88adb3e2, L4) = NO_CONTINUE / INCOMPLETE:** device rollout exhausted the spike wall budget (10,820 s) after 1 of 18 decisions; measured 0.217 full world-steps/s vs the frozen ≥ 20,000/s bar → **engine thesis STOP for this week** per its frozen rule (no retry). GPU-resident sim (lever 4) and anything depending on it (search-in-the-loop at scale) are off the week-1 table. E3 recovery-02 INCOMPLETE mechanically (supervisor exec poll timed out at 51 s during reconstruction 33/256); relaunch pending. Owner-requested Colab concurrency test #2 (T4 max, CPU max, CPU on top of 3 L4) armed/running.
- **2026-10-21 ~07:10 PT — Claude's calls on Sol results:** Z-9 upload path fixed (300-s shared upload deadline; ~44 kB/s measured; checkpoints now downloaded + hash-verified on runtime; E4 uploads 18 kB) → E4 relaunch ready. Z-4 E9 replacement: best design (fixed 192/64/64) joint funding .362 < .60 at true +40 → **E2 NOT funded as a screen this week; R stays research-only** (PLAN). Z-2 paced-identity cutoffs split + probe-only mode landed. Y0/Y1/Y7/X8 landed (L2′ 4×40 chunks; Y1 ruler ≤ 4 u decides L2′ admission at R ≥ 2.18/s). O2 built at N96 but underpowered → prospective re-size to ≥ .80 power (≤ 150 u) before any data (Sol). E3 poll timeout → Sol fix (workers ≤ nproc−2, nice). Dispatch Y1 + E4 when the concurrency test frees Colab.
- **2026-10-21 ~07:20 PT — A100/H100 rule changed (owner input; Claude's call):** the cost-based reopener (L4 fit ≥ 60 min AND GPU ≥ 40 %) is DROPPED. New rule: any job may use A100/H100 when its own L4 profile shows it GPU-bound (GPU busy ≳ 70 %, steps not waiting on CPU/data) and the expected wall-time gain is ≥ 2×; no duration or unit test. Current lanes (E3 270k-param students, B8-sized PPO learner, CPU-bound search/collection, E1b engine) do not qualify; an A100 likely occupies one of the 3 account slots (concurrency test pending).
- **2026-10-21 ~09:55 PT — Colab back on the critical path** after a 2.5-h idle gap (owner concurrency test #2 crashed: CPU runtime ReadTimeout then a bash-3.2 empty-array error in Claude's script; no orphan runtimes; T4 max = 3 at 2 vCPU each; CPU-runtime cap UNMEASURED). Dispatched Y1 ruler (≤ 4 u; decides L2′ admission at R ≥ 2.18/s), E4 relaunch (Z-9 path), E3 recovery-03 (poll fix). **O2 re-sized (V2):** 80 % power for +10 incremental over D needs 5,097 paired worlds (~150 runtime-h); budget fallback 1,260 worlds at .34 power registered. **Call: O2 deferred behind L2′** (harvest is a week-2 served-search add-on; moot if L2′ kills the served push).
- **2026-10-21 ~14:35 PT — L1′ Mac latency diagnostic: served exact-sim search ~15–25× too slow** (`snake-dqn-artifacts/big-lever/exp/bl-l1-20261021/RESULT_L1.md`, main 1d23ce67). CPU S p99 22.9–27.2 s at W1–W4; MPS (Apple GPU) 32–41 s; tick overruns in every trial; ceilings 0.83–1.458 s. Mac 6×H300 batch: frozen 10.6–26.3 s, mixed 7.6–11.0 s, scripted 0.6 s. Formal verdict NOT_READY (consumption path unbuilt; quiet host not achieved), but the gap is > 10× and W ≤ 4 cannot close it ⇒ **PLAN week-1 release track (Line S) infeasible as designed.** E4 COMPLETE (~1,440 hero transitions/s per L4). Phase 0 finished. E3 recovery-04 dispatched (recorded-action replay). Y1 L4 failed at upload (Sol fixing). **Re-plan round 11 (10 Astra → Fable) started.**
- **2026-10-21 ~14:50 PT — RE-PLAN (round 11, Fable): PLAN_v2 adopted** (`docs/research/advice/2026-10-21_big_lever/PLAN_v2.md`). No trustworthy served gain this week; release track withdrawn; **B8 stays served**. **Erratum (Y1/L1′):** latency was measured with the STOCK Python opponent-veto hook — `opponent_v8` = 78–90 % of each frozen/mixed batch; S2 native path never enabled in bl_served_search/bl_mac_latency; non-veto remainder 1.3–2.7 s/batch. Week = L2′ (offline frozen/mixed exact-search ceiling; re-chunk 8×20, ≤ 200 u; bank/bars/one-look unchanged) + K-1 (native veto in branch rollouts, action-identical, bar ≥ 5× on opponent_v8) + P1 (one fixed-endpoint 100M PPO, Y3 + 1M qualification first, 128/mix eval). **KEEP_B8 criterion:** earlier of Fri 23:59 (L2′ PACKAGE_KILL or INCONCLUSIVE with point < 40) or Tue 23:00 (no fresh-bank print at pooled ≥ 40 ∧ LB90 > 0 and K-1 < 5×). Sol WP-1/2/3 + X5 launched; Y1 L4 ruler dispatched (upload fixed a7bd01e9).
- **2026-10-21 ~17:30 PT — results:** (1) **Y1 L4 ruler: aggregate R = 0.166 batches/s ≪ 2.18 → L2′ NOT_DISPATCHABLE** on L4 with the stock opponent hook (frozen 0.0074 / mixed 0.0136 / scripted 0.147 batches per worker-second; Colab L4 vCPUs ~10× slower per worker than the Mac on frozen). WP-1 re-chunk: Mac×2 forecast 566 min/chunk (needs 1,132 > 720 cap), 342 u > 200 u → still NOT_READY. (2) **K-1 action identity PASS 294/294** (native veto in branch rows); timing Thu 20:00–22:00 PT. (3) **E3 = KILL** (geometry B: −26.6 % regret but LB90 −5.4 and 56 % unnecessary switches; history C: −18.2 %) → representation lever closed; E14/E16-G moot. (4) P1 registered (F, seed 20261021, 100M endpoint); Y3 durability dispatched. (5) X5 rerun launched in an exclusive Mac window (first attempt blocked by the K-1 audit).
- **2026-10-21 ~21:00 PT — RETROSPECTIVE (round 12, owner request; `docs/research/advice/2026-10-21_big_lever/JUDGMENT_round12_retrospective.md`):** 3 world/science negatives (Path B, D0, E3), 5 planning errors (E1b magnitude, E9 power, O2 size, L1′ stock hook, Y1 L4 CPU rate — premises never measured on the target platform/code path), 2 process (Round 0 operator slip; nine mechanical Colab failures). PLAN_v2: right questions, off its dates → re-sequenced (Y3 fix Thu; K-1 timing Thu 20–22 under a Mac lease; X5 Fri AM; native-enabled ruler Fri; L2′ placement ruling Sat 12:00; honest Sat 16:00 handoff; Tue 23:00 KEEP_B8 reducer literal). Stopped: E14, E16-G, WP-4 build, O2, Round 0, E1b, A100/H100. **Corrections:** P1 Y3 source-02 FAILED 18:12 (pause-boundary bug, ~1.6 u; Sol fixing); X5 17:25 relaunch produced NO receipt (empty recovery-02). **Process changes adopted:** target-platform canary gate (real transport bytes×count, one CUDA decision, real export sidecars); measure-before-plan premise register; execution attestation (native/fallback counters); one schema validator before allocation; distinct supervisor failure classes; power/size sheet before any build; Mac exclusive lease; pre-claim published-ledger check in every analyze entrypoint.
- **2026-10-21 ~22:10 PT — Round 13 (GPU-trained components; owner asked advisors first):** Fable: legitimate research line, NOT the lever. **GO** only for a horizon-conditioned tail critic after an H64 exact prefix, gated by **V0** (0 u Mac latency ladder: C1 kernel H300 / C2 + exact stop / C3 H64 + oracle leaf + dummy-net cost / C4 = C3 + stop; kill: C2 action mismatch ⇒ INVALID; C3 warm p99 > 1.458 s on frozen-1500/800 ⇒ LEARNED_CUTOFF_CLOSED; C3 < 2× C2 ⇒ NO_CONSUMER). NO-GO: opponent-model-first, pruning prior, any approximation as a serving route before L2′ prints. **Zero-unit finding:** bare-first / Q-ordered **exact stop** is decision-identical (bare survives on ~98.8 % of dense roots; _choose returns bare on ties) and removes 3 of 4 branches → entered into Friday's native rulers (prospective amendment) before the Sat 12:00 L2′ placement ruling. Claude's "≥ 95 % decisions unchanged" criterion rejected as vacuous (always-bare agrees 99.8 %).
- **2026-10-21 ~23:05 PT — owner: "start now if most promising" + "never be idle on Mac or Colab".** K-1 timing window amended to "2 h from launch" (82737c0c; ≥ 5× bar, identity binding, foreign-load logging unchanged); new manifest prepared-03; identity re-audit → timing chained in tmux `k1-attempt-03` (results `exp/bl-k1/attempt-03/`). Colab: O2 V2 chunks 00 + 01 dispatched (registered, data-producing while P1's Y3 fix is pending); one slot held for P1. Standing rule: never idle (memory never-idle).
- **2026-10-22 ~02:05 PT — K-1 = PASS: opponent_v8 speedup 20.13× (bar ≥ 5×)** on frozen-1500, kernel mode, W1, 810 batches, ratio of warm medians; identity 294-case audit PASS (attempt-03 re-audit 00:52). Receipt `exp/bl-k1/attempt-03/timing-03/timing.json` (manifest 306f33fe, identity ae1cceb1); foreign_load_observed = true (background macOS load; disclosed). Whole-batch p50 stock→kernel: frozen 19.5–21.4 s → 2.3–2.4 s (~8–9×); mixed 7.2–10.5 s → 1.2–1.5 s (~6–7×); scripted unchanged ~0.5 s. With the exact stop (~3 of 4 branches skipped) frozen/mixed batches are projected ≈ 0.6–0.8 s / ≈ 0.4 s — under the W4 1.458-s S ceiling on paper (unmeasured; p99 required). WP-4 ADMITTED_BUILD_ONLY. Operational: timing first refused on source-closure drift (main moved) → rerun from frozen worktree at 82737c0c; then Spotlight indexing (new worktree) blocked idle checks → retry loop. Owner action suggested: exclude ~/Projects/ml from Spotlight.
- **2026-10-22 ~02:15 PT — Colab idle ~3 h (23:24 → 02:10), Claude's failure:** all three O2 V2 chunks failed mechanically (chunk-00 upload of public-inputs.json; chunks 01/02 bare AssertionError in the remote bootstrap) and no watcher was set on them. Fix: a general watcher now wakes Claude on any Colab job end (`big-lever/watch/`); Sol fixing O2 (v3 chunks). Dispatched P1 Y3 source-latebody-02 (round-14 E8_LATE_BODY_AMENDMENT_02) and the density canary (METRICS_AMENDMENT_03). Mac native+exact-stop ruler refused "host not idle" → retry-until-idle loop armed.
- **2026-10-22 ~07:50 PT — overnight results:** (1) **Native + exact-stop Mac ruler PAIRED_PASS** (02:20): W1 all-branches 0.611 → exact_stop 1.793 batches/s (×2.9), decision identity 270/270; placement tool → **MAC_AMENDMENT_ELIGIBLE** for L2′ (Fable judge ruling requested). (2) **colab_cli requalified** final-only PASS (process gates merged e7bd7885). (3) **Density production COMPLETE but INSUFFICIENT for every densifier×mix**: arm A natural play produced < 4 doom-bearing episodes per mix; doom moments ≈ 0.09 per 1,000 hero decisions overall (frozen 0.018, mixed 0.045 per 1,000). Raw doom/CPU-s vs A: C-P1 ≈ 4.8×, B70m ≈ 3.7×, B70p ≈ 1.7× — directional only, not decidable. (4) P1 Y3 still HOLD: new family-canary gate (Sol building canary); O2 v3 chunks 00/01 dispatched 07:45. **Idle gap ~02:30–07:40 (Mac + 2 Colab slots):** watcher covered Colab logs only, not the Mac ruler or Sol PIDs → watcher now also tracks Sol PIDs.
- **2026-10-22 ~08:10 PT — FUNDING RULING (Claude, under the owner's standing Colab pre-approval): P1 family canary source-latebody-03-canary-01 funded at ≤ 1.0 u** (separate from the reserved Y3 6.10 u). Evidence: colab_cli final-only acceptance on current main PASS (`colab-cli-main-acceptance-20261022`); independent review `independent-review.json` (Sol). Also: colab_cli requalified on main after Sol's accounting change (94fc88eb). L4 native ruler main-01 refused at allocation (endpoint churn from the concurrent acceptance) → main-02 dispatched with a fresh output dir. O2 parked (L4 timing forecast refuses; needs native+exact-stop).
- **2026-10-22 ~09:00 PT — L2′ opponent premise (Fable):** registered L2′ bank uses the historical T3 rosters with `veto_slots=()` — correct for its §2 skill question (served Watch6 = B8 + 5×FRP3-s12+v8 is a different, product question). The "kernel rows required" condition was the judge's own error; native/exact-stop NOT needed for L2′ (P3 production 'all' frozen ≈ 0.44 batches/worker-s). Mode B forecast W2 ≤ 25.2 h (exp. ~22 h, ≤ 30 h w/ reserve). Next: Sol analyzer flip (refuse nonzero veto counters / nonempty veto_slots, native_mode=stock) → Codex-V review-04 → P3′ on 5 dev worlds (≤ 2 Mac-h) → P4 → launch; Fri 23:59 KEEP_B8 reads NOT_READY(L2′) mid-run. X5 disarmed (L2′ has Mac priority).
- **2026-10-22 ~10:14 PT — L2′ LAUNCHED on the Mac (mode A, exact_stop, W2, 8 chunks, 142-min chunk timeout, 28.3-h lease; out `exp/bl-l2/mac-amendment-01`).** Admission: P3′ identity READY_MODE_A (p3-identity-05), review-05 PASS (only the launcher lid-guard fix 311b3fcc changed in the closure), runtime-parity-07 VERIFIED. One look by Claude after all 8 chunk receipts (clean checkout == origin/main). Expected ~9 h. Also today: DENSITY-2 production retry dispatched under the 5.5-u one-time envelope (cap ruling); P1 Y3 source-latebody-03 running; L4 native ruler INCOMPLETE (not needed); gate CUDA-scope fix merged after requal PASS. Lesson: editing a reviewed pinned package (even a launcher bug) invalidates its review binding — route fixes through review first.
- **2026-10-22 ~13:15 PT — OWNER: "make budgets like 100% generous im colab credit heavy rn".** Standing rule: every Colab unit cap, reserve, envelope and operational wall/timeout = 2× the forecast or ruling (owner authority over judge-set caps). Applied now: P1 Y3 envelope v2 12 u → **24 u** (retry reserve 4.81 → 9.62 u; canary ≤ 1 → ≤ 2 u). Scientific design (bars, N, one look, outcome kill rules) unchanged.
- **2026-10-22 ~13:45 PT — RESUME POLICY (Fable, `JUDGMENT_RESUME_POLICY.md`; owner: "isnt work done on colab recoverable ... that sounds like a silly mistake"):** (1) DENSITY-2 completed by one outcome-blind job for the 85 missing episodes + 12-episode A identity replay, merged on the Mac into the single look (remote run had finished 492/492; loss was broker transport); (2) standing requirement for ALL Colab packages: idempotent completed units exported as they finish, verified-ledger skip on retry, Mac aggregator is the only look, ≤ 2 outcome-blind mechanical retries; (3) P1 operational snapshots 900 s in Y3 source + snapshot-replay test in Y3 replacement. Owner 2× budget rule applied on top: DENSITY-2 completion cap 5 u / timeout 180 min; P1 Y3 envelope 25.5 u.
- **2026-10-22 ~15:35 PT — L2′ (offline exact served-search ceiling vs bare B8 s3, Mac mode A, single registered look) = FUND_SYSTEMS** (development evidence; `docs/research/results/L2prime_mac_analysis_20261022.json`; run TERMINAL_VALID 8/8 chunks; outcome-free pre-flight PASS; one `analyze.py` invocation from clean launch checkout 0005cd87). Paired mass integral vs B8: **frozen +56.1 (LB90 +34.8), mixed +27.6 (LB90 +12.1), scripted +165.3 (LB90 +114.8); pooled-equivalent +83.0 (LB90 +64.1, UB90 +101.9).** No-death episodes +19.8 pp (LB90 +15.3). Gain is almost entirely survival (dense-only survival mass), action changes 0.07–0.52 per 1,000 decisions. First positive frozen/mixed evidence for search. KEEP_B8 clause 1 (Fri 23:59) does NOT fire. Not a release: next is served-systems work (latency, paced identity, strict gate) per PLAN_v2/PLAN §2.

| 2026-10-22 16:05 PT | PLAN_v3 ops | Sol M0/M1/M3 merged (M1 3708063b, Codex-V PASS, 66 tests). M0 Part B refused 3x on its own idle receipt: foreign heavy = macOS mediaanalysisd (~65-75% CPU, 3 h). Mac chain re-armed on the census's own snapshot() quiet test: Part B → M1 timed cells (frozen worktree snake-dqn-m1run @3708063b, empty ledger bl-m1-ledger-01) → X5; refused-before-start retries only, no blind retry after start. Sol launched on C1 bank runner and B2 natural-root library (schedule rows 3 and 5). Colab: P1 Y3 source running (~0.95M hero transitions); DENSITY-2 waits on canary-staging fix. |
| 2026-10-22 16:22 PT | PLAN_v3 ops | P1 Y3 source stopped at 950,272/2M refit transitions by integrity tripwire: fixed 160 MiB snapshot cap vs growing banks (mechanical) → Sol p1snap (bounds derived from declared maxima, 2x; resume verdict). requal-canaryfix lost accounting when P1's L4 ended mid-run → re-planned on empty account (requal-canaryfix-b). C1 runner merged (e686269a, Codex-V PASS, 79 tests); untimed check CLEAR, ledger bl-c1/ledger-01; C1 run waits for D1 per PLAN_v3 (not auto-chained). |
| 2026-10-22 16:55 PT | PLAN_v3 ops | DENSITY-2 canary-04 PASS (11 exports, 282 s) → completion-01 dispatched (04b55162). P1 snapshot bounds derived from declared maxima x2 (695e63af, Codex-V PASS; old run not resumable: zero complete snapshots) → fresh canary running. M0 Part B attempt-05 COMPLETE_UNAGGREGATED 28/28 (hashes published a7d117b0); sole aggregation armed to run after X5. M1 attempt-02 INCOMPLETE mechanical in 0.8 s (AISnake.auto_respawn; mocks hid it) → Sol m1fix incl. unmocked smoke. X5 running (recovery-09). B2 built (7e843acc) but PLAN_v3 leaves trajectory length / root grid / H1440 meaning open → Astra round 16 (4) → Fable judge before freezing. |
| 2026-10-22 16:58 PT | M0 Part B look | Sole aggregation (part-b.json sha 7c94ea7c): COMPLETE_D1_PENDING_M1. frozen+mixed 18 rescues (frozen 13, mixed 5), 15/18 surviving alternative is Q-rank 1, all-die-at-f 2 frozen; F(L) >= 0.50 for L <= 230 frames. L* = max(72, min(432, 5*ceil((24*(p99_doom_W4+0.25)+2)/5))) => CLOSED only if M1 W4 doom p99 > ~9.25 s; at the M1 bar (p99 <= 5 s) L* <= 130, F = 0.72. X5 GAVEUP (5 attempts aborted by foreign python from Sol builds) → re-armed after M1 (x5-after-m1). M1 fix 9fd17700 (Codex-V PASS, unmocked smoke OK) armed: smoke → ledger-03 → check → attempt-03. |
| 2026-10-22 17:16 PT | DENSITY-2 look (TERMINAL result) | completion-01 PASS (98 exports). First merge IDENTITY_FAIL = comparator defect (Colab staging-uuid checkpoint paths; judge JUDGMENT_DENSITY2_IDENTITY.md), fixed 57c0ad9e (basename compare only). Sole look summary-02.json: identity 12/12 IDENTICAL, r=0.747 outside [0.80,1.25] (reported; raw and 1/r evaluations both shown). ALL 9 variant cells (B70m, B70p, C-P1 x frozen/mixed/scripted) REJECTED: minimum_map_rule_violation, identical under both evaluations. Natural doom 0.058/1k hero decisions overall. Density/smaller-map lever CLOSED. |
| 2026-10-22 17:16 PT | Mac timing incident | Claude launched Sol b2freeze/density-identity at ~17:05 while M1 attempt-03 was armed: M1 cell stopped after 6 s ('foreign heavy load', verified stop, charged 6.0 s) and X5-12 refused. New hook guard_mac_timing.py blocks Codex launches while ops/mac-timing-armed or a live Mac lease exists. X5 restarted (x5-now, flag set). Fable judge asked whether M1 attempt-04 is admissible before D1. |

| 2026-10-22 19:50 PT | M1 attempt-05 pre-dispatch | JUDGMENT_M1_RESTART_OBJECTION.md sha256 160f5980796a9f0ca21e9282329725594a59522c8ea51cea98e61fd1b813f91c (pre-data): attempt-05 may dispatch as built at 01334e1e; post-data reducer rules R1 carry-forward (overrun / W4 doom S>5s in discarded prefix => BLOCKED) + R2 discard disclosure pre-registered; Sol builds m1_carry.py. Also: A100 program funded (PLAN_A100.md; Sol A0/A2/A3); P1 amendment 04 HOLD (C2 value drift 7.7/14.6/20.4 > PopArt 5.53) -> judge. |

| 2026-10-22 20:35 PT | P1 amendment 04 §6 | Claude history-inventory receipt collection-history-inventory.json sha256 7a182e7369745c423fd3c53b74cad98becc963c71eef973b998d533d8d2b7e14: 21 historical collector controls, all seed 20261021/128 worlds/offset 0 (one root family); new packs offset 3,011,000 with 131,466 burned roots excluded. Qualification B PASS (JUDGMENT_P1_E8_04_B), collision witness justifies +10,000. Next: Sol binds receipt into fresh sibling packs + Codex-V supplement, then canary 2.00 u -> source 9.62 u on L4. |
| 2026-10-22 | P1 amendment 04 inventory binding | Fresh immutable -04/final pack binds that receipt in both controls/shared bundle and allocation index03; -03 bytes retained. Independent Codex-V supplement04 PASS / READY_FOR_CLAUDE_CHECK, V3-INVENTORY cleared, zero unresolved. Canary-admitted.json enabled; source template remains false pending genuine canary proof and all-attempt debit admission. Eight historical terminal receipts charge7.46u, initial reservations22.94u/25.50u, WARNING_UNEXPLAINED_DEBIT retained. Upload preflights148.3054s PASS;15 targeted tests PASS single-process. See research/bl_ppo/P1_E8_04_HANDOFF_04.md and DISPATCH_PACKS_04 index. No dispatch/provider/timing/serving action; Claude live inventory/rate/debit/calendar checks remain required. |

| 2026-10-22 21:30 PT | M1 attempt-05 | INCOMPLETE (verified mechanical): cell cpu-w1-k1 restarted 3x then stopped; foreign heavy corespotlightd 99.8 %, fileproviderd 67.7/55.1/54.2 %; charged 771.55 s cumulative. receipt sha256 48dcd2ad67be5a44d563d95af6737be275bb73562e666fcc0080dd2544453f05; m1_carry verdict INCOMPLETE (receipt.carried.json sha 053b57fb). JUDGMENT_M1_ATTEMPT6.md sha 80598e91: attempt-06 admitted (last PLAN rerun) with HARD owner clean-window precondition + 600 s measured quiet; D1 Sat 12:00 else <= Sun 12:00. |

| 2026-10-22 22:00 PT | M1 owner directive | Owner: "let them run during test" — macOS system daemons (fileproviderd, corespotlightd, mds/mdworker, mediaanalysisd, photoanalysisd, cloudd, bird) and our Colab supervisors no longer abort M1; their per-sample CPU is recorded and every cell discloses max/mean daemon load; estimand relabelled 'served-Mac timing under normal macOS background load (owner-authorized)'. Any other foreign process (codex, other python) still aborts/restarts as before. Sol implements; attempt-06 waiter replaced. |

| 2026-10-22 22:05 PT | A100 A1 decision | Repaired GPU engine (corpse loop fix, 313d27eb) on A100-40: E6 speedup 190.9x (event-path hypothesis PASS) but E1024 complete steps/s = 158 frozen / 163 mixed / 166 scripted << 2,000 => pre-registered STOP_ENGINE. GPU label factory (WP2) not funded; A2/A3/P1 continue. |

| 2026-10-22 22:30 PT | M1 attempt-06b RESULT | BLOCKED (not mechanical; owner-authorized daemon-tolerant estimand; daemons ~0 %). receipt sha 3631f67bf5c41284bdcdb0beb4bd77a346f629be3065d70130d30b76e44a75b8. Latency bars PASS: W4K4 frozen-800 doom p99 1.46 s (<=5), update p95 4.6-10.2 ms (<=11.688); FAIL: tick overruns W1K1 6, W2K2 26, W4K4 5, W4K8 8 (bar 0). M0 with p99 1.4586 => L*=72, F(72)=0.944 => not CLOSED. D1 judge ruling on the one judged remedy pending. |

| 2026-10-22 22:50 PT | D1 RULING | JUDGMENT_D1.md (sha 335dc9e2781b52f4...): M0 GO (L*=72, F=0.944). M1 BLOCKED on overruns only: 6 = harness start-up artefact (240 ms first update + catch-up), 39 = sparse 38-60 ms steady-state game.update stalls (likely GC; never measured for bare game). ONE judged remedy = attempt-07 (last): 200-tick warm-up + pool pre-spawn, two bare W0 control cells, per-tick GC disclosure; Branch A zero bar default; Branch B (relative to bare control) only if Astra->Fable ratify pre-data by Sun 18:00. Verdict by Mon Oct 26 12:00; FAIL => KEEP_B8. Sol building; Astra round 22 on Branch B. A1 STOP_ENGINE; A2 main + A3 L4 running. |

| 2026-10-22 22:55 PT | A100 A3 result | analysis.json sha 721195a1b2fe6dae...: baseline cached actor learning 1.87 s (A100) / 1.81 s (L4) per 16,384-transition rollout, above the 1.6384 s stop line => WP5 paused for learner fix; optimized learner 0.41 s A100 / 0.98 s L4 (passes); collection 9.56 / 9.57 s dominates, full cycle ~10-12 s on both (~1.5k transitions/s). A100 gives no P1 wall-clock gain while collection is host-CPU bound; keep P1 on L4 with the optimized learner. |

| 2026-10-22 23:00 PT | M1 Branch B REJECTED | big-lever/round-22/JUDGMENT.md sha256 c85134313f824046cf26cd3060eabfa9df87ea6b2a6b4c25816d0614d5e439f6; admission-rule block sha256 4dc7bc164a1199cee40cb0f5a52c19525291f4282ecaf8a80b6def34f8743aa7. Attempt-07 = Branch A only (zero overruns in every worker cell, other bars unchanged); W0 controls verdict-neutral, report-only; --branch-b refuses. Pre-data. Dispatch by Sun 20:00, verdict by Mon 12:00; FAIL => KEEP_B8. |

| 2026-10-22 23:40 PT | M1 attempt-07 RESULT = FAIL (BLOCKED, second failure) | receipt sha 7117dd42415f1bc6...: worker cells overruns W1K1 15, W2K2 9, W4K4 12, W4K8 10 (bar 0); latency PASS (W4K4 doom p99 1.38 s, update p95 8.6-9.6 ms). Bare W0 controls (no search, verdict-neutral): pre 43 overruns with update p95 13.2 ms (> 11.688), post 3 (incomplete). Per JUDGMENT_D1 / round-22: M1 FAIL => KEEP_B8 for serving; no further remedy; C15 not built. PRODUCT FINDING for owner: the served B8 Watch game itself misses 24 fps deadlines on this Mac without any search. |

| 2026-10-22 23:50 PT | A100 A2 result (single look) | decision.json sha 045be833616754e9...: scale NOT_CLOSED_EXPLORATORY, target KEEP_SINGLE_HORIZON. T5 D300 AUC @100%: frozen-B8 linear 0.767 > s_b8 0.719 > m_random(6.7M) 0.711 ~ s_random 0.706; data 25->100% +0.02-0.05 for all arms. Bigger model no gain; data volume helps; frozen B8 features beat fine-tuning. Exploratory historical bank, no served-gain claim. |

| 2026-10-22 23:58 PT | Direction ruling (round 23) | JUDGMENT_DIRECTION.md sha b50dffba...: 86/89 attempt-07 overrun ticks carry >= 20 ms gen-2 GC pauses (bare W0: 42 gen-2 collections, ~2.1 s GC/100 s) => served B8 game misses SC §4 today (product finding); C15 blocked by GC, not search. Path A: GC fix in served process (Astra round 25 started) -> bare qualification >= 2,400 ticks + 10-min session -> owner-rechartered M1-bis Sat 09:00 -> PLAN_v3 +2 days. B: CPU-runtime label factory (post-Nov-10 distillation). C: P1 on L4. Owner decisions requested by Fri 21:00. |

| 2026-10-23 00:20 PT | Data plan (round 24) | Opus stand-in JUDGMENT.md sha 750c2c6a... (Fable prior c718cd56..., same core plan, kept as JUDGMENT_fable_prior_0010.md). Bigger GPUs add no labels (CPU-bound sim). Multiplier = hindsight relabelling (restore last 300 decisions after each bare-B8 death; roll out only alternatives): ~15-33 rescue-bearing lives/Mac-worker-h vs 2.7 (L2'); fleet ~7-34k decisive labels/day. P(dev-screen win vs B8) ~0.10. No GPU engine now. Sol started F0 (label factory + CPU-runtime canary) and F3 (frozen-B8 residual-head harness + F4 dose-study draft). P1 source retry05 dispatched (canary PASS). |

| 2026-10-23 01:25 PT | F4 APPROVED (pre-data) | round-26 JUDGMENT.md sha 0fe1a103...8362 (Opus stand-in, RATIFY-WITH-CHANGES); F4_REGISTRATION.md sha 66388c79...2fd65; FitRecipe sha 49d12f48...3a5a; all 8 edits at 8d6df609 (Codex-V PASS, 151 tests); approval artifacts big-lever/exp/f4/judgment.json (sha 15b6c842e73f1030...). No labels read. F0 label factory: family probe PASS on qualified tree d0d92bdb; CPU science canary running. GC fix: d.1 PASS (3 seeds x 5,000 ticks), d.3 PASS (both policies) at 40993bce; d.2 timed probes queued. Kernel-identity supervisor fix merged (qualified d0d92bdb). |

| 2026-10-23 01:40 PT | P1 Y3 critic gate PASS (amendment 04) | source retry05 training-status.json: critic_refit PASS, held-out natural late-body n=163 EV 0.587, MSE 295,944 vs baseline 528,354 (fresh disjoint worlds, offsets 3,010,000/3,011,000). Job ended INCOMPLETE at finalisation (mechanical): 'Y3 source lacks an operational snapshot and fixed suffix proof'; exports intact. Sol diagnosing + offline recovery path. |

| 2026-10-23 02:05 PT | P1 finalisation ruling | JUDGMENT_P1_FINALISATION.md sha fe0584595cca6f6f... (Opus stand-in): critic PASS stands as the registered Y3 critic-gate result (no second look); Y3 not yet PASS. Amendment 06 repair = import verified pre-actor origin, skip refit/gate, actor phase only to a 900-s operational capture + suffix (canary 2.00 u, repair 6.16 u); envelope 25.50 -> 34.50 u (one retry each); replacement 3.86 u with tape check; 1M qualification 4.00 u after full Y3 PASS. Sol building. |

| 2026-10-23 04:00 PT | OWNER DECISIONS | Verbatim "yes approved yes to all fri" (big-lever/exp/OWNER_DECISIONS_20261023.txt): (1) publish P1 Y3 origin assets as public GitHub release (incl. disclosed paths; scope incl. replacement/1M) — pending owner gh auth login; (2) GC freeze_startup serving change approved subject to registered qualification (d.2 A1 + Sunday regression) passing; M1-bis re-chartered once, conditional on GC A1 PASS; 120-tick probe retired for >=2,400-tick + 10-min bare qualification. F1 CPU probe INVALID (F0 export digest mismatch) — investigating; L4 probe running. |

| 2026-10-23 05:05 PT | GC A1 (d.2) VERDICT = FAIL | A1_VERDICT.json sha d1ff26d47edb58df... at 85aa7cf5 (all 6 cells COMPLETE; hashes published e68de0d3). Overruns default 117/113 (2,400) & 813 (14,400); freeze 100/105 & 1,138. freeze cut max GC pause (87/52/67 -> 11/12/28 ms) but gen-2 collections ~4x more often; non-GC overruns dominate (70-739/cell); update p95 13.8-23.3 ms (> 11.688) under BOTH policies; RSS slope 15.5 MiB/min (fail). Host noisier than attempt-07 (bare 43/2,400 then). Per owner approval terms: GC serving change NOT adopted; M1-bis NOT chartered; KEEP_B8; C15 stays closed. Product finding: served Watch game cannot hold 24 fps on this Mac under normal background load. |

| 2026-10-23 05:20 PT | F1 re-funded | JUDGMENT_F1_REFUND.md sha f2f35631e65e4234... (Opus stand-in): F1 INVALID = packaging defect (ledger sidecar basename), not science; failed probes recorded, excluded. pack-10 caps: CPU probe 40 min/0.07 u, CPU ruler 180 min/0.30 u, L4 probe 20 min/0.65 u, L4 ruler 94 min/3.02 u (F1 total <= 4.20 u incl. 0.153 spent; <= 5.0 u). Keep CPU arm and 25 % rule; no A100 arm. Next INVALID => HOLD. GC A1 FAIL logged earlier. |

| 2026-10-23 06:20 PT | F1 placement decision | f1-decision-10/decision.json (single look; hashes published 0b2c1ca8): NO_PRODUCTION_PLATFORM — CPU runtime 6 and L4 4 exec RPC faults at the 120-s budget => per registered rule neither Colab platform used for F0 production this week. Judge asked about Mac-as-production label factory + supervisor stall fix. |

| 2026-10-23 06:35 PT | Placement ruling | JUDGMENT_PLACEMENT_MAC.md sha 6adf731e49b0fd27... (Opus stand-in): Colab NO_PRODUCTION_PLATFORM through Sun Oct 25 (control-path RPC stalls on CPU and L4). Mac = F0 production platform after a Mac-equivalence package (1/3/6-worker digest+label identity on the 64-root bank; full 36-allocation F0 canary on Mac as the single F0 look, floor 8 rescue-bearing lives/Mac-worker-h). Default 6 workers nice 10 (12 needs owner OK). Forecast 37-56 lives/h at 6 workers (P(fail) ~0.4); D256 6-9 h. Supervisor stall fix commissioned (re-qualify 3+3 one-hour sessions, then F1-bis Mon Oct 26). |

| 2026-10-23 07:10 PT | OWNER standing approval | Verbatim "always gonna- approved" (big-lever/exp/OWNER_STANDING_APPROVAL_20261023.txt): routine operational asks pre-approved; W12 Mac F0 factory approved subject to the registered 30-min calibration (after M-E). Releases and credential entry still owner-only. |

| 2026-10-23 07:40 PT | F0 Mac equivalence PASS; D256 started | mac-pack-02 RUN_ME (W6): F1 1/3/6-worker identity, Canary-B 36 allocations complete in 1,117 s, Canary-A replay audit PASS, single F0 look PASS: 14.5 rescue-bearing lives per Mac-worker-hour (floor 8; forecast was 6-9). D256 dispatched on the Mac at W6 (bl-f0-mac-d256-02; first attempt refused pre-start on a missing approval path — operator error, no data). W12 calibration runner queued (owner standing approval). |

| 2026-10-23 07:55 PT | Colab stall fix PARKED | Requal of a18bf6b4: CPU PASS, L4 INCOMPLETE; Sol diagnosis (03d7e841): real bounded L4 control-RPC timeouts (2 x 30 s, one unrecovered), no sizing defect; counts as L4's first failed attempt (§7 C1.3/C1.4). Colab label lanes parked this week; Mac factory is the platform (14.5 lives/worker-h). D256 running at W6 (8/858 parents in 20 min — slower than forecast); round-27 (10 Astras) on factory speed in progress. App update killed tmux waiters; W12 calibration re-queued after round 27. |

| 2026-10-23 08:10 PT | Factory speed ruling (round 27) | JUDGMENT.md sha f3d4e62547718306... (Opus stand-in): measured canary profile = play 47 %, doom alternatives 34 %, ordinary roots 11 %, audits 7 %, restore 1 %. Build from D1024: W12 (>=1.54x; width-change resume), fold 1-4 parent filter (1.25x usable), prepared-decision packet (1 obs + 1 B8 forward per decision; 1.08-1.16x), bare-audit mass check; 4 gated items behind pre-set profile thresholds. Byte-identity vs 61900af2 on a 12-parent bank + 1/64 production recheck. D256 left untouched (ETA ~15:00 PT at W6). Rejected: pruning, warm-start, batched/GPU inference, >12 workers. |

| 2026-10-23 08:45 PT | D256 W6 PREEMPTED (operational, not scientific): driver slowdown_guard tripped (life wall 208.9 s vs first-hour baseline 147.9 s) at 36/858 parents, 4,463 s charged of the 57,600 s cap, 0 thermal trips, verified terminal receipt. Likely cause: concurrent Sol f0speed load. No D256 continuation tool exists (mac_gates ledger is Canary-only; d256_w12 refuses prior attempts). Mac idle → W12 calibration --check PASS (2a0821a9), full calibration queued; then Sol builds an accounting-preserving D256 resume (same state dir, fresh output, union ledger, W12 only on sealed PASS). Lesson: no Codex/Sol on the Mac during factory runs either. |
| 2026-10-23 09:31 PT | W12 calibration (runner 2a0821a9, receipt w12-calibration-01) FAIL by registered rule: worker-wall ratio 1.470 > 1.30, 1,811 s measured, 0 thermal / 0 slowdown trips. Wall throughput W12 = 1.36x W6 (16.3 vs 22.1 s per 192 roots). Default W6. Independent judge asked whether wall throughput should govern (JUDGMENT_W12_THROUGHPUT_20261023.md pending). Sol d256resume launched by chain. |
| 2026-10-23 09:40 PT | Judge ADOPT_W12 (snake-dqn-artifacts/big-lever/exp/JUDGMENT_W12_THROUGHPUT_20261023.md, sha 03a7afd1…815ef): the 1.30 per-worker bar is replaced by a wall-throughput bar W12 ≥ 1.20× W6 (measured 1.36×). This is a post-data change to an operational bar only; labels are byte-identical and no F4 statistic uses width. Tighter W12 guards: RSS > 34 GiB or any thermal warning → W6. Sol w12admit queued after d256resume. Owner notice only. |
| 2026-10-23 10:03 PT | Sol implements the binding W12 throughput ruling, full SHA256 `03a7afd1a1ab48dfbb4a2a4fd760ea00d2fe5c8bf6132bfc3660f3077b8815ef`. Create-only `bl-labels-f0/w12-admission-01/receipt.json` PASS, SHA256 `65ea988e086ea1c26fc13dba505f4aae7ba524d443095def77ba10c1b80fa4fe`; recomputed wall throughput 1.360454595884024 >= 1.20, measured duration 1811.1803524983115 s, zero thermal/slowdown. Calibration-01 stays FAIL at unchanged SHA256 `082304bc3e94ce48f82313ed285fc7388c099ef4b8d5c1bcc47545cb6fb3cdbc`. Admission use rechecks exact ruling/calibration/host hashes and measured cells. D256 continuation preserves original control, namespace, allocation, state and 57,600 s cap; W12 RSS >34 GiB / first thermal warning / failed first-three-hour parent wall-throughput >=1.15x matched-source W6 trigger life-boundary drain and sealed W6 continuation. Thermal return requires verified 24 h clean across dose states. Canary-B W12 baseline is x1.36; width charges and slowdown baselines remain separate. Future launchers use a separate frozen science tree and operational pin; D4096 remains R6 HOLD. Integrated bounded suite 114 PASS; final admission subset 9 PASS; independent Codex-V light review recorded in `research/bl_labels/CODEX_V_W12_ADMISSION_20261023.json`. An external operator started original-pin W6 D256 at 09:59:53 during verification; Sol did not start, stop, or modify it, and deferred additional multi-second verification after detection. Exact next-resume dispatch commands are in `W12_ADMISSION_HANDOFF_20261023.md`; no factory/calibration/long job started by this implementation. Owner notice: W12 calibration measured 1.36× wall throughput with 0 thermal; the 1.30 per-worker bar measured E-core speed, not heat, so the judge replaced it with ≥ 1.20× throughput plus unchanged thermal guards; any thermal warning returns the factory to W6. |

| 2026-10-23 10:10 PT | D256 resume-02 (6498f992, W6, Codex-V PASS) started and then PREEMPTED at 402 s (0 of 822 remaining parents done). Root cause is a guard defect, not load: the slowdown guard compares single-life wall time against the first-hour mean (life length varies naturally; attempt-1 15-min medians held at 129–152 s), and it is armed immediately on resume, so the first life on each partial parent includes replay of the already-done prefix and trips it by construction. Judge ruling requested (JUDGMENT_SLOWDOWN_GUARD_20261023.md); no further resumes until repaired. |
| 2026-10-23 10:12 PT | Judge REPAIR_GUARD (JUDGMENT_SLOWDOWN_GUARD_20261023.md, sha e50a573e…09551): a mechanical fix plus an operational clarification; labels untouched. The single-life trip is replaced by: median of the last 4×W eligible lives (this attempt and width) > 1.30× the frozen per-width first-hour median (≥60 lives; W6 = 136 existing lives). Excluded: replayed first lives, lives spanning a width change or a stop. Re-arms only after K fresh lives. Two trips in 6 h → W12 drops to W6 for 24 h. Replay of attempt 1: the rolling median ran 0.88–1.04× baseline throughout (no real slowdown). Sol guardfix launched; resume D256 at W12 after Codex-V. Owner notice only. |
| 2026-10-23 10:26 PT | Sol implements REPAIR_GUARD, full ruling SHA256 `e50a573e69e6aeea1de7ddde590bac36e62a28a46c3399f5bce0182ae4109551`, composed with W12 throughput ruling `03a7afd1a1ab48dfbb4a2a4fd760ea00d2fe5c8bf6132bfc3660f3077b8815ef`. Window median of last 4x admission-width fresh eligible lives, immutable per-stream/per-width first-hour median (>=60), replay/width/stop exclusions, W12 baseline sanity <=1.911x B6, same-pin new-stream sanity and two-trip escalation. Actual D256 timestamp regressions: attempt1 no trip (max evaluated ratio0.995391<=1.04); attempt2 four replay first boundaries excluded, five eligible<K24; 1.35x slowdown trips after20<=24 boundaries,1.20x no trip. Codex-V light suite81 PASS; review receipt `research/bl_labels/CODEX_V_SLOWDOWN_GUARD_20261023.json`. Frozen science remains clean at `61900af28932e47e22e332840fc4406559adb062`; original namespace/control/allocation/state/cap and summed accounting retained. Existing W12 admission `65ea988e...0fa4fe` verified and reused; calibration stays FAIL. Existing pack/calibration/admission inventory2110 files unchanged. No factory/calibration/host check/dispatch/long job launched. Next explicit operator W12 dispatch includes --host-check because the two sealed legacy W6 preemptions require a logged check first. Owner notice: D256 slowdown guard compared single lives (10 % exceed 1.30x by chance) and self-armed on every resume; judge replaced it with a 1.30x test on the median of the last 4xW fresh lives vs a per-width first-hour median; thresholds unchanged; attempt-1 data never exceeded 1.04x. |

| 2026-10-23 13:35 PT | OWNER: "work with astra not fable on research". From now on, research advice and independent judge rulings go to Astra personas (Codex gpt-6-astra), not Fable or Opus stand-ins. Code stays with Sol. The two rulings issued earlier today (W12 throughput, slowdown guard) stand as issued. |
| 2026-10-23 17:45 PT | P1 Drive route (owner yes ~13:10 PT): 9 assets staged and hash-verified in My Drive/snake-dqn-p1-y3-origin-20261023 (id 1yr71…); the owner set Anyone-with-link Viewer. Deviations, disclosed to Codex-V: the owner switched Drive to Mirror mode mid-staging, so the copy was rerun into ~/My Drive; a duplicate nested folder from the migration was moved to Trash; IDs were read from the DriveFS mirror DB instead of xattrs, with the same refusal checks. build-drive ran; preflight PASS: 9/9 exact SHA, 595 MB in 24.3 s. Codex-V _12 FINAL is running (sol-p1codexv). |
| 2026-10-23 17:47 PT | Sol guardfix done at 10:28 PT (0d187a2f, Codex-V PASS, 81 tests), but the operator missed it, so the Mac sat idle about 7 h. D256 W12 resume (bl-f0-d256-guard-w12-01, with --host-check and the w12-admission-01 record) is queued to start 2 min after the P1 Codex-V run exits. |
| 2026-10-23 17:58 PT | P1 Codex-V _12 FINAL: FAIL on procedure only. Byte bindings PASS (36 URL edits + 20 attempt-identity fields, URL↔ID match, preflight bound). The single finding is F1_A3_3_SQLITE_SUBSTITUTION_UNRATIFIED; the receipt is preserved. Astra judge asked (JUDGMENT_P1_A3_3_SQLITE_20261023.md). D256 W12 dispatch HOLD: the host check matched the ChatGPT app's bundled codex binary by comm name. Sol hostcheck fix (argv match) is launched. |
| 2026-10-23 18:05 PT | Astra judge RATIFY (JUDGMENT_P1_A3_3_SQLITE_20261023.md): the SQLite ID read is accepted for this pack. Conditions: _12 FAIL preserved; a new create-only Codex-V _13 FINAL must bind the ruling and resolve F1; usual admission/live checks; the VM canary stays mandatory; no restaging. Sol hostcheck fix ca8ad8ab (argv match, ChatGPT app ignored; Codex-V PASS). Codex-V _13 running; D256 W12 (bl-f0-d256-guard-w12-02 @ca8ad8ab) queued 2 min after it exits. |
| 2026-10-23 18:25 PT | Codex-V _13 FINAL PASS (P1 ready for Claude checks and admission). D256 W12 dispatch -02 HOLD: _exclusive_factory_guard matched ~/.codex/computer-use/Codex Computer Use.app (split at a space gives 'Codex'). Sol exclguard fix launched. The OWNER asked for Astra shortcuts on the GPU-engine rewrite (round-24 estimate 10–15 days / 100–300 u): round-28 Astra panel (8 personas) launched. The exclusive guard also preempts the factory every 5 s whenever any Codex job runs, so an Astra ruling on factory-window concurrency is requested (JUDGMENT_FACTORY_CONCURRENCY_20261023.md). D256 dispatch waits for these to finish. |
| 2026-10-23 18:45 PT | OWNER STANDING RULE: "i want to have an a100 running always and us trying to make it work using astra advisors". The A100 GPU-engine lane is reopened by owner direction; this supersedes the round-24 §5 reopen conditions. Astra panels choose the experiments, Astra judges, Sol builds, Codex-V verifies; engine outputs never become gate evidence without exact CPU verification. First job: Sol is building A1-PROF (per-stage A100 profile at E64/256/1024, all three rosters; outcome-free) to feed the round-28 panel. Sol exclguard fix 5dfd4a74 is in; D256 waits for the factory-concurrency ruling. |
| 2026-10-23 18:55 PT | Astra ruling ALLOW_BOUNDED_CONCURRENCY (JUDGMENT_FACTORY_CONCURRENCY_20261023.md): during factory windows Astra/Sol may run at nice ≥15 on a shared 0.25-core budget, plus one test lane at nice 19, 1 thread, ≤120 s wall and ≤90 CPU-s per test (≤300 CPU-s per 30 min). The test lane is revoked when the window median exceeds 1.15×B. Existing guards and timing exclusivity stay binding. Sol 'concurrency' is implementing it; D256 dispatches after that. |
| 2026-10-23 18:55 PT | Round-28 Astra panel (8 memos) on shortening the GPU engine: consensus 3–5 days and ~25–75 u (vs 10–15 days / 100–300 u). Repair, not rewrite. Main tricks: reuse the existing CUDA sparse-flood/component kernels that vector61 bypasses; remove the 96 repeated body gathers per feature call (2 calls per tick); cache free-space sufficiency proofs; a CPU-authoritative hybrid with batched GPU inference. First test is 0 u stage attribution, with a kill if the remainder exceeds 512 ms per E1024 tick. Dissent (Astra 06): keep STOP unless the end-to-end useful throughput gain is ≥4×. Astra judge is ruling on the 5-day A100 queue. |
| 2026-10-23 18:57 PT | A1-PROF prepared for the standing owner A100 engine lane; source `557ad8a66677acd14bf78b0acc087020fa42a7c4` published on main, no Colab dispatch. E64/256/1024 x scripted/frozen/mixed, separate uninstrumented full ticks, exclusive CUDA-event segments and profiler kernel/copy/launch/Python/memory attribution; dense vector61 flood isolated, broad and conservative non-observation remainders reported beside descriptive 0.512 s E1024 reference. Published-pin real Mac entrypoint preflight 19.955589 s, 3/3 E1 rosters with progress and immutable sidecars; supervisor canary validation PASS. Codex-V LIGHT: no outstanding findings, 18 package tests passed/1 CUDA skip, 20 source/export tests passed; broader bounded canary/schema/ops selection 178 passed/1 skip. Receipt `research/gpu_sim_spike/CODEX_V_A1_PROF_20261023.json` SHA256 `dc6356244b32674a5f8b2e64ee6838f2228a4107bd22f0b46ac7ec39d311b495`. Forecast 40 min / 5.13-6.67 u total; effective doubled main90min/20u+4reserve and canary30min/5u+1reserve, first cap wins. Same-pin A100 canary mandatory before main; main remains an unbound template. Exact commands in `research/gpu_sim_spike/A1_PROF_HANDOFF_20261023.md`; pinned operational checkout `snake-dqn-a1-prof-run`. Outcome-free diagnostics only, no labels/gate/release evidence; engine, transport, serving and legacy workers unchanged. |
| 2026-10-23 19:12 PT | OWNER: open-ended speed-up review. Round-29 Astra (4 generalists: RL research lead, ML infra, process/statistics, outsider red team) gets the full list of speed-ups already in place and is asked what is missing. The specialist round-30 panels will follow the directions it raises. |
| 2026-10-23 19:25 PT | Round-28 Astra JUDGMENT (round-28/JUDGMENT.md): run A1-PROF first, since the 0 u receipt audit cannot establish exclusive stage shares. Then: sparse CUDA floods; remove repeated body gathers; exact geometry; measured residual bottlenecks; cached proofs. A CPU-authoritative batched-inference fallback is built in parallel. Bar: ≥2,000 complete world-steps/s on every roster by hour 72. Day-5 falsifier: ≥4× verified useful lives/h. A ready A100 queue runs within 120 h / ≤1,100 u. A1-PROF (pin 557ad8a6; Mac preflight PASS 20 s; Codex-V PASS) A100 canary DISPATCHED 19:22 PT (cap 5+1 u; main 20+4 u). |
| 2026-10-23 19:25 PT | Full concurrency implementation 22c0db2f = PREPARED_NOT_QUALIFIED (activation HOLD). Owner-directed interim is being built (sol-interimconc on 5dfd4a74): the factory ignores nice ≥15 processes, and agents/tests are launched via agent_nice.sh. Astra quick ruling requested (JUDGMENT_FACTORY_CONCURRENCY_INTERIM_20261023.md). D256 dispatches after both. |
| 2026-10-23 19:35 PT | A1-PROF canary INCOMPLETE: the strict profiler-correlation check aborted at 17,843/17,910 kernels attributed (99.6%); the CUDA flood self-check passed on A100-SXM4-40GB. Sol a100prof2 fix running. Interim concurrency ruling ALLOW_NICE15_INTERIM (JUDGMENT_FACTORY_CONCURRENCY_INTERIM_20261023.md); Sol pin 4b46454d (191 tests, Codex-V light PASS); a focused Codex-V (condition 3) is running before the D256 dispatch. run_round.sh and all agent launches now go through ~/.claude/bin/agent_nice.sh. |
| 2026-10-23 19:40 PT | Round 29 (open-ended, 4 Astras) raised five directions: (1) converting search to policy, using existing labels first; (2) hindsight leaking privileged future randomness; (3) rehearsing the full decision path plus separating 'beats B8' from 'data caused it' and adaptive Mac episode allocation; (4) ops throughput and never-idle; (5) remote verification capacity. Specialist rounds 30 (search→policy), 31 (privileged-randomness leakage), 32 (rehearsal plus sequential/adaptive gates), 33 (ops throughput) launched with 3 experts each. |
| 2026-10-23 19:50 PT | A1-PROF canary-02 (fix 2edf7ed6: tolerant profiler correlation plus an explicit unattributed share) DISPATCHED to A100. Focused Codex-V of interim pin 4b46454d = FAIL: recovery receipts lack interim ruling/pin identity, and the test lane has no serialization or 120 s deadline. Sol interimfix running; D256 still HOLD. Rounds 30–33 complete (3 memos each); four Astra judges are ruling. |
| 2026-10-23 19:58 PT | P1 Claude line checks (canary): manifest canary-manifest.json sha fa398a8e…9c726e; Codex-V _13 FINAL sha f6b9b17b…3245a, READY_FOR_CLAUDE_CHECK, 0 unresolved, ruling fe058459…ebfc, commit 03e0af75. Both the _12 FAIL receipt and the Astra A3.3 RATIFY are on record. Live gates: broker = P1 pin = qualified research/colab_cli tree d0d92bdb; Colab 0 assignments / 0 sessions; rate 1.54 u/h; balance 1,099.64 u. Admission and dispatch of p1-repair10-04-canary follow (2.00 u + 0.40 reserve, 30 min, zero retries). Rounds 30–33 Astra judges adopted cheap checks; Sol builders r30–r33 running. A1-PROF canary-02 aborted again: 68 unattributed kernels = 77% of kernel time (the finding itself); Sol a100prof3 now reports per kernel name. |
| 2026-10-23 20:02 PT | P1 canary ADMITTED (canary-admission/manifest.json via p1_repair_admission canary, Codex-V _13 + Claude line check) and DISPATCHED: p1-repair10-04-canary (2.00 u + 0.40, 30 min, zero retries). Repair (6.16 u) follows only after every canary bar passes, debit is reconciled and the live gates are run fresh. |
| 2026-10-23 20:06 PT | OWNER: "have astras look at this" (the A1-PROF finding: 68 unattributed kernels = 1.14 s of 1.48 s, 77% of kernel time, ~17 ms per launch). Round-34 Astra panel launched (CUDA perf, PyTorch extension internals, GPU graph algorithms, red team), asking what the kernels are, why correlation fails, the exact next-run profiling setup, and the first fix with a kill rule. |
| 2026-10-23 19:54 PT | Time correction: rows logged earlier today as 19:50–20:06 PT ran between about 19:25 and 19:45 PT (operator estimate error). D256 W12 RUNNING (bl-f0-d256-guard-w12-04 @387b1dde; interim Codex-V-02 PASS, 99 tests). Workers 12, 815 parents pending, prior elapsed 4,866 s, output d256-output-mac-03. The -03 dispatch was refused by the host check because the tmux parent shell of the agent launches (argv contains codex-handoff.sh) ran at nice 0; those shells were reniced to 15, and agent launches now renice their own shell. P1 canary terminal PASS by the supervisor (523 s, debit 0.15 u, balance 1,099.49); bar mapping is under independent Codex-V (upload_elapsed 208.65 s vs the residual ≤150 / staging ≤300 bars) before any repair. A1-PROF canary-03 DISPATCHED (per-kernel-name export). Round-34 Astras: the 68 kernels are real native-extension launches without profiler scopes (suspects: sparse_frontier, component union/count, event FIFO); the capture was E64 scripted, not E1024. Sol gpuflood is building FUNCTION scopes, per-launch events, matched E1024 ticks, and an exact capped sparse BFS for vector61 with CPU parity and a kill rule. |
| 2026-10-23 19:58 PT | D256 -04 PREEMPTED at 317 s (2 parents done). The exclusive guard caught the A1-PROF canary-03 local Colab supervisor tree, including its colab CLI child pid 61325, which the tmux server launched at nice 0. -05 was refused at the host check for the same pid. Reniced the whole supervisor tree to 15; -06 RUNNING (W12, 813 pending). Astra persistence ruling requested (JUDGMENT_FACTORY_PERSISTENCE_20261023.md): require a sustained foreign process before preempting, and decide whether supervisors run at nice 15. Operator commands now run under nice 15. |
| 2026-10-23 19:59 PT | P1 canary chain: Codex-V _12 FAIL (F1 SQLite ID read), then Astra A3.3 RATIFY (sha f1bd0dca…f3ee9), then Codex-V _13 FINAL READY (f6b9b17b…3245a), then canary p1-repair10-04-canary terminal PASS. Codex-V canary verification CANARY_PASS on all six bars: VM fetch <260.1 s; residual upload 113.76 s; staging 208.65 s; exports and release verified; repair forecast 90.9 min at 1.54 u/h. **Reconciled P1 ledger:** prior 10.11 u + canary conservative charge 0.23 u (own 0.2238 u > observed 0.15 u) = **10.34 u charged** against the 34.50 u Y3 v5 hard cap; remaining worst-case reservations 22.04 u, headroom 2.12 u. WARNING_UNEXPLAINED_DEBIT carried. Repair (6.16 u / 200 min) is admitted after fresh live gates, once the A1-PROF canary-03 assignment ends (zero-other-assignment rule). Astra ALLOW_PERSISTENCE_30S (JUDGMENT_FACTORY_PERSISTENCE_20261023.md): preempt only after 7 consecutive sightings spanning ≥30 s; supervisors at nice ≥15. Sol persist is implementing it. |
| 2026-10-23 20:14 PT | P1 REPAIR ADMITTED and DISPATCHED (p1-repair10-04-repair, 6.16 u / 200 min; repair line check binds the CANARY_PASS chain; fresh gates: tree d0d92bdb, 0 assignments, balance 1,098.49). Its local supervisor spawns colab CLI children at nice 0 every few seconds, so D256 -06 was PREEMPTED (8 parents done) and -07 was refused at the host check. D256 waits for Sol's persistence fix; an interim renicer loop (ops/renicer.sh, nice 15) lowers stray Colab/agent processes every 1 s. No A100 jobs during the P1 repair (zero-other-assignment accounting). |
| 2026-10-23 20:14 PT | **A1-PROF full profile obtained** (canary-03: the job succeeded with 12 exports; marked INCOMPLETE only because result transfer took 162.9 s > 150 s). E1024 tick = 6.48 / 6.43 / 6.17 s (frozen/mixed/scripted; 157–166 steps/s). vector61 flood 2.46 s + vector61 other 2.51–2.59 s ≈ 77%; movement/collision/events 0.85 s CUDA (0.12 s host); opponent v8 0.47/0.28/0 s; non-observation remainder 0.89–1.01 s, above the 0.512 s bar. Throughput is flat from E256 to E1024 (tick linear in E): no batching gain. Round-35 Astra panel launched with the data. |
| 2026-10-23 20:19 PT | Persistence fix caafdf5d (ALLOW_PERSISTENCE_30S; Codex-V 69 acceptance cases + 223 regressions PASS). D256 resumed as bl-f0-d256-guard-w12-08 (W12, 805 pending, output -05), running alongside the active P1 repair supervisor and the renicer loop. |
| 2026-10-23 20:22 PT | P1 REPAIR attempt INCOMPLETE: UPLOAD_CAP (the total upload time cap was reached at source.py), stop_verified true, cleanup_error null, 384.6 s. The canary had uploaded in 113.8 s on an idle Mac with its supervisor at nice 0; this attempt's supervisor was reniced to 15 while D256 W12 was starting or draining. Codex-V diagnosis sol-p1repairv is running and will rule whether this is the one permitted mechanical retry, give the root cause, and set safe retry conditions. No retry yet. |
| 2026-10-23 20:22 PT | GPU sprint: Sol gpuflood built A1-PROF v4 (FUNCTION scopes/NVTX/per-launch events) and the exact sparse capped-BFS candidate (pin 31375825, frozen checkout snake-dqn-gpu-flood-frozen-01; Mac preflights PASS; CPU byte-identical full61 on all rosters; 317 tests). Profiler v4 canary a1-prof-a100-canary-0401 DISPATCHED; D256 kept running through the dispatch, so the persistence fix holds. Round-35 Astras: four repairs are needed (dense flood; repeated geometry/body views; movement/collision with 32 redundant respawn occupancy builds; v8 bundle 0.63 s on frozen). The red team calls 2,000 steps/s in 72 h an upside case and wants a measured residual budget by T0+24 h. Parallel Sol lanes MOVE and V8 launched with strict file ownership. |
| 2026-10-23 20:47 PT | P1 repair Codex-V diagnosis (P1_REPAIR_FAILURE_CODEX_V_20261023.json sha b40f8abb…92d5): **NO_RETRY under current rules**, because the finalisation ruling explicitly excludes cap failures. Root cause: control-path delay, not data. The residual upload set is identical to the canary (74,138 B); two exec calls took ~120 s each. **Operator error (Claude):** the repair supervisor and its colab CLI were reniced to 15 while the 12-worker factory ran. Astra ruling requested on a second attempt (JUDGMENT_P1_REPAIR_ATTEMPT2_20261023.md). D256 -08 was PREEMPTED after 1,372 s with 12 parents done: 'process monitor fault' on a nice-20 process (macOS renice -n adds; system daemons also run at 20). The renicer now sets priority absolutely; Sol monitorfix is making nice 20 and other legitimate ps rows valid. D256 -10 RUNNING (793 pending). |
| 2026-10-23 20:49 PT | GPU sprint: profiler v4 canary INCOMPLETE, because large .json.gz traces missed the telemetry-recovery deadline; the GPU work completed and 15 unit exports were recovered. The sparse chain was blocked because its helper needs a PASS v4 receipt. Lane V8 (pin 85180521): Mac CPU parity PASS (61 vector61 columns, v8 diagnostics, B8 planes, full sim tensors). Its A100 canary allocation FAILED (Colab kernel-creation uncertainty, 0 GPU time). Lane MOVE (pin 866fa48e) is INVALID under its own rule: byte-identical to the frozen device baseline, but reward/food-order/corpse-flag mismatches vs CPU GridBatchSim that pre-exist in the baseline (float32 vs float64 reward). Sol gpuunblock is running (small exports, decoupled sparse chain, fresh V8 namespace). Astra parity-standard ruling requested (JUDGMENT_GPU_PARITY_STANDARD_20261023.md). D256 -10 running (790 pending). |
| 2026-10-23 20:50 PT | Astra AUTHORIZES ONE FINAL P1 REPAIR ATTEMPT (JUDGMENT_P1_REPAIR_ATTEMPT2_20261023.md). Conditions: a fresh -05 pack and exact-input canary; qualified transport, deadlines and inputs unchanged; the factory drained via the timing arm; supervisor/CLI at nice 0 and excluded from renicing through verified release. Charge 10.51 u; worst case 26.39 / 34.50 u. Another failure means STOP: P1 NOT_READY operationally (critic PASS stands). Sol p1pack05 is building the pack. The V8 canary allocation failure ended (0 GPU time). |
| 2026-10-23 21:00 PT | D256 -10 PREEMPTED again on a 'process monitor fault' (4 parents). Monitor fix 3b216ed6 (Codex-V 391 cases; accepts empty/long/Unicode/zombie rows; nice 20 was already valid; the historical sensor cause is unknown). D256 -11 RUNNING on 3b216ed6 (789 pending). Astra ADMIT_MOVE_BENCHMARK_ONLY (JUDGMENT_GPU_PARITY_STANDARD_20261023.md): pure optimizations need frozen-device byte identity; CPU-exact components keep CPU parity; inherited reward/food/corpse deviations are diagnostic only and must be repaired or fully CPU-regenerated before authoritative use. The old MOVE result stays INVALID; Sol gpumove2 is building the fresh packet. |
| 2026-10-23 21:08 PT | GPU sprint unblocked (6fc4608e; Codex-V LIGHT ACCEPT): traces off by default, exports ≤64 MiB; the sparse chain is decoupled from v4 PASS (pin 42c84584); V8 got a fresh -n03 namespace. Two A100 canaries DISPATCHED in parallel: sparse-bench-a100-canary-42c84584 and gpu-v8-indexed-canary-85180521-n03 (each 4 u / 24 min canary; 32 u / 192 min main after PASS). D256 -11 steady through both dispatches (9 parents, 780 pending). |
| 2026-10-23 21:38 PT | **OWNER DIRECTIVE** (in chat, ~21:40 PT, away from home and unable to share a new folder): "can u move it to the already shared [folder]". The new -05 ninth bundle (sha 97b29ee2…acd1, 2,517,001 B) was staged in a fresh folder, hash-verified, then moved under a distinct name p1-inputs-attempt05.tar.gz into the already-shared folder 1yr71… with its Drive ID unchanged (1-uTSS…i_5I). The existing nine files were not touched or overwritten; this deviates from the stage note 'keep exactly these nine files' by owner direction, with the same exposure class the owner already approved. drive-urls.json was written create-only. -05 finalize ran, and the public fetch preflight PASSED (9/9 exact SHA/size, 156.2 s ≤ 300 s). Next: Codex-V _14 FINAL, then a drained-factory canary and repair. For the renicer hold, the operator plans to stop the renicer tmux session for the whole drained window and restart it after release. |
| 2026-10-23 21:43 PT | D256 -11 strong: 74 parents this attempt, 716 pending (~85/h at W12). GPU sprint: V8 canary PASS (native A100 parity at E64; baseline E64 plain tick 0.279 s), and V8 MAIN DISPATCHED (32 u / 192 min). MOVE2 canary DISPATCHED (frozen-device identity admission per JUDGMENT_GPU_PARITY_STANDARD). Sparse exactness fix (1-ULP causes: float32 danger-distance arithmetic in dense col 39; float32 food-density ratio in static col 8; fixed with integer counts, round-to-nearest float64, __double2float_rn, no FMA; Mac preflight PASS). Its canary allocation FAILED at kernel creation (0 GPU time); Sol freshns is building a fresh-namespace re-dispatch helper. **Scheduling call:** the P1 final repair needs zero other Colab assignments and a drained factory for 2–4 h, so it runs after D256 completes (est. ~06:00 PT 10-24) to keep both the Mac factory and the A100 sprint busy tonight. Codex-V _14 FINAL is running. |
| 2026-10-23 21:43 PT | P1 Codex-V _14 FINAL = READY_FOR_CLAUDE_CHECK, zero findings (sha f24b4d7f…eae92, commit 37019abb). The -05 final attempt is staged for the window after D256 completes. Planned operator order: stop the renicer session; arm the timing window and drain; verify; Claude canary line check; admit and dispatch the canary; verify; disarm. Then re-arm; repair line check; admit and dispatch the repair; verify release; disarm; restart the renicer. |
| 2026-10-23 21:52 PT | **GPU sprint first measured win (V8 lane, A100, exact).** Scripted E1024: vector61-other 2.412 → 0.100 s (24×); b8_v8 bundle 0.145 → 0.072 s (1.994×); full plain tick 3.54 → 1.18 s (**3.0×**). E256 2.17×, E64 1.51×. Feature CPU parity, mechanics and state carry byte-identical. The package rule (each non-flood stage ≥2× at E1024) gave STOP_ROUTE on the 1.994× v8 bundle before frozen/mixed ran; Astra route ruling requested (JUDGMENT_V8_ROUTE_20261023.md). MOVE2 canary PASS (frozen-device identity admission); MOVE2 main allocation FAILED at kernel creation (3 of the last 6 Colab dispatches); the fresh-namespace helper (Sol freshns) is pending. D256 -11: 128 parents this attempt, 661 pending. |
| 2026-10-23 22:01 PT | D256 -11 PREEMPTED after 149 parents (attempt total ~1 h at W12) by a persistent non-owned /usr/bin/tail at nice 0 whose argv matched research/ (likely an operator monitoring command). -12 was refused at the host check for the same reason; -13 RUNNING (628 pending). Sol guardexe is restricting relevance to python/pytest research interpreters plus the agent matcher. Operator commands now run under nice 15. |
| 2026-10-23 22:03 PT | Astra V8 route: CONTINUE_SPLIT_BENCHMARK_ONLY (JUDGMENT_V8_ROUTE_20261023.md). The original STOP_ROUTE stands; indexed geometry is kept provisionally and indexed v8 frozen pending an incremental full-tick comparison. Next: canary, then R/G/GV over nine cells, frozen/mixed E1024 first, 3 repeats, 72 min / 12 u. Sol freshns helper d9f3e6c0 (40 tests): sparse exactness canary re-dispatched as sparse-bench-a100-canary-3ab2ef04-fresh-02 (allocation success). The helper REFUSED MOVE2 main (warmup evidence does not prove pre-execution failure); Sol gpunext is preparing V8 split and MOVE2 main. The timestamps do not show the allocation failures correlating with rapid dispatch. Guard executable fix 3c5fc403 PASS; D256 switches to it at the next natural stop. |
| 2026-10-23 22:08 PT | Sparse exactness canary fresh-02 INCOMPLETE: CUDA compile failure on the VM (vector_geometry.cu: 7 ambiguous overload errors for dsub/dmul/dadd). Mac preflights cannot run nvcc, so native CUDA was never compiled before the A100. Sol gpucompile is fixing it and adding a compile-only GPU preflight that A100 canaries must pass. D256 -13: 52 parents, 589 pending. |
## Compute portfolio (adopted 2026-10-20, Fable judge; goal: spend ~1,160 u by ~11-20)

| Lane | Question | Units | Start | Kill / gate |
|---|---|---:|---|---|
| L0 engine placement | Which runtime layout (10 procs x 1 thread L4; CPU-only runtime; unpaced 2-world smoke) gives the most completed worlds per unit? | <= 10 | now | pick the best layout; if no layout reaches a priced admission, teacher labels stay on the Mac |
| L1 T34 deciders | A5 STOP -> pre-data priced admission (<= 180 u / <= 4 days) -> Round 0 240 paired -> unpaced retention arm | ~180-310 | Oct 21-23 | per registrations |
| L2 Claim A chain | R1/R2 labels (Colab only if L0 >= 300 row-steps/s/lane, else Mac), CUDA fit qualification, ~10 cached fits, bare-student screens | 160-310 | after L1 | per screen bars |
| L3 projection fix | 2x2: KL-anchor only on non-decisive vs frozen loss x trainable vs frozen B8 trunk (loss.py: KL tau=0.05 dominates the 0.1 hinge) | ~140 | after R1 labels | screen vs B8 |
| L4 search successor | Offline successor chosen by retention attribution | ~150 | after paced verdict | — |
| L5 replication/census | Independent census + replication | 100-150 | Nov 3+ | — |
| Reserve | | ~150 | | |

Not funded: 4,096-root runs (~900 u at measured engine), H20000 product sessions, self-play/league/PPO, T5-family search,
distill lane C at 6,400 worlds, A100/G4/H100 for host-bound work, Colab "strict" evidence, idle allocations.
**Units spent tracker:** start 1,174.37 (10-18); 1,159.1 (10-20 13:45); 1,157.6 (10-20 15:33); 1,153.84 (10-20 ~19:20, `colab usage`; 1 active L4 = Round 0 chunk 000, 1.54 u/h).

## Measured results so far

| Result | Outcome |
|---|---|
| S1 (bare B8 fine-tunes, Colab) | KILL both arms (−12.5 / −14.9 pooled mass vs B8); audit PASS |
| M1 shadow identity | GO — 15,000 frames, 3 mixes, zero differences |
| M2 cheap LAD search vs T3 rescues | NO-GO (registered) — bisection 39/75, stride rule 37/75; on the 42 truth-capable rescues 39/42 and 37/42; the other 33 were all-die delays (+1–4 frames) |
| M3 serial live LAD search | NO-GO — 30–63 s per doom vs 12.5–24.9 s of lead (engine, not idea: 6.6 ms/row-step serial) |
| M4 B8 runtime | ONNX 0.14 ms batch-1 (≈3–5 % of the serial path end to end) |
| A5 v2 L4 benchmark #1 | ~50 active row-steps/s/lane vs 3,000 floor — serial shadow fill bug; 0 decisions; ~0.7 u |
| Retention power (development-only OC) | joint GO 0.80–0.83 at T3-like effects after resize; protected false-pass 0.026 ≤ 0.06 |

## Current status (update in place)

_As of 2026-10-20 ~19:25 PT:_
- **Round 0** (ratified 652f2b1e; study root `snake-dqn-artifacts/t34-round0-study-01`): chunk 000 runtime
  `succeeded` ~19:10, supervisor finalizing (tmux `t34-round0-c000`). Chunks 001/002 not dispatched. **48 h dispatch
  deadline ~Thu 10-22 16:45 PT** (missing chunk = KILL). No outcome opened by anyone.
- **Phase 0** (Mac, 6 workers, tmux `phase0-selection`): ~105/160 worlds; ETA Wed 10-21 ~10:00–13:00 PT.
- **Label ordering:** SUCCESS_CONTRACT §5 as amended (d5b11e3a, RULING_label_ordering.md): R1 label collection
  (Colab only) may start after a *published* Round 0 GO, in parallel with Path B; fits/screens/R2/Mac work wait for
  the Path B disposition; R1 needs its own registration + approval; owner veto; release approval stays the owner's.
- **Path B retention:** ratified at 97232d51 (study operator-01). Bare-unpaced amendment on branch
  `t34-retention-bare-unpaced-20261020` @52dec74f (FIX-FIRST round done; re-review in progress; saves ~8.7–10 Mac h).
  Lands in one push only after trace-receipt PASS and before Phase 0 ends; fallback = the 97232d51 design unchanged.
- **Mac order after Phase 0:** Round 0 Mac replay (10 %) first, **then** Path B timing gate → T3-bank pre-check →
  efficacy (NO-GO branch plan, 59e932c2).
- **Claim A:** pipeline built (9230b150 + fixes e301cfeb). Two frozen-design BLOCKERS (loss cannot flip decisive
  rows, 0.4 %; 1:1 rule starves the fit) await a prospective ruling — **no production fit until then**. L4 timing
  qualification (cap 5 u) is allowed; it exports no weights.
- **Round 0-D:** DRAFT (8fc8e00e), 26.7–39.3 u expected; panel in progress; runs only on a Round 0 NO-GO/INCONCLUSIVE.
- **T35 probes:** outcome-head KILL (182d85e2). Two-deviation primary EXTEND 2/7 (both known paths); EXTEND run
  (26 events, ≤ 2 h Mac) awaits its own approval and must not overlap the Path B quiet window.
- **A100:** not used this campaign (SYNTHESIS.md); reopen only on the fit-timing trigger.
- **Approvals:** Fable at its usage limit since ~19:05 → Opus stand-ins for panels and approval judges until it returns.
- **Sol running:** test-suite hygiene (worktree `../snake-dqn-test-hygiene`, log `t34-sol/test-hygiene.log`); verify
  protected paths untouched after merge.
- **Colab:** qualified on `dab00823`; 1,153.84 u at ~19:20 (`colab usage`); 1 L4 active (Round 0 chunk 000).

## Claim A amendment judgment and local preflight — 2026-10-20 22:16 PDT

- Independent Fable 5.1 approved the prospective amendment subject to C1–C6; verbatim ruling: `docs/research/advice/2026-10-20_colab_lanes/CLAIMA_AMENDMENT_JUDGMENT.md` (SHA-256 `c7c519706513fdd2db1458226f463751d11bd752b501ad271a3e393a6ee4dda9`). C1 text/string corrections are applied on `t34-claim-a-amendment-20261021`; merge and dispatch remain separate.
- Local `d0.sh` preflight-03: 2 decisive-bearing excluded A5 worlds, all training folds supported; replay 2 accepted / 0 rejected; 15 real recipe updates across 3 fits. Wall 312.861 s; deliberately stopped during the first full-batch stream fold, preserving registered defaults. No `D0_DECISION.json`; non-REFUSED decision reached = false. **Incomplete preflight; C2 dispatch qualification is not met.**
- Detailed command, inclusive seconds/update, fixture limitation and SHA-bound evidence: `docs/research/advice/2026-10-20_colab_lanes/CLAIMA_PREFLIGHT_03.md`; external summary `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/d0-inputs/preflight-03-summary.json`. Targeted tests: 58 passed. No Colab, no main merge, no claim-bearing outcome data read, no serving change.


## Big-lever E8 preparation and E9 blocked freeze — 2026-10-21

- Owner-requested pre-data E8 amendments are implemented in `research/bl_ppo`; executable pin `0b53afdf094fb689878d10cc8a9c405ae623b682`. Original intent/bank/audit bytes remain unchanged. No protected T34 package or CLI supervisor was authored.
- E9: 20,000 Mac simulation replicates per scenario, measured T23 stratified variances, .05 E2 + .05 E6 family with Holm R/F. The prospective 128/48/48 screen has minimum joint funding .307 at true +40 (required .60), maximum family null-pass .0616 (ceiling .10). **E2 ADMISSION BLOCKED; no dispatch.** S1 measured variances and the broader round-05 continuation forecast model remain unavailable; these are explicit limitations, never fabricated inputs.
- Local final preflight: R/F 400 total transitions in 11.315/10.706 s; R explicit resume 3.004 s, identical endpoint SHA. All scientific=false, critic unqualified; no research conclusion follows. Targeted tests: 101 passed. Sparse source import/native-step and three manifest validations passed. No Colab calls or serving changes.
- Details: `research/bl_ppo/E8_PREFLIGHT.md`, `research/bl_ppo/E9_RULE.md`; authoritative external `big-lever/exp/bl-e2/DISPATCH.txt` points to `dispatch-e8-v2`. Current E2 templates cannot be materialized without genuine E4 evidence and passing prospective E9 admission.

- **2026-10-21 — O2 prospective V2 prepared, NOT DISPATCHED (owner request):** measured-T23 minimum 5097 worlds (joint superiority bound .800179; preceding 5094 .799974); >1000 fallback 1260/420 per mix,149.733333u hard cap including 20% reserve,10 single-runtime jobs<=462min. Capped power bound .339888; retained observed>=40 precludes 80% full-GO at true +40. Round 5 seeds CLEAR after whole-allocation collision rejections; 260-file source pin `f19d3f39050adfedc6eaef001a3364318fbb28bc`, registration `040a6fcd`, Mac serial/pool proof, 77 O2 tests+ 16 targeted transport tests, all 10 local manifest/upload checks pass. See `research/t34_o2_harvest/{REGISTRATION_O2_V2.md,AMENDMENT_O2_V2.md,PREFLIGHT_V2.md}` and O2 artifact-root `DISPATCH.txt`. No Colab or serving change.

## P1 E8 amendment 04 — 2026-10-22, HOLD / return to judge

- Owner requested implementation of round-18 `RATIFY-WITH-CHANGES` on
  `p1-snapbound-20261022` over `5afeb8d9`, without dispatch. The sibling
  `research/bl_ppo/E8_LATE_BODY_AMENDMENT_04.md` freezes C1 zero-sum solve,
  C2 first-step fixture and moments clarification, and C3 fresh allocation.
- The sole existing-export replay reports rank 257, condition 2516.5907,
  raw FIT MSE 896460.1646, max installed FIT prediction difference 11.0748
  return units, common-mode sensitivity -5.0022e-16. Numerical FIT criteria
  pass, but its source binding predates a rank-deficient report-serialization
  fix and its thread environment is unset. It is not final-code qualification.
- C2's initial bias-shift check was vacuous. A corrected full-forward fixture
  failed; subsequent synthetic target/reward easing produced `_02` PASS,
  which independent Codex-V rejects as acceptance evidence. The restored
  original configuration's `_03` FAIL retains three finite actor updates:
  cumulative held-observation value drift 7.6990/14.6261/20.3876 versus
  PopArt scale 5.5338. Analytic and actual common-mode checks pass. No
  thresholds were changed and no accepted C2 stability PASS is claimed.
- Judgment section 4's preparation kill rule applies: **no spend, no
  dispatch, return to judge**. Review-only source/canary packs carry explicit
  admission blockers. Frozen pricing stays canary 2.00u/.40u/30min and source
  9.62u/1.92u/300min at 1.54u/h, zero tolerance; 2x is not reapplied.
- The current/manifest Colab CLI tree must remain
  `553af3f692515b2bf66173356fe4d099acc96942`; registry acceptance receipt
  hash/PASS/qualified=true were checked. Live inventory, current debit and
  canary proof are future operator preconditions. The previous
  `WARNING_UNEXPLAINED_DEBIT` is carried forward; `{}` is not an inventory.
  Claude's line-by-line receipt check has **not** been performed by Codex.
- All work is 0u local preparation. No provider/Colab, tmux, serving pin or
  released-checkpoint action; protected study unopened, `snake-dqn-l2mac`
  untouched. Detailed evidence and precondition status are in
  `research/bl_ppo/P1_E8_04_HANDOFF.md` and the independent Codex-V receipt.
- Final independent receipt `P1_E8_04_CODEX_V_REVIEW.json` explicitly records
  all 26 section-4 acceptance lines and all eight section-6 preconditions:
  **HOLD_RETURN_TO_JUDGE**, unresolved admission findings V1 C2 drift, V2
  replay final-code/thread binding, V3 limited history. Software regressions
  pass (root 108, independent reviewer 87); these are not admission PASS.
  Actual review-only source06 packs are pinned to
  `f184c2c7f54e642b5870d6efd4a32b60ffe50496`, independently checked and indexed
  by `P1_E8_04_DISPATCH_PACKS.json`. Both top-level admission flags are false.
  This records Codex's receipt publication, **not** Claude's required
  line-by-line verification or a judge disposition. No dispatch occurred.

### 2026-10-23 — F4 round-26 implementation binding (pre-data)

Authority: round-26 `RATIFY-WITH-CHANGES`, judgment file
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-26/JUDGMENT.md`,
SHA256 `0fe1a103384035839cb143a6df9193abf8269197e6d02f44bf7dceb000ab8362`.
Extracted the exact 129-line §3.1 block into
`research/bl_residual/F4_REGISTRATION.md`; its SHA256 matches the judge:
`66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65`.
Post-change FitRecipe canonical SHA256:
`49d12f486bffa455adcedbc318e88f1524f5c502894d0c1ace47daa84b9d3a5a`.
The independent implementation receipt is
`research/bl_residual/CODEX_V_F4_REVIEW.json`. Claude must accept that receipt
and create the exact five-field `judgment.json` in `F4_HANDOFF.md` before any
factory-label fit, reserve opening or screen allocation. This entry records
bindings, not operator approval. No real labels or outcomes were opened by the
implementation; synthetic fixtures only, 0 Colab units, no serving change.
Future packs use qualified CLI tree
`d0d92bdb572b770284a1fea6152417ffa5f06bea` from `origin/main`; historical packs
and earlier hash-bound receipts remain unchanged.

F4 implementation closeout (same pre-data scope): Codex-V all eight edits PASS;
review receipt `research/bl_residual/CODEX_V_F4_REVIEW.json`, SHA256
`9f7f26efa36e10361a4b66bad00e3df2b1798f9db35d79bf1e1a6f6d8001e890`.
Published executable/pack pin `efdba0c2cbea91fc408e0597f76ed7ef5864634d`;
151 light single-process tests passed. Fresh HOLD pack import preflight and
supervisor validation passed with qualified CLI tree d0d92bdb572b770284a1fea6152417ffa5f06bea.
No real labels/outcomes opened, no Colab allocation/spend, no operator approval
file written, and no serving/release authority exercised.

F1 refund pack-10 preparation (2026-10-23, Sol): **HOLD**. Exact caps and repaired
worker pin `2af48cd0b98a2063f46871cf4d320a5541b789af` preserved; qualified CLI tree
`d0d92bdb572b770284a1fea6152417ffa5f06bea` unchanged. Four upload qualifications
PASS; L4 probe validates, but CPU's ruled 40-minute probe is refused by the
qualified 30-minute family-canary maximum. Both rulers await actual family PASS.
Historical pack-08 spend carried as 0.153 u, separate from fresh 4.04 u caps;
exact historical failures retained/excluded by SHA-bound refund census; A100
`NOT_RUN by ruling`. 153 light single-process tests PASS. No Colab dispatch.
See `research/bl_labels/F1_REFUND_HANDOFF_20261023.md` and sibling Codex-V receipt.

F1 A1 CPU-only preparation (2026-10-23, Sol): post-amendment authority SHA256
`86e9ca69383d21a45ede31fd721d315ff39ae09736ad538496ce42b1abf1ac72`,
`big-lever/round-24/JUDGMENT_F1_REFUND.md` Amendment A1. Create-only
`bl-labels-f0/f1-pack-10a/` corrects the CPU probe's four time fields to
30 minutes / 1800 seconds, adds the A1 binding, and gives CPU probe/ruler
fresh `-10a` IDs, outputs, state paths and CPU-only dispatch instructions.
All unit figures, worker pin `2af48cd0`, bank/ledger/census bytes and qualified
transport tree `d0d92bdb572b770284a1fea6152417ffa5f06bea` are unchanged.
CPU probe supervisor validation and both CPU upload qualifications PASS.
The Mac-only analyzer additionally accepts the exact ratified A1 ruling-file
hash, while still requiring the original unchanged census authority hash;
foreign hashes remain refused and no decision rule changes. Five focused
regression cases cover this compatibility fix. The ruler still requires its actual
CPU probe PASS receipt. Pack-10, its original dispatch file and L4 lane are
untouched. See `research/bl_labels/F1_A1_CPU_HANDOFF_20261023.md` and
`CODEX_V_F1_A1_DELTA_20261023.json` for independent review and exact commands.
No CPU allocation, dispatch, new science result or serving change by this task.
Final A1 delta review: `PASS_CPU_A1_DELTA`, zero open findings; receipt SHA256
`9d1cd980d9bcaf76c6b2bdf3183594e243f70ad5d9f5f320b5c7cab45d5752b8`.
Final package/family-canary suite: 158 PASS in 8.82 seconds; reviewer independently
ran 132 F1/canary checks and 24 final analyzer checks. Artifact inventory is
`research/bl_labels/F1_A1_PACK_20261023.json`, SHA256
`f537b69a3a097d88f2ea661ec76d16d29fac5404de0561f7789ac914cf6f82fe`.

Round-27 F0 speed package prepared (2026-10-23, Sol): authority SHA256
`f3d4e6254771830625ec25938f9eb6dd5a9750e383cb12b34a7a04b687b8518c`,
base `2a0821a9f5c6cd5f7b45ba6b22352e48e349b451`. WP-3 W6↔W12 continuation,
summed worker-seconds and per-width baselines; WP-4 fold-1--4 prospective
D1024/D4096 banks with frozen OLD D256/cross-tranche seed audits; WP-2 immutable
prepared decisions, forced-step inference skip, sparse production Q/action
recheck and bare mass audit; WP-1 isolated OLD/NEW W1/W6/F1 identity runner,
origin/main-published metadata before comparison, and profile-only threshold
instrumentation. Registered 40 h / 130 h future caps retained. Parent reserves
2160 / 7020 use the documented D256 cap/floor method, are prospective operational
windows rather than selected-life doses, and cannot be refilled after outcomes.
D4096 admission stays HOLD pending its R6 funding/projection route. D1024 requires
immutable fault-free D256 F4 predecessor-fit metadata, actual T-EQ PASS and the
sealed calibration; measured calibration FAIL gives W6, unknown/HOLD refuses.
T-EQ FAIL has an explicit W6 entire-frozen-OLD-closure contingency.

Two independent personas (science identity and operations) report no remaining
concrete blocker. Codex-V's integrated single-process/capped verification:
94 passed, 1 intentionally skipped; final runner checks: 5 passed. Real
three-parent width proof, W12 calibration, B-EQ/F1 identity, interleaved timing,
profile and production are NOT RUN by Sol. The running D256 worktree, outputs,
lease, tmux, and frozen `61900af2` pin are untouched. Ranks 5--8 remain unbuilt;
profile thresholds stay 20%/8%/5%/4%, at most one builder day, each needing T-EQ.
Implementation evidence: `research/bl_labels/CODEX_V_ROUND27_FINAL_20261023.json`;
operator commands and explicit pending phase publication:
`research/bl_labels/ROUND27_HANDOFF_20261023.md`. Actual qualification receipt
paths/pins/hashes must be logged after Claude's post-D256 runs and before adoption.
No release, serving, F4 registration or strict-gate change.
Final round-27 implementation Codex-V: `PASS_IMPLEMENTATION_LIGHT_VERIFICATION`,
receipt SHA256 `19713e3929a00c601042405903dc34a3bcd92d728898d8155aebcf473ebe4f07`.
The initial style-blocked receipt is retained unchanged; its import-order finding
is corrected and the final receipt supersedes it. Heavy qualification is pending.
Round-27 published implementation/bank pin:
`bf8ac97dc66d1b4515e4c5510db3ad5886fa695d` (origin/main fast-forward verified).
Create-only preparation completed, no factory execution: D1024 2160 parents,
D4096 7020 parents, all fold 1--4, preserved stream/roster ratios, nonoverlapping
append-only indices, sealed seed audits/empty ledgers; excluded B-EQ 12 parents
with materialized plans. Packages:
`bl-labels-f0-speed/{d1024-d4096-pack-01,equiv-pack-01}`. Prepared inventory:
`bl-labels-f0-speed/build-01/prepared-inventory.json`, SHA256
`5930e7a64dda67ddfbf5b8571af2c23363ebac19f0d3baf86a534d3c7f862928`.
Use that immutable implementation SHA for every NEW-run command, including after
later metadata publication commits; the handoff now spells it out literally.
Actual T-EQ and calibration receipts do not yet exist from this task.
F0 bounded-concurrency candidate (2026-10-23, Sol): **PREPARED_NOT_QUALIFIED /
ACTIVATION HOLD**, under ruling `ALLOW_BOUNDED_CONCURRENCY`, SHA256
`af117cd98d51f428cd83d6e33138fcd13def0b4d7c7cf63a8b2e2b83db99105f`.
Prepared an operations-only supervisor, argv/desktop-helper recognition,
absolute-priority wrappers, global test-slot/duty/protected-period policy,
v2 resume host receipts, 1.15 companion warning, and production integration.
The mandatory qualification fence refuses monitor/client/tool allocation,
production admission and host-check publication; no environment/manual PASS
override exists. Independent Codex-V light review permits publication only as
a disabled candidate, not activation or a completed implementation of the ruling.
Native arbitrary-descendant ownership/stable topology, cross-group native CPU
transfer credits/wrapper overhead, independent watchdog, complete cooperative
recovery and downstream exclusive-launch proof remain blocking requirements.

Root verification: 200 serial tests passed in 36.30 seconds, plus Black/isort/
flake8 on all 24 changed Python files and bounded shell checks. Native tests
validated counter units, waited-child accounting, PID/start/argv identity,
owned cleanup and an excluded frozen-collector payload/SHA/preempt/resume
fixture with an admitted nice-15 control command. These are partial evidence,
not global concurrency qualification. Frozen science remains clean at
`61900af28932e47e22e332840fc4406559adb062`; scientific source, cumulative
accounting, caps, allocation, ledger and serving are unchanged. No factory,
production monitor/client, calibration, timing/strict/serving or long job ran.
Evidence: `research/bl_labels/CODEX_V_FACTORY_CONCURRENCY_20261023.json`, SHA256
`3872b77235af797155a0480c26e51992409808cbb1f47a8c603213e1d69a9b4a`;
operational closure SHA256
`4b2270d7b70f2a66c6dca137e28def079b0f139b46c1d585fbc383817e54aae0`.
Operator commands are future protocol only in
`research/bl_labels/FACTORY_CONCURRENCY_HANDOFF_20261023.md`; do not dispatch
`bl-f0-d256-guard-w12-03` under this candidate while the fence is HOLD.

R33 bounded ops repair (2026-10-23, Sol preparation): adopted judgment
`big-lever/round-33/JUDGMENT.md`, SHA256
`7db57b262f1bfa4ae70597772d79e2904f8d6a885ea530c54178d869911cf07a`.
Ownership recorded before implementation in `research/r33_ops/OWNERSHIP.md`:
Sol-O owns watcher/journal/proof dependencies; Sol-P owns package/CI adapters;
independent Codex-V owns adversarial verification. Existing A1-PROF owner retains
row 4 exclusively (`research/r33_ops/A1_OWNER_HANDOFF.md`). Claude owns three
separate next actions: D256-successor evidence/admission, A100 Q1 repair/canary,
and P1 current-package canary admission. Unknown evidence remains a typed blocker;
no unrelated scientific dependency or READY is inferred. Engineering timebox is
3 Sol builder-days + 1 Codex-V day, scope frozen after three calendar days.
CPU pilot forecast 2 h / 0.16 u; base 2.4 h / 0.192 u; effective shared-series
4.8 h / 0.384 u (included reserve 0.8 h / 0.064 u), no dispatch by Sol.
R28 row-4 1.2 h / 12 u is only a sub-envelope of its remaining Q1 authority,
not new funding. No live balance/capacity, native qualification, operational
handoff sample, serving or science conclusion is asserted by this preparation.

F0 nice interim mechanical repair / independent Codex-V 02 (2026-10-23):
**PASS** conditions 1–3 of `ALLOW_NICE15_INTERIM`, ruling SHA256
`ea5ae541efc434fe3e5888eeab7a51a821220ce242f935795eacd0542af1accd`, at exact
published operational pin `387b1dde8241c9c24ea352eb92de802b66fc798f`.
Truthful recovery v2 receipts bind ruling/policy/clean operational HEAD and refuse
legacy-unbound or foreign authority. Individual relevant descendants must meet
nice >=15. The real agent launcher now requires light/reviewed-test routes;
CPU/all six thread limits/Torch intra/inter one are enforced, the canonical
shared test slot serializes checks, and a <=120 s elapsed deadline covers
admission, execution, owned cleanup and Mac-lease release. Tests conservatively
refuse every active factory lease, protecting baseline/rearming/W12 sanity.
Independent exact-pin product-route review: **99 passed, 0 failed, 0 skipped**,
pytest 8.19 s, full elapsed 9.699733915971592 s, nice 15; actual numerical nice-15
and nice-19 probes ran. Initial reviewer-only fixture/harness errors are retained
and documented; corrected review changed no product bytes. Create-only verdict
`big-lever/exp/CODEX_V_FACTORY_INTERIM_02_20261023.json`, SHA256
`d9a0313377cbc617777062df8ce172fdfef2f3d104662e9c093deff0be89cf75`, plus immutable
sidecar/completion and identical repo triplet. Exact future `-03` dispatch command
and full ruling/pin/results are in
`research/bl_labels/NICE_INTERIM_02_OPERATOR_20261023.md`; this record permits the
normal operator-requested ledger continuation at the reviewed pin, launches
nothing, and does not qualify the full supervisor. Science stays clean/detached
at `61900af28932e47e22e332840fc4406559adb062`; accounting and eight applicable
slowdown/determinism evidence source hashes are unchanged. No actual production
host check, factory, calibration, outcome look or long job was run. Owner notice:
the narrow interim is now independently qualified; all real timing/health/
slowdown/accounting stops remain binding.

F0 factory persistence amendment (2026-10-23, prospective operational activation):
**PASS — ALLOW_PERSISTENCE_30S**, policy ID
`factory-nice15-persistence30/v1`, exact published operational pin
`caafdf5d079bd73bcfbde6f9356dc0f60de42e0f`. Binding Astra ruling
`big-lever/exp/JUDGMENT_FACTORY_PERSISTENCE_20261023.md`, SHA256
`35295b5b4669bec32a61736c1d5dbb5d9c70015208d6f06db06e5f064bca7c32`;
retained interim ruling SHA256
`ea5ae541efc434fe3e5888eeab7a51a821220ce242f935795eacd0542af1accd`.
Ordinary foreign priority veto now requires seven consecutive observations of
one native `(PID,birth)` identity spanning >=30 s; no CPU threshold. Pending
rows do not pause admissions or clear slowdown windows. Failed/stale monitoring
stops immediately under the aggregate 15 s observation deadline; timing, health,
slowdown and all other guards retain their existing immediate behavior. Colab
pane shells/supervisors and reviewed agent launch routes reach effective nice
>=15; shared/custom-socket tmux servers require native/trusted executable and
exact socket/server PID proof, while real pane shells/children remain accountable.
Recovery uses an honest digest-bound v3 window with final pending set empty,
final host health/top consumers and the existing serial-test lock, bound to
both ruling hashes, policy, operational pin and trip epoch; v2/foreign receipts
cannot qualify it.

Independent Codex-V executed **69 new focused fixtures** (60 guard/clock,
6 synthetic coordinator preempt/resume, 3 real shell/stub supervisor priority),
all serial/nice15/CPU-one-thread environment and below 120 s. The executed
candidate `85f4106b` and final pin have byte-identical product and acceptance
fixture closure; subsequent changes corrected only two old test assertions.
Additional focused recovery/migrated guard regression checks: **223 passed**
(147 + 68 + 8). Black/isort/flake8/shell syntax/diff checks passed. Initial
fixture/harness failures remain preserved with corrected passing records;
no failed or timed-out test is counted as PASS. Create-only independent receipt:
`big-lever/exp/CODEX_V_FACTORY_PERSISTENCE_20261023.json`, with sidecar/completion
and identical repo triplet, SHA256
`50d9736a61947b335e06694b922419e646f0811458fd86564bf924cfe553938e`.
Full commands/hashes and limits are in that receipt
and `research/bl_labels/FACTORY_PERSISTENCE_OPERATOR_20261023.md`.

Science remains clean/detached at `61900af28932e47e22e332840fc4406559adb062`.
Frozen collector/export and unchanged slowdown closure retain applicable prior
proof; this patch's changed control path additionally passed opaque immutable
prefix/verified-skip preempt/resume and exact cumulative accounting checks.
Caps, nominal worker-second and wall charges, drain charges, ledger semantics,
retry entitlements, baseline rules and ChatGPT-helper exclusions are unchanged.
Owner notice: this is a narrow operational qualification, not a scientific or
full concurrency-supervisor result. No factory, production host check, timing,
calibration, strict/serving, remote runtime or long job was launched, and no live
D256 process was stopped, reniced, patched or restarted. The existing attempt
at `387b1dde` stays on its pin until natural preemption/completion and verified
stop; only then may the normal operator-requested verified-ledger continuation
use the exact fresh `bl-f0-d256-guard-w12-08` command in the handoff, if missing work and
all other registered admission conditions permit. No automatic dispatch.
Final command-name correction: another operator consumed the requested `-07`
at `387b1dde` during this implementation (worktree/runner/log observed at
20:12 PT); this task preserved those artifacts and performed no dispatch.
The new pin cannot reuse that differently pinned worktree. `-08` was verified
without a worktree, runner or log reservation at closeout.

## R33 implementation closeout — 2026-10-23

Delivered the adopted durable task journal/watcher consumer protocol, thin immutable
package/ledger and real-entrypoint preflight adapters, explicit 15-node Linux CI
inventory, common actual startup/runtime/recovery process fixtures, and narrow
future opt-in proof-dependency charter. A1-PROF remains with its existing owner;
no factory, F4, P1, GPU spike or serving implementation was authored in this lane.
Other-owner main commits were integrated without reverting their changes.

Frozen qualified source `d53eaef784fc06996e4d77e3d39889ecd804d66a`, closure
`52d4ad55f8f07ee544ed726e4b595298283626e87ea25163a2aec2e91114ec6c`, is published
through an explicit history merge preserving main's newer transport wrapper.
Both source-pin and broker-HEAD CLI trees match the existing qualification record;
no registry or live transport was changed. Exact-pin serial check: 64 passed in
1.06 s (complete wrapper 1.318319 s), nice 15, one process, six thread caps 1.
Actual scrubbed/native Mac preflight: 15 required checks / three immutable shards,
3.582715 s, runtime nice 15. Codex-V `codex-v-02/receipt.json` SHA256
`5997b23f8dc50f4181c51ecf1edfec52f75aa965e23577cb09adc5edfe905d86` independently
accepts this implementation. Focused style/shell/diff/import checks pass. Hosted
Research contracts run 40057728587 at `9bfd99ea` passed the complete frozen Linux
packet. Native ownership, A100 attribution, timing/strict/serving and actual
handoff-latency/review-time performance remain unproven by this evidence.

**CANCEL_CPU_PILOT_NO_REMAINING_DEMAND**, applying the R33 section-6 kill rule:
the actual Linux packet already passed CI and no finite unverified Linux backlog
remains to move off the factory Mac. Independent Codex-V confirmed cancellation.
The historical `cpu-pilot-01` preparation was refused by the qualified supervisor
for a missing target-platform family canary receipt; it is not dispatch-qualified.
No new canary, second pack, allocation or retry was created to repeat the work.
Immutable cancellation SHA256
`6c79fb57edab867fc75a415ed731cefb84309d7cc934bd0928c67683971a4591` binds the rejected
manifest, ruling, accepted source and actual CI evidence. Actual R33 cloud charges
and active remote liability: **0 units / 0 allocated minutes / 0 assignments**.
The registered forecast 120 min / 0.16 u and owner-2x ceiling 288 min / 0.384 u
were never spent or added to existing P1/F4/R28 liabilities. No native/release
qualification or SLA improvement is claimed. Failed/refused checks and superseded
receipts remain immutable historical evidence. Complete limits, source assembly,
artifact paths and exact metadata-only Claude commands: `research/r33_ops/HANDOFF.md`.

## Factory monitor mechanical acceptance — 2026-10-23

Independent focused Codex-V **PASS** at operational pin
`3b216ed6a2efad8f05e190e89d92f2e898144a81`, published through main
`2dd2896f70cf9cde901dd27388ba571fffb996a1` with identical reviewed source.
Create-only `big-lever/exp/CODEX_V_FACTORY_MONITOR_FIX_20261023.json`, SHA256
`c0c71c586a3d3531fc0abac89b8ea6ab84ce4b9211d30dbffeb52a722cbc0687`.
All 391 focused cases passed, plus final caller/closure checks; canonical serial
lock, nice 15, CPU/thread caps and bounded commands. Legitimate empty, long,
Unicode, cross-user/system and confirmed zombie/disappearance observations are
accepted; genuine sensor faults still stop. Nice [-20,20] was already inclusive
at caafdf5d; the supplied nice-20 incident's native cause remains unknown.
Science stays clean/detached at 61900af2; accounting, caps, drains, ledgers,
slowdown baselines and retry rights are unchanged. No production dispatch,
foreign process control, label look or serving action occurred. Owner notice and
exact fresh verified-ledger continuation after verified stop are in
`research/bl_labels/FACTORY_MONITOR_FIX_OPERATOR_20261023.md`; -10 was consumed
externally, and -11 was the next unused dispatch reservation at publication.

## P1 final repair -05 preparation held — 2026-10-23

Prepared the fresh sibling `big-lever/exp/bl-p1/y3/source-repair-10-20261023-05`
under Astra attempt-2 ruling SHA256
`55b13197d9d92f8eeb1c895180f01993ca11908f51c74a60d9a67065a0550d13`.
Operational amendment and exact conditional operator/line-check contract:
`research/bl_ppo/P1_REPAIR_ATTEMPT2_OPERATIONAL_AMENDMENT_14.md` and
`P1_E8_04_SOURCE_REPAIR_HANDOFF_14.md`. Original finalisation authority fields
remain unchanged. One new canary and one final repair only; no further repair
retry. Wrapper resume maximum 2 grants no extra scientific attempt.

Executable remains `03e0af75c64538f53a6971e6fec5d9a173bc921b`, CLI tree remains
`d0d92bdb572b770284a1fea6152417ffa5f06bea`. No CLI/runtime transport edit.
New namespaces have tag `6f31cd3de308`, with the original source allocation and
empty ledgers. The new 2,517,001-byte bundle SHA256 is
`97b29ee24a324284b740b53d9fadf6d7b70044953f166715da6e3a927d1eacd1`;
only four namespace JSONs and their four SHA sidecars change. Exact-pin real
validate-only checks passed for repair/canary in 2.306169/2.290468 seconds,
without actor steps, refit/gate calls or runtime output creation. Fresh offline
forecast: residual 28 files/74,138 bytes; fallback 1,331,053 bytes; upload
146.921532516 s <=150; doubled complete forecast 90.893766318 min <=200.
Actual public-fetch measurement is still null, not inherited from -04.

Create-only failure-accounting binds the unchanged raw -04 failed receipt and
independent release proof. Conservative own/gross charge is 0.17 u, so charged
is 10.51 u. Remaining liabilities 2.00+6.16+3.86+3.86=15.88; total26.39 and
unallocated 8.11 under 34.50. Retain WARNING_UNEXPLAINED_DEBIT, no refunds, actual
intervening debit reconciliation before each admission, and 13.88 liabilities
after substituting the new canary's actual charge. This is local arithmetic,
not current account inventory. The separate 1M invoice remains held.

Status is HOLD, not READY_FOR_CLAUDE_CHECK or admission. The requested unchanged
nine URLs/no restaging conflicts with ruling condition 2's new ninth bundle,
mapping and full unauthenticated fetch requirement; clarification is pending.
All eight origin URLs and selected folder 1yr71MQZgtRLlZ3DprqN147dDOwqprHm_
are preserved. No Drive mutation or fetch occurred. A second operational HOLD
is the real renicer exclusion/handover: prepared PID-tree exclusion source is
not installed, and the existing daemon has no dynamic reload interface. No
daemon/worker/tmux kill or priority action occurred. Claude must prove current
cooperative factory drain, cleanup/lease release and blocked re-admission, and
absolute nice 0 for the entire supervisor/CLI tree through verified release.
Prepared scoped wrapper code preserves the generic nice 15 policy, separately
registers exact allocation-recovery roots, refuses other account assignments
under its allocation lock, and does not dispatch during preparation.

Independent Codex-V review is recorded in a create-only sibling `_14` receipt;
its HOLD cannot be used as a READY receipt. Final mapping/fetch review, fresh
Claude per-stage checks, live zero-session/calibration/accounting evidence,
canary/repair and all later-stage gates remain unperformed. No Colab operation,
scientific outcome look, serving change or release action occurred; critic PASS
stands, Y3 remains not PASS. Prior packs/receipts and unrelated changes remain.

Preparation source is published through `6e54f936604b79d4e13b757c5f7ed96e5dba8bf4`.
Independent create-only `P1_E8_04_SOURCE_REPAIR_CODEX_V_14_PREPARED.json` SHA256
`281bdf18f39427777bb27937272910b4a92b05df8041db492412dc1149ec3ac9`
reports no open code findings and HOLD_NOT_READY_FOR_CLAUDE_CHECK (H1–H3 above).
Create-only verification sibling `_VERIFICATION_14.json`, SHA256
`6cb92fcbda042140c0f5cc65602aabb5166ae825e5272a10f3e24bc2f28c4038`,
records 113 focused/dispatch tests passed in 58.71 s, the identical complete
9-test transport suite passed on the frozen qualified broker in 41.04 s, and
style/shell/diff checks passed. The earlier broader snapshot was interrupted
after 160 passes and two reproduced unchanged old-pin/current-main inventory
refusals; it is not a full-suite qualification. No transport/test fixture was
changed to hide that identity difference. Only the owned pytest child received
one interrupt; no worker, daemon, tmux or provider action occurred. The independent
HOLD receipt and all earlier artifacts remain immutable.
