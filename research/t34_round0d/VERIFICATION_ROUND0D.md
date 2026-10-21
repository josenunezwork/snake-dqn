# Round 0-D build verification (2026-10-20, pre-data; DRAFT)

This record is not part of the source closure. It was written before any Round 0 analysis existed:
`t34-round0-study-01/analysis/` was absent at commit and at push. No Round 0 chunk, pair, trace or outcome file
was opened. The seed scan excluded the live study root.

## Untouched hash-bound packages

`git diff origin/main...HEAD` is empty for:

- `research/t34_round0`
- `research/t34_pathb`
- `research/t34_exit_b8`
- `research/t34_exit_b8_fit`
- `research/t3_rollout_labels_20261015`
- `research/colab_cli`

Every reuse is by import.

## Bank and audit

- **Allocation:** 240 selection worlds (96/72/72) plus 2 A6 fixtures and 30 A5-D worlds, 272 seeds in total.
  Rows are identical to Round 0's `_rows` under the new namespace (tested).
- **Explicit disjointness:** the Round 0 bank (242 seeds), Path B retention v2 (177), Phase 0 (172), and the T3
  dense census (32). Zero hits.
- **Text scan:** docs/, research/, src/, scripts/ and the artifact tree (JSON/JSONL/MD/PY/YAML/SH/CSV/TSV,
  hidden and ignored files included); about 30 s single-threaded. Zero hits. Positive controls (one seed from
  each explicit bank) were all found. As a manual check, a Round 0, a Path B and a Phase 0 seed were each found
  in both research/ and the artifact tree, and a non-existent integer was not found.
- **FRP training exclusions:** 7 inventories, zero hits.

## Tests

`tests/test_t34_round0d.py`: **21 passed** (including one `slow` real-engine test), about 5 s on one thread under
nice 10. Coverage:

- **Controller:**
  - the selection rule matches T3's verbatim expression on 2,000 random cases;
  - Amendment 2 label roles;
  - the trigger clock: first L ≥ 600, every 5th decision, no reset on length dips;
  - all-die delays execute;
  - depth is truncated at H, and the teacher config is restored after each batch;
  - an empty advisory set falls back to bare;
  - candidates are advisory-legal only.
- **Bank:** fresh and disjoint, equal to Round 0's `_rows`; any collision rejects the whole allocation.
- **History** (real temporary git remote):
  - Round 0-D tags never match `refs/tags/t34-round0/*`;
  - the Round 0 ledger path is refused;
  - an in-flight attempt is skipped, and its analysis reservation is refused;
  - an abandoned (unlocked) claim is KILL;
  - a seal must be committed and pushed;
  - an interrupted analysis seals a KILL decision.
- **Analysis:**
  - `summarize` is the same function object as Round 0's;
  - a point-bar failure is STOP;
  - the descriptive decomposition never gates.
- **Sizing:**
  - A5-D kill below 8 worlds/h, fidelity HOLD refusal, and the 9 MiB export guard;
  - Round 0's chunk formula, with multiples of 10, estimate ≤ 180 min, and timeout ≥ 2 × estimate;
  - the 48 u ceiling charges unsettled attempts at their cap;
  - the dense R1 arithmetic.
- **Sealed labels:** refused for any purpose other than R1, and refused without a published GO. The public
  trace has its labels stripped.
- **Closure:** the source closure binds Round 0's rule documents, this pre-registration, and the NO-GO plan.

## Mac preflight (A6, real entrypoint; excluded fixtures; one thread; nice 10; commit `9d26fa3d`)

| Run | Worlds | Wall (`/usr/bin/time real`) | Producer elapsed | Per-arm rollout s (bare / teacher) | Metric s per arm |
|---|---|---|---|---|---|
| a6-01 | scripted fixture (H500) | **5.05 s** | 3.99 s | 1.31 / 0.84 | 0.19 / 0.19 |
| a6-02 | scripted + frozen fixtures (H500) | **9.71 s** | 8.73 s | 1.30 / 0.85; 2.02 / 2.04 | ≈ 0.2–0.27 |

`validate` on a6-02 returned **PASS**, bound to the source of `9d26fa3d`. The checks:

- every world is excluded and H500;
- public traces carry no labels;
- sealed exports exist and are hash-bound;
- the run used Darwin, one thread and CPU.

At H500 the hero never reaches L ≥ 600, so the dense search does not run in A6 by design. A6 sizes only the bare
and metric components: 10 × H500 gives 13–20 s bare per world and about 4.7 s metric per pair, below the 92 s and
8 s placeholders used in the ESTIMATED draft forecast. The search path is exercised by the real-engine test, by
the probes below, and by A5-D. Ratification needs a fresh, source-current A6 after the final commit, because
main has moved since this run.

## Pre-registration engine checks (development only, not evidence; Mac, one thread)

All of these used T3's own historical dense census receipts, b8_s3. No Round 0 or Round 0-D outcome was involved.

- **Prefix identity (env 0).** Bare stepping through the Round 0 harness reproduced T3's length, greedy action
  and advisory set at triggers 1913, 1918 and 1923. The first batch's per-action frames-survived matched T3
  exactly (300 for each of {1, 2, 4, 5}).
- **First changed decision (env 1).** At decision 2592, this harness gave greedy 1 → chosen **4** (a boost),
  matching T3 [2592, 1, 4]. Per-action frames {0: 300, 1: 297, 2: 300, 3: 300, 4: 300, 5: 300} were identical
  to T3's. The choice of 4 among the 300-frame ties exercises the highest-Q tie-break.
- **First changed decision (env 6).** At decision 3751: greedy 1 → chosen **0**, matching T3 [3751, 1, 0].
  Frames {0: 300, 1: 299, 2: 14, 3: 300, 4: 300} were identical.
- **Batch cost.** One full D = 300 batch of 6 candidates cost 2.3 s (scripted) to 3.5 s (frozen) on the loaded
  Mac, about 500–800 row-steps/s. T3 dense census worlds run about 600–716 batches, so on the order of 25–40
  minutes per world per core. This is consistent with E8's 15–23 worlds/h on a K = 12 L4 lane, and it is why
  A5-D fidelity is priced at 90 minutes rather than 60.

The full A5-D fidelity check still runs on L4: all 8 worlds, complete endpoints and every changed action.

## Sizing arithmetic found during the build

Both findings are in the reviewer questions.

- At 2.0 u/h with Round 0's chunk formula, Round 0-D's expected spend is 26.7–39.3 u for r = 23–15. The
  plan's "18–32 u" counted only teacher lane-hours. The draft therefore uses a ratification gate of expected
  ≤ 40 u and a charged ceiling of 48 u (40 + 20%).
- Dense R1 at 560 fresh worlds is 24–37 lane-h, or **49–75 u**, against the plan's "15–25 u, ≤ 30 u". The plan
  appears to have priced wall hours instead of lane-hours, missing a factor of three. As registered, R1 is
  fundable only at ≤ 30 u, so a larger R1 needs a new approval.

## Re-scope verification (2026-10-20 ~21:30 PT, branch `t34-round0d-rescope-20261021`, pre-data)

At the time of writing, `t34-round0-study-01/analysis/` did not exist. No Round 0 study file was opened. The
inputs were the Opus stand-in panels (integrity, science, ops), JUDGMENT §5 action 3 and §7e/§7f, the
RULING on the Round 0 GO file, and the boost-floor amendment (`research/t34_round0_amend/rule.py`, REPORT).

Tests: `tests/test_t34_round0d.py` has **29 passed** (1 thread, `-n 2`, nice 10). Together with
`test_t34_round0_analysis.py` and `test_t34_round0_fixes1_history.py`, 78 passed. New tests cover:

- **Relative boost floor.** `analyze.amend` is `rule.amend`. At B8-like rates (about 0.1%) the registered floor
  gives STOP and the amended floor gives GO. A teacher share below half the bare share gives STOP.
- **Rate-to-action table:**
  - KILL when r < 5;
  - wall-time KILL;
  - HOLD on a fidelity DIFFERENCE, with EXPLAINED_TIE accepted;
  - ceiling equal to 1.2 × expected + headroom;
  - replay timeout at least max(estimate + 25, 2 × estimate).
- **Derived ceiling** never strands the last chunk while one lane is in flight. Breaching it on settled charges
  alone is detected.
- **A5-D rate** comes from process-seconds and row-steps (ρ, R = max(T3, measured), min(r_rows, r_completed)),
  never from a time window. Nothing measured gives HOLD.
- **Dense R1 yield rule:** N = ceil(300/y), capped at 200 u and 24 h on 3 lanes. An unreachable held-out target
  or zero yield is not fundable.
- **Fidelity tie rule** is narrow: a near-tie at the first divergent batched decision only.
- **Ceiling seal:** a settled-charge breach seals `ceiling` INCONCLUSIVE, and `require_chunks` then raises
  `CapExpiredHistory`, not KILL. This closes the integrity BLOCKER.
- **In-flight limit:** 2 live attempts are counted. A concurrent shared probe does not make a dead attempt look
  alive.
- **Round 0 routing** of the governing verdict follows RULING §3 for GO, scientific STOP and INCONCLUSIVE, fault
  STOP, integrity INCONCLUSIVE, KILL, and non-governing verdicts.
- **Import pins** match the current bytes and agree with the amendment's `registered_sources`.
- **Sealed labels:** no module except `labels.py` reads `round0d` `sealed/` paths.

Protected packages are still untouched: `research/t34_round0`, `t34_round0_amend`, `t34_pathb`, `t34_exit_b8`,
`t34_exit_b8_fit`, `t3_rollout_labels_20261015` and `colab_cli`.

The A6 and the earlier engine checks above predate this re-scope. Activation requires a fresh, source-current A6
on the final approved commit, followed by A5-D.

## FIX-FIRST revision (2026-10-20 night, pre-data; `REVIEW_RESCOPE.md`)

The branch was rebased on main `d62070e5`. `t34-round0-study-01/analysis/` still did not exist, and no Round 0
study file was opened.

1. **`make_chunk_manifests` KeyError (BLOCKER).** FORECAST.json now uses only fields that `sizing.plan` always
   emits. It is built before any create-only write, so a missing field cannot leave partial manifests behind.
   **Test:** an end-to-end build of all chunk manifests and FORECAST.json from a re-scoped plan, using the real
   supervisor `validate`. Ratification is stubbed out.
2. **Wave-quantized forecast (BLOCKER).** Round 0-D now uses its own partition (deliberate difference 4), and
   Round 0's pinned `size_round` is no longer called.
   - Chunks are multiples of 12 pairs, and each is forecast as setup + ceil(n/12) whole waves.
   - Timeouts are at least setup + waves × max(A5-D slowest world, wave) + 25.
   - A5-D now records censored in-progress elapsed times.
   - The replay sample follows Round 0's ceil(10%) per chunk.
   - **Test:** no chunk is stranded at r ∈ {6.5, 7, 8, 10, 15, 23}, with every wave at 1.3× the mean world and
     the other lane in flight charged at its full cap.
3. **Outcome table.** It now has a survival-guard STOP row and a precedence rule: fault, then scripted or mass,
   then survival, then boost.
4. **Ceiling.**
   - The derived ceiling must be ≤ 120 u, otherwise KILL.
   - The operative ceiling is exactly 120 u. `require_registration` enforces 120 u and `max_in_flight` = 2.
   - The honest 2-lane threshold is about r = 6.5, and no third lane is used.
5. **Setup and HOLD path.**
   - Setup minutes are frozen at max(15, ceil(observed) + 5).
   - The pre-registered Mac re-run (`a5d.py resolve-hold`) gives EXPLAINED_PLATFORM only when the Mac
     reproduces T3 and there is a near-tie (≤ 1e-4) at the first Mac–L4 action divergence. Its records are
     hash-bound into the receipt. **Test:** resolution is accepted only from EXPLAINED_PLATFORM records.
6. **Fallback route (review ruling 4).** The fallback requires the exact late-publication refusal text, and the
   refusal file's SHA-256 must appear in CAMPAIGN_LOG.md.

**Tests:** `tests/test_t34_round0d.py` has 37 passed. With the two Round 0 test files, 86 passed (1 thread,
`-n 2`, nice 10).

## FIX-FIRST revision 2 (2026-10-20 night, pre-data; `REVIEW_RESCOPE_2.md`)

No Round 0 study file was opened. This section supersedes the "honest 2-lane threshold about r = 6.5" lines
above, which used mean-wave wall time.

1. **N1 straggler schedule (BLOCKER).** `sizing.plan` charges each chunk at its slow-world makespan
   (setup + ceil(n/12)·max(slowest, wave), 2.0 u/h) and replays the 2-lane admission (`admission_schedule`)
   against 120 u − replay cap; a stranded chunk or straggler total + replay cap > 120 u is KILL ("measured
   straggler schedule strands under 120 u"). The wall gate uses the same straggler makespans. **Tests:**
   slowest = 1.6 × wave at r = 7 is KILL (also with the wall gate disabled); a 9 × 6 grid of r × slowness
   checks that every RATIFY survives an independent 2-lane simulation at its own slowest world. At the
   placeholder 1.3 × wave, r = 6.5–8 now KILL on the straggler wall; the threshold is about r = 9.
2. **120 u is a hard total.** The replay cap is reserved (`chunk_admission_ceiling` = 120 − replay cap, enforced
   by `require_registration` and used by `dispatch`), and the admitted chunk is charged its full cap.
3. **HOLD binding.** `resolved_fidelity` accepts a resolution only if its L4 export hash equals the verified A5-D
   receipt's `fidelity-<i>.json` export, its checkpoint is B8 s3, the Mac reproduced T3 with no differences,
   and ε = 1e-4. `a5d.platform_tie` makes the two legs exclusive: greedy top-2 gap only at unbatched decisions,
   tied-longest gap only at batched ones. Both are tested.
4. **Outcome precedence** is computed by `analyze.outcome_reading` and recorded in `decision.json`; tested.
5. **Docstring** in `dispatch.py` names the 120 u admission ceiling. The DRAFT registration was regenerated.

**Tests:** `tests/test_t34_round0d.py` has 89 passed (1 thread, nice 10).
