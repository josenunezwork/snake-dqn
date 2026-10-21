# T34 Path B v2 retention — pre-data amendment: the bare arm runs unpaced

Status: **APPROVED pre-data amendment.** It takes effect only through a ratified registration that binds this
file's sha256, both in `source_hashes` and in its `amendments` pointer. Such a registration exists only after the
independent review, Fable approval and a final-source trace-receipt PASS, all recorded in
`docs/research/final_run_20261019/CAMPAIGN_LOG.md`. It supersedes the registration ratified at source 97232d51
(study `t34-retention-v2-operator-01`). That registration is void for execution and was never executed. The
amendment was made pre-data: no timing attempt, pre-check row or efficacy row of the retention study existed, and
no `refs/tags/t34-retention/*` or ledger entry existed.

## 1. Rationale

The 97232d51 runner paced **both** Mac arms at 24 fps. After every tick, `_collect` sleeps
`max(0, 1/fps − elapsed)`. The pre-check (`for arm in ("bare", "paced")`, `fps=24`) and the efficacy loop
(`arm_fps = 24` for every arm in `MAC_ARMS`) applied this to the bare comparator as well. Preregistration line 219
says the bare arm is "paced at 24 fps".

The bare arm is released B8 s3 with no controller, so pacing measures nothing about it. A bare tick takes 2–11 ms
(§5) and then sleeps for the rest of the 41.7 ms period. Reviewers put this sleep at about 10–12 h of the ~27 h Mac
critical path: advice round 2026-10-20_a100, REVIEW_C Gap 2, with the arithmetic confirmed in REVIEW_A row 12.

## 2. Exact change

Line numbers refer to the amended source bound with this file.

| Site | 97232d51 | Amended |
|---|---|---|
| `retention_v2.py:69–72` constants | — | `BARE_FPS = 0`, `PACED_FPS = 24`, `ARM_FPS = {"bare": 0, "paced": 24}`, `BARE_UNPACED_AMENDMENT` (this file) |
| `retention_v2.py:82` `DEFAULT_STUDY_ROOT` | `t34-retention-v2-operator-01` | `t34-retention-v2-operator-02` (fresh study id) |
| `retention_v2.py:485–505` `require_arm_rates`, called from `require_registration` (`:522`) | — | Runs at phase start, before reservation and admission, in the pre-check and efficacy phases. It refuses unless `ARM_FPS` is exactly the amendment, the registration has `paced_fps: 24` and `bare_fps: 0`, and `amendments` equals `[{"path": <this file>, "sha256": <its bytes>}]`. |
| pre-check `_precheck_rows` (`:1451–1460`) | `open_terminal(row, 24, arm)`, `fps=24` | `ARM_FPS[arm]` |
| efficacy `run_registered` (`:1636–1639`) | `arm_fps = 24` | `arm_fps = ARM_FPS[arm]` |
| dev `smoke` CLI (`:1848–1858`) | `opener(row, 24, arm)`, `fps=24` | `ARM_FPS[arm]` |
| `_collect` (`:1263–1276`, `:1323–1325`, `:1374–1378`) | At fps 0, always called `session.settle_between_ticks` (the unpaced-arm clock hold) | `arm == "bare"` at fps 0 runs free: no sleep and no settle (the adapter's settle raises for bare). It records `unpaced_clock = "bare_free_running_no_controller"`, with `overruns = 0`. `arm == "paced"` at fps 0 raises `RegistrationError`. The Colab `unpaced` arm is unchanged. |
| `registration_retention_v2.py:350–360`, `:617–618` | — | The builder emits `bare_fps: 0` and `amendments: [binding of this file]`. `_check_common` refuses a registration without them. |
| `source_binding_v2.DECLARED_DATA_FILES` | — | Adds this file, so `source_hashes` binds its bytes |

Unchanged: the paced arm, the timing gate (`timing_v2`, including its paced OFF probes), host guard, caps,
contention budget, banks, seeds, pairing and order, OC/power, analysis and decision rules.

The preregistration is sha-bound and is not edited. This file is its sibling amendment. Preregistration line 219
now reads "bare: unpaced (free running)". The study root on preregistration line 26 now reads
`t34-retention-v2-operator-02`.

## 3. Why bare outcomes cannot change

The bare path is `adapter_v2.open_terminal | open_precheck | open_watch(row, fps, "bare")` → `RunSession.step` →
`_collect`. Every channel through which wall-clock time could enter was checked:

1. **No time budget or deadline.** `RunSession(arm="bare")` builds no `PathBSearchEngine` and no controller
   (`adapter_v2.py:74–77`). `decide`, deadlines, late/stale fallbacks, `settle_between_ticks` and
   `discard_pending_assistance` therefore do nothing for bare. The only clock reads in `step` are `perf_counter` and
   `thread_time` (`adapter_v2.py:168, 222–237`). They feed telemetry fields that `_collect` does not return.
2. **Construction does not depend on fps.** `open_terminal`/`open_precheck` do `del fps` (`adapter_v2.py:318, 331`).
   `open_watch` calls `set_speed(fps or 24)` (`:357`), so fps 0 and fps 24 build the same speed 24.
   `GameSession.step` reads the clock and `speed` only inside the release-monitoring branch. `new_session` sets
   `_release_monitoring = False` after construction (`research/t15_mixed_roster_20261017/episodes.py:79`). Even if
   that branch were active, it is telemetry only. The Path B serving hook is `None` (`SNAKE_SERVE_PATHB=off`;
   asserted in `test_t34_pathb_adapter_v2`).
3. **Simulation, opponents and Watch game.** `GridBatchSim`/numba, the bank-engine world, the opponent policies
   (scripted, frozen B8, FRP3+v8) and the live `src/game` update contain no clock reads that affect actions. The
   only `time.*` call in `src/simd_env` is the v8 latency counter at `vector61_policy.py:877/910`. It only feeds
   `BoostAwareCounters`, and the identity veto digests strip timing (`t34_shadow/identity.py:207`).
4. **RNG.** Each world is seeded at construction: `build_world` uses the row seed, `new_session` calls
   `set_world_seed`, and `open_watch` restores the `random` state. Pacing consumes no RNG. The `HostGuard` sampler
   thread uses only `ps`/`pmset`/`ioreg`/`os.getloadavg`. The arms run serially in one process.
5. **Floating point.** The process, the thread pools (all capped at 1) and the rowwise fp32 CPU kernels are the same
   in both modes. Sleeping does not change the order of operations.
6. **Scoring.** `_collect` metrics are per-tick functions of session state. For bare, `overruns` becomes 0, which is
   correct because there is no period. The analyzer reads overruns only from the paced arm and the timing probes,
   and only range-checks bare overruns. `unpaced_clock` has no consumer. `host_contended_ticks` is consumed only for
   timing probes. `bare_prefix_trace` (the Colab cross-platform check) is a physical-state hash stream and is
   unchanged.

**Empirical identity: `tests/test_t34_pathb_bare_unpaced_v2.py`** (real B8 artifacts, threads = 1). It covers each
terminal mix (scripted, frozen, mixed) for up to 1,300 ticks on unregistered development seeds, with at least 480
decisions each. The scripted and frozen heroes die at ticks 504 and 1,272, which exercises the terminal
early-break path. The mixed hero survives all 1,300 ticks; no mixed development seed among 50 died within 1,300
ticks. It also covers a Watch6 session through a hero death and respawn. Each row runs three ways:
paced at 24 fps with real sleep, unpaced (fps 0), and unpaced with adversarial irregular 0–18 ms stalls. Across the
three runs, every `_collect` outcome field is byte-identical (canonical JSON sha256), including mass, survival,
deaths, decisions, action counts, `bare_prefix_trace` and controller state, and so is the final world `_digest`.
Only the pacing bookkeeping fields are excluded: `overruns`, `unpaced_clock` and `host_contended_ticks`. The test
also asserts that the paced run actually slept. The file matches the trace-receipt glob `tests/test_t34_pathb_*v2.py`.
With `T34_TRACE_RECEIPT=1`, a missing B8 artifact fails the test instead of skipping it, so the receipt must show
these tests as passed with 0 skipped.

## 4. Effect on validity, caps, contention budget, host guard and timing gate

- **Estimand.** The estimand is the paced-assisted arm minus released bare B8, paired per registered world or
  session. Bare B8 has no latency-dependent behaviour, so paced bare and unpaced bare are the same random variable
  for each world, and the paired difference is unchanged. Real-time effects on the candidate (missed deadlines lead
  to bare fallback) remain entirely in the paced arm.
- **Timing gate.** Unchanged. The processing and delivered-cadence bars come from the timing phase's own paced ON/OFF
  Watch probes at 12 and 24 fps, not from efficacy ticks or bare records.
- **The one asymmetry: host state carried over into the paced arm.** In each row, bare runs first and paced runs
  second, each in a fresh, independently seeded session. The paced arm now starts right after a bare burst at about
  100% of one core (seconds to about a minute), instead of after a 5–25% duty-cycle bare run. `_between_rows` pauses
  only for power, lid or thermal reasons, never for load, so nothing resets host state between the arms. The paced
  arm's outcome does depend on the clock, so this is the only channel through which the amendment could reach a
  decision statistic. On the 18-core host, one core is far below the guard's 3.0 load-excess threshold and any
  thermal trigger. Any effect would add load at the start of the paced run, which can only cost the candidate
  deadlines. **Any effect is therefore against the candidate (conservative)** and cannot create a false GO.
- **Host-guard sensitivity.** `load_1m` includes the runner's own process; other-process CPU excludes it. A
  free-running bare arm adds about 1.0 to `load_1m`, compared with about 0.1–0.25 when paced. This narrows the
  headroom under the 3.0 excess threshold (sustained more than 60 s) during bare rows. The only possible
  consequence is more contended seconds (towards INCONCLUSIVE), never a changed outcome.
- **Contention budget.** The ledger is `contended_seconds / registered phase cap`: 2% of 8 h for the pre-check and
  2% of 40 h for the study. The denominator is fixed. A shorter wall time means fewer 5-second samples, so measured
  exposure can only shrink. `check()` per tick is a cached read, so a faster tick loop adds no sampling.
- **Caps.** The elapsed caps (pre-check 8 h, study 40 h, timing 1 h) are unchanged and gain headroom. Caps are still
  checked every tick on the free-running path. CPU seconds per tick are unchanged because sleep uses no CPU, so the
  60 CPU-h cap is unaffected. There is one process and no parallel worlds, as registered.
- **Product (Watch6) bars.** Persistence (decision-normalised deaths) and growth (full-H5000 mass) are clock-free for
  bare.

## 5. Expected Mac hours

Measured unpaced bare cost on registered rows (Mac, threads = 1, under Phase 0 load, early-game ticks):

| Row | ms/tick |
|---|---|
| Terminal scripted | 5.0 |
| Terminal frozen | 1.8 |
| Terminal mixed | 1.9 |
| Watch6 | 10.1 |

An independent panel measured 2.4 and 11.3 ms/tick. Ticks late in the game, with long bodies, cost more. The plan
below assumes 10 ms per terminal tick and 15 ms per Watch6 tick.

| Phase | Bare exposure, paced (97232d51) | Bare, unpaced (planning) | Saved |
|---|---|---|---|
| Pre-check (275,586 alive T3 frames + 26 death ticks) | 3.19 h | ≤ 0.77 h | ≈ 2.4 h |
| Efficacy terminal (144 worlds, ~0.8 alive) | ≈ 6.7 h | ≤ 1.6 h | ≈ 5.1 h |
| Efficacy Watch6 (32 × 5000 ticks, full horizon) | 1.85 h | ≤ 0.67 h | ≈ 1.2 h |
| **Total** | **≈ 11.7 h** | **≤ 3.0 h** | **≈ 8.7 h (up to ~10 h at measured rates)** |

The Mac retention chain shrinks from about 27–28 h to about 18–19 h after admission. Upper bounds if every world
survives: pre-check 7.41 h → ≤ 4.6 h, study 20.37 h → ≤ 12.9 h. Timing (≈ 0.36 h) is unchanged.

## 6. Binding re-ratification order

All steps run on branch `t34-retention-bare-unpaced-20261020` in the authoring checkout that will publish the JSON.
`main` is not touched until step 8, so the ratified 97232d51 design remains the runnable fallback until then.

1. Make the review fixes and push the branch. Get the independent re-review.
2. Get Fable approval. Record the review and the approval in CAMPAIGN_LOG (landing in step 8).
3. Regenerate the OC on the branch in the authoring checkout, with the same inputs and new paths:
   `retention_oc_v2 --reps 20000 --seed 3420261020 --max-seconds 60` (repeat with `--resume` until complete).
   **Acceptance:** every field of the new OC except `source_commit` and `source_hashes` must be byte-equal to the
   ratified `RETENTION_OC_V2.json`, including `rates`, `pathb_joint_go_power`, `watch6_product_power`,
   `development_summary`, `scenarios`, `fixed_vs_sequential`, `world_pool_sha256` and `simulator_hash`. Any other
   difference means **STOP**, because it would mean the amendment changed a statistic. Copy the receipt over
   `RETENTION_OC_V2.json`, commit it, and record the key-level diff in CAMPAIGN_LOG.
4. Run `registration_retention_v2 build --study-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-v2-operator-02 --checkpoints …`
   in that same checkout, so that `history_ledger.repo` is the publishing checkout and the ledger is
   `research/t34_pathb/ledger/t34-retention-v2-operator-02.jsonl`. The draft carries `bare_fps: 0` and the
   `amendments` pointer to this file. Then run `validate`, commit, and push the **branch**. Leave operator-01
   untouched as the superseded record.
5. Create a fresh detached worktree at the draft's `source_commit`.
6. Write the trace receipt from that frozen worktree to a new file,
   `…/t34-retention-v2-operator-02/trace-receipt.json`. It runs the `tests/test_t34_pathb_*v2.py` glob with
   `T34_TRACE_RECEIPT=1` on the Mac that holds the B8 artifact. It must show the bare-unpaced identity tests
   **passed with 0 skipped**, and an overall PASS.
7. Run `ratify --source-commit <draft source_commit> --trace-receipt <receipt>` and write the ratified JSON on the
   branch. Overwrite `REGISTRATION_RETENTION_V2_RATIFIED.json` with the new bytes; git history keeps the 97232d51
   bytes. Commit on the branch.
8. **Single push to main**, only after the receipt PASS. It contains the code, OC, draft and ratified JSON, and the
   supersession edits: a CAMPAIGN_LOG/STATUS line marking registration @97232d51 SUPERSEDED pre-data (VOID for
   execution), and README_V2/HANDOFF commands with `E=…operator-02`, `FROZEN=`, `SOURCE_COMMIT=` set to the new
   source. The `snake-dqn-retention-fixes6` and `snake-dqn-retention-trace-97232d51-20261020` worktrees are
   retired, and the Colab unpaced arm is dispatched only at the new source commit (`unpaced_colab` and
   `make_unpaced_manifest` bind the same `source_hashes` and refuse an old-commit dispatch). Then run
   `validate --ready` from the frozen worktree.
9. If anything fails before step 8, run the ratified 97232d51 design unchanged. It costs the ~8.7 h and loses
   nothing scientifically.
