# T34 Path B v2 retention — pre-data amendment: the bare arm runs unpaced

Status: **DRAFT — needs an independent review and Fable approval before re-ratification.** It is not ratified.
It must be ratified before Phase 0 ends (about Wed 2026-10-21 10:00 PT) so the retention handoff loses no Mac hours.
Branch `t34-retention-bare-unpaced-20261020`. Not merged to main. `REGISTRATION_RETENTION_V2_RATIFIED.json` is unchanged.
Pre-data: no timing, pre-check or efficacy row of the retention study exists (CAMPAIGN_LOG 10-20 ~15:00: execution
waits for Phase 0 and quiet admission).

## 1. Rationale

The ratified runner paces **both** Mac arms at 24 fps. `_collect` sleeps `max(0, 1/fps − elapsed)` after every tick
(`retention_v2.py:1288–1289` at 97232d51). The pre-check (`:1407–1416`) and the efficacy loop (`:1593–1601`,
`arm_fps = 24` for every arm in `MAC_ARMS`) use it for the bare comparator too. Preregistration line 219 says the bare
arm is "paced at 24 fps". The bare arm is released B8 s3 with no controller. Pacing measures nothing about it: a bare
tick takes 2–11 ms (measured below) and then sleeps for the rest of the 41.7 ms period. Reviewers estimate the sleep
at 10–12 h of the ~27 h Mac critical path (REVIEW_C Gap 2; REVIEW_A row 12 confirms the arithmetic).

## 2. Exact change

| Site | Before | After |
|---|---|---|
| `retention_v2.py` constants | — | `BARE_FPS = 0`, `PACED_FPS = 24`, `ARM_FPS = {"bare": 0, "paced": 24}` |
| pre-check `_precheck_rows` | `open_terminal(row, 24, arm)`, `_collect(..., fps=24)` | `ARM_FPS[arm]` in both places |
| efficacy `run_efficacy` | `arm_fps = 24` | `arm_fps = ARM_FPS[arm]` |
| dev `smoke` CLI | `opener(row, 24, arm)`, `fps=24` | `ARM_FPS[arm]` |
| `_collect` | `fps == 0` always called `session.settle_between_ticks` (unpaced-arm clock hold) | A session whose `arm == "bare"` at fps 0 runs free: it does not sleep or settle. It records `unpaced_clock = "bare_free_running_no_controller"` and has `overruns = 0`. A `paced` session at fps 0 raises `RegistrationError`. The Colab `unpaced` arm is unchanged. |
| `require_registration` / `registration_retention_v2._check_common` + builder | — | The registration must carry `"bare_fps": 0`. The builder writes it. The old ratified JSON (which has no such key) is refused. |
| `source_binding_v2.DECLARED_DATA_FILES` | — | Adds this spec, so the new registration binds its bytes |

The paced arm, the timing gate (`timing_v2`, including the paced OFF probes used for delivered cadence), the host
guard, the caps, the banks, the seeds, the analysis and the decision rules do not change. The preregistration is
sha-bound, so it is not edited. This file is its sibling amendment. Read prereg line 219 as "bare: unpaced (free
running), per the 2026-10-20 bare-unpaced amendment".

## 3. Why bare outcomes cannot change (code audit)

The bare path is `adapter_v2.open_terminal | open_precheck | open_watch(row, fps, "bare")` → `RunSession.step` →
`_collect`. It was checked for every channel through which wall-clock time could enter:

1. **No time budget or deadline.** `RunSession(arm="bare")` builds no `PathBSearchEngine` and no controller
   (`adapter_v2.py:74–75`). So `decide`, deadlines, late/stale fallbacks, `settle_between_ticks` and
   `discard_pending_assistance` are all no-ops for bare. The only clock reads in `step` are `perf_counter` and
   `thread_time` for the telemetry fields `last_update_ms` and `shadow_upkeep_ms`. No decision or analysis reads them.
2. **Construction is fps-independent.** `open_terminal` and `open_precheck` do `del fps`. `open_watch` does
   `set_speed(fps or 24)`, so fps 0 and fps 24 both build speed 24. `GameSession.step` reads `speed` only for
   release-monitor telemetry, which is disabled (`_release_monitoring = False`). Its Path B hook is `None`
   (`SNAKE_SERVE_PATHB=off`; asserted in `test_t34_pathb_adapter_v2`).
3. **Simulation, opponents and Watch game.** `GridBatchSim`/numba, the bank-engine world, the opponent policies
   (scripted, frozen B8, FRP3+v8) and the live `src/game` update have no clock reads that affect actions. The only
   `time.*` in `src/simd_env` is `vector61_policy.py:877/910`, a v8 latency counter that does not feed back into
   actions.
4. **RNG.** Each world is seeded at construction (`build_world` from the row seed; `new_session` →
   `set_world_seed`, and `open_watch` restores `random` state). Pacing consumes no RNG. The `HostGuard` sampler thread
   uses only `ps`, `pmset`, `ioreg` and `os.getloadavg`. Arms run serially in one process, so no RNG is shared across
   concurrent work.
5. **Floating point.** The process, the thread pools (all capped at 1) and the rowwise fp32 CPU kernels are the same
   in both modes. Sleeping does not change the order of operations.
6. **Scoring.** `_collect` metrics depend only on per-tick session state. `overruns` becomes 0 for bare, which is
   correct with no period. The analyzer reads overruns only from the paced arm and the timing probes
   (`analyze_v2.py:414–440`). `bare_prefix_trace` (the Colab cross-platform check) is a physical-state hash stream
   and does not change.

**Empirical identity** (`tests/test_t34_retention_bare_unpaced.py`, real B8 artifacts, threads=1). The tests run a
real terminal world (72 ticks) and a real Watch6 session (48 ticks) three ways: paced at 24 fps with real sleep;
unpaced (fps 0); and unpaced with adversarial irregular 0–18 ms stalls before ticks. In all three, every `_collect`
outcome field is byte-identical (canonical JSON sha256), including mass, survival, deaths, decisions, action counts,
`bare_prefix_trace` and controller state. The final world `_digest` is also identical. Only the pacing bookkeeping
fields (`overruns`, `unpaced_clock`, `host_contended_ticks`) are excluded. The paced run is checked to have actually
slept (wall time ≥ decisions/24).

## 4. Effect on caps, contention budget, host guard and timing gate

- **Comparison validity.** The estimand is paced-assisted minus released bare B8, paired per world. Bare B8 has no
  latency-dependent behaviour, so a paced bare and an unpaced bare are the same random variable for each world. The
  paired difference is unchanged. Real-time effects on the candidate (missed deadlines → bare fallback) stay entirely
  in the paced arm.
- **Timing gate.** Unchanged. The processing and delivered-cadence bars use the timing phase's own paced ON/OFF Watch
  probes at 12 and 24 fps, not efficacy bare records.
- **Contention budget.** The ledger is `contended_seconds / registered phase cap` (2% of 8 h for the pre-check, 2% of
  40 h for the study). The denominator is fixed and does not change. Shorter wall time means fewer 5-second samples,
  so measured exposure can only shrink. Contention cannot change bare outcomes (item 1). The paced arm's contention
  exposure is the same.
- **Host load.** The bare arm now uses about one core continuously instead of a ~5–25% duty cycle. The guard
  excludes its own process tree from "other CPU". It does count its own load in `load_1m`, but one runnable thread
  is far below the 3.0 excess threshold. The paced arm already runs the main thread plus one assistance worker (two
  threads), which remains the registered maximum. One process, no parallel worlds, as registered.
- **Caps.** The elapsed caps (pre-check 8 h, study 40 h, timing 1 h) are unchanged and now have more headroom. CPU
  seconds per tick are unchanged because sleep uses no CPU, so the 60 CPU-h cap is unaffected.
- **Order/state coupling.** In each row, bare runs before paced in a fresh, independently seeded session. The only
  carry-over is host state (load average, temperature) when the paced arm starts. One core for under a minute on an
  18-core machine is far below the guard's thresholds and below the paced arm's own load.
- **Product (Watch6) bars.** Persistence and growth compare per-session decision-normalised deaths and full-H5000
  mass. Both are clock-free for bare.

## 5. Expected Mac hours

Measured unpaced bare cost on registered rows (Mac, threads=1, during the Phase 0 load, early-game ticks): terminal
scripted 5.0 ms/tick, frozen 1.8, mixed 1.9; Watch6 10.1 ms/tick. Late-game ticks with long bodies cost more. For
planning, assume 10 ms (terminal) and 15 ms (Watch6).

| Phase | Bare exposure, paced (ratified) | Bare, unpaced (planning) | Saved |
|---|---|---|---|
| Pre-check (275,586 alive T3 frames + 26 death ticks) | 3.19 h | ≤ 0.77 h | ≈ 2.4 h |
| Efficacy terminal (144 worlds, ~0.8 alive) | ≈ 6.7 h | ≤ 1.6 h | ≈ 5.1 h |
| Efficacy Watch6 (32 × 5000 ticks, full horizon) | 1.85 h | ≤ 0.67 h | ≈ 1.2 h |
| **Total** | **≈ 11.7 h** | **≤ 3.0 h** | **≈ 8.7 h (up to ~10 h at measured rates)** |

The Mac retention chain goes from about 27–28 h to about 18–19 h after admission. All-survive upper bounds:
pre-check 7.41 h → ≤ 4.6 h; study 20.37 h → ≤ 12.9 h. Timing (≈ 0.36 h) is unchanged.

## 6. Re-ratification checklist (orchestrator)

1. Independent review of this spec and of the diff on branch `t34-retention-bare-unpaced-20261020`.
2. Fable approval judge, recorded in CAMPAIGN_LOG.md.
3. Merge. Rebuild the registration with `registration_retention_v2` (it now emits `bare_fps: 0` and the new source
   inventory, including this spec). Commit and push. Freeze a new worktree at the new source commit. Get a
   trace-receipt PASS.
4. The ledger and fixed phase directories are unaffected, since no phase has a terminal record yet.
