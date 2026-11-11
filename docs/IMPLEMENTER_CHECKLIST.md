# Implementer & Reviewer Checklist (`docs/IMPLEMENTER_CHECKLIST.md`)

Walk this checklist whenever authoring or reviewing changes in `snake-dqn`. See [STATUS.md](STATUS.md) for current release state and [REGISTERED_ITEMS.md](REGISTERED_ITEMS.md) for active frozen items.

## 1. Before You Edit

- [ ] **Check [REGISTERED_ITEMS.md](REGISTERED_ITEMS.md):** Verify no constant, rollback bar, seed namespace, pre-registered cap, or SHA-pinned file you touch is registered/frozen (code contracts in §2 such as `percentile95` and `replay_window_p95` may change with tests + `1` review only if the meaning of the registered `p95` bars is kept). Never change a registered value to make a test or run fit.
- [ ] **Respect frozen & SHA-256-bound paths:** Never edit sha256-bound docs under `docs/research/**` (registrations, intents, receipts, rulings, prepare-bound docs). Appending `CAMPAIGN_LOG.md` rows and adding sibling erratum/status files is allowed when in scope. Do not edit `docs/experiments/**`, `tests/fixtures/**`, `saved_snakes/**`, the `14` files in `research/gpu_sim_spike/V8_SPLIT_FROZEN_COMPONENTS.json`, `research/colab_cli/**` unless qualified as a new tree (Fast-process #5), `tools/ops/**` (unless explicitly scoped + ops-reviewed), `src/evaluation/safety_veto*.py`, `src/evaluation/strict_promotion.py`, `src/scripts/tournament_eval.py`, `research/watch_replay/{engine.py,run.py}`, or `research/watch6_l1_confirm/**`.
- [ ] **Create-only immutable evidence:** Write exported payloads via temp file + `fsync` + atomic rename/link, then write `.sha256`, then write `.complete` **last**. Never rewrite, truncate, or touch a `.complete`-marked file or an existing receipt.
- [ ] **Held-out & contaminated namespaces:** Never open, read, train on, tune on, or select checkpoints from held-out, sealed, or contaminated namespaces (`t34-gpu-sitl-selectivity-20261109-v1`, `t34-gpu-sitl-screen-20261109-v1`, `t34-openstop-gate-20261104-v1`, `t34-hero-probe-20261103-v1`; `t34-hero-probe-20261103-v1` is design-contaminated and is never a gate). Listing namespace strings in seed-disjointness guards is fine. Releases require the confirmation lane and owner sign-off.

## 2. Scope Discipline

- [ ] **Minimal diff:** Fix only what was requested in your own git worktree and branch (local commits only; never `git push`, merge to `main`, or edit sibling worktrees).
- [ ] **Behaviour-preserving cleanups:** A refactor, formatting pass, or cleanup must never alter runtime behaviour (such as checkpoint/artifact lookup paths, environment fallbacks, tie-break order, or log/telemetry schemas).
- [ ] **Never weaken guards:** Do not relax assertions, tolerances, tripwires, or failure counts to make a test or preflight pass.

## 3. Tests

- [ ] **Fail-before / pass-after proof:** Every bug fix or behavioural guard includes a targeted test that fails without the fix and passes with it.
- [ ] **Hermetic & portable:** Never write tests that depend on this Mac's `/Users/...` absolute paths or unversioned local state (`snake-dqn-artifacts`, downloaded checkpoints, `~/.config/colab-cli`). Use `monkeypatch` and `tmp_path` for unit tests, and `@pytest.mark.skipif` when checking optional real artifacts or `mps`/`cuda` backends.

## 4. Honest Reports & Independent Review

- [ ] **Report only what ran:** Claim only commands and test suites actually executed in your current session, with exact `passed / skipped / failed` counts, file paths, and commit SHAs.
- [ ] **Independent review:** Carbon (the `agy` lane) implements; Claude Opus 5.5, run as a separate fresh `agy` reviewer, reviews Carbon's builds (`CAMPAIGN_LOG.md` `2026-11-10 PT 11:35`). Astra or a separate fresh `agy` run is also allowed (`STATUS.md` Fast-process #2). Never the author run.

## 5. Mac Hygiene

- [ ] **Thread caps + `nice`:** Prefix every Python, pytest, Numba, or PyTorch command with all `7` thread caps (`tools/sandbox/admission.py:37-46`) and `nice -n 15` (`tools/ops/agent_nice.sh:10-20`), pointing `SNAKE_DQN_PYTHON` (`tools/sandbox/admission.py:32`) at the main checkout's `venv/bin/python` when running in a git worktree without `./venv`:
  ```bash
  PYTHON="${SNAKE_DQN_PYTHON:-./venv/bin/python}"
  OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 \
  nice -n 15 "$PYTHON" -m pytest -q <targets>
  ```
- [ ] **Process cap (`<= 2`):** Run at most `2` concurrent processes (standing `agy` dispatch constraint protecting Mac `3×2`-thread Tier-1 slots, `STATUS.md:322` / `docs/research/compute_policy_2026-10-02.md`); `make test-fast` only when no governed Mac timing / strict / qualification work is running (`docs/INDEX.md` §5), and run targeted serial `pytest` inside `<= 2`-process agent runs.
- [ ] **Clean exit:** Terminate all background child processes before replying and verify none remain (`agy` dispatch constraint).
- [ ] **Hands off live services:** Owner rule `2026-11-07` (`STATUS.md:33`, `CAMPAIGN_LOG.md` `2026-11-07 PT 18:18`): the served Watch app runs persistently in `tmux` session `snake-web` (`tools/ops/watch_server.sh`) on `:8000`; agents must never start a second server on `:8000` (`web/serve.py`) or create, attach to, or kill `tmux` sessions.

## 6. GPU / Colab Preflight & Dispatch

### Before any dispatch (Mac, `0` units)
- [ ] **Run the real entrypoint on `mps`, not only `cpu`:** Catches `cpu`/`cuda` tensor mixes before Colab. Policy, runtime, and simulator must stay on one device across init and checkpoint resume.
- [ ] **Test the manifest, not only the code:** `required_exports` must list relative **payload paths only** (never `.sha256` or `.complete` sidecars; validate with `research.colab_cli.schema.validate_manifest` / `validate_export_inventory`), and resume paths must never rewrite a `.complete`-marked file.
- [ ] **Verify per-phase wall time against the limit:** Have implementer and reviewer check per-phase wall-time bounds against `timeout_minutes` (if within `~25%` of the limit, reuse built worlds across phases, compile native CUDA extensions for the active GPU's capability only while still checking declared `TORCH_CUDA_ARCH_LIST`, and pre-load lazy extensions before timed loops). Cut runtime first. Exploration wall-time and unit caps — including `gpu_sitl` Step B1b smoke and pilot — are runaway guards the operator may raise and log with no judge amendment (Owner ruling `2026-11-10` #2); only confirmation-lane science caps frozen pre-data require a pre-data judge amendment (never alter science bars or held-out seeds).
- [ ] **Emit per-phase `START` / `END` stamps:** Log wall-clock start, end, and elapsed seconds per phase so any timeout identifies where time went.

### Native-CUDA / `gpu_sitl` shake-out & dispatch roles
- [ ] **L4 shake-out first for new native-CUDA / `gpu_sitl` code (`prepare --mode shakeout`):** Require a **`PASS` receipt** on L4 (`~0.2–0.8` u, dev namespaces only, non-evidence) before new GPU code runs on registered hardware (Owner ruling `2026-11-10` #6; `CAMPAIGN_LOG.md` `2026-11-10 PT 04:50 & 11:35`); CUDA lazy-init (`torch.cuda` init/`set_device` before `reset_peak_memory_stats`) and native `.cu` paths never run on the Mac. In exploration (incl. `gpu_sitl` Step B1b smoke and pilot), any GPU type (`L4` included) may be used (Owner ruling `2026-11-10` #2). Unchanged job packages with only parameter/seed changes follow Fast-process #3.
- [ ] **Re-qualify native-CUDA / `gpu_sitl` edits & pin pilot to smoke:** Any merge/edit to `research/gpu_sitl/` or native-CUDA code needs a **new** L4 shake-out `PASS` before running on registered hardware (Owner ruling `2026-11-10` #6; `CAMPAIGN_LOG.md` `2026-11-10 PT 11:35`). **No code change** is allowed between a passing smoke and its pilot (the pilot pins the smoke's commit).
- [ ] **Generous dev wall cap (`30 min`, not `15 min`):** Session setup and slow control calls alone can take `5+ min` on dev/shake-out runs; exploration wall-time and unit caps (incl. B1b smoke/pilot) are runaway guards the operator may raise and log, while confirmation science caps follow the pre-data judge amendment rule above.
- [ ] **Implementer vs operator dispatch roles:** Implementers prepare the packet and run `tools/ops/dispatch.sh --dry-run`. The operator dispatches, retries and runs `reconcile_gpu_pending.py` (dispatching exclusively via `tools/ops/dispatch.sh`, never calling `colab new/exec/stop/upload/download` directly; exploration B1b smoke/pilot may use any GPU incl. `L4`, and scarce GPU stock is polled `~1/min` rather than every `20 min` per Owner ruling `2026-11-10` #2 & #4).

### When a run fails
- [ ] **Inspect `receipt.json` before blaming code:** Check `failure_class` and slow `cli_calls` first. `UPLOAD_CAP`, `RPC_EXEC_POLL_TIMEOUT`, allocation errors with no endpoint, and a missing `hardware.json` mean no project code ran — retry with no code change.
- [ ] **Automatic retries vs slow control plane:** Follow Fast-process #4 automatic bounded outcome-blind retries after verified stop + `0` outputs (`503` no-stock, allocation refused, single transport timeout). Only when the Colab control plane itself is repeatedly timing out on staging/RPC while Mac/broker health (`colab usage` time, ping) is normal, space retries `30–45 min` apart and stop at the first run that clears staging (`CAMPAIGN_LOG.md` `2026-11-10 PT 08:00`).
- [ ] **Reconcile `503` no-stock leftovers (operator):** After an `H100`/`A100` `503`, the operator retires any leftover `A100` pending record with `tools/ops/reconcile_gpu_pending.py` before the next attempt, using a fresh packet, job name, and output directory per attempt.
