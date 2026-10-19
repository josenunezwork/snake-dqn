---
name: snake-dqn-mac-ops
description: |
  Use whenever you write a launcher, start, monitor or stop a long-running LOCAL job on the owner's Mac for
  /Users/josenunez/Projects/ml/snake-dqn (training lanes, evaluations, screens, strict gates, pod runners,
  Drive watchers): the venv interpreter, device choice (CPU vs MPS), thread caps, tmux detached sessions,
  caffeinate, AC-power/lid/thermal guards, CPU slot locks, what you must never kill, and Google Drive for
  desktop paths. Also when a prompt says "run this overnight", "in tmux", "in the background", or asks you to
  check or stop a running job.
---

# Snake DQN: running long jobs on the Mac safely

Machine: Apple M5 Pro, 18 CPU cores (6P + 12E), 20-core GPU (MPS), 64 GB. Facts verified 2026-10-18.

## Interpreter and devices

- Always `./venv/bin/python` (Python 3.12, torch 2.9.1) from the repo or a worktree with the `venv`
  symlink; otherwise `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`. System `python3` has no deps.
- `SNAKE_DQN_DEVICE` (`cpu|mps|cuda|auto`) is read by `src/core/device_manager.py` and propagates to
  spawned workers (parent `override_device()` does NOT). Auto-detection NEVER picks MPS: Apple-Silicon
  default is CPU because the tiny ApexNetwork is ~5x faster on CPU than MPS. MPS is opt-in
  (`SNAKE_DQN_DEVICE=mps` / `--device mps`).
- **But the M3-C/B8 trainer recipe runs its learner on MPS** (`train_m3c.py ... --device mps`, ~3-5k
  transitions/s; learner-bound by MPS sync waits). Review lesson 2026-10-16: a "CPU only" instruction meant
  for the implementer leaked into launch scripts and a lane trained at 159/s for 7 h. Launch scripts use
  the RECIPE's device even though your own tests run on CPU; after launch, read the first log row's rate.
- torch.compile on MPS gives wrong gradients (auto-falls back); do not enable it locally. GPU is never used
  for gate evidence (not bit-exact).
- Determinism/eval work: `SNAKE_DQN_DEVICE=cpu` plus one thread per process.

## Thread caps and the compute policy

- Export in every launcher: `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
  VECLIB_MAXIMUM_THREADS=1` (per process; parallelism comes from processes, not BLAS threads).
- Policy (owner 2026-10-02, calibrated 2026-10-03): Tier-1/dev work up to **3 CPU slots x 2 threads**
  with the thermal guard; **strict gates exactly 2 slots**; always AC + lid open. Slot locks are files
  `cpu-slot-1.lock`, `cpu-slot-2.lock` (strict pool) and `cpu-slot-3.lock` (dev-only) under
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909` (`dev_screen.DEFAULT_SLOT_LOCK_ROOT`);
  the dev-only slot-3 lock is created with `research/compute/slot_setup.py --root <that root> --create`; hold slots via
  `research/apex_safety_20260926/dev_screen.py --use-slot-locks --slots N [--slot-pool 3 --thermal-guard]`
  or wrap any command: `./venv/bin/python research/perf_sim_20261007/with_slots.py --slots 2 -- CMD ...`
  (sets OMP=1 and CPU device, refuses on battery or lid closed).
- Thermal guard: `research/compute/thermal_guard.py` (stdlib) reads `pmset -g therm` + an episode-slowdown
  detector (defaults 12 / 1.60 / 8); fails CLOSED on unknown state; a trip pauses admission then stops
  with `thermal: ...` -- never relabels results. Use `--thermal-guard` on screens; strict runners have it.
- Check load before launching: `top -l 1 | head -12`, `pmset -g therm`, `tmux ls`. Do not start a third
  heavy lane while a strict gate holds its 2 slots.

## Power and lid guards (every long-run launcher)

```bash
pmset -g batt | grep -q "AC Power" || { echo "refusing: on battery" >&2; exit 2; }
ioreg -r -k AppleClamshellState | grep -q '"AppleClamshellState" = No' || { echo "refusing: lid closed" >&2; exit 2; }
```

Python equivalents: `sequential_runner.on_ac_power()`, `with_slots.lid_open()`. The 2026-10-03 incident:
lid closed on AC -> clamshell sleep + "Dark Wake Thermal Emergency"; the launcher had checked only AC.
Never launch long runs on battery (owner rule). Wrap the job in `caffeinate -dims <cmd>` (or
`/usr/bin/caffeinate -is` for a coordinator) so the Mac stays awake while it runs; a watchdog that must
outlive the terminal uses a detached process (`setsid`/`start_new_session=True`).

## tmux: launch, watch, stop

- One detached session per job, unique name, absolute paths, log to a file with noclobber:
  ```bash
  tmux new-session -d -s <job>-<YYYYMMDD>-<attempt> \
    "cd /Users/josenunez/Projects/ml/snake-dqn-<wt> && set -C && caffeinate -dims env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 \
     ./venv/bin/python research/<study>/<script>.py ... > /Users/josenunez/Projects/ml/snake-dqn-artifacts/<study>/<run>.log 2>&1"
  ```
  Shell-quote carefully; prefer a committed `lane*.sh` with a `--check` mode that prints the commands
  without starting (pattern: `research/t4a_two_sided_20261016/lane0.sh`: verifies registered input hashes,
  refuses existing outputs/logs, `set -C`, `caffeinate -dims`, exit status captured).
- Watch: `tmux capture-pane -p -t <name> | tail -40`, `tail -f <log>`, `tmux ls`. Jobs stream progress
  (done/total, ETA) every ~2 min; a silent log is a bug to fix before relaunching.
- **Never**: `tmux kill-server`; `kill`/`pkill`/`killall` by pattern (`pkill -f`, `kill $(pgrep -f ...)`);
  killing any PID you did not verify with `ps -o pid,ppid,comm,args -p <pid>`. The tmux SERVER's argv
  looks like `tmux new-session -d -s <first-session> ...` and matches job patterns; killing it (2026-10-03)
  destroyed every session, interrupted a RunPod runner mid-cleanup (pod kept billing) and killed a queued
  sweep. A Claude-side hook (`~/.claude/hooks/guard_kill.py`) blocks these; you have no such hook, so the
  rule is yours to keep. Never touch `git-sync` or any session you did not start.
- Stop ONE job you own: its own shutdown path (stop file, cancel command) first; else
  `tmux kill-session -t <exact-name>`. Pod runners are the exception: never kill their session or move
  their run dir while alive; wait for `EXIT=` in the log (see `snake-dqn-runpod`).
- Relaunch: confirm the old process exited (`ps -Ao pid,command | grep <script>` as a READ), then a NEW
  session name and a NEW run/output dir per attempt; never reuse or overwrite a prior attempt's outputs.
- Fixed-N pre-registered runs are never stopped early (invalid); sequential gates stop only by their own rule.

## Artifacts and Google Drive

For a **new Colab** job, the Mac runs `research/colab_cli` as the broker; its local state/receipt and verified
artifact directory are the authoritative exchange. It uploads inputs and imports bounded hash-verified exports with
temporary files plus atomic rename. Drive-for-desktop may receive a verified backup afterward, but the runtime never
mounts Drive. Do not invoke `colab auth` or `colab drivemount`; those are human-only. The Drive guidance below is
legacy Q1/notebook-worker support only.

- Run outputs, logs, checkpoints: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/<study>-<date>/...`
  (not a git repo; reference by path + sha256 in docs). Never write into `saved_snakes/`.
- Google Drive for desktop mount: `/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/`;
  the Colab exchange root is `.../My Drive/t16_colab` (Colab sees it as `/content/drive/MyDrive/t16_colab`).
  Writes from the Mac sync asynchronously: copy to a temp name then rename, verify with sha256 on the other
  side, and never create the same shared directory from two machines (Drive makes `name (1)` duplicates --
  `q enqueue` on the Mac creates all shared dirs; see `snake-dqn-queue-jobs`). Mac-side Drive watchers
  read specific files (heartbeats, receipts, logs) on a modest interval rather than listing whole trees.
- Checkpoints that bundle replay buffers can be 600 MB+; keep them out of git and out of Drive unless the
  job needs them.

## Before you say done

1. Launcher has: venv interpreter, explicit device per the recipe, thread caps, AC + lid checks,
   caffeinate, unique output dir/log with refusal of existing ones, streamed progress, and a `--check`/dry
   mode you actually ran.
2. You did not start anything the spec did not ask for; if you did launch, report the tmux session name,
   log path, PID, and the first log row's rate; give the operator the exact stop command.
3. Report times in Pacific time (owner is in California), and list every `tmux ls` session you saw but
   did not touch.
