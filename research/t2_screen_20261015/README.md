# T2 learner-regime screen

Development screen, not gate evidence. Implements
`docs/research/specs/t2_probe_spec_2026-10-15.md` without edits to the trainer,
`src/`, `web/`, the original probe, or governance inputs.

Run from this worktree on `t2p-20261015`:

```sh
OMP_NUM_THREADS=1 PYTHONPATH=$PWD ./venv/bin/python -m pytest -q tests/test_t2_screen.py
./venv/bin/python -m flake8 research/t2_screen_20261015 tests/test_t2_screen.py
PYTHONPATH=$PWD ./venv/bin/python -m research.t2_screen_20261015.watch_t2
```

The live watcher runs in the dedicated `t2p-run` tmux session. It checks training
`status.txt` and final checkpoint/summary presence, evaluates CTRL and baseline
first, and retains a two-worker ceiling even after training finishes. Workers
run with `nice -n 10`, one torch thread and `OMP_NUM_THREADS=1`. Session names are
read only to record training-lane presence. No training session receives input.
The singleton lock is inherited by workers, preventing an overlapping restart
while any worker from the previous watcher is still alive.

The external output directory is
`../snake-dqn-artifacts/redesign-t2-probes-20261015/`. Each job owns one atomic
`rows/NAME__rROUND__MIX.json`; complete matching rows are reused on restart.
Checkpoint digests and evaluation settings are validated before reuse. Events,
commands and worker PIDs are recorded in `watcher.jsonl`; stdout/stderr are in
`logs/`; `progress.json` is updated every 30 seconds. Failed probes are retried
at most three times per watcher execution and recorded in `failed_probes.json`.
Training failures are observed, never repaired by this watcher.

Round 60 is the initial dev bank. `seed_bank_audit.json` proves disjointness from
historical generated distillation manifests, NI banks, rounds 50–52, and the
extension round 61. EXTEND requests add exactly one 96-world scripted bank for
the affected arms and CTRL, with shared CTRL jobs evaluated once. After complete
probes, the watcher writes `probes.jsonl`, `telemetry.json`, `analysis.json`, and
`docs/research/t2_screen_result_2026-10-15.md`.

Rows retain per-world mass integral, deaths, alive frames, death frame, length
at death, and cause. Alive frames count post-transition survival, exactly as
`survival_fraction` in the evaluator. A fatal transition contributes zero alive
frames. Death frames are one-based; length at death is logical mass immediately
before the fatal action. Surviving heroes have null death frame and length.
Missing holdout agreement stays missing and cannot satisfy the GO threshold.
