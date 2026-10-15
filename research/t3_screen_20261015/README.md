# T3 label-bank development screen

Development screen, not gate evidence. Implements the binding pre-registration
and Amendment 1 in `docs/research/specs/t3_screen_prereg_2026-10-15.md`.

`watch_t3.py` imports the T2 evaluator, checkpoint readiness, job identity and
strict output validator; it does not copy or change the evaluation engine.
It finishes all five CTRL finals and the incumbent frp3-s12 + v8 before probing
T3 finals. Candidate probes are bare-network/no-veto, H5000, rowwise ego2s,
advisory masks, numba vector61, fast anchors, round 62 with 96 scripted and
32 frozen/32 mixed worlds. The one-time registered extension uses 96 scripted
round-63 worlds for eligible arms and shared CTRLs. The initial decision and
extension-exhausted flag are retained; no second extension or promotion occurs.

The watcher requires a passing pre-first-probe `seed_bank_audit.json`. This binds
both rounds, the historical NI/distillation and observed probe inventories, and
the training label bank's manifest. It remains an immutable launch receipt;
subsequent screen output is not historical evidence of earlier seed use.

The dedicated tmux session is `t3-probe`. Its output directory is
`../snake-dqn-artifacts/redesign-t3-probes-20261015/`. Each worker owns an atomic
output and log. Complete rows are reused only after checking checkpoint SHA,
world keys, evaluation settings and survival accounting. A singleton lock is
inherited by children. At most two probe processes run while training is active;
up to four only after all 15 training statuses are terminal and both training
lane names are absent. Reading session names is the only interaction with
training tmux sessions; no input, termination or mutation is sent to them.
Only the watcher's own child handles are cleaned up on exceptions.

After all probes/extensions, the watcher creates one deterministic 200-decisive-
state sample (seed 20261015), then re-rolls each available T3 final with the
existing T3 engine. Bank observations are reconstructed exactly from source
worlds before changing the continuation network. Re-roll outputs bind the bank,
sample and checkpoint. It reports surviving labelled actions and raw argmax
agreement, with advisory argmax agreement separately. Training labels came from
B8 simulator rollouts; the served artifact stays a bare network.

Commands (from this worktree):

```sh
OMP_NUM_THREADS=1 ./venv/bin/python -m pytest -q tests/test_t3_screen*.py
./venv/bin/python -m flake8 research/t3_screen_20261015 tests/test_t3_screen*.py
OMP_NUM_THREADS=1 ./venv/bin/python -m research.t3_screen_20261015.watch_t3
```

`progress.json`, `reroll_progress.json`, `watcher.jsonl` and individual logs are
operational evidence. Completion writes `analysis.json`, `telemetry.json`,
`rerolls.json`, `probes.jsonl`, and
`docs/research/t3_screen_result_2026-10-15.md`. Git publication remains a root-agent
operation: commit, fetch/merge origin/main, push HEAD:main, push t3p-20261015.
