# T4c tightened two-sided development screen

Binding protocol: `docs/research/specs/t4c_probe_spec_2026-10-16.md` and
`t4c_prereg_2026-10-16.md`, including Deviation 1. This screen has no promotion
or serving authority.

The `watch_t4c` module extends the T3/T4a screen tooling and uses the existing
T2 rowwise evaluator through `probe_t4c`. It evaluates final checkpoints on
round 66: 96 scripted, 32 frozen and 32 mixed H5000 worlds. All five CTRL finals
and the incumbent finish before any candidate is evaluated. Candidate probes
require a successful lane receipt, final checkpoint and summary. Only the
registered decision can request the single round-67 scripted extension.

Owned computation uses CPU, single-threaded math, and at most two simultaneous
probe workers throughout the run. The watcher never queries tmux and never
controls training. Only its own child handles are cleaned up on failure.
The abandoned CPU T4c run is pruned from freshness inventory discovery.

After a final completes all three initial gameplay mixes, its historical r62
drift capture may occupy an idle worker slot. Gameplay jobs take priority and
both job types share the same two-worker cap. The final diagnostic phase
revalidates each saved capture and its input hashes before computing readouts.

The launch audit is preserved as `seed_bank_audit_round66.json`. If extension
is needed, freshness is checked again and preserved separately as
`seed_bank_audit_round67_launch-<UTC timestamp>.json`. Resumed probes validate checkpoint identities,
world keys, evaluation settings, survival accounting and terminal lengths.

The extra observer only reads detached post-frame state. It records logical
length after frame 5000, with dead heroes zero. Unit coverage includes a real
CPU H5000 comparison on previously used round-60 worlds, confirming identical
mass, death, cause and survival records with and without this observer.

Run from this worktree with the project interpreter:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
./venv/bin/python -u -m research.t3_screen_20261015.watch_t4c
```

The root agent owns the dedicated `t4c-probe` tmux session. Its wrapper and
logs are in `../snake-dqn-artifacts/redesign-t4c-probes-20261016/`.
`progress.json` and `watcher.jsonl` describe execution; they are not completion
or decision evidence. The final report is
`docs/research/t4c_screen_result_2026-10-16.md`. Publication is a root-agent
operation after verification: commit, fetch and merge `origin/main`, push
`HEAD:main`, then push `t4p-20261016`.
