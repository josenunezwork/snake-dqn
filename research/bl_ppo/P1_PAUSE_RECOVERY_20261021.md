# P1 Y3 failure diagnosis and exact-entrypoint preflight

This sibling note supersedes the old handoff's source-02/replacement-02/1M
preparation instructions. The registration, F recipe, seed 20261021, 100M
endpoint, evaluation bank, E8 thresholds and registered budgets are unchanged.
No Colab calls or dispatch occurred during this correction. No live Y3 PASS is
claimed. The failed source evidence and every earlier preparation are retained.

## What actually failed

Source-02's exported `origin/training-status.json` records `INCOMPLETE`,
`critic_qualified=false`, zero actor transitions/updates and no snapshot.
It collected 2,523,136 critic-refit transitions, including 2,010,624 fit-world
transitions, then failed the registered held-out EV/MSE critic gate. The
supervisor receipt is `INCOMPLETE` with `stop_verified=true`. The wrapper checked
`PAUSED` before critic qualification and masked this with the generic pause
exception. This is E8 `NOT_READY`, not evidence that zero was rounded incorrectly.
The old exception discarded the measured failed metrics; its zero/null support
fields cannot establish actual missing support or recover the original EV/MSE.

Literal `pause_at=0` is the post-qualified-refit, pre-actor origin. A successful
refit must produce a snapshot at exactly zero actor transitions and updates.
Y3's excluded durability branches also request zero, but stop after their
registered action/update/episode minimums; their actual actor counts are positive.
Qualification resumes the original origin, never a durability endpoint.

Whole-world rounding remains `ceil(1,000,000 / 128) * 128 = 1,000,064`, overshoot
64. Zero stays zero. All five tranche boundaries and the 100M endpoint are exact.

## Correction

- `p1_contract.py` centralizes the unchanged frozen execution dimensions and
  integer whole-world rounding. There is no tiny-mode CLI/environment override.
- `p1_runtime.py` reports the E8 refusal with its status, counters and tripwires;
  it verifies actual actor counters and snapshot boundary metadata at every pause
  and exact predecessor on resume. `run.py`/`recovery.py` persist requested,
  rounded and actual boundaries and distinguish exact from durability snapshots.
- `refit.py` retains failed finite EV/MSE/support evidence without admitting actor
  updates. Nonfinite derived metrics enter the existing nonfinite tripwire before
  strict JSON serialization; finite qualification thresholds are unchanged.
- The exact-entrypoint preflight exposed two additional latent fingerprint bugs:
  the ordinary bank's NumPy RNG was unsupported, and the simulator's object-array
  kill metadata was hashed as heap addresses. Fingerprints now include NumPy RNG
  state and recursively hash object-array values with dtype/shape framing.
  Sensitivity regressions prove changed RNG or kill metadata still changes hashes.

## Mac verification

`test_p1_entrypoints.py` calls the real `p1_runtime.main` for Y3 source and
replacement, qualification and all five tranches. It runs actual B8 CNN/PPO
updates, spawned native-v8 collector worlds, flat transitive file transport,
snapshot export/restore and full action/state comparison. Only in-process test
fixtures reduce dimensions/horizon/bank sizes and supply calibration/held-out
test vectors to the real unchanged E8 gate. Those fixtures are software evidence,
never a scientific E8 qualification or replacement-L4 Y3 PASS.

The passing tiny run has origin actor=0, 1,024 matched continuation actions,
32 actor updates, 64 actual episode boundaries, qualification 129 ->130, and
resumes at 256/512/768/1,024/1,280. Only the last tranche exports an endpoint.
The companion test refuses a failed E8 at zero with no actor update or snapshot.

Final package command (166 passed in 64.21s):

```sh
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q research/bl_ppo --basetemp=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/mac-entrypoint-preflight-08
```

Use a fresh `--basetemp` for subsequent runs: pytest clears that directory.
The retained final log is `registration/pausefix-04/pytest.log` below the artifact
root. Changed Python passes Black/isort/flake8; shell syntax and diff checks pass.

## Current preparation and operator hold

Executable pin: `65a5391356f9e6a1c31e29c50fb288fb3521ff4e`, published on main;
fix commits `61fb11d29141ef4a1c402d66d0efff920223bc59` and
`617d1e706c806898b09f4c395ed58db4b925aa6c`.

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/`.
Current manifest directories:

- `y3/source-pausefix-04/`
- `y3/replacement-pausefix-04/`
- `qual-1m/prepare-pausefix-04/`

Each contains `manifest.json`, `DISPATCH.txt`, preparation and Z-9 receipts,
offline validation, verified unpacked control and completion/hash sidecars.
`registration/pausefix-04/manifests.json` binds their exact hashes and forecasts;
its sibling `DISPATCH.txt` indexes operator instructions. Attempt-02 and the
incomplete pausefix-03 preparation are superseded and retained.

All three are **PREPARED_NOT_DISPATCHED**, with an explicit `HOLD:` blocker and
`scientific_admission=false`. Offline schema/upload validation does not admit a
scientific retry. No inventory, HTTPS snapshot URLs, successful E8 origin or Y3
PASS has been manufactured. Independent disposition of source-02's E8 failure,
failed-attempt accounting within the existing caps, real preceding terminal and
release evidence, transitive catalogs, admission receipts and fresh inventory
remain required before a new preparation can admit any dispatch. Nothing in this
note authorizes retraining past that failed E8 gate or changing its thresholds.

No protected research packages, supervisor code, served checkpoints, serving
pins, tmux sessions or sealed study directory were edited or operated on.
