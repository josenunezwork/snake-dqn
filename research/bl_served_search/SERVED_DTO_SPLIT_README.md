# W0/K0 authoritative-game / DTO separation

This is the opt-in `dto_split_w0_k0` arm of `served_tick_probe.py`, implementing
section B, first row of “Smallest next experiment after the bare panel” in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-rule-argon-1027/RULING.md`.
It supplies an architectural experiment, without selecting a timing result or
replacing the pending bare panel. No search runs and no actions are applied by
the DTO worker. Production serving code, checkpoints and defaults are unchanged.

`served_dto_split_manifest.json` is a new identity. The existing
`served_tick_manifest.json` retains its original bytes. The successor retains
the same three seeds, matched W0 timer/inline/fresh-process controls, 24 fps,
200 warm-up + 2,400 measured ticks per cell and original interleaving, replacing
only the real-Hub cell with the new split arm. The seed registry allocation is
explicitly the existing `served-tick-20261026` allocation for paired controls;
these are engineering worlds, not a fresh efficacy bank. The recorder observer
control remains separate. Exposure, order, transport bounds, no scored commands,
and base 3,600s / effective 7,200s runaway timeout are frozen in the new manifest.
The timeout conservatively retains the bare panel envelope; successor duration
has not been measured. There is no automatic retry or extension.

The coordinator uses the existing spawn/Pipe lifecycle from `isolated_game.py`.
Its child owns the authoritative real mixed Watch6 game. That child starts a
second spawned interpreter which owns stock inspector forwards, `build_frame`,
and WebSocket JSON construction. Inspector models are detached once before
pacing; the worker receives no live game, mutable policy, dispatch queue or RNG.
The facade uses the original `Ego2sServingPolicy.hero_activations` method on the
frozen observation and an inspector-only copy of the model. The FRP inspector
uses its original vector network. The game retains snapshot-equivalent cache
refreshes, because omitting them would change its future policy state.

Each export is immutable pickle bytes containing bounded ordered geometry,
counters, control-state inputs, exact hero observation/state and reset generation.
Neither the final DTO nor inspector activations are constructed in the game
process. The game charges observation refresh, state copying, packing, slot
publication and reply polling to its full absolute-clock tick. The update bar
still brackets `GameState.update`, separately from export/snapshot cost.

Transport has one slot, a 4 MiB input bound, an 8 MiB wire-output bound, at most
six snakes, 10,400 segments per snake and 20,000 food items. A process lock proves
publication ordering: FREE → REQUEST → REPLY → FREE. The worker copies the input
and owns the output until the game acknowledges the reply. The game tries the
lock once; a busy slot drops the new publication and increments a fallback
counter. It does not wait, spin, grow a queue or reuse mutable game memory. Lost
publications invalidate complete-output exposure, even if the game timing clears.
Renderer failure invalidates the attempt; it cannot apply actions or block a tick
on shutdown. Failed cells retain their attempted tick trace and partial transport
evidence. The worker exits when its authoritative owner disappears, including
when that owner died while holding the slot lock. Stop uses a shared flag, so killing the worker cannot abandon an
Event condition used by shutdown. Shutdown escalates only for the owned child.

Commands are applied by the authoritative owner in strictly increasing sequence
before step/export. The supported research control surface is playing, hero,
food, speed and reset. Reset increments a separate generation; export sequence
never rewinds with game frame. Obsolete replies are counted and withheld, and
previous wire output is cleared on generation change. No commands enter scored
cells. Offline tests replay ordered commands, death/respawn, resets, long bodies,
B8 and FRP inspector selections and a dead hero against inline snapshot behavior.
They also test immutable ownership, fixed backpressure, invalid/abrupt worker
failure, startup rejection and idempotent shutdown.

Raw evidence includes cold startup/first service, warm service/S latency,
worker input copy through output publication, game copy/publish/poll wall and CPU,
step/update/export/full-tick wall and CPU tails, wake lateness, game/worker peak
RSS and existing host covariates. Reply polling visibility is timestamped separately
from worker publication. No shell, hashing or disk export runs in a scored tick.
Final DTO hashes, model transfer, drains and any fallback final diagnostic output
are outside scoring. A final diagnostic publication cannot restore exposure.

Reduction validates source/host/parity and complete transport exposure first.
Malformed, stale, lost, overflow, crash or changed evidence is INVALID; missing
cells are INCOMPLETE. Any absolute overrun or required-cell update p95 above
11.688 ms is FAIL. Otherwise the result is engineering PASS for this panel only.
Low-background subsets remain descriptive. No p99 or search-wave gate is added.

The endpoint is WebSocket-shaped DTO/JSON construction. It does not open a socket
or qualify real Hub/client cadence or client latency; those remain separate
endpoints under the ruling. State/DTO/action parity is engineering evidence, not
search efficacy or serving release authorization.

## Operator commands

All commands are local. Use a fresh output directory for every attempt. The
check and smoke commands do not acquire a timing lease. The full panel uses the
existing host guard and lease; only the operator should run it when the shared
Mac window is available. This implementation handoff did not run that panel.

```bash
/Users/josenunez/Projects/ml/snake-dqn-served-split/research/bl_served_search/run_tick_probe.sh --manifest /Users/josenunez/Projects/ml/snake-dqn-served-split/research/bl_served_search/served_dto_split_manifest.json --check

/Users/josenunez/Projects/ml/snake-dqn-served-split/research/bl_served_search/run_tick_probe.sh --manifest /Users/josenunez/Projects/ml/snake-dqn-served-split/research/bl_served_search/served_dto_split_manifest.json --smoke --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-dto-split-20261027/smoke-01

/Users/josenunez/Projects/ml/snake-dqn-served-split/research/bl_served_search/run_tick_probe.sh --manifest /Users/josenunez/Projects/ml/snake-dqn-served-split/research/bl_served_search/served_dto_split_manifest.json --classified-host-load --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-dto-split-20261027/panel-01

/Users/josenunez/Projects/ml/snake-dqn-served-split/research/bl_served_search/run_tick_probe.sh --reduce /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-dto-split-20261027/panel-01/raw.json --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-dto-split-20261027/panel-01-reduced
```

The full panel requires committed, clean tracked source and admitted host
conditions. A check that reports host refusal on the shared Mac does not qualify
or invalidate unexecuted timing. Preserve failing cells and all raw evidence.
Do not merge, push or change serving defaults as part of these commands.

## Implementation verification (2026-10-27)

Final scoped suite: **501 passed, 1 skipped in 13.57s**. The skip is the existing
real desktop process integration opt-in (`SERVED_DESKTOP_INTEGRATION=1`), which
was not enabled on the shared Mac. This includes 21 new split checks.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-served-split
nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_served_dto_split.py tests/test_served_tick_probe.py tests/test_served_tick_lifecycle.py tests/test_served_tick_trace.py tests/test_served_tick_report.py tests/test_served_production_probe.py tests/test_served_probe_host.py tests/test_served_host_load.py tests/test_served_desktop_apps.py tests/test_served_desktop_history.py tests/test_served_desktop_integration.py tests/test_served_remote_supervision.py tests/test_served_supervision_io.py tests/test_served_supervision_identity.py tests/test_served_supervision_cpu.py tests/test_served_supervision_report.py
```

Earlier checkpoints used these same exact environment settings. The targeted
command below initially stopped after five passes when the deliberate crash test
exposed the Event shutdown deadlock; only that test run was interrupted. After
the stop-flag fix it passed 9 tests in 7.19s, and later passed 20 in 7.60s before
the additional abandoned-lock case. The broader command above first passed 498
with the same one skip before the last lifecycle cases were added.

```bash
nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_served_dto_split.py

nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q tests/test_served_dto_split.py tests/test_served_tick_probe.py tests/test_served_tick_lifecycle.py tests/test_served_tick_trace.py tests/test_served_tick_report.py
```

The five-file selection passed **95 tests in 9.15s** before the final lifecycle
checks. All final files passed Black, isort, flake8 and `git diff --check`:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/bl_served_search/dto_split.py research/bl_served_search/dto_split_report.py research/bl_served_search/served_tick_probe.py tests/test_served_dto_split.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/bl_served_search/dto_split.py research/bl_served_search/dto_split_report.py research/bl_served_search/served_tick_probe.py tests/test_served_dto_split.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/bl_served_search/dto_split.py research/bl_served_search/dto_split_report.py research/bl_served_search/served_tick_probe.py tests/test_served_dto_split.py
git diff --check
```

The CLI smoke was run twice during implementation, once with output
`/tmp/astra-served-dto-split-smoke-20261027`, then at final source with the command
below. Both reported `SMOKE_OK_NOT_TIMING`, inline/isolated and inline/split
state/DTO/action parity, three completed exports (one warm-up, two measured),
and zero busy/crash/stale counts. The final transport evidence also passed
`validate_transport`. These two-tick samples are functional evidence only.

```bash
nice -n 15 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.served_tick_probe --manifest /Users/josenunez/Projects/ml/snake-dqn-served-split/research/bl_served_search/served_dto_split_manifest.json --smoke --output /tmp/astra-served-dto-split-final-smoke-20261027
```

Argon independent static reviews were invoked through `agy --model
gemini-4-argon-high` with self-contained source packets and no tool access.
The full reviews found no blocking findings. Review output was kept under
`/tmp/astra-served-dto-argon-*-review.txt`. No timing or efficacy PASS is claimed.
No push, merge, serving integration or long panel was performed.
