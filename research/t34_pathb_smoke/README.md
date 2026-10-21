# Path B v2 deliverability smoke (development, excluded)

The precommitted [stopping rule](STOPPING_RULE.md) governs this development smoke.
No output is admissible to any registered verdict. No registration or production
Path B source is changed. The controller and adapter are imported read-only.

The fixed input inventory is the five historical M3 doom samples and the 75
historical action changes in 22 original T3 source trajectories. Every root is
reconstructed from its original checkpoint/roster and earlier saved T3 overrides;
saved greedy actions and lengths are validated. Start at max(previous change+1,
doom-499), preserving D500 lead. M3 sources use their original continuous Watch
rosters. This is a conditional interval replay, not continuous efficacy.

Each interval starts a fresh actual RunSession/controller, with cold maintenance
charged from before construction, and executes at 24 fps until the first applied
change or the original doom. Absolute frame deadlines additionally reject an
application that only appeared timely because a tick overran. No tick waits for
search. Receipts retain overruns, achieved rate, full controller state, detection
completion/live frame, complete improving evaluations, search returns and events.
The primary denominator is fixed at 75 historical changes, including delay-only
changes; H500 rescue versus delay counts are separate. Historical D300 survival
LADs are preserved with their original qualification. An independent backward
scan finds the exact last H500 v2 improvement (rescue OR strict survival delay),
so operational lead is reported without mislabelling D300 LAD as H500 survival.
Those diagnostic scans run outside the paced windows and supply no search cache.

N=1 uses the unchanged native branch batch. N=6 uses the existing controller
worker plus five helper threads, splitting the eight chronological branch points
into stable contiguous chunks. Each chunk calls the unchanged native evaluator.
A locked first-winner latch prevents later callbacks replacing the winner. Local
cooperative cancellation drains other chunks at their next row boundary before
publication; this drain cost is included. The controller retains one outstanding
job, full digest/life/epoch fences and bare fallback. This is an experimental
six-thread transport, not a candidate amendment. Main thread + six workers = 7;
all numerical pools are capped at one, no background host sampler is created.

Run from the worktree root with the primary venv:

```sh
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb_smoke.run --check
/usr/bin/caffeinate -dims env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb_smoke.run
```

Default create-only output:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-smoke-20261021/`.
Wall cap: 6,900 seconds including reconstruction and diagnostics. AC, open lid
and thermal guards are checked before launch and approximately every 30 seconds.
Existing jobs remain untouched; host contention is disclosed, not filtered away.

After the terminal receipt is complete, audit and render it with:

```sh
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb_smoke.report --check
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_pathb_smoke.report
```

The renderer verifies hashes and fixed paired coverage, then creates `audit.json`
and `RESULT.md` without overwriting earlier artifacts. Source-hash verification
requires the execution source files to remain available unchanged. The committed
[RESULT.md](RESULT.md) is the human-readable copy; raw data stays in the artifact root.
