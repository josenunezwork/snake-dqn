# TRAIN-01-07 — Argon review fixes and real CPU screen

Outcome: all three implementation findings are fixed; offline admission passes.
The real E32 distribution screen is **INCOMPLETE_TIMEOUT**, not a screen pass.
The resident CPU fallback could not finish all three mixes within 45 minutes.
No runtime was dispatched.

## Engineering-screen change

At the owner's explicit request on 2026-10-27, terminal rate, starvation share
and collision share are compared only when **both arms have at least 16 terminal
events for that mix**. Otherwise each comparison records `insufficient_support`
and cannot fail. The other four metrics retain the existing 25% relative-gap
rule; supported terminal comparisons use the same threshold. This is a
**dev-sandbox engineering-screen change**, documented in TRAIN01.md, with no
change to the PPO recipe, natural H5000 task, endpoint, allocations or release
rules. The local baseline compared only four gross-support metrics; the repair
also records the three requested terminal metrics and their support decisions.
Collision share uses actual v1 wall/self/head/body codes 1-4. These engines have
no starvation terminal mechanism, so starvation share is zero.

The watchdog now refuses either `.sha256` or `.complete` sidecar for either
payload before opening or changing diagnostic files. Screen start telemetry uses
the instantiated task class, and partial test-only reports list their actual
requested mixes.

## Real E32 CPU execution

The main invocation requested scripted/frozen/mixed, E32 per mix, 256 frames,
with ordinary full-capacity resident tasks and CPU reference tasks. Both B8 and
frozen-network inference used CPU. Launches used `nice -n 15`, all thread caps 1,
Torch intra/inter-op threads 1, AC/lid checks, shared CPU slot locks and
caffeinate. Inherited parent priority made effective process nice 20. macOS
reported no thermal or performance warning during the run.

| Mix | CPU reference | Resident CPU fallback |
| --- | --- | --- |
| scripted | 256/256, complete, 3.977 s | 256/256, complete, 1411.952 s |
| frozen | 256/256, complete, 9.718 s | Partial; last report 201/256 at main elapsed 2681.440 s |
| mixed | 256/256, complete, 7.536 s | Partial; last report 188/256 at companion elapsed 1062.902 s |

The main timer stopped execution at **2700.0094977089902 seconds**. Its terminal
trace was in resident vector free-space feature work. To exercise mixed worlds
before that cutoff, a second free CPU slot ran the same real E32 mixed adapter
for the remaining overall window: **1092.0093772920081 seconds**. That companion
used `mixes=("mixed",), test_only=True` only because the API requires that flag
for partial-mix calls; its tasks retained `tiny=False`, E32 and natural H5000.
Its terminal trace was in resident vector features. Both exited nonzero through
their fixed timers, with no automatic retry. All four owned worker/wrapper PIDs
were verified gone afterward. Final partial frame counts above are the last
reported progress, not reconstructed exact interruption counters.

No complete all-mix comparison was returned, so material gaps are **not assessed**.
No late-life or CPU/CUDA identity conclusion follows. The main invocation began
with the final acceptance algorithm applied to baseline 9582c313 before its code
commit; only Black formatting changed afterward. The companion ran after the
published code and packet commits. Driver completion-time HEAD fields must not
be interpreted as a frozen launch pin.

## Verification and packet

- Scoped resident/sandbox/ops/mocked-supervisor suite: **414 passed, 838.37 s**.
- Frozen broker preflight: **226 passed, 27.99 s**, then
  `PASS_TRAIN01_PREFLIGHT seeds=114299008 schema=PASS registry=PASS tests=PASS allocations=0`.
- Frozen broker admission tests: **40 passed, 0.25 s**.
- Focused regressions: **30 passed, 1.88 s**. They cover nonzero 1-vs-2 death
  counts, both 15/16 support boundaries, supported terminal-rate/collision gaps,
  the unchanged sparse-support mass gate, all four immutable-sidecar cases and
  stable partial-mix telemetry. Independent review found no actionable issue.
- Changed-file Black/isort/flake8, repository `make lint` (149 Python files),
  launcher shell syntax and `git diff --check`: **PASS**.
- Fresh detached runtime: `152af938acc32da6aeea9fcb5a300f470d96c53c` at
  `/Users/josenunez/Projects/ml/snake-dqn-train01-07-runtime.noindex`.
- Fresh detached broker: `395061afc75e278634d40def2910b620b5c4f7d2` at
  `/Users/josenunez/Projects/ml/snake-dqn-train01-07-broker.noindex`.
- Both checkouts remained clean. All 186 source paths and five inputs verified;
  the seed audit is byte-identical to packet 06 and the new ledger is empty.
- Packet: `packet-train01-07`; identity: `sandbox-a-train01-20261027-07`.
  Manifest SHA-256: `dab8df4c7a68669eb596519a6976d759023a54063eee7b6ee7e778e1802d6a17`.
- `dispatch.sh --dry-run`: **PASS_DRY_RUN sandbox-a-train01-20261027-07 allocations=0 live_accounting=NOT_CHECKED**.

Exact commands, logs, hashes and progress are bound by
[VERIFICATION_TRAIN01_07.json](VERIFICATION_TRAIN01_07.json); artifacts are in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-07-engineering-20261027`.
The operator command remains in [packet-train01-07/DISPATCH.txt](packet-train01-07/DISPATCH.txt).
No live allocation, provider accounting/session query, A100 kernel/timing or
Inductor qualification was performed. No tmux session, `exp/mac-lease.json*`,
serving code, released checkpoint, qualified `research/colab_cli` source or
historical sealed packet was changed.
