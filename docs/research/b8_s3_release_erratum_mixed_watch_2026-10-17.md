# B8 seed 3 release erratum: mixed Watch default — 2026-10-17

**Outcome: RELEASE, owner-approved.** Watch's default exhibition roster is now
**mixed**: the released **b8-s3 hero without a serving veto** at slot 0 faces
**frp3-s12 opponents with the v8 veto** at slots 1 through N−1. The owner approved
this switch after T15 `SERVING_PASS` and T13b's fixed roster rule, as recorded in
the [T15B implementation spec](specs/t15b_switch_watch_default_spec_2026-10-17.md).

This sibling erratum publishes the draft in the unchanged
[T15 qualification report](t15_mixed_roster_qualification_2026-10-17.md) and
supersedes the all-B8 default wording in the unchanged
[original B8 release record](b8_s3_release_2026-10-17.md) and
[monitoring erratum](b8_s3_monitoring_erratum_2026-10-17.md). Their qualification
records and hash-bound evidence remain historical and unchanged.

## Released behavior and rollback

`WATCH_ROSTER_RELEASED_DEFAULT = WATCH_ROSTER_MIXED`. Unset or blank
`SNAKE_SERVE_WATCH_ROSTER` selects mixed. Invalid words fall back to the released
mixed default with a visible reason. Opponent checkpoint/pin/wrapper refusal
restores all-B8 with a visible reason and no partial mixed roster.

Opponent veto counters are labeled as opponent counters. The B8 hero is never
in `wrapped_ids` and has no serving veto counter. B8's v8-derived training labels
and live input features do not mean the B8 hero is wrapped by a serving veto.
The hero checkpoint remains `b8-s3`, SHA256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`;
opponents use `frp3-s12`, SHA256
`eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`.

To roll back the Watch roster, restart the server you operate with:

```bash
SNAKE_SERVE_WATCH_ROSTER=all_b8 ./venv/bin/python web/serve.py
```

Unsetting the variable selects mixed again. Checkpoint rollbacks remain as
recorded in the original release. Play, the checkpoint default, training and
checkpoint bytes remain unchanged. This switch introduces no Play or
leaderboard-comparability change.

T7's claim remains **more mass at equal survival against the served agent, with
no hero veto**; it is not a claim of greater safety. T15 qualifies this serving
configuration. It does not isolate the effect of the opponent checkpoint from
the effect of its wrapper, and provides no new promotion evidence.

## Monitoring by roster mode and size

| Watch roster | Driven snakes | Hero deaths / 1k baseline | Rollback when deaths / 1k exceed | Source |
|---|---:|---:|---:|---|
| mixed | 6 | 0.272 | 0.544 | T15: 34 deaths / 125,000 decisions, 25 H5000 episodes |
| all_b8 | 6 | 0.424 | 0.848 | T12 |
| all_b8 | 8 | 0.496 | 0.992 | T13b |
| all_b8 | 12 | 0.768 | 1.536 | T13 |

Equality does not trigger. Mixed N6 logical-length p50 is 248 and p95 is 841.
There are no registered mixed Watch8/12 death baselines; those sizes report
`NOT_HEALTHY` for missing death baselines even though their latency is qualified.
The monitor never borrows or pools another mode or size's baseline.

Mixed default-speed `GameState.update` latency uses the exact retained T15 p95
values, with rollback above 1.5× the matching value:

| Mixed Watch case | Update p95 baseline (ms) | Rollback above (ms, rounded) |
|---|---:|---:|
| Watch6 | 7.792219356633723 | 11.688329 |
| Watch8 | 9.519156138412654 | 14.278734 |
| Watch12 | 13.141333474777639 | 19.712000 |

All-B8 and Play retain their T12B latency baselines; all-B8 Watch8 still has no
qualified latency baseline. The 66.667 ms qualification bound and server ceilings
remain: mixed Watch6 at most 24 fps, Watch8/12 at most 12 fps. These are processing
budgets, not achieved-cadence promises. Latency excludes the first 200 updates
after construction/rebuild/reset and uses eligible 12 fps samples only. Deaths
and actual hero decisions count at every speed, including warm-up.

The predefined observation window for **each roster mode and size** is at least
seven days from its first retained valid Watch tick **and** 100,000 actual hero
decisions, whichever ends later. Keep complete stderr across restarts and do not
pool all-B8 history into the mixed window. The owner check-in is 2026-10-24;
continue each incomplete window after that date. Rates use all retained samples
in that group's window; rollback thresholds are checked before window completion.
Any served fallback (including a refused mixed opponent pin/wrapper) or ego2s
exception remains a rollback trigger. Mixed monitoring follows the fixed B8
roster hero even while the inspector selects an opponent. Legacy Watch records
without a roster-mode field remain all-B8; mixed records explicitly emit
`roster_mode=mixed`. The CLI labels mixed groups `mixed_watch6`/`mixed_watch8`/
`mixed_watch12` and retains `watch6`/`watch8`/`watch12` for all-B8.

Read retained telemetry with:

```bash
./venv/bin/python scripts/monitor_b8_release.py logs/b8-release.stderr.log
```

The script reports thresholds and window progress; it does not stop or restart
servers. A short startup smoke cannot establish a healthy completed release
window. No persistent server or monitoring job is installed by this task.

## Live verification and validation

Both configurations were started with all `SNAKE_SERVE_*` variables cleared,
except the explicit `all_b8` rollback selector in its case. Each used an owned
loopback port and temporary scores database, CPU and one BLAS thread. Health and
state endpoints succeeded, eight WebSocket frames advanced from 0 through 7,
and both servers were stopped and reaped. No tmux session or foreign process
was touched. The pre-existing `saved_snakes` symlink was preserved.

Exact identity/scope lines from the final startup smoke:

```text
# default
INFO web.backend.served_checkpoint served-checkpoint: name=b8-s3 requested=None sha256=3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8 path=/Users/josenunez/Projects/ml/snake-dqn-t15/saved_snakes/b8_s3_t7_20261016.pth reason=None
INFO web.backend.safety_veto_serving safety-veto-serving: active=True scope=watch_opponents mode=watch wrapped_ids=[1, 2, 3, 4, 5] flags={'watch_hero': True, 'play_ai': False} checkpoint_sha256=eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723 strict_checkpoint_match=True wrapper_source_sha256=faf3695fa9d60550f1f681c04b8c8aba0447fecaa34cd75f16f75efe91e4ac05 variant=v8 variant_requested=v8 wrapper_sources_match=True wrapper_method=free-space-veto/v8-space-and-head(lambda=8.0) reason=None
INFO web.backend.served_checkpoint served-checkpoint: watch_roster=mixed requested=mixed hero=b8-s3 opponent=frp3-s12 opponent_pins=[] reason=None
# all_b8
INFO web.backend.served_checkpoint served-checkpoint: name=b8-s3 requested=None sha256=3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8 path=/Users/josenunez/Projects/ml/snake-dqn-t15/saved_snakes/b8_s3_t7_20261016.pth reason=None
INFO web.backend.served_checkpoint served-checkpoint: watch_roster=all_b8 requested=all_b8 hero=b8-s3 opponent=None opponent_pins=None reason=None
INFO web.backend.safety_veto_serving safety-veto-serving: active=False scope=watch_hero mode=watch wrapped_ids=[] flags={'watch_hero': True, 'play_ai': False} checkpoint_sha256=3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8 strict_checkpoint_match=False wrapper_source_sha256=None variant=v8 variant_requested=v8 wrapper_sources_match=None wrapper_method=None reason=the safety veto applies only to a vector61 Apex policy
```

[Startup records](../../research/t15b_watch_default_20261017/results/startups.json)
retain selectors, endpoint state, process IDs and shutdown results. The adjacent
stderr and frame captures retain the raw observations. This is a startup/stream
smoke, not another serving qualification or a completed monitoring window.

The expanded focused Python suite passed **182 tests** (one existing
FastAPI/Starlette deprecation warning). The first full-suite run identified five
additional historical T13/T13b tests whose original all-B8 posture now needs an
explicit test-only selector; all **10 tests** in those two modules pass with
that pin and every original assertion intact. Frozen harness files are unchanged.
The frontend passed **178 tests across
21 files**, and its TypeScript/Vite production build passed. Black, Flake8,
isort and whitespace checks passed. An independent read-only review checked
hero identity, default/invalid/rollback contracts and monitoring separation;
regressions cover fixed B8 telemetry while inspecting an opponent, durable
roster refusal, and per-roster eligible latency.

The required full local Python suite uses `./venv/bin/python -m pytest -q`
(including slow tests), with one OMP/OpenBLAS/MKL thread. GitHub CI on the
published main SHA separately runs frontend tests/build, lint, numerical-plan
checks, Torch parity, local-agent startup and `pytest -q -m "not slow"`.
Final completion requires both the local full suite and that exact main CI run
to pass; the task's final handoff records their results and the CI URL.
