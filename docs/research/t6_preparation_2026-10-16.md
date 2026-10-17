# T6 RESTART preparation — 2026-10-16

The pre-registration was committed and pushed before implementation and the
real-store MPS smoke. The smoke passed. Both 8M launch lanes are prepared and
were checked only with `--check`; no 8M training or evaluation was started.

## Registration and implementation

- [Pre-registration](specs/t6_prereg_2026-10-16.md): commit `ac2f78d`, integrated
  and pushed as `e7981a23`; immutable SHA256
  `640786c239580aab46701c520091d5455956dd699cb61be64cddce310d2f9307`.
- Telemetry and fixed recipe: `d219f5143ba1409c95aeaf1c6f23be7f183b66c2`, pushed
  before the smoke. Added integer decision-time L>=800 decision/death counts for
  restarted and fresh worlds, their per-1,000 rates, and null for empty denominators.
- Every shared parsed argument matches the recorded M6 CTRL_s10 summary except
  output path and explicitly pinning its effective `per-group` opponent default.
  Required restart/tripwire flags are the only active experimental additions.
- The stale actor-parity fixture was pinned to pre-T6
  `f3375b26d410c4de0feb2cd049c6021171902e53`, retaining strict message/ring byte
  equality and including the pre-existing T4f boost telemetry.

## Actual smoke

The exact command, environment, artifact hashes and checks are in
[t6_smoke_validation_2026-10-16.json](t6_smoke_validation_2026-10-16.json).
It ran from `/Users/josenunez/Projects/ml/snake-dqn-t6`, seed 10 from B8_s0,
`--device mps --actors 2 --worlds 128 --transitions 200000`, using the full
10,885-snapshot store. Environment device was MPS, not a CPU override.

| Measurement | Observed |
|---|---:|
| Hero decisions | 200,228 (normal message rounding) |
| Trainer wall time | 79.095 s |
| Trainer rate, one lane | 2,531.498 decisions/s |
| End-to-end time including startup | 114.570 s |
| End-to-end rate | 1,747.641 decisions/s |
| Restart world fraction | 0.203125 = 26/128 |
| Restart / fresh decisions | 38,864 / 161,364 |
| Restart L>=800 decisions / deaths | 27,896 / 14 |
| Restart L>=800 deaths per 1,000 | 0.501864 |
| Fresh L>=800 decisions / deaths | 0 / 0 (rate null) |
| Exit / actor exits | 0 / [0, 0] |
| Ring overruns | 0 |
| Halt | null |

Greedy boost rate was 0.15265% then 0.10900% at the two 100k reports;
below-700 rates were 0.02639% then 0.04473%. Both remained below the registered
2% / 1% thresholds. Existing nonhalting Q-scale flags appeared at both reports
(Qmax 141.05 / 140.07); they were not tripwire halts. All required restart/fresh
fields were present and interval counts reconcile with summary totals.

The final smoke checkpoint is an infrastructure artifact, not an 8M final or
model-quality result. No fresh-world L>=800 conclusion is possible from the zero
denominator. The store's restore behavior and training outcome are separate
questions; this preparation does not confer promotion.

## Unstarted launch lanes

External scripts, with identical tracked copies under
`research/t6_restart_20261016/`:

- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t6-mac-20261016/lane0.sh`
  — seeds 10, 12, 14.
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t6-mac-20261016/lane1.sh`
  — seeds 11, 13.

Each `--check` validated pinned inputs and every snapshot digest and printed full
commands using MPS from this worktree. The check outputs are `lane0-check.txt`
and `lane1-check.txt` beside the scripts. No `RESTART_s*` output or lane-start
receipt exists. Scripts refuse existing outputs/logs, hash drift and arm KILL;
a nonzero/halting child records KILL and stops subsequent seeds. They inspect
only their own arm receipts and never signal unrelated sessions or processes.

Focused verification: 16 restart/snapshot tests, 9 boost/calibration integration
tests and 4 launch tests passed. Launch tests use mocked child execution and
confirm that `--check` cannot launch. Shell syntax, formatting, lint and diff
checks passed. The independent recipe/telemetry review found no actionable issue.
The separately reported seed audit must be refreshed at evaluation launch.

No tmux session was accessed. The bounded smoke's actors exited cleanly; no
8M run, evaluation, promotion, or unrelated process control was performed.

## Evaluation seed audit

The executable audit is `research/t6_restart_20261016/seed_audit.py`; its saved
receipt is [t6_seed_audit_2026-10-16.json](t6_seed_audit_2026-10-16.json).
It checks rounds 72/73 against prior registered history, the T3 namespace, all
1,200 T4e planned collection worlds and all 319 source-world seeds of the
hash-bound full snapshot store. JSON/JSONL scanning includes ignored and hidden
paths, with only the exact current output receipt excluded.

A separate conservative inventory covers 41 reconstructable recorded/planned
training configurations. Dead heroes and respawn waits prevent using
worlds-times-frames as a decision lower bound; the inventory instead guarantees
only fresh segment-start decisions and includes actor partitioning and queue
prefetch. A generated seed-prefix collision caps the envelope because the
trainer refuses such a prefix. Potential collisions in this envelope would
block the audit rather than be reported as proven actual world reuse.

Seven legacy summaries lack enough arguments for this reconstruction and are
listed explicitly in the receipt. Neither this inventory nor text scans prove
anything about unrecorded/external experiments. This preparation is not an
evaluation launch qualification: refresh the audit against then-current evidence
and resolve any remaining history coverage before probing. Six focused seed-audit
tests passed, including collision and ignored/hidden-path coverage.
