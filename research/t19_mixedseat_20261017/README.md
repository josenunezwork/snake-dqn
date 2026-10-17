# T19 mixed-seat tooling status (2026-10-17)

This directory records prepared tooling for step 4 of
`docs/research/specs/h100_preruns_plan_2026-10-17.md`. It is not a launch
receipt, training result, promotion result, or serving change.

The `n3` suite in `research/t16_colab/run_smoke.py` contains exactly two
static-pool rows: 128 worlds, fp32 B8 recipe, raw rings, gather threads 3,
three learner seats, and either three or four actors. It uses the trainer's
source-defined frozen six-policy pool. The bundled B8 bytes are explicitly
passed with `--league-b8-checkpoint`; the dispatcher binds all pool identities
to their checkpoint SHA-256 values at startup.

Each completed N3 row must lift the learner-decision (D), replay-row (R), and
update (U) ledger; per-policy live rows; mean and p95 policy lag; and frozen
GPU-forward time. Missing league telemetry makes the sample invalid. N3 is a
measurement-only suite and has no comparison or promotion verdict.

The trainer extension is opt-in via `--learner-seats {1,3,6}` with `--actors >= 1`.
N=1 is explicitly the legacy hero-only collector, including its historical opponents.
N=6 is the existing all-snakes collector. N=3 requires BV8 observations and permutes
learner seats with a separate seeded RNG at each segment; ownership survives respawn.
Frozen slots independently draw uniformly from the full static pool and retain that
assignment for the segment. The six permanent policies are B8-s3, FRP3-s12 bare,
FRP3-s12+v8, champion+v8, greedy_food, and random_safe. Both v8 wrappers are supported.
Models remain resident in each actor; vector policies/scripts stay on CPU, while
frozen B8 uses the actor device. Pool records bind checkpoint, observation, mask,
wrapper and source identities and are saved in checkpoint metadata and the summary.

`learner_decisions` (D) alone advances the collection schedule; `replay_rows` (R)
funds updates, including warmup debt and segment tails. Batch-scale 2 retains 256
R/update. `per_policy_live_rows` includes a `learner` total plus frozen policy IDs;
only learner rows enter the observation ring, replay and labels. Lag is measured in
learner update publications per received actor message. Smoke aggregation uses the
interval histograms for the scored-window mean and p95. `frozen_forward_gpu_s` is
cumulative CUDA event time for frozen BV8 forward/selection, excluding CPU features;
it is zero on CPU/MPS. Smoke rows expose its window delta as
`frozen_forward_gpu_seconds`, D/R deltas, `measured_updates` (U), and
`live_rows_by_policy`. Actual CUDA execution is not qualified by the Mac tests.

PFSP, dynamic snapshots/cache eviction, exploiter scheduling, payoff adapters and
exact training resume are deferred. N=1 static-league exploiters are also outside
this legacy-compatible N=1 slice. The existing 16-policy admission bound is enforced;
there are no dynamic resident models in this six-policy static pool.

`research/t16_colab/mac_n3_baseline.sh --check` validates and prints the
prepared MPS N=3 command. Without `--check` it is a bounded 420-second run
with a 120-second warm-up exclusion and a nominal 300-second timing budget. The
observed scored interval is between log boundaries and is reported as
`measured_seconds`; it is not guaranteed to span all 300 seconds. It refuses
to overwrite an existing receipt. The MPS learner/actors, 128 worlds, two actors
and fixed B8 recipe follow the T16 Mac baseline; raw rings, gather threads 3 and
the longer budget match the new league workload rather than reusing the older
packed-ring 180-second rates. The script binds the frozen checkpoint root
with `SNAKE_CKPT_DIR` (defaulting to the established source checkout).
`SNAKE_PYTHON` overrides the interpreter; linked worktrees otherwise reuse the
main checkout's venv. The script checks clean actor shutdown and complete league
metrics before writing `throughput.json`, and cleans up its own temporary rings.
`research/t18_h100pod_20261017/commands_n3.json` contains the corresponding
unlaunched 30-minute H100 command. It is separate from the existing T18
`commands.json`, whose fixed P0/smoke03 workload is unchanged.

No pod, long training, or tmux job was launched while preparing these files.

Verification on the Mac: 66 focused league/shutdown/runner tests passed; 15
non-slow trainer tests passed; 50 legacy M6/actor/league compatibility tests
passed; and 77 input-bundle/environment/notebook/sampler/comparison/pod-runner
tests passed. The batch-scale checks (including the bounded 12k-transition
fast-learner regression) passed. The 26 non-slow compile tests passed with the
command-local `DYLD_FALLBACK_LIBRARY_PATH=/usr/lib` needed by this host's
TorchInductor shared libraries; no global environment was changed. These sets
overlap and are not a full-repository test count. Slow suites were not required.

A separate bounded real CPU N=3 integration smoke completed 1,032 learner
decisions, 924 replay rows and three updates (`floor(924 / 256)`), with all six
pool identities recorded, actor exit `[0]`, no halt and no ring overrun. The
remaining replay-token debt was 156 rows. This verifies execution and accounting,
not throughput or learning quality. CUDA execution, the prepared Mac baseline,
and H100 measurements remain unrun.
