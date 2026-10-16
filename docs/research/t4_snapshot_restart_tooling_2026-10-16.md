# T4b snapshot restart tooling — 2026-10-16

Build-only tooling for the snapshot-restart curriculum in
`specs/t4b_snapshot_restart_spec_2026-10-16.md`. No research training, evaluation
gate, or promotion was launched. All verification uses CPU, with at most two
concurrent processes. Existing processes and tmux sessions were not touched.

## Snapshot construction and identity

`research/t4_snapshot_restart_20261016/build.py` accepts a T3 bank directory or
its collector root. It reconstructs decisive rows from the immutable per-world
receipts and raw shards under `worlds/`. Counter rows are excluded. The original
T3 bank retained every decisive source row.

The existing collector did not persist a complete action tape. The builder first
reconstructs the pinned B8 world using its seed, exact roster, and hash-checked
hero/opponent checkpoints. It verifies selected action, frame, mask, observation
bytes, and available length/Q metadata against the source shard. It records all
six snakes' actions, persists the compressed tape, then independently replays
that tape from the original initial state. Every saved boundary must match the
first pass's full-state hashes. Reconstruction does not run branch labelling or
perform any optimizer update.

Snapshots are taken **before** decision preparation: a snapshot at decision `d`
has `sim.frame == d`; its bank selector row has `frame == d + 1`. Ordinary
`step_with_policy` therefore performs lifecycle preparation exactly once after
restore. The compressed NPZ contains only arrays and explicitly typed JSON,
loaded with `allow_pickle=False`; it contains no Torch modules or executable
pickles. It preserves raw body storage, padded grids, ordered food, RNG state,
last-step facts, opponent roster, and Vector61 trend/carry/selection state plus
explicit featurizer parameters. Standalone restore is independent of the
receiving process's global feature settings; actor grafts reject mismatches. Grid
views are rebound to their padded storage. Immutable checkpoint bytes remain
external and hash-bound. The manifest pins every snapshot; each snapshot binds
its source receipt, source shard, action tape, and source decision.

Standalone restore retains the collector's terminal-hero semantics. Actor grafts
retain the trainer's ordinary hero respawn behavior and configured reward/gamma
schedule. These are explicit training semantics, not claims of source-policy
gameplay identity. Opponent groups and scripted seed tables are rebuilt after
grafting. Standalone source parity uses rowwise CPU inference. Actor grafts keep
the actor's existing batched opponent inference (and configured optional merged
forward mode); they do not claim cross-forward-mode float32 identity.

## Trainer contract

The multi-actor trainer accepts `--restart-bank PATH --restart-frac F`, with
`F=0` by default. An active bank is validated before actor/shared-memory startup.
At each segment, `round(worlds * F)` uniformly selected worlds are replaced with
snapshots sampled with replacement. A separate deterministic RNG keyed by run
seed and segment makes selection independent of the actor's exploration RNG.
At fraction zero there is no snapshot loading, sampling, or restart telemetry.

Each restored world resumes at its saved frame and stops when its own clock
reaches `--segment-frames`. Fresh worlds continue from zero to the same endpoint.
Active masks suppress decisions and stepping after a world's deadline; its
n-step window is flushed independently, using the existing segment truncation
bootstrap convention. Snapshot frames at or beyond the segment end are rejected.

Only newly generated hero decisions count toward `--transitions`, including
ordinary decisions after a restarted hero respawns. Prefix reconstruction,
snapshot loading, and inactive worlds add zero budget. Restarted decisions enter
the same reward/n-step/TD pipeline as fresh decisions; no label-bank loss is
introduced by this feature. As in the existing trainer, segment-tail decisions
count toward the decision budget even when the final row has no next observation
and is omitted from replay during truncation flushing.

Active logs include `restart_frac_actual` (selected worlds / total worlds),
`restart_decisions`, `fresh_decisions`, deaths for both groups, decision-time mean
lengths, and mean pre-death lengths. The summary includes total counts and length
sums. The restart decision share can be lower than the world share because late
snapshots have shorter remaining segments. A world's restarted/fresh label lasts
for that entire segment, including respawns.

## Verification and CPU timing

The measured artifact store is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4b-snapshot-restart-20261016/benchmark-1000-v2`.
The build command above was run with that destination and `--limit 1000`.
Worlds are ordered by a deterministic hash of their receipt names, avoiding a
single-arm lexicographic prefix. This bounded sample contains 1,000 decisive
snapshots from 32 source worlds across all five B8 arms: 708 scripted, 226 mixed,
and 66 frozen snapshots. It is a tooling measurement, not a balanced experimental
bank. The full-bank command below has no sample limit.

**Measured reconstruction time: 512.249 seconds per 1,000 snapshots (8m 32.2s)**
on one CPU process with one numerical thread. This is an actual 1,000-snapshot
measurement, not an extrapolation. It includes world construction, B8 action-tape
generation, saved-row/full-state checking, independent recorded-action replay,
and compressed tape/snapshot writes. Receipt/checkpoint validation and raw-shard
loading precede each timed reconstruction; the separate continuation parity
checks are excluded. Timing was measured with another bounded verification or
read-only task permitted in the second process slot, not on an isolated machine.
The 1,000 compressed snapshots occupy 56,736,379 bytes (about 56.7 kB each),
excluding action tapes and the manifest.

The first sampled world in each mix passed **300/300 exact continuation frames**,
with independently recomputed hero/opponent actions and full simulator/RNG/
Vector61 state checked at every frame. All 1,000 saved boundaries also matched
their original action-generation pass's full-state hashes after independent
recorded-action replay. Source restore keeps the terminal hero behavior even
after death; this check is distinct from actor respawn semantics.

Focused tests verify disk round-trip and 300-frame parity with deaths/respawns,
nonzero-row batch grafting and unchanged neighbors, scripted seed rebinding,
opponent carry/trend memory, digest/config/deadline refusal, independent n-step
flushes, and restart/fresh telemetry. The actual no-learning CPU actor smoke
uses five worlds, fraction 0.2, a frame-30 snapshot, and frame-45 segment ends;
it crosses into a second segment and checks retirement and transition counts.
An `F=1.0` case also checks all-world early segment retirement. Separate
in-process actor checks admit real decisive scripted, frozen, and mixed snapshots
from the measured store into ordinary TD messages. No learner is started.

Final merged-tree validation: **50 passed, 2 intentionally deselected training-run
tests, 25.75 seconds**. This includes existing actor/n-step/trainer tests and the
concurrently merged T4a tests. The default-off actor's messages and observation
ring arrays match pre-change commit `865ae05` byte-for-byte; only wall-clock
timing fields are excluded. Black, isort (Black profile, 100 columns), flake8, and
`git diff --check` pass for the changed Python files. Independent read-only review
found no actionable correctness issue; its two residual coverage suggestions
(all-world retirement and real-bank actor dispatch) are covered by the final tests.

The machine-readable evidence, exact test command, artifact digest, and source
digests are in [the validation receipt](t4_snapshot_restart_validation_2026-10-16.json).
Final store manifest SHA256:
`2fe2a95e847ae5d8429548ae2bb4370a8a68ec44d5198b64c8efdb91fee942af`.

## Reproduction and future arm (not launched)

Build the full snapshot store in one CPU process. The create-only destination
must not already exist:

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
./venv/bin/python -m research.t4_snapshot_restart_20261016.build \
  --bank /Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/bank \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t4b-snapshot-restart-20261016/full
```

Exact command for a future CPU arm, after the full store exists and that
experiment is selected. This is an executable tooling example, not a ratified
experiment or authorization to launch. One actor plus one learner uses two
numerical processes; observation transport uses the trainer's default mmap files.

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
./venv/bin/python research/redesign_m3c_20261008/train_m3c.py \
  --seed 10 \
  --init /Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s0/final.pth \
  --demo-data /Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008/data \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t4b-snapshot-restart-20261016/RESTART_s10 \
  --device cpu --actor-device cpu --actors 1 --worlds 10 \
  --transitions 8000000 --segment-frames 5000 \
  --reward mass --mass-kappa 1 --kill-scale 0 \
  --gamma-start 0.998 --gamma-end 0.998 --gamma-hold 0 --gamma-anneal-end 1 \
  --eps-end-at 1 --mixed-frac 0.25 \
  --anchor-mode v8margin --onpolicy-labels --label-frac 1 \
  --anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0.1 \
  --obs-version bv8 --v61-backend numba --opp-forward per-group \
  --learner-fast --batch-scale 2 --obs-pack --checkpoint-every 2000000 \
  --restart-bank /Users/josenunez/Projects/ml/snake-dqn-artifacts/t4b-snapshot-restart-20261016/full \
  --restart-frac 0.2
```

A paired control would keep the same recipe and seed, use a separate create-only
output directory, and omit both restart arguments. The learning outcome remains
unknown; this task supplies construction and correctness evidence only.
