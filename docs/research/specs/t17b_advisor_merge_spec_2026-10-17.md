# T17b spec: fold the advisor's T17 review into main, then smoke-03 (2026-10-17)

Read first: `docs/research/advice/2026-10-17_t17_plan_review.md` and
`docs/research/advice/2026-10-17_colab_slower.md` on branch `origin/claude/upbeat-allen-uvxqvd`
(cloud advisor). Main already has T17 (Codex, `a660cb22`: `--gather-threads`, ingest cleanup, shutdown,
smoke-03 suite). The advisor's branch has an overlapping shutdown fix (`train_m3c.stop_actors`,
`tests/test_m3c_actor_shutdown.py`) and its own pre-registered smoke-03 matrix (`run_smoke.py --matrix
smoke03`, config keys `obs_pack`, `batch_scale`, `precision`), plus compare/notebook/test edits.

Work in `../snake-dqn-t17` on branch `t17b-20261017` (from origin/main). Commit often; rebase on origin/main
and push to main when green. Do not touch running tmux jobs. Invariants from the T17 spec still hold
(byte-identical batches for the same flags; defaults unchanged).

## B1. Merge the advisor branch

Merge `origin/claude/upbeat-allen-uvxqvd` into this branch and resolve conflicts:
- Shutdown: keep ONE implementation. Prefer whichever is simpler while keeping both test files' guarantees
  (back-pressured actors exit 0 within seconds; a stuck process still gets terminated under one shared
  deadline; stop-aware puts).
- Smoke matrix: ONE mechanism (`--suite` or `--matrix`, keep both names as aliases if trivial). Config keys
  supported: `worlds`, `actors`, `self_play`, `gather_threads`, `obs_pack`, `batch_scale`, `precision`,
  `actor_queue_frames`.

## B2. Row-chunked gather

`--gather-threads N` must use N workers even when there are fewer rings than N: split each gather's
destination rows into N contiguous chunks across rings. Batch byte-identity test extended to 2 rings with
3 and 4 threads, with rejected rows present (compaction path) and with the labels' re-check.

## B3. P0 learner bench in the notebook (runs first, ~5 min)

Before the smoke matrix, run `research/perf_learner_20261013/bench_learner.py --devices cuda` on the
runtime with: batch scale 2 and 4; fp32 and bf16; compiled (reduce-overhead); packed vs raw rings; ring
dir `/dev/shm` vs local disk (log `df -h /dev/shm`); gather threads 0 and 3. Add whatever cells the bench
lacks (raw ring, gather threads) behind flags. Write the JSON next to the smoke results on Drive. It must
report host-gather ms and GPU-leg ms per update separately.

## B4. smoke-03 matrix (pre-registered; 300 s per row; fits the runner cap and notebook budget)

All self-play, 128 worlds, `--actor-queue-frames 4`, reduce-overhead compile:
1. control: a2, packed, g0, bs2, fp32
2. a2, raw, g0, bs2, fp32
3. a3, raw, g3, bs2, fp32
4. a4, raw, g3, bs2, fp32
5. a6, raw, g3, bs2, fp32
6. a4, raw, g3, bs4, fp32
7. a4, raw, g3, bs2, bf16
Plus one profiled 60 s window on row 4 in a separate unscored run (keep cProfile off scored rows).
If budget allows only six rows within the cap, drop row 5 first; do not exceed the runner cap silently.

Per row report: tr/s, updates/s, transitions/update, learn_s per update, policy_lag_mean, GPU/CPU util,
actor exit codes. Verdict (compare.py): PASS if updates/s >= 84 AND transitions/update within 2% of the
row's expected value (256 x batch_scale/2) AND policy_lag_mean <= control's; rows 6-7 are recipe arms,
report them as THROUGHPUT_ONLY (they can show a fallback exists but are not a recipe pass).

## B5. Mac reference cell

Run the same bench cell set on this Mac (`--devices mps`, plus `cpu` if quick) and record ms/update host vs
device in the results note, so we know whether T17 also moves the Mac baseline (21 upd/s).

## Deliverable

Merged + pushed to main; notebook pinned to the final commit, RUN_NAME `smoke-03`; results note appended
with B2 bench numbers, B5 Mac numbers and the pin. Tests: the T16/T17 suites + new tests green.
