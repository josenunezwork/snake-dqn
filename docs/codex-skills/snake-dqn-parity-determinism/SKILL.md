---
name: snake-dqn-parity-determinism
description: |
  Use whenever a change in /Users/josenunez/Projects/ml/snake-dqn must be proven to change nothing observable:
  simulator/featurizer/veto/pool/learner optimizations, numba kernels, process pools, resume paths, or any
  claim of "identical" / "bit-exact" / "same math". Also use when a Mac-vs-pod/Colab identity check fails or
  when writing an oracle test against the frozen pre-change implementation. Covers tests/test_grid_sim_parity.py,
  test_ego_live_identity.py, test_perf_sim_identity.py, the T17/T22/T24 oracle pattern, serial==parallel proofs,
  and the float32 Q-tie cross-platform rule.
---

# Snake DQN: parity, identity and determinism

When parity work needs a new Colab development runtime, use the lease-fenced `research/colab_cli` supervisor and
retain its pinned-input/hash-verified-export receipt. Do not route it through the Drive queue; that transport is
legacy Q1-only.

This project promotes on bit-exact evidence. "Close enough" is never accepted for a same-math claim;
a numerics change (bf16, compile reassociation, batched BLAS) is labelled as such and screened separately.

## The parity ladder (what already exists; extend, do not duplicate)

| Layer | Test / harness | What is compared |
|---|---|---|
| Live game vs BatchSim vs GridBatchSim | `tests/test_simd_parity.py`, `tests/test_grid_sim_parity.py` (runs every test on NumPy and numba kernels via the `grid_kernels` fixture) | per frame: bodies, lengths, alive, heading, boost/hunger/respawn counters, ordered food, corpses, per-env RNG state, advisory/legal/resolved masks, rewards, done, death causes, kill credit, events; big-body regime (serpentines 150-900) |
| Observation | `tests/test_ego_live_identity.py` (sim `ego_view` vs `ego_live_adapter` on `PyRefGame`), `test_obs_parity.py`, `test_ego_raster_bv8.py` | local/global planes and scalars **bytewise** (`tobytes()`), both featurizer backends; gate-world version `research/redesign_scope_20261007/ego_live_identity.py` |
| Eval engine | `test_eval_runtime_parity.py`, `test_eval_controlled_parity.py`, `test_simd_parity_h5000_check.py`, `test_simd_parity_v7v8_check.py` | H5000 records, veto chains live vs simd |
| Hot-path optimizations | `tests/test_perf_sim_identity.py` + `tests/perf_sim_reference.py` (verbatim main@db2ef7a twins) | each optimized function vs its frozen twin on seeded adversarial worlds; floats by `repr` (0.0 vs -0.0 fails) |
| Trainer internals | `test_perf_learner.py` (fast == reference bitwise on CPU, `fused_adam=False`), `test_perf_actor.py`, `test_obs_pack.py`, `test_t17_gather.py` (frozen pre-T17 gather oracle: staging bytes incl. padding, RNG, compaction, 1/2/3 rings x 2/3/4 threads), `test_m6_reward.py` (replays pre-M6 actor from git) | ring rows, staging blocks, RNG state, messages |
| Frozen pool / v8 kernels | `tests/test_t22_frozen_pool.py` (fixture `tests/fixtures/t22/league_pool_pre_t22.py` from 4c1e1b2; recorded 96x12x6 action stream), `tests/test_t24_v8_bodymap.py` (`fixtures/t24/v8_nb_pre_t24.py`, SHA `72947d97...`) | actions, learner raster/v8 bytes, carry/trend memory, world + RNG state through forced deaths and respawns |
| Whole episodes across trees | `research/perf_sim_20261007/{episode_harness,actor_harness,compare_identity}.py` (`--root` = `git archive <rev>` tree) | per-frame world digests, every network forward's float32 bytes, records |
| Serial == parallel | `docs/research/q0_mass_worker_parity_20261018.json`, `tests/test_q0_mass_execution.py`, T23 `parity.py` (`tests/test_t23_parity.py`) | SHA-256 of rows/candidates/reservoir from original serial vs worker-serial vs N spawned workers |

Run the relevant slice before claiming identity, e.g.
`./venv/bin/python -m pytest -q -m 'not slow' tests/test_grid_sim_parity.py tests/test_ego_live_identity.py`
and `-k '(obs_pack or m3c or t17 or t22 or t24) and not mps'`.

## How to write an oracle test against the frozen pre-change code

1. **Freeze the reference, independently of the working tree.** Either vendor the unedited module into
   `tests/fixtures/<tag>/<module>_pre_<tag>.py` and pin its SHA-256 in the test (T22/T24 pattern), load it
   with `importlib.util.spec_from_file_location` under a private module name; or `git show <rev>:<path>`
   at test time (`BASE_REV` in test_m6_reward.py; `--reference` in bench_frozen_pool.py:60). Never let the
   oracle import the candidate's helpers (T24 made the oracle import the frozen `v8_nb` explicitly so both
   sides could not share the new kernel). Do not "fix" or modernize reference files: they are evidence.
2. **Separate everything else too:** separate simulators, runtimes, policies and RNGs per side, driven in
   lockstep by one recorded action stream (store it: `tests/fixtures/t22/actions_seed741.npz`, SHA in README).
3. **Compare bytes, not values with tolerance:** `np.array_equal` + `tobytes()`; for Python floats use
   `repr`; for dataclass results compare every stored field including lazy sentinels.
4. **Stress the proof's assumptions, not the happy path:** long/coiled bodies, overlapping snakes, heads on
   and outside walls, integer coordinates off the lattice, wrapped ring indices, duplicate and shuffled row
   selections, dead rows, forced deaths and respawns, exact Q ties on every slot, stale caches, rebinding,
   >63 snakes fallbacks (T24), rejected-row compaction and padding bytes (T17).
5. **Both arms of a flag:** off == old bitwise; on == old except exactly the documented term, recomputed
   independently in the test (test_m6_reward.py).
6. **Record the receipt:** commit, source SHA-256s, platform, command, pass counts (research/<tag>/README.md,
   `verification*.txt`). Receipts bound by sha256 are never edited; add a sibling erratum/status file.

## Serial == parallel

For process pools (eval worlds, bank audits, Colab workers): deterministic per-unit seeds, one world per
worker, results merged in a fixed order independent of completion order; prove with a test that hashes the
serial output and the N-worker output (N >= 2, include a "worker-serial" middle arm) and asserts equality
(`q0_mass_worker_parity_20261018.json`: `reservoir_rows_equal: true`, `parallel_rows_sha256 == ...`).
For threads inside the learner (`--gather-threads`): partition contiguous destination rows, write into
preallocated staging, keep the RNG draws on the main thread so the draw order is unchanged; the test
compares the whole staging block including padding and the post-copy validity re-check (`_FlakyMulti`).
Caution: `pgrep -f` matches its own command line; wait on PIDs/output files instead.

## Cross-platform identity: the float32 tie rule

- Mac (Accelerate BLAS) and x86 (MKL AVX2) sum in different orders: raw Q differs at ~5e-5 absolute
  (docs/research/runpod_x86_consistency_2026-10-03.md). Decisions usually agree (28/28 there) but **exact
  float32 Q ties flip** (~1 in 50 H10000 episodes; FRP-v5-S2 1/54 failure: actions 2 and 5 both
  99.66497040 at frame 3997). This is not a code bug and not a waiver.
- Consequences (STATUS.md, governance_amendment_strict_on_runpod_..._status_2026-10-08.md): strict gates and
  serving qualification run on the Mac; any Mac<->pod/Colab identity check must **explain** divergences -
  record a per-decision action-hash stream with the top-2 Q margin and accept a mismatch only when the
  first divergence is a near-exact tie (<= ~4 ulp) on the Mac side; keep the per-world single-CPU-model rule
  (every arm and control of one (mix, world) on one worker/CPU). Epsilon tie-breaks alone are not enough;
  float64 action selection changes behaviour on ties (references would need regeneration).
- Same-platform results and paired per-world comparisons stay valid. Colab/pod parity receipts are
  "tested CPU action identity on that host", never cross-architecture qualification; say so in the README
  (T22/T24 wording).
- Mac replays are deterministic: `random.seed`, per-env `_rngs`, torch single thread. Keep thread limits at 1
  and `SNAKE_DQN_DEVICE=cpu` for identity runs; MPS/CUDA floats differ (`--opp-forward merged-device` is
  documented as not same-math).

## Mistakes that already happened

- Assuming bitwise Mac<->x86 identity after a 28/28 sample; the tie failure appeared later (S2 1/54).
  The 54/54 pass of FRP-v5-H was a lucky draw.
- Batch-N vs batch-1 torch forwards treated as identical: Q differs by ~6e-6, `torch.equal` False at batch 2.
- NumPy `arctan2` proposed for a libm `atan2` path: 1-ulp sector-boundary flips (perf_plan §2, not done).
- Oracle sharing code with the candidate (fixed by explicit frozen-kernel import in T24).
- Caches not bound to frame/simulator identity (f15fb292) and frozen pool models dropped across
  segments (99c27b0e): every cache needs identity keys + a stale-cache test.
- Numba scratch shared across threads (f7e74417): use thread-local workspaces.
- Resume paths that silently changed segment gamma / actor boundary (10236da1, c53b5d37): a resume is a
  parity claim too - restore then compare counters, schedules and the next rows.
- Writing identity "PASS" on a sample too small to see ~2% per-episode events; report the sample size and
  what the receipt does not qualify.
