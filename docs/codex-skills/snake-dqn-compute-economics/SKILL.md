---
name: snake-dqn-compute-economics
description: |
  Use whenever you size, price, place or schedule compute for /Users/josenunez/Projects/ml/snake-dqn: choosing
  Mac vs Colab (L4 / A100 / H100 / G4 / CPU High-RAM runtimes) vs RunPod, estimating units or dollars for a
  training arm, evaluation bank, teacher rollout or audit, setting est_minutes/timeout for a Drive-queue job,
  judging whether a GPU would help an actor-bound trainer, or reviewing a budget in an advice/spec document.
  Contains the measured throughputs (transitions/s, updates/s), owner-supplied unit rates, cost-per-transition
  arithmetic, the 4x rule, and queue/runtime lifetime facts. Load with snake-dqn-advisor-brief for advice and
  snake-dqn-colab-jobs when writing launchers.
---

# Snake DQN compute economics (measured facts, as of 2026-10-18)

## Colab dispatch boundary

For every new Colab job, price and submit the manifest through `research/colab_cli`, not the Drive queue. The
supervisor admits at most three account-wide assignments including notebook workers, uses OAuth2 with assignment
retries disabled, and applies measured first-cap-wins wall/unit limits with reserve. `research/colab_queue` is legacy
for the two running notebook workers and Q1 only. Charge CLI setup, export, and idle time; no automatic retry or
purchase. `colab usage` is the live account-rate evidence; historical rates below are planning inputs.

Repo `/Users/josenunez/Projects/ml/snake-dqn`; paths relative. Primary sources:
`docs/research/t16_colab_smoke02_results_2026-10-17.md` (T16 smoke-02/03, N=3, T22, T24 appends),
`docs/research/t16_mac_baseline_2026-10-17.md`, `research/perf_quiet_20261013/RESULTS.md` (Mac + 4090),
`docs/research/advice/2026-10-17_after_t23_compute.md`, `2026-10-17_colab_slower.md`,
`2026-10-17_a100_vs_h100.md`, `2026-10-17_colab_review.md`, `2026-10-18_overnight_a100.md`,
`research/colab_queue/README.md`, `docs/STATUS.md` (Operating rules). Rates marked **owner-supplied** are
planning inputs, not quotations: record the live rate from Colab's resources panel at admission.

## 1. Workload shape (why GPUs rarely help here)

- Learner: ~1.5-1.64M-parameter ego2s-bv8 network; one fp32 `--batch-scale 2` update = 768 TD rows + 256
  anchor rows, issued once per **256 ingested transitions**. Honest throughput metric = **updates/s**;
  transitions/s = 256 x updates/s. 4x the Mac = ~84 upd/s (~21.6k tr/s).
- Actors: Numba simulator + featurizer + v8 feature kernels on CPU cores; opponent policies (frozen vector61 +
  v8 veto chains, scripted) also on CPU. Hero/N=3 configurations are **actor-bound**; all-snakes self-play
  moves forwards onto the GPU and becomes **learner-bound**.
- x86 cloud cores are slow relative to the Mac: Colab ~3.7x slower per 128-world actor frame (63 ms vs ~17 ms,
  `2026-10-17_colab_slower.md`); RunPod EPYC ~2.4x slower per actor (`perf_quiet` RESULTS). Colab has 12 vCPU /
  6 physical cores on A100 runtimes; the measured L4 runtime also had 12 vCPUs (2026-10-17) (`2026-10-17_colab_review.md`).
- GPU leg of an A100 update is ~8-12 ms and hides under ~19-23 ms of host work until gather threads are on;
  after that the A100 learner caps at ~78-83 upd/s fp32 bs2 even alone. bs4 is linear in cost (no gain);
  bf16 reaches 117 upd/s isolated but only 47.6 in a full run (unexplained) and is a numerics change.

## 2. Measured throughput table

| Platform / config | tr/s | upd/s | Bound by | Source |
|---|---:|---:|---|---|
| Mac M5 Pro, MPS, hero-only, 128w x 2 actors | 5,283 | 20.6 | learner (~48 ms/update) | t16_mac_baseline |
| Mac, self-play (`--all-snakes-learner`) | 5,459 | 21.3 | learner | t16_mac_baseline |
| Mac, N=3 league (3 learner seats + static 6-policy pool) | 5,214 | 20.4 | learner (46.5 ms/update); lag 0.75 | t16 "Mac N=3 baseline" |
| Mac end-to-end B8 recipe, 128w x 2, bs2 (quiet) | 5,364 | - | learner busy 0.90 | perf_quiet RESULTS |
| Colab A100-40GB hero, a2 / a4 / a8 | 4,062 / 8,014 / 9,138 | 15.9 / 31.3 / 35.7 | actors (6 cores) | t16 smoke-02 |
| Colab A100 self-play smoke-02 (any a2/a4) | ~11.1-11.5k | 43-45 | learner host thread | t16 smoke-02 |
| Colab A100 self-play smoke-03, a4 raw g3 bs2 fp32 | 15,927 | 62.2 | learner GPU leg (84% busy) | t16 smoke-03 |
| Colab A100 isolated learner fp32 bs2 (no actors) | ~20-21k capacity | 78-83 | GPU leg | t16 smoke-03 / N=3 P0 |
| Colab A100-80GB N=3 league, a3 / a4 / a6 | 4,396 / 5,938 / ~7,720 | 17 / 23 / 30 | actors: frozen-pool CPU veto chains | t16 "A100 N=3" |
| A100 N=3 after T22+T24 host fixes (projection, NOT measured) | ~13.6-16.0k (<= 3.07x Mac) | - | remaining actor work | t16 T22/T24 appends |
| Colab L4 N=3 (12 vCPU) | 7,505 (a3) / 7,625 (a4), 29-30 upd/s, GPU 87-90% | - | GPU (L4) | t16_colab_smoke02_results (L4 section, commit 6f8e8013) |
| RunPod RTX 4090 learner, compile reduce-overhead, bs2 | 22,212 ceiling | 87 (11.5 ms) | - | perf_quiet CUDA |
| RunPod 4090 actor, B8 recipe, 128 worlds, one actor | 4,056 (2,987 long bodies) | - | EPYC core | perf_quiet CUDA |
| RunPod pods end-to-end (M3-B, 5 x 20M) | ~5.9k combined (4.25x Mac) | - | actors (3-4 per run) | docs/STATUS.md M3-B row |
| H100 | **not measured**; projection 28-32k tr/s after host fix, only if GPU-bound | - | - | t16 smoke-02 diagnosis; colab_slower |
| G4 (RTX PRO 6000 Blackwell, sm_120) | **not measured**; needs torch cu128 (`setup_runtime.sh` picks cu128 for compute cap >= 10) | - | - | research/t16_colab/setup_runtime.sh |
| N=1 task-aligned collector | **no qualified rate**; budget at 1/3 of N=3 as a sensitivity only | - | - | q0q1 PREREGISTRATION |

Rule of thumb from the record: more actors beyond the learner's consumption only raise policy lag (a2 = a4 in
tr/s, lag 4.4 -> 9.0); more GPU does nothing for actor-bound rows; "4x at the fixed fp32 recipe is not
reachable on the A100" (smoke-03 conclusion).

## 3. Prices and cost per transition

Unit rates (Colab Pro+, ~1,250 units available on 2026-10-17; ~1,300 per the colab skill):

| Runtime | Rate | Status |
|---|---|---|
| L4 | 2 units/h | owner-supplied planning input (public figures 2.5-5) |
| A100 (80GB) | 6.8 units/h | owner-supplied planning input (older public figures 9-12) |
| A100-40GB High-RAM | ~5.3 units/h | observed 2026-10-18 (`overnight_2026-10-18.md`) |
| H100 | ~13 units/h | public figure; scarce; not needed for current work |
| T4 | ~2 units/h | public figure; not used |
| CPU High-RAM | **unknown**; measure and charge it, never record zero | q0q1 PREREGISTRATION |
| RunPod 32-vCPU CPU pod | ~$0.89-0.95/h | `docs/research/t4_pod_plan_2026-10-15.md` |
| RunPod 4090 secure pod | bench session ~$0.5 total; M3-C 15 x 20M runs ~$7.9; M3-B 5 x 20M $3.92 | perf_quiet, m3c learnings, STATUS |
| RunPod serverless | real cost ran ~6-10x catalog; fallback only | docs/STATUS.md |

Derived cost per transition (pure training, excludes setup/export/eval; recompute if the rate changes):

| Placement | tr/s | Million transitions per unit | One 8M seed | Five seeds (40M) |
|---|---:|---:|---:|---:|
| L4 at 7.6k, 2 u/h | 7,600 | 13.7 | 17.5 min, 0.58 u | 1.46 h, 2.92 u |
| A100 at 7.7k (N=3), 6.8 u/h | 7,700 | 4.1 | 17.3 min, 1.96 u | 1.44 h, 9.81 u |
| A100 at 14k (projection), 6.8 u/h | 14,000 | 7.4 | 9.5 min, 1.08 u | 0.79 h, 5.40 u |
| A100 self-play at 15.9k (smoke-03), 6.8 u/h | 15,900 | 8.4 | 8.4 min, 0.95 u | 0.70 h, 4.76 u |
| A100-40GB idle/allocation at 5.3 u/h | - | - | - | 5.3 u per hour regardless of work |
| Mac at 5.2k | 5,200 | no Colab debit; occupies strict-gate capacity | 25.6 min | 2.14 h |

Decision rule from the compute advice: A100 must finish the identical job > 3.4x faster than L4 (6.8/2) to
cost fewer units; at 7.7k vs 7.6k it is ~1.01x, even 14k vs 7.6k only 1.84x. **L4 for replicated small-model
online RL; CPU runtimes for simulation/evaluation/teacher branches; A100 for batched offline fits or a declared
deadline or a job L4 cannot hold in RAM; H100 only after a profile shows kernel-busy > 80% at the fixed recipe.**
Five extra minutes of runtime = 0.167 L4 units or 0.567 A100 units. Report aggregate runtime-hours, not elapsed
time across parallel workers; parallelism buys time-to-answer, not units or evidence.

## 4. Evaluation and audit costs (usually the binding item)

- Mac, single thread, B8 bare hero, H5000 world: ~13-15 s when the hero survives, ~4 s on an early death
  (`docs/research/q0_mass_mac_timing_20261018.json`). Old vector61+veto gate episodes: ~18-33 s/episode on Mac
  slots, up to ~70 s under thermal throttling (`docs/experiments/apex_veto_v8_2026-10-03/README.md`,
  `apex_veto_strict_gate_2026-09-27/README.md`).
- Colab CPU: ~2-3x slower per core than the Mac; a Q0 audit that ran 480 H5000 episodes on one core hit its
  4 h timeout, and T23 evaluation used one worker per job (`~/.codex/skills/snake-dqn-colab-jobs/SKILL.md`).
  Always use a process pool (`--workers`, default `min(cpu_count-1, 10)`) with deterministic per-world merge.
- Bank sizes to price: Q1 = 5 seeds x 128 worlds x 5 identities = 3,200 episodes; Q2 = 1,920; a 64/32/32
  development bank per seed; strict gates run n up to 600/mix with calibration (FRP3 strict stopped at
  n=207/mix). Price from a disjoint timing fixture at the real length distribution, never from training tr/s.
- Teacher/rollout jobs are priced in **branch world frames** (e.g. P0 pilot 512 x 6 x 4 x 128 = 1.57M frames;
  Q3b upper bound ~615M frames for 640 episodes), measured inclusive of clone, features and policy inference
  (`2026-10-17_after_t23_compute.md`, Q3; `_rl.md`, P0).
- Mac strict planning ceiling proposed at 72 wall-hours on <= 2 single-thread strict workers; size the bank from
  development paired variances before accessing strict outcomes.

## 5. When GPU helps vs CPU-bound actors (diagnostic recipe)

1. Read the sampler columns in the smoke row: GPU% mean/max, CPU%, `cpu_affinity`, oversubscription, and
   `policy_lag` (`research/t16_colab/sampler.py`, `run_smoke.py`).
2. CPU near 100% on physical cores with GPU low -> actor-bound: the only levers are cheaper frames (T22/T24-style
   byte-identical kernels), fewer/larger actor frames, or more runs per session. Chip upgrades do nothing.
3. GPU > ~70-80% with CPU < 80% and `learn_s` ~ wall -> learner GPU-bound: consider H100 (fp32 67 vs 19.5
   TFLOP/s), but at ~13 u/h it must still beat L4/A100 on units per completed seed.
4. GPU ~50% and `learner_wait` small -> learner host-thread-bound: gather threads (`g3`), raw rings instead of
   `--obs-pack` (RAM 10-11 GB for 4-6 actors), ingest cleanup. These are byte-identical and allowed.
5. Anything that changes replay ratio, batch, precision or actor count per arm is a recipe change -> paired
   quality arm, not a speed-up.

## 6. Legacy queue and runtime lifetime facts

Drive job queue (`research/colab_queue/README.md`):
- Job JSON: full 40-char `commit`, repo-relative Bash `script`, `requirements.gpu` in {any, A100, L4, cpu}
  (`any` excludes CPU runtimes; CPU work meant for a GPU worker must still request that GPU), `min_vcpu`,
  `est_minutes`; `timeout_minutes` defaults to 2x est; setup limit 1,800 s; `output_dir` job-private under the
  Drive root; `depends_on` needs current-generation `done` receipts.
- Worker heartbeat every 15 s; stale after 300 s; a worker with no eligible job for 10 min releases the runtime
  (`drive.flush_and_unmount` then `runtime.unassign()`), unless the notebook's `KEEP_RUNTIME = True` sets
  `IDLE_MINUTES=100000`. Changing env does not affect a running worker. Never requeue on a stale heartbeat alone.
- Bounded-visibility claim protocol (not exactly-once): pre-create shared directories from the Mac; concurrent
  `mkdir` from two runtimes produced `evaluation` and `evaluation (1)` in T23.
- Launcher hygiene: measure >= 3 real units on the Mac, scale by 2-3x for Colab cores, set timeout >= 2x;
  stream progress every ~2 min; resumable units; local disk first, then Drive
  (`~/.codex/skills/snake-dqn-colab-jobs/SKILL.md`).

Colab runtime facts:
- Colab's FAQ: hardware availability and limits vary; a subscription reserves no particular accelerator; H100s
  are scarce (keep an assigned one by chaining jobs). Sessions disconnect: checkpoint full state to Drive
  asynchronously every few million transitions, resume by verified snapshot (`research/t16_colab/long_run.md`).
- Idle A100 time is billed at the allocation rate (overnight 2026-10-18: ~5.3 u/h with 40 s of useful work);
  record idle debit separately from job-attributed cost.
- Units have historically expired 90 days after purchase (`2026-10-17_colab_h100.md`); check the expiry date.
- Python 3.12 uv venv on a 3.13 kernel; torch cu126 for A100/L4/H100, cu128 for Blackwell; numpy 1.26.4 pins.

RunPod facts (`docs/STATUS.md`; Claude skill `~/.claude/skills/runpod/NOTES.md`, HARDWARE.md, OPTIMIZATION.md):
- Per-step rule: use RunPod only if projected wall-clock >= 4x the Mac for that step, measured incl. setup and
  transfers; waivers are per step and recorded. Cheap CPU pods first; serverless only as fallback. Spot is gone;
  v1 API dies 2026-11-15; a pod-side self-delete probe returned 403 (owner decision pending), so H100 pod runs
  are paused. Strict gates never run on pods.
- Pods are actor-bound at ~4k tr/s per run: run 3-4 actors per run and several runs per pod; devel image needed
  for `torch.compile` (`perf_quiet` RESULTS).

## 7. Budget envelopes currently in force (do not exceed without the owner)

- Q0+Q1: <= 150 units (+25 by Amendment A1 for A100-priced Q0), 24 GPU-runtime-h, 40 CPU-runtime-h, 20% reserve
  per envelope; Q0 cap 40 units; Q1 training <= 60 units + admitted CPU eval; <= 3 concurrent runtimes.
- Reserve 100 units for Q2 or P0/Q4; total initial commitment <= 250 of ~1,250. Retain >= ~1,175 units until the
  first tranche yields a decision (`2026-10-17_after_t23_compute.md`, "Commitment").
- RunPod: standing permission for the improvement loop with per-step caps ~$10; account balance/limits per the
  runpod skill.
