---
name: colab-jobs
description: Launch, checkpoint, monitor and recover snake-dqn training/benchmark jobs on the owner's Colab compute. New jobs = Colab CLI supervisor research/colab_cli (no owner clicks); the Drive queue research/colab_queue + worker notebooks is legacy, kept only for already-running work (Q1, 2026-10-18). Covers manifests, Mac preflight, monitoring, recovery, result import. Use with the colab skill whenever work runs on Colab.
---

# Colab jobs (snake-dqn)

Canonical operator guidance is `docs/colab-cli.md`. For every new job, use its supervisor interface rather than a
direct `colab new`/`exec`/`upload`/`download`/`stop`; only the existing Q1 workers may use the Drive queue.

## New jobs: Colab CLI supervisor (default since 2026-10-18)

Every new job kind goes through `research/colab_cli/` (see `docs/colab-cli.md` for exact commands) — owner instruction
2026-10-18 "update everything to now use colab cli except for whats running". Checklist:
1. Pre-registration first for any learning experiment; manifests pin a full commit, Bash `.sh` script, env, requested
   GPU plus any pre-priced `allowed_gpus` substitution list, est/timeout, and fresh output dir. No Mac-only paths:
   ship hash-bound inputs in the uploaded bundle. At admission, account for every pre-existing session endpoint in
   `budget.other_assignments` with a reviewed positive rate; unaccounted endpoints refuse the allocation. After
   allocation, take a bounded provider-rate calibration (up to 120 seconds, three stable positive reads five seconds
   apart, stable within 0.01 units/hour) after subtracting reconciled other assignments and allowing provider lag.
   Observed must be <= priced upper-bound rate + 0.05 units/hour; unknown, unstable, unexplained, or over-ceiling
   calibration is INCOMPLETE. Thereafter provider-rate drift is checked against the observed rate, while the job cap
   stays the priced upper bound times elapsed time. The account balance is only a
   baseline-subtracted reconciliation check with five minutes of provider-lag allowance. It records a warning, never
   a raw-account cap that charges Q1 or another lane to this job.
   Acceptance pricing defaults to CPU 0.5 and L4 2.0 units/hour, overridable with `--cpu-rate`/`--l4-rate`;
   the measured CPU and L4 rates are outputs rather than required inputs.
2. **Mac preflight of the real entrypoint** on a tiny real input with the manifest env (measured per-unit time x units
   / workers -> est; timeout >= 2x; progress lines every ~2 min; parallel workers for CPU-bound loops). Codex must
   load the `snake-dqn-colab-jobs` skill.
3. Submit via the supervisor; it allocates (max 3 concurrent incl. notebook runtimes), verifies runtime GPU
   model/capability and writes `hardware.json`, uploads, launches, polls progress files, pulls completed snapshots
   to the Mac (then Drive-for-desktop backup), downloads results, stops, verifies release, and records account usage
   after stop. The job creates consistent optimizer/replay/actor/target/RNG/counter state; the supervisor transports
   and hash-verifies it. Exports use 8 MiB bounded chunks with a 60-second per-chunk bound, while renewal is
   throttled to at most every 30 seconds, then receive local hash verification before atomic publication; one export
   may be as large as the manifest's bounded cap (at most 2 GiB). Only idempotent reads receive their bounded
   three-attempt retry; an uncertain mutating kernel operation is never retried. An uncertain
   kernel RPC is not resubmitted to a possibly busy kernel; reconnect through state/files and retry only after
   independent backend evidence says the kernel is idle.
4. On lease expiry, the runtime checkpoints and releases via the Colab tunnel GET/XSRF-POST endpoint flow, not
   `google.colab.runtime.unassign()`, which has no notebook frontend in a detached subprocess. The Mac supplies a
   private, short-lived access token with exactly `colaboratory` and `userinfo.email` scopes; no refresh token,
   client secret, or token URI reaches the runtime. The credential is account-scoped rather than endpoint-bound and
   remains visible to trusted same-user runtime code; redirects are denied and naive expiry is normalized as UTC.
   Three release attempts have a 130-second total bound, and the runtime requires lease + grace + 10 seconds + that
   bound before it starts the child. Backend stop has up to three 75-second attempts and must prove the exact endpoint
   disappeared from `sessions`; a cleanup exit or CLI success is not release proof.
5. Never trust a CLI exit code alone: check the job's receipt (status, hashes, counters), locally verified exports,
   spend ledger, and endpoint disappearance.
6. Import results into the repo doc, commit + push to main, report to the owner (Pacific time).
The first use of the supervisor requires its <=4-unit acceptance test to pass.

The L4 acceptance arm simulates a Mac disconnect by suppressing renewal. It passes only if the exact assignment
disappears from backend sessions within 300 seconds after lease expiry, before broker fallback cleanup. A missing artifact, credential
scope/expiry failure, unaccounted assignment, lingering VM, cleanup error, or timeout ambiguity is `INCOMPLETE`.

Everything below documents the **legacy Q1-only** notebook/Drive path, still needed only to run and babysit the
already-authorized Q1 work. It must not be used to create or route a new Colab job.

Prereq: `colab` skill (runtime + Drive mounted). Code on the runtime lives at `/content/t16-snake-dqn`
(pinned checkout made by `notebooks/colab_throughput_smoke.ipynb`), venv `/content/t16-colab-venv`, inputs
`/content/t16-colab-extracted/t16-colab-inputs` (B8 init, demo, frozen-pool checkpoints; bundle sha
229be294...). Training env: `SNAKE_CKPT_DIR=$IN/checkpoints PYTHONPATH=/content/t16-snake-dqn`.

## Notebooks

- `notebooks/colab_throughput_smoke.ipynb`: setup + P0 learner bench + a smoke matrix (`MATRIX` = smoke02 /
  smoke03 / n3). Writes `$DRIVE/t16_colab/<RUN_NAME>/colab_throughput_results.json` (status preflight ->
  completed/incomplete/failed). P0 failures are non-fatal (receipt `p0_receipt.json`).
- `notebooks/colab_long_run.ipynb`: long runs with async full-state snapshots (`--full-state-every`,
  local + Drive dirs, keep last 2, sha256 manifests) and auto-resume from Drive; recipe block must be filled
  from a committed pre-registration. Full replay snapshots briefly stall the learner (multi-GB copy); use
  `--full-state-replay none` unless exact resume is required.

## Launch pattern (resumable)

1. Commit a launcher under `research/<task>/run_*_colab.sh` that loops over units (seeds/rows), skips units whose
   result already exists on Drive, trains to local `/content/...`, then copies to `/content/drive/MyDrive/...`,
   and appends start/exit lines to a Drive `run.log`. Example: `research/t23_r0_screen_20261017/run_t23_colab.sh`.
2. Pre-registration first for any learning experiment (commit before data).
3. On Colab, one-line cell: fetch by commit sha from raw.githubusercontent.com and start with nohup (see colab).
4. After a disconnect: new runtime -> run the notebook's setup (or the smoke notebook once) -> relaunch the same
   script; finished units are skipped, long runs resume from the newest Drive snapshot.

## Legacy Q1 Drive queue for dedicated runtimes

For the existing Q1 lane only, use `research/colab_queue/` when its dedicated Colab runtimes pull independently schedulable,
pinned jobs. It replaces the old concurrent `CLAIMED`-marker pattern below: Drive can duplicate simultaneous
names (`x` and `x (1)`), so a shared marker is not a safe cross-runtime claim. Keep the manual resumable launcher
pattern for a single runtime or an already partitioned legacy job; do not point two runtimes at the same manual
seed list.

On the Mac, publish a fully specified job before opening workers. `--drive-root` is the Drive *base*, while queue
metadata is below `$DRIVE/queue` and each job `output_dir` is below `$DRIVE`:

```bash
DRIVE='/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab'
q() { ./venv/bin/python research/colab_queue/q.py --drive-root "$DRIVE" "$@"; }
q enqueue path/to/job.json
q status
q watch --interval-seconds 10
```

The JSON needs a full job `commit`, repository-relative `script`, string `env`, `requirements` (`gpu`,
`min_vcpu`, `est_minutes`), `depends_on`, Drive-base-relative job-private `output_dir`, and integer `priority`.
`gpu: any` requires a real non-CPU GPU; `gpu: cpu` uses only CPU runtimes. `enqueue` pre-creates shared queue
directories and output directories; job scripts must publish only inside `QUEUE_OUTPUT_DIR`.

Start one of `notebooks/colab_queue_worker.ipynb`, `colab_queue_worker_a.ipynb`,
`colab_queue_worker_b.ipynb`, or `colab_queue_worker_c.ipynb` per dedicated runtime. Set the notebook's full
`QUEUE_REF` to the published queue implementation commit. It is distinct from the job's full `commit`: the queue
pin fetches `worker.sh`, while the job pin selects setup and the launcher. The worker waits 120 seconds for claim
visibility, uses the lexicographically smallest `(worker_id, token)` claim, writes 15-second heartbeats, re-runs
setup when the job commit changes, and records GPU, commit, timing, log, and terminal receipt.

Drive coordination is bounded visibility, not exactly once. A stale heartbeat (300 seconds by default) is an
operator warning, never an automatic retry. First cancel a job, then verify the owning runtime and process stopped
before requeueing with the required acknowledgement:

```bash
q cancel JOB_ID --reason 'operator stop'
q requeue JOB_ID --stale-owner-stopped --reason 'runtime verified stopped'
```

Workers release only dedicated runtimes after 10 idle minutes with no locally eligible job. They flush Drive before
calling `runtime.unassign()` and report a failed release for manual owner action. A queue containing only blocked,
claimed, cancelled, CPU, or other-GPU work is locally empty for an incompatible worker, so watch the Mac queue
before assuming a runtime will stay allocated. See `research/colab_queue/README.md` for the schema, failure model,
T23 illustrative-only examples, and local test boundary.

## Monitoring from the Mac

Watch Drive, not the browser (screenshots lag; the pane may be hidden):
```bash
D="$HOME/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/<run>"
tail -5 "$D/run.log"; tail -1 "$D/<unit>/log.jsonl"
```
Gotcha: a process-pattern match (pgrep with -f) also matches its OWN shell command line - a follower looping
`while pgrep -f "train_m3c.py --seed 7103"` never ended (2026-10-17). Bracket one character
(`"train_m3c[.]py --seed 7103"`) or wait on a PID / output file. New concurrent work uses the CLI supervisor, never
the Drive queue or `CLAIMED` markers. Existing Q1 manual launchers remain safe only when units are partitioned before
launch and each runtime has a disjoint seed list.

Use a background until-loop on the final marker (status completed/failed/incomplete, `all done` line) rather
than sleeps. The results JSON exists from preflight on - don't treat its existence as completion.
Key telemetry: `log.jsonl` (transitions, updates, rate, learn_s, env_s), `telemetry.jsonl` (learner_wait_s,
queue_depth, frozen_forward_s, policy_lag_mean/p95, featurizer_s, v8_s). Learner-bound = queues full, small
learner_wait; actor-bound = queues ~0, large learner_wait.

## Throughput facts (2026-10-17, A100-80)

- Mac baselines: all-six self-play 5,459 tr/s; N=3 static pool 5,214 tr/s / 20.4 upd/s. 4x bar = 84.4 upd/s.
- A100 self-play a4 15.9k tr/s / 62 upd/s (GPU-bound). Isolated learner fp32 bs2 82.7 upd/s, bf16 118.
- A100 N=3 is actor-bound by the frozen pool's CPU work: a4 5.9k, a6 7.7k tr/s (fix in progress, t22).
- bs4 gives no throughput gain (cost linear); bf16 is a numerics change (measure-only unless screened).
Records: `docs/research/t16_colab_smoke02_results_2026-10-17.md`.

## After results

Import into the repo doc, commit + push to main, report to the owner; release the runtime if nothing is queued.
