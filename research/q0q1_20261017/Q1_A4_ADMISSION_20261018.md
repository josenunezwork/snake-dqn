# Q1 Amendment A4 admission — 2026-10-18

Outcome: **ADMITTED_FOR_DISPATCH; 120 manifests validated; NOT ENQUEUED.**
This is a capacity/admission record, not Q1 scientific evidence or a release.

A4 was committed as `cc061149` before the new capacity measurements. The exact
owner quote is appended to PREREGISTRATION.md and bound by the new
OWNER_ADMISSION_20261018_A4.json. The pre-A4 registration prefix and older owner
packets are unchanged. A4 raises only the combined GPU-hour cap to 37 (29.6
usable); CPU stays 40 (32 usable). Seeds, recipe, banks, thresholds, Q1 allocation,
and the A3 inclusive accounting formula stay unchanged.

The final generator ran from clean published source
`f478257e48377f99e4da69b0c332b79d434af765`. The NEW Drive packet is:

`q0q1-20261017/q1/jobs_approved_a4_r2_final`

The earlier A3 refusal and all historical packets remain intact. The new
historical-parity bridge checks the exact A3 source/registration/validator/tool
hashes, unchanged numerical dependencies, and the real old-serial/current-serial/
current-pool row identity. Historical receipts retain their original hashes.
Every queued job still generates fresh parity against the new Mac reference.

## Forecast and measurement

| Resource | Combined forecast including Q0 | Usable limit | Nominal cap |
|---|---:|---:|---:|
| GPU-hours | 29.468238339864364 | 29.6 | 37 |
| CPU-hours | 26.38860158845773 | 32 | 40 |
| Units | 113.93965340722725 | 168.63035524611342 | 203.63035524611342 |

Q0 units are 68.63035524611341; Q0 CPU-hours are 0.06305555555555556 (cap 8).
Q1 training uses 12.243442220561551 GPU-hours plus 3.8298203517698304
setup/export/attempt-overhead hours: **16.07326257233138 hours**, costing
32.14652514466276 units (training cap 60). Q1 evaluation uses
9.457481279614042 pooled CPU-hours plus **16.86806475328813 serial overhead
hours**: **26.32554603290217 CPU-hours**, costing 13.162773016451087 units.

The Mac real entrypoint evaluated the four RESERVED parity worlds, repeated
twice for the pool (eight real H5000 episodes, no Q1 scored-bank look).
Three single-thread spawned workers took **22.020308082996053 seconds** inclusive
of startup/merge/shutdown. Serial per-world times were
3.9875397499999963, 5.474307541997405, 7.947725250007352, and
7.626636583008803 seconds (mean 6.259052281253394). Raw canonical rows are
byte-identical across the frozen A3 serial loop, current serial, and current
spawn pool. The measured speedup is 2.2739199679359947; normalized worker
efficiency is **0.7579733226453316**, below the 0.8 ceiling.

Forecast per-world time is max(measured L4 56.45208326124998 seconds,
3 times measured Mac serial mean). Thus the historical 56.45-second floor
continues to dominate; the tiny parity fixture never lowers the historical
per-world reserve. Scored CPU-hours = 3200 * per-world seconds /
(3600 * 7 * efficiency). Seven is a forecast floor for at least eight vCPUs;
actual workers = max(1, min(cpu_count - 1, 10)): seven on eight vCPUs, ten on twelve.
The four parity episodes/job and 381.4419980733728 seconds/job staging/export
reserve remain SERIAL: 100 * (4 * per-world seconds + overhead) / 3600.
No pool speedup or pool-efficiency penalty is applied to serial work.

The three-core Mac probe does not establish seven-core Colab scaling. The 3x
per-core factor, historical L4 floor, measured efficiency, and 20% envelope
reserve are conservative planning inputs; real Colab throughput remains to be
observed. Input pricing remains the bound A3 receipt: L4 2 units/hour owner
planning rate, CPU 0.5 conservative units/hour, A100 5.3 observed units/hour.
No fresh billing rate or live cloud state was measured here.

## Runtime plan and stopping conditions

Keep the qualified **L4 with at least 12 vCPUs** for all 20 training jobs. Its
additional projected busy time is 16.07326257233138 hours. Per-job queue estimates
are 30 or 68 minutes; outer timeouts are 98 or 136 minutes, at least twice the
estimate. The internal 5100-second training cap and fixed recipe are unchanged.

CPU jobs require **gpu: cpu and at least eight vCPUs**. A100/L4 workers cannot
claim them. There are 100 jobs of 32 scored worlds each, with 16-minute queue
estimates and 32-minute timeouts. Aggregate projected CPU time is 26.325546 hours;
two continuously busy CPU runtimes would need about 13.16 hours each, before
rounding and dependency idle time.

The owner reports an A100 and an L4 notebook worker alive. Start one CPU worker
in the third available slot. Release the A100 before creating a second CPU
worker; run no more than L4 + CPU + CPU concurrently. Stagger the second CPU
worker until there is sustained eligible backlog (approximately hour 3 is a
planning example), aiming to finish near hour 17: one CPU allocation up to
17 hours and the staggered allocation up to 14 hours totals **31 CPU-hours**.
Including Q0, this allocation plan is 31.063056 hours, below 32 usable. This is
a planning bound, not a promised makespan. Shorten/release idle CPU allocations;
monitor actual aggregate hours rather than treating idle notebooks as free.
Use worker notebooks in order with their blocking keep-alive cell; replacement
VMs require fresh setup and Drive authorization. No runtime was changed here.

GPU usable headroom is only **0.131761660135636 hours = 7.90569960813816 minutes**.
Do not retain an idle A100 while Q1 runs. With the bound Q0 snapshot, additional
GPU allowance to the usable limit is 16.205024232467015 hours. Charge any actual
setup, copies, idle allocation, failed attempts and cleanup; stop at the first
applicable cap. The JSON admission forecast does not account for any unrecorded
GPU allocation since that snapshot and is not evidence of current remaining
credit. Parallelism reduces elapsed evaluation runtime, not the episode count.

Each completed world is written locally, exported to the queue-precreated
job-private evaluation-partials directory, and included in the final archive.
Resume imports require an explicitly reviewed Q1_RESUME_ARCHIVE and matching
Q1_RESUME_ARCHIVE_SHA256, a fresh job id/output directory, and unchanged scientific
identity. **Partial reuse is same-runtime only**: changed host/environment/parity
identity is refused. No automatic retry, rerun, enqueue or seed replacement was
performed or authorized by this record.

## Verification and evidence

All **339 scoped Q0/Q1/queue tests passed** (23.82 s), including the fixed mock
serialization test. The process-pool byte proof was executed through the real
q1_evaluate CLI, separately from synthetic unit tests. Black, isort, flake8,
repository make lint, and git diff --check passed. The independent review found
and verified fixes for resumed-manifest truncation and retained-unit export.

The additional whole-repository non-slow run was **not green**: 7678 passed,
26 skipped, 22 failed, 5 warnings (398.64 s), including two native worker crashes.
One failure was the mocked serialization test, corrected to workers=1 and then
passed in the 339-test scoped run. The remaining failures and diagnosis are
recorded in Q1_A4_TEST_LIMITATIONS_20261018.md. No unrelated trainer, serving,
frontend, or global compiler changes were made to chase that broader baseline.

From this worktree, exact final scoped verification was:

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$PY" -m pytest -q tests/test_q1_*.py tests/test_q0*.py tests/test_make_q1_jobs.py \
  tests/test_requalify_q0b.py tests/test_colab_queue*.py
make PY="$PY" lint
git diff --check
```

Exact generation argv is in the packet's generator-invocation.json; its real
CLI exited 0. Read-only validation used research.colab_queue.core.validate_job
on all 120 manifests, checked both external Q0 dependencies are currently done,
verified every bound evidence digest, no ID/output collisions, portable paths,
timeout >= 2x estimate, and that the pinned commit exists on origin/main.
It did not call Queue.enqueue. There were zero pre-existing Q1 jobs in the queue.

Mac evidence root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a4-20261018-r2`.
The earlier r1 capacity probe is preserved; r2 binds the final evaluator bytes.
Logs: r2/scoped-tests.log; r1/test-fast.log; r2/generator.log.

| Packet artifact | SHA-256 |
|---|---|
| study-plan.json | 66652214a7414f8ef6bcfd620c67199e1ce7d4981aca5024fd32d02807dd4c53 |
| admission.json | 9454135e463a47899b31c2b86412ac1ac4cb3ea897163f3616fe222134aed976 |
| a4-compatibility.json | c3c47554328bb73d523983ee0420c5b99db04aea2eaf33b0a27df18762ab50e1 |
| manifest-validation.json | d449c57bfedc5af76ff3ede3f87b2984651eb8ceae73da2eb375cd80b004ec65 |
| artifact-manifest.json | 346d237c21942431328259ee4ecebf8c7e1c18f60ef9a5db9a43cdbf21550938 |
| evidence/cpu-timing-receipt.json | 58914936bf5053243b17bc04309c7da60f4cadb834cb51e64a78207fd8d35469 |

## Operator enqueue commands — prepared, not executed

First arrange the runtime plan above. The following enqueues the exact reviewed
20 training, 20 shared B8, and 80 candidate-evaluation manifests in that order.
Queue dependencies still control when they execute. Files are already staged
on the Mac's Drive mount; synchronization is asynchronous, and runtime-side
hash checks remain mandatory.

```bash
set -euo pipefail
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
Q=/Users/josenunez/Projects/ml/snake-dqn-q1-a4/research/colab_queue/q.py
DRIVE='/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab'
JOBS="$DRIVE/q0q1-20261017/q1/jobs_approved_a4_r2_final"
for spec in "$JOBS"/q1-train-*.json "$JOBS"/q1-eval-b8-*.json "$JOBS"/q1-eval-[abcd]-*.json; do
  "$PY" "$Q" --drive-root "$DRIVE" enqueue "$spec"
done
```

No tmux session (including git-sync), released checkpoint, serving pin, live
notebook, existing queue receipt, or prior approved manifest was changed.
Only the explicitly authorized A4 append changed the registration; historical
A3 packets remain immutable.
