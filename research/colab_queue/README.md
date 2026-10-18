# Colab Drive queue

`colab_queue` lets dedicated owner-owned Colab runtimes take compatible, pinned
jobs from one Drive-backed queue, then release themselves when they have no
eligible work. It is an operations tool, not an experiment authorization:
commit the launcher and any required pre-registration before enqueueing a
learning job.

One Mac is the control plane; its local writer lock does not coordinate multiple publisher machines. It publishes jobs and creates every shared Drive
directory. A Colab worker only writes its own uniquely named claim, heartbeat,
log, and receipt files.

## Prerequisites and boundaries

Use the repository venv on the Mac and this Drive root:

```bash
DRIVE='/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab'
q() { ./venv/bin/python research/colab_queue/q.py --drive-root "$DRIVE" "$@"; }
```

`q.py` defaults to that same Drive root if it is run on this Mac. The queue
metadata always lives below `$DRIVE/queue`; a job's `output_dir` is relative to
`$DRIVE`, never to the repository or the `queue` directory. The Colab mount of
the same root is `/content/drive/MyDrive/t16_colab`.

Run a queue worker only in a dedicated runtime. It may unassign that runtime
after it is idle, so do not use it alongside notebook work or another owner
workload. Queue workers do not launch pods, inspect live Colab capacity, or
authorize research. They also do not touch existing tmux jobs.

## Job JSON

Pass one JSON object to `enqueue`. Every field below is required unless marked
optional. `commit` is an immutable, full 40-character Git SHA, not a branch,
tag, or abbreviated SHA.

| Field | Meaning |
| --- | --- |
| `id` | Unique queue identifier containing letters, digits, `.`, `_`, or `-`. |
| `commit` | Full repository commit whose setup and job script will run. |
| `script` | Relative path to a Bash launcher inside that checkout (run with `bash`). |
| `env` | Object of string environment-variable names and string values. |
| `requirements` | Object with `gpu`, positive integer `min_vcpu`, and positive integer `est_minutes`. |
| `depends_on` | IDs that need current-generation `done` receipts. |
| `output_dir` | Relative, job-private output directory below the Drive root. |
| `priority` | Integer; higher values run first, then IDs sort ascending. |
| `timeout_minutes` | Optional positive integer job limit; default: twice `est_minutes`. |
| `output_subdirs` | Optional relative directories created below `output_dir`. |

`requirements.gpu` is one of `any`, `A100`, `L4`, or `cpu`. `any` requires an
actual non-CPU GPU; it does not select a CPU runtime. `cpu` selects only a CPU
runtime, preserving GPU capacity for GPU jobs. The detected GPU and CPU affinity
must also satisfy `min_vcpu`.

```json
{
  "id": "example-train-7106",
  "commit": "0123456789abcdef0123456789abcdef01234567",
  "script": "research/example/run_colab.sh",
  "env": {"EXAMPLE_SEED": "7106"},
  "requirements": {"gpu": "A100", "min_vcpu": 12, "est_minutes": 90},
  "depends_on": [],
  "output_dir": "example/seed-7106",
  "priority": 100
}
```

Replace the illustrative SHA, script, and values with a real published job.
`enqueue` validates the schema, creates `queue/`, its shared subdirectories,
per-job receipt directories, and the output directory. It rejects output paths
outside the Drive root, inside `queue/`, or overlapping another job's output.
Job scripts must treat `QUEUE_OUTPUT_DIR` as already existing and job-private.
The worker also supplies `QUEUE_JOB_ID`, `QUEUE_GENERATION`, `QUEUE_WORKER_ID`,
`QUEUE_CLAIM_TOKEN`, and `QUEUE_DRIVE_ROOT`. These and its `PYTHONPATH`/venv-first
`PATH` override job env values. Use `output_subdirs` for required shared output
directories; create scratch directories only on local runtime storage.
A requeue reuses the output directory: inspect partial artifacts and make the
launcher resume safely or use a new job ID/output directory for a clean run.

## Drive layout and claim protocol

The Mac-side enqueue command creates this layout before a runtime can claim
work:

```text
t16_colab/
  queue/
    pending/       # job specifications
    claimed/       # one directory per job; unique claim files
    done/          # immutable successful receipts
    failed/        # immutable failed receipts
    heartbeats/    # current worker records
    control/       # cancellation and generation fences
    locks/         # reserved queue layout; the Mac writer lock itself is local
  <output_dir>/    # job logs and artifacts, created by enqueue
```

Google Drive can turn simultaneous creation of one name into `name` and
`name (1)`. The queue never treats a directory creation or a filename as an
atomic lock. It scans the union of duplicate directory aliases, rejects
non-identical duplicate records, and each contender writes a claim named with
its worker ID and a fresh token. Contenders wait 120 seconds by default for
Drive visibility; the winner is the lexicographically smallest
`(worker_id, token)` among claims in the generation. A cached late contender is refused once a visible owner has started or produced a
terminal receipt. A running worker keeps checking that it remains the winner and stops its owned process group when it
is cancelled, fenced, or loses arbitration.

This is deliberately a bounded-visibility protocol, not an exactly-once
system. If Drive hides a competing claim beyond the settle window or partitions
a runtime, two scripts can begin. Scripts must keep outputs job-private,
validate their published artifacts, and be safe to inspect after a failed
attempt. The queue never automatically reclaims a claim, even when a heartbeat
is stale.

## Mac control plane

All commands print JSON. The usual sequence is:

```bash
# Publish the job and create its Drive directories.
q enqueue path/to/authorized-job.json

# Inspect states, claims, receipts, and heartbeat ages. A heartbeat is stale
# after 300 seconds by default.
q list
q status example-train-7106
q watch --interval-seconds 10
```

The checked-in [T23 train example](examples/t23_train_seed_7101.json) and
[evaluation-shard example](examples/t23_eval_shard_0_of_3.json) are illustrative
only: T23 R0 is closed **KILL**, and their
adapters fail unless an operator explicitly enables their local example guard.
They do not authorize a run or change any governance decision.

To stop a job, first cancel it. Cancellation fences its current generation and
causes an observing worker to stop its owned process group:

```bash
q cancel example-train-7106 --reason 'operator stop'
```

For a failed, cancelled, or stale claimed job, inspect `status`, verify that
the owning runtime and its job process have actually stopped, then start a new
generation. Requeue refuses a claimed job whose winning owner has a fresh matching heartbeat. The
acknowledgement is intentionally explicit:

```bash
q requeue example-train-7106 --stale-owner-stopped --reason 'runtime verified stopped'
```

Do not use a stale heartbeat alone as proof that a process is dead. Requeue
increments the generation and attempt, retains old claims and receipts for
audit, and fences them; it does not make a live stale owner safe.

## Starting runtimes

Open a different notebook file for each concurrent runtime:

- `notebooks/colab_queue_worker.ipynb`
- `notebooks/colab_queue_worker_a.ipynb`
- `notebooks/colab_queue_worker_b.ipynb`
- `notebooks/colab_queue_worker_c.ipynb`

Each notebook has exactly four code cells, executable in order with **Runtime >
Run all**: configure idle release and mount Drive, fetch `worker.sh` from a full
`QUEUE_REF`, start it with `nohup` in the background, then block in the final
**keep-alive** cell while that worker is alive. Keep the last cell executing so
background work is accompanied by an active notebook cell. It prints a UTC
timestamp, worker alive yes/no, the last `worker.log` line (up to an 8 KiB tail),
and the current queue job from that worker's heartbeat immediately and every
600 seconds. It wakes immediately when the worker exits, prints a final status,
and ends; missing logs or heartbeats are reported without stopping monitoring.
This does not guarantee that Colab will retain the runtime.

The first cell defaults to `KEEP_RUNTIME = True`, setting
`os.environ['IDLE_MINUTES'] = '100000'` before launch. Set it to `False` before
Run all to remove the override and restore the worker's default 10-minute idle
release. This flag controls queue-worker idle release; the final cell keeps
executing in either mode and ends when the worker exits. Configure the flag
before starting a new worker; it does not change an already running process.

The checked-in `QUEUE_REF` pins the published queue implementation. Change it
only to another full published implementation commit when upgrading. That pin is deliberately independent from a job's `commit`: the
former controls the queue worker code, while the latter controls the repository
checkout, `setup_runtime.sh`, and job script.

On every job whose commit differs from the prepared one, the worker fetches and
runs `research/t16_colab/setup_runtime.sh` from that job commit, verifies the
checkout is at the requested commit with no tracked changes, then starts the
job with `nohup`. Its Drive output directory receives the job log. An immutable receipt under
`queue/done/` or `queue/failed/` records worker, GPU, commit, timing, and exit
information. Pending specs remain as the job registry; use `q list` to see state
rather than counting files in `pending/`. Setup has a default 1,800-second limit. Jobs use their
explicit `timeout_minutes` or twice their estimated minutes. The actual
assigned GPU and CPU affinity are detected on the runtime, rather than trusted
from the notebook's selected type.

Workers write heartbeats every 15 seconds by default. A worker with no locally
eligible job for 10 minutes by default releases its dedicated runtime. "No
eligible" includes jobs blocked by dependencies, claims, incompatible GPU or
vCPU requirements, or cancellation, so an A100 worker can release while only
CPU, L4, or blocked jobs remain. Before release it invokes the runtime kernel's
Python to call `drive.flush_and_unmount(timeout_ms=120000)` and then
`runtime.unassign()`. The worker retries a failed release three times with
15-second delays; if all attempts fail, it reports that the owner must inspect
the local worker log and manually disconnect the dedicated runtime. The
[Colab Drive implementation](https://github.com/googlecolab/colabtools/blob/main/google/colab/drive.py)
defines `flush_and_unmount` as flushing outstanding Drive writes, and
[Colab's runtime implementation](https://github.com/googlecolab/colabtools/blob/main/google/colab/runtime.py)
defines `unassign()` as requesting unassignment of the current runtime. Its
result is not covered by local tests.

## Verification

Run the local queue contract tests from the repository root:

```bash
./venv/bin/python -m pytest -q tests/test_colab_queue.py tests/test_colab_queue_safety.py tests/test_colab_queue_worker.py
# If the project venv is unavailable, use a Python with the dev dependencies,
# including the repository-required pytest 9 or newer:
python3 -m pytest -q tests/test_colab_queue.py tests/test_colab_queue_safety.py tests/test_colab_queue_worker.py
```

They simulate three workers, duplicate-name Drive directories, dependency and
capability selection, fencing, heartbeat staleness, worker process ownership,
and the four notebook bootstrap contracts. They do not mount Drive, allocate a
Colab runtime, run a T23 job, or prove Drive propagation or hosted-runtime
release behavior.
