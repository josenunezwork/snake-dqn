# T23 R0 evaluator

This implements the R0 decision rules in `PREREGISTRATION_T23.md` and sections 4
and 6 of `docs/research/advice/2026-10-17_h100_preruns_rl.md`. It is a development
screen for long-run planning. A GO is neither a strict pass nor a release.

Execution note, 2026-10-17: the owner requested CPU evaluation on Colab, with a
Mac parity reference. This changes the preregistration's Mac execution location;
its checkpoint selection, world counts, statistics and thresholds remain fixed.
The original preregistration is preserved byte-for-byte. Strict qualification
and serving qualification still run on the Mac.

## Inputs and bank

Training seeds are 7101–7105. Each seed has 64 scripted, 32 frozen and 32 mixed
fresh worlds at H5000, distinct across training seeds and mixes. Candidate and
B8 share the exact ordered opponent roster and world seed; only slot-zero hero
changes. Both use CPU rowwise inference, advisory masking, and no veto. The
hero's death is terminal and opponents retain the inherited evaluation respawn
semantics. Mass is summed after each frame, with dead frames zero, then divided
by 5000. Alive-frame fraction and no-death episode survival are separate metrics.

`WORLD_BANK.json` fixes the initial bank, one reserved extension bank and four
separate parity worlds. Its T23 allocation is new; the roster definitions come
from the same strict-design materializer used by B8 release evaluation.
`SEED_AUDIT.json` records the collision audit and evidence hashes. No T7 gameplay
producer is invoked. The audit covers recorded banks and visible manifests;
it cannot establish absence from an unrecorded experiment.

Released B8 is pinned to
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
The candidate files are `seed-<s>/ckpt_8000k.pth`, and descriptively
`ckpt_2000k.pth` and `ckpt_4000k.pth`. Actual checkpoint transition metadata and
manifest hashes are checked; the trainer can cross a milestone on a message
boundary. Only 8M decides. The evaluator reads training evidence and never
launches, resumes or modifies training.

## Three fresh Colab runtimes

Use `notebooks/colab_worker.ipynb` in **three new runtimes**, leaving existing
training runtimes alone. CPU runtimes are sufficient: `setup_runtime.sh` now
selects a CPU Torch wheel when no GPU is present. Evaluation hides CUDA even
if a GPU runtime is selected.

Set `COMMIT` to the full published evaluator commit, and set `SCRIPTS_REF` to
the same commit. Keep these values identical across all runtimes. Set:

```python
JOB = 'research/t23_r0_screen_20261017/run_t23_eval_colab.sh'
```

Use one of these values in each runtime:

```python
JOB_ENV = 'T23_SHARDS=0/3 T23_WORKERS=2'  # runtime 1
JOB_ENV = 'T23_SHARDS=1/3 T23_WORKERS=2'  # runtime 2
JOB_ENV = 'T23_SHARDS=2/3 T23_WORKERS=2'  # runtime 3
```

The notebook mounts Drive, prepares the pinned runtime, then starts this job.
Wait until all five training seeds have finished and their evidence has synced
to Drive. Inputs are read from
`/content/drive/MyDrive/t16_colab/t23-r0/seed-<s>/`; evaluation outputs have their
own namespace. `T23_SHARDS` also accepts comma-separated selectors for a serial
set, for example `0/3,1/3`. Shards are zero-based. Two CPU workers is a conservative
default for ordinary Colab CPU allocations; each worker uses one math thread.

Every runtime first reruns the four fixed H5000 B8 parity worlds and compares
against `parity/mac.json.gz`. Any disagreement stops that job before selection
worlds run. The comparison records the first differing action and available
float32 Q margins/ULPs; a tie does not excuse a mismatch. This follows the
diagnostic requirement in
`docs/research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md`.
The referenced `mac-pod-float-ties.md` memory and diagnostic branch document are
not present in this checkout. Colab parity remains unverified until these
runtime receipts exist; a Mac baseline alone is not cross-platform evidence.

Relaunch the same shard selector after an interruption. Completed outputs are
reused only after their identities and hashes validate. Never change a shard's
checkpoint bytes, world allocation or source during a run. Do not run the same
shard concurrently on multiple runtimes. The notebook's last `train_m3c` process
count belongs to its old training example; read `/content/job.out` and the T23
evaluation logs for this job's status.

## Local invocation

Use the project's existing venv from the main checkout if this worktree has no
`venv`. The evaluator defaults to the Mac Drive root
`/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/t23-r0`.
Set `SNAKE_CKPT_DIR` to the pinned frozen checkpoint directory before importing
the evaluator if it differs from the main checkout's `saved_snakes`.

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.t23_r0_screen_20261017.evaluate \
  --shard 0/3 --workers 2 --include-descriptive \
  --b8-checkpoint /Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s3/final.pth
```

## Analysis and extension

The analysis script checks evidence before making a scientific decision.
Missing identities, hash mismatches, incomplete counts, cap expiry and unclean
exits produce INVALID/INCOMPLETE. For valid 8M data it averages the three mix
means equally per training seed, then computes G and one-sided 90% Student-t
limits across the five seed effects (df=4). It reports every seed, paired-world
variances, hierarchical paired-bootstrap sensitivity, and descriptive 2M/4M
results. The per-seed scripted −0.15 guard refers to alive-frame fraction;
the GO scripted mass point guard is a different quantity.

KILL precedes GO. If neither applies, the first look is EXTEND. Freeze the same
checkpoint hashes, evaluate the reserved extension exactly once, and recompute
using both banks. If neither rule then applies, the verdict is INCONCLUSIVE.
No new training, third look, checkpoint swap or replacement seed is permitted.
The unsupported low-boost floor is reported as unevaluable, as preregistered.

The JSON verdict and Markdown report are analysis artifacts, not completion
proof for a still-running shard. Commands and the verification receipt below
describe only tests and the Mac parity sample performed during implementation.

After all three shards finish, run in one prepared Colab runtime:

```bash
cd /content/t16-snake-dqn
/content/t16-colab-venv/bin/python -m research.t23_r0_screen_20261017.analyze \
  --input /content/drive/MyDrive/t16_colab/t23-r0/evaluation \
  --out /content/drive/MyDrive/t16_colab/t23-r0/evaluation/analysis
```

This writes `t23_verdict.json` and `t23_verdict.md`. The same command works on
the Mac with its venv and Drive paths. Incomplete shards produce an invalid
report; their absence is never interpreted as a successful or failed arm.
If a verified training tripwire stopped a seed before 8M, the overall report
retains INVALID/INCOMPLETE precedence and separately records the training KILL
disposition. A completed, otherwise valid run with a recorded tripwire receives
the registered KILL verdict.

For an EXTEND only, preserve the first report as `evaluation/t23_initial_verdict.json`.
On each of the same pinned runtimes run the evaluator directly with its original
shard selector, adding:

```bash
--phase extension \
--prior-verdict /content/drive/MyDrive/t16_colab/t23-r0/evaluation/t23_initial_verdict.json
```

Keep `--root`, `--bank`, `--b8-checkpoint`, `--shard` and `--workers` as above.
Omit `--include-descriptive` for this second look: the original complete 2M/4M
results remain descriptive, and only 8M needs the additional worlds. The
evaluator binds a copy of the original verdict and checkpoint hashes in
`evaluation/extension_authorization.json`. Analyze the same evaluation root
again to combine both phases. Changed initial receipts or checkpoint bytes,
an extension without the first EXTEND verdict, and any third look are refused.
