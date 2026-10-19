# Q1 A5 final admission handoff — 2026-10-18

Outcome: **REFUSED; zero manifests; NOT ENQUEUED; L4 released.**

The actual Mac generator ran from clean published commit
`f5b2716571f017decc40aba11842bf0c4c829270` in
`/Users/josenunez/Projects/ml/snake-dqn-q1-a5`, branch
`t31-q1-a5-20261018`. Both origin/main and the remote branch contain that source.
A5 and the accounting correction were committed before this generator invocation.
The generated record is capacity evidence, not Q1 scientific data or a release.

The complete per-runtime table, evidence limitations and unchanged unit/CPU caps
are in [the A4 sibling correction](Q1_A4_ALLOCATION_CORRECTION_20261018.md).
Missing assignment starts use the disclosed Oct17 18:00 PDT bound; the unresolved
earlier L4 may be counted twice. These are conservative policy bounds, not observed
charges. The requested prospective L4 assumption is dispatch Oct18 20:00 PDT,
charged through 20:30 PDT. The actual earlier release is not used to seek admission.

| Resource | Complete upper-bound forecast | Usable | Nominal |
| --- | ---: | ---: | ---: |
| GPU-hours | 105.62604035010915 | 36 | 45 |
| CPU-hours | 51.3027682551244 | 32 | 40 |
| Units | 362.55279816111386 | 168.63035524611342 | 203.63035524611342 |

All three limits refuse admission. The frozen Q0 68.63035524611341-unit and
8 CPU-hour subcaps also refuse. Q1 training itself remains 32.14652514466276 units,
below its unchanged 60-unit limit; this does not cure the combined refusal.
No further raise, reserve borrowing, scientific change, automatic retry or seed
replacement is authorized. No 120-manifest validation claim is made for this batch:
the generator correctly stopped before creating manifests. The old A4 packet is
preserved and must not be enqueued under A5.

## Actual generator and evidence

The NEW output directory is:

`q0q1-20261017/q1/jobs_a5_refused_r1`

It contains staged evidence, study-plan.json and admission.json. The real CLI
returned **2** and the terminal admission status is **REFUSED**. Validation
confirmed zero q1-*.json manifests, an empty Q1 queue, exact fixed caps, complete
historical attempt coverage and unchanged pre-A5 registration bytes.

The following is the exact archived invocation from that clean source commit;
it is evidence of the completed invocation, not a request to repeat it. The
existing output directory is create-only and its dispatch assumption expires.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-q1-a5
env SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.q0q1_20261017.make_q1_jobs --commit f5b2716571f017decc40aba11842bf0c4c829270 --inventory-receipt '/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/q0q1-20261017/admission/inventory-receipt.json' --q0-mass-archive '/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/q0q1-20261017/q0/audit-t23-mass-a100r5/result.tar.gz' --q0-init-archive '/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/q0q1-20261017/q0/audit-b8-init-a100r6/result.tar.gz' --q0b-l4-receipt '/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/q0q1-20261017/q0b/capacity-l4-20261018r6-a3/result.tar.gz' --q0b-a100-receipt '/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/q0q1-20261017/q0b/capacity-a100-20261018r6-a3/result.tar.gz' --historical-mac-parity-reference '/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/q0q1-20261017/admission/mac-parity-reference.json' --current-mac-parity-reference /Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a4-20261018-r2/mac-current-reference.json --cpu-timing-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a4-20261018-r2/cpu-timing.json --q0-accounting /Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a5-20261018/allocation-accounting-v2.json --q0b-l4-job-id q0b-capacity-l4-20261018r6 --q0b-a100-job-id q0b-capacity-a100-20261018r6 --drive-output-dir /content/drive/MyDrive/t16_colab/q0q1-20261017/q1/jobs_a5_refused_r1 --out-dir '/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab/q0q1-20261017/q1/jobs_a5_refused_r1' --owner-admission /Users/josenunez/Projects/ml/snake-dqn-q1-a5/research/q0q1_20261017/OWNER_ADMISSION_20261018_A5.json
```

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a5-20261018`.
The invocation JSON, log, final artifact manifest and validation JSON are
create-only evidence. Old admissions, receipts, manifests and accounting remain
untouched. `allocation-accounting.json` is an earlier local development receipt;
only **allocation-accounting-v2.json** is bound by the actual invocation.

## Verification and limits

The final scoped suite on published source passed **387 tests in 44.43 seconds**:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-q1-a5
env SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_q1_*.py tests/test_q0*.py tests/test_make_q1_jobs.py \
  tests/test_requalify_q0b.py tests/test_colab_queue*.py tests/test_colab_cli*.py
```

The independent focused review passed **92 tests in 6.73 seconds** across
allocation accounting, qualification, generator, real synthetic admission/runtime
integration and queue jobs. Mutation probes rejected omitted workers/claims,
shortened original attempts, changed hardware and inflated credit limits.
The synthetic success path validates all 120 generated test manifests; it does
not represent real-packet admission. The actual refusal entrypoint emitted none.

Repository `make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint`
passed. Black (line length 100), isort (black profile) and flake8 passed on all
nine changed Python files; `git diff --check` passed.

The additional non-slow gate was **not green**: **7726 passed, 19 failed,
26 skipped, 6 warnings in 473.71 seconds**, with two native worker segfaults.
It began while the synthetic A5 fixture migration was underway. Its 12 Q1
integration failures used old fixtures; the final 387-test run above includes
all repaired integration tests and passes. The other seven failures are:

| Unresolved full-gate failures | Evidence |
| --- | --- |
| Three M6 reward tests | Actor payload-key drift and produced_monotonic comparisons |
| T15 source-closure test | Built frontend dist absent from isolated worktree |
| Two T21 actor-control tests | Native worker crashes during short trace/restart tests |
| T4 restart actor parity | Frozen baseline payload-key mismatch |

These match the failure classes documented before A5 in
[the A4 limitations record](Q1_A4_TEST_LIMITATIONS_20261018.md). Corresponding
actor/reward/T4/T15/T21/frontend source was not changed by A5. This run does not
independently prove all seven fail identically on every baseline, and a whole-repo
green gate is not claimed. No unrelated repair or environment change was made.

Exact broad command:

```bash
env SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 4 -m 'not slow'
```

## Operator commands and runtime state

**Do not enqueue the old A4 packet or this refused A5 record.** No runtime is
required for Q1 under this decision. If a later independent study is proposed,
it needs its own authority; A5 permits no further raises.

Read-only inspection:

```bash
/Users/josenunez/.local/bin/colab --auth oauth2 sessions
cat /Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a5-20261018/validation.json
cat /Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a5-20261018/generator-invocation.json
cat /Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a5-20261018/artifact-manifest-final.json
```

The existing colab_queue_worker_b L4 session was terminated via Colab's
Manage sessions UI under the owner's conditional A5 release decision. The
account-wide CLI reported **No active sessions found on server** at
01:57:53 and 01:58:38 UTC on Oct 19 (18:57:53 and 18:58:38 PDT on Oct 18).
No new runtime was allocated and no notebook cell was executed.

No tmux sessions, released checkpoints, serving pins, scientific code, seed bank
or old hash-bound evidence were modified. The explicitly requested A5 appendix
is the only registration change; its old prefix is byte-identical. Owner admission
A5 is a new file, and older owner admissions are preserved.

## Artifact digests

| Artifact | SHA-256 |
| --- | --- |
| study-plan.json | `ad21e20b017f44d9bd65c47d498e619d019c771e3eda3a927077a3ef1c2587c5` |
| admission.json | `62d0e6e8aee82a25c6794b0a7fee983ad9f701236e5aa12b5281d85ea15adfe6` |
| allocation-accounting-v2.json | `d2ae90f8e3cba530fc4cab78ec3501cf81442b2564675e0fe784417bb1225cc0` |
| allocation-bounds-v2.json | `4e03513e89e053ca2b233d55c1a573f13a64a5cdaafdd1abaf6b28fa9e9bb4fe` |
| generator-invocation.json | `defd8c4758db8ba7df40341b5bc29bda18baae0b43642c82d80b907e5ca9fc96` |
| generator.log | `87824b9bb5fb4b0dbe3e25916321360388a4a0a7e7d91025c220b72327a0532c` |
| validation.json | `3c15217d45da6e43a1075d5bcdb86f692029b2dd592176a2c6b6b1fe164a2254` |
| release-sessions.json | `dc993f3549ec525cad5ace46f94d8c7f7b6fdc26d5da791f99f1faa805ff9242` |
| final-scoped-tests.log | `efb34aadd701b8f2a20de15a07265e83a357dec33ccaef65c82dcc1f68fb9620` |
| full-tests.log | `3e0f0b19c0afe676a7baedbac2d2ceff0035c0af676da987c0d15e6d87cb138c` |
