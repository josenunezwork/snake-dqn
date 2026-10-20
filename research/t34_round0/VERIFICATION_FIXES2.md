# T34 Round 0 FIX-FIRST #2 verification — 2026-10-20

Status: **DRAFT_VALIDATED_NOT_RATIFIED**. All four rulings in
[FIX-FIRST #2](../../docs/research/specs/2026-10-20_t34_round0_fixes2.md) are implemented.
Executable/rule source commit: `b60b0e3a25ac391ec40dc5dee82772901750c139`, published to
origin/main and `t34-round0-fixes2-20261020` before A6. Implementation commit:
`c96617c6` (abbreviated; the full source commit above includes the main merge).
Subsequent verification-document commits do not change the 844-file bound closure.

## Implemented behavior

- Preregistration supplement §3/§5, erratum and README distinguish supervisor
  wall-cap timeout / first unit cap (**INCONCLUSIVE**, no extension) from a PASS
  receipt whose recorded run exceeded its registered timeout (**KILL**).
  Receipt-settlement tests cover both timeout reasons, exactly-on-time PASS and
  over-time PASS. No settlement rule or statistical threshold changed.
- `history.finalize` seals a structural `schedule_complete.json` only for a
  completed chunk attempt. The shared pre-reservation guard checks the latest
  attempt of every registered chunk. Missing chunks and sealed failures, including
  pending mechanical reruns, refuse both replay and analysis through the original
  48-hour deadline without consuming a phase or reading terminal JSON/outcomes.
  A failed second attempt also waits until the deadline; no third attempt is
  authorized. A successful second attempt completes the schedule. Missing or
  failed chunks after the deadline still produce the sole KILL decision.
  Structural markers are included in the immutable terminal inventory; full
  terminal validation still follows reservation. Old seals without a completion
  marker fail closed. Early STOP/cap/KILL and unfinished/erased claims preserve
  their existing settlement paths.
- Shared faults now live in `research/t34_exit_b8/faults.py`. Policy, teacher and
  Round 0 runner import that module. The original Round 0 path re-exports all
  seven names, including `CapExpired`, with identical class objects; legacy
  exception imports retain identity. Shared teacher code has no study-fault import.
- FIX-FIRST #2 is explicitly included in the source-bound rule documents.

Independent read-only review found no blockers. It verified pre-reservation
ordering, latest-attempt handling, both consumers, deadline boundaries,
cap/STOP/erased-claim preservation, fault identity, and the source-inventory
requirement to commit the new shared module before A6.

## Fresh excluded A6 and sizing

A6 root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020/a6-fixes2-01`.
The requested fresh directory was absent before launch. The Mac was on AC with
the lid open and no reported thermal warning. The run used caffeinate, CPU,
one spawned worker, one numerical/inference thread, W1, two excluded H500 paired
worlds and a 600-second cap. It completed in **23.800773541006492 seconds**, with
two pairs and zero drops. No selection-bank gameplay ran.

| Pair | Bare rollout seconds | Teacher rollout seconds | Bare metric seconds | Teacher metric seconds |
|---|---:|---:|---:|---:|
| 1 | 9.190805916994577 | 3.9432962909922935 | 0.15530595800373703 | 0.1624077910091728 |
| 2 | 3.045130499987863 | 5.491168000007747 | 0.19438633299432695 | 0.19517125000129454 |

Pair 1's bare rollout is about 3.02 times pair 2's, consistent with a first-pair
cold/JIT outlier; no separate profiling was performed to isolate that cause.
The registered **maximum over both pairs is retained**, with no warm-up exclusion
or alternative aggregation. The H500-to-H5000 factor remains 10, so the adapter
uses **91.90805916994577 bare seconds/world** and **3.895575829956215 metric
seconds/pair**. Slower saved A5 teacher throughput remains
**54.37214064869172 worlds/hour**. The 15-minute setup allowance and 3x CPU
projection remain unchanged. These are prospective sizing assumptions, not a
measurement of paired L4 throughput.

`A5_RECEIPT.json` and `VERIFICATION.json` passed the README sizing/validation
commands. The validation binds the current **844 source/rule/input files**,
released B8 checkpoint identity and both pair/arm trace hashes. It confirms
two excluded H500 pairs, CPU/one thread and no fit labels.

## Rebuilt draft

Fresh draft root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-fixes2-20261020/draft-01`.
Package validation and all **three** supervisor-schema draft manifests pass;
`source_current_preflight=true`. Registration remains DRAFT with
`ratified=false` and `dispatch_authorized=false`.

| Chunk | Pairs | Scripted/frozen/mixed | Estimate min | Timeout min | Export MiB | Unit cap | Reserve |
|---|---:|---|---:|---:|---:|---:|---:|
| 000 | 100 | 40/30/30 | 166 | 332 | 1125 | 11.066666666666666 | 2.2133333333333334 |
| 001 | 100 | 40/30/30 | 166 | 332 | 1125 | 11.066666666666666 | 2.2133333333333334 |
| 002 | 40 | 16/12/12 | 76 | 152 | 450 | 5.066666666666666 | 1.0133333333333334 |

At the unchanged 2.0 units/hour rate, initial cap is **27.2 u**, reserve is
**5.44 u**, combined initial liability is **32.64 u**, and the maximum including
one manual mechanical rerun per chunk is **65.28 u**. These are draft bounds,
not spending or funding approval. The 240 selection identities, bank/audit bytes,
mix proportions and decision thresholds are unchanged.

`VERIFICATION_FIXES2.json` records artifact paths and SHA-256 identities, including
A6 inputs/receipts, sizing, draft registration, chunk plan and three manifests.

## Exact verification commands

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-round0-fixes2
export PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export BLACK_NUM_WORKERS=1

# Initial schedule/settlement selection: 43 passed in 13.48 s.
"$PY" -m pytest -q -n 0 tests/test_t34_round0_fixes1_history.py tests/test_t34_round0_fixes1_chunks.py
# Final complete Round 0 + adjacent suites: 139 passed in 41.64 s.
"$PY" -m pytest -q -n 0 tests/test_t34_round0_*.py tests/test_t34_v2_policy.py tests/test_t34_v2_exports.py tests/test_t34_teacher_v2.py
# Fault lane subset: 42 passed in 0.96 s (included in final suite).
"$PY" -m pytest -q tests/test_t34_round0_fixes1_faults.py tests/test_t34_round0_runner.py tests/test_t34_v2_policy.py tests/test_t34_teacher_v2.py

make PY="$PY" lint
"$PY" -m black --check --line-length 100 research/t34_round0 research/t34_exit_b8/faults.py research/t34_exit_b8/policy.py research/t34_exit_b8/teacher_v2.py tests/test_t34_round0_*.py
"$PY" -m isort --profile black --check-only research/t34_round0 research/t34_exit_b8/faults.py research/t34_exit_b8/policy.py research/t34_exit_b8/teacher_v2.py tests/test_t34_round0_*.py
"$PY" -m flake8 research/t34_round0 research/t34_exit_b8/faults.py research/t34_exit_b8/policy.py research/t34_exit_b8/teacher_v2.py tests/test_t34_round0_*.py
bash -n research/t34_round0/run.sh
git diff --check

export A6=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020/a6-fixes2-01
export A5ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991
"$PY" -m research.t34_round0.bank --preflight-inputs "$A6/inputs"
caffeinate -dims env SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes \
  "$PY" -m research.t34_round0.runner --preflight \
  --bank "$A6/inputs/bank.json" --audit "$A6/inputs/audit.json" \
  --checkpoint /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth \
  --output-dir "$A6/output" --runtime-id t34-round0-a6-fixes2-01 \
  --start 0 --count 2 --device cpu --threads 1 --seconds 600
"$PY" -m research.t34_round0.sizing \
  --a5-benchmarks "$A5ROOT/a5-l4-s0-output/exports/benchmark.json" \
                  "$A5ROOT/a5-l4-s1-output/exports/benchmark.json" \
  --preflight-dir "$A6/output" --setup-minutes 15 --out "$A6/A5_RECEIPT.json"
"$PY" -m research.t34_round0.validate --preflight-dir "$A6/output" \
  --a5 "$A6/A5_RECEIPT.json" --out "$A6/VERIFICATION.json"
"$PY" -m research.t34_round0.draft --a5 "$A6/A5_RECEIPT.json" \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-fixes2-20261020/draft-01 \
  --study-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01
```

All final lint commands passed (149 project source/backend files, 26 scoped
Python files); Bash syntax and whitespace checks passed. A first Black check
requested history.py line wrapping, which was formatted before the final pass.
The final test suite contains 119 Round 0 cases and 20 adjacent cases. Tests
use bounded synthetic fixtures / temporary Git remotes; no live provider work.
The full project nonslow and transport suites were not rerun for this bounded
research change. No training, Mac strict gate or serving qualification is claimed.

## Operator handoff and constraints

No further execution is required to complete FIX-FIRST #2. For read-only
revalidation, call `research.t34_round0.validate.validate(A6/output,
A6/A5_RECEIPT.json)`; do not reuse create-only CLI output paths. The completed
A6 paths must not be rerun or overwritten. Future execution still requires
independent ratification and the frozen-checkout/publication procedure in README.
No production admission, new approval receipt, claim tag, replay, scientific
decision, release or funding change was made.

No live Colab or tmux command ran; no session, worker, released checkpoint or
serving pin was changed. No gameplay ran beyond the specified two-pair A6.
The explicitly requested FIXES1 supplement/erratum were corrected prospectively
under FIXES2; historical preregistration/amendment bytes, banks, audits and existing
artifact receipts remain unchanged.
