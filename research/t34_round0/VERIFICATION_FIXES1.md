# T34 Round 0 FIX-FIRST #1 verification — 2026-10-20

Status: **DRAFT_VALIDATED_NOT_RATIFIED**. Implements every required ruling in
`docs/research/specs/2026-10-20_t34_round0_fixes1.md`.
Final executable source: `838ae93e025cca4c695e4c50a603beb8b7654bf3`, published to
origin/main and `t34-round0-fixes1-20261020`. Later verification-document commits
do not alter the bound executable/rule closure. No gameplay arm, excluded A6,
Colab allocation/operation, tmux operation, ratification or release occurred.

## Implemented rules

- Canonical `REGISTRATION_ROUND0.json` must have identical bytes on origin/main,
  published after the frozen source commit, before dispatch. Its study root is
  immutable. Every admission scans all origin `t34-round0/*` claim tags and rejects
  a different root. Tests use isolated temporary bare Git remotes.
- Dedicated non-finite Q/loss, illegal-action and one-action-share exception types
  determine scientific faults. Message contents cannot create a scientific STOP.
  Trace, pair and chunk serialization failures are operational INCOMPLETE and
  can qualify for the single registered manual mechanical rerun with verified
  release and an outcomes-unseen attestation.
- Analysis reserves its fixed directory before terminal inspection. Missing
  terminal files or erased local claims, including an erased second attempt,
  produce its sole KILL decision. Structural settlement markers let pre-deadline
  checks recognize early closure without reading terminal JSON or outcomes.
- Producer wall caps, runtime timeout, and exact broker/accounting cap refusals
  settle INCONCLUSIVE, without extension. A proven cap precedes mechanical
  recovery. A replay request after a cap refuses before reservation, preserving
  the sole INCONCLUSIVE settlement. Abort/unexplained interruption stays KILL.
- The unchanged bank has a derived 4:3:3 scripted/frozen/mixed execution view.
  Chunks end on ten-pair stratum boundaries. Exact world IDs, the 48-hour
  ratification-derived dispatch deadline, immutable admission time and per-chunk
  completion interval are bound and checked. Incomplete healthy schedules refuse
  an early analysis without consuming the look; a missing chunk after deadline
  writes KILL. Operational history is validated before opening chunk exports.
- The launcher checks `T34_ROUND0_PROCESSES`, matching the manifest. Its real Bash
  entrypoint rejects a tampered count before source/input/world execution.
- README, ERRATUM_FIXES1.md and PREREGISTRATION_ROUND0_FIXES1.md state the same
  cap/deadline/fault rules. The new supplements are source-bound; original
  hash-bound preregistrations, amendments, banks, audits and receipts are unchanged.

The optional extra dispatch qualification hash check was not added. Preparation
and manifest generation still check the existing qualified transport. No
`research/colab_cli` file was edited.

## Rebuilt draft and source delivery

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-fixes1-20261020/`.
Final draft: `draft-02/`; `draft-01/` retains the earlier published source snapshot.
Both are explicitly DRAFT and cannot pass production registration admission.
`VERIFICATION_FIXES1.json` records the final artifact paths and SHA-256 hashes.

The draft builder rechecks the saved A5 adapter against the two retained A5
benchmark/supervisor receipts and the prior A6 measurement bytes. It validates
the package and both complete supervisor-schema draft manifests without
allocating. It reports `source_current_preflight=false`: saved historical A6
costs price this draft but do not certify changed executable source. A new,
source-current excluded A6 and independent ratification remain future dispatch
requirements. None was run in this implementation task.

| Chunk | Paired worlds | Scripted/frozen/mixed | Workload, min | Estimate, min | Timeout, min | Export, MiB | Cap, u | Reserve, u |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| 000 | 140 | 56/42/42 | 178.25652862276928 | 179 | 358 | 1575 | 11.933333333333334 | 2.3866666666666667 |
| 001 | 100 | 40/30/30 | 131.61180615912093 | 132 | 264 | 1125 | 8.8 | 1.7600000000000002 |

At 2.0 units/hour, initial cap plus 20% reserve is **24.88 u**; maximum liability
including one manual mechanical rerun per chunk is **49.76 u**. These are
prospective bounds, not spend. The 15-minute setup allowance, slower saved A5
rate and conservative 3x CPU projection are preserved. The 240 paired identities
and statistical thresholds remain fixed. A2 selects 14+10=24 paired worlds.

Local source preparation passed for the final draft manifest: **844 files**,
**3,406,107 compressed bytes**, **13,551,531 unpacked bytes**, each below the
32 MiB transport limit. `source-02/` retains the plan, archive and qualification
summary. This is local delivery preparation, not provider execution evidence.

## Verification commands and results

All test invocations pin native numerical threads to one and use `-n 0`.
At most two test invocations ran concurrently; no xdist auto fanout was used.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-round0-fixes1
export PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
"$PY" -m pytest -q -n 0 \
  tests/test_t34_round0_registration.py tests/test_t34_round0_manifest.py \
  tests/test_t34_round0_runner.py tests/test_t34_round0_integrity.py \
  tests/test_t34_round0_analysis.py tests/test_t34_round0_fixes1_chunks.py \
  tests/test_t34_round0_fixes1_faults.py tests/test_t34_round0_fixes1_history.py \
  tests/test_t34_v2_policy.py tests/test_t34_v2_exports.py
"$PY" -m pytest -q -n 0 tests/test_colab_cli_supervisor.py \
  tests/test_colab_cli_source.py tests/test_colab_cli_admission.py
BLACK_NUM_WORKERS=1 make PY="$PY" lint
BLACK_NUM_WORKERS=1 "$PY" -m black --check --line-length 100 \
  research/t34_round0 research/t34_exit_b8/policy.py research/t34_exit_b8/teacher_v2.py \
  tests/test_t34_round0_*.py
"$PY" -m isort --profile black --check-only \
  research/t34_round0 research/t34_exit_b8/policy.py research/t34_exit_b8/teacher_v2.py \
  tests/test_t34_round0_*.py
"$PY" -m flake8 research/t34_round0 \
  research/t34_exit_b8/policy.py research/t34_exit_b8/teacher_v2.py tests/test_t34_round0_*.py
bash -n research/t34_round0/run.sh
git diff --check
"$PY" -m research.t34_round0.draft \
  --a5 /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020/a6-03/A5_RECEIPT.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-fixes1-20261020/draft-02 \
  --study-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01
```

Final package/adjacent suite: **112 passed in 35.57 seconds**: the original 68
Round 0 cases retained (fault cases updated to require types), 34 new FIX-FIRST
regressions and 10 adjacent policy/export cases. An intermediate source passed
111 tests; the final cap/replay adjustment separately passed 45 affected tests
before this full 112-case rerun. No failing test was removed.

Supervisor/source/admission suite: **92 passed in 528.60 seconds**. These tests
use fake providers and temporary source/account state; no live allocation ran.
The final two suites total **204 passing tests**.

Project lint (149 source/backend files), scoped Black/isort/flake8, Bash syntax,
whitespace, draft validation and source-archive preparation passed. Source archive
preparation used `research.colab_cli.source.prepare(ROOT, first_draft_manifest,
Path(artifact_root) / "source-02")`, with the exact draft source paths and pinned
commit. The shell process-count regression executes the real launcher and
requires exit 2 before any runtime work.

Independent read-only review identified and verified fixes for receipt terminal
fallback, all serialization boundaries, teacher timeout races, partial schedules,
early STOP with healthy retained chunks, erased claims, reserve-before-terminal
ordering and cap-before-mechanical precedence. Its final reread found no blockers.
The final replay-after-cap adjustment has its own focused regression.

The full project nonslow suite was not run; the bounded research, adjacent and
transport suites above are the qualification scope. No live L4 run, source-current
A6, cross-platform gameplay parity, Mac strict gate, serving qualification or
release is claimed. No running worker, released checkpoint, serving pin or
existing hash-bound evidence was changed.
