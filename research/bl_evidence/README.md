# PLAN_v3 M3: file-backed evidence boundary

This module checks existing files only. It does not run games, allocate Colab,
change a registration, qualify C15, or authorize release. The five required
negative controls are in `tests/test_bl_evidence.py`. Test-generated corpora are
protocol conformance fixtures; their provenance fields do not turn them into
measured production evidence.

`loader.load_clock(report, expected_source_digest, expected_C)` returns a typed
`ClockEvidence` or raises. `clock_problems` converts all malformed/missing evidence
into named defects. Both expected digests come from the consumer's frozen inputs,
not from the report being checked. A field-only v1 report is insufficient.

A v2 report has schema `bl-evidence-clock/v2`, verdict `PASS`, an absolute `root`,
`candidate_sha256`, `source_binding_sha256`, an absolute-path `source_sha256`
closure, an `execution` file reference, and exactly four `cases`. References are
`{"path": "relative/path", "sha256": "64 lowercase hex characters"}`. Source
bytes are rehashed and the closure must match the externally expected digest.
All evidence paths stay inside the root; links and `..` escapes are refused.

The execution file uses `bl-evidence-execution/v1`: expected candidate/source
bindings, `platform: Darwin`, `environment: local-mac`,
`scope: production-clock-seam`, `synthetic: false`, `real_clock: true`, and
`real_transport: true`. This is an operator-attested execution record, not the
platform of the validator. Hashes bind those attestations; they cannot authenticate
an untrusted producer lying about its execution host. The authorized operator must
supply the frozen inputs and preserve the execution provenance chain.

Each case (`on-time`, `publication-late`, `consumption-late`, `wrong-frame`) has
`accepted`, `reason`, `scheduler_verdict`, `journal_seal`, and `latch`.
The loader opens the entire journal through `read_journal`, checks the chain and
completion markers, rehashes the typed candidate identity (including scheduler),
and independently runs `replay_events`. It derives the disposition from the
journal rather than trusting the summary. Header `binding_sha256` equals the
expected source digest; initial state binds `execution_sha256` and declares the
same nonsynthetic production scope.

Each latch is a completed `bl-evidence-game-latch/v1` JSON file with candidate,
execution and journal-seal digests, the decision's `request_id` and `action`, and
an integer `applied_ns` captured by the game after it actually latches the action.
It must be no earlier than the journal's final event; accepted actions must be
strictly before their consumption cutoff. The existing controller's pre-return
`commit` event cannot substitute for this record. This checks clock disposition;
full state/score/game replay remains a separate prerequisite. Until a production
provider writes these artifacts, clock admission remains `NOT_READY`.

JSON evidence, execution records, latches and exported reports use the existing
create-only `export_json` contract: finite canonical JSON + SHA256 sidecar +
completion marker published last. Journal seals pin all chunks. Missing evidence
blocks admission, without changing the scientific result or triggering a retry.
No budgets, timeouts or retry policies change in M3; there is no job to resume.

`strict_pins.strict_evidence_problems` opens all existing release evidence and
compares it to the immutable owner-approved FRP3 and B8 release anchors. It
checks checkpoint/veto identity, audit/decision clauses, and cross-file links.
Runtime checks reuse the frozen audited release result; they do not recompute
5,805 historical scientific records on every session construction. A new
checkpoint, assisted candidate digest, or rehashed forged summary cannot inherit
these anchors. A new Claim B packet validator is a later PLAN_v3 item.

Run the light, targeted checks from the worktree root:

```sh
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_bl_evidence.py
```

When real production artifacts exist, the offline CLI checks a caller-pinned
completed report and writes a new immutable diagnostic receipt (exit 0 PASS,
exit 2 NOT_READY). It grants no serving or release authority:

```sh
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_evidence \
  --report /absolute/production/clock.json --report-sha256 FROZEN_REPORT_SHA256 \
  --source-binding-sha256 FROZEN_SOURCE_SHA256 --candidate-sha256 FROZEN_C \
  --receipt /absolute/fresh-output/clock-check.json
```
