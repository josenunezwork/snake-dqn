# Codex-V R32 independent verification

This file records the independent acceptance surface for R32-1/R32-2. The
fixtures are synthetic and excluded from scientific admission; no factory labels,
protected study directory, gameplay bank, release pin, or dispatch endpoint is
used.

`research/r32_rehearsal/ORACLE.json` is the hand-calculated expected matrix. It
contains one case for each of the ten ordered R9 labels, plus exact equality,
`NOT_RUN_BY_RULE` versus missing required Watch6, `TOP_PASSED_DOSE_FAILED`, mass
pass/survival fail, and mass pass/Watch fail. The cases are built independently
in `research/r32_rehearsal/decisions.py` by reusing the existing synthetic F4
fixture helper. `payload_case` exposes the reducer input, while `run_case`
returns the observed label, static expected label, full decision, and explicit
`pass`; the reducer is not used to generate the expectations.

`research/r32_rehearsal/matrix.py` owns the portable lifecycle matrix. Its three
positive traversals are clean completion, interrupted/resumed completion with
the same verified union (including interrupts before and after marker
publication), and scientific stop with downstream units exactly
`NOT_RUN_BY_RULE`. Its twelve faults are wrong sidecar basename, truncated
payload, marker-before-payload, stale source proof, wrong checkpoint bytes,
foreign namespace/allocation, duplicate immutable unit, ancestor split leakage,
missing full-state snapshot, unknown runtime stop, timeout after export, and a
changed output filename attempting a second look. A timeout after export retains
the verified units and returns `STOP_UNVERIFIED`; it cannot complete or authorize
replacement.

The light lane is intentionally one process at a time, single-threaded, at nice
15. The root operator should run:

```bash
nice -n 15 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m pytest -q -n 0 tests/test_r32_rehearsal.py
```

Acceptance is 3/3 positive paths, 12/12 faults, all ten R9 labels, and the
separate missing-versus-not-run and timeout assertions. This artifact records
the acceptance definition; test outcomes and source hashes belong in the root
closeout after the root-owned single light test lane runs.

The delivery truth table in `ORACLE.json` keeps two receipts separate. The
portable worker receipt at `main-output/exports/receipt.json` is an export
accounting diagnostic and cannot assert release or `stop_verified`. The
qualified supervisor receipt at `main-output/receipt.json` is the only receipt
that can support reduction, and it must bind the saved manifest, source, spec,
exported-unit path/hash inventory, and positive accounting. Exported unit files
remain immutable evidence; the registered Mac terminal claim must agree with
the supervisor receipt. An unknown or `STOP_UNVERIFIED` stop refuses
replacement and reduction.

The final engineering review also checks the package subcontracts. The consumer
preflight now requires its exact schema, all six expected consumer source hashes,
the synthetic production-admission refusal, and SHA-verified exported
checkpoints. The reducer binds the supervisor manifest, pinned `spec.json`,
verified input-ledger bytes, producer source archive, export path/SHA/size
inventory, positive accounting, and terminal success or an explicitly classified
mechanical failure. The input ledger is the supervisor-bound resume skip set; the
reducer's `ledger_path` is the separate Mac-verified output union of all recovered
completed units. Every output row must match a recovered supervisor export by
canonical unit ID/name, path, size, and SHA before the terminal claim is written.
The Mac plan records the source repository, pinned commit, managed
working-directory contract, and the `tools/ops/dispatch.sh --mac --repo ...`
prefix; no dispatch is performed here.

Codex-V acceptance is limited to these synthetic engineering subcontracts. The
full consumer matrix, strict validator, serving registry, and release readiness
remain `BLOCKED`; this report makes no whole-consumer or strict-pass claim.
Root reported the prior 21-test light run PASS and an earlier 24-test run PASS
in 16.50 seconds; the post-fix repeat remains the authoritative pending result.

Final root evidence supplied after the producer source/spec/ledger binding fixes:

- The combined light/package lane passed 24 tests, with 76 deselected, in
  25.35 seconds.
- The actual default consumer preflight `consumer-02` passed under the 2700
  second hard alarm, including the tiny three-update fit and fresh-process
  consumer checks; no H3 worlds or Watch smoke were run.
- Portable tiny CLI preflight `preflight-01` passed with `done=3` and
  `expected=3`.
- The root subtree is qualified at `d0d92...`; publication remains the root
  integration step and no Colab allocation or dispatch occurred.

Independent engineering verdict: **PASS for the synthetic R32 lifecycle,
oracle, resume, receipt-binding, package, and tiny consumer-preflight
subcontracts**, with the required 3/3 positive paths, 12/12 lifecycle faults,
ten-label decision oracle, and package refusal coverage represented by the
reported root test lane. **BLOCKED for full consumer strict validation,
serving-registry admission, and release readiness**; the consumer preflight is
an engineering trace and does not establish the T34 strict/registry gates.
