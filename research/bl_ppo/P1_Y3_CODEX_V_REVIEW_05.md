# Independent source review — P1 Y3 source-latebody-05

Verdict: **PASS for software preparation**, 2026-10-22. No unresolved
findings; `no_new_cuda_path=true`. Independent read-only reviewer
`/root/independent_review`, separate from both implementation lanes and the
root integrator. This is neither dispatch authority nor a live L4 Y3 PASS.

The review covers amendment 05, ledger/input/admission preparation, full-state
recovery, source tape persistence, both replacement replays, and the affected
collector, runtime, qualification and export seams. Exact executable hashes
are bound in the immutable source-review receipt after publication. A separate
artifact review binds the actual canary and source templates after preparation.

The frozen invoice is 25.50 u total, preserving 4.67 u historical conservative
debit. Source: estimate 75 min, timeout 300 min, cap 9.62 u, reserve 1.92 u.
Replacement: 30 / 120 min, 3.86 / .78 u. Canary: 15 / 30 min, 2.00 / .40 u.
Only caps/reserves/timeouts are doubled; rates and estimates are unchanged.
L4 rate ceiling remains 1.54 u/h, zero tolerance, empty other-assignment map.

Review findings resolved before publication:

- A restored source must retain the original pre-actor origin and complete
  action-tape prefix. Latest verified snapshot selection checks the embedded
  namespace/allocation/index/unit ID as well as the file digest. Refit and
  support continuation retain their phase bookkeeping and exclusions.
- Wall time cannot choose a replacement endpoint. Both replays use the
  source's fixed absolute counter. A periodic-only mismatch emits
  `Y3_PASS_BOUNDARY_ONLY`, with operational recovery eligibility false.
- Mutable pre-actor banks are exported as content-addressed immutable
  components, keeping the primary snapshot below its frozen size bound.
- NumPy serialization loses the three interior-grid/padded-buffer aliases.
  `collector._GridShard.restore` now verifies their values and rebinds those
  views. This fixes a real interrupted-refit parity failure without changing
  fresh simulator dynamics or weakening exactness.
- Operational trace references must be flat basenames. The replay test hides
  the original source tree after transport so a replacement cannot silently
  read source-local files. Both Y3 phases require the new replay amendment.
- Code review and artifact review use separate receipts, avoiding a circular
  source-template hash. DISPATCH uses the actual shared venv and the wrapper's
  positional name/manifest interface, with explicit broker/cwd/dispatch text.

Independent verification: focused admission/ledger suite **17 passed**;
five forged embedded operational identities refused before training; a
10,000-row production-cardinality ordinary bank serialized and restored in
**129,531,629 bytes**, below 192 MiB. Root's whole-package suite recorded
**221 passed in 218.82 s**, including both replays and interrupted refit/actor
entrypoints. Later admission checks and exact preparation evidence are recorded
in the final preparation index; CPU software evidence is never live L4 proof.

Existing model/learner operations, CUDA RNG capture and restore remain in use;
there is no new CUDA operator, kernel, precision or scientific recipe. No
provider operation, runtime allocation, serving change or release occurred.
Prior registrations and receipts are immutable. The excluded study directory
and `../snake-dqn-l2mac` were not accessed.
