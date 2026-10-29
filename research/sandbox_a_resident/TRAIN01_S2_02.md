# TRAIN-01 S2-02 — per-attempt envelope correction

Prepared only under the owner's explicit 2026-10-29 continuation instruction and the
standing development sandbox authority. Outputs remain simulator-D development evidence.

`sandbox-a-train01-20261028-s2-01` is **BURNED** and must never be reused. Its supervisor
receipt at `/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-s2-attempt-01/output/receipt.json`
records `INCOMPLETE`, `endpoint: null`, no allocation call, and
`Refused('insufficient balance for cap plus study reserve')`. `stop_verified: false` is
not a failed stop: allocation never occurred. Preserve that identity, packet and receipt.
This reseal is a fresh initial attempt, not a trajectory continuation or a consumed
completion retry. No training data or scientific outcome from S2-01 exists.

The new identity is `sandbox-a-train01-20261028-s2-02`. Freeze a **690-minute per-attempt
estimate** and **1,380-minute effective timeout** (exactly 2x, 23 hours). Price the entire
timeout at each ceiling: H100 12 u/h, base/effective cap 138/276 u; A100 8 u/h,
base/effective cap 92/184 u. Base/effective reserve is zero. The full unchanged
384-update workload remains forecast at 2,009.04 minutes on the observed A100 recipe;
690 minutes is an attempt planning window, not a claim that all updates fit in it.

Keep compact full-state snapshots every 16 updates, verified Mac exports, empty initial
ledger, allocation index 2 and training offset 2,000,000,000. S2-01 never allocated, so
its exhaustively audited seed domain can be reused without duplicating any training.
The frozen control is unchanged except for source commit. No new seed audit is needed:
reuse and verify the exact audit and disjointness bytes sealed with S2-01.

The continuation builder narrowly admits S2-02 and its c1/c2 descendants and caps each
of those attempts at 690 base / 1,380 effective minutes. It retains the uncapped
remaining-work forecast and all 384 updates, proxy banks, RNG and counters. All other
packet families retain their existing envelopes. A completed snapshot is not a scientific
look. Continue only from the latest verified full-state snapshot after mechanical failure,
verified stop/release and missing-unit checks; at most two manual completion attempts.
No automatic continuation. An early failure may exhaust the three-attempt allowance
before the endpoint; report INCOMPLETE in that case without raising caps mid-series.

Before sealing, read-only account observation found r6's cap 369 u and P09e's cap 14.63 u,
reserves zero, so committed liability is 383.63 u. Admission requires balance strictly
above 659.63 u for H100, or 567.63 u for A100. At the observed 883.73 u balance both fit;
the prior H100 cap required 1,199.63 u and failed. The retained accounting evidence calls
the real `admit_account` with live sessions/rates and active records, without changing them.
The actual dispatch still refreshes all account, hardware and host checks.

Packet, source pins, clean detached checkouts, checklist, tests, independent Argon review,
accounting evidence, dispatch dry-run and the exact command are recorded in the sibling
SEAL/verification record and
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-train01-s2b/SUMMARY.md`.
