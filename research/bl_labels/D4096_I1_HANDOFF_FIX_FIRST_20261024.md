# D4096 I1 FIX-FIRST operator addendum — 2026-10-24

This sibling addendum supersedes only the disclosure, discard handling and release-accounting
instructions in `D4096_I1_HANDOFF.md`. The original handoff, pack note, A1 control bytes and all
existing immutable evidence remain unchanged. All B1/B2, funding, identity, host and review gates
still apply. No factory or provider operation was performed by this fix round.

D4096's prospective design is **nested mixed-platform (Mac + Colab L4 CPU) collection,
tie-tolerant L4 admission**. Owner decision:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/fable-20261024/OWNER_DECISION_TIE_TOLERANT_20261024.md`,
SHA-256 from its `.md.sha256` sidecar:
`003399e7ee6835d622ecf48fc04d2e862926c61991f8426243b1bf62db300b20`.
Only replay-audit-classified float ties are admitted; unexplained mismatches close L4. The ≤50%
ceiling, rate-sized share, L4-only placement, PASS funding and separate identity/fault gates stand.

New readiness, B2, attempt-start/terminal, I3 projection, discard, rebinding and release receipts carry
`collection_design` and `owner_decision`. `actual_execution_placement` reports the observed placement
or explicitly that D4096 has not started; the prospective design alone never claims L4 execution.
I3 projections explicitly say `l4_ledger_status: ABSENT` or `BOUND`. An absent ledger's legacy empty
accounting object is not a statement that L4 cost zero, nor proof of a complete L4 inventory.

The pack's sibling note is `D4096_MIXED_PLATFORM_ADDENDUM_20261024.md`. A1 remains exactly the two
approved source substitutions; allocation, cap, seed audit, empty ledger and original copied control
bytes are preserved. Do not regenerate or amend any of those frozen inputs.

## Check before first start

Use the following command from the clean, published fix checkout. It is the same real-pack dry-run
entry point as before. Missing B1/B2 seals remain HOLD with `factory_started=false`. Do not substitute
fixtures. The actual start still requires independent review acceptance, actual B1/B2 PASS and all
original host/funding gates; this addendum does not start collection.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-d4096-i1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export D4096_PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/d4096-old-equiv-02-pack-01
nice -n 15 ./venv/bin/python -m research.bl_labels.d1024_w12 \
  --stage d4096 \
  --control "$D4096_PACK/d4096-control.json" \
  --ledger "$D4096_PACK/d4096-verified-ledger.json" \
  --output "$D4096_PACK/d4096-output-mac-01" \
  --state-dir "$D4096_PACK/d4096-state-mac-01" \
  --admitted-control "$D4096_PACK/d4096-admitted.json" \
  --speed-equivalence-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/equiv-02/receipt.json \
  --calibration-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-admission-01/receipt.json \
  --old-factory-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/equiv-pack-02/old-reference \
  --sealed-readiness "$D4096_PACK/d4096-sealed-readiness.json" \
  --projection-receipt "$D4096_PACK/d4096-mac-h-projection.json" \
  --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  --frp3 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth \
  --owner-w12-ok --check
```

## Discard only for an unrecoverable HOLD

For either explicit labels or terminal receipts, stage must be inferred/validated as `d1024-fit`
(KILL_FIT/INSUFFICIENT) or `r-a` (STOP_FUNDING/INVALID/HOLD). A receipt-based HOLD now also requires
`--unrecoverable`; reading a seal revalidates the same semantics. Recoverable HOLD preserves the
collection for later recovery and cannot create the permanent NOT_RUN_BY_RULE seal. Receipt stage
arguments, if supplied, must agree with the receipt schema.

After the operator has established that an R-A HOLD is unrecoverable, the receipt route is:

```bash
: "${RA_UNRECOVERABLE_HOLD_RECEIPT:?Set the sealed terminal R-A HOLD receipt path}"
nice -n 15 ./venv/bin/python -m research.bl_labels.d4096_sealed seal-not-run \
  --out "$D4096_PACK/d4096-not-run-by-rule.json" \
  --readiness "$D4096_PACK/d4096-sealed-readiness.json" \
  --terminal "$RA_UNRECOVERABLE_HOLD_RECEIPT" --terminal-stage r-a --unrecoverable
```

Presence of the discard payload, digest sidecar or completion marker—including a dangling symlink—
blocks collection/publication. Never remove malformed evidence to bypass that HOLD.

## Release accounting before any mixed-platform fit

The release command now requires one of the two explicit accounting declarations below. The writer
checks exact hash-set equality between the separate complete attempt inventory and the L4 accounting
ledger. Missing failed, setup or identity attempts are refused, even when they exported no labels.
Every listed supervisor receipt is rehashed. The release binds both immutable metadata files and the
attempt count/total charge. The complete-inventory attestation remains the operator's responsibility;
no provider query or automatic discovery is performed, and an attempt omitted from both inputs
cannot be detected by this local comparison.

Create the inventory with `research.bl_labels.exports.publish_json` in a fresh path. Its schema is:

```json
{
  "schema": "bl-labels-d4096-l4-attempt-inventory/v1",
  "series_id": "<the canonical PASS funding series_id>",
  "registration_sha256": "<F4_REGISTRATION_SHA>",
  "dose": 4096,
  "all_attempts_listed": true,
  "attempts": [
    {"receipt": {"path": "<absolute supervisor receipt.json>", "sha256": "<sha256>"}}
  ]
}
```

Include every D4096-related charged L4 attempt, including failed collection, setup and identity work.
Use the existing `bl-residual-d4096-l4-accounting/v1` ledger with one row per identical receipt hash
and its measured `charged_units`; never infer missing cost as zero. Preserve all original attempts.

After canonical PASS funding and a complete funded union, for any L4 attempts:

```bash
: "${D4096_FINAL_LEDGER:?Set the complete funded final union ledger path}"
: "${D4096_FUNDING_RECEIPT:?Set the canonical PASS funding receipt path}"
: "${D4096_L4_LEDGER:?Set the sealed complete L4 accounting ledger path}"
: "${D4096_L4_ATTEMPT_INVENTORY:?Set the sealed complete L4 attempt inventory path}"
nice -n 15 ./venv/bin/python -m research.bl_labels.d4096_sealed release-collection \
  --pack "$D4096_PACK" --ledger "$D4096_FINAL_LEDGER" \
  --funding-receipt "$D4096_FUNDING_RECEIPT" --code-commit "$(git rev-parse HEAD)" \
  --l4-ledger "$D4096_L4_LEDGER" --l4-attempt-inventory "$D4096_L4_ATTEMPT_INVENTORY" \
  --out "$D4096_PACK/d4096-collection-released.json"
```

Only if no L4 attempt occurred, use the same command with `--no-l4-attempts` instead of both L4
arguments. That emits `ABSENT_NO_L4_ATTEMPTS_ATTESTED` and null bindings, not a zero-charge snapshot.
A mixed union cannot use that attestation. A Mac-only final union still needs L4 accounting if L4
setup/identity/failed attempts occurred. The old release command without either declaration refuses.
I3's later fit-pack builder still binds the exact release and funding receipts and final ledger hash.
