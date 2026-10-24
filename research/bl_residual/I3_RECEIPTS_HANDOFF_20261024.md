# I3 D4096 receipt producers

Admission and verification updates from the independent FIX-FIRST review are in
[I3_FIXFIRST_HANDOFF_20261024.md](I3_FIXFIRST_HANDOFF_20261024.md). That handoff
supersedes this document's admission requirements and integration status.

Implements §1's zero-unit defect and §5 step 9 of
`JUDGMENT_D4096_START_20261024.md`, SHA256
`ccf1e0cd8d60a2019c983ab2714d58663ca42e2a228a0ddf78c16216c2ef83fb`.
This is code and synthetic metadata-fixture evidence. No production receipt,
collection, fit, funding look, provider query or dispatch was performed.

## Ownership and interfaces

- I3 owns `collection_receipts.py`, the `job.py` admission change, its tests and this handoff.
- I1 owns all `research/bl_labels/*` changes and the B2 Mac-h writer. I3 does not implement that writer.
- I2 owns `f4_ra_prepare.py` and `funding.py`; I3 does not edit them.

`collection` reads the sealed collection ledger, its pinned digest, its listed
`runtime-result.json` terminal accounting receipts, and the sealed predecessor
fit receipt/control. It never follows a ledger unit path, sealed-rate path or
state-directory path. It checks the control's source ledger hash and the fit's
control hash, allocation, recipe and namespace. The source is final only when
all earlier attempts are PREEMPTED and the last is COMPLETE without pending
parents; every listed attempt is charged, including attempts with zero exports.
It preserves the original D256 driver's already-established 1 ms cumulative
clock tolerance only for that exact driver hash on its first W6 attempt.

The output is `bl-residual-collection-receipt/v1`. Mac placement explicitly
charges zero provider units. `selected_rescue_lives` is the consumer's legacy
field name: it contains the **dose-defined selected increment**, 256 and 768,
not a rescue count read from data. Their denominator is 1,024.

`projection` reads the two sealed collection receipts and I1's sealed B2 receipt.
It writes `bl-residual-d4096-projection/v1`, binding both collection receipt
hashes and B2's hash. It does not produce or reinterpret B2. The original
measured-unit formula, fit identities, funding refusal, placement tree and
480-unit projection bar remain in `job.py`. The zero exception additionally
requires both Mac/zero-unit receipts, their predecessor-control source ledger
bindings, the 256/768 denominator and B2's exact bound hash and arithmetic.

## B2 contract for I1

No B2 writer was published on `origin/main` when I3 was implemented. The accepted
schema is **`bl-labels-f0-d4096-mac-projection/v1`**, demonstrated by
`tests/fixtures/bl_residual_i3_b2_mac_projection.json` and
`test_i1_b2_schema_fixture`. This file is synthetic, never operational authority.
I1 must export its real receipt create-only with SHA256 and completion sidecars.

Required fields are the fixture's schema, judgment SHA, `placement: mac`,
`verdict: PASS`, `selected_lives: 1024`, `increment_lives: 3072`, `margin: 1.2`,
`wall_cap_hours: 130`, dose-keyed `collection_ledger_sha256` and
`cumulative_wall_seconds` maps, and finite `projected_mac_hours` equal to
`sum(cumulative_wall_seconds) / 3600 / 1024 * 3072 * 1.2`. Values above 130 h
refuse, including a mislabeled PASS. Do not infer the denominator from rescues.
The sources and wall values must be the same final D256/D1024 ledgers/accounting
used by I1, sealed before D4096 collection dispatch under B2.

## Separate L4 R7 accounting

`l4_tail_accounting` is separate from `projected_units`. With no tails, the
projection records an explicit empty snapshot: zero charged units, no attempts,
base cap 480 and effective cap 960. This is **not a claim about future charges**.
When L4 attempts exist, pass `--l4-ledger` with a sealed metadata object:

```json
{
  "schema": "bl-residual-d4096-l4-accounting/v1",
  "series_id": "THE_FROZEN_SERIES_ID",
  "registration_sha256": "66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65",
  "dose": 4096,
  "attempts": [
    {"receipt_sha256": "SHA256_OF_ONE_TERMINAL_ACCOUNTING_RECEIPT", "charged_units": 1.0}
  ]
}
```

The L4 operator/adapter owns this ledger's measurement provenance and completeness:
include all charged attempts/setup/canaries/exports, including failures and
permitted identity work. I3 checks the sealed declared accounting and cap; it
does not query providers or independently certify source billing. Unknown,
non-finite, negative, boolean and duplicate charges refuse. The sum must be
≤960; 480 is the base projection bar, not an extra effective spend ceiling.
Never hide L4 charges in a Mac collection receipt or assume a zero-unit Mac
projection grants L4 admission. Funding and the L4 judgment still apply.

## Operator commands (not executed by I3)

First obtain Claude-V ACCEPT of I3 and I1's compatible B2 output. Do not run this
against the live D1024 factory. Run only after D1024's terminal final union and
fit publication are verified; use immutable accounting metadata. These commands
do not dispatch or open any label, outcome or rescue payload.

Set the two operator-supplied paths to the **existing sealed** D1024 stage-fit
receipt and I1 B2 receipt. A stage-fit receipt is produced by the existing
`job.write_stage_fit_receipt` machinery after the fit, outside this metadata-only
producer. Do not fabricate it or substitute a supervisor receipt.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-receipts-i3
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu PYTHONDONTWRITEBYTECODE=1
export DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib
I3_F4=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4
I3_OUT="$I3_F4/i3-receipts-01"
I3_D256=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/d256-union-final-01/verified-ledger.json
I3_D1024=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/d1024-union-final-01/verified-ledger.json
: "${I3_D1024_STAGE_FIT:?Set the accepted sealed D1024 stage-fit receipt path}"
: "${I3_B2_MAC_PROJECTION:?Set the accepted sealed I1 B2 receipt path}"
I3_SERIES=$(nice -n 15 ./venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["series_id"])' "$I3_F4/d1024-pack-01/inputs/control.json")
I3_RESERVE=$(nice -n 15 ./venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["reserve_allocation_sha256"])' "$I3_F4/d1024-pack-01/inputs/control.json")

nice -n 15 ./venv/bin/python -m research.bl_residual.collection_receipts collection \
  --ledger "$I3_D256" --ledger-sha256 "$(shasum -a 256 "$I3_D256" | cut -d' ' -f1)" \
  --stage-fit "$I3_F4/d1024-metadata-staging-03/inputs/d256-fit.json" \
  --fit-control "$I3_F4/d1024-metadata-staging-03/inputs/d256-control.json" \
  --dose 256 --series-id "$I3_SERIES" --reserve-allocation-sha256 "$I3_RESERVE" \
  --out "$I3_OUT/d256-collection.json"

nice -n 15 ./venv/bin/python -m research.bl_residual.collection_receipts collection \
  --ledger "$I3_D1024" --ledger-sha256 "$(shasum -a 256 "$I3_D1024" | cut -d' ' -f1)" \
  --stage-fit "$I3_D1024_STAGE_FIT" --fit-control "$I3_F4/d1024-pack-01/inputs/control.json" \
  --dose 1024 --series-id "$I3_SERIES" --reserve-allocation-sha256 "$I3_RESERVE" \
  --out "$I3_OUT/d1024-collection.json"

# Mac-only snapshot: if L4 work has charged any units, add --l4-ledger with its
# complete sealed R7 ledger. Never reuse the empty snapshot after charged tails.
nice -n 15 ./venv/bin/python -m research.bl_residual.collection_receipts projection \
  --d256 "$I3_OUT/d256-collection.json" --d1024 "$I3_OUT/d1024-collection.json" \
  --mac-projection "$I3_B2_MAC_PROJECTION" --out "$I3_OUT/d4096-projection.json"
```

The future D4096 fit pack must copy the receipt triplets into its input root and
bind their basenames under `stage_receipts` **and** their digests in `input_hashes`:

| Stage key | Artifact |
|---|---|
| `D256_COLLECTION` | I3 D256 collection receipt |
| `D1024_COLLECTION` | I3 D1024 collection receipt |
| `D4096_PROJECTION` | I3 D4096 unit projection |
| `D4096_MAC_PROJECTION` | I1 B2 Mac-h projection |
| `D4096_L4_ACCOUNTING` | Same sealed L4 R7 ledger, when supplied |

All existing predecessor fit/control and funding stage keys remain required.
Then the pack operator runs the existing metadata admission entrypoint, with
`I3_D4096_INPUTS` set to the freshly prepared fit-pack input directory:

```bash
: "${I3_D4096_INPUTS:?Set the prepared D4096 fit-pack input directory}"
nice -n 15 ./venv/bin/python -m research.bl_residual.job --preflight-main --input-dir "$I3_D4096_INPUTS"
```

No D4096 fit pack or release is created or authorized by this handoff. Production
receipt generation and Claude-V acceptance remain operator steps.

## Verification and review

Final verification used nice 15, all listed thread variables set to 1, no xdist,
and at most two Python processes including the CLI child. The explicit
`DYLD_LIBRARY_PATH` above prevents `conftest.py` from launching its macOS OpenMP
restart parent. Earlier exploratory runs passed but missed that waiting parent
and briefly had three Python processes in the CLI test; the operator was told
and both final selections were rerun with the corrected environment.

With the environment exports above and the worktree root as cwd, the exact final
commands were:

```bash
nice -n 15 ./venv/bin/python -m pytest -q -p no:xdist -p no:cacheprovider tests/test_bl_residual_collection_receipts.py
# 69 passed in 2.43 s, including both real receipt CLI commands on synthetic metadata.
nice -n 15 ./venv/bin/python -m pytest -q -p no:xdist -p no:cacheprovider tests/test_bl_residual_pipeline.py -k 'only_exact_judged_registration or all_scientific_controls or d4096_staging_refuses'
# 3 passed, 7 deselected in 0.70 s; no training or payload test selected.
nice -n 15 ./venv/bin/python -m black --check --line-length 100 research/bl_residual/collection_receipts.py research/bl_residual/job.py tests/test_bl_residual_collection_receipts.py
nice -n 15 ./venv/bin/python -m isort --profile black --check-only research/bl_residual/collection_receipts.py research/bl_residual/job.py tests/test_bl_residual_collection_receipts.py
nice -n 15 ./venv/bin/python -m flake8 research/bl_residual/collection_receipts.py research/bl_residual/job.py tests/test_bl_residual_collection_receipts.py
git diff --check
```

All three style checks and whitespace validation passed. Independent read-only
source review found no remaining blocking defect after fixing source-ledger
provenance and the L4 accounting omission case. This is an implementation review,
not the judgment's still-required Claude-V ACCEPT or a production B2 review.
The broad test suite was not run: verification stayed on synthetic metadata and
the existing metadata-only admission tests while the D1024 factory was live.
