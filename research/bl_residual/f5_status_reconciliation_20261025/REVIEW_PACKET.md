# F5 metadata-only ID reconciliation: revised acceptance packet

**SUBMITTED, NOT ACCEPTED. Real scan unrun.** Prepared in `snake-dqn-f5annex`, branch `sol-f5-scan-reconcile-20261025`, from clean baseline `02d29811`. This packet requests independent acceptance of revised pins and separate renewed one-scan authorization. Neither is issued here. The old explicit never-retry authorization does not transfer.

| Pin | File | SHA-256 |
|---|---|---|
| source | `research/bl_residual/f5_status_extractor.py` | `49f7cb0fcadc2b95569aabe6d2e1613d2f46c5ed5aec6c24f1cbeebf60a157df` |
| tests | `tests/test_f5_status_extractor.py` | `f1084cbbfb547a7e6d02f2328e9cef1930a7a0860be0d275be385ee25c382def` |
| contract | `research/bl_residual/f5_status_reconciliation_20261025/extractor-contract.json` | `2d9edb5ab598c6ce5ba61412e51757fcfde0b9a09552eebbebb55956e8254cb9` |
| inventory | `research/bl_residual/f5_status_reconciliation_20261025/extractor-input-inventory.json` | `f9e1de54440d892600a738839f48a70b7e5988ee75621c319bbf350cefb77589` |

Original ruling: `d6e2f029da689bbbe44e6c9c337ad989202a36af1fc26bcadad2f6f3be7e4fa6`. Astra refusal ruling (exact copy `ASTRA_SCAN_REFUSAL.md`): `a77f026a07052cf86abf4094d73703276d68d6529b19d0b116b1739b8eaf44b7`. Prior four-pin receipt/contract/inventory and all closure annexes remain unchanged; revised source acceptance uses this folder. The historical `f5_status_extractor.py.sha256` describes the prior source revision; it is deliberately not used by this package. Use `extractor-review-pins.json` here.

The historical `_table` at `research/bl_residual/f4_d256_package.py:141` appends retained rows, preserves original life IDs, and emits no independent life-presence/count table. A zero-ordinary-row sibling can disappear. The frozen 969-life inventory has no retained-row counts, and neither do its bound UNION ledger wrappers. This makes the explanation plausible, **not proven for the real file**. No original life JSON, real NPZ, D4096, or `t34-round0-study-01/` was opened during this work. No outcome values were decoded.

## Exact reconciliation policy

The parser still opens only the same eight metadata NPY members; all nine outcome/observation members stay unopened. All existing bounds, ancestry/fold exclusions, stratum caps, both-strata requirement, status projection, action enumeration and full U coverage are preserved.

It records exact sorted frozen and observed ID sets, missing and extra sets, and a classification for every difference. There is no canonicalization or alias creation. Extra IDs always refuse. Missing selected IDs always refuse, even if a count witness says zero. A missing ordinary-only sibling requires complete metadata count evidence over every frozen ID, explicitly ordinary=0 and doom=0 for that sibling. Evidence must obey the original per-life caps and agree with every observed life/stratum count. Nonzero counts, absent/malformed/incomplete/foreign evidence, count contradictions or any other difference refuse. The count evidence receives a canonical SHA-256. Zero-row omissions contribute no invented row, action or status; the frozen life denominator remains 969.

Evidence is embedded in the accepted input inventory as `retained_row_count_metadata`: schema `bl-f5-retained-row-count-metadata/v1`, exact container SHA, exact frozen original-life-source SHA, metadata provenance, and full `life_id -> {ordinary, doom}` counts. No provenance path is followed. Independent acceptance must establish the authenticity and original conversion correspondence of that metadata; a nonempty provenance string alone proves neither. Counts derived merely from missing IDs are forbidden.

**Current submitted inventory deliberately contains null evidence.** It preserves the exact original 960 selected + nine sibling bindings and canonical life-source digest. Available metadata cannot establish a real zero-row omission. If the later authorized scan finds missing siblings, this package records and refuses them. Genuine new count metadata requires a sibling inventory/contract/pin revision and independent acceptance before it can permit those omissions. This packet does not claim the real refusal is repaired by a proven real-data witness.

`Refused.reconciliation` carries the metadata sets/classifications and contradiction IDs. Redacted exception text carries only a fixed category. The operator CLI persists a structured REFUSED metadata report and returns 2. There is no retry loop.

## One-shot output and authority

The original reservation is untouched: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/F5_STATUS_SCAN_20261026T012513Z.json`; stat before/after shows size 0, inode 284427663, mtime 1792977954. Its sidecar remains absent. The fresh proposed output is `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/F5_STATUS_SCAN_RECONCILED_20261025_REVIEW01.json`; it has not been created.

The CLI reads only two fixed, bounded metadata receipt paths under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/f5-scan-reconcile-review01/`. The independent acceptance must use schema v2, decision ACCEPT_STATUS_ONLY_EXTRACTOR, a nonempty independent reviewer, both exact ruling hashes and the four current pins. Separately, renewed authorization must use schema bl-f5-renewed-scan-authorization/v1, decision AUTHORIZE_ONE_METADATA_SCAN, a nonempty authorizing identity, the same pins, the exact fresh output and refusal-ruling hash. The two `.template.json` files are intentionally invalid pending templates, not grants; neither real receipt exists.

Output creation is exclusive and uses no-follow directory FDs. Both CLI and API enforce the empty exclusive regular-file reservation at the exact output inode; the API writes/fsyncs RESERVED before source access. It cannot reuse that reservation after interruption or failure. Existing files, including an empty prior reservation, refuse. Final success/refusal metadata replaces only the new task-owned reservation. No cleanup or automatic retry occurs. The original immutable NPZ identity and source-descriptor checks remain enforced before/after projection.

## Verification and review

56 synthetic tests passed in 0.82 seconds; Black check, isort check and flake8 passed on both changed Python files. Tests ran nice 15, one math thread, no xdist or pool, at most the macOS isolation parent plus one child. Tests cover the full 969/960+9 case; absent/nonzero/malformed/foreign zero-row evidence; missing selected and extra IDs; exact count agreement; forbidden-member mutation and member-open traces; acceptance/renewal gates; create-only success/refusal output; repeated API use; and output ancestor substitution. The historical ceiling U=46,512 remains fully enumerated. All fixtures and synthetic receipts are OS-tempfile data. No real-input smoke was run.

```bash
nice -n 15 env PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ./venv/bin/python -m pytest -q -o addopts= -p no:xdist tests/test_f5_status_extractor.py
```

`IMPLEMENTATION_REVIEW.md` records an independent read-only implementation review. Two operator-gate findings were fixed and re-reviewed; final reviewed source SHA is `49f7cb0fcadc2b95569aabe6d2e1613d2f46c5ed5aec6c24f1cbeebf60a157df`. No remaining findings were reported within that bounded review. This is not scientific acceptance or authorization.

## Proposed exact command — NOT EXECUTED

Only after fresh independent acceptance and renewed authorization at the fixed paths below. With the current null count witness, any missing sibling remains a refusal; authorize a diagnostic reconciliation only if that limited result is intended. Never transplant the prior never-retry grant.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f5annex
nice -n 15 env PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ./venv/bin/python -m research.bl_residual.f5_status_extractor \
  --acceptance /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/f5-scan-reconcile-review01/acceptance.json \
  --authorization /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/f5-scan-reconcile-review01/authorization.json \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/F5_STATUS_SCAN_RECONCILED_20261025_REVIEW01.json
```

No scientific outcome, mechanism selection, label completion, fit, post-mortem, compute dispatch, serving change or adoption is authorized by this submission.
