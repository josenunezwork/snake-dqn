# Verified package metadata archives

Prepared and offline verified. No Colab allocation, upload, fit, strict gate,
serving change, or release was performed. Operator dispatch remains separate.

The package change implements recommendation 1 from
`snake-dqn-artifacts/t34-sol/uploaddiag/UPLOAD_CAP_DIAGNOSIS.md`.
`research/metadata_archive.py` packages local files no larger than 2 MiB into one
deterministic tar with an internal per-file SHA256/size manifest. Larger payloads
remain direct inputs. The transport-required `public-inputs.json` remains a direct
row, selected by its destination rather than the source file's basename.

The package launcher verifies the bound outer SHA, exact member inventory,
per-file size/SHA, safe nested destinations, and intact export sidecars/markers
before publishing verified members. Existing destinations and links are refused;
completion markers publish last. Original package readers then repeat their
ledger/allocation/sidecar checks. Re-preparation may reuse an existing complete
archive only after a deterministic rebuild proves identical bytes; no archive,
input, ledger, or historical evidence is overwritten.

F4's original `d256-pack-02`, source `c2c82ff2`, inputs and old device-fix runbook
are retained. All 16 reconstructed input hashes match that pack. FitRecipe,
registration, protocol, fit implementation, seeds, labels, updates, and scientific
source bytes are unchanged. Registration SHA256:
`66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65`;
FitRecipe SHA256:
`49d12f486bffa455adcedbc318e88f1524f5c502894d0c1ace47daa84b9d3a5a`.

## Frozen source and artifacts

Every final source pin retains qualified transport tree
`d0d92bdb572b770284a1fea6152417ffa5f06bea` unchanged from its prior family pin.
No transport requalification was needed or performed. Main's separate transport
work arriving from concurrent upstream commits was preserved during integration;
these commands use the qualified family checkouts below.

| Family | Full frozen pin | Qualified checkout |
| --- | --- | --- |
| F4 | `6313fe8fce990915ef1c3702ff1687169513b624` | `/Users/josenunez/Projects/ml/snake-dqn-f4-metadata-qualified-02.noindex` |
| V8 split | `669405e2bef629f7f69bc951cf49486d8b0a87bf` | `/Users/josenunez/Projects/ml/snake-dqn-v8-metadata-qualified-02.noindex` |
| MOVE2 | `fe404912436fe2f3f01db78f4a1778fa176ac056` | `/Users/josenunez/Projects/ml/snake-dqn-move2-metadata-qualified-02.noindex` |
| Sparse | `b7dfa1fc35321a388813122b3acfe033d1367782` | `/Users/josenunez/Projects/ml/snake-dqn-sparse-metadata-qualified-02.noindex` |

The MOVE2 checkout has the standard ignored `venv` symlink to the primary
project environment, as required by its generated dispatch scripts; its Git
checkout remains clean.

Artifact paths below are relative to
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/`.

| Family | Final package | Local input rows before/after | Planned stages before/after | Archived members | Tar bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| F4 | `f4/d256-metadata-pack-02` | 16 / 3 | 18 / 5 | 14 | 1,300,480 |
| V8 split | `gpu-v8-metadata-20261024-02` | 11 / 2 | 13 / 4 | 10 | 163,840 |
| MOVE2 | `gpu-move2-metadata-20261024-02` | 10 / 2 | 12 / 4 | 9 | 266,240 |
| Sparse canary | `gpu-sparse-metadata-20261024-02/packet` | 8 / 2 | 10 / 4 | 7 | 30,720 |

Planned stages include `bootstrap.sh`, `source.py`, and local input rows; they
are not observed provider transfer counts. Each job uses one metadata tar.
F4 retains B8 and the training NPZ as two direct payloads. Each GPU package
retains the public checkpoint catalogue as its other local row. The existing
17,876-byte sparse exactness fixture is inside its tar.

Sparse also retains the prior source-size repair: compiler source fallback
137,321 bytes (48 files, 1 MiB bound), sparse fallback 1,544,023 bytes (424 files,
64 MiB bound), including the unpack helper. Its compiler-only prerequisite is
prepared at the new same pin, with its existing caps and proof checks.

## Completed evidence

Final verification ran serially with one numerical thread, at absolute nice 15 for the
verification driver and pytest processes. The existing MOVE2 scripts add their
own `nice` wrapper, lowering that child priority further. No live Mac job or
its source checkout was modified.

- Focused package/helper tests: 66 passed in 0.78 s. Final qualified-source
  helper/F4/supervisor/schema/canary/qualification suite: 203 passed in 443.40 s.
  Black, isort, flake8, Bash syntax, and diff checks passed for changed files.
- Exact production archives replayed through the real `python -m
  research.metadata_archive unpack` CLI on the Mac, followed by the unchanged
  package ledger readers. F4 also passed the real metadata-only
  `research.bl_residual.job --preflight-main`; no real fit or reserve look.
- Final-pin GPU real-entrypoint preflights completed scripted/frozen/mixed E1
  units through archive unpack: V8 20.716 s, MOVE2 12.981 s, sparse 7.306 s.
  These are excluded CPU plumbing checks, not CUDA qualification or results.
- `supervisor.validate` passed all four final canary/pending manifests and the
  sparse compiler manifest. F4/V8/MOVE2 main templates returned the expected
  missing-family-canary HOLD. Sparse's package guard returned the expected
  missing-native-compile HOLD. No proof was substituted.
- Every actual source fallback inventory includes the unpack helper. All four
  qualified Git trees and manifest input hashes were checked.
- The exact generated V8 and MOVE2 dispatch preparation prefixes were rerun
  without the broker exec. Both passed, and every artifact file hash remained
  unchanged. `REPREPARE_VERIFICATION.json` records the checks.

Consolidated source/input checks and archive identities:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/metadata-archive-20261024-01/FINAL_VERIFICATION.json`.
The verification driver and complete command logs live beside it. VM replay
reports are `f4/d256-metadata-verification-02/verification.json` and
`archive-verification/verification.json` under each GPU artifact root.

The initial `*-01` metadata preparation artifacts and pins are superseded,
retained as diagnostic evidence. Their integration check exposed the catalogue
source-name/destination distinction before dispatch. Only the `*-02` artifacts
and pins above are the operator packages.

## Operator commands

Each line is a separate operator action. Wait for terminal PASS and verified
runtime release before the dependent line. The unchanged wrappers refresh live
accounting/rates, enforce caps, and refuse absent proof. These commands were
prepared, not dispatched.

F4 canary, using the already-prepared package (do not rerun `repack` into it):

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-f4fix/research/bl_residual/RUN_D256_NEXT_METADATA_20261024.sh canary
```

After its same-pin family canary PASS/release:

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-f4fix/research/bl_residual/RUN_D256_NEXT_METADATA_20261024.sh seal-main
bash /Users/josenunez/Projects/ml/snake-dqn-f4fix/research/bl_residual/RUN_D256_NEXT_METADATA_20261024.sh fit
```

V8 split:

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-metadata-20261024-02/dispatch-canary.sh
# Only after this same-pin canary passes and releases:
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-metadata-20261024-02/dispatch-main-after-canary.sh
```

MOVE2 main has a new package pin and therefore needs a new same-pin canary:

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move2-metadata-20261024-02/dispatch-canary.sh
# Only after this same-pin canary passes and releases:
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move2-metadata-20261024-02/dispatch-main-after-canary.sh
```

Sparse's compile prerequisite first, then the canary after its verified PASS:

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-sparse-metadata-20261024-02/packet/dispatch-compile.sh
# Only after the same-pin compiler job passes and releases:
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-sparse-metadata-20261024-02/packet/dispatch-canary.sh
```

If the qualified dispatcher's existing accounting retry produced the compile
receipt in `output-retry-1`, explicitly set `NATIVE_COMPILE_RECEIPT` to that
receipt on the sparse canary command. The binder verifies its identity and
refuses changing an already-bound canary. There are no automatic workload
completion retries or fabricated compile/canary receipts in this preparation.

No live-upload latency improvement, actual GPU fit, benchmark result, strict
pass, serving qualification, or release is claimed. Fewer file-control calls
reduce exposure but do not eliminate stalls on bootstrap/source/archive calls.

Exact final qualified regression command (the wrapper sets absolute priority 15
and all numerical thread limits to 1 before exec):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f4-metadata-qualified-02.noindex
python3 /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/metadata-archive-20261024-01/run_nice15.py \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist \
  tests/test_metadata_archive.py tests/test_bl_residual_f4_d256_package.py \
  tests/test_colab_cli_supervisor.py tests/test_colab_cli_schema_failures.py \
  tests/test_colab_cli_canary.py tests/test_ops_qualification.py
```

Full output: `metadata-archive-20261024-01/final-qualified-tests.log` beneath the
`t34-sol` artifact root above. The package preparation commands are preserved in
`prepare02.py` with individual `*-02` command logs beside it.

Independent Codex-V LIGHT: `PASS_PREPARED_ONLY`, zero open findings. Tracked
receipt: `tools/ops/CODEX_V_METADATA_ARCHIVE_20261024.json`, SHA256
`a656cbfabddeebbeb90048458eb03ae19e969fd8831fe19d78fc0b98b4f33ef3`. This is a byte-identical copy of the reviewer-authored
`metadata-archive-review-20261024-01/CODEX_V_LIGHT_REVIEW.json` under the
`t34-sol` artifact root. The review binds the final pins, manifests, archives,
VM verification, regression log, and repeated-preparation evidence.
