# R30-1 URL checkpoint and metadata upload repair

Prepared and dry-run only. No dispatch, allocation, stop, live account query,
scientific fit/look, gameplay, serving change or release was performed.

Frozen runtime: `98cc98a779df5b11bcb79d7dc40860d9976d2608`, checkout
`/Users/josenunez/Projects/ml/snake-dqn-r30-l4-urlbundle-02-20261024`, branch
`t34-r30-l4-urlbundle02-20261024`. The runtime is published in origin/main's
ancestry. Its qualified `research/colab_cli` tree remains
`d0d92bdb572b770284a1fea6152417ffa5f06bea`; main's newer transport was preserved.

`pack.py` replaces local inputs above 1,000,000 bytes with exact size/SHA-bound
HTTPS rows only when their identity matches an existing public release asset in
`research/t34_exit_b8/PUBLIC_INPUTS.json`. B8 uses the existing `s1-inputs-20261019`
release URL, preserving destination `b8.pth`, 6,562,005 bytes and SHA256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
The unchanged qualified supervisor verifies the URL on Mac before allocation and
again on the runtime before the package starts. Only B8 is fetched; no unused
league checkpoint archive is needed. Unknown/private inputs above 1 MB refuse
preparation rather than being uploaded or published.

The remaining six control/ledger/sidecar files become one deterministic
10,240-byte `metadata.tar`. The existing `research.metadata_archive` helper
verifies the outer SHA, exact inventory and every original size/hash, rejects
paths/links/conflicts, and publishes completion markers last. `colab.sh` unpacks
before the unchanged worker runs. Local input rows drop from seven to one;
planned Contents stages including bootstrap/source drop from nine to three.
There is also one public URL row. No transport constants/caps changed.

## Qualification evidence

An active resumed W6 D1024 factory prevents the existing exclusive numerical-test
launcher from admitting a fresh test run. R30's older ordinary-window launcher
also refuses this resumed attempt identity. Neither guard nor the factory was
changed. The independent Codex-V light reviewer recommended a separate mechanical
proof rather than relabelling the old numerical PASS as current execution.

`packaging_check.py` accepts only the fixed immutable c1bf44c8 numerical preflight
SHA `1eeaf536bb67176201bcef70cb1316516a18784a05a9cbb643391220ed59b193`.
It verifies that source against its original Git pin and permits only four
mechanical paths over that base: pack, launcher, the new checker and the launcher
test fixture. Every other original source binding and the shared archive/catalog
must remain byte-identical. The new proof has its own
`r30-packaging-preflight/v1` schema, exact current source identity and
`synthetic_only:true`. Real scientific inputs still require the original full
current-source Mac preflight. No main is admitted by the new adapter.

Final proof: **PASS, 9.5568 seconds, nice15, 260 bound source files**, zero numerical
imports or numerical tests. It invokes the real archive CLI, verifies all original
metadata bytes and export triples, and checks repeat extraction, bad archive SHA,
unknown private large input, foreign/stale proof, and scientific-main refusal.
The CLI has a 120-second overall alarm. Historical numerical evidence is the
previous 78-test PASS, explicitly inherited and not rerun. The updated real Bash
fixture is preserved for the next admitted numerical test window. No new CUDA
consumer or remote staging success is claimed.

Exact final proof command, run from the frozen checkout:

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.bl_round30.packaging_check \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/sol-verification/urlbundle-98cc98a7-20261024-01 \
  --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
```

That create-only proof is already present; do not rerun into the same destination.
The earlier candidate fcc87807 and its log are retained as diagnostic evidence:
its local replay used macOS's symlinked `/var` temp path, which the archive helper
correctly refused. Final replay resolves the temporary root to its physical path.

Final package directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/qualified-98cc98a7-urlbundle-r301-01`.
It contains immutable `job.json`, `metadata.tar`, `PUBLIC_INPUT_PREFLIGHT.json`,
`UPLOAD_QUALIFICATION.json`, `DISPATCH.sh`, `DISPATCH_DRY_RUN.json` and
`PREPARATION.json`, with their SHA/completion sidecars.

Public-input preflight streamed and verified all 6,562,005 B8 bytes from the actual
HTTPS URL. Source/upload qualification returned `PASS_NOT_DISPATCHED`: one local
input; 27,995 normal upload bytes including bootstrap/source; 65.6293-second normal
and 90.8705-second fallback forecasts. These are modelled forecasts, not measured
Colab staging times. The actual pinned sparse-source closure was built locally.
Black (`--check --workers 1 --line-length 100`), isort (`--check-only --profile black`),
flake8 (`--jobs 1`), Bash syntax for the launcher and generated command script,
and `git diff --check` passed. No pytest/full suite was run in the live factory
window. Independent Codex-V report is the sibling
`round30_upload_codex_v_light_20261024.json`.

L4 caps remain estimate 5 minutes, timeout 20 minutes, 2/3 unit, rate 1.54 units/h,
reserve 1/9 unit, max export 128 MiB and default lease 180 seconds. The short-canary
exemption remains. Both earlier attempts are INCOMPLETE, stop_verified=true and
zero exports; their immutable receipts remain intact. Their combined priced
elapsed liability is 0.3064836622 units at 1.54 units/h, not a provider-debit
measurement. Future scientific accounting must retain all prior failures.
Live account assignments/balance are checked by the operator wrapper at dispatch.

## Exact operator command

The command below passed with an additional `--dry-run`:
`PASS_DRY_RUN r30-1-rehearsal-urlbundle-98cc98a7-01 allocations=0 live_accounting=NOT_CHECKED`.
All new runtime-output/state/log/manifest/reservation destinations remained absent.
It was never executed without `--dry-run`.

```bash
/Users/josenunez/Projects/ml/snake-dqn-r30-l4-urlbundle-02-20261024/tools/ops/dispatch.sh r30-1-rehearsal-urlbundle-98cc98a7-01 /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/qualified-98cc98a7-urlbundle-r301-01/job.json --broker /Users/josenunez/Projects/ml/snake-dqn-r30-l4-urlbundle-02-20261024 --cwd /Users/josenunez/Projects/ml/snake-dqn-r30-l4-urlbundle-02-20261024 --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/broker-r301-rehearsal-urlbundle-98cc98a7-01
```

No tmux/process/lease, released checkpoint, serving pin, prior frozen checkout,
or hash-bound historical receipt/document was modified.
