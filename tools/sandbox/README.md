# Standing A100 Development Sandbox

Authority: [owner-adopted rule](../../docs/research/governance_a100_sandbox_20261026/ADOPTED.md), adopted 2026-10-26. Jobs require automated preflight and a committed ten-line manifest. No per-job judge, registration or timing canary is needed. Claude dispatches through the existing qualified broker. Preflight never calls Colab. The host-only admission adapter invokes the unchanged qualified supervisor only through dispatch.

## Initial input registry

`SANDBOX_INPUT_REGISTRY.json` is an exact positive allowlist. It binds metadata sources by SHA256, the selected source-life IDs, ultimate parents, immutable F0 payload paths/sizes/hashes, and filtered NPZ hashes. All unlisted lives, banks and payload hashes fail closed. Outcome labels are admitted only as development training data.

| Selection | Before exclusions | Admitted lives | Admitted parents | Admitted rows |
|---|---:|---:|---:|---:|
| D256 prefix | 192 | 136 | 31 | 12,512 |
| D1024 including prefix once | 960 | 730 | 291 | 65,825 |

Source ledgers contain 3,024 allocated parents. The original F4 hash-fold rule excludes 167 fold-zero parents. The census receipt lists 29 parents (already among fold zero), and the frozen R30 inventory lists 81 validation parents. The exclusion union is 248 parents; 230 of the 960 selected fit lives are removed. Strict, probe, serving, F5 evaluation, natural archive, unused tails, D4096 and protected round-zero banks are categorically excluded by the positive closed-fit allowlist. Their payloads are never opened to enumerate them; this is not an exhaustive historical seed audit. No F5 package, state, evaluation world or output is consumed.

Sources are the published D256/D1024 final union ledgers; non-F5 `metadata-v3/corpus-inventory.json` from the closed R30 corpus; the F4 registration; the R30 post-F4 ruling; and `research/bl_residual/f4_postmortem_signal_fix_20261025/development-parents.json`. The exact source paths and full hashes are recorded in the registry. `build_registry.py` checks the original immutable pins before deriving lists from metadata alone. `prepare_input.py` reads only the admitted life files, checks their ledger size/hash/completion/wrapper identity, and preserves every ordinary and rescue row using the existing F4 adapter. It never opens a combined partition containing excluded parents.

Generated NPZs remain outside Git under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-sandbox-b-20261026/inputs/`. Preserve the published registry and original inputs. A new dataset requires a new registry revision and preflight. Rebuilding uses a fresh output path:

```sh
./venv/bin/python -m tools.sandbox.build_registry --out /tmp/sandbox-registry-metadata.json
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  nice -n 15 ./venv/bin/python -m tools.sandbox.prepare_input --dose D1024 --output /absolute/fresh/d1024.npz
```

The metadata-only builder emits the membership portion. The projection receipt printed by the fixed selector supplies the registered dataset `path`, `sha256` and `row_count` fields, which must be committed before use. Never admit an arbitrary NPZ by assigning it a hash: derive it with the fixed selector first. `validate_npz` checks registered bytes before reading lineage, and `validate_rows` checks exact admitted life/parent membership and row identities.

## Worlds and future studies

`world_seed(study, idx)` takes the first eight bytes of SHA256 of UTF-8 `a100-dev/<study>/<canonical decimal idx>`, interpreted as unsigned big-endian uint64. Study slugs are canonical lowercase names. Do not truncate to uint32, reduce modulo a generator limit, or rename an existing parent. Explicit world records must match those derived seeds exactly and avoid the registry's closed-ledger seed set. Extend the canonical exclusion registry before new development allocations. Historical confirmation-bank collision audits remain the responsibility of that future registered crossing; hash salts alone do not certify historical freshness.

## Preflight and manifests

```sh
./venv/bin/python -m tools.sandbox.preflight research/sandbox_b_heads/sandbox-manifest.json
```

The manifest has exactly ten physical JSON lines with ten top-level fields: schema, study, inputs, salt, endpoint, timeout, excluded_studies, owner_rule, outputs, crossing. `manifest.schema.json` describes the machine contract. Inputs carry registry kinds and exact paths/hashes. The broker packet is bound by hash under `outputs.broker_manifest`. Timeout is exactly twice its declared base. New packets use an explicit `runaway-guard` base rather than claiming a CPU smoke predicts A100 duration. The original measured-estimate spelling remains accepted for existing Job A packets. Outputs carry `simulator-D`; crossing is capped at one frozen checkpoint SHA. Per-job scientific selection and execution parameters live in the source-bound broker packet and job code.

Preflight verifies committed manifest/registry bytes, source pins, input membership, broker consistency, unchanged qualified transport tree `5833e54e9c5396121cbbf99a5cd4e52006e564fd`, and focused tests at nice 15 with single math threads and no xdist. `--skip-tests` is a diagnostic check, never an admission PASS. No API token, live assignment inventory or provider operation is needed for offline preflight. Dispatch remains responsible for fresh account-wide capacity, registered-study coexistence and separate assignment isolation.

Exports are development evidence at most. A frozen candidate crosses into Mac evaluation only through a separately committed registration, judge review and fresh paired bank. Mac strict gate, serving qualification and owner release authority remain unchanged.

## Dispatch admission

A broker packet declares `sandbox_admission` with schema
`standing-a100-development-sandbox-admission/v1` and an absolute `manifest_path`
to its committed ten-line sandbox record. That record binds the original broker
packet by SHA256; the reverse reference has no digest, avoiding a hash cycle.
No `canary` field or `SNAKE_COLAB_CANARY*` instrumentation is permitted.

`tools/ops/dispatch.py` selects `tools/sandbox/launch.sh` for this explicit route
in both validation and execution. The launcher preserves AC power, open-lid,
`umask 077` and `caffeinate -is` guards. Its process-local adapter replaces only
the family receipt requirement with full committed sandbox preflight and tests.
It checks sandbox input paths before the qualified validator hashes any bytes, and retains transfer policy, explicit device and required-export checks. The
qualified supervisor continues to own all allocation, accounting, lease, export
and verified-release behavior; `research/colab_cli` remains tree `5833e54e`.
Ordinary jobs use the original launcher and original canary checker unchanged.

Admission refuses unknown/registered controls, unlisted inputs, protected paths,
foreign study scripts, unpinned source bytes, and changed execution parameters.
Outputs stay under the designated `a100-sandbox-b-20261026` or `sandbox-a-resident` artifact root, preventing writes into registered-study namespaces. Only documented supervisor defaults, fresh live assignment rates, and the
existing single exact accounting-allocation retry may differ from the sealed
packet. The live dispatch path always uses a separate new assignment. It waits
if an active broker record names F5 or another protected study: this route does
not infer F5 coexistence permission from available assignment slots. A dry-run
does not inspect live account capacity or prove coexistence.
