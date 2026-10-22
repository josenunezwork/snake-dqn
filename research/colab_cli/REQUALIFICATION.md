# Telemetry/export stall requalification — pending Claude execution

The final published commit on **`supervisor-telemetry-20261022`** requires a new
source-bound `final-only` CPU + L4 acceptance before merge or production use.
The handoff names the exact SHA. Earlier qualified transport bytes do not cover
`renew_lease`, compact telemetry, the incremental export ledger, or bounded ZIP
export batches. No Colab operation was run while implementing this correction.
The incident and recoverability record is
[TELEMETRY_STALL_20261022.md](TELEMETRY_STALL_20261022.md).

Claude runs **`tools/ops/requalify.sh`**, which saves a review plan, consumes one
execution attempt, checks exact source/manifests, and updates the ops qualification
registry only after the existing final-only checks pass. These steps replace
the earlier direct acceptance commands; use this wrapper workflow. Keep the
existing **4.0-unit envelope / 0.8 reserve**, wall/unit caps and all ten required
acceptance gates from [ACCEPTANCE.md](ACCEPTANCE.md). Do not run the full-state
extension or a production retry as part of this correction.

1. Fetch this branch into the clean worktree below. Confirm `HEAD` and the remote
   branch equal the handoff SHA and `git status --short` is empty. Do not change
   any transport bytes between planning, acceptance and merge.
2. Run the complete offline CLI suite (including the 493/562-file virtual-latency
   regressions, running-to-terminal case, partial-export failure, archive
   integrity, caps and verified cleanup) with the project interpreter.
3. Prepare a reviewed, current endpoint-to-positive-rate map. `{}` is valid only
   for an actually empty account. Plan offline, inspect the printed
   `ops-review.json`, its exact manifests, source inventory and budget. Execute
   that saved plan once. A failed/interrupted attempt requires a new reviewed
   plan; do not rerun its execution marker or allocate an automatic replacement.

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn-telemetry
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
cd "$REPO"
git fetch origin
git status --short
git rev-parse HEAD origin/supervisor-telemetry-20261022
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -m pytest -q -n 2 tests/test_colab_cli_*.py
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
"$REPO/tools/ops/requalify.sh" plan --broker "$REPO" \
  --other-assignments "$ACCOUNT_MAP" \
  --state-parent /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/telemetry-requalification-20261022
# Review the printed saved plan, then set PLAN to that exact ops-review.json.
PLAN=/absolute/path/from/plan/ops-review.json
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

4. Independently review `acceptance-result.json`, CPU/L4 broker receipts and all
   ten observations. Require exact generation/source identity, positive rate
   calibration, the 67,108,864-byte verified artifact, original scanner and
   checksum/marker evidence, bounded telemetry recovery, and independent backend
   disappearance. `require_final_only` and `verify_plan_sources` must pass for
   these exact bytes. The live CPU probe must exercise at least one multi-file
   batch through `collect_exports`: verify its `.zip` transfer, every original
   payload/sidecar/marker and matching receipt ledger. If the fault sequence
   prevents this, it has not exercised the new batch path; record NOT_QUALIFIED
   and prepare a separately reviewed bounded acceptance attempt, never waive it.
   The hundreds-of-files scale proof is the offline regression, not a live
   throughput claim. Existing wall/unit caps still govern failed transfers.
5. Confirm the new registry's qualified commit/tree/result hash match the final
   commit and reviewed receipt. Only Claude may then perform the normal reviewed
   merge. Any subsequent executable transport change requires another acceptance.
   Keep original incident and acceptance attempts immutable. Scientific outputs
   from the two incomplete jobs remain incomplete; no study rerun is authorized
   by this code change.

## Family canary protocol

Every science dispatch now declares a **nonempty `required_exports` list** of
producer payload paths relative to `output/`. Each real producer must write the
payload plus `.sha256` and `.complete`; listing a path in the manifest does not
establish the contract. Both source preparation and validation use
`schema.validate_manifest` (including accounting's exact `_manifest_rates`
parser). Source preparation writes a create-only `schema-validation.json`
sidecar, which includes the normalized manifest digest and required exports.
Successful terminal admission repeats the export-inventory check.

Prepare an excluded, small workload from the science template with the same
entrypoint, full actual upload inventory, and target GPU. There are small
package generators:

| Package module | Real entrypoints retained |
| --- | --- |
| `research.bl_prefix.canary` | `run.sh` |
| `research.bl_info_value.canary` | `job.sh` |
| `research.bl_ppo.canary` | `colab.sh`, `p1_colab.sh` (distinct mode families) |
| `research.bl_e16.canary` | `colab.sh` (arm-bound family) |
| `research.bl_served_search.canary` | `run_l2.sh`, `run_ruler.sh` |
| `research.gpu_sim_spike.canary` | `colab.sh` |
| `research.t34_claim_a.canary` | `label.sh`, `pipeline.sh` |

Each module takes the same interface:

```sh
"$PY" -m research.bl_ppo.canary \
  --template /absolute/reviewed-science-template.json \
  --manifest /absolute/fresh-canary.json \
  --output /absolute/fresh-canary-output --job-id p1-canary-01 \
  --env-json /absolute/excluded-small-workload-env.json --timeout-minutes 30
# The generated manifest and .validation.json sidecar are review material.
"$LAUNCH" --state-dir /absolute/fresh-canary-state validate /absolute/fresh-canary.json
# Claude runs the reviewed canary through `run`, never direct provider commands.
```

The generator preserves all actual inputs and their hashes. `--env-json` changes
only normal package workload controls; it does not substitute a toy transport or
fake decision function. It does **not** invent a scientifically authorized tiny
bank, lower package admission bars, bypass fixed endpoints, or remove producer
exports. If a family has no lawful small input/phase, it remains NOT_READY until
its owner prepares one. Packages that fetch public artifacts still fetch their
real public artifacts. Include all relevant dtype fixtures, including `_paint`
paths for GPU simulator families, in the small real workload. A pass only covers
branches exercised by that canary; instrumentation is not exhaustive branch proof.

A canary must finish its whole session within 30 minutes, complete real staging
within **150 seconds** (half the 300-second staging cap), recover every declared
export through **`runtime._exports`**, and have verified backend release. The
science device is independent of the rented hardware: **only jobs declaring or
using CUDA require an observed CUDA action decision**. Explicit CPU-only jobs
on L4 (including DENSITY-2's twelve-vCPU host) retain every transport, scanner,
sidecar, source, budget and release gate, without a CUDA-decision requirement.

`canary.job_device` derives the device from the manifest's `device`, environment
`DEVICE` / `*_DEVICE` values (including `SNAKE_DQN_DEVICE`), and literal
`--device value` / `--device=value` arguments in environment values and the pinned
Bash entrypoint (simple manifest-bound `$VAR` / `${VAR}` arguments are resolved).
Unresolved local variables, defaults or computed device flags refuse; the gate
also refuses device arguments reassigned by the entrypoint and accounts for
explicit device environment assignments there. It never executes a shell to
guess device choices. Hardcoded `--device cuda` cannot
be hidden by a CPU environment default. Device choices inside Python remain an
explicit manifest contract, and observed CUDA use also requires attributable
CUDA decision proof. Values must
be `cpu` or `cuda[:index]`; missing, automatic (`auto`), unsupported, malformed,
or unresolved choices refuse before allocation. Multiple CPU and CUDA stages
make this a CUDA job. The manifest must declare all entrypoint device choices;
hardware availability never supplies a missing declaration. The saved canary
and production device classes must match, preventing a CPU proof from admitting
a CUDA dispatch.

Canaries install an observer in the pinned venv: a CUDA job's real package must execute
an action-space `argmax` or `multinomial` on CUDA and synchronize it successfully.
Hardware availability, a flag saying `cuda`, a generic warmup, CPU decisions, or
a manually written boolean do not qualify as CUDA evidence. CPU action selection
does qualify CPU science on rented GPU hardware when all other evidence passes.

The broker writes `canary_evidence` inside its normal create-only receipt only
after all these checks. A science manifest binds that receipt as:

```json
{
  "required_exports": ["receipt.json"],
  "canary": {
    "receipt": {
      "path": "/absolute/fresh-canary-output/receipt.json",
      "sha256": "<SHA256 of that receipt>"
    }
  }
}
```

`validate` re-derives the proof from the saved manifest, measurements, recovered
exports and receipt. It binds entrypoint + operational mode + source commit +
complete transport executable inventory. It refuses changed code, incomplete
release, missing sidecars/scanner evidence, missing CUDA decisions for CUDA jobs, ambiguous
devices, CPU-canary/CUDA-production substitution, uncovered
export paths, more files, larger files, changed input hashes/destinations/source paths or a different
hardware target. Every allowed GPU target needs its own canary; use a single
qualified target per manifest. Canary runs and recovery attempts still spend
from the package's registered envelope; this feature grants no new budget.
Previously qualified manifest syntax and fixed acceptance probes remain usable,
but science manifests must acquire this new evidence before dispatch. No frozen
T34 package or existing manifest bytes were rewritten to supply it.

## Local verification and other process changes

Run the complete local CLI suite with the project interpreter:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$PY" -m pytest -q -n 2 tests/test_colab_cli_*.py
```

The new offline tests cover map-vs-list accounting, nested budgets, missing
required exports, separate failure classes, real-scanner canary evidence,
transport volume/count and source/hardware changes, and CUDA observation
attribution. These are local regressions and do not prove live transport.

Mac timing and identity entrypoints use `research.mac_lease`. All worktrees
share the primary checkout's `exp/mac-lease.json`; acquisition is atomic and
release requires the same owner and lease ID. The receipt includes the lease.
Expired records may be replaced; operators must price the lease duration for
the whole work and retain existing job-level time bounds. This helper does not
stop an arbitrary nonparticipating process.

Registered BL decision CLIs now require `--authoring-repo` and `--ledger`
(repository-relative). `research.published_ledger.require_published_ledger`
checks the clean authoring checkout and freshly fetched `origin/main` ledger
bytes **before reading result inputs**. Analysis receipts include the proof and
its SHA256. Publish the ledger first, then run analysis from a clean authoring
checkout. Worker computation remains separate: P1 evaluation exports pairs for
Mac analysis, and compute-only E3/prefix reports need their `decide` wrapper to
publish a claim. Frozen `t34_round0*`, `t34_pathb`, `t34_exit_b8*` and
`t34_round0d` analysis code is excluded from this change.

E6/E7 raw production exports retain the exact expected world rosters from the
audited bank. The fenced wrapper uses those rosters for the original completeness
checks; E7 selection rows contain three ancestors per world and cannot substitute
for that roster. Older exports lacking the roster fields refuse the new wrapper.

## Historical result (2026-10-22 ~02:30 PT, Claude operator)
final-only acceptance on e9ce99f2: **PASS, qualified=true** — receipt
`snake-dqn-artifacts/colab-cli-process-gates-acceptance/acceptance-1792660644/acceptance-result.json`
(CPU + L4 jobs; final artifact 67,108,864 bytes, Mac-verified, ≥1.58 MB/s). Merged to main.
That historical PASS does not qualify this branch's telemetry/export correction.
Current result: **PENDING — no Colab operations performed for this correction.**
