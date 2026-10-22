# Process-gate requalification — pending operator execution

This branch (`gate-cuda-scope-20261022`) changes executable `colab_cli` code. The earlier qualification in
`QUALIFICATION.md` does **not** qualify these bytes. No Colab allocation, canary,
or live acceptance was run during implementation. Merge remains blocked until
Claude runs and independently reviews the source-bound **final-only** acceptance.
Do not replace the existing qualification hashes with current hashes without that
PASS receipt. Do not run the full-state extension for this change.

## Exactly which acceptance gates to rerun

Run the complete `final-only` CPU + L4 acceptance once on the final published
branch commit, within the existing **4.0 unit envelope, 0.8 reserve** and its
existing first-wall-or-unit caps. Preserve every create-only attempt receipt.
The required gates from `ACCEPTANCE.md` are:

1. `account-assignment-spend`: pre-existing endpoint-to-rate map and separate
   elapsed-rate charge / other-assignment baseline.
2. `rate-calibration`: stable positive baseline-subtracted CPU and L4 observed
   rates, within the priced ceiling and tolerance.
3. `fresh-oauth-refresh`: fresh terminal identity and usage without a browser prompt.
4. `silence-timeout`: bounded timeout after a gap greater than 60 seconds,
   without re-submitting uncertain work.
5. `url-no-second-allocation`: sessions before/after browser attachment identify
   the same assignment.
6. `client-loss-reconnect`: same generation and remote job, one completion record.
7. `failure-and-corrupt-export`: intentional child failure and rejected corrupt
   download, with the new distinct `failure_class` and last traceback line.
8. `final-artifact-download`: exactly **67,108,864 bytes**, real scanner sidecars,
   verified recovered hash, measured transfer seconds and goodput lower bound.
9. `mac-loss-watchdog-unassign`: L4 lease expiry and backend disappearance within
   300 seconds, before broker fallback cleanup.
10. `backend-stop-disappearance`: independent sessions proof of exact endpoint removal.

`busy-full-snapshot`, `exact-resume`, and `checkpoint-overhead` remain
`NOT_REQUIRED`, not observed passes. Acceptance mode uses its fixed probe
entrypoint and is exempt from the science-family canary prerequisite so it can
bootstrap qualification. It cannot generate a science-family canary receipt.
The acceptance program, not a hand-edited receipt, selects its actual gate IDs.

Claude must requalify the final published SHA reported in this branch's handoff,
before merging it to main or dispatching DENSITY-2. Use that exact clean checkout;
the old main acceptance does not cover the device-rule change. Do not edit any
transport Python/shell bytes between acceptance and merge. From that checkout,
use the launcher and a fresh state parent:

```sh
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
REPO=/Users/josenunez/Projects/ml/snake-dqn-gatefix
cd "$REPO"
git fetch origin
git status --short                    # must be empty
git rev-parse HEAD origin/gate-cuda-scope-20261022  # both must equal handoff SHA
LAUNCH="$REPO/research/colab_cli/launch.sh"
STATE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-cuda-scope-requalification-20261022
# Prepare a reviewed JSON endpoint-to-positive-rate map from the live inventory.
# {} is valid only when that inventory is actually empty.
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
"$LAUNCH" acceptance --profile final-only \
  --repo "$REPO" --state-dir "$STATE" --other-assignments "$ACCOUNT_MAP" \
  --colab /Users/josenunez/.local/bin/colab
# Review the generated plan and its current account map first. Claude then runs:
"$LAUNCH" acceptance --profile final-only \
  --repo "$REPO" --state-dir "$STATE" --other-assignments "$ACCOUNT_MAP" \
  --colab /Users/josenunez/.local/bin/colab \
  --execute-after-review --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

Independently review the new `acceptance-*/acceptance-result.json`, its bound
CPU/L4 broker receipts, all ten required observations, the recovered 64 MiB
artifact hash, positive rate calibration, and exact backend disappearance.
`require_final_only(result, REPO)` and `verify_plan_sources(...)` must both pass
against these exact bytes. Retain the previous qualification and attempt files.
No family-canary PASS or scientific admission is implied by this acceptance.

After a reviewed PASS, update qualification by the established source-bound
procedure and merge only through the operator's normal review. A later Python
or shell change in `research/colab_cli` invalidates this acceptance again.

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
That PASS predates this branch's executable device-rule change and does not qualify it.
Current result: **PENDING — no Colab operations performed for this correction.**
