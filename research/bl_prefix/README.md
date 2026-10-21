# BL prefix experiments E6 and E7

Development experiments only, excluded from every registered T34 verdict.
No result here authorizes a release, serving change, or automatic training run.
The frozen authorities are `REGISTRATION_E6.md`, `REGISTRATION_E7.md`, and their sibling
identity/allocation/compatibility addenda. They were published before any gameplay.

E6 measures whether 64 candidate decisions, taken 100 decisions before a late
B8 death, improve remaining H5000 mass after B8 resumes. It includes ordinary
surviving roots as a collateral guard. E7 searches the same death worlds at
100/150/200 decisions before death with eight stochastic prefixes, selects on
training RNG, and confirms with two independent paired future RNG streams.
Both use the shared T4b full-state restore path and actual 128-decision history.

The bank has 128 death worlds (44 scripted /42 frozen /42 mixed), 128 matched
ordinary worlds, and all three E7 ancestors. Matching uses distinct surviving
worlds of the same mix at the same frame, within 100 length cells. Matching or
history exhaustion gives INCOMPLETE. Source collection stops at 4096 worlds.
The source seed audit must pass before collection; the sealed live artifact
directory is pruned before traversal. Published bank identities cover registered
worlds without reading sealed outcomes. Opaque or unrecorded runs remain an
explicit audit limitation.

## Entry points and inputs

Run from the repository root using the primary checkout's existing Python:
`/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`.

- `python -m research.bl_prefix.seeds` creates the Mac collision inventory receipt.
- `python -m research.bl_prefix.e6 --plan PLAN --out FRESH_DIR [--preflight]`
  runs E6 through the real restoration and continuation code.
- `python -m research.bl_prefix.e7 --plan PLAN --out FRESH_DIR [--preflight]`
  runs E7 through that same code.
- `python -m research.bl_prefix.make_manifest --experiment e6|e7 --plan PLAN
  --output-dir FRESH_ABSOLUTE_MAC_DIR --out MANIFEST --commit FULL_SHA`
  creates a locally validated, immutable supervisor manifest. Optional
  `--input DEST=PATH` binds additional uploaded files, and `--estimate-minutes`
  must fit the frozen budget with a two-times margin.

A runtime plan uses `input:filename` for supervisor-uploaded paths. The dispatcher
resolves these beneath the upload root and refuses traversal and Mac/Drive
paths. A plan with `ready: false` is a reviewable binding template: the runner
refuses it before loading gameplay or creating outputs. Local structural
manifest validation is distinct from scientific input readiness.

Scientific E6 requires real final R/F E2 checkpoints and their hash-bound
completed-training/critic receipts. The critic gate is >=2M refit transitions,
held-out late-body EV >=.1, and MSE below the time/length baseline. B8 fixtures
are permitted only in explicit preflight mode. Scientific E7 requires the
complete E6 source bank and never silently recollects a replacement bank.

## Transport and resource boundary

`run.sh` is the child script for the unchanged qualified `research/colab_cli`
supervisor. It uses Python 3.12/bootstrap dependencies and CPU policies on an
L4 assignment, with one native math thread per worker. Scientific runs use exactly
12 processes and deterministic merge order. Local preflight uses nice10 and
at most two single-thread processes. No launcher in this package allocates or
calls the Colab CLI. This build does not dispatch.

Every export is create-only payload + `.sha256` + `.complete`, with the marker
written last. The tests exercise the actual supervisor export discovery and
mutation fence. Each export is bounded at 64MiB by the manifest; split larger
collections into separately completed files. This uses the qualified final-only
transport profile and makes no exact trainer-resume claim.

E6: 4u total, .8u reserve, 2u/h planning ceiling, 96min usable session timeout.
E7: 20u total, 4u reserve, 2u/h ceiling, 480min usable timeout (10h absolute
registration ceiling). Setup, JIT, collection, primary, secondary, confirmation,
export, idle and failures all count. The first wall/unit cap wins. No retry.
Account-wide assignments remain capped at three; the operator must bind the
current reviewed assignment/rate inventory before any future dispatch.

The forecast counts phases that the judgment's headline estimates omit:
37.12M maximum E6 world-steps including all 4096 source worlds, selected-source replay and the secondary
protocol; 24.96M maximum E7 world-steps including all three ancestors, eight
training continuations, paired training B8 and paired confirmation. Root-specific
remaining horizons and early terminal deaths can lower actual work. A measured
forecast that cannot fit must stay NOT_DISPATCHABLE; the package never shrinks
N or drops required protocols to make a cap fit.

## Verification and handoff

Targeted package tests only, plus selected offline supervisor validation/export
tests. Do not run the whole suite for this isolated package. See `PREFLIGHT.md`
for measured Mac numbers and limitations. Concrete plans, manifest paths,
forecasts and operator handoff are stored outside Git at:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e6e7/DISPATCH.txt`.

Never edit the protected T34 packages, transport implementation, serving pins,
released checkpoints or frozen registration bytes. Add a sibling pre-data
addendum if a rule must be clarified.

## E10 population bridge (registered before new gameplay)

`REGISTRATION_E10.md` adds the round-3/4 reporting contract without rewriting
frozen E6/E7 registrations. Collection retains every screened source, selected
and unselected eligibility, and death/survivor facts. E7 selection exports all
24 training prefix outcomes per source and the two selected-prefix confirmation
streams. The continuation traces supply the common-alive, candidate-only, and
B8-only mass terms and the productive dead-tail check.

Historical E7 confirmation/point labels remain visible alongside Wilson LB90.
The additional productive readout requires the original confirmation predicate,
B8 death, candidate H5000 survival and >=90% tail retention in both streams.
A historical TRANSFER_ADMISSIBLE label alone never admits an archive: E10 also
requires productive population coverage, feasible tau, and its envelope check.
The bridge uses prevalence times EARLY-stratum mean, never balanced EPD.

Run the reporting-only `research.bl_prefix.e10` entrypoint on verified completed
scientific bank/E6/E7 exports. SAMPLING_E10_ADDENDUM.md distinguishes eligible sources not selected by design
from missing selected outcomes. U is prevalence times selected-world mean;
missing selected training/confirmation rows make the envelope INCOMPLETE.
Coverage, overlap-aware uncertainty and stopping sensitivity are exported.
All mined worlds are burned for later evaluation. No E10 result grants release.

`production_preflight.py` measures burned full-H5000 source and continuation
fixtures through the real worker code, with reacting opponents and one native
math thread per worker. It is operational timing only. Run under Mac AC/lid,
thermal and slot guards with nice10 and at most two workers. The measured rate
still requires the explicit conservative L4 CPU slowdown assumption.

Resource preparation and scientific admission are separate fields. A manifest's
`bl_prefix.resource_readiness.ready` means its hash-bound measured forecast fits
the prepared budget with the two-times timeout margin inside 24 hours. The
existing `bl_prefix.ready` and plan `ready` remain false until final scientific
inputs exist and pass the unchanged admission checks. `DISPATCH.txt` lists each
missing dependency. A resource-ready manifest never bypasses that runtime fence.
