# A100 retention and frozen bundle qualification — 2026-10-23

Status: **PREPARED, NOT QUALIFIED; no acceptance dispatched.**

Offline implementation verification and the explicitly retained baseline
failures are recorded in [A100_RETENTION_VERIFICATION.md](A100_RETENTION_VERIFICATION.md).

Authority: the binding Astra ruling at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-36/JUDGMENT.md`,
sections 3–4 and 6–8. Implementation belongs to Sol-Ops. Independent acceptance
belongs to Codex-V; Claude/operator dispatches. Simulator, integration harness,
geometry, mechanics, observation and flood files retain their separate owners.
Ruling SHA-256: `156c80b412064defd88281424e4c84e6a732d9cb8bc6e163561ef352ffb5fe3b`.

This candidate is developed in `/Users/josenunez/Projects/ml/snake-dqn-retain`,
branch `t34-a100-retain-20261023`. It does not replace the historical
`QUALIFICATION.md`, mutate a prior acceptance receipt, publish qualification, or
modify a running broker. At implementation baseline the ops registry identifies
qualified commit `a36a821de98f7ef48b730ab454fa2870b0962a5e` and transport tree
`d0d92bdb572b770284a1fea6152417ffa5f06bea`. The operator must keep that accepted
broker available until the candidate earns acceptance. Merging source is not
operational adoption; transport tree checks continue to refuse unqualified bytes.

## Initial admission

An A100 chain contains at least two frozen, already-qualified main packets. Every
packet retains its independent current-pin native canary, family restrictions,
input and source identity, resume declaration, output root, manifest and receipt.
A retained receipt truthfully reports that the assignment remains allocated;
only the final independently verified release supplies disappearance evidence.
Retained canary-to-main promotion is outside this authorization.

After acceptance the ready-work chain's base envelope is 2 hours / 20 units and
its effective ceiling is 4 hours / 40 units. Each packet's original cap and the
remaining global allowance also apply; the first applicable cap wins. No ready
admitted work means release after at most 15 charged handoff minutes. There is no
open-ended idle loop or automatic completion retry.

Dependency and compiler reuse must be immutable and keyed by source, dependency
lock, Python ABI, Torch/CUDA versions, compiler identity and flags, and GPU SM.
A mismatch rebuilds or refuses. Credentials, scientific state, output files and
job roots are never cache entries. Each child starts from its frozen inputs.

## Acceptance envelope and required evidence

Forecast: **2 A100 allocation-hours**. Base: **2.4 hours / 24 units**. Effective
series ceiling: **4.8 hours / 48 units**, including every attempt and reserve,
not an additional allowance for each job. Conservative A100 admission ceiling:
**10 units/hour**. Reserve work before data. Use the sole sprint A100 only in a
planned boundary window; do not displace I0, Q5 or the candidate freeze. Reconcile
original sealed T0, round-28 remaining allowance and liabilities before dispatch.
The original round-28 deadlines and caps remain controlling.

Every acceptance manifest is at most 60 minutes with an explicit acceptance
exemption, or uses genuinely resumable frozen units. The effective envelope is
already doubled; never double it again. One original and at most two separately
prepared outcome-blind completion attempts share the same series ceiling.
Mechanical retries require verified failure, stop and missing units; scientific
failure parks the feature.

The independent acceptance must establish all of the following:

1. Final-only regression PASS on the final changed executable inventory before
   native retention; exact source and manifest pins remain unchanged throughout.
2. Native A100 two-main-packet retention on the same assignment, separate roots,
   inputs, outputs and immutable receipts, and continuous account accounting.
3. Cache invalidation for source/config/dependency/compiler/SM changes and clean
   state isolation. Reuse includes useful setup, not merely a surviving VM.
4. Mac disappearance/lease expiry, stale generation, child failure, handoff cap,
   accounting refusal, export failure, interrupted second launch and recovery.
   A stale lease cannot run later work or keep the endpoint alive. Watchdog
   disappearance remains within 300 seconds after lease expiry where applicable,
   observed independently before fallback cleanup.
5. Cumulative bootstrap, job, idle, export and stop debit, three-assignment cap,
   other liabilities, first-cap-wins, and blocked replacement after uncertain
   ownership or release.
6. Fixed cold-versus-retained comparison: at least 50% less second-job setup and
   at least 25% less total allocated wall, including transfers and handoff.
   Missing evidence is INCOMPLETE. A valid timing failure parks retention while
   the existing cold bundled path remains available.

Codex-V independently reviews native evidence before adoption. Offline tests and
a prepared package cannot establish these live gates. Release/serving authority
remains with the owner.

## Frozen experiments per allocation

A bundle freezes experiments, variants, cells, repeats, dependencies and branch
rules before data. Cells execute serially under one dependency environment and
all changed extensions compile/load before expensive cells. Each experiment has
independent acceptance; failure closes its dependent branch. Unrelated already
admitted cells may continue only after clean state restoration and verified
isolation. Scientific stops never become mechanical retries.

Exports use deterministic namespace plus frozen allocation index, immutable
payloads, size/SHA sidecars, and completion markers published last. Upload an
empty verified ledger on the first attempt. Subsequent ledgers are built on the
Mac from path/size/hash only, skip exactly verified units, and reject foreign
namespaces, allocations and duplicate IDs. The registered Mac reducer alone
opens scientific outcomes; remote summaries stay sealed. Preserve registered N
and look points.

Record allocation, bootstrap, compile, warmup, useful work, export and review
wait separately. Bundle adoption also requires at least 20% fewer allocated
minutes per completed valid cell, with packaging charged to I0 rather than O1.
The Sol-I owner supplies I0/R1/Q5 scientific packets; this operations change does
not create new experiments, revise native acceptance or alter scientific stops.

## Preparing a research family after O1 acceptance

Keep the existing cold native-canary release path. On that **A100-only** canary,
set `COLAB_RETENTION_CAPTURE_PINS=1` before freezing the package. This captures
actual Python ABI, full Torch/CUDA version, installed-distribution digest,
requirements bytes, full source tree, host/CUDA compiler identities, compiler
flags and SM in the cold canary receipt. It does not retain its assignment.
Bind that released receipt independently for every main packet and copy its
measured `cache_evidence.pins` into `retention_chain.cache`. Never invent
compiler or package-version values. A changed resolution of a ranged/nested
requirements file therefore refuses even when the requirements text is equal.

The frozen `retention_chain.shared` contains `chain_id`, `retention_minutes`
and `retention_units`; all links have identical shared and cache values.
`retention_chain.qualification` is a path/SHA binding to the independent
`snake-dqn-a100-retention-review/v1` review record. Main-only chain dispatch:

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn-retain
"$REPO/tools/ops/dispatch.sh" UNIQUE_CHAIN_NAME /absolute/first-main.json \
  --chain-manifest /absolute/second-main.json \
  --retention-minutes 240 --retention-units 40 \
  --broker "$REPO" --cwd "$REPO"
```

The ordinary qualified-tree, family canary, resume, source/input, account and
scientific admission checks still apply. H100's existing chain API remains
available under its original rules; no new H100 spending is funded by O1.
