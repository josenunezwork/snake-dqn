# Path B v2 — Amendments 2–4

The v2 implementation is an offline, opt-in assisted hero over released B8 s3.
Read [PROTOCOL_V2.md](PROTOCOL_V2.md) and the prospective
[build details](PROTOCOL_V2_BUILD_RULES.md) before using it. The historical v1 files
and evidence remain intact. The registered runner imports only the v2 adapter,
controller, engine, timing and analysis modules. The separate descriptive
unpaced collector and manifest builder are included in the frozen source
inventory. Serving defaults stay OFF.

The retention package remains **DRAFT_PENDING_OPERATOR_RATIFICATION**. Its
pre-data Colab amendment runs only the unpaced descriptive arm through the
already qualified CLI transport. The Mac runs the timing gate, T3 pre-check, and
the bare/paced study records. The Colab arm uses the same 144 terminal worlds
and 32 product sessions, but it cannot affect a Mac verdict, fault, multiplicity
calculation, or Claim B decision. A missing or failed Colab row remains
descriptive evidence; only an unconditional fault in the Mac bare/paced arms can
KILL. See the [descriptive-arm amendment](PREREGISTRATION_RETENTION_V2.md#amendment-3-addendum--pre-data-descriptive-unpaced-colab)
and [contention/fault amendment](PREREGISTRATION_RETENTION_V2.md#amendment-4-addendum--pre-data-contention-accounting-and-arm-scoped-faults)
for the binding rationale and caps.

`engine_v2.py` maintains a D500 exact continuation, stops its future at hero
death, and searches still-future decisions earliest-first. Windows of at most
eight consecutive branch points and their alternatives share
a native multi-environment simulator batch, retain independent opponent/v8
carry, and use exact rowwise PyTorch CPU forwards. The worker reuses its action
and active-row scratch. Native batch assembly remains a per-window cost.
No v1 or serial LAD fallback is used. ONNX is unavailable in this implementation;
enabling it requires a later explicit backend and gameplay identity gate.

`controller_v2.py` owns one worker and at most one outstanding maintenance/search
request. Every real post-step digest is retained while the worker is busy.
Published predictions permit immediate digest rejection; action execution also
checks session, hero life, checkpoint, generation and exact target frame. Late,
stale, unconfirmed and unavailable results use bare B8. Logs distinguish the
proposal, execution, fallback, horizon rescue and all-die delay.

`adapter_v2.py` uses the T3 captured-world seam for terminal worlds. Its Watch
adapter drives the actual T15 `GameSession` with a separate exact Grid mirror
and fails on physical/RNG, carry or v8 divergence. Product opponents are five
fixed FRP3+v8 policies. Full-tick timing includes this bridge and snapshot/digest
cost; it must not be presented as an already qualified production hook. It
does not start a server, write a score or write release-monitor telemetry.

## Commands

Run the local smoke commands in this section from this worktree with the
existing primary interpreter. They use no tmux, service, or cloud allocation.
The later **Descriptive unpaced Colab arm — operator only** section is the sole
exception and dispatches only through the qualified supervisor.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-v2-pathb2
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python

# Serialize checks that can construct a real background worker during Phase 0.
"$PY" -m pytest -q tests/test_t34_pathb_*v2.py

# Prepare a reused diagnostic roster; this constructs a plan and runs no episode.
"$PY" - <<'PY'
import json
from pathlib import Path
from research.redesign_scope_20261007 import grid_h5000_identity as gid
from research.t3_rollout_labels_20261015.bank_engine import planned_rosters
rows = planned_rosters(gid._context(1), [991827], ["mixed"])
with Path("/tmp/t34-pathb-v2-smoke-input.json").open("x") as stream:
    json.dump(rows, stream, indent=2)
PY

# Tiny real terminal and Watch ON/OFF smoke; create a fresh output directory.
"$PY" -m research.t34_pathb.retention_v2 smoke \
  --input /tmp/t34-pathb-v2-smoke-input.json --frames 2 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-v2/smoke-01

# Tiny timing plumbing check during Phase 0; these are contended observations.
"$PY" - <<'PY'
import json
from pathlib import Path
from research.t34_pathb.adapter_v2 import open_watch
from research.t34_pathb.timing_v2 import run_probes
result = run_probes(
    lambda arm, fps: open_watch({"seed": 991827}, fps, arm),
    ticks=2, warmup=0, contended=True,
)
with Path("/tmp/t34-pathb-v2-timing-smoke.json").open("x") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
PY
```

The following commands are **operator-run**, after implementation verification.
They are not launched by the implementation task. The T3 input must contain the
original 64 dense source trajectories: B8 s0 and B8 s3 over the same 32 scripted
seeds. The pre-check preserves each trajectory's original checkpoint and roster.
These are 64 reused trajectory cases, with 32 unique physical seeds; no
independent-world inference is made. The >=50/64 rule is unchanged. The fresh
registered study uses released B8 s3 exclusively. Every row binds its original
ceiling receipt, checkpoint, arm/env coordinate and materialized source hash.

Use the complete timing-first operator sequence in
[Retention registration package](#retention-registration-package). Timing must
pass before constructing the pre-check gameplay; the input materializer alone
only reads historical records.

Mac timing, pre-check, and bare/paced execution require Mac CPU, AC, open lid
and admitted thermal telemetry. The harness refuses existing output roots. No
automatic retry, resume, extension or replacement bank is provided. Numerical
pools are capped before imports.

## Registration schema and evidence

The operator prepares `schema:"t34-pathb-v2-registration/v1"`, `ratified:true`,
`source_commit` (the exact detached execution `HEAD`), `source_hashes` from
`retention_v2.source_inventory()`, `arms:["bare","paced","unpaced"]`,
`mac_arms:["bare","paced"]`, and the fixed `unpaced_colab` policy,
`paced_fps:24`, `terminal_per_mix:48`, `product_sessions:32`, `horizon:5000`,
`workers:1`, `max_elapsed_hours:40`, `max_cpu_hours:60`, `precheck_max_elapsed_hours:8`,
`fresh_disjoint_banks:true`, and the complete materialized `terminal_rows`,
`product_rows`, `timing_row`, `timing_ticks` and `timing_warmup`. Freeze the load
profile and full contract bars in the committed preregistration. The registered
entry-point import graph determines the source inventory: it contains only the
transitive modules and data files actually read by `retention_v2`, `analyze_v2`,
`timing_v2`, `host_guard_v2`, `registration_retention_v2`, `retention_oc_v2`,
`unpaced_colab`, and `make_unpaced_manifest`, plus their imports. It is not a
whole-`research/` inventory.
The unpaced descriptive arm uses the same successor with no pacing sleep. Between
simulated ticks the harness holds the simulation clock while its sole worker
finishes maintenance and search, then uses the ordinary action boundary. This
reference has no rewind. It is exported by the separately priced Colab path;
the paced arm never waits, and only Mac work is charged to the registered
40 elapsed-hour / 60 CPU-hour study caps.

Copy the exact measured `load_policy` keys from the profile to `load_profile`;
the runtime computes contention from fresh samples. The profile freezes 5-second
sampling, the load/CPU thresholds, a >60-second continuous persistence rule,
and `max_contended_fraction: 0.02`. At admission, the runner records the
measured idle baseline and CPU count with those thresholds. It then adds 5
seconds for each contended sample. It checks the 2% allowance only when a phase
ends: the timing phase uses its registered 40-hour study cap, the pre-check its
separate 8-hour cap, and the study its 40-hour cap. An early contended sample or
interim excess never stops a phase; an end-of-phase excess is INCONCLUSIVE for
that phase. It pauses only between rows, after dropping pending assistance
requests as counted fallbacks.

Before each timing, pre-check, or study phase, the host must meet a 120-second
quiet admission window covering raw load and other-process CPU, as well as AC,
lid, and thermal telemetry. A failed window refuses to start: the command exits
nonzero without `decision.json` or a terminal phase record, and the operator may
retry later. This refusal does not create a scientific verdict.
Product and timing rows declare `mix:"mixed"`; the timing seed is also disjoint
from every study seed. A nested terminal `roster` must repeat its outer row's
seed and mix exactly.

`checkpoints` is a list of `{path,sha256}` objects, including released B8 s3 and
every frozen opponent, including Watch FRP3 s12. Coverage is checked against
every terminal checkpoint spec before an episode starts. `profile`, `bank`, `seed_audit`, `preregistration`,
`trace_receipt` and `operating_characteristic` are
each `{path,sha256}` mappings. The preregistration bytes must exist in committed
HEAD. The bank JSON must reproduce the registered rows exactly. The seed audit
must pass, report no hits/collisions, and contain exactly the registered seed
union. Mixes and product sessions are disjoint; strict-gate banks stay unused.
The pre-check receipt binds raw paired outcomes and the same source inventory.
Trace and operating-characteristic receipts must bind that exact source. The
operator must produce the study-specific fixed-N survival-band operating-
characteristic check before running; implementation tests do not provide it.
The registered load profile freezes measured quiet-host thresholds. Record
large deaths at pre-action length >=800 and late deaths at frame >=2000, with
their overlap; these are diagnostics.

The source inventory is a deterministic import-graph walk over the registered
entry points. It includes their local transitive Python dependencies and data
files read at runtime, including required T3/T33 dependencies and the frozen
profile/preregistration assets. It excludes generated registration and OC files,
which would make the closure recursive. Changing a closure member requires a
new prospective source freeze; an opaque success flag or CLI exit status is not
proof.

## Acceptance table

| Requirement | Verification / deciding evidence |
|---|---|
| Exact batched v8 continuation and B8 tie/mask semantics | Independent reference traces and focused engine tests; see `VERIFICATION_V2.md` |
| One thread, one request, bounded confirmation history, stale/late fallback | Controller queue, identity, mismatch and overflow regressions |
| Real terminal + continuous Watch adapter | Tiny actual adapter smoke, physical/carry/v8 comparison |
| T3 pre-check >=50/64 at 24 fps | Separately authorized development pre-check with an 8 h cap; full run reserved for operator |
| Fresh 48/mix H5000 +32 paired Watch6 sessions | Registered runner implemented; full run reserved for operator |
| P7 paired-world/session inference and frozen Holm family | Analyzer rule applies the paced one-sided p <= .05 worst-case threshold; unpaced has no decision p-value |
| First-result p95 <=2 s | Engineering target; no final timing claim from implementation smoke |
| 24 fps update p95 <=11.688 ms, zero 12/24 fps overruns | Registered probes; 5-second samples add 5 seconds when contended, and only an end-of-phase excess above 2% of that phase's registered cap is INCONCLUSIVE |
| All contract skill/product bars | Full registered decision; no efficacy claim from smoke |
| Colab unpaced descriptive record | Per-world immutable export and shared bare-prefix physical-state-hash comparison; missing/invalid exports stay descriptive |
| Default serving OFF / no release | No serving-selector or pin change |

Passing this implementation's tests does not pass Claim B. A development GO
admits fresh Mac strict-gate preparation; release and wider-roster qualification
remain separate operator decisions.

## Retention registration package

The 2026-10-20 package is **DRAFT_PENDING_OPERATOR_RATIFICATION**. Read
[PREREGISTRATION_RETENTION_V2.md](PREREGISTRATION_RETENTION_V2.md), including the
explicit near-boundary INCONCLUSIVE precedence, before ratification. The draft
JSON binds the committed bank, audit, profile, checkpoints and development-only
OC receipt. Its only deferred evidence binding is the final-source trace receipt.
It is unauthorized; pending quiet-host conditions are not observed facts.

`registration_retention_v2` only allocates, hashes, validates and writes
create-only registrations. It cannot start gameplay. The original 113-seed allocation
was audited before bank publication; **do not run allocate again** on these
namespaces, because their declarations are now searchable history. A collision
rejects the whole allocation. A future namespace needs a prospective sibling
erratum, never a replacement seed. The resized bank fixes 144 terminal rows, 32 product
rows and one separate timing seed, with exact historical terminal rosters.

The [pre-data resize ruling](../../docs/research/specs/2026-10-20_t34_retention_resize.md)
increases terminal N32 -> N48/mix and Watch sessions 16 -> 32 after the low-power
review, with 40 elapsed h / 60 CPU-h caps. All 113 original seeds/rows/rosters
are retained; 64 additions use the frozen allocator's next indices and are
audited together against the retained bank, Round 0 draft and prior history.
The updated seed audit embeds the original audit and its historical hashes.
`resize --out-bank NEW --out-audit NEW` is create-only and accepts only the
original N32/16 inputs; do not rerun it on the published resized bank.
Minimum Mac pacing exposure is (144+32)*2*5000/24/3600 = **20.370 h**, plus
~.361 h timing/warmup and Mac setup. The separately
authorized reused-T3 pre-check has its own 8 h elapsed cap and is not debited
against the 40 h/60 CPU-h study caps. Stop the study at its first cap as
INCONCLUSIVE. The unpaced arm has a separate 42.533128-unit Colab envelope,
including a 7.088855-unit reserve; it is not debited to the Mac caps. Before a
Colab allocation, the operator runs the required Mac preflight of two
development worlds at short frames. It produces no efficacy row and cannot
qualify timing or any contract bar. The new OC power numbers and complete
verification are in [VERIFICATION_RETENTION_RESIZE.md](VERIFICATION_RETENTION_RESIZE.md).

Package validation (safe during implementation; no pre-check or efficacy):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-unpaced-colab
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
"$PY" -m research.t34_pathb.registration_retention_v2 validate
```

### Descriptive unpaced Colab arm — operator only

The Mac-only two-development-world preflight is development work authorized by
this pre-data amendment. It is not an efficacy run: it persists no world metrics,
uses two development rows outside the registered bank, and is limited to four
frames. Production Colab collection begins only after the operator has published
the ratified registration. Provide the two fully materialized development rows
as `{"rows":[...two rows...]}` at the path below; the runner rejects a
registered-bank seed. Do not dispatch from this implementation task or replace
the qualified `research/colab_cli` transport.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-unpaced-colab
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
E=/absolute/path/to/a-fresh-t34-unpaced-operator-root
# The preflight accepts the current source-matching draft registration.
REG=research/t34_pathb/REGISTRATION_RETENTION_V2.json
CAT=research/t34_exit_b8/PUBLIC_INPUTS.json
mkdir -p "$E"

# The public release is downloaded and hash-checked into a fresh local cache.
"$PY" -m research.t34_exit_b8.public_inputs --catalog "$CAT" --out-dir "$E/inputs"

# Mac-only transport/entrypoint preflight: exactly two development rows, <=4 frames.
# $E/dev-rows.json is operator-prepared and has {"rows":[...two disjoint rows...]}.
T34_UNPACED_PREFLIGHT=1 \
T34_UNPACED_PYTHON="$PY" \
T34_UNPACED_INPUT_ROOT="$E/inputs" \
T34_UNPACED_OUTPUT_ROOT="$E/preflight" \
T34_UNPACED_REGISTRATION="$REG" \
T34_UNPACED_CATALOG="$CAT" \
T34_UNPACED_DEVELOPMENT_ROWS="$E/dev-rows.json" \
T34_UNPACED_SOURCE_COMMIT="$("$PY" -c 'import json; print(json.load(open("research/t34_pathb/REGISTRATION_RETENTION_V2.json"))["source_commit"])')" \
T34_UNPACED_FRAMES=4 \
bash research/t34_exit_b8/unpaced_retention.sh
```

The preflight creates `$E/preflight/preflight.json` with `.sha256` and
`.complete` sidecars. It records the two-world transport exercise and platform
configuration but no efficacy metric, timing result, or decision input.
Manifests pin this small campaign launcher so the unchanged shared broker
rechecks qualified transport bytes during every later validation and dispatch,
including saved manifests. The launcher executes the hash-bound Path B script.

For the production descriptive export, replace `REG` with the published
`REGISTRATION_RETENTION_V2_RATIFIED.json`. Before building or allocating a lane,
inspect the account with `"$COLAB_BIN" --auth oauth2 usage` and `sessions`.
Set `OTHER_ASSIGNMENTS` to the complete endpoint-to-positive-rate map for every
existing assignment, then set `LANES` to the remaining account capacity. It must
be 1 through 3, and the new lanes plus existing assignments must never exceed
three. The empty map and `LANES=3` are valid only when the sessions list is
empty. A CPU runtime requires a measured positive provider rate and
`--cpu-rate-units-per-hour`; do not use an L4 rate for it. Cold setup, export and
timeout are priced by each manifest and must remain inside the 42.533128-unit
aggregate envelope, including its 7.088855-unit reserve.

```bash
LAUNCH=/Users/josenunez/Projects/ml/snake-dqn-unpaced-colab/research/colab_cli/launch.sh
COLAB_BIN=/Users/josenunez/.local/bin/colab
REG=/absolute/path/to/REGISTRATION_RETENTION_V2_RATIFIED.json
OTHER_ASSIGNMENTS='{}'  # Only after `"$COLAB_BIN" --auth oauth2 sessions` is empty.
LANES=3                 # Set to 3 minus the count of OTHER_ASSIGNMENTS.
COMMIT="$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$REG")"

for SHARD in $(seq 0 $((LANES - 1))); do
  "$PY" -m research.t34_pathb.make_unpaced_manifest \
    --registration "$REG" --catalog "$CAT" \
    --output-dir "$E/colab/shard-$SHARD" --out "$E/manifest-$SHARD.json" \
    --shard "$SHARD" --shard-count "$LANES" --nproc 12 --commit "$COMMIT" \
    --other-assignments "$OTHER_ASSIGNMENTS"
done

# `--check` only prints the guarded command. The second validation performs the
# no-allocation manifest check. Give each lane its own durable supervisor state.
for SHARD in $(seq 0 $((LANES - 1))); do
  STATE="$E/colab-state-$SHARD"
  "$LAUNCH" --check --state-dir "$STATE" --colab "$COLAB_BIN" \
    validate "$E/manifest-$SHARD.json"
  "$LAUNCH" --state-dir "$STATE" --colab "$COLAB_BIN" \
    validate "$E/manifest-$SHARD.json"
done

# After reviewing the validated manifests and live assignment capacity, dispatch each lane once.
for SHARD in $(seq 0 $((LANES - 1))); do
  STATE="$E/colab-state-$SHARD"
  tmux new-session -d -s "colab-cli-t34-unpaced-$SHARD" \
    "'$LAUNCH' --state-dir '$STATE' --colab '$COLAB_BIN' run '$E/manifest-$SHARD.json'"
done
```

The supervisor retrieves each completed lane under
`$E/colab/shard-N/exports/unpaced/`: `shard.json` and
`worlds/{terminal,product}-NNN.json` each have a SHA-256 sidecar and completion
marker. A missing, failed, or hash-invalid shard remains a descriptive missing
row. After the Mac runner has already written its create-only `decision.json`,
attach whatever verified shard receipts are available; this command preserves
that decision and only prints the descriptive comparison:

```bash
"$PY" -m research.t34_pathb.retention_v2 analyze \
  --out /absolute/path/to/the-complete-Mac-retention-root \
  --unpaced-shards "$E"/colab/shard-*/exports/unpaced/shard.json
```

The OC receipt contains 20,000 replicates per declared proxy/stress model,
protected-regression scenarios, +40 mass power and a fixed-N/sequential cost
comparison. T3's 32 physical seeds are resampled separately for s0/s3; they are
not 64 independent worlds. Frozen/mixed terminal variance is a disclosed proxy.
Power for a positive mass bound is distinct from power to meet point gain +40;
product, boost and no-death bars can reduce joint success further. The extended
OC reports T3-shaped +.10/+.25 scripted survival crossed with +40/+60/+92 mass,
joint GO probabilities conditional on timing/boost compliance and unchanged product growth, and 32-session
Watch6 paired deaths/1k power at 20%/40% reduction from T15/T13b baselines. If
T3-like joint GO power is below .5, report it for operator decision; keep N48; do not resize again.
Fixed N stays fixed. The simulator is one-thread, checkpointed every 250 replicates, with a
60-second command budget. To reproduce it prospectively into **new** paths:

```bash
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out /absolute/path/new-oc.json \
  --state /absolute/path/new-oc-state.json \
  --reps 20000 --seed 3420261020 --max-seconds 60
# If INCOMPLETE, repeat that exact command with --resume; do not change its inputs.
```

The following is an **operator-only sequence**, not run by this implementation
job. The registration remains a draft. Obtain an actual final-source PASS trace
receipt; the old source receipt is stale after these fixes. Phase 0 must finish
before timing. Runtime admission measures host conditions and competing processes;
there are no hard-coded Phase-0-finished or zero-job attestations.

Build order is binding: commit the implementation, create the fresh
development-only OC into a new artifact path, copy and commit its retained
receipt, then build and commit the draft registration. This makes the draft's
`source_commit` include the committed OC receipt. The generated OC is excluded
from the executable import closure to avoid a recursive source hash, but its
separate registration binding remains required. The draft stays unratified until
the operator completes the later trace-and-ratification step.

Publish the ratified JSON in the authoring worktree before timing or pre-check.
The exact ratified bytes must be published there, but execution occurs only from
a fresh worktree detached at the ratified `source_commit`. Pass the published
registration to that frozen worktree by absolute path. Do not run these gameplay
commands during implementation:

```bash
AUTHOR_ROOT=/absolute/path/to/the/authoring-worktree
cd "$AUTHOR_ROOT"
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-v2-operator-01
TRACE=/absolute/path/actual-final-source-trace-receipt.json
READY="$AUTHOR_ROOT/research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json"
mkdir "$E"
"$PY" -m research.t34_pathb.registration_retention_v2 validate
SOURCE_COMMIT=$("$PY" -c 'import json; print(json.load(open("research/t34_pathb/REGISTRATION_RETENTION_V2.json"))["source_commit"])')
"$PY" -m research.t34_pathb.registration_retention_v2 ratify \
  --source-commit "$SOURCE_COMMIT" --trace-receipt "$TRACE" \
  --operator Claude --acknowledge-near-boundary-precedence --out "$READY"
git add research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json
git commit -m 'Ratify T34 retention before timing and pre-check'
git fetch origin
git merge origin/main
git push origin HEAD:main
git push origin t34-unpaced-colab-20261020

# Create the owned execution worktree. It must remain detached, clean and stationary
# through timing, the optional pre-check and final analysis.
FROZEN=/absolute/path/to/owned-t34-retention-frozen
git worktree add --detach "$FROZEN" "$SOURCE_COMMIT"
git -C "$FROZEN" diff --quiet
git -C "$FROZEN" diff --cached --quiet
test "$(git -C "$FROZEN" rev-parse HEAD)" = "$SOURCE_COMMIT"
cd "$FROZEN"
"$PY" -m research.t34_pathb.registration_retention_v2 validate \
  --registration "$READY" --ready

# FIRST: admission requires 120 quiet seconds. If refused, this exits nonzero
# without a decision.json or terminal receipt; remedy host conditions and retry.
# Then run quiet-host ON/OFF Watch6 timing at 24 and 12 fps (~.36 h).
"$PY" -m research.t34_pathb.retention_v2 timing-gate \
  --registration "$READY" --out "$E/timing"
# Stop unless the create-only timing gate receipt says PASS.
# A timing failure writes decision.json (KILL-timing); no pre-check or efficacy.
"$PY" -m research.t34_pathb.precheck_v2 --out "$E/t3-dense-64.json"
"$PY" -m research.t34_pathb.retention_v2 precheck \
  --registration "$READY" --timing-gate "$E/timing/timing_gate.json" \
  --input "$E/t3-dense-64.json" --out "$E/precheck"
# Stop unless the actual pre-check reaches >=50/64.
"$PY" -m research.t34_pathb.retention_v2 run \
  --registration "$READY" --timing-gate "$E/timing/timing_gate.json" \
  --precheck "$E/precheck/precheck.json" --out "$E/retention"
"$PY" -m research.t34_pathb.retention_v2 analyze --out "$E/retention"
```

Host monitoring samples asynchronously every 5 seconds and never sleeps inside
collection or between ticks. It pauses only between completed rows, drops pending
assistance as counted fallbacks, and does not exclude or retry a row. Before each
phase it requires 120 continuous quiet seconds with acceptable AC, lid, and
thermal telemetry plus quiet raw load and other-process CPU. The admission record
includes the measured idle load baseline, CPU count, and configured thresholds.
If admission does not complete, the command refuses to start with a nonzero exit
and writes neither `decision.json` nor a terminal phase record; retry only after
the host is quiet.

After admission, load excess above the measured idle baseline plus the registered
3.0 allowance and a known Phase 0/research pattern above the registered CPU
threshold count only after each has persisted continuously for more than 60
seconds. Every contended sample adds 5 seconds and retains raw load and
PID/command evidence. At phase end only, the
runner compares contended seconds to 2% of the phase's registered elapsed cap:
40 hours for timing, 8 hours for pre-check, and 40 hours for study collection.
An excess is INCONCLUSIVE for that phase; an early contended sample or interim
excess does not end it. AC/lid/thermal changes pause between rows and a
non-acceptable state persisting 120 seconds is INCONCLUSIVE. Unavailable
telemetry cannot pass; pause time is charged. The profile and validator use the
same policy keys; no load attestation is accepted as a measurement.

Timing and pre-check admission also require their original phase
`source_validated.json` certificate, written after the source monitor closes; the
certificate binds the receipt hash and source commit. Missing certificates or a
sibling `source_failure.json` block the next phase even after source restoration.

Only the runner's create-only `decision.json` is the verdict. Re-analysis labels
its calculations descriptive and preserves that file. Efficacy ticks >=41.7 ms
are reported diagnostics with bare fallback included; only registered timing
probes enforce zero overruns. Update p95 <=11.688 ms gates 24 fps; 12 fps reports
p95 and gates overruns/cadence. Near-zero mass uncertainty applies only when mass
is the sole failing category and product passes.

Exactly one complete fixed-N look. Preserve incomplete/fault evidence; no
automatic retry, efficacy resume, bank replacement or extension. The first of
40 h elapsed / 60 CPU-h / $0 wins for the study; the pre-check has its own 8 h
elapsed cap and is never debited to that study budget. The timing gate is charged
to the 40-hour study cap, which is also its registered contention denominator.
Product sessions and timing use actual offline Watch6; no release-monitor or
leaderboard writes. Development GO only admits fresh Mac strict-gate preparation.
Client latency, Watch8/12, Play, serving qualification, visible assisted
labelling and owner-approved release remain separate. Teacher label rounds are
not funded by this package.

The frozen-worktree fence covers the standalone `analyze` command as well as
timing, pre-check, and efficacy. If a source change is discovered near a final
write, the runner preserves the evidence and writes `source_failure.json` to
invalidate the otherwise preserved decision; it never silently substitutes a
new result.

Package verification, exact preparation commands and the development-only OC
results are recorded in [VERIFICATION_RETENTION_V2.md](VERIFICATION_RETENTION_V2.md).
