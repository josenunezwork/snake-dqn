# Path B v2 — Amendment 2

The v2 implementation is an offline, opt-in assisted hero over released B8 s3.
Read [PROTOCOL_V2.md](PROTOCOL_V2.md) and the prospective
[build details](PROTOCOL_V2_BUILD_RULES.md) before using it. The historical v1 files
and evidence remain intact. The registered runner imports only the v2 adapter,
controller, engine, timing and analysis modules. Serving defaults stay OFF.

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

Run from this worktree. Use the existing primary interpreter. Every command
below is local; no tmux, service or cloud allocation is involved.

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

```bash
"$PY" -m research.t34_pathb.precheck_v2 \
  --out /absolute/path/t3-dense-64-materialized.json

"$PY" -m research.t34_pathb.retention_v2 precheck \
  --input /absolute/path/t3-dense-64-materialized.json \
  --out /absolute/path/fresh-v2-precheck

# Only after >=50/64, Phase 0 has finished, and full pre-registration is committed:
"$PY" -m research.t34_pathb.retention_v2 run \
  --registration /absolute/path/registered-v2-retention.json \
  --precheck /absolute/path/fresh-v2-precheck/precheck.json \
  --out /absolute/path/fresh-v2-retention

# Exactly one registered look; preserve every immutable input/output.
"$PY" -m research.t34_pathb.retention_v2 analyze \
  --out /absolute/path/fresh-v2-retention
```

Real execution requires Mac CPU, AC, open lid and admitted thermal telemetry.
The harness refuses existing output roots. No automatic retry, resume, extension
or replacement bank is provided. Numerical pools are capped before imports.

## Registration schema and evidence

The operator prepares `schema:"t34-pathb-v2-registration/v1"`, `ratified:true`,
`source_commit` (an ancestor of execution HEAD with identical execution bytes),
`source_hashes` from `retention_v2.source_inventory()`, `arms:["bare","paced",
"unpaced"]`, `paced_fps:24`, `terminal_per_mix:32`, `product_sessions:16`,
`horizon:5000`, `workers:1`, `max_elapsed_hours:32`, `max_cpu_hours:48`,
`fresh_disjoint_banks:true`, and the complete materialized `terminal_rows`,
`product_rows`, `timing_row`, `timing_ticks` and `timing_warmup`. Freeze the load
profile and full contract bars in the committed preregistration. The unpaced
descriptive arm uses the same successor with no pacing sleep. Between simulated
ticks the harness holds the simulation clock while its sole worker finishes
maintenance and search, then uses the ordinary action boundary. This reference
has no rewind. The paced arm never waits. Settle time stays outside ON timing
and counts against the study's elapsed/CPU caps.

Set `load_profile:{"quiet_ratified":true}` and `contended:false` only after the
registered quiet-host condition actually holds. Product and timing rows declare
`mix:"mixed"`; the timing seed is also disjoint from every study seed. A nested
terminal `roster` must repeat its outer row's seed and mix exactly.

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
The registered load profile must explicitly ratify quiet-host execution. Record
large deaths at pre-action length >=800 and late deaths at frame >=2000, with
their overlap; these are diagnostics.

The source inventory covers tracked Python in `src`, `research`, and
`web/backend`, configuration YAML, protocol/governance and serving pins. It
includes T3/T33 imports. Changing any executable dependency requires a new
prospective registration; an opaque success flag or CLI exit status is not proof.

## Acceptance table

| Requirement | Verification / deciding evidence |
|---|---|
| Exact batched v8 continuation and B8 tie/mask semantics | Independent reference traces and focused engine tests; see `VERIFICATION_V2.md` |
| One thread, one request, bounded confirmation history, stale/late fallback | Controller queue, identity, mismatch and overflow regressions |
| Real terminal + continuous Watch adapter | Tiny actual adapter smoke, physical/carry/v8 comparison |
| T3 pre-check >=50/64 at 24 fps | Implemented; full run reserved for operator |
| Fresh 32/mix H5000 +16 paired Watch6 sessions | Registered runner implemented; full run reserved for operator |
| P7 paired-world/session inference and Holm family | Analyzer rule and boundary tests |
| First-result p95 <=2 s | Engineering target; no final timing claim from implementation smoke |
| Update p95 <=11.688 ms, zero 12/24 fps overruns | Quiet-host registered probes after Phase 0; contended smoke cannot qualify |
| All contract skill/product bars | Full registered decision; no efficacy claim from smoke |
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
create-only registrations. It cannot start gameplay. The full 113-seed allocation
was audited before bank publication; **do not run allocate again** on these
namespaces, because their declarations are now searchable history. A collision
rejects the whole allocation. A future namespace needs a prospective sibling
erratum, never a replacement seed. The bank fixes 96 terminal rows, 16 product
rows and one separate timing seed, with exact historical terminal rosters.

Package validation (safe during implementation; no pre-check or efficacy):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-prereg
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
"$PY" -m research.t34_pathb.registration_retention_v2 validate
```

The OC receipt contains 20,000 replicates per declared proxy/stress model,
protected-regression scenarios, +40 mass power and a fixed-N/sequential cost
comparison. T3's 32 physical seeds are resampled separately for s0/s3; they are
not 64 independent worlds. Frozen/mixed terminal variance is a disclosed proxy.
Power for a positive mass bound is distinct from power to meet point gain +40;
product, boost and no-death bars can reduce joint success further. Fixed N stays
fixed. The simulator is one-thread, checkpointed every 250 replicates, with a
60-second command budget. To reproduce it prospectively into **new** paths:

```bash
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out /absolute/path/new-oc.json \
  --state /absolute/path/new-oc-state.json \
  --reps 20000 --seed 3420261020 --max-seconds 60
# If INCOMPLETE, repeat that exact command with --resume; do not change its inputs.
```

The following is an **operator-only sequence**, not run by the implementation
job. Phase 0 must have finished and the Mac must be quiet, on AC, lid open and
thermally admitted. No live service, tmux or cloud command is needed. Do not
continue if validation fails. Numerical sources must be committed and unchanged
through trace verification, pre-check, ratification and efficacy. The trace
receipt must come from actual final-source identity/trace verification and bind
`retention_v2.source_inventory()`; the old implementation receipt is stale for
this expanded inventory. This package deliberately does not manufacture it.

First obtain the actual Phase 0 completion receipt and final-source PASS trace
receipt, then set their existing absolute paths below. The completion receipt
is evidence the operator reviews; mere file existence does not prove Phase 0
finished. The operator authorizes the pre-check separately, then runs:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-prereg
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-v2-operator-01
export PHASE0_RECEIPT=/absolute/path/actual-phase0-completion-receipt.json
TRACE=/absolute/path/actual-final-source-trace-receipt.json
mkdir "$E"

"$PY" -m research.t34_pathb.registration_retention_v2 validate
"$PY" -m research.t34_pathb.precheck_v2 --out "$E/t3-dense-64.json"
"$PY" -m research.t34_pathb.retention_v2 precheck \
  --input "$E/t3-dense-64.json" --out "$E/precheck"
```

Stop unless that actual pre-check reached **>=50/64**. It is reused development
data, not Claim B evidence. Capture observed load only after verifying Phase 0
completion and absence of other project compute. The command below records real
load/power/lid/thermal readings and the reviewed completion receipt hash; the
`phase0_finished` and zero competing-job fields are operator attestations to
those checks. Run it only when they are true:

```bash
export RETENTION_OPERATOR_EVIDENCE="$E"
"$PY" - <<'PY'
import hashlib, json, os
from datetime import datetime, timezone
from pathlib import Path
from research.compute.thermal_guard import read_pmset_therm, parse_pmset_therm, thermal_reasons
from research.perf_sim_20261007 import with_slots
phase0 = Path(os.environ['PHASE0_RECEIPT']).resolve(strict=True)
thermal = read_pmset_therm()
ac, lid = with_slots.on_ac(), with_slots.lid_open()
ok = bool(thermal.get('available')) and not thermal_reasons(parse_pmset_therm(thermal.get('text') or ''))
assert ac and lid and ok, 'AC/lid/thermal admission failed'
record = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'phase0_finished': True,
    'phase0_completion_evidence': str(phase0),
    'phase0_completion_sha256': hashlib.sha256(phase0.read_bytes()).hexdigest(),
    'other_project_cpu_jobs': 0,
    'load_average': list(os.getloadavg()),
    'ac_power': ac, 'lid_open': lid, 'thermal_ok': ok,
    'thermal_telemetry': thermal,
}
with (Path(os.environ['RETENTION_OPERATOR_EVIDENCE']) / 'quiet-host.json').open('x') as stream:
    json.dump(record, stream, indent=2, allow_nan=False)
    stream.write('\n')
PY

# Bind the exact HEAD recorded by the actual pre-check; later JSON-only commits are allowed.
SOURCE_COMMIT=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["source_commit"])' "$E/precheck/precheck.json")
READY=research/t34_pathb/REGISTRATION_RETENTION_V2_RATIFIED.json
"$PY" -m research.t34_pathb.registration_retention_v2 ratify \
  --source-commit "$SOURCE_COMMIT" --precheck "$E/precheck/precheck.json" \
  --trace-receipt "$TRACE" --load-record "$E/quiet-host.json" \
  --operator Claude --acknowledge-near-boundary-precedence --out "$READY"

# Publish ratification before efficacy; preserve the draft's bytes.
git add "$READY"
git commit -m 'Ratify T34 paced retention before efficacy'
git fetch origin
git merge origin/main
git push origin HEAD:main
git push origin t34-retention-prereg-20261020

# Any intervening execution-source change blocks this validation; do not bypass it.
"$PY" -m research.t34_pathb.registration_retention_v2 validate \
  --registration "$READY" --ready --precheck "$E/precheck/precheck.json"
"$PY" -m research.t34_pathb.retention_v2 run \
  --registration "$READY" --precheck "$E/precheck/precheck.json" \
  --out "$E/retention"
"$PY" -m research.t34_pathb.retention_v2 analyze --out "$E/retention"
```

Exactly one complete fixed-N look. Preserve incomplete/fault evidence; no
automatic retry, efficacy resume, bank replacement or extension. The first of
32 h elapsed / 48 CPU-h / $0 wins. Product sessions and timing use actual offline
Watch6; no release-monitor or leaderboard writes. Development GO only admits
fresh Mac strict-gate preparation. Client latency, Watch8/12, Play, serving
qualification, visible assisted labelling and owner-approved release remain
separate. Teacher label rounds are not funded by this package.

Package verification, exact preparation commands and the development-only OC
results are recorded in [VERIFICATION_RETENTION_V2.md](VERIFICATION_RETENTION_V2.md).
