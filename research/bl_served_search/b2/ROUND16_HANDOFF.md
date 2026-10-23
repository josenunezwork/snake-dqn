# B2 round-16 freeze and blocked dispatch packs — 2026-10-22

This prospectively supersedes the unresolved-parameter and admission paragraphs
of HANDOFF.md. Authority: the owner's instruction treating the full Frozen JSON
block of external `big-lever/round-16/JUDGMENT.md` as RATIFIED_BEFORE_DATA. The
exact JSON is retained in `round16/B2_PARAMETER_RULING.json`; its authority record
binds the judgment and qualified CLI record by SHA. No B2 gameplay, real preflight,
family canary, production allocation or outcome look was run for this handoff.

## Frozen design

Tracked: `research/bl_served_search/b2/round16/registration/b2-design.json`.
Identical external dispatch input:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-b2/round16-20261022/registration/b2-design.json`.
SHA-256: `08e51f03ed938d1ce5b18be53d9e5222c8b1cbbf84d3b194ff3d178f173f54f8`.

- 14,400 continuous-respawn Watch6 decisions, bare B8 s3 vs five FRP3 s12+v8.
- Absolute decision grid 0,5,...,14395, never reset on respawn; stride 5.
- Up to four global bottom-hash natural alive L>=800 roots per trajectory,
  all legal actions plus the unchanged bare control; no outcome-based selection.
- Escape endpoint 1,440 TOTAL transitions: H300 plus 1,140 additional transitions.
- Compute timeout base 337.5 s / effective 675 s; six one-thread workers.
- Namespace `bl-c15-b2-natural-roots-20261022/v1`.
- `b2_allocate.derive(0)` is the lowest CLEAR complete allocation: 32 production
  seeds plus excluded canary seed 1988948902. Full audit, positive controls,
  explicit banks and training exclusions retained. Empty verified ledger included.
- No scientific/selection/controller change. The only runtime change exports
  operational timing and a separate canary admission allowlist; source pins and
  fresh real preflight are required for it. Outcomes stay sealed until the sole
  32-unit Mac merge.

## Admission rule implemented

`b2_admission.evaluate` rederives the unchanged qualified family proof and hashes
all recovered and actual Drive backup bytes. It parses `canary-admission.json`
only, never `trajectory.json`. Required: terminal PASS and release proof, at least
one actual serial/parallel root check, identical branch bytes, all native execution
counters reconciled, worker threads torch/numba 1/1, six workers and exact design.

EST = supervisor whole-session seconds minus only the measured canary serial
replay phase; missing phases conservatively use the whole session. Production
requires EST <=427.5 s (7.125 min), same commit/spec/ledger/checkpoint uploads and
`--canary-receipt`. An arbitrary supplied estimate cannot bypass this rule.

Only if EST exceeds 427.5 s and EST minus 4800/measured-parent-fps is positive and
<=427.5 s, `RESCALE_REQUIRED` permits 9600 decisions, unchanged grid/endpoint/N,
a fresh whole-round 33-seed CLEAR audit, new digest and new excluded <=1-u canary.
A create-only claim shared across worktrees consumes this authorization once,
even when the original spec/report is reused with a different output directory.
A rescaled-canary admission failure, missing timings, non-timing failure or failed
arithmetic goes to the judge. Eligibility/yield/outcomes never trigger anything.
No rescale is executed by normal admission or production preparation.

## Qualified transport and pack status

Tree `research/colab_cli` MUST remain
`553af3f692515b2bf66173356fe4d099acc96942`. Qualification record:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json`;
qualified commit `f21a92edb86550df5263dbed4be7e2ff0bfa4c17`. Each pack and generated
manifest records this file's SHA, accepted-result SHA and tree. Preparation
checks acceptance PASS and published pin/current runtime source equality; the
family gate rechecks transport identity. No qualified CLI bytes were changed.

External `round16-20261022/dispatch-pack` contains a full published source pin,
one excluded-canary request, all 32 production requests and executable adapters.
These are explicitly BLOCKED requests, not fabricated validated manifests.
They finalize real manifests only when Claude supplies a passing current-source
Mac preflight and live assignment-rate map; production additionally requires
successful real canary admission. This is necessary because the owner reserves
the Mac for M1 and forbids executing that preflight in this task.

## Exact Claude commands

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-b2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
REPO=/Users/josenunez/Projects/ml/snake-dqn-b2
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-b2/round16-20261022
PACK="$ART/dispatch-pack"
INPUTS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-b2/preflight-20261022-01/inputs
SPEC="$ART/registration/b2-design.json"

# CLAUDE ONLY, after M1 releases the Mac: AC, lid open, shared lease free.
# Existing held() refuses a busy lease; never steal it. This runs 3 tiny real units.
mkdir "$PACK/preflight"
SNAKE_DQN_PYTHON="$PY" nice -n 10 bash "$REPO/research/bl_served_search/run_b2.sh" \
  --tiny-preflight --inputs "$INPUTS" --out "$PACK/preflight/preflight.json"

# Before EVERY allocation: read-only account evidence, no provider mutation.
/Users/josenunez/.local/bin/colab --auth oauth2 usage
/Users/josenunez/.local/bin/colab --auth oauth2 sessions
# Write PACK/other-assignments.json as endpoint -> reviewed positive u/hour;
# include all live assignments. Use {} only if sessions independently proves empty.
# Confirm L4 rate fits the pinned 2 u/hour price; unknown/excess rate refuses.
# Reconcile spent units + concurrent worst-case liabilities against 40 u.

# Finalizes the canary manifest after preflight, then calls ONLY ops dispatch.
bash "$PACK/canary.sh"
# Dispatch returns after allocation calibration, NOT terminal success.
# Wait for terminal PASS, verified stop, exports and Drive backup.
CANARY_RECEIPT="$PACK/canary-manifests/canary/receipt.json"
bash "$PACK/admission.sh" --receipt "$CANARY_RECEIPT"

# Only ADMIT_UNCHANGED permits production. Refresh account map before each call.
# C first; B only after DENSITY-2 releases it; A remains P1. No automatic loop.
bash "$PACK/production.sh" --index 0
# Each call materializes the 32-index manifest inventory with current accounting
# and dispatches ONLY the requested index, one trajectory per allocation.
# Wait for its terminal receipt/release; stage and verify before next in this lane:
RUN="$PACK/production-00-manifests/trajectory-00"
SNAKE_DQN_PYTHON="$PY" bash "$REPO/research/bl_served_search/run_b2.sh" \
  --stage --spec "$SPEC" --source "$RUN/exports/trajectory.json" \
  --receipt "$RUN/receipt.json" --index 0 --exports "$PACK/staged"
# Repeat explicitly for missing indices 1..31, after each prior unit is staged.
```

The adapters enforce evidence admission and reject an index already in the staged
ledger. They do not schedule/monitor lanes or enforce the stage-before-next lane
barrier: Claude must enforce that operator sequence and cross-allocation accounting.
Never interpret the return of dispatch as completion. A mechanically failed unit
with a verified immutable completed export is retained after stop/backup; never
recompute it. Missing units get at most one prepared mechanical completion series,
with a new ledger and same-input real canary, never an automatic retry. Second
failure goes to the judge. A fresh recovery pack/output is required; original
pack/output/receipt bytes remain immutable.

Only for a verified `RESCALE_REQUIRED`, Claude may execute the one pre-registered
mechanical option below. It is not a yield remedy and is never automatic:

```bash
"$PY" -m research.bl_served_search.b2_admission --spec "$SPEC" \
  --canary-receipt "$CANARY_RECEIPT" --out "$PACK/admission-rescale.json" \
  --rescale-out "$ART/mechanical-rescale-01"
# Then build a fresh pack against mechanical-rescale-01/registration/b2-design.json
# using the SAME published pack.json commit; rerun no gameplay on the Mac.
# Current-source tiny preflight can be reused (design is not a source change).
# The new spec/ledger MUST receive its own full-design excluded Colab canary.
```

After exactly 32 staged verified units, `--make-ledger` to a fresh path and one
`--merge` as in HANDOFF.md. Keep alternative outcomes sealed until that merge.
<8 distinct H300-positive lives prints INSUFFICIENT; >=8 LIBRARY_COMPLETE.
Neither verdict is C15/L1 timing, strict, serving or release evidence. The
judgment's yield sub-readings and parent-hazard report remain obligations at that
single look. No serving changes are authorized.

## Units forecast and validation

At frozen 2 u/hour: production at admission ceiling is 7.6 u, canary <=1 u,
forecast <=8.6 u; optional timing rescale adds <=1-u canary (<=9.6 u forecast).
Production at session wall cap is 15.2 u, plus <=1 u canary =16.2 u wall-derived
worst case; manifest liabilities including reserves are 20 u original and
40 u all attempts. Unit cap 0.59375 u, reserve 0.11875 u, wall 14.25 min;
canary cap 1 u, reserve .2 u, wall 24 min. Already doubled, never double again.
12-minute canary estimate is its conservative admission envelope, not a measured
throughput result. Actual production estimate is unknown until the real canary.

Light tests and independent Codex-V receipt are recorded in the sibling review
receipt. No real entrypoint preflight, multiprocess test, benchmark or heavy suite
was run on the M1-reserved Mac. Two engine tests (spawn pool and real restore)
were explicitly excluded; real family parity is still required on Colab.
