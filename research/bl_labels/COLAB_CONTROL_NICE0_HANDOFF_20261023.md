# Colab control nice-0 operator handoff

This handoff records the implementation boundary for Astra's
`ALLOW_COLAB_CONTROL_NICE0` ruling. It is **prepared, not qualified for ordinary
overlap**. The sole next overlap action is one already-authorized, exact-pin
transport canary under the procedure below. This file neither creates an
authorization, funds a canary, starts a factory, nor runs a provider command.

The operator needs the existing qualified-Colab workflow, the governing
canary's sealed package, and the W12 production admission procedures. The
desired outcome is narrow: prove that a positively identified, nice-0 Colab
control tree can complete its existing bounded control work while an already
lawful ordinary W12 factory continuation runs, without weakening any factory or
transport guard. It is not a science run or a throughput measurement.

## Binding state and boundaries

- Policy ID: `factory-colab-control-nice0/v1`.
- Ruling: `ALLOW_COLAB_CONTROL_NICE0`,
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/JUDGMENT_COLAB_SUPERVISOR_PRIORITY_20261023.md`,
  SHA256 `2ec03d85be07d4d9264e8aba105567ee23b40e364313c0da91fd57c215b0d64a`.
- Frozen legacy scientific pin: `61900af28932e47e22e332840fc4406559adb062`.
  This patch changes no scientific source, frozen loader, label/export rule,
  slowdown statistic, baseline, health/timing guard, or accounting code relative
  to operational base `3c5fc403`. Existing D1024 speed-equivalence and admission
  requirements still bind its separately frozen science pin.
- Factory shape remains W12 at nice 10. W11, a reserved core, affinity,
  real-time priority, negative nice, increased deadlines, added retry, and new
  resource quota are outside this authorization.
- The dedicated broker must be a separate worktree at
  `/Users/josenunez/Projects/ml/snake-dqn-colab-control-nice0-20261023.noindex`.
  Do not modify a qualified `research/colab_cli` tree in place. Requalification
  of that separate revision uses the established qualification route; this
  implementation must not hand-edit the existing qualification registry.
- The P1 final-repair canary is not an overlap candidate. Its governing ruling
  requires a quiet factory through verified release. Do not use P1 to establish
  this proof, and do not spend an extra canary, reuse its final repair attempt,
  or change its fixed caps, deadlines, or retry count.

`OPERATIONAL_PIN` below is deliberately a placeholder for the independently
reviewed, published operational commit. Set it to the full 40-character pin
reported with the independent receipt before any operator action; do not
substitute the frozen science pin or a later unreviewed commit.

## Preconditions for the one overlap canary

All of these conditions are required before dispatch. A failed condition is a
HOLD, not a reason to edit a manifest, raise a cap, or try another canary.

1. An independent focused verification has passed for the exact
   `OPERATIONAL_PIN`, and its create-only receipt binds this ruling, the policy,
   the new host-check v4 schema, the positive identity classifier, the operator
   renicer, both nice-0 launcher layers, and unchanged science closure.
2. The dedicated broker tree is clean, detached at the separately qualified
   broker revision, and the normal qualification check accepts both its `HEAD`
   and the exact manifest commit. A source-tree edit, a matching executable, or
   `-m research.colab_cli.supervisor` text does not establish qualification.
3. The chosen package is the **next already-authorized** transport canary. Its
   manifest, `DISPATCH.txt`, source pin, input digests, timeout, unit cap,
   reserve, retry envelope, and output/state paths are already sealed. The
   package's existing canary thresholds remain the only PASS thresholds.
4. Its immutable manifest includes the following binding, in addition to every
   existing supervisor and resume/admission field:

   ```json
   {
     "colab_control_nice0": {
       "policy_id": "factory-colab-control-nice0/v1",
       "ruling_sha256": "2ec03d85be07d4d9264e8aba105567ee23b40e364313c0da91fd57c215b0d64a",
       "overlap_status": "PREPARED_NOT_QUALIFIED_FOR_OVERLAP"
     }
   }
   ```

   The `PREPARED_NOT_QUALIFIED_FOR_OVERLAP` value admits this one proof attempt;
   it does not admit an overlapping main dispatch.
5. An ordinary W12 factory continuation is already lawfully running, with its
   normal fresh-attempt, health, timing, AC/lid, thermal, disk, RSS, slowdown,
   baseline, lease, worker-deadline, integrity, accounting, and export guards
   active. Do not start, rebaseline, preempt, patch, or restart a factory merely
   to create this observation window. Numerical tests remain excluded from
   baseline formation, window rearming, and throughput-sanity intervals.
6. There is no P1 quiet-window restriction or other study-specific quiet-host
   requirement on the candidate. No overlapping paid main job begins before a
   passing canary has been independently recorded.

## Exact overlap-canary procedure

The following variable declarations deliberately require the operator to supply
the already-sealed package paths. They do not create a new manifest or output
directory. Every path except `OPERATIONAL_PIN` is absolute.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-f0
BROKER=/Users/josenunez/Projects/ml/snake-dqn-colab-control-nice0-20261023.noindex
OPS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops
OPERATIONAL_PIN=REVIEWED_PUBLISHED_OPERATIONAL_PIN
CANARY_NAME=FRESH_ALREADY_AUTHORIZED_CANARY_NAME
CANARY_MANIFEST=/absolute/path/to/the/sealed-canary-manifest.json
CANARY_STATE=/absolute/fresh/canary-state
CANARY_DISPATCH=/absolute/path/to/the/sealed-package/DISPATCH.txt

test "$(git -C "$REPO" rev-parse --verify "$OPERATIONAL_PIN^{commit}")" = "$OPERATIONAL_PIN"
test "$(git -C "$BROKER" status --porcelain --untracked-files=all -- research/colab_cli)" = ""
"$REPO/tools/ops/qualification.py" check \
  --repo "$BROKER" --record "$OPS/colab-cli-qualified.json"
"$REPO/tools/ops/qualification.py" check \
  --repo "$BROKER" --record "$OPS/colab-cli-qualified.json" --ref "$OPERATIONAL_PIN"
```

Before the command below, independently record the ordinary W12 attempt ID,
its current armed/baseline state, and its normal process/lease evidence. Do not
modify those records. Then dispatch exactly once with a fresh name and fresh
state path:

```bash
OPS_DIR="$OPS" "$REPO/tools/ops/dispatch.sh" \
  "$CANARY_NAME" "$CANARY_MANIFEST" \
  --broker "$BROKER" --cwd "$BROKER" --state-dir "$CANARY_STATE" \
  --dispatch-txt "$CANARY_DISPATCH" --colab-control-nice0
```

The dispatcher copies the sealed manifest and creates a create-only control
admission at `$OPS/colab-control-admissions/$CANARY_NAME.json`. Its exec-only
wrapper creates `$OPS/colab-control-admissions/$CANARY_NAME.root.json` only
after observing inherited absolute nice 0, then starts the approved launcher.
A nonzero inherited nice, malformed manifest binding, qualification mismatch,
unreadable native identity, stale/PID-reused record, unexpected source closure,
or unapproved descendant must refuse before allocation or lose the exception;
none can be corrected with a text-based renice.

During the one canary, collect only non-scientific operational telemetry:

- root/wrapper/supervisor and each admitted CLI/helper's PID, PPID, native birth
  identity, executable, role, and observed nice;
- RPC elapsed and timeout, transfer elapsed, remaining bounded deadline, and
  staging/renewal/export/verified-stop milestones;
- the ordinary factory's attempt identity, armed/baseline state, and any existing
  health, slowdown, timing, or stop reason.

Do not inspect label outcomes or held-out payloads. The canary must exercise its
existing staging, lease renewal, export verification, and verified release path
while W12 continues under its unchanged ordinary guards. The factory monitor and
renicer must inspect every descendant: only positively admitted transport roles
are exempt, and an unapproved local numerical child remains a visible foreign
process under the normal policy.

PASS requires the canary's pre-existing thresholds, verified immutable exports
when that package expects exports, verified backend release, no unauthorized
local work, absolute nice 0 throughout the admitted control lifetime, and every
factory/transport guard still honored. Record the ruling SHA, operational pin,
qualified broker pin/source closure, unchanged science pin, independent result,
and canary evidence in the campaign log and the create-only independent receipt.

On any failure, drift, malformed observation, guard trip, or uncertain release,
preserve the evidence and halt new overlap admissions. Do not increase a
deadline, switch to W11, rebaseline, rerun the canary, consume a replacement
attempt, or dispatch an overlapping main job. The existing supervisor retains
its bounded cleanup and release responsibility. Any later quiet maintenance
window or new spend/retry question returns to a judge.

## D1024 dispatch form after independent implementation verification

After verified stop, activate changed factory code only through a fresh lawful
continuation, preserving frozen baselines and cumulative history. Never patch a
live factory. A standalone factory continuation does not require or establish
paid transport-overlap qualification.

This form is for a future already-admitted D1024 factory continuation. It is not
the overlap canary, does not manufacture an F4 input, and does not waive
D1024's sealed speed-equivalence, calibration, F4, ledger, output, state,
health, timing, W12, or funding gates. Use a fresh dispatcher name. The
operational pin is separate from the frozen science closure and must be the same
full reviewed pin used by the verified implementation receipt.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-f0
OPERATIONAL_PIN=REVIEWED_PUBLISHED_OPERATIONAL_PIN
F4_CONTROL=/absolute/path/to/sealed-f4-control.json
F4_INPUTS=/absolute/path/to/sealed-f4-staging-directory

test "$(git -C "$REPO" rev-parse --verify "$OPERATIONAL_PIN^{commit}")" = "$OPERATIONAL_PIN"
test -f "$F4_CONTROL"
test -d "$F4_INPUTS"

F4_CONTROL="$F4_CONTROL" F4_INPUTS="$F4_INPUTS" \
  "$REPO/tools/ops/dispatch.sh" --mac --repo "$REPO" \
  bl-f0-d1024-w12-FRESH_NAME "$OPERATIONAL_PIN" -- \
  /bin/bash research/bl_labels/RUN_D1024.sh
```

`RUN_D1024.sh` refuses without its required F4 paths and performs its own sealed
input, source, speed-equivalence, calibration, checkpoint, output/state, and
admission checks before it reaches the factory driver. A refusal leaves the
existing scientific allocation, baseline, accounting, and immutable evidence
unchanged. Do not replace `OPERATIONAL_PIN` with
`61900af28932e47e22e332840fc4406559adb062`; that value remains the frozen science
closure, not the operational dispatcher revision.
