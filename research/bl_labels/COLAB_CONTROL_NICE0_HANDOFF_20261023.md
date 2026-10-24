# Colab control nice-0 operator handoff

This handoff records the implementation boundary for Astra's
`ALLOW_COLAB_CONTROL_NICE0` ruling. It is **prepared, not qualified for ordinary
overlap**. The sole next overlap action is one already-authorized, exact-pin
transport canary under the procedure below. This file neither creates an
authorization, funds a canary, starts a factory, nor runs a provider command.
No particular canary or overlap proof exists at this handoff; the procedure is
conditional on a later independently verified, already-authorized package.

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

`OP_PIN` below is deliberately a placeholder for the independently reviewed,
published operational commit. Set it to the full 40-character pin reported with
the independent receipt before any operator action; do not substitute the frozen
science pin or a later unreviewed commit. The separate dedicated CLI candidate
is pinned now: `CLI_PIN=3f26b4279601948ffa7f255a4127acad3336a52d`.

The two pins bind different source boundaries. `OP_PIN` identifies the
operational dispatcher, factory classifier, launch wrapper, and proof consumer;
`CLI_PIN` identifies the separately versioned broker candidate and its
`research/colab_cli` source closure. The current registry remains qualified for
the old tree. The candidate cannot be dispatched until the normal **final-only
acceptance source qualification** has independently qualified its exact tree and
updated the registry through that existing route. Do not hand-edit the registry
or infer qualification from the candidate checkout being clean.

## Preconditions for the one overlap canary

All of these conditions are required before dispatch. A failed condition is a
HOLD, not a reason to edit a manifest, raise a cap, or try another canary.

1. An independent focused verification has passed for the exact `OP_PIN`, and
   its create-only receipt binds this ruling, the policy,
   the new host-check v4 schema, the positive identity classifier, the operator
   renicer, both nice-0 launcher layers, and unchanged science closure.
2. The dedicated broker tree is clean, detached at `CLI_PIN`, and the completed
   final-only acceptance source qualification has made the normal qualification
   check accept both its `HEAD` and the exact manifest commit. A source-tree
   edit, a matching executable, or `-m research.colab_cli.supervisor` text does
   not establish qualification.
3. The chosen package is the **next already-authorized** transport canary. Its
   manifest, `DISPATCH.txt`, source pin, input digests, timeout, unit cap,
   reserve, retry envelope, and output/state paths are already sealed. The
   package's existing canary thresholds remain the only PASS thresholds.
4. Before it is sealed, the exact authorized-canary draft includes the following
   binding, in addition to every existing supervisor and resume/admission field:

   ```json
   {
     "colab_control_nice0": {
       "policy_id": "factory-colab-control-nice0/v1",
       "ruling_sha256": "2ec03d85be07d4d9264e8aba105567ee23b40e364313c0da91fd57c215b0d64a",
       "overlap_status": "PREPARED_NOT_QUALIFIED_FOR_OVERLAP",
       "authorized_overlap_canary": {
         "authority_path": "/absolute/path/to/the-existing-authorization",
         "authority_sha256": "SHA256_OF_THAT_EXACT_AUTHORITY",
         "quiet_host_required": false,
         "manifest_sha256": "CANONICAL_DRAFT_MANIFEST_DIGEST"
       }
     }
   }
   ```

   The `PREPARED_NOT_QUALIFIED_FOR_OVERLAP` value admits this one proof attempt;
   it does not admit an overlapping main dispatch. `authority_path` is absolute,
   must exist, and its bytes must match `authority_sha256`. `quiet_host_required`
   is the JSON boolean `false`: a quiet-host study, including P1, is refused.
   The exact sealed manifest must use `CLI_PIN` as its full `commit` and must be
   accepted by the qualified broker's normal commit check. `OP_PIN` remains the
   exact checkout head for the operational dispatcher and classifier source.
5. An ordinary W12 factory continuation is already lawfully running, with its
   normal fresh-attempt, health, timing, AC/lid, thermal, disk, RSS, slowdown,
   baseline, lease, worker-deadline, integrity, accounting, and export guards
   active. Do not start, rebaseline, preempt, patch, or restart a factory merely
   to create this observation window. Numerical tests remain excluded from
   baseline formation, window rearming, and throughput-sanity intervals.
6. There is no P1 quiet-window restriction or other study-specific quiet-host
   requirement on the candidate. No overlapping paid main job begins before a
   passing canary has been independently recorded.
7. The live renicer has been activated through its operator-owned process
   lifecycle, as described below. Replacing its script on disk is insufficient.

## Prepare the canary digest before sealing

The dispatch gate hashes a deterministic JSON representation of the full parsed
manifest after removing **only** its own
`colab_control_nice0.authorized_overlap_canary.manifest_sha256` field. It does
not accept a raw-file hash and it does not omit authority, budget, source, or
any other manifest field. Run this read-only snippet against the mutable draft
before the package's existing sealing step; it writes no file.

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
MANIFEST_DRAFT=/absolute/path/to/mutable-authorized-overlap-canary.json

"$PY" - "$MANIFEST_DRAFT" <<'PY'
import copy
import hashlib
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
canary = manifest["colab_control_nice0"]["authorized_overlap_canary"]
if canary.get("quiet_host_required") is not False:
    raise SystemExit("quiet_host_required must be JSON false")
bound = copy.deepcopy(manifest)
bound["colab_control_nice0"]["authorized_overlap_canary"].pop("manifest_sha256", None)
print(hashlib.sha256(json.dumps(bound, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
PY
```

Place the printed value in the draft's sole `manifest_sha256` self-reference,
then use the package's existing immutable-sealing procedure. Never edit a sealed
manifest, replace its sidecars, or recompute this value as a repair action.

## Exact overlap-canary procedure

The following variable declarations deliberately require the operator to supply
the already-sealed package paths. They do not create a new manifest or output
directory. Every path except `OP_PIN` is absolute.

```bash
OP_ROOT=/Users/josenunez/Projects/ml/snake-dqn-f0
BROKER=/Users/josenunez/Projects/ml/snake-dqn-colab-control-nice0-20261023.noindex
OPS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops
OP_PIN=REVIEWED_PUBLISHED_OPERATIONAL_PIN
CLI_PIN=3f26b4279601948ffa7f255a4127acad3336a52d
CANARY_NAME=FRESH_ALREADY_AUTHORIZED_CANARY_NAME
CANARY_MANIFEST=/absolute/path/to/the/sealed-canary-manifest.json
CANARY_STATE=/absolute/fresh/canary-state
CANARY_DISPATCH=/absolute/path/to/the/sealed-package/DISPATCH.txt

test "$(git -C "$OP_ROOT" rev-parse HEAD)" = "$OP_PIN"
test "$(git -C "$BROKER" rev-parse --verify HEAD)" = "$CLI_PIN"
test "$(git -C "$BROKER" status --porcelain --untracked-files=all -- research/colab_cli)" = ""
"$OP_ROOT/tools/ops/qualification.py" check \
  --repo "$BROKER" --record "$OPS/colab-cli-qualified.json"
"$OP_ROOT/tools/ops/qualification.py" check \
  --repo "$BROKER" --record "$OPS/colab-cli-qualified.json" --ref "$CLI_PIN"
```

The two qualification commands are post-acceptance checks, not a route to
qualify the candidate. With the presently old registry they must refuse. First
perform the established final-only acceptance source-qualification procedure on
the separate candidate; do not run a paid canary until that procedure has
completed and normal checks accept the exact tree.

Before the command below, independently record the ordinary W12 attempt ID,
its current armed/baseline state, and its normal process/lease evidence. Do not
modify those records. Then dispatch exactly once with a fresh name and fresh
state path:

```bash
OPS_DIR="$OPS" "$OP_ROOT/tools/ops/dispatch.sh" \
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

## Activate and verify the real renicer before the canary

The artifact `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/renicer.sh`
has been replaced with the shared-classifier wrapper, SHA256
`3640228e636c66f633230b74dfffef8f246bec10a490fb24e5401d361d30e100`.
It checks its source files against immutable code revision
`e2c1fb1f811915dbacfa2f86c6a32effb9a4f3c9`; the final operational pin must
retain those exact classifier/renicer bytes. The former substring loop is
preserved create-only at `ops/renicer.before-colab-control-nice0-20261023.sh`,
SHA256 `0335a7acc86a25b294cb650353cac78cc214e7b771dba780db5ffcee3670fb7b`.
No running renicer was stopped or restarted by this implementation; an old
in-memory loop can still be active. The replacement `tools/ops/renicer.py`
obtains the common classification through
`factory_persistence.process_rows` and its shared
`research.compute.colab_control` classifier, but these reviewed repository bytes
are not activated merely by writing or replacing a file.

Before the canary, the operator must use the existing operator-owned control
mechanism to stop the one verified old-loop process and start the reviewed
shared-classifier renicer. Do not target a process by a substring or restart an
unverified process. Record the old and new PIDs, native birth identities,
executables, exact native argv vectors, working directories, observed nice, and
the activated source SHA256. Verify the replacement's exact native identity with
`research.compute.factory_companion_argv.native_argv`, and verify that its first
full sampling iteration uses `process_rows`/the shared classifier. The factory
guard and this active renicer must agree about every admitted root and every
descendant. A failed native observation is a HOLD; no broad renice or daemon is
authorized as a workaround.

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

After a passing canary, create a **new create-only sibling** proof; never
overwrite the initial independent Codex-V implementation receipt. The proof is
consumed by `tools/ops/control_overlap.py` and therefore must be immutable with
its SHA-256 sidecar and completion marker. Its required top-level fields are:

- `schema: "factory-colab-control-overlap/v1"`, `verdict: "PASS"`,
  `policy_id`, `ruling_sha256`, `operational_pin: OP_PIN`,
  `cli_pin: CLI_PIN`, and `qualified_broker_source_closure` equal to the
  candidate's `HEAD:research/colab_cli` tree.
- A nonempty `independent_reviewer` and a `gates` object with each of
  `existing_canary_thresholds`, `staging`, `lease_renewal`, `export_verified`,
  `stop_verified`, `ordinary_w12`, `all_admitted_nice0`,
  `no_unauthorized_local_work`, `factory_guards_honored`, `no_factory_stop`, and
  `transport_guards_honored` set to JSON `true`.
- An `evidence` object whose `canary_manifest`, `canary_receipt`,
  `control_telemetry`, and `factory_process_telemetry` each provide absolute
  `path` and matching `sha256`. The referenced terminal receipt must have
  `stop_verified: true` and `cleanup_error: null`; it must bind the manifest's
  exact SHA-256, `commit: CLI_PIN`, and nonempty matching `job_id`.

Only then may a later immutable manifest set
`overlap_status: "QUALIFIED_FOR_OVERLAP"` and bind the sibling through
`overlap_receipt: {"path": "/absolute/path/to/proof.json", "sha256": "..."}`.
`control_overlap.py` rechecks these fields against the current exact
operational root, CLI checkout, and qualified source closure. The proof has not
been created and no overlap is qualified at this handoff.

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
OP_PIN=REVIEWED_PUBLISHED_OPERATIONAL_PIN
F4_CONTROL=/absolute/path/to/sealed-f4-control.json
F4_INPUTS=/absolute/path/to/sealed-f4-staging-directory

test "$(git -C "$REPO" rev-parse --verify "$OP_PIN^{commit}")" = "$OP_PIN"
test -f "$F4_CONTROL"
test -d "$F4_INPUTS"

F4_CONTROL="$F4_CONTROL" F4_INPUTS="$F4_INPUTS" \
  "$REPO/tools/ops/dispatch.sh" --mac --repo "$REPO" \
  bl-f0-d1024-w12-FRESH_NAME "$OP_PIN" -- \
  /bin/bash research/bl_labels/RUN_D1024.sh
```

`RUN_D1024.sh` refuses without its required F4 paths and performs its own sealed
input, source, speed-equivalence, calibration, checkpoint, output/state, and
admission checks before it reaches the factory driver. A refusal leaves the
existing scientific allocation, baseline, accounting, and immutable evidence
unchanged. Do not replace `OP_PIN` with
`61900af28932e47e22e332840fc4406559adb062`; that value remains the frozen science
closure, not the operational dispatcher revision.
