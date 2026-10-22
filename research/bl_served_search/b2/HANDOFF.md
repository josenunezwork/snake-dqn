# PLAN_v3 B2 — prepared code, NOT_READY for dispatch

Worktree: `/Users/josenunez/Projects/ml/snake-dqn-b2`, branch `c15-b2-20261022`.
Authority: external `big-lever/PLAN_v3.md`, SHA-256
`d23690617519262bd1b1a3e6c5d8b414ef952003b5b92c86842c9f5fa4210201`, read in full.
No B2 allocation, family canary, production trajectory, result look or serving change has run.

## Blocking facts

1. PLAN_v3 §1 B2 fixes 32 trajectories, continuous respawn, FRP3 s12+v8,
   native kernel, at most four hash-selected L>=800 roots, all legal H300
   alternatives, H1440 escape continuations, death causes and complete snapshots.
   It does not state the trajectory horizon or candidate-root grid, nor whether
   H1440 is total depth or 1,440 additional frames after H300. The independent
   audit found a ten-minute advisor proposal, but that proposal specified 24
   trajectories; it is not authority to fill missing frozen design values.
   A dated pre-data ruling must supply those fields. `DESIGN_PENDING.json`
   deliberately has no production defaults. No cap, N, endpoint or frozen source
   was changed to address these omissions.
2. The real shell-entrypoint tiny Mac preflight was attempted and refused before
   constructing gameplay: the shared Mac lease is held by `claude-m0-part-b`.
   Do not bypass, steal or replace it. Run the command below when Claude releases
   the lease. A separate one-transition snapshot diagnostic executed earlier;
   it is not the required shell-entrypoint preflight receipt and is not timing
   qualification. No complete preflight or production forecast is claimed.
3. No same-pin real-transport B2 family canary exists. `b2_package.prepare`
   refuses missing/stale Mac preflight and refuses production without the
   supervisor's real family proof. The canary must cover the identical uploaded
   design/ledger/checkpoints and source pin. A changed ledger needs a new family
   canary because the qualified gate binds input digests.

## Implementation

- `b2_engine.py`: GridBatchSim continuous-respawn Watch6, bare B8 s3 versus five
  independently carried FRP3 s12+v8 policies; kernel-only executed-row counters;
  no injected geometry. Bottom-four root hashing uses namespace, frozen
  trajectory index, hero life and selector frame. Selection finishes before any
  alternative is evaluated. Alternatives include every game-legal action,
  independently of the advisory safety mask. B8 retains its unchanged advisory
  bare action and normal-action fallback. Every H300 survivor
  continues the same restored branch to the explicitly frozen endpoint.
- Full root snapshots contain simulator arrays, ordered food, per-world RNG,
  runtime memory/carry, per-slot veto counters/state, phase, hero policy, life,
  decision, and Python/NumPy/Torch RNG state. Padded-grid aliases are restored.
  These internal pickle snapshots must only be loaded from trusted, hash-verified
  B2 exports under the matching source closure; they are not an untrusted format.
  The merged library references
  verified unit files rather than duplicating all snapshot bytes; use
  `b2.restore_library_root(library, staged_dir, fixture_id)` for the C15/L1 seam.
- `b2_units.py`: deterministic IDs, outcome-independent hash selection, create-only
  payload/sidecar/completion triples, Mac path/size/hash ledger, portable runtime
  ledger validation, exact verified-unit skipping, foreign/duplicate rejection.
- `b2.py`: explicit `--check`, `--make-ledger`, `--run`, `--stage`, `--merge` and
  tiny preflight. Stage checks verified release and terminal status (PASS, or a
  classified mechanical failure after a completed unit export), manifest allocation/index,
  broker export digest and actual Drive backup bytes before copying a unit.
  Merge verifies exactly 32 unique completed units before opening outcome JSON;
  recomputes candidate inventory from selector traces, checks branch horizons and
  death reconciliation, and restores each root without advancing gameplay;
  it counts distinct rescue-positive hero lives using the H300 bare-death /
  alternative-survival definition and separately reports escape-endpoint survival.
  Fewer than eight H300 rescue-positive lives prints INSUFFICIENT. Neither
  LIBRARY_COMPLETE nor INSUFFICIENT grants C15/L1 qualification or release.
- `b2_package.py` and `run_b2.sh`: offline manifest preparation and real payload;
  qualified transport remains unchanged. Six spawned one-thread branch workers
  merge in fixed action order; the excluded natural-root canary compares actual
  serial and parallel branch bytes. Zero exercised canary roots is NOT_READY,
  never evidence of no search value. Every allocation contains exactly one
  trajectory. Claude dispatches solely through `tools/ops/dispatch.sh`; verify
  release and the Drive-for-desktop backup before counting/staging that unit. A completed, hash-verified unit from a
  mechanically failed attempt is retained after verified stop and backup; it is
  never recomputed merely because the whole-job receipt says INCOMPLETE.
  A Colab runtime crash can therefore affect only its one trajectory. Drive is
  never mounted or accessed in the CLI runtime. The existing supervisor copies
  verified exports to Drive-for-desktop after release.

## Frozen input requirements

Before preparing, publish a create-only `b2-design.json` with sidecars. The schema
is `bl-plan-v3-b2/v1/design`, status `FROZEN_BEFORE_DATA`, exact plan SHA above,
`trajectory_frames`, `root_stride`, `root_grid: absolute-trajectory-decision`,
`workers: 6`, `escape_total_frames` (1440 total or 1740 if
additional), `unit_timeout_seconds`, new `namespace`, 32 distinct fresh uint32
`seeds`, a distinct fresh `canary_seed`, roster
`bare-b8-s3-vs-5-frp3-s12-v8`, a CLEAR Mac `freshness` audit whose
`seed_allocation_sha256` is `ceiling.digest_value([*seeds, canary_seed])`, and
`allocation_digest = ceiling.digest_value(the entire object before that field)`.
The full create-only `freshness-audit.json` is loaded, its allocation, historical
controls, training inventory and collisions are checked, and its digest is bound
by `freshness_report_sha256` in the design. It and its sidecars are uploaded.
The ruling, full freshness report and namespace audit must be retained with the
registration. Use the existing whole-allocation seed auditor, including history
and training exclusions; do not manufacture a CLEAR summary or replace seeds
individually. Missing fields remain NOT_READY.

## Budget — already doubled; never double again

PLAN_v3: canary <=1 u; production forecast <=20 u; effective series cap 40 u.
The implementation conservatively includes canary in the original 20-u envelope.

| Item | Base before PLAN's doubling | Effective envelope |
|---|---:|---:|
| Original canary | 0.5 u | 1 u |
| Original 32-unit production series | 9.5 u | 19 u |
| One mechanical completion series, including new canary | 10 u | 20 u |
| All attempts combined | 20 u | 40 u |
| Each trajectory allocation | 0.296875 u | 0.59375 u |
| Each trajectory reserve | 0.059375 u | 0.11875 u |
| Each trajectory whole-session wall limit | 7.125 min | 14.25 min |
| Canary reserve | 0.1 u | 0.2 u |
| Canary whole-session wall limit | 12 min | 24 min |

Rate is the reviewed L4 rate 2 u/hour, NOT doubled. Confirm it against current
provider accounting before admission; unknown or changed rates refuse dispatch.
Estimates must include allocation/bootstrap, trajectory, up to four full branch
sets, exports and release. Production estimate must be <=7.125 minutes per whole
allocation; a tiny three-frame smoke does not establish that forecast. A forecast
that does not fit is NOT_READY; preserve evidence and obtain a ruling rather than
raise caps or shorten the design. No useful-result speed claim exists yet.

Reserve at least three whole-session minutes for setup/export/release when fixing
`unit_timeout_seconds`. First applicable wall/unit/study cap wins. Charge all
canaries, failures, setup/export and any ops accounting allocation retry to the
same 40-u series ledger. Before each dispatch Claude must reconcile all receipts
and refuse a next allocation whose worst-case liability exceeds the remaining
series envelope. The manifest enforces its per-allocation caps; cross-allocation
series accounting belongs to the operator.

PLAN_v3 allows one prepared outcome-blind mechanical completion attempt; second
failure goes to the independent judge. The required resume declaration retains
`max_outcome_blind_retries: 2` for the ops schema, but this package does not use it
to expand PLAN_v3's one-completion allowance. Retry only after verified mechanical
failure, verified stop and missing units; use a fresh manifest/output/receipt and
Mac-verified ledger, never recompute an already verified trajectory. No automatic
completion retry exists. No N shrink or scientific-failure retry is permitted.

## Exact Claude commands

All commands start in the handed worktree. Nothing below authorizes taking the
Mac lease from another job. Paths named `registration` below are required future
frozen inputs; they do not exist yet. No production manifest is supplied as ready.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-b2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
REPO=/Users/josenunez/Projects/ml/snake-dqn-b2
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-b2
INPUTS="$ART/preflight-20261022-01/inputs"

# Safe now: explicitly reports NOT_READY for the unresolved frozen design.
SNAKE_DQN_PYTHON="$PY" bash "$REPO/research/bl_served_search/run_b2.sh" \
  --check --spec "$REPO/research/bl_served_search/b2/DESIGN_PENDING.json"

# Only after the current Mac lease is released; choose a fresh receipt directory.
mkdir "$ART/preflight-claude-01"
SNAKE_DQN_PYTHON="$PY" nice -n 10 bash "$REPO/research/bl_served_search/run_b2.sh" \
  --tiny-preflight --inputs "$INPUTS" --out "$ART/preflight-claude-01/preflight.json"

# After the pre-data ruling + audited allocation exist; initial empty ledger.
SNAKE_DQN_PYTHON="$PY" bash "$REPO/research/bl_served_search/run_b2.sh" \
  --make-ledger --spec "$ART/registration/b2-design.json" \
  --exports "$ART/staged" --out "$ART/registration/verified-ledger-check.json"
SNAKE_DQN_PYTHON="$PY" bash "$REPO/research/bl_served_search/run_b2.sh" \
  --check --spec "$ART/registration/b2-design.json" \
  --ledger "$ART/registration/verified-ledger.json" --inputs "$INPUTS"
```

The allocator builds the complete 32+1 seed bank and initial empty ledger after
Claude records the dated pre-data ruling as a completed JSON export. It rejects
whole collision rounds and retains every rejected audit; it does not run gameplay:

```bash
"$PY" -m research.bl_served_search.b2_allocate \
  --ruling "$ART/B2_PARAMETER_RULING.json" --out "$ART/registration"
```

To prepare the canary, write a fresh `other-assignments.json` from read-only
`colab --auth oauth2 usage` / `sessions`, with every live endpoint's reviewed
positive debit rate. Set `BACKUP` to an absolute directory under the actual configured
`GoogleDrive-<configured-account>/My Drive` root; arbitrary local backups and
symlink escapes refuse admission. Staging derives its exact backup destination
from that root plus the manifest job and receipt generation. Set `PIN` to the
published implementation SHA, and `EST` to the reviewed whole-unit
estimate; unresolved values are hard blockers, not example defaults.

```bash
"$PY" -m research.bl_served_search.b2_package \
  --spec "$ART/registration/b2-design.json" \
  --ledger "$ART/registration/verified-ledger.json" --inputs "$INPUTS" \
  --preflight "$ART/preflight-claude-01/preflight.json" \
  --out "$ART/canary-attempt-01" --backup "$BACKUP" --commit "$PIN" \
  --probe --estimate-minutes "$EST" --other-assignments "$ART/other-assignments.json"
# Set CANARY_MANIFEST to the one generated .json payload (not its sidecars).
"$REPO/tools/ops/dispatch.sh" bl-b2-canary-01 "$CANARY_MANIFEST" \
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state/canary-01" \
  --dispatch-txt "$ART/DISPATCH.txt"
```

Wait for terminal PASS, verified exports and session disappearance. Production
preparation repeats the package command with a fresh `--out`, omits `--probe`,
and adds `--canary-receipt /absolute/canary-runtime/receipt.json`. The unchanged
supervisor rederives the real family proof, including same pin and uploaded bytes.
Then dispatch each generated trajectory manifest through the same `dispatch.sh`
command pattern, with a unique name/state directory and fresh accounting. Do not
call `colab new/exec/upload/download/stop` or the supervisor directly.

The generated manifest executes the **real `--run` command** below through the
qualified payload; it is not a Mac command to invoke manually:

```bash
python -m research.bl_served_search.b2 --run \
  --spec "$COLAB_INPUT_DIR/b2-design.json" \
  --ledger "$COLAB_INPUT_DIR/verified-ledger.json" \
  --inputs "$COLAB_INPUT_DIR" --index "$BL_B2_INDEX" --out "$COLAB_OUTPUT_DIR"
```

After each production allocation, verify/stage it before dispatching the next unit
in that lane. `RUN` is the actual supervisor output directory, `INDEX` its frozen
manifest allocation index. Staging does not parse trajectory outcome JSON.

```bash
SNAKE_DQN_PYTHON="$PY" bash "$REPO/research/bl_served_search/run_b2.sh" \
  --stage --spec "$ART/registration/b2-design.json" \
  --source "$RUN/exports/trajectory.json" --receipt "$RUN/receipt.json" \
  --index "$INDEX" --exports "$ART/staged"
# After all units (or for outcome-blind mechanical recovery), use a fresh ledger path.
SNAKE_DQN_PYTHON="$PY" bash "$REPO/research/bl_served_search/run_b2.sh" \
  --make-ledger --spec "$ART/registration/b2-design.json" \
  --exports "$ART/staged" --out "$ART/registration/verified-ledger-final.json"
# ONE Mac look, only once the ledger contains all 32 verified trajectory IDs.
SNAKE_DQN_PYTHON="$PY" bash "$REPO/research/bl_served_search/run_b2.sh" \
  --merge --spec "$ART/registration/b2-design.json" \
  --ledger "$ART/registration/verified-ledger-final.json" \
  --exports "$ART/staged" --out "$ART/library.json"
```

Keep alternative payloads sealed until that merge. Export the create-only library
triple to the C15 fixture/L1 consumer; four distinct natural frozen/mixed doom
fixtures and their clock bars must still be measured on the Mac. No B2 receipt
stands in for those measurements, strict evidence or release authority.

## Review and verification state

The first independent Codex-V review used explicit
`CODEX_MODEL=gpt-6.1-sol` through `/Users/josenunez/.claude/bin/codex-handoff.sh`
in review-only mode. It requested corrections to legal-action coverage, reducer
semantics, full freshness-report binding and Drive-root/run binding. Those
corrections and targeted negative controls are implemented; the second review
receipt records the final reviewed hashes. Review is implementation evidence only.

The current preflight refusal is retained create-only at
`exp/bl-b2/preflight-20261022-01/blocked.json` in the artifact root, with
SHA/completion sidecars. No Colab units have been spent by this implementation.

Cross-platform limit: root metadata retains source float32 Q values and its actual
natural bare action. Mac restoration verifies that source decision and full state;
it does not assume Mac/x86 argmax equality. Colab labels remain development labels.
C15/L1 must construct their Mac all-branch reference on the restored natural state
and explain any Q-tie divergence before claiming action identity. No Colab result
or restored snapshot supplies a Mac timing, strict-gate or serving PASS.
