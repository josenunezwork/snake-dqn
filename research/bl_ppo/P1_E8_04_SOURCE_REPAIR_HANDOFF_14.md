# P1 -05 final repair preparation: handoff 14

Status: **HOLD — not READY_FOR_CLAUDE_CHECK**. No dispatch is authorized from
these templates. The local preparation is complete enough to expose the exact
instruction conflict: the ruling requires a new ninth control bundle/URL while
the request preserves all nine URLs with no restaging. The eight scientific
origin URLs remain unchanged. No Drive or Colab action has occurred.

The separate operational amendment is
`P1_REPAIR_ATTEMPT2_OPERATIONAL_AMENDMENT_14.md`. Authority hash:
`55b13197d9d92f8eeb1c895180f01993ca11908f51c74a60d9a67065a0550d13`.
Executable pin: `03e0af75c64538f53a6971e6fec5d9a173bc921b`.
Qualified CLI tree: `d0d92bdb572b770284a1fea6152417ffa5f06bea`.
The acctfix broker has that tree; do not fast-forward it to current main.

The prepared new bundle is `97b29ee24a324284b740b53d9fadf6d7b70044953f166715da6e3a927d1eacd1`.
It changes exactly four namespace JSON payloads plus four SHA sidecars. Other
bundle members, origin-import evidence, science and executable pin are preserved.
The immutable `preparation.json` includes an overlarge root-level diff caused by
the first diff formatter; `preparation-compact-05.json` is the equivalent leaf
diff sibling. The original evidence is not rewritten. Both real validate-only
controls passed at the exact pin, with no actor/refit/gate execution or runtime
output creation. The failed repair accounting record binds the original failed
receipt and release proof and records 10.51 u charged, never a forged PASS.

The second HOLD is an operational integration prerequisite: the current
every-second renicer has no dynamic exclusion/reload interface. A prepared
replacement exclusion implementation exists, but was neither installed nor
substituted for a live daemon. Do not kill workers, daemons or tmux to install it.
The operator must establish a reviewed lawful handover before using the nice-0
dispatch branch. Source code alone cannot prove that the actual renicer excludes
the entire supervisor/CLI tree. No field below may be marked true on that basis.

## Exact operator order after the holds are resolved

These are conditional commands, not a runnable dispatch authorization. Missing
final mapping, final fetch proof, final independent READY receipt or live evidence
must cause HOLD. Replacement and 1M remain held and have no new dispatch command.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-p1snap
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/y3/source-repair-10-20261023-05
BROKER=/Users/josenunez/Projects/ml/snake-dqn-acctfix
REVIEW=/Users/josenunez/Projects/ml/snake-dqn-p1snap/research/bl_ppo/P1_E8_04_SOURCE_REPAIR_CODEX_V_14_FINAL.json
```

Before arming, finalize the new bundle's independently reviewed mapping and run
one exact-inventory Mac preflight. No frozen URL tinkering or fetch loop:

```bash
"$PY" -m research.bl_ppo.p1_attempt2_pack finalize --output "$PACK" --urls "$PACK/drive-urls.json"
"$PY" -m research.bl_ppo.p1_repair_transport preflight --manifest "$PACK/canary-manifest.json" --output "$PACK/public-fetch-preflight-drive-05.json"
```

Obtain final independent Codex-V with exact manifest/mapping/fetch/accounting
bindings, zero findings and READY_FOR_CLAUDE_CHECK. Record its digest and the
amendment/ledger in CAMPAIGN_LOG and publish before dispatch. Preparation review
is not that receipt. Complete the reviewed renicer handover before arming; supply
all four planned root records to its exclusion (canary/repair, original and the
existing exact allocation recovery). Verify that the old policy cannot still
renice the control tree. Never alter the frozen broker transport.

For each stage, use a separate owned window; arm only after preparation. Set
STAGE to canary, then repeat the full window for repair after canary release,
independent canary verification and accounting reconciliation:

```bash
STAGE=canary
JOB_ID=$("$PY" -c 'import json,sys;print(json.load(open(sys.argv[1]))["job_id"])' "$PACK/$STAGE-manifest.json")
"$PY" -m research.bl_ppo.p1_attempt2_window arm --owner "$JOB_ID" --deadline-seconds 1800 --evidence "$PACK/$STAGE-arm.json"
```

Retain the returned owner token securely in the stage's local line check. Let
the canonical arm request cooperative drain; await the bounded factory cleanup.
Identify the actual active attempt from the current factory ledger and its
matching journal; set FACTORY_RESULT to its terminal runtime-result.json and
FACTORY_PID to its recorded launcher PID. These live values are intentionally
not fabricated in this handoff. Require wait/reap and worker cleanup, no live
descendants, absent primary Mac lease and re-admission blocked. Then:

```bash
"$PY" -m research.bl_ppo.p1_attempt2_window verify-drain --factory-result "$FACTORY_RESULT" --factory-root-pid "$FACTORY_PID" --evidence "$PACK/$STAGE-drain.json"
git -C "$BROKER" rev-parse HEAD:research/colab_cli
git -C "$BROKER" rev-parse 03e0af75c64538f53a6971e6fec5d9a173bc921b:research/colab_cli
git -C "$BROKER" status --porcelain -- research/colab_cli
/Users/josenunez/.local/bin/colab --auth oauth2 sessions
/Users/josenunez/.local/bin/colab --auth oauth2 usage
```

Require both trees exactly d0d92bdb, no working transport diff, zero other
assignments and calibrated L4 <=1.54 u/h with zero tolerance. Record immediate
inventory and all intervening P1 debit/liabilities in a new create-only
`$STAGE-reconciliation.json`, schema `bl-p1-attempt2-live-reconciliation/v1`,
status PASS, warning WARNING_UNEXPLAINED_DEBIT,
`additional_attributable_debit_units` nonnegative. Bind source evidence and UTC
time. Set EXTRA_DEBIT to that exact actual additional debit, never a guessed 0.

```bash
"$PY" -m research.bl_ppo.p1_attempt2_accounting cumulative canary --held-manifest "$PACK/canary-manifest.json" --failure-accounting "$PACK/failure-accounting-04.json" --reconciliation "$PACK/canary-reconciliation.json" --additional-attributable-debit-units "$EXTRA_DEBIT" --output "$PACK/canary-cumulative.json"
```

Claude now writes the exact canary line check below, with create-only SHA and
completion sidecars, after completing and publishing its campaign checks. Then:

```bash
"$PY" -m research.bl_ppo.p1_attempt2_admission canary --manifest "$PACK/canary-manifest.json" --output "$PACK/canary-admission/manifest.json" --codex-v "$REVIEW" --claude "$PACK/claude-canary-line-check.json" --cumulative "$PACK/canary-cumulative.json" --preflight "$PACK/public-fetch-preflight-drive-05.json" --broker "$BROKER"
/Users/josenunez/Projects/ml/snake-dqn-p1snap/tools/ops/dispatch.sh p1-repair10-05-canary "$PACK/canary-admission/manifest.json" --broker "$BROKER" --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap --dispatch-txt "$PACK/canary-admission/DISPATCH.txt"
```

DISPATCH.txt must be a fresh gate-specific operator authorization after all
checks, not this held handoff. The wrapper verifies absolute nice 0 and registers
the fresh actual control root before the supervisor starts. It refuses inherited
nice 15; it does not try to raise priority. Sample the entire root/descendant
tree, recording host load and nice at before-dispatch, during-staging and
export-release. Keep the arm, factory drain and exclusion through verified release:

```bash
"$PY" -m research.bl_ppo.p1_attempt2_window sample --root-record "$PACK/canary-control-root.json" --phase during-staging --evidence "$PACK/canary-staging-priority.json"
"$PY" -m research.bl_ppo.p1_attempt2_window sample --root-record "$PACK/canary-control-root.json" --phase export-release --evidence "$PACK/canary-export-priority.json"
```

The existing exact accounting-allocation retry receives root-record and
registration-evidence filenames with `-allocation-retry-1` before `.json`;
sample that real root if it is the allocated tree. This is not a canary retry.

Independent Codex-V must verify the actual new canary receipt: exact pin,
transport and repair input SHA/destination/transport; all nine VM fetches <=300s,
residual transfer <=150s, staging <=300s, recovered verified exports, endpoint
disappearance/cleanup, charge, and twice measured forecast <=200min at 6.16u.
The create-only `$PACK/P1_CANARY_VERIFICATION_CODEX_V_05.json` must contain
status CANARY_PASS, unresolved_findings [], attempt2_ruling_sha256,
repair_manifest_sha256 and canary_receipt_sha256. Historical -04 proof cannot
satisfy it. Any failure closes this authorization: no repair dispatch.

Wait for the wrapper to resolve its bounded exact allocation recovery and for
every registered control root to disappear. Select the actual final allocation
output, never the original failed-accounting receipt while recovery is active.
Use CANARY_OUT below for independent verification and all later charge/admission
bindings. Disarm only after fresh sessions/receipt release verification:

```bash
/Users/josenunez/.local/bin/colab --auth oauth2 sessions
CANARY_OUT="$PACK/canary-output"
if test -f "$PACK/canary-output-retry-1/receipt.json"; then CANARY_OUT="$PACK/canary-output-retry-1"; fi
"$PY" -m research.bl_ppo.p1_attempt2_window disarm --owner-token "$OWNER_TOKEN" --release-proof "$CANARY_OUT/receipt.json" --root-record "$PACK/canary-control-root.json" --root-record "$PACK/canary-control-root-allocation-retry-1.json" --evidence "$PACK/canary-disarmed.json"
```

For repair repeat arm (owner=repair job ID, deadline-seconds=12000), complete
drain, nice/exclusion/live account checks and fresh reconciliation. Substitute
the actual canary charge and preserve exactly 13.88 u liabilities:

```bash
"$PY" -m research.bl_ppo.p1_attempt2_accounting cumulative repair --held-manifest "$PACK/repair-manifest.json" --failure-accounting "$PACK/failure-accounting-04.json" --reconciliation "$PACK/repair-reconciliation.json" --additional-attributable-debit-units "$EXTRA_DEBIT" --canary-receipt "$CANARY_OUT/receipt.json" --output "$PACK/repair-cumulative.json"
"$PY" -m research.bl_ppo.p1_attempt2_admission repair --manifest "$PACK/repair-manifest.json" --output "$PACK/repair-admission/manifest.json" --codex-v "$REVIEW" --claude "$PACK/claude-repair-line-check.json" --cumulative "$PACK/repair-cumulative.json" --preflight "$PACK/public-fetch-preflight-drive-05.json" --broker "$BROKER" --canary "$CANARY_OUT/receipt.json" --canary-verification "$PACK/P1_CANARY_VERIFICATION_CODEX_V_05.json"
/Users/josenunez/Projects/ml/snake-dqn-p1snap/tools/ops/dispatch.sh p1-repair10-05-repair "$PACK/repair-admission/manifest.json" --broker "$BROKER" --cwd /Users/josenunez/Projects/ml/snake-dqn-p1snap --dispatch-txt "$PACK/repair-admission/DISPATCH.txt"
"$PY" -m research.bl_ppo.p1_attempt2_window sample --root-record "$PACK/repair-control-root.json" --phase during-staging --evidence "$PACK/repair-staging-priority.json"
"$PY" -m research.bl_ppo.p1_attempt2_window sample --root-record "$PACK/repair-control-root.json" --phase export-release --evidence "$PACK/repair-export-priority.json"
/Users/josenunez/.local/bin/colab --auth oauth2 sessions
REPAIR_OUT="$PACK/repair-output"
if test -f "$PACK/repair-output-retry-1/receipt.json"; then REPAIR_OUT="$PACK/repair-output-retry-1"; fi
"$PY" -m research.bl_ppo.p1_attempt2_window disarm --owner-token "$OWNER_TOKEN" --release-proof "$REPAIR_OUT/receipt.json" --root-record "$PACK/repair-control-root.json" --root-record "$PACK/repair-control-root-allocation-retry-1.json" --evidence "$PACK/repair-disarmed.json"
```

Before repair disarm confirm actual receipt/export verification, endpoint absent,
cleanup_error null and stop_verified true. Scientific source success separately
requires COMPLETE_SOURCE_CONTINUATION, unchanged critic/counters, real anchor and
suffix, and prefix MATCH. Any final repair failure exhausts this route; retain its
label/evidence/debit and STOP. Disarming only permits lawful factory continuation.

## Exact Claude line-check fields

Both create-only JSON checks require the following fields. A true value is a
record of an actual completed check, not a template default:

| Field | Required value |
| --- | --- |
| status | PASS |
| stage | canary or repair, matching this manifest |
| ruling_sha256 | fe0584595cca6f6fcb28071692e52e4fe8460f3ab90fc29429c027203ff9ebfc |
| attempt2_ruling_sha256 | 55b13197d9d92f8eeb1c895180f01993ca11908f51c74a60d9a67065a0550d13 |
| transport_ruling_sha256 | 2fb9e3914a5eb86e9cd5c42495a1d8213fa1475aa6cf0c72d14c83350de5bc5e |
| a3_3_exception_ruling_sha256 | f1bd0dca5499c1766addb2145e35dc3e6291c6d9d97f175425c68d7f0d9f3ee9 |
| manifest_sha256 | Exact final held stage manifest digest |
| codex_v_sha256 | Exact final READY independent review digest |
| cumulative_accounting_sha256 | Exact stage cumulative record digest |
| public_fetch_preflight_sha256 | Exact -05 nine-input Mac preflight digest |
| qualified_transport_tree | d0d92bdb572b770284a1fea6152417ffa5f06bea |
| campaign_log_recorded | true, only after published campaign/ledger record |
| origin_invariants_verified | true |
| empty_ledger_namespace_verified | true |
| publication_mapping_verified | true |
| exact_mac_fetch_verified | true |
| factory_drain_verified | true |
| factory_re_admission_blocked | true |
| control_nice_zero_verified | true |
| complete_control_tree_renicer_exclusion_verified | true |
| quiet_window_through_release_committed | true |
| account_inventory_zero_other_assignments | true |
| l4_rate_calibrated_le_1_54_zero_tolerance | true |
| ac_lid_thermal_x5_k1_m1_verified | true |
| attempt_limit_one_canary_one_final_repair | true |
| control_window.arm_path | /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/mac-timing-armed |
| control_window.owner_token | Actual owned stage arm token |
| control_window.root_record | Fresh absolute PACK/canary-control-root.json or repair-control-root.json |
| control_window.registration_evidence | Fresh absolute PACK/canary-control-registration.json or repair-control-registration.json |
| evidence | Fresh paths/SHA, PID/PPID/nice/start/load samples, active-attempt cleanup/journal, arm, lease, renicer handover, current account inventory/calibration, reconciliation and publication references |
| canary_verification_sha256 (repair only) | Exact new CANARY_PASS independent verification digest |

No missing final manifest, review, mapping, preflight, live fact or token is
invented here. Current preparation cannot pass admission. No old receipt, source
pack, released checkpoint, serving pin or pre-existing untracked canary file was
modified. No Colab, live worker, tmux, daemon or priority was touched.
