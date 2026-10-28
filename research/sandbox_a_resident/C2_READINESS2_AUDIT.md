# TRAIN-01 c2 admission correction audit

Astra implementation, starting at `b2f0ddd38cfed3035ab20966888ced41ba3d8bda`.
Independent review remains with Opus. This is offline tooling evidence, not a c2
seal, dispatch authorization, release decision, or science result.

## Refusal-path comparison

Compared every deleted/changed refusal in
`git diff e971d41f1adbbdc3e6bf655f66f33cc3849c04e5..HEAD -- research/sandbox_a_resident/training_resume.py research/sandbox_a_resident/training_preflight.py`,
then compared the working corrections. A read-only scout independently inventoried
the old predicates; Astra reconciled the inventory against the implementation.

| Baseline refusal or exception | Current enforcement / intentional compatibility change | Regression evidence |
| --- | --- | --- |
| Persisted terminal regular file, mapping, single JSON document, identity, no completed status | `_retained_files`, `_retained_json_records`, `_scan_json_faults`, `_check_retained_execution`; `allow_completed` exists only for explicit inspection | execution filename variants, retained admission, corrected clean capture |
| Terminal returncode 124 / nonmechanical child exit | Recursive execution check; `verify_mechanical_stop` still directly calls `_check_terminal_fault_evidence` on receipt terminal | child kill 137/-9/-15 reason twins; watchdog tests |
| Terminal error, last traceback/error, tails, canary and failed-export markers | Last fields accept only a complete `TRAIN01_PROGRESS {object}` scanned recursively. Tails parse progress lines; exception syntax is refused independently of marker membership. Export and scanner inventories retain failed-stage/faulthandler checks | every mechanical golden x every exempt diagnostic/transport surface; 36 actual retained-poll file twins; native child-23 capture |
| Distribution screen mapping and material gaps | Dedicated publication-level gate restored, before generic file scans | exact-message good/bad twin; nonmapping publication |
| Progress mapping, failed stage and error | Dedicated publication-level gates restored | separate exact-message stage and error twins; malformed publication |
| Stall-watchdog mapping and failed stage | Dedicated publication-level gate restored | exact-message twin; malformed publication |
| Symlinked output/training/header/poll/stack/summary | Whole-tree regularity check runs before opening scientific publications; summary retains explicit symlink-or-exists check | retained-sidecar/symlink suite; full scoped suite |
| Nonempty faulthandler stack | Publication gate and export descriptor check retained | existing exact-message regression |
| Completed summary or staged summary | All original summary suffix checks retained; explicit completed-output inspection can opt in | existing summary and corrected completion tests |
| Header corruption, mapping, preparation/trajectory phase | Strict JSON retained scanner; explicit nonmapping snapshot-header refusal restored; phase checks retained | scientific publication mapping matrix and retained failed-header tests |
| Every poll, including earliest / staged, regularity, exactly one mapping | Whole-tree scan plus poll loop; only explicit nonexecution JSONL can hold many records | retained filename / JSONL / early-poll regressions |
| Poll job/generation aliases, nested state/terminal shape and identity | Recursive normalizer; a state/terminal mapping carrying execution keys resets data vocabulary. Scalar state/terminal metrics remain ordinary data | retained identity matrix and metric terminal-count regression |
| Identity-matched-poll count | Requires both job and generation in one scope. Sole consumer is host-ENOSPC recovery, which requires at least one matched poll plus exact count/hash inventory. This strengthens identity proof | recovery suite; no receipt-backed consumer depends on weaker job-only count |
| Unknown statuses/reasons and malformed values | Execution status/reason checks remain strict; scientific scopes now have explicit per-scope producer status/reason allowlists. 32-hex run generations remain bound in data scopes | per-scope error/fail/nan/non_finite/unstable/rejected/skipped/unknown twins, reason and UUID twins |
| Receipt's own reason | Validated first; nested terminal reason only explains receipt-level returncode | top-level reason twin |
| Mechanical receipt status/failure class/cleanup/stop proof | Exact envelope with `stop_verified is True`, clean cleanup, class-specific cause; no missing-key exemption | every mechanical golden with absent stop_verified; accounting arbitrary-error twin |
| Duplicate/alias keys | Exact raw duplicates refuse everywhere. Normalized collisions refuse for execution vocabulary; unrelated metric keys retain raw spelling | normalized execution duplicate matrix; A/a and valueLoss/value_loss metric example |
| Broken JSON-looking log line | Bracketed text warnings may use raw scan. Damaged JSON with quoted execution vocabulary refuses; quoted/camelCase exit aliases are recognized in text | truncated execution fields x job/supervisor logs; quoted exit alias matrix |
| Mechanical transport records and CLI footer | Native control/transfer records exempt only in exact validated receipt scope. Receipt echo must equal validated receipt. Native terminal event has an exact key/identity/status/count shape. Footer must match validated cause and bound original path exactly | existing receipt/footer/historical twins plus novel exception injection across every exempt transport surface |
| Staged source archive/executable bytes | Existing pinned Git/member inventory verification retained; no exact gzip-byte requirement was introduced | source archive self-consistent-tamper regression |
| Sidecar contents and orphan digests | Bounded exact producer alphabets/prefixes; complete marker/digest still require valid payload; orphan full digest needs matching staged payload | sidecar corruption/symlink/orphan regressions |
| Partial payload proof | Full immutable binding or byte slice at native offset; empty writes now return `None`, never an iteration-0000 identity. Complete mutable JSON rewrites scan independently and never become ledger units | native 17 MiB three-chunk download + interrupted chunk-1 twin; mutable rewrite and empty-payload regression |
| Scan-root existence / unreadable directories | Missing root explicitly refuses. `os.walk(onerror=...)` refuses unreadable directories; never silently skips a root or subtree | missing-root and synthetic permission-denial tests |
| Git preflight boundary | No refusal was deleted from `training_preflight.py`. OSError raises ValueError; inherited GIT redirects are scrubbed. Seal HEAD-blob check precedes noncheckout sibling fallback | existing preflight Git boundary matrix; full packet suite |

No deleted baseline refusal remains justified solely by a generic text denylist.
The mechanical cause set is unchanged by this correction. It uses qualified
supervisor-owned facts and exact complete messages, not fake-provider stderr.
No retry, science cap, snapshot-selection policy, or completion-attempt count changed.

## Producer corrections and limits

The first-capture corpus is immutable and remains a compatibility corpus. Tests now
assert its exact admission baseline hash and `e971d41f` source ancestor. Its old
runtime timeout/lease fixtures represent retained-terminal shapes, not proof that a
self-releasing runtime always leaves those bytes available to the broker.

`fixtures/supervisor_goldens_v2` adds corrected captures: a native clean terminal
observed during retention, a real `sys.exit(23)` child, and runtime timeout/lease
self-release removing the fake assignment. The latter produce a session-disappearance
receipt with no downloaded terminal and are explicitly refused for ordinary
continuation. Live offline capture tests also preserve native runtime terminal/state
observations. Provider/host loss after downloading a previously native-produced tiny TRAIN-01
full-state publication, a verified 17 MiB byte fixture, and part of a second large
download is inspected, fault-twinned, and still refused as continuation
without a qualified stop/recovery proof. The large byte fixture is not a valid
training checkpoint and never enters a ledger.

Capture provenance records dirty status, HEAD tree, working-seam hashes, and the
runtime/trainer version boundary. New captures have no `receipt_fabricated` assertion.
Capture calls ordinary fixture helpers and a CLI method seam, not pytest internals
or globally patched subprocess.run. Replay commit writes use private object stores
and require the actual commit tree to be available. Temporary roots use the platform
temp directory. An unsuccessful host-loss capture is renamed `.failed`.

**F4 remains a qualified-transport limitation:** `runtime.py` measures the final
retention limit from job start. `supervisor.py` only supplies an explicit longer
limit for retained chains; ordinary c2 cannot be claimed to have a fresh full export
window. The harness no longer forces a 0.2-second hybrid clean state. No
`research/colab_cli` source was edited, and no c2 spec/seal was generated. A separate
transport repair/requalification or a reviewed safe operating envelope is needed
before relying on a full export window after a late completion.

**F11 publication boundary:** a field-name-only audit of the two existing historical
roots found 26 JSON files, no protected outcome-key matches, zero exported payloads,
and no terminal in either receipt. No result values were reported. No authoritative
F5 protected-field list was supplied or located by filename, so this is not F5
publication clearance. No push was performed; original evidence bytes remain intact.
