# Review-04 checker repairs and pack -07

Status: HOLD_OFFLINE. Final-only prerequisite unrun; independent review-05 ACCEPT absent.

Authority: JUDGMENT_L4_V2_REVIEW04.md, SHA256
75bcbbb9e6ae72ff6bcf728d9bab61a25127e43d61d2c5f83a73da73e65c3f22.
This prospective record supersedes pack -06 preparation instructions and the
publication/freeze wording in REVIEW03_FIXES_20261024.md. Prior bytes stay intact.

- F1: optional CPU run detection includes manifests, state, outputs, receipts,
  runtime records, dispatch reservations/copied manifests/logs and retained
  publication inventories. Every evidenced session is assessed. Missing
  manifests preserve demonstrable C1-C6 faults through the fallback; clean
  incomplete evidence blocks admission without closing the lane.
- F2: receipt status alone never becomes a counted injection fault. Child-only
  exits remain nonqualifying with zero counted classes. Release/cleanup failures
  retain classifier C3, proven session wall/debit breaches retain supervisor C3,
  and integrity failures retain C5. C1-C6 definitions and transport are unchanged.
- Final binding: before the one-look marker, require_review verifies the frozen
  package commit explicitly, current/pinned source inventory, plan/proof/trace/V4
  helper and R/J/H/review03/review04 authorities. The actual supplied plan digest
  must equal ACCEPT artifact_sha256.plan, including when an alternative plan is
  otherwise self-consistent. Review refusal consumes no look and creates no fault.
- Publication: HEAD never moves between the prerequisite and any later wave.
  After EVERY selected wave is terminal, exactly one evidence-only commit+push
  on ops-l4-faults-v2-20261024 is allowed, containing only the plan's
  docs/research/results/rpc-qualification-<ns>-ledger.json triplet (.json,
  .json.sha256, .json.complete), before the one look. Source/helper/plan/proof
  bytes stay frozen. Final source verification uses the package commit, not HEAD.
  Publication attribution: Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>.

Pack -07 uses FAKE_CLI_PROOF_20261024_V2_07.json and V4 guarded commands. Its
REVIEW05_BINDINGS.json is a binding request, never an ACCEPT. The independent
judge's ACCEPT belongs at the existing CLAUDE_V_L4_REQUAL_02_20261024.json path.
The V4 helper has the V3 command behavior with the fresh pack and commit pins.

After exact review-05 ACCEPT: final-only CPU+L4 prerequisite (two slots; ruler
terminal is not required) -> both arms v2 PASS -> three concurrent L4 sessions
(ruler terminal, A100 idle at a job boundary, three free slots) -> mandatory CPU
injection -> optional CPU loops only with free slots -> terminal-wave ledger
publication -> one look. No automatic dispatch or retries. A counted fault ends
attempt 2; EVIDENCE_MISSING blocks admission without closing the lane.

Offline proof generation runs all 17 qualification suites in-process at nice 15,
threads=1, with one differential child at a time. The transport tree remains
364c5d4b3276102f6ca907dfadcf28774854d076; main's CLI tree remains d0d92bdb.
