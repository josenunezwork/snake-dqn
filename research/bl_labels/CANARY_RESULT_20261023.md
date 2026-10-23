# F0 CPU canary review — 2026-10-23

**Verdict: INCOMPLETE; incomplete-but-informative, not admitted for tranche 1.**
The registered outcome look remains closed. This is not an INVALID result and
does not justify discarding or recomputing verified units. No production,
student fitting, strict-gate or release authority follows from this review.

Authority: round-24 `JUDGMENT.md` §4 F0, frozen `INTENT.md`, and the
owner-authorized whole-session replacement in `SESSION_CAPS_AMENDMENT.md`.
The evaluator is the existing `exports.aggregate_attempts` path at source
`816b366912eb41a499a8c64161b1b16e3ef90a39` (unchanged in this review).

## Verified evidence

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0`.
Original `canary-output-05/receipt.json` SHA256:
`bfd06b23cc5606f889810699ff4cfec16bb2f1b77e9466d8d61ff0db5167e343`.
The immutable review evidence is in `canary-review-06/`:
`resume/verified-ledger.json`, `aggregate/verified-ledger.json`, and
`mechanical-verification.json`, each with SHA sidecar and completion marker.
`provenance-verification.json` also checks the receipt-bound canonical
`canary-output-05/manifest.json`, all manifest input hashes, all exported sizes,
and the pinned qualified CLI tree. The canonical manifest SHA256 is
`373efb17799984071fcace5857570c14ab03d7677b0b3a6fa867a4f55a266b49`;
its dispatch-refreshed assignment accounting explains its different bytes from
the original pack template.

All **44** receipt-listed payloads match their SHA256 and completion triplets.
They include **42** life/session payloads. The registered aggregate validates
namespace, frozen allocation identity, duplicate IDs, session/life dependencies,
and INVALID markers. It confirms **31/36 completed session parents**, comprising
12 frozen, 12 mixed, and 7 scripted. Missing are `gate-scripted-007` and
`watch6-000` through `watch6-003`.

The runtime's 36 `completed` strings list visited allocations, including those
returning INCOMPLETE; they do not establish completed immutable sessions.
This follows directly from `runtime.run` appending every returned allocation
before deriving the overall incomplete status.

## Admission clauses

| Registered clause | Evidence | Result |
| --- | --- | --- |
| Immutable export integrity | 44/44 receipt payloads verified | PASS for exported evidence |
| Restore digest checks | 1,012 checks in completed sessions; restoration raises on mismatch | No reported mismatch in completed evidence |
| Bare reroll audit | 82 rerolls; 0 bare-audit mismatches in completed sessions | No reported mismatch in completed evidence |
| INVALID precedence | No durable INVALID marker; runtime `invalid_reason=null` | No INVALID observed; missing units remain unverified |
| Fixed allocation complete | 31/36 sessions | INCOMPLETE, takes precedence |
| Rescue-bearing lives | Registered aggregate `summary=null` | Withheld until allocation complete |
| Decisive labels/life and H1440 reversal rate | Registered aggregate withholds outcomes | Not computed; no partial outcome look |
| Rescue-bearing lives/Mac-worker-hour equivalent ≥8 | Neither complete numerator nor measured Mac denominator exists | PENDING_MAC_WORKER_EQUIVALENCE; no PASS/KILL inference |
| Mac recorded-action replay audit | Required 64-root audit not run in this light review | Pending an unleased quiet gap |
| Stall census | Runtime says `NOT_MEASURED` | Unknown; cannot assert zero stalls |

CPU runtime: Linux x86_64 (no more specific CPU model exported), nproc=2,
two workers, one numerical thread each. Runtime elapsed **6465.875398051 s**;
supervisor elapsed **6609.8826731670415 s**. Runtime worker-hours are
**3.5921529989172223 Colab worker-hours**. This is explicitly not a measured
Mac-worker-hour denominator. A hypothetical or historical conversion cannot
be substituted for the registration's measured ruler.

Completed-session worker-second buckets are play 5848.144230246039,
restore/replay 147.68554597700336, doom alternatives 4157.488650483004,
ordinary 1393.0782356499972, audit 854.9947302329992, and export
1.7744422880005004. These omit missing-session work and are operational
diagnostics, not the whole-canary throughput denominator.

The receipt proves `stop_verified=true`, with no cleanup error. The last
accounting crosscheck reports 0.1460503710694443 own elapsed units and
`WARNING_UNEXPLAINED_DEBIT` (account debit below prediction by
0.038802918781260964 u); that warning is retained, not recast as science
INVALID or a free retry. The session budget assigned 6465.125438631978 s to
collection after measured setup and a 540 s export/margin reserve. The cap
stop is consistent with that collection deadline.

## Next permitted steps

Use the verified resume ledger for any manually admitted mechanical completion;
preserve N, namespace, source identities and all existing payloads. At most two
completion attempts are permitted, only after verified stop and missing-unit
checks, and all prior setup/failures must be charged against the frozen series
envelope. This review does not resize that envelope or dispatch a completion.
If the remaining envelope cannot fund completion, obtain a prospective envelope
amendment before dispatch, retaining this incomplete result and its evidence.
There is no scientific basis here for a from-zero rerun or an INVALID re-canary.

F1 can independently measure placement on an outcome-blind frozen 64-root bank.
It cannot supply a Mac conversion merely by comparing two x86 hosts. Tranche 1
still needs the completed F0 aggregate, measured Mac equivalence, and required
Mac action-tape replay audit before admission.

Independent Codex-V receipt: `canary-review-06/CODEX_V_CANARY.json`, SHA256
`56c33c8489497e7f0dd3687fe1e7ffc2e00112c9a864899ff550009f1c06e04f`.
The verifier independently confirmed the export hashes, completion metadata,
integrity counters and outcome-look discipline, with 19 narrow tests passing.
The root's single-process F0/CLI guard selection passed 109 tests in 8.38 s.
