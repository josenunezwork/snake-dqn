# Refused prerequisite evidence

These are exact, immutable byte copies from acceptance-1792878940 under
big-lever/round-24/rpc-qualification-08/prerequisite-state/requalify-jpkn4_pg/
in the sibling snake-dqn-artifacts directory. Receipts and observers are copied
from outputs/cpu and outputs/l4; injection manifests are from manifests/.
The checkpoint is the 3848-byte acceptance toy payload, copied only to verify its
hash; tests never deserialize it. The log is the original ops/logs/
l4v2-attempt2-prerequisite-execute.log. No gameplay labels are included.

Authority: Astra JUDGMENT_L4_V2_PREREQUISITE_REFUSAL.md, SHA256
b8ffb06bc9d91a2e6dd57720a08cdb880141b50debb9c051d188e51db82d7261.

The raw receipts remain EVIDENCE_MISSING with zero counted C1-C6 faults.
Prospective-counterexample tests create separate in-memory operation records;
they never backfill or requalify either historical receipt. In a prospective
operation with the same raw poll sequence, the genuine timeout has ordinal 44
and the successful sidecar has ordinal 82. Recovery then exceeds the frozen
three-attempt transfer bound and remains C6. This is a disclosed prospective
conflict for Astra, not proof of a counted fault in the historical run.
