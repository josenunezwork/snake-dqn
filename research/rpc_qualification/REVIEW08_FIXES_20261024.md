# Review 08: prospective C6 observer accounting

Authority: Astra's `JUDGMENT_L4_V2_REVIEW07_C6_SCOPE.md`, section 3,
SHA256 `739786fa3fbb710b8bf7d607ea31e61f2fdb6887f297a0720830d484c1b5a9b1`.

This successor repairs evidence and classification only. The raw observer
`transfer_attempt` remains the complete per-child poll ordinal. The classifier
derives the C6 allowance count after validating the pre-dispatch injection
manifest; only failed, exactly matched `planned_absence_probe` exclusions are
discounted. Real failures and completed reads remain attempts. Absence probes
do not reset the count, and arbitrary observer reads receive no exemption.

Recovery still requires the same prospective parent/child/remote lineage,
complete raw ordinal census, a later completed read with its own verified digest,
and a matching verified whole-transfer record. Earlier completed reads without
their own verification remain nonqualifying. Normal transfer-helper limits stay
at three attempts. The real refused fixture sequence retains raw ordinals 44
and 82; its prospective counterpart has two actual attempts only with all 80
lawful absence exclusions and final verification. The historical receipts
remain `EVIDENCE_MISSING`, with zero established counted faults.

`admission.py` binds this authority, `rpc-qualification-10`, the regenerated
`FAKE_CLI_PROOF_20261024_V2_10.json`, V7 helper, and the fresh create-only
`ASTRA_L4_V2_REPAIR_REVIEW08_ACCEPT_20261024.json` path. The review request is
`REVIEW08_BINDINGS.json`; it is not an ACCEPT. The consumed -08 ACCEPT and the
unissued -09 ACCEPT cannot authorize this changed package.

The successor includes the complete qualification proof, behavior differential,
source inventory, plan, preservation inventories, verification report, helper
dry-run, and ACCEPT-ready artifact bindings. Packs -08 and -09 are preserved;
only a sibling `rpc-qualification-09.SUPERSEDED_BY_10.json` marks supersession
for future review. Independent Astra ACCEPT is required before any fresh
prerequisite planning or execution. No prerequisite or provider call is part
of this implementation.

The transport pin remains `d68ba07c350d3f642879ee6df2cf9af9d34f0601`, tree
`6415984f6cedc17a828a6a0d8fffc4519f7cb273`. No command, cadence, retry,
deadline, cap, fence, release decision, or transport behavior changes.
