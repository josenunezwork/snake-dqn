# Review-03 repairs and pack -06

Status: HOLD_OFFLINE pending the separate exact-package review-04 ACCEPT and
fresh final-only prerequisite. No Colab calls or dispatches are authorized by
this implementation record.

Authority: `big-lever/exp/fable-20261024/JUDGMENT_L4_V2_REVIEW03.md`, SHA256
`77fd5e294fdf183349a4b0adbd6e2cc36ccbb8c458e4d1dd26af474ad39236f6`.
This record supersedes the preparation instructions for pack -05. Existing
pack/proof bytes remain immutable; pack -05 receives only a sibling marker.

- B1: failed transfer parents match the prospective injection selector through
  the realized injection ID, failure type, direction, remote suffix and frozen
  parent-event count. Other failures, including the same suffix without the
  realized injection identity, still count as C6.
- B2: the existing single bounded end-of-session census also observes only the
  SHA256 and byte size of `client-start.json` and `client-finished.json`. Local
  download identities/digests are retained until that census. The producer
  compares the independent observations before recording verification. A
  mismatch produces C6; absent evidence remains EVIDENCE_MISSING. The census
  does not move earlier in the lifecycle, retry, or clear an uncertain fence.
- S1: the differential against transport tree `4eefe6b1` includes normal,
  slow/failed-census, near-deadline and designed-watchdog lifecycle paths.
  The census is an explicit authorized observation delta, including its
  latency. The real CPU acceptance producer also reaches the v2 classifier.
- S2: the generic requalification command identifies lane source/lineage
  independently of the registry flag and refuses main's registry before
  execution or a consumed-plan marker. A detached source copy is covered.

`admission.require_review` requires package commit, transport pin and tree,
complete current/pinned source inventory (including executed ops wrappers),
and exact plan/proof/trace/V3-helper plus R/J/H/review03 authority digests.
The review record uses `artifact_sha256` keys `plan`, `proof`, `trace`, `helper`,
`authority_sha256` keys `R`, `J`, `H`, `review03`, and `transport_tree`.
`review_artifact_bindings(repo, package_commit)` computes those fields for an
independent reviewer; it does not issue ACCEPT or waive any admission gate.

Offline proof regeneration uses `research.rpc_qualification.fake_proof` with
the create-only output `FAKE_CLI_PROOF_20261024_V2_06.json`. Pack preparation
uses `research.rpc_qualification.manifests prepare` and fresh destination
`big-lever/round-24/rpc-qualification-06`. Exact identities, verification logs,
and the V3 helper live in the frozen external pack and implementation handoff.

After review-04 ACCEPT, freeze this branch until attempt 2 settles. The operator
must establish that the ruler is terminal before the prerequisite, and that
the A100 yields at a job boundary before the three-concurrent-L4 wave. Order:
final-only CPU+L4 prerequisite (both v2 PASS), required L4 wave, mandatory CPU
injection proof, optional CPU loops. Publish ledgers before the one look.
EVIDENCE_MISSING refuses admission without closing the lane; a counted v2
fault in any dispatched session ends attempt 2. No main merge is permitted.
