# R32 acknowledged next actions — passive table

**Purpose.** This is the small, passive transition table ordered by R32-3. Claude
records coverage, READY, acknowledgment, dispatch state, and any later exact hashes.
It does not create a READY state, declared coverage, acknowledgment, command,
allocation, retry, or dispatch authority.

**Audience and prerequisite.** Claude is the operator of this table; Codex-V may
test stale-proof cases as part of R32-2. A row may be changed from `BLOCKED` only
when Claude records the exact prerequisite bytes and the evidence that its named
owner has supplied. During a coverage gap, record that gap. Do not infer an
acknowledgment from a commit, a process, a chat mention, or this template.

**Target and stop rule.** During coverage that Claude has explicitly declared,
the target is acknowledgment within 15 minutes of a recorded READY timestamp.
The target creates no automatic dispatch or retry. One false READY disables
automatic readiness classification; retain the row as a passive record and send
the evidence to the responsible owner.

**R32-3 cap.** This is text/light-check preparation only: 0.25 Sol-day plus
0.25 Claude-day, at most 0.125 Mac core-hour, and 0 units. Stop at the 0.25
Sol-day limit; do not turn the table into a dashboard project.

## Current template state

All operational fields below are deliberately unpopulated. This draft did not
inspect live processes, account balance, coverage schedules, receipts, or pending
handoffs. `BLOCKED` means "not established by this document," not that a job has
failed or that an operator is unavailable.

| Transition | Required immutable prerequisites and hashes | Next operator action (non-executable) | Responsible operator / resource | Status and blocked reason | READY timestamp / coverage / acknowledgment |
|---|---|---|---|---|---|
| P1 actual VM canary/admission handoff | **BLOCKED:** pack pin, current Codex-V receipt, Astra ruling binding, admission record, VM-canary prerequisites, and exact command were not bound here. | Claude verifies the P1-owned handoff against its own pack, then records its actual next action in the P1 record. | Claude / P1's separately admitted resource only. | **BLOCKED:** R32 may not restage, complete, or reserve P1; P1 takes priority over R32's optional transport probe. | **NOT RECORDED** / **NOT DECLARED** / **NOT RECORDED** |
| F4 D256 → D1024 collection | **BLOCKED:** immutable D256 predecessor-fit metadata, required T-EQ/calibration evidence, D1024 plan, ledger, receipt hashes, and admitted funding are not bound here. | F4's existing operator follows its registered serial chain; Claude records the exact command only after F4's own prerequisites are verified. | Existing F4 owner / F4-qualified path. | **BLOCKED:** F4 registration and factory ownership are outside R32; no partial dose, refit, extra reserve look, or pipeline action is authorized. | **NOT RECORDED** / **NOT DECLARED** / **NOT RECORDED** |
| D1024 fit → joint Look R-A | **BLOCKED:** final D1024 bytes, source/recipe digest, reserve census, Look R-A receipt, and required F4 hashes are unknown here. | Existing F4 owner performs only the registered fit and look order; Claude records any later READY state from the F4 evidence. | Existing F4 owner / Mac only where F4 registers it. | **BLOCKED:** D1024 must precede Look R-A; this table cannot create a reserve look or decide funding. | **NOT RECORDED** / **NOT DECLARED** / **NOT RECORDED** |
| D4096 funding/admission → F4 gameplay | **BLOCKED:** F4's funding disposition, budget projection, D4096 collection/fit/probe/Look R-B evidence, final checkpoint SHA-256, and fresh-bank plan are unknown here. | Existing F4 owner applies its registered stop/advance rules; Claude records the exact F4-controlled next action only after the governing F4 artifacts exist. | Existing F4 owner / registered F4 placement. | **BLOCKED:** F4's D4096 route remains its own serial route. R32 does not buy collection, alter caps, or send a checkpoint to gameplay. | **NOT RECORDED** / **NOT DECLARED** / **NOT RECORDED** |
| F4 H5000 → conditional Watch6 | **BLOCKED:** F4 H5000 decision, all registered GO-bar evidence, Watch eligibility, exact bank and controller hashes are unknown here. | Existing F4 owner follows the conditional Watch rule in F4; Claude records whether Watch is `NOT_RUN_BY_RULE` or actually due from the governing F4 disposition. | Existing F4 owner / Mac only per F4. | **BLOCKED:** no outcome is inferred. A dose failure may leave product work unrun and does not imply that it passed. | **NOT RECORDED** / **NOT DECLARED** / **NOT RECORDED** |
| R32 rehearsal/fault report → consumer-readiness report | **BLOCKED:** one traceable exported-byte consumer manifest, source/fixture/consumer hashes, Codex-V 3/3 and 12/12 matrix, and blocked-edge list are not bound here. | Sol and Codex-V complete their isolated R32-1/2 evidence; Claude records the report rather than treating a fixture as production authority. | Sol/Codex-V, then Claude / admitted light-check capacity only. | **BLOCKED:** synthetic/rehearsal evidence cannot establish strict efficacy, paced timing, Watch benefit, serving qualification, or release. | **NOT RECORDED** / **NOT DECLARED** / **NOT RECORDED** |
| Fixed-D4096 delivery contingency | **BLOCKED:** qualifying F4 `TOP_PASSED_DOSE_FAILED` flag in a published registered disposition; actual final D4096 SHA-256; no top-controller guard failure; and a new Astra-judged, pre-data funded registration are all absent from this draft. | Claude may submit the complete future packet to Astra only after every prerequisite exists; Astra alone judges the new registration. | Claude → Astra / no resource allocation in this draft. | **BLOCKED:** [DELIVERY_DRAFT.md](DELIVERY_DRAFT.md) is a charter and truth table, not an admission token. | **NOT RECORDED** / **NOT DECLARED** / **NOT RECORDED** |
| Future fixed-D4096 development/product/strict/serving path | **BLOCKED:** fresh collision-audited bank allocations, study-specific hashes, priced resource plan, strict request/audit/review interfaces, serving-qualification plan, and owner release approval are unknown. | The future registration must order and fund each stage before any data; Claude records any admitted command in its own row. | Future registration owners / platform determined prospectively. | **BLOCKED:** no automatic two-arm 128/mix screen, strict gate, serving qualification, or release follows from the contingency. | **NOT RECORDED** / **NOT DECLARED** / **NOT RECORDED** |

## How to update a row

1. Preserve this initial state as the pre-evidence record. Do not replace
   `BLOCKED` with a summarized claim.
2. Record the exact prerequisite file paths and SHA-256 values, the responsible
   operator, platform, resource reservation, and a copyable command from that
   package's owner.
3. Record READY only after those prerequisites are verified. Add an ISO-8601
   timestamp, the declared coverage window, and the operator's explicit
   acknowledgment with timestamp.
4. If a prerequisite later becomes stale, mark the row `BLOCKED — STALE PROOF`,
   retain the previous values as historical evidence, and do not dispatch or
   retry automatically.

## Boundaries

- This table is limited to acknowledgment and next-action visibility. It is not a
  dashboard, scheduler, admission ledger, or scientific decision record.
- It does not modify `research/bl_residual/F4_REGISTRATION.md`, F4 factory work,
  Round-28 A100 work, P1 packs, serving, release pins, or other rounds.
- The source order remains F4's registered serial path: D256 fit; D1024
  collection and fit; joint Look R-A; D4096 funding/admission and fit/Look R-B;
  then the F4 H5000 and only conditionally Watch6. This table does not permit
  partial doses, pipeline reserve looks, or checkpoint substitution.

## Source boundaries

- R32 ruling: `big-lever/round-32/JUDGMENT.md` §§2, 4, 6–9.
- F4 serial path and result labels: `research/bl_residual/F4_REGISTRATION.md`
  §§R6–R9.
- Existing P1 priority and the published missed-handoff context:
  `docs/research/final_run_20261019/CAMPAIGN_LOG.md` entries 191–195.
