RULING: For SUCCESS_CONTRACT §5 and every Round 0-conditioned lane, "Round 0 GO" (and every other Round 0 verdict) means the decision in the published, sealed `$E/amendment-boost-floor/decision.json` (`governing: true`), never the registered `$E/analysis/decision.json`. The only exception is a recorded `decide` refusal for late publication, and even then the decision must be ratified first.

# RULING: which Round 0 file governs R1 and Round 0-D (independent judge, Opus stand-in, 2026-10-20 ~20:10 PT)

Judge: Opus, standing in for Fable because Fable is at its usage limit (owner's choice). I wrote none of the documents ruled on. I made this ruling prospectively, before any look: nothing under
`snake-dqn-artifacts/t34-round0-study-01/` was opened, and no Round 0 outcome is known to me.

## 1. The conflict

- The §5 Amendment (`SUCCESS_CONTRACT.md:75-88`; ruling `../2026-10-20_a100/RULING_label_ordering.md`) lets R1
  label collection follow "a published Round 0 GO". Its conditions define that phrase as "the published registered
  `decision.json` = GO only".
- The owner-approved boost-floor amendment (merge 30b2739a; spec
  `../../specs/2026-10-20_t34_round0_boost_floor_amendment.md` §3 item 3, §3.3; approval
  `snake-dqn-artifacts/round0-boost-review/APPROVAL.md`) came later. It makes
  `<study_root>/amendment-boost-floor/decision.json` "the sole governing Round 0 verdict". The registered file is
  kept as evidence only, and its `decision` field "is not reported as the Round 0 outcome".
- JUDGMENT §7b (`JUDGMENT.md:153-156`) flagged the clash.

## 2. Ruling

1. **The governing file decides.** "Round 0 GO" in the §5 Amendment (and in the label-ordering ruling §3) means
   **`$E/amendment-boost-floor/decision.json` with `governing: true`, `decision: "GO"`, and
   `evidence_authority: "TIER_1_DEVELOPMENT"`**. It must also be *published*: its `phase_terminal.json` is sealed,
   and its line in `research/t34_round0_amend/ledger/<root>.jsonl` is on `origin/main`. The §5 Amendment's word
   "registered" was written before 30b2739a. Its purpose was to rule out any unregistered or discretionary
   verdict. The amended verdict meets that purpose: it is a pre-registered, deterministic, pre-look function of the
   same sealed records, and the owner approved it. So this is a later, owner-approved instrument taking precedence,
   not a reinterpretation. A registered GO is therefore not enough, and neither is a registered STOP that the
   amendment did not flip.
2. **One fallback only.** The registered `analysis/decision.json` governs only if `decide` refuses for exactly one
   reason: the amendment was not on `origin/main` before the analysis admission (spec §3 item 3). In that case the
   refusal output must be preserved and logged in CAMPAIGN_LOG before any Round 0-conditioned lane is registered.
   Every other refusal must be fixed and `decide` re-run (spec §3.3(b)). A post-claim failure, or a remote amend tag
   without a sealed `phase_terminal.json`, is a governing **KILL** (spec §3.3(a), (c)). None of these cases falls
   back to the registered file.
3. **Nothing is authorized before the governing verdict is published.** A sealed registered verdict alone opens no
   lane: not R1, not Round 0-D, and no mechanical re-registration.
4. **Every Round 0-conditioned lane binds the governing verdict.** This covers L2 / sparse R1, L7 / Round 0-D and
   dense R1, a mechanical Round 0 re-registration, and any later lane keyed on "R0". Each such registration must
   record the governing `decision.json` SHA-256, its `phase_terminal.json` SHA-256, and the amendment-ledger
   commit. Every other §5 Amendment condition is unchanged: R1 needs its own registration, review loop and approval;
   it uses fresh disjoint seeds; it admits no Round 0 output; R1/R2 fits wait for the Path B disposition; and the
   owner's veto stands.

## 3. How the governing verdict routes (pass-through included)

When the registered verdict comes before the boost clause, the amendment copies it **unchanged**
(`applied: false`; spec §3 item 2, `decide.py:248`). A copied verdict therefore routes exactly as if the amended
rule had produced it. The rows below use only fields of the governing file, so the routing needs no judgment.

| Governing `decision` | Distinguishing fields | Route (NO-GO plan §3; JUDGMENT §3) |
|---|---|---|
| GO | `evidence_authority` = TIER_1_DEVELOPMENT (applied true or false) | **GO plan.** R1 / L2 may be registered under the §5 Amendment. Round 0-D is shelved. |
| STOP, scientific | `unconditional_faults` empty, and `reason` does not start with "unconditional" | **Round 0-D** (L7), after it is ratified on main. No sparse R1. |
| INCONCLUSIVE, scientific | `evidence_authority` = TIER_1_DEVELOPMENT (point bars pass, an interval fails) | **Round 0-D**, same as scientific STOP. Round 0 is never extended. |
| STOP, fault | `unconditional_faults` non-empty, or `reason` starts with "unconditional" (includes the rolling-100k action-collapse STOP) | **Bug branch.** Fix teacher_v2, then re-register Round 0 on a fresh namespace. No Round 0-D and no R1. |
| INCONCLUSIVE, integrity | `evidence_authority` = INVALID_OR_INCOMPLETE (cap or deadline expired, invalid evidence) | **Mechanical branch.** New Round 0 registration on a fresh namespace with the same rule. No dense substitution, no Round 0-D, no R1. |
| KILL | `evidence_authority` = INTEGRITY_FAILURE (registered history failure copied through, or a post-claim amendment failure) | **Mechanical branch**, as for integrity INCONCLUSIVE. The sealed KILL is final for this study. A fresh registration needs its own approval and reserve units. It is not evidence about the teacher premise, so it does not open Round 0-D. |

No row authorizes Round 0-D except a *scientific* STOP or INCONCLUSIVE in the governing file. No row authorizes R1
except a governing GO.

## 4. Text to append, and why appending is safe

- **SUCCESS_CONTRACT.md:** append a dated "§5 Amendment clarification" after the existing §5 Amendment. It is
  append-only, and the ratified §5 text and the §5 Amendment text are left unchanged. A CAMPAIGN_LOG row goes before
  "## Compute portfolio".
- **Hash binding, confirmed.** `SUCCESS_CONTRACT.md` is SHA-bound by registrations that hash their **own frozen
  checkout**, not main's working file:
  - Round 0 binds `9f34ef2f…` and runs from `snake-dqn-round0-frozen-01` at 882b0fa2
    (`research/t34_round0/common.py:11-16`, `ROOT` = the checkout).
  - Path B retention v2 binds `cb3300bc…` through `source_binding_v2.source_inventory(repo)` under its frozen fence.
    It runs from `snake-dqn-retention-v2-frozen-02` at 81c54731. That checkout is clean and its file hashes to
    `cb3300bc…`, as checked now.
  - Its ledger repo `snake-dqn-retention-v2-author` is on branch `t34-retention-bare-unpaced-20261020`, not main,
    and must not be fast-forwarded until its analysis is complete. That rule already exists and this ruling keeps it.
  - The boost-floor `decide` hashes only its amendment JSON, the ratification and the six registered sources
    (`decide.py:83-90`), not the contract.

  So appending to main's file changes no bound hash and does not touch any running or ratified study, as with
  d5b11e3a.
- **One consequence.** Round 0-D (still DRAFT) lists `SUCCESS_CONTRACT.md` in its source binding
  (`research/t34_round0d/common.py:36`). Its registration must be generated from a checkout that includes this
  append. A draft built earlier must be regenerated, not hand-patched.
