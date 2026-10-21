# RULING — R1 label collection vs the Path B retention verdict (independent Fable judge, 2026-10-20 ~18:30 PT)

Judge: independent Fable (wrote none of the advice, reviews, contract or amendments). Read-only; the only
file written is this one. Nothing committed. Round 0 outcomes were **not** opened: I read the supervisor log tail
(`snake-dqn-artifacts/t34-round0-study-01/chunk-000-attempt-01.log`, elapsed 2,562–4,111 s), which carries only
export paths, SHA-256s and sizes — operational telemetry as the registration allows
(`research/t34_round0/PREREGISTRATION_ROUND0.md:148-149`). No pair file was opened.

## VERDICT: APPROVE WITH CONDITIONS

1. **Governing text today: SUCCESS_CONTRACT §5.** Until amended, no R1/R2 label round may be funded or dispatched
   before the Path B retention verdict. DESIGN_V2:103's "or Round 0" does not create permission.
2. **A prospective amendment permitting R1 *label collection* (Colab, non-Mac) after a published Round 0 GO and
   before the Path B verdict is legitimate** under the project's rules, provided the conditions in §3 are met. It is
   an amendment of owner-ratified text, made under the owner's 2026-10-20 approvals delegation; it is not an
   interpretation, and it must be recorded and reported as an amendment.
3. **Scope is label collection only.** R1 fits, R1 screens, R2 labels and anything on the Mac stay gated on the Path
   B disposition (or a later prospective ruling). Round 0 GO still does not dispatch anything by itself.
4. **Release authority: the owner's OK governs releases.** STATUS.md:22's sentence routing "release/serving changes"
   to a Fable judge is an over-broad transcription and should be corrected by a dated operator edit; a Fable judge
   may approve everything up to serving-qualification readiness and recommend release.

---

## 1. Which text governs

**Ranking.** STATUS.md:10-14 (owner-ratified authority block) lists the binding documents "in order":
SUCCESS_CONTRACT → DESIGN_V2_SHADOW (+ amendments) → P7 → unsuperseded FINAL_PLAN. The Round 0 registration
repeats the same order verbatim (`PREREGISTRATION_ROUND0.md:12-15`). DESIGN_V2 itself declares it "Binds with
SUCCESS_CONTRACT.md (**unchanged**: claims, bars, serving constraints)" (`DESIGN_V2_SHADOW.md:8`), and Amendment 2
is an "operator decision under owner delegation" (`:74, :80`) — a lower tier than owner ratification
(`SUCCESS_CONTRACT.md:3`).

**The contract is unambiguous.** `SUCCESS_CONTRACT.md:61-68`:
- "Path B deadline-feasibility check runs before R1/R2 label rounds" (:61);
- pass branch: "label rounds become optional research (owner decides whether to fund R1/R2)" (:65-66);
- fail branch: "the label rounds proceed only on a Round 0 GO" (:67-68).

Both branches condition label funding on the Path B *verdict*; Round 0 GO is a necessary condition in the fail
branch, not a sufficient one in either. The reading "labels may start on Round 0 GO alone" is not available from
this text.

**DESIGN_V2 is internally split and the split resolves to the contract.** `DESIGN_V2_SHADOW.md:93` ("Deciding
sequence (**no label rounds funded before it**)") restates the contract. `:103-104` ("Teacher label rounds (R1+)
wait for (b) or Round 0") is a one-line sequencing note written at 01:00 PT after M2/M3 NO-GO; read with :93 and
:8 it can only mean "labels wait on the retention study's branch outcome, and in the KILL branch also on Round 0" —
the same structure as the contract. If it meant more, it would contradict the higher-ranked contract and lose.
The ratified retention registration reads the same way: KILL → "teacher labels then wait on Round 0"
(`research/t34_pathb/PREREGISTRATION_RETENTION_V2.md:350-353`), and labels "remain separate operator decisions"
(`:538-540`; also `README_V2.md:498-499`, `PROTOCOL_V2.md:87-88`). FINAL_PLAN:7/:59 ("R1+R2 funded after Round 0
GO") is the superseded pre-contract rule — the contract's §5 heading is literally "Sequencing change".

**Conclusion:** SUCCESS_CONTRACT §5 governs. REVIEW_B §3.1 (`REVIEW_B_decision.md:61-67`), REVIEW_A #11
(`REVIEW_A_evidence.md:23`) and Astra P4 call 2 §2 (`astra_p4_critical_path.md:193-206`) all reach this reading;
I concur.

## 2. Is a prospective amendment legitimate?

Tested against each rule the question names.

**(a) Freeze before data** (`CAMPAIGN_LOG.md:15-16`; `P7_STATISTICS.md:6-7`; `PREREGISTRATION_ROUND0.md:7-8`
"use a prospective sibling amendment before affected data"). The data this amendment could influence is the
*post-Round-0 funding decision*, i.e. it must be frozen before anyone sees a Round 0 outcome. Round 0's outcomes are
sealed by registration (`PREREGISTRATION_ROUND0.md:90-91` "No interim outcome peeks"; `:148-149`; `README.md:135,
172-173`; `HANDOFF_2026-10-20.md:147-149`), chunk 000 is mid-run, and retention has not started
(`CAMPAIGN_LOG.md:56, 62`). An amendment landed now is pre-data for both studies. **Satisfied, conditional on §3(B).**

**(b) Never relabel a registered gate.** The amendment changes no bar, N, look, decision rule, bank or cap of Round 0
or retention; it changes what action may follow a published *GO*. Round 0's own text already anticipates this:
"GO permits planning label rounds, never fitting, dispatching another study, a strict pass, serving qualification,
or release **by itself**" (`PREREGISTRATION_ROUND0.md:21-22`). The amendment must keep that sentence true: R1 is
dispatched by its *own* approved registration, not by Round 0 GO. **Satisfied, conditional on §3(C),(E).**

**(c) No outcome peeking.** The amendment's trigger is the registered `analysis/decision.json` value GO
(`README.md:227`), produced only after all chunks are terminal and the 10% replay. It must not be conditioned on any
other Round 0 quantity. **Satisfied, conditional on §3(B).**

**(d) Authority to amend owner-ratified text.** The owner's standing rule of 2026-10-20 (STATUS.md:22; AGENTS.md:21;
HANDOFF:8-9; CAMPAIGN_LOG:23-24) sends "funding gates" and other approvals to one independent Fable judge, recorded
in CAMPAIGN_LOG and reported to the owner; the owner also delegated next-step calls (CAMPAIGN_LOG:23-24). Amendment 2
already amended the owner-approved DESIGN_V2 under that delegation (`DESIGN_V2_SHADOW.md:74`). A change to *when a
Colab label round may be funded* is a funding-gate decision squarely inside the delegation; and the owner's separate
standing instruction to spend Colab units on decision-producing lanes without waiting for the Mac (STATUS.md:22)
removes the contract §5 rationale (advice: "before expensive label rounds",
`docs/research/advice/2026-10-19_t34_success_skeptic.md:25`) for a Colab-only, Mac-free label collection. The
amendment is **legitimate under delegation**, but it is an amendment, not an interpretation, so it must be written as
one and reported to the owner, who may veto it (§3(G)).

**(e) Effect on calendar is real but bounded.** The practical saving is ≤ 1–2 days and only if the production
pipeline (REVIEW_A #10; REVIEW_B §0) is ready first; for R1 labels (β=1, roots = B8+teacher, no student needed:
`FINAL_PLAN.md:22`) the blocking student fan-out in `label.py:108-111` is not required, so R1 collection is the one
Claim A step that can honestly run early.

## 3. Conditions (all required)

**(A) Form.** A dated, prospective "§5 Amendment (2026-10-20)" *appended* to SUCCESS_CONTRACT.md (no in-place edit of
the ratified text), plus an Amendment 2 addendum to DESIGN_V2 replacing the ":103 'or Round 0'" sentence with the
unambiguous rule below, plus a CAMPAIGN_LOG decision row citing this ruling, all in one commit pushed to main.
Suggested rule text: *"R1 teacher-label collection (Colab only, bare B8 s3 + teacher, β=1) may be registered and,
once its registration is approved, dispatched after a published Round 0 GO, in parallel with the Path B retention
phases. R1/R2 fits, R1/R2 screens, R2 labels and any Mac work for Claim A wait for the Path B disposition
(GO/KILL/INCONCLUSIVE decision.json). Round 0 GO authorizes nothing by itself."*

**(B) Timing / no-peek attestation.** The commit must land **before** Round 0's sole analysis is invoked
(`$E/analysis/decision.json` must not exist) and before any Round 0 outcome value has been inspected by anyone; the
CAMPAIGN_LOG row carries the operator's attestation and the chunk/attempt state at commit time. For hygiene, also
before retention `timing/attempt-01` (free: retention has not started). If decision.json already exists when the
amendment is proposed, this ruling is void and the contract §5 ordering stands.

**(C) Round 0 GO means the registered GO only.** Dispatch may follow only a published `decision.json` = GO
(`PREREGISTRATION_ROUND0.md:113-125`); STOP or INCONCLUSIVE → no R1 (contract §5 fail branch is unchanged). No
inference from unfinished chunks (`:104`).

**(D) R1 needs its own registration and approval.** Standard loop (`CAMPAIGN_LOG.md:17-19`): independent review →
FIX-FIRST → Sol → re-review → RATIFY → one independent Fable approval judge (funding gate, STATUS.md:22). The
registration must: use fresh, collision-audited namespaces with seeds disjoint from the Round 0 bank, the retention
bank and strict banks (`P7_STATISTICS.md:22-24`; `PREREGISTRATION_ROUND0.md:60-67`); admit **no Round 0 output** to
labels or fitting (`PREREGISTRATION_ROUND0.md:148`); bind source like Round 0 (frozen worktree, A6, receipts);
state caps inside portfolio lane L2 (`CAMPAIGN_LOG.md:70`; ~29 u/800 worlds); and declare that its outputs are
label data only — no fit, screen, selection or claim may be computed from them until the Path B disposition is
published.

**(E) No Mac contention.** While any retention phase (timing, pre-check, efficacy) is running or admitted, R1 does no
Mac work beyond reading logs: preflight/A6 for R1 either completes before retention admission or runs after the
retention phase in a gap, never concurrently (retention's quiet-host and 2 %-contention rules,
`CAMPAIGN_LOG.md:50-51`). Mac replays for R1 (FINAL_PLAN §7 A2) wait for the Path B disposition.

**(F) Independence of the two studies.** R1 label results may not be cited in, or influence, the retention verdict
or its interpretation; the retention KILL clause ("labels then wait on Round 0") remains satisfied. If retention is
GO, R1 continues as the "optional research" the contract already names (`SUCCESS_CONTRACT.md:65-66`), with the
funding call now made by a Fable judge and reported to the owner (STATUS.md:22), not silently assumed.

**(G) Owner report and veto.** The amendment is reported to the owner in the next status/handoff with this ruling
linked. An owner objection before R1 dispatch cancels the amendment; after dispatch it stops R1 at the owner's call,
with collected data retained unused.

**(H) Claims hygiene.** No wording anywhere may describe R1 labels as "funded by Round 0" or as evidence; Round 0
remains Tier 1 development evidence (`PREREGISTRATION_ROUND0.md:19-22`).

## 4. Release authority (REVIEW_B §3.2)

Texts in tension: `SUCCESS_CONTRACT.md:57` "owner OK required for release" (owner-ratified, ranked first);
`CAMPAIGN_LOG.md:23-24` "releases/serving changes still need the owner" (roles line, 10-20); memory record of the
10-20 00:15 delegation ("releases still need OK"); versus `docs/STATUS.md:22` / `AGENTS.md:21` (committed 13:32,
`4248f00c`) "any other decision needing approval (… release/serving changes) goes to one independent Fable judge".

Ruling: **the owner retains release approval.** The contract's sentence is specific, owner-ratified and top-ranked;
the STATUS.md sentence is the operator's generalised transcription of an approvals rule whose own recorded carve-out
("safety-prohibited actions still the owner's", HANDOFF:9) shows it was not an exhaustive transfer, and no document
records the owner saying "a Fable judge may release". A general delegation recorded by the operator does not
override specific owner-ratified text. Operationally: a Fable judge may approve strict-gate registrations, serving
qualification and a *release recommendation*; the serving-default switch itself needs the owner's OK. STATUS.md:22
and AGENTS.md:21 should be corrected by a dated operator edit (not a relabel) to say so; if the owner later states
explicitly that release approval is delegated, record that statement and update the contract by dated note.

## 5. Not ruled on (noted for the operator)

- Amendment 2 §3(a) "~1.5 h Mac" pre-check estimate vs the ratified runner's ~6–7 h (REVIEW_B §3.3): an estimate,
  not a registered value; a dated CAMPAIGN_LOG note suffices. No action under this ruling.
- A later prospective ruling may extend §3's scope to Colab-only R1 fits once the CUDA fit qualification exists; not
  granted here because the contract's Path B branch logic is still the governing structure and the calendar gain
  before the retention verdict is near zero.
