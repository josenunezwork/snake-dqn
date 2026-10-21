RULING: KILL_STANDS

# Round 0 INTEGRITY_FAILURE: does the sealed KILL stand? (independent Fable judge, 2026-10-21 ~01:00 PT)

Judge: Fable 5.1, independent; I wrote none of the instruments ruled on. Read-only. I opened nothing under
`chunks/*/attempt-*/job/exports/` (file metadata only, see §2.5); no Round 0 outcome is known to me.

## 1. Facts verified

| Fact | Where |
|---|---|
| Registration RATIFIED at commit 882b0fa2, study root `snake-dqn-artifacts/t34-round0-study-01`, operator "fable-approval-judge-20261020 (Claude operator)", 3 chunks of 100/100/40 pairs | `snake-dqn-round0-fixes2/research/t34_round0/REGISTRATION_ROUND0.json` |
| All three chunk attempts sealed COMPLETE; remote claim tags exist for every chunk, replay and analysis | `git ls-remote` tags `t34-round0/t34-round0-study-01/*`; repo ledger lines 1-3 |
| Mac replay PASS: 24 pairs (4:3:3 per chunk), 0 divergences, allocated 100/100/40 = completed, 0 dropped. `replay.json` carries only hashes, counts and identities, no metrics | `replay/replay.json`, `replay.log` (00:12:33-00:14:15 PT) |
| Analysis phase claimed 00:15:27.3 PT (admission `dispatched_at_unix_seconds` 1792566927.3; claim commit 59ca2fe9). The replay ledger line was still only local; it was committed in 850b08dc at 00:15:36 and pushed at 00:15:37 (5ba76312), i.e. about 10 s after the claim | `analysis/admission.json`; `git show 850b08dc`; `git reflog show origin/main` |
| `analyze_paths` → `require_chunks` → `require_published_history` raised "receipt-hash ledger must be committed and pushed before continuing" (`history.py:137-138`) before `_admission` or any `chunk.json`/pair record was opened (`analyze.py:338-347` vs `:366+`); `analyze.py:585-590` maps `HistoryError` → `decision KILL, evidence_authority INTEGRITY_FAILURE`; `run_phase` sealed the phase COMPLETE with that decision | `analysis/decision.json`, `analysis/phase_terminal.json` (status COMPLETE, reason "source-fenced phase complete") |
| `analysis.log` is 0 bytes. Nothing printed, no summary, no clause values | `t34-round0-study-01/analysis.log` |
| Boost-floor `decide` ran after the analysis ledger line was published, passed its own preconditions, and copied the verdict through: `applied: false, "registered verdict precedes clauses"`, `governing: true`, KILL / INTEGRITY_FAILURE; sealed and its ledger line is on origin/main (115955f2) | `amendment-boost-floor/decision.json`, `phase_terminal.json`; `research/t34_round0_amend/ledger/t34-round0-study-01.jsonl` |
| Today, from the frozen checkout, `history.require_published_history(registration)` PASSES (published ledger == retained terminal history for all five phases; every remote claim validates), and `history.require_chunks(...)` now refuses with "the sole outcome look has already been reserved" (`history.py:332-333`) | run read-only by me, 2026-10-21 ~00:55 PT |
| Frozen checkout `snake-dqn-round0-frozen-01` is clean at 882b0fa2 | `git status --porcelain` empty |

So the record is internally consistent: the integrity property the failed check protects (append-only published
history matching sealed terminals) is now verifiably intact, no outcome was computed or printed, and the KILL
carries zero information about the teacher. The question is purely one of governance.

## 2. Ruling and reasons

### 2.1 The sealed KILL is final for study `t34-round0-study-01`

Four binding, pre-look instruments say so, and the code enforces it:

1. `ERRATUM_INTEGRITY_SIZING.md` (part of the ratified closure): "`analysis/decision.json` is the sole scientific
   verdict ... A source-fence failure or interrupted phase is sealed as KILL; **no substituted verdict or
   re-analysis**." and "No world replacement, efficacy resume, bank extension or second look."
2. `PREREGISTRATION_ROUND0_FIXES1.md` §3/§5: "Integrity failures take precedence over outcome inference" and the
   analysis directory "is reserved before terminal history inspection; an unfinished claim yields its KILL
   decision.json". The claim-first ordering that produced this KILL is the registered design, not a bug: the README
   (research/t34_round0/README.md, "Publish the ledger using the block above before analysis") told the operator
   what the code would otherwise do.
3. `README.md` "Future Mac replay and sole fixed-N decision": "a second analysis or copied output path cannot create
   another look. Published immutable history and remote claims are checked even if source or local artifacts were
   restored."
4. `RULING_round0_go_file.md` §3 (independent judge, pre-look, 2026-10-20 ~20:10 PT; appended to SUCCESS_CONTRACT
   §5 as the "Amendment clarification"): governing KILL with `evidence_authority = INTEGRITY_FAILURE` → "**Mechanical
   branch** ... The sealed KILL is final for this study. A fresh registration needs its own approval and reserve
   units. It is not evidence about the teacher premise, so it does not open Round 0-D." That ruling expressly
   contemplated "registered history failure copied through", which is this case, and pre-committed its remedy.

I will not override a pre-look routing ruling after the fact. "Integrity KILL is final" is only worth anything if
it holds in the one case where everyone agrees the data are fine; otherwise every future integrity KILL invites the
same plea, and the plea will come after the look at least once.

### 2.2 Why a "remedial pre-look amendment" is not legitimate here

The project has one accepted model for a post-collection, pre-look analysis amendment: the boost-floor amendment.
Its legitimacy rested on a mechanical precedence test the project itself wrote (`decide.py:verify_precedence`): the
amendment must be on origin/main **before the analysis admission timestamp**. That is the line between "changed the
rule before the look was reserved" and "changed the rule after". A remedial amendment would necessarily be committed
after 00:15:27 PT on 10-21 and so fails the project's own test. Calling it "pre-look" because no number was printed
substitutes an unverifiable claim (nobody read the exports) for a verifiable one (commit time < claim time).

Two further mechanical facts make "re-run the registered analysis code unchanged" impossible as stated:

- `require_active_phase(registration, "analysis")` (`history.py:39-43`) and the guard at `history.py:332-333` refuse
  any consumer whose active phase is not `<root>/analysis`, and that directory and its remote tag are consumed.
  A remedial consumer would therefore have to either lie about its phase or re-implement the ~150-line evidence
  admission loop of `analyze_paths` (`analyze.py:366-520`) without the history guard. Either is new,
  outcome-touching code written after the data exist, by the same operator who needs the result. That is exactly
  the surface preregistration exists to keep closed.
- The governing verdict is the amendment file, not the registered one (go-file ruling §2.1). A remedial path would
  also need a second boost-floor pass, i.e. a third decision file and a second amendment tag, where spec §3.3 and
  the go-file ruling say "no second look".

Outcome-blindness is credible (empty log, exception before any record read, replay reports hashes only) but it is
not provable: the five export files whose access times post-date creation by more than 5 s are trace/pair JSON files
with reads within a minute of creation (supervisor-side), and APFS access times did not record the replay's own
reads at 00:12-00:14, so filesystem metadata cannot establish anything either way. The project should not build a
verdict on an attestation when it has a cheaper, fully registered alternative (§3).

### 2.3 DESCRIPTIVE_ONLY is the worst option and is forbidden

A descriptive look buys no verdict (the go-file ruling binds every Round 0-conditioned lane to the governing file,
and `README.md`: "Numerical summaries grant no evidence authority") while burning the one asset the sealed data still
have: nobody knows the answer. Once the operator knows the number, a fresh-namespace re-registration of the same
teacher is no longer a pre-data decision (whether to fund it, and how to read it, become outcome-informed). The 240
pairs therefore stay sealed. Exception, only after the search-teacher family is formally closed in CAMPAIGN_LOG with
no further Round 0 registration planned: a descriptive opening may be logged, labelled NOT_EVIDENCE, and never cited
in any registration.

## 3. The registered remedy, and whether it is worth running now

Remedy (already pre-committed): **mechanical re-registration on a fresh namespace with the same rule**, own approval,
own reserve units. Nothing else is authorized by this KILL: no Round 0-D (stage-2 `ratify` needs a *scientific*
NO-GO), no R1.

Cost estimate from the ratified sizing: 3 chunks, unit cap 11.07 + 2.21 reserve each (envelope ~40 u); chunk 000
actually ran 7,859 s at ~1.5-2.0 u/h, so expected spend is ~15-20 u for all three, ~3 h wall on three lanes (or
~8 h on one), plus Mac replay (~2 min) and analysis (minutes). Operator work: new namespace + bank + collision seed
audit + registration JSON + 48 h dispatch window, all with tested machinery and no new consumer code.

Is it worth it? It is cheaper, faster and cleaner than the remedial path (which needs Sol code, a persona review,
an approval and still produces a contestable verdict), and it is the only route that can yield a citable Round 0
number. Whether that number is still wanted is a portfolio question I answer conditionally:

- **Fund it** if, within 48 h, at least one live lane states in writing that it will consume the governing Round 0
  verdict (R1 registration under §5, Round 0-D stage 2, or a big-lever registration that cites the offline
  teacher-vs-B8 ceiling by SHA). Then it is a decision-producing ~20 u lane and matches the owner's spend goal.
- **Otherwise close the lane**: log "Round 0 premise question unanswered; data sealed", keep the 240 pairs sealed
  (§2.3), and let the big-lever program plan without an offline teacher ceiling. With Path B at STOP and Claim A D0
  at KILL, nobody should spend a day and 20 u to gate lanes that are themselves being retired.

## 4. Mechanical conditions for the fresh-namespace re-registration (if funded)

1. New namespace root (e.g. `t34-round0-premise-20261021/v1`), new `BANK_ROUND0`-style bank and seed audit disjoint
   from the Round 0, retention, strict and `t34-round0-premise-20261020` banks; any audit hit rejects the whole
   allocation (prereg "Fixed controllers and world bank"). New study root; new ledger file; new tag prefix
   `t34-round0/<new-study-id>/`. `_reject_other_study_roots` only checks the one ratified root, so the old tags do
   not block this.
2. Same rule: thresholds, arms, H5000, 240 pairs 96/72/72, 4:3:3 strata, replay quota, boost-floor amendment
   (re-declared for the new root, committed before the new analysis claim), Colab L4 K=12, Mac replay. No clause
   edits. Comparator stays bare B8 s3 `3e649783...`.
3. Source commit: 882b0fa2 may be reused verbatim (frozen checkout exists and is clean) **or** a new commit whose
   only functional change is the process fix in item 5. A new commit requires the registered A6 preflight and the
   review loop again (prereg: "saved historical A6 costs may price a draft but cannot certify changed source").
4. Registration binds, per the go-file ruling §2.4, the SHA-256 of the old governing `amendment-boost-floor/decision.json`
   (94ad1344...), its terminal (`phase_terminal.json`, ledger line 115955f2) and this JUDGMENT.md, and states that the
   prior study is KILL/INTEGRITY_FAILURE with no outcome read.
5. Process fix, required whichever commit is used: the analysis runbook step must be executed through a wrapper
   (shell is enough if 882b0fa2 is reused) that refuses to invoke `analyze` while the authoring repo has an unpushed
   or uncommitted ledger (`_local_ledger != _published_ledger`) or a dirty tree. This check reads no outcome and
   claims nothing, so it is safe pre-claim. Prefer landing it in `run_phase` before `reserve` for the analysis phase
   (a refusal, not a KILL) if a new commit is taken anyway.
6. Approval: the package (bank, audit, registration draft, forecast, wrapper) goes through the usual Sol
   implementation → 2-3 persona review → one independent judge who is not me and did not author any Round 0
   instrument. Owner is informed; owner veto stands; release authority is untouched. Dispatch within 48 h of the
   new ratification, else that registration lapses (FIXES1 §5).
7. Until the new governing verdict is published: no R1 registration, no Round 0-D stage 2, no citation of any Round 0
   number anywhere, and the old exports remain unopened. The old study's five phases and both ledgers are never
   edited, deleted or relabelled.

## 5. Record

- CAMPAIGN_LOG row "2026-10-21 ~00:20 PT — Round 0 governing verdict = KILL / INTEGRITY_FAILURE (operator error,
  Claude)" is accurate and should stand; append a row pointing at this ruling and the funding decision in §3.
- This file is advice; it edits no hash-bound document. Not committed by me.
