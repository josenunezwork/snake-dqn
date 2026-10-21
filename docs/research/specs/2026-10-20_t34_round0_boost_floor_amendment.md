# T34 Round 0: boost-floor analysis amendment (pre-look)

**Status: DECLARED PRE-LOOK, 2026-10-20 (PT). Under review; not yet merged to main.**
It governs only after it is merged to `origin/main` *before* the Round 0 analysis
phase is reserved. The `decide` entrypoint checks this and refuses otherwise.
Owner decision: 2026-10-20 ~19:55 PT. Package: [`research/t34_round0_amend/`](../../../research/t34_round0_amend/).
Machine-readable declaration: [`AMENDMENT_BOOST_FLOOR.json`](../../../research/t34_round0_amend/AMENDMENT_BOOST_FLOOR.json).

## 1. Why the floor has to change

The registered Round 0 clause is
`"delivered_boost_share": 0.005 <= boost <= 0.10` (`research/t34_round0/analyze.py:170`).
It applies to the teacher arm's pooled share (`analyze.py:156-158`). It is a *point* clause, so a
failure is STOP no matter what mass and survival show (`analyze.py:178-202`;
PREREGISTRATION_ROUND0.md:79, 120, 125-127).

An independent check (`snake-dqn-artifacts/boost-bar-check-20261020/REPORT.md`) opened no
Round 0 study files. It found:

- **Bare B8 s3 boosts on 0.1206% of decisions.** This uses the exact definition, over
  39 *excluded* A5 v2 worlds (190,779 decisions): scripted 0.130%, frozen 0.108%, mixed 0.106%.
  Every other source is also below 0.5%: T13 0.15-0.24%, T3 CTRL 0.19-0.39%, and 0 boosts in
  the Round 0/0-D preflights.
- The teacher can choose boosts, but in A5 it added only about 0.007 percentage points. The
  teacher arm is therefore expected to sit about 4x below the 0.5% floor, so Round 0 would
  **STOP regardless of mass or survival**.
- The 0.5% floor first appeared in FINAL_PLAN/MERGED_PLAN on 2026-10-19 with no recorded
  rationale. It was then copied into SUCCESS_CONTRACT:32 and P7:47. B8 s3's own release
  (T7 strict, T12 serving) never applied a boost clause. The 10%
  ceiling does have a rationale: T3 students spilled into 23-27% boost.

The floor's evident purpose is to catch a controller that suppresses boosting. An absolute 0.5%
cannot do that for a base policy that boosts 0.12% of the time. It would fail the comparator
itself. A floor relative to the paired bare arm keeps the purpose. The ceiling is unchanged.

## 2. Exact amended rule

On the 240 paired worlds, with the same definition for both arms (boost = delivered hero action
`>= 3`, `runner.py:287`, one decision per trace row, pooled as Σboosts / Σdecisions over all
mixes rather than averaging per-world ratios):

```
0.5 × bare_share  <=  teacher_share  <=  0.10        (both inclusive; bare_share = 0 → floor 0)
```

- **Floor:** compared exactly in integers:
  `2 · teacher_boosts · bare_decisions >= bare_boosts · teacher_decisions`.
  Display rounding never matters.
- **Ceiling:** the registered float expression `teacher_boosts / teacher_decisions <= 0.10`,
  unchanged.
- The clause keeps its name (`delivered_boost_share`) and stays a **point** clause. Its failure
  is STOP, exactly as before.
- **Nothing else changes:** every other clause and threshold, every one-sided tail, fault
  precedence, the STOP → GO → INCONCLUSIVE order, the clear-survival-failure rule, all
  KILL/INCONCLUSIVE evidence outcomes, the replay, the chunk/dispatch machinery, the caps and the
  ledger.

## 3. Mechanism

Round 0 cannot be edited. The frozen worktree at `882b0fa2` must stay byte-clean: any tracked or
untracked change is source drift, which is KILL. The ratification also binds the source
inventory, and the preregistration says changes go in "a prospective sibling amendment before
affected data". So the amendment is a **separate package that runs after the registered
analysis and changes only one clause**:

1. **Registered analysis, unchanged.** The frozen checkout runs `research.t34_round0.analyze`
   exactly as the README says. It writes and seals `<study_root>/analysis/decision.json` and
   appends its ledger entry, which is then published. Dispatch, replay and analysis code are
   untouched.
2. **Amendment `decide`, run once.** The command is
   `python -m research.t34_round0_amend.decide --ratification R --out <study_root>/amendment-boost-floor/decision.json`.
   It runs from a clean checkout whose HEAD is on `origin/main`. It does the following:
   - **Refuses with no side effects** in any of these cases:
     - the amendment JSON does not bind this exact ratification (file SHA and canonical SHA,
       commit `882b0fa2`, study root);
     - any imported registered source (`analyze.py`, `common.py`, `history.py`, `__init__.py`,
       `retention_history_v2.py`, `eval_stats.py`) differs from the ratified commit's blob;
     - the registered analysis is not sealed, its inventory changed, or its ledger entry is not
       on `origin/main`;
     - the remote `t34-round0/<root>/analysis` claim is missing;
     - the amendment JSON was not published on `origin/main` (first-parent, exactly once, never
       edited) in a commit whose committer time precedes the analysis admission's
       `dispatched_at_unix_seconds`;
     - the amendment's own remote claim, phase directory or ledger entry already exists.

     No outcome file is parsed before the claim. Sealed files are only hashed.
   - **Claims create-only.** It pushes a unique remote tag
     `refs/tags/t34-round0-amend/<root>/boost-floor`. A non-force push refuses an existing tag.
     It then creates `<study_root>/amendment-boost-floor/` (`exist_ok=False`) with an
     `admission.json`. The tag sits outside `refs/tags/t34-round0/<root>/` on purpose: Round 0's
     claim-inventory check (`history._claims` vs local phases) and its other-root guard never
     see it.
   - **Evaluates.** It reads the registered `decision.json` and re-checks that the sealed
     inventory is unchanged since the claim.
     - If the registered verdict precedes the clauses (KILL, INCONCLUSIVE evidence, any
       unconditional or rolling-window fault STOP, any authority other than
       `TIER_1_DEVELOPMENT`), it is copied **unchanged** (`applied: false`).
     - Otherwise it re-reads the same pair records. It follows `decision.json.chunk_sha256` →
       `chunk.json` (SHA-checked) → each `pairs[]` entry through the same `_bound_path`
       SHA-256 binding. It re-runs the **registered** `summarize` and requires its output to
       equal the registered decision field by field (everything except
       `evidence_authority`). It then computes both arms' pooled boost counts and re-evaluates
       only `delivered_boost_share` (`rule.amend`).
   - **Writes and seals.** It writes `decision.json` (schema `t34-round0-decision-amended/v1`,
     `governing: true`). That file keeps every registered field, and adds
     `registered_decision`, `registered_reason`, `registered_clauses` and a
     `boost_floor_amendment` block with both arms' counts, shares, floor, ceiling, and the
     registered and amended clause values. It then seals `phase_terminal.json` (a full
     inventory hash) and, under a lock, appends `{"phase", "terminal_sha256"}` to
     `research/t34_round0_amend/ledger/<root>.jsonl` in the registered history-ledger repo
     (the authoring checkout).
   - **After the claim, never refuses.** Any reproduction mismatch, SHA mismatch, malformed
     record or interruption writes a sealed **KILL** (`INTEGRITY_FAILURE`). This mirrors
     Round 0's `run_phase`.
3. **Governing verdict.** Once this amendment is published before the analysis claim,
   **`<study_root>/amendment-boost-floor/decision.json` is the sole governing Round 0 verdict.**
   The registered `analysis/decision.json` is retained as evidence. Its `decision` field is not
   reported as the Round 0 outcome. If `decide` refuses because the amendment was not published
   in time, the registered decision governs, unamended.

### Does the registered analysis record the bare arm's boosts?

**Not in `decision.json`.** `summarize` returns only the teacher's `boost_share`
(`analyze.py:156-158`). The bare counts are in every exported `pair.json` as
`arms.bare.{boost_decisions, decisions, action_counts}`. The registered analysis has already
validated them:

- `_validate_arm` reconciles `boost_decisions == sum(action_counts[3:])` and
  `sum(action_counts) == decisions`;
- `analyze_paths` requires `action_counts` to equal the hero column of the exported trace and
  `decisions` to equal the trace length.

So the bare share uses the identical definition, comes from the identical records, and is
reached through the identical hash chain. No new file type is opened; traces are not re-read.

### 3.3 Failure and refusal semantics (R-A; Opus approval judge condition 1, 2026-10-20)

- (a) If a remote tag `refs/tags/t34-round0-amend/<root>/boost-floor` exists without a sealed
  `amendment-boost-floor/phase_terminal.json`, that is a governing KILL (INTEGRITY_FAILURE).
- (b) Any `decide` refusal other than "amendment was not published before the analysis was reserved" must be
  fixed and `decide` re-run. Such a refusal never returns governance to the registered verdict.
- (c) A post-claim failure is a sealed KILL. It is never a fallback to the registered verdict.

## 4. Integrity argument

- **Declared before any look.** This document and the amendment JSON are written while chunks
  are still running and no outcome exists. The code enforces the precedence: the amendment must
  be on `origin/main` before the analysis admission timestamp.
- **No discretion.** The amended verdict is a deterministic function of the same sealed
  records, fixed in code before the look. Knowing the registered decision cannot change it, and
  no choice remains to make after seeing it.
- **One look.** There is a single create-only remote claim, a create-only directory, and a
  ledger check. Deleting local evidence does not reopen the look: the remote tag still refuses
  it (tested). Round 0's own one-look machinery is untouched. Its `analysis` phase is still the
  only Round 0 phase that reads outcomes; the amendment only re-reads records that phase
  already consumed.
- **Minimal change.** Tests show every other clause, every summary field, and the
  decision/reason are identical to the registered `summarize`. The oracle is the registered
  code itself: boost counts enter no other clause, so editing only teacher boost counts until
  the *registered* clause equals the amended value must reproduce the amended decision exactly
  (120 randomized studies, covering GO, STOP and INCONCLUSIVE and all four clause flips).
- **Frozen source preserved.** Nothing in `research/t34_round0/`, the frozen worktree or the
  authoring worktree changes. The amendment imports the registered `summarize` and refuses if
  its bytes differ from `882b0fa2`.
- **Fail-closed.** Refusals happen only before the claim. After the claim, every failure is a
  sealed KILL, so an operator cannot quietly choose between the two verdicts.

## 5. Operator runbook (after review, judge approval and merge to main)

The merge must land **before** the Round 0 analysis is reserved. Chunks keep running from the
frozen worktree, and replay is unchanged.

```bash
# 0. Once, before analysis: merge this branch to main and record the merge in CAMPAIGN_LOG.md.
# 1. Registered replay + sole analysis from $FROZEN exactly as research/t34_round0/README.md;
#    publish the analysis ledger append from $AUTHOR_ROOT (README "publish" block).
#    Do not report analysis/decision.json.
# 2. Amendment, immediately after step 1, from a clean detached checkout of origin/main:
git -C /Users/josenunez/Projects/ml/snake-dqn fetch origin main
git -C /Users/josenunez/Projects/ml/snake-dqn worktree add --detach \
  /Users/josenunez/Projects/ml/snake-dqn-round0-amend-01 origin/main
cd /Users/josenunez/Projects/ml/snake-dqn-round0-amend-01
OMP_NUM_THREADS=1 "$PY" -m research.t34_round0_amend.decide \
  --ratification "$AUTHOR_ROOT/research/t34_round0/REGISTRATION_ROUND0.json" \
  --out "$E/amendment-boost-floor/decision.json"
# 3. Publish the amendment ledger before reporting:
cd "$AUTHOR_ROOT"
git add -- research/t34_round0_amend/ledger/t34-round0-study-01.jsonl
git commit -m 'Preserve T34 Round 0 boost-floor amendment terminal receipt'
git fetch origin && git merge origin/main && git push origin HEAD:main
# 4. Report "$E/amendment-boost-floor/decision.json" as the Round 0 verdict.
```

A `REFUSED (nothing claimed or written)` exit leaves no trace and may be fixed and rerun, for
example after publishing the analysis ledger. A sealed KILL is final.

## 6. Round 0-D

Round 0-D imports Round 0's `summarize` unchanged (`research/t34_round0d/analyze.py:339`), so
as written it would hit the same STOP. It is still a **DRAFT** (`REGISTRATION_ROUND0D_DRAFT.json`),
so it should adopt the rule **in source before its ratification**, not through a post-hoc
wrapper:

- in `round0d/analyze.py`, right after `result = summarize(records)`, apply
  `research.t34_round0_amend.rule.amend(result, pooled_boosts(records, "teacher"), pooled_boosts(records, "bare"))`
  before the window-fault override. `amend` keeps fault precedence, and the override then
  forces STOP as before. This way `sealed_labels_admissible_to_r1` follows the amended
  decision;
- report `bare_boost_share` next to `teacher_boost_share` in its descriptive block;
- add `research/t34_round0_amend/rule.py` to Round 0-D's source binding and registration text,
  and state the amended clause in `PREREGISTRATION_ROUND0D.md`;
- add a test equivalent to `test_every_other_clause_and_the_order_match_registered_summarize`.

Not done here, because this task changes only Round 0's clause.

**Path B v2 is out of scope.** Its `analyze_v2.py:242-254` applies the same 0.5% floor to the
paced arm as a non-overridable KILL (report item 7), and it needs its own pre-data decision.

## 7. Attestation

Written by an Opus agent on 2026-10-20 (PT). It did not open, list, hash or compute anything
under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/`, and did not
modify `research/t34_round0/`, `../snake-dqn-round0-frozen-01` or `../snake-dqn-round0-fixes2`.
The only rates used come from the boost-bar REPORT, which draws on excluded A5 v2 worlds and
earlier studies. Every test uses synthetic records and temporary Git remotes. No Round 0
outcome was seen.

## 8. Open questions for the reviewer

1. **KILL after the claim.** A crash after the claim makes the governing verdict KILL, even
   though the registered analysis may be valid. This mirrors `run_phase`. The alternative is
   to fall back to the registered verdict on a mechanical failure.
2. **Timestamp precedence.** Precedence uses the committer time of the first-parent
   `origin/main` commit that added the amendment versus the analysis admission's wall-clock
   timestamp. Both could be forged by an adversarial operator. Is that, plus the
   CAMPAIGN_LOG record, sufficient?
3. **Where `decide` runs.** It runs from a separate clean `origin/main` checkout and writes its
   ledger into the authoring checkout, which is the registered `history_ledger.repo`. Should it
   instead use its own repo-tracked ledger outside `research/t34_round0_amend/`?
4. **Bare share of 0.** A bare share of exactly 0 gives floor 0, per the owner's rule, so a
   teacher at 0% passes. Is that intended?
