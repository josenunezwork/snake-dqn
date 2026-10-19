---
name: snake-dqn-preregistration
description: |
  Use whenever you write, amend, implement or report against a pre-registration, protocol, rule spec,
  decision analyzer or RESULTS/result document for a snake-dqn study in
  /Users/josenunez/Projects/ml/snake-dqn (research/*/PREREGISTRATION*.md, research/*/protocol.md,
  docs/research/specs/*prereg*.md, *_result_*.md, RESULTS_*.md, decision.json/verdict producers), or when
  you are about to launch a study, extend it, pick a checkpoint/seed, or spend budget on it. It encodes
  the evidence discipline: prereg committed before data, fixed decision rules, amendments before data,
  owner-release guards, caps, and how results are written.
---

# Snake DQN: pre-registration and evidence discipline

Why this exists (`docs/research/governance_tiers_2026-09-26.md`): 12 of 13 invalid studies in Sep 2026
failed on tooling or rules, not science, and the one thing governance must prevent is changing the rule
after the result (`docs/research/advice/2026-10-17_release_decision.md`). Model files to copy from:
`research/redesign_t7_phase_r_20261016/PREREGISTRATION_T7.md` (fresh-bank one-look Phase R),
`research/t23_r0_screen_20261017/PREREGISTRATION_T23.md` (short dev screen), 
`research/q0q1_20261017/PREREGISTRATION.md` (dependency graph, budgets, factorial),
`research/redesign_m4_20261013/PREREGISTRATION_M4.md`, `research/frp3_checkpoint_serving_20261005/protocol.md`
(serving lane). Results: `docs/research/t7_phase_r_result_2026-10-16.md`,
`research/t23_r0_screen_20261017/RESULTS_T23.md`, `docs/research/m6_result_2026-10-14.md`.

## 1. Order of operations (never reorder)

1. Advice/spec exists (`docs/research/advice/*.md`, `docs/research/specs/*_spec_*.md`) and names the
   question, comparator, endpoint and rules it adopts.
2. Write `PREREGISTRATION*.md` (or `protocol.md`) in the study package `research/<study>_<yyyymmdd>/`.
   Header says what it is: `DRAFT_PENDING_OWNER_REVIEW` until the owner ratifies; then the ratification
   line is added at the top (T7: "RATIFIED 2026-10-16 by the owner's explicit instruction ..."; Q0Q1:
   "Owner approval (2026-10-18, chat): ..." with the original header preserved below it).
3. Commit and push **before any data exist** ("Written and committed BEFORE any T23 training data exists").
   Rule tests and the implementation receipt pass before the frozen run plan is created.
4. Allocate banks, run the seed/collision audit, commit `SEED_AUDIT*.json`. A collision rejects the whole
   allocation; register the next round in a sibling erratum (T7 errata 74 -> 75 -> 76), never swap worlds.
5. Create-only `plan.json` / `start.json` / intent binding commit SHA, source hashes, checkpoint bytes,
   banks, rosters, profiles, rule spec sha. The runner refuses a different source/commit identity.
6. Run. Monitor only operational progress, errors and hashes. **No outcome metric is examined before the
   registered look.**
7. Isolated audit (`python -I`, stdlib-only auditor, separate from the producer), independent review,
   then the decision, then the result doc. Publish evidence hashes.

Nothing in a registration authorizes compute, dispatch, serving changes or release unless it says so
(Q0Q1 prereg: "This document does not authorize real training, Colab/pod dispatch, tmux operations,
serving changes, or release decisions").

## 2. What a pre-registration must fix

For a planned new Colab arm, name the `research/colab_cli` transport and its manifest fields, account-wide
assignment cap, wall/unit caps, lease/export limits, and receipt. Do not authorize a Drive queue job except for
legacy Q1 recovery; the supervisor does not turn a registration into spending approval.

- **Question + comparator + endpoint**: e.g. "terminal-hero H5000 full-horizon mass integral vs released
  bare B8 `3e649783...`, dead frames zero". Pin every identity by SHA-256 (candidate finals, incumbent,
  champion, wrapper variant/method, config digests, deployment profile digests).
- **Design**: training seeds (collision-audited against research/, docs/, scripts/), world counts per mix
  per seed (T7 192/96/96; T23/Q1 64/32/32), namespaces/round keys, horizons (H5000 exact prefix of
  H10000), prefix controls, which checkpoint decides (only the final; 2M/4M descriptive), one look or a
  sequential schedule, extension bank if any (reserved **once**, pre-allocated).
- **Statistics**, exactly: per-seed equal-mix mean, HK pool with df=4 or one-sided 90% Student-t across
  seeds, LB90/UB90 definitions, feasibility formula, candidate-selection rule (80% WS lower endpoint,
  ties -> lowest seed), survival tolerance (1e-9).
- **Decision rules with precedence** (see section 3) and the exact labels that may be emitted.
- **Tripwires** (training): non-finite loss/Q/grad, >90% one-action share in a full 100k window, greedy
  boost >2% overall or >1% below length 700. A tripwire kills that arm; no replacement seed.
- **Budgets/caps**: units, GPU-hours, CPU-hours, episodes, per-job timeouts; "stop at the first applicable
  cap", reserve 20% before dispatch, charge setup/copies/failed attempts (Q0Q1 section "Shared governance
  and budget"). Measure per-unit time before setting est/timeout (`snake-dqn-colab-jobs`).
- **Execution constraints**: platform (Mac CPU only for strict/serving), threads = 1 per worker, worker
  count, tmux session owned by the task, AC/open-lid/thermal guard, artifact root (create-only), Git
  branch/worktree, push cadence (`git fetch origin && git merge origin/main`, push to main and the branch).
- **Disclosures to carry**: historical sensitivity tables that motivated the study (T7 carried M4 look-2
  -0.156/-0.188, census -0.039/-0.157, pooled -0.003/-0.050) labelled "historical, never re-scored".
- **What happens after** each outcome, including "if it fails, record it, apply no second amendment".

## 3. Decision labels and their precedence

Apply in this order, per arm (T23 `PREREGISTRATION_T23.md` section 6; Q0Q1 "ordered decisions"):

1. **INVALID / INCOMPLETE**: missing rows/hashes/identity, unclean exit, sample-count mismatch, failed
   parity, cap expiry, counter-reconciliation failure. Preserve evidence; no auto-rerun, no seed
   replacement, no training extension, no intermediate-checkpoint substitute. Not a scientific result:
   "must not be relabeled as a pass or fail".
2. **KILL**: tripwire; pooled alive-frame diff < -0.03 on any mix; any seed scripted alive diff < -0.15;
   pooled no-death survival < 0.50 on any mix; `upper90(G) <= 0`. KILL precedes GO.
3. **GO** (development; "long-run planning only" / "to fresh Mac Phase R preparation"): `lower90(G) > 0`,
   mean G >= 20, scripted mass point >= 0, >= 4/5 seeds positive, all guards pass. A GO is neither a
   strict pass nor a release.
4. **EXTEND** once (if registered, T23) with the pre-reserved bank and the same checkpoints; then
   **INCONCLUSIVE**. Q1 registered **no** extension: "every other valid result" is INCONCLUSIVE.
5. Strict/release gates emit **STRICT_PASS** or **FAIL** only (T7), plus retained failure reasons and the
   non-binding futility flag (`NO_GO_EARLY` when UB90 < G=50 and clauses fail).

Rules the project has made explicit, repeatedly:
- "No extra training, no best-checkpoint swap, no seed replacement after data." (T23)
- "A healthy arm is not stopped for a poor 2M curve; no intermediate checkpoint may replace the final." (Q1)
- "A complete pass has precedence over this nonbinding futility." (T7)
- Fixed-N registrations are never stopped early; stopping invalidates them (rule of 2026-10-02).
- Choose at most one recipe; rank by `lower90(G)`, then mean G, then a fixed identifier order (Q1: D,C,B,A).
- A preservation certificate (`lower90(G) > -10`, scripted point >= -10) admits only the next study; it never
  makes an inferior recipe "equal".

## 4. Amendments

- Allowed only **before the affected data exist**, as a **sibling file** (`SEED_ALLOCATION_ERRATUM_T7.md`,
  "Amendment 1 ... before seeds 7104/7105 started; no evaluation data exists", Q0Q1 "Amendment A1 (before
  any Q0/Q1 data)"). Original bytes stay unchanged (hash-bound by `prepare`/plan).
- State what it supersedes (only the round/key; only execution location; only the cap) and that
  thresholds, clauses, N, identities, look discipline and serving scope are unchanged.
- Execution-only changes (GPU type, runtime count, CPU-on-Colab evaluation) still need an amendment and
  make non-conforming units INVALID ("A seed on a different GPU type is INVALID for this screen").
- Survival-guard amendment for the redesign line was explicitly **one** ("no second amendment after a
  second failure", `docs/research/specs/guard_clause_amendment_2026-10-15.md`). Do not draft another.
- Governance amendments (survival band v2, sequential Phase R) are prospective and opt-in per
  registration; `prepare` refuses unless the ratification section reads exactly `- Decision: ratified`
  with no changes. Never relabel a closed study under a later rule.

## 5. Owner release and executable guards

Registrations are enforced in code, not by convention:
- Q0/Q1 jobs refuse to run unless `Q0Q1_REGISTRATION_STATUS=APPROVED_Q0Q1_20261017` **and**
  `Q0Q1_OWNER_RELEASE=APPROVED_Q0Q1_20261017` (`research/q0q1_20261017/run_queue_job.py:
  require_owner_release`; `queue_jobs.py REGISTRATION_STATUS = "DRAFT_PENDING_OWNER_REVIEW"`;
  `make_q1_jobs.py` injects the release token; `q0b_measure.py` checks it). The overnight 2026-10-18
  guard correctly blocked `q0-audit-*` jobs while the prereg was DRAFT (`docs/research/overnight_2026-10-18.md`);
  never bypass or spoof these env vars.
- Phase R producers refuse a non-smoke plan without the ratification line (M4 `phase_r_m4.py`); T7's
  runner rejects any source/commit identity other than the frozen one.
- Strict template `prepare` binds the amendment doc sha and ratification text; serving harnesses pin
  `PROTOCOL_SHA256` in code; a non-smoke intent with another sha cannot pass.
- Served checkpoints are pin-gated by `web/backend/served_checkpoint_pins.json` (fill only with the
  study's `fill_strict_pin.py` after decision + audit + review verify).
- Owner approvals are quoted verbatim with date and channel ("everything approved for today dont ask me
  again", 2026-10-18 chat) and scoped to what was approved. Standing authority exists for the improvement
  loop with per-step caps (~$10 pod steps, 4x rule); serving releases and roster switches needed explicit
  owner OK (T12C, T15B).

## 6. Budgets and compute in a registration

- State units/hours/episode caps per phase and the global envelope; record observed debit rates (never
  record an unknown rate as zero). Unused reservations are still part of the forecast.
- Colab/pod work is development evidence; strict and serving stay on the Mac (`docs/STATUS.md`).
- Per-step RunPod rule >= 4x faster than the Mac, measured incl. setup/transfer; waivers are per step and
  recorded. Never launch long runs on battery.
- Register `K` actors, `W` worlds, `N` seats and the learner-decision overshoot bound `K*W*N-1`; a receipt
  exceeding it is INVALID (Q1 "Registered counter overshoot").

## 7. Writing the result document

Pattern (`t7_phase_r_result_2026-10-16.md`, `RESULTS_T23.md`, `m6_result_2026-10-14.md`,
`t12_b8_serving_qualification_2026-10-17.md`):
1. Title with study, date; first line **"Outcome/Decision/Verdict: <registered label>"** and one sentence
   of what it does and does not authorize ("Serving qualification and release remain a separate
   planner-approved step"; "A GO is neither a strict pass nor a release").
2. Pointer to the prereg (+ amendments), commit SHAs for run and evaluator, platform, seeds, wall time,
   whether any interim look, extension, rerun or deviation occurred (list deviations explicitly, e.g.
   "Actor count deviation ... recorded as a deviation"; "first launch ran on CPU by mistake (deviation 1)").
3. **Clause table**: estimate, SE, threshold, PASS/FAIL per registered clause, with exact (unrounded)
   values; futility/feasibility numbers; candidate-selection table.
4. Comparators spelled out per row (vs served agent / vs champion / vs CTRL / vs B8) and the metric
   (mass integral, alive-frame, episode survival).
5. Sensitivity and disclosures: historical tables carried forward "disclosed only, never re-scored";
   narrow margins stated ("passed by 0.034 at SE 0.023"); what the rosters did not gate.
6. Operational notes and failures that cost time (duplicate Drive folders, process-group signal error,
   one-core eval) so the next registration avoids them.
7. Evidence hashes: decision/verdict JSON, audit JSON, independent review, receipt, manifest; artifact
   root path; which bytes are frozen. Publication commit SHAs (pushed to main).
8. "Implication / next" one paragraph; never promote a label (a KILL closes that design "as designed";
   a NULL "not adopted").
Also update `docs/STATUS.md` ("Latest results" row) and `docs/research/INDEX.md`; STATUS wins over the
result doc if they later diverge, so keep them consistent. Erratum-only edits for anything hash-bound.

## Before you say done
- Prereg (and every amendment) committed and pushed before the data it governs; header states status.
- Decision rules have explicit precedence and exact thresholds; labels are the registered ones only.
- Seed audit committed; banks fresh; identities pinned by SHA-256; counter/overshoot bounds registered.
- Caps and est/timeout derived from measured per-unit times; owner release token/guard in place and
  not bypassed.
- Result doc follows section 7, names comparator and metric for every number, records deviations, and
  says what the result does not authorize.
