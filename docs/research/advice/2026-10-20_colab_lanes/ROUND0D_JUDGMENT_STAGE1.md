RULING: APPROVE

# Round 0-D rescope, stage 1 (pre-data ratification): independent judge, 2026-10-20 night PT

Judge: Fable 5.1, independent (wrote none of the package, the reviews, or the rulings). Read-only. Nothing under
`snake-dqn-artifacts/t34-round0-study-01/` was opened; no Round 0 outcome is known to me. No jobs, tmux or Colab
were touched. This doubles as the final re-review of FIX-FIRST 2 (REVIEW_RESCOPE_2.md).

Target verified: branch `t34-round0d-rescope-20261021` @ `9730fed22a2dc584f6f5eb37abb8d25d565dee66`, clean checkout
at `.claude/worktrees/agent-a276d68378b46d40d`.

- tree `research/t34_round0d` = `9d8870d8a35fed3d14b1eb6edc064488d74e22e9`
- blob `tests/test_t34_round0d.py` = `5ccb1aed73505ec301909a1b9929f363c610a54b`
- `PREREGISTRATION_ROUND0D.md` sha256 `9fbb572d89d069354b703d84483fc26b2d8312d4b22eb5a6c70e0fe63b0b6b06`
- `REGISTRATION_ROUND0D_DRAFT.json` sha256 `5f7edcb830c4f300fbbae785008ace47df58d07fc75fab3571d95eb41ef7f937`
- `VERIFICATION_ROUND0D.md` sha256 `aff53d46464212c51171d7bdb0f68df88810db04d5ea0a31c762a5b45c8a6c00`

Tests: `tests/test_t34_round0d.py` run as instructed (nice 10, one thread): **89 passed** (was 37 at 392a97f9).
`registration.draft()` regenerated into my scratchpad from 9730fed2 is identical to the committed DRAFT except
`built_from_commit` (see §3).

## 1. Review-2 blocker N1 (slow-world stranding): FIXED in code, test and prereg

`git diff 392a97f9..9730fed2` was read in full (10 files, +522/-131).

- `sizing.plan` (`sizing.py:196-233`) now charges every chunk at its straggler makespan
  `setup + ceil(n/12)·max(slowest, wave)` at 2.0 u/h (`straggler_minutes` from `chunk_record`, `:140`), runs the
  2-lane admission simulation `admission_schedule` (`:245-265`) against `120 − replay cap`, and KILLs with reason
  "measured straggler schedule strands under 120 u" if any chunk is refused or `Σ straggler + replay cap > 120`.
  The gate sits before the wall gate, so it kills on its own (tested with the wall cap disabled,
  `tests:536-553`).
- The simulation matches the dispatcher: chunk k admitted while chunk k−1 is in flight at full cap, earlier chunks
  settled at their charge, the admitted chunk at its full cap (`admission_spend`, `:312-342`;
  `dispatch.py:243-266` uses the same function with `chunk_admission_ceiling` and `prior_spend` charges, which
  include failed/mechanical-rerun attempts).
- The wall forecast now uses straggler makespans (`:218-222`).
- Reviewer's requested test exists: slowest = 1.6×wave at r = 7 → KILL, and a 9×6 grid (r × slowness) where every
  RATIFY is re-checked by an **independent** simulation written in the test (`_stranded_independently`,
  `tests:495-513`, does not call `admission_schedule`) plus `admission_spend` per chunk.
- Prereg Sizing section, condition table and ESTIMATED table updated (`PREREGISTRATION_ROUND0D.md:425-472`); I
  recomputed the table from the regenerated draft and the nine rows match (threshold at placeholder slowness
  ~r = 9; r = 6.5–8 KILL on straggler wall; r = 6 KILL on straggler schedule, first stranded chunk 18).

## 2. Review-2 non-blocking notes: all fixed

| Note | Where | Verified |
|---|---|---|
| HOLD resolution not bound to the run | `sizing.resolved_fidelity` (`sizing.py:408-447`) | A record counts only if `l4_export_sha256` equals the supervisor-verified receipt export `fidelity-<i>.json` for that world (index aligned with `a5d.fidelity_status` enumeration order and the export name at `a5d.py:232`), `checkpoint_sha256 == B8_SHA`, `mac_reproduces_t3 is True`, no `mac_differences`, `tie_epsilon == 1e-4`, schema and `world_id` match. INCOMPLETE worlds (no export) can never be resolved. Tested (`tests:568-611`), including missing receipt/exports. |
| Greedy-gap leg explaining a batched divergence | `a5d.platform_tie` (`a5d.py:408-438`), `resolve_hold` | Legs are exclusive: a label at the divergence (batched) → tied-longest Q gap only; otherwise greedy advisory top-2 gap. `explained` additionally requires exact T3 reproduction. Tested (`tests:614-640`). Prereg text updated (`:375-387`). |
| 120 u was an admission ceiling, not a hard total | `chunk_admission_ceiling = 120 − replay cap`; new chunk charged at its cap | Enforced in `require_registration` (`registration.py:177-183`: must equal 120 − replay unit_cap, no stranded chunk, straggler total ≤ 120) and used by `dispatch.py:244`. Replay manifest caps at timeout × 2.0 u/h (`replay.py:187-190`). Hence settled chunks + in-flight caps + replay cap ≤ 120 by construction, given the supervisor's hard stop. Prereg (`:280-291`) now says "hard total" and discloses the residual (a world slower than A5-D measured → integrity INCONCLUSIVE, never a silent overrun). |
| Outcome precedence "computed mechanically" with no code | `analyze.outcome_reading` (`analyze.py:97-127`), recorded as `outcome_reading` in decision.json | Precedence fault → point bars → survival → boost; an unmatched STOP raises InvalidEvidence. Never alters `decision`. Tested (`tests:400`). Prereg updated (`:234-238`). |
| Wall forecast ignored stragglers | `sizing.py:218-222` | Fixed with N1. |
| Stale `dispatch.py` docstring | `dispatch.py:5-11` | Fixed. |

## 3. §7e / §7f / RULING_round0_go_file requirements: hold

- Relative boost floor imported (`analyze.py:375-383`, pinned `t34_round0_amend/rule.py`), not the absolute floor.
- Frozen r→action table: KILL r < 5; RATIFY only if expected ≤ 120 u and wall ≤ 24 h on the 2 registered lanes;
  now additionally derived ceiling ≤ 120, straggler schedule admits every chunk, straggler wall ≤ 24 h. Stricter
  than §7e, never looser.
- A5-D: 300 min timeout, 10 u cap (`make_manifest.py:30-32`, `:212`), slowest-world sizing feeds timeouts and now
  the straggler charges; pre-registered Mac-only HOLD path with ε = 1e-4.
- r from completed process-seconds, `min(r_rows, r_completed)` (`sizing.py:82-118`), unchanged since review 2.
- Replay timeout ≥ max(estimate + 25, 2×estimate, 60) (`sizing.py:273`).
- Ceiling formula per ops finding 5 retained as the derived-ceiling gate; the operative ceiling is 120 u.
- Dense R1 yield rule ≤ 200 u, ≤ 24 h on 3 lanes, held-out minimum reachable (`dense_r1_plan`), labels-only until
  the PB disposition (fit preconditions recorded).
- Round 0 binding: `ratify` requires `bind_round0_verdict` (governing file, phase_terminal, ledger commit),
  `require_registration` demands `route == "ROUND0D"`; draft `dispatch_condition` names the governing file.
- Nothing else regressed: the diff touches only the Round 0-D package and its tests; generic Round 0 settlement
  helpers are imported unchanged.

## 4. The draft's `built_from_commit` (392a97f9): no rebinding required

The DRAFT is regenerated before each commit, so its `built_from_commit` can never equal the commit that contains
it; it is informational only. Nothing consumes it: `ratify` binds `commit = head(repo)` at activation
(`registration.py:385`), `require_registration` demands `head(repo) == commit` and `source_hashes` equal to the
ratified closure, and `require_published_registration` demands the ratification commit be on origin/main after
the source commit. The DRAFT's `preregistration.sha256` already equals the final prereg bytes
(`9fbb572d…`), and its content is byte-identical to a regeneration from 9730fed2 apart from that one line.

**Decision:** stage 1 ratifies the *content* pinned in the header above, not a commit id. Do not amend the DRAFT
to rebind it (an amend would only move the lag to the next commit). The stage-2 activation (`ratify`) will bind the
real source commit automatically.

## 5. Conditions (mechanical; no further review needed)

1. **Merge without edits.** Merge `t34-round0d-rescope-20261021` to main (fast-forward or merge commit) such that
   main's tree for `research/t34_round0d` is `9d8870d8…` and `tests/test_t34_round0d.py` is blob `5ccb1aed…`.
   Any byte change under those two paths before stage 2 lapses this approval and needs a fresh judge pass.
2. **Stage-2 activation binds that main commit.** `registration.ratify` must run with `head(repo)` = a commit on
   origin/main whose tree satisfies condition 1 (the code enforces the rest: governing Round 0 scientific
   STOP/INCONCLUSIVE, source-current A6 preflight after the final commit, A5-D receipt with fidelity
   PASS/EXPLAINED_TIE/EXPLAINED_PLATFORM and a RATIFY row). Stage 1 authorizes no dispatch and no A5-D beyond
   what the package already allows (`may_run_pre_verdict`, ≤ 10 u).
3. **Exposure disclosure at activation.** Record in CAMPAIGN_LOG that L7 exposure before dense R1 is the 120 u
   study hard total plus A5-D ≤ 10 u (priced separately per §7e), i.e. ≤ 130 u; dense R1 stays ≤ 200 u under
   its own yield rule. This is a log line, not a code change.
4. **Record this ruling.** Reference this file and the hashes above in the ratification commit message or
   CAMPAIGN_LOG entry.

## 6. Non-blocking observations (no action required)

- The "hard total" rests on the supervisor's timeout hard stop; a settled receipt's elapsed time can exceed the
  timeout by billing granularity (minutes). At ~2 u/h this is pennies and is already covered by the disclosed
  integrity-INCONCLUSIVE path.
- The straggler model charges every wave of a multi-wave chunk at the slowest world; this is conservative (biases
  toward KILL), which is the right direction for a budget gate.
- A single mechanical rerun of a chunk is charged via `prior_spend`; it is not pre-simulated by
  `admission_schedule`. If a rerun pushes settled charges over the ceiling the dispatcher seals INCONCLUSIVE,
  as pre-registered.

Blockers: none.
