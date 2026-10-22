RULING: YES — DENSITY-2 completion may make ONE more canary allocation (`canary-04`) after the canary-staging metric fix is merged and the Colab CLI lane is re-QUALIFIED on that tree, followed (on canary PASS) by the completion job exactly as JUDGMENT_RESUME_POLICY §1 authorizes (completion-01, and the pre-authorized mechanical completion-02 under its condition 5). The three canary failures are gate/supervisor events, not completion attempts: no completion job has ever been dispatched (`production_completion_attempts_consumed = 0`), no episode of the registered namespace ran, no outcome was read, and in two of the three the remote skip-ledger job itself succeeded (9 planned − 3 skipped, exports hash-verified) while only a mis-specified timing rule (177.8 s "staging" = GitHub source delivery + uploads, actual uploads 21.2 s) refused it. The `maximum_canary_attempts = 2` in the pack is the operator's own packaging rule, not a judge's cap, and is replaced by this ruling. HARD STOP: if canary-04 fails for ANY cause (gate, supervisor, transport, runtime, allocation), DENSITY-2 is TERMINAL immediately, with no further judgment, no gate re-definition, no further canary. Terminal means: the one terminal look is `analysis.py merge` over the 408 verified exports (allowed), reported under the caveats in §3 — C-P1 is undelivered (0/72, no claim of any kind), B70p/mixed is judged at n = 24/36 by the registered rule without relaxation, missingness is transport-order (alphabetical tail), so within-cell unbiased but underpowered; no imputation, no replacement units, no pooling with the pilot, and the completion option is extinguished the moment that look is taken.

# DENSITY-2 third-attempt judgment — one more canary, then terminal

Judge: independent Fable, 2026-10-22, read-only. Inputs read in full: `big-lever/JUDGMENT_RESUME_POLICY.md`,
`exp/bl-density/JUDGMENT_DENSITY2_CAP.md`, `big-lever/LIVE_FACTS.md`, the three completion packs
`density2-completion-20261022-{01,02,03}/` (`DELIVERY.json`, `canary-output/receipt.json`, `source-delivery.json`,
`upload-measurement.json`, `warmup-measurement.json`; export inventories by path/size/hash only — no outcome field of
any episode, canary or production, was opened), and the `production-output-02` export tree listing (408 `.complete`).
`t34-round0-study-01/` not opened. No jobs, no Colab calls, no commits.

## 1. Facts (verified from receipts)

| pack | canary job | what failed | where | units | remote job | exports |
|---|---|---|---|---|---|---|
| -01 | `…-56f20aee` | `bootstrap.sh` first upload stalled 134.1 s (fresh-runtime warm-up); staging rule fired | supervisor | 0.13 (1120.04 → 1119.91) | `succeeded`, 9/9 scheduled done in 47.9 s | 11, hashes in inventory |
| -02 | `…-96e9b776` | `GuardError: tracked session lacks kernel probe metadata` | allocation guard | 0.00 (1119.79 → 1119.79) | never started | 0 |
| -03 | `…-0e7bbd30` | warm-up probe PASS 3.65 s; `upload-measurement 177.8 s` vs 150 s rule; six real uploads sum **21.2 s** (4.7 / 3.3 / 3.4 / 3.5 / 2.9 / 3.4, all `verified`); remainder = GitHub source delivery (`source-delivery.json: transport github`) | canary metric definition | 0.21 (1119.35 → 1119.14) | `succeeded`, 9/9 scheduled done in 49.6 s | 11, hashes in inventory |

- All three: `stop_verified true`, `failure_class SUPERVISOR`. Canary namespace is `bl-density-round14b-canary-20261022`,
  distinct from the production namespace; production seeds untouched.
- `production_completion_attempts_consumed = 0`; completion ledger `skip-ledger.json` sha `70b252ec…` identical across
  all three packs; `verified_prior_episodes = 408`; `missing_registered_episodes = 84` (72 C-P1 + B70p/mixed indices
  22, 25–35). The RESUME_POLICY text says "85" and "24–35" — a miscount there; 492 − 408 = 84 is the verified figure.
  The runner plans "allocation minus ledger" mechanically, so no manifest text depends on the list; this note corrects
  the record only.
- Completion-series spend to date: 0.34 u measured (0.13 + 0 + 0.21). DENSITY-2 series to date ≈ 4.50 + 0.34 = 4.84 u.
- The substantive canary PASS criteria of RESUME_POLICY §1.1 (skipped units not recomputed; non-skipped exports
  recovered and hash-verified) were met by the remote jobs in packs -01 and -03; only the timing rule refused them.
  That evidence cannot itself serve as the canary proof (the identity rule requires proof == current == manifest pin,
  and the supervisor tree is changing), but it is the reason this is a gate defect and not a transport failure.

## 2. Why one more, and why only one

- **No scientific cost.** The decision to complete was taken outcome-blind in RESUME_POLICY and remains so: nothing
  observed since is an outcome. A further canary adds no look, no unit, no sizing change. The attempt cap in
  RESUME_POLICY §1.5 ("at most two completion attempts") counts completion jobs; it was written for lost units after
  episodes ran, not for a gate that refuses on its own metric. None has been consumed.
- **Owner rules.** Budget rule is "2× generous, never idle"; the entire completion series is 0.34 u so far and the next
  canary ≤ 0.5 u. Budget is not the constraint. The idle-lane cost is the real cost, and it is paid only while Sol
  fixes and requalifies — lawful under JUDGMENT_DENSITY2_CAP §4.1.
- **Why not unlimited.** "The gate was wrong, fix it, rerun" cannot be open-ended: each redefinition after a refusal is
  a post-hoc rule change on operational data, and an unbounded loop converts a pre-data gate into "retry until green".
  Bounding it at one further allocation, decided now before the fix is written, keeps the gate pre-data for the run
  that matters.

## 3. Conditions (all mandatory)

1. **Fix before canary, frozen before dispatch.** The staging metric must time only the supervisor's own uploads (the
   `transfers` list, or an equivalent measured span that excludes source delivery), with its threshold stated in the
   requalification note; the fixed rule must be re-scored against the three existing receipts and the results recorded
   (expected: -03 PASS on uploads 21.2 s; -01 FAIL on the 134 s stall, which the warm-up probe now covers; -02 n/a).
   Re-scoring is evidence for this ruling, never a substitute for canary-04. After requalification the tree is
   pinned; any later supervisor edit requires requalification again but does NOT grant another canary.
2. **Canary-04** = the RESUME_POLICY §1.1 skip-ledger canary, same `skip-ledger.json` (sha `70b252ec…`), same
   `canary-ledger.json` (sha `1fac1b13…`), ≤ 12 episodes, ≤ 0.5 u / ≤ 20 min, PASS criteria unchanged. New pack,
   fresh output dir, own receipt; prior packs immutable.
3. **Then completion-01** exactly per RESUME_POLICY §1.2–§1.4 (same commit f1904777 for the science, same allocation,
   `one_terminal_look = true`); `prior_spend_units` must carry every charge of the series incl. all canaries
   (≥ 4.84 u + canary-04, measured conservatively). Condition 5 of RESUME_POLICY (mechanical completion-02) stays in
   force unchanged — it is the only further job this ruling leaves open.
4. **Hard stop.** Canary-04 failure of any kind, or completion-01 and completion-02 both failing, ⇒ TERMINAL per §4
   below. No judge may be asked for a fourth canary or a third completion; a future transport fix does not reopen
   DENSITY-2 (it may inform a NEW registration of a new study only).
5. **No peeking**, unchanged: until `merge` writes `summary.json`, no tool or person opens an outcome field of any of
   the 408; ledger builders read path/size/sha256 only; remote per-attempt `summary.json` sealed. A peek voids
   completion and the study reverts to INCOMPLETE_PEEKED.

## 4. Terminal — what "terminal on what is verified" permits

If the hard stop fires, DENSITY-2 is closed to new units forever and the following is the one and only look:

- **Allowed:** `analysis.py merge` (or the registered summarize path on the Mac) over exactly the verified exports
  present at that time (408 today; more only if completion-01/02 delivered some), with the registered thresholds,
  features, stride, horizon and per-X×mix rule unchanged. Report `attempted/complete` per cell.
- **Caveats that must appear in the result:**
  - Missingness is the alphabetical tail of the transport inventory — determined by file order, independent of any
    outcome — so the delivered subset of a partially delivered cell is an unbiased subsample of that cell; power is
    reduced, nothing else.
  - **C-P1 arm: 0/72 delivered.** Every C-P1 cell is `INSUFFICIENT_DELIVERY`. No C-P1 statement of any sign, including
    "no evidence of effect", may be made; the arm is simply unmeasured.
  - **B70p/mixed: 24/36.** The registered rule is applied as registered at n = 24; if the rule's admission depends on
    an interval or count that n = 24 cannot satisfy, the cell is `INSUFFICIENT`; it may be REJECTED by the rule but
    can be ADMITTED only if the rule as written passes at n = 24 — no relaxed, re-powered or imputed version.
  - Parent-run overhead (`finalize_cpu`) is lost with production-02's summary and, absent a completion run, cannot be
    charged uniformly; CPU endpoints are reported on per-episode `cpu_seconds_by_stage` with overhead marked
    NOT MEASURED, and any cell whose verdict would change under any non-negative uniform overhead is `INSUFFICIENT`.
  - No imputation, no re-weighting toward the registered allocation, no replacement episodes, no pooling with the
    pilot, no re-sizing.
- **Order of decisions.** The terminal look may be taken only after the hard stop has fired. Once it is taken, the
  completion option is extinguished permanently (a completion after a look would be an extension, forbidden by
  RESUME_POLICY §2 and the spare-slot ruling).

## 5. Must not change

Pre-registered thresholds/features/stride/horizon/B8 bytes and amendment SHAs; no third enlargement; pilot never
pooled; ≤ 180-min wall cap per job; L4 1.54 u/h ceiling; canary identity rule (proof == current == manifest pin);
Claude dispatches, Codex never; served hero B8 s3 untouched; release OK stays with the owner; sealed study closed.

Not committed by the judge.
