# Spec: B8 fresh-bank Phase R under the Tier 1 guard (2026-10-16)

Owner: "ask Fable and go with that"; the advisor chose option A
(`docs/research/advice/2026-10-16_phase_r_choice.md`, binding, including its pre-registration details).
Implementer: Codex. Read AGENTS.md, the advisor's choice note, `docs/research/specs/guard_clause_amendment_2026-10-15.md`
(Tier 1 / Tier 2), the M4 Phase R package `research/redesign_m4_20261013/` (rule_spec_m4.py, m4_select.py,
phase_r_m4.py, PREREGISTRATION_M4.md), `research/sequential_phase_r/` (example_rules.py, hooks), the M4
result `docs/research/m4_phase_r_result_2026-10-13.md`, and `docs/research/t5b_risk_and_fuse_2026-10-16.md`.

1. Pre-registration FIRST (`research/redesign_t7_phase_r_20261016/PREREGISTRATION_T7.md`, committed before
   any gameplay): hero = the five B8 finals (`redesign-m4b-mac-20261013/B8_s{0..4}/final.pth`, sha256-pinned),
   no veto, the M4 serving path; baselines frp3-s12 + v8 (served agent, incumbent) and champion_a5 + v8 on the
   same worlds. Fresh bank namespace/round with a seed audit (unused by every previous bank, incl. dev probe
   rounds 50-73, the census, M3/M4 Phase R namespaces, the T3/T4 label/return/snapshot collections). Worlds per
   seed: 192 scripted, 96 frozen, 96 mixed; ONE look at full N; keep only the existing mass futility. Clauses:
   M4's rule with c6 replaced by Tier 1 exactly (c2 alive-frame vs served agent >= -0.03 pooled per mix;
   per-seed alive-frame floor >= -0.15 on each seed's own bank; absolute pooled episode survival >= 0.50 per
   mix); efficacy (mass vs incumbent), mi10, guard mass point, positive-seed count unchanged. Tier 2 (episode
   survival vs champion >= -0.05 pooled) reported, not deciding, with the sensitivity table. Candidate seed:
   the existing rule in example_rules.py applied to seeds that pass their per-seed floor, ties to lowest seed,
   fixed now. Disclosures: world count chosen prospectively for power citing the 38.8% risk estimate from
   existing banks; B8 trained with v8 labels; the M4 PARTIAL motivated the amendment; Tier 2 open. After
   STRICT_PASS: Mac serving qualification exactly as for frp3-s12 (reuse that procedure), rollback env vars.
   If it fails: record, no second amendment, frp3-s12 + v8 stays served.
2. Rule tests: synthetic records proving every clause, the one-look decision and the candidate rule.
3. Run it on the Mac: CPU evaluation, up to 6 single-thread processes (a training run may use the MPS GPU and
   some CPU concurrently; never touch it or any tmux session/process you did not start), tmux `t7-phase-r`,
   resumable, hash-bound records, AC/thermal guard as in previous Phase R runs. Do not use RunPod.
4. Decision + report `docs/research/t7_phase_r_result_2026-10-16.md`: every clause with estimate/SE/threshold,
   decision (STRICT_PASS / FAIL with reason), candidate seed, Tier 2 line, sensitivity table. Do NOT change any
   serving default; serving qualification and release are a separate, planner-approved step.
Git: branch t7-20261016 in /Users/josenunez/Projects/ml/snake-dqn-t7; commit often; after each commit
`git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t7-20261016`.
Poll until the decision is pushed. Final message: decision, clause table, candidate seed, Tier 2, wall time, sha.
