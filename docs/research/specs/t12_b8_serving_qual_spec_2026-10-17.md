# Spec: B8 seed 3 extended Mac serving qualification (2026-10-17)

The owner delegated the release decision to the cloud advisor, which decided RELEASE conditional on this
qualification: `docs/research/advice/2026-10-17_release_decision.md` (binding: checks 1-8, governance, release
mechanics, rollback). Implementer: Codex. Read AGENTS.md, that note, `docs/research/t7_phase_r_result_2026-10-16.md`,
the frp3 precedent (`frp3_serving_result_and_release_2026-10-06.md`, `research/frp3_checkpoint_serving_20261005/`
protocol + harness + pin tool), `web/backend/served_checkpoint.py`, `served_checkpoint_pins.json`,
`safety_veto_serving.py`, `ego2s_policy.py`, `ego_live_bv8.py`, `session.py`, STATUS.md.

Scope of THIS job: implement everything needed and RUN the qualification, but do NOT flip the released default.
1. New sibling protocol (do not edit the frp3 one) covering checks 1-8 with pass criteria; pin its sha in the harness
   and audit before any non-smoke episode; seeds disjoint from every bank in the T7 freshness audit.
2. Code: `b8-s3` registry entry (T7 checkpoint sha 3e649783..., artifact path, gitignored copy under saved_snakes/);
   a sibling pin tool binding T7's decision.json + isolated audit + independent review hashes (refuses unless audit
   PASS and STRICT_PASS recomputes; never overwrites); `resolve_served_checkpoint` fallback chain for a selected
   b8-s3 whose pin/bytes fail: frp3-s12 + v8 if its pin verifies, else champion, with a reason; S6-style test matrix
   (empty env, champion, frp3-s12, b8-s3, unknown, unfilled pin, corrupted bytes, VETO_VARIANT=v7 with no name).
   Veto absent for b8-s3 by BOTH guards (install_serving_vetoes inactive with the vector61 reason; sha not in
   v8_gated set; pin veto_variant none). Leaderboard: if scored runs do not record the served checkpoint name, add it.
3. Run the qualification on the Mac exactly per the pinned protocol, with b8-s3 SELECTED via
   `SNAKE_SERVE_CHECKPOINT=b8-s3` (not by changing the default): live-observation identity (P2-style parity for
   seed 3's pinned bytes; lockstep tests), 25 Watch + 25 Play episodes, latency at 6 and 12 snakes in Watch and
   Play (frame-time p50/p95, highest sustainable speed), UI checks (inspector groups, no veto counters, confidence
   split on 1,500 frames), Train refused, both rollbacks (SNAKE_SERVE_CHECKPOINT=frp3-s12 and =champion) exercised
   live with the expected stderr lines. Run latency only when tmux `t11-lanes` (a training run) has finished, so the
   machine is quiet; everything else may run alongside it (at most 4 processes). Never touch tmux sessions or
   processes you did not start.
4. Report `docs/research/t12_b8_serving_qualification_2026-10-17.md`: each check PASS/FAIL with evidence, the
   monitoring baseline (Watch hero deaths per 1,000 decisions, length distribution), the exact one-line default
   change and expected stderr, and a draft release record (claim + disclosures per the advisor). Tests pass; flake8.
Git: branch t12-20261017 in /Users/josenunez/Projects/ml/snake-dqn-t12; commit often; after each commit
`git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t12-20261017`.
Final message: check table, latency numbers, monitoring baseline, any FAIL with cause, sha. Do NOT change
CHECKPOINT_RELEASED_DEFAULT.
