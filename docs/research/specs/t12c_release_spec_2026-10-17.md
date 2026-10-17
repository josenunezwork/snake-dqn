# Spec: release B8 seed 3 as the served Watch hero (2026-10-17)

All 8 checks of the extended qualification passed (`docs/research/t12_b8_serving_qualification_2026-10-17.md`, incl. the
check-4 amendment). The advisor's decision (`docs/research/advice/2026-10-17_release_decision.md`, "Release mechanics" and
"Monitoring") is binding. Implementer: Codex. Read AGENTS.md, that note, the frp3 release record
(`frp3_serving_result_and_release_2026-10-06.md`), `web/backend/served_checkpoint.py`, STATUS.md.

1. Set `CHECKPOINT_RELEASED_DEFAULT` to the b8-s3 name. Fallback order as qualified (b8-s3 -> frp3-s12 + v8 -> champion).
   No veto flag, obs-spec or engine default changes. Update tests that assert the old default.
2. Verify live with the released default env: start the server briefly and confirm the stderr lines
   `served-checkpoint: name=b8-s3 ... sha256=3e649783...` and `safety-veto-serving: active=False ... reason=applies only
   to a vector61 Apex policy`; then `SNAKE_SERVE_CHECKPOINT=frp3-s12` (frp3-s12 + v8 active) and `=champion`
   (champion + v8) each start correctly. Record the exact lines.
3. Release record `docs/research/b8_s3_release_2026-10-17.md` mirroring the frp3 one: claim "more mass at equal survival
   against the served agent, with no veto" (not "safer"); evidence (T7 Phase R STRICT_PASS + Tier 2, T12 qualification);
   disclosures (trained with v8 labels, served without a veto, Tier 2 margin 0.034 above threshold at SE 0.023,
   self-play roster ungated, speed ceilings for ego2s rosters, leaderboard rows before/after not comparable);
   rollback instructions (`SNAKE_SERVE_CHECKPOINT=frp3-s12` or `=champion`, restart); the monitoring plan: first week or
   100,000 Watch hero decisions, roll back on any fallback reason/exception in the ego2s path, Watch hero deaths/1k
   above 0.848 (2x the 0.424 baseline), or p95 frame time >50% over the qualification measurement; owner check-in at
   7 days. Add a STATUS.md entry and an INDEX row.
4. A small monitoring script `scripts/monitor_b8_release.py` (or under research/) that reads the server's logs/telemetry
   (whatever the server already records; add minimal counters if needed: served checkpoint name, fallback reason,
   exceptions, Watch hero decisions and deaths, frame time p95) and prints the three trigger metrics vs thresholds.
Tests; flake8; frontend tests if touched. Never touch tmux sessions or processes you did not start; stop any server you
started. Git: branch t12-20261017; commit often; after each commit `git fetch origin && git merge origin/main`,
`git push origin HEAD:main`, `git push origin t12-20261017`.
Final message: the default change, the verified stderr lines (all three configurations), release record path, monitoring
command, sha.
