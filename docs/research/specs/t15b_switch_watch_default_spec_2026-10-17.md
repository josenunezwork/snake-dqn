# Spec: switch the released Watch roster default to mixed (OWNER-APPROVED 2026-10-17)

The owner approved the switch ("approved") after T15 SERVING_PASS (`docs/research/t15_mixed_roster_qualification_2026-10-17.md`)
and the T13b roster rule. Implementer: Codex. Read AGENTS.md and that report (incl. its draft erratum).
1. `web/backend/served_checkpoint.py`: `WATCH_ROSTER_RELEASED_DEFAULT = WATCH_ROSTER_MIXED`, with a comment naming the
   approval and the rollback (`SNAKE_SERVE_WATCH_ROSTER=all_b8` + restart).
2. Update every test that encodes the old default (known failures: tests/test_web_watch_roster.py default/invalid-word
   tests and the artifact-denial subprocess test, tests/test_web_served_checkpoint.py::TestLiveStack default test,
   tests/test_web_ego2s_serving.py default test, tests/test_t12_episodes.py watch smoke, tests/test_web_safety_veto.py
   default-off payload/log tests; find any others). Keep each test's original intent: where it tested all-B8 behaviour,
   pin `SNAKE_SERVE_WATCH_ROSTER=all_b8`; add explicit tests that the unset env now serves the mixed roster (b8-s3 hero
   unwrapped, frp3-s12 + v8 opponents, scope watch_opponents, hero not in wrapped_ids) and that invalid words fall back to
   the released default (mixed) with a reason. Do not weaken any assertion about the hero being unwrapped.
3. Publish the erratum: `docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md` from the T15 draft (now released,
   owner-approved), and add the mixed-roster monitoring row: Watch6 mixed baseline 0.272 hero deaths/1k, rollback trigger
   > 0.544; extend `scripts/monitor_b8_release.py` (per roster mode and size) + tests. Update STATUS.md and the INDEX.
4. Verify live: start the server briefly with no env and confirm the stderr lines (served-checkpoint b8-s3; safety-veto
   active=True scope=watch_opponents, hero not wrapped), then with SNAKE_SERVE_WATCH_ROSTER=all_b8 (veto inactive).
   Stop any server you start. Run the full Python suite and the frontend tests; flake8/black.
Git: branch t15-20261017; commit; `git fetch origin && git merge origin/main`, `git push origin HEAD:main`,
`git push origin t15-20261017`; confirm the GitHub Actions run on main is green with gh.
Final message: tests changed, stderr lines, CI run URL, sha.
