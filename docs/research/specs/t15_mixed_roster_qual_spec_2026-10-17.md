# Spec: T15 mixed Watch roster, implementation + qualification (2026-10-17)

The T13b rule (`docs/research/t13b_six_snake_roster_2026-10-17_results.md`, rule committed before data) selected the
mixed roster (unwrapped b8-s3 hero, frp3-s12 + v8 opponents) as the planned Watch default, pending this qualification.
Binding: `docs/research/advice/2026-10-17_after_t13.md` step 2 (identity; veto scope limited to opponents and reported
as such; fallbacks; latency with the wrapped vector61 opponents; release erratum stating opponents carry a veto and the
hero does not). Implementer: Codex. Read AGENTS.md, the T12 qualification protocol/harness/report (reuse its
structure; write a new sibling protocol, pinned before any non-smoke episode), `web/backend/served_checkpoint.py`,
`safety_veto_serving.py`, `session.py`, and the T13/T13b mixed-roster harness hook.
1. Code, default-off: a served Watch roster mode "mixed" (hero = released default b8-s3, unwrapped; opponents =
   frp3-s12 with the v8 veto) selectable by env (e.g. SNAKE_SERVE_WATCH_ROSTER=mixed); a veto scope "watch_opponents"
   reported in the stderr line (active=True scope=watch_opponents wrapped_ids=[1..n-1], hero not wrapped); fail-closed
   behaviour if frp3-s12's pin fails (fall back to all-B8 with a reason). Play unchanged. Tests incl. a matrix.
2. Qualification (sibling protocol, pinned first): identity of hero and opponent observations vs the eval engine for
   the mixed roster (like T12 check 1), veto scope check (hero never wrapped, every opponent wrapped), fallbacks,
   latency at 6/8/12 snakes in Watch with the mixed roster (update p95 < 66.667 ms; zero overruns at the default
   12 fps and at the existing ego2s speed ceilings; adjust ceilings for the mixed roster if needed, server-enforced),
   25 Watch episodes no errors, UI shows the opponents' veto counters and none for the hero.
3. Report `docs/research/t15_mixed_roster_qualification_2026-10-17.md` with PASS/FAIL per check, the one-line default
   change, expected stderr, and a draft release erratum. Do NOT change any serving default.
Never touch tmux sessions or processes you did not start. Git: branch t15-20261017; commit often; after each commit
`git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t15-20261017`.
Final message: check table, latency, the proposed default change, sha.
