# Spec: close the campaign properly (docs + CI), 2026-10-17

Read AGENTS.md and `docs/research/advice/2026-10-17_what_next.md` item 1 (binding). Implementer: Codex. No compute.
1. Retrospective `docs/research/redesign_campaign_retrospective_2026-10-17.md`: the goal ("beat frp3-s12 + v8 without a
   veto"), the result (B8 seed 3 released 2026-10-17: Phase R STRICT_PASS Tier 1 + Tier 2, T12 qualification), the
   closed late-death arms (M6, Huber T2/T2b, T3, T4 TWO_SIDED / TWO_SIDED_T, RETURN / RETURN_C, T5, T6, T11 LOCAL30) each
   with its one-line result and doc link, and the T8-T10 mechanism (self-built pocket over 50-100 decisions; no early
   separator; horizon AUC ~0.92 at 30 -> ~0.75 at 300 for every input; no certificate). Cite INDEX rows; no new claims.
2. STATUS.md: a standing decision "late-death arms for bare networks are closed; reopen only with new evidence of a
   different mechanism or a new observation/architecture", plus the release as the current served state.
3. Fix stale docs: README.md (deployed reference) and docs/README.md (released agent) to b8-s3 with the claim wording
   from the release record and the rollback.
4. Fix CI (`.github/workflows`): make `lint + tests` pass: black formatting in `src/core/runtime_contract.py` (black 26),
   align `requirements-dev.txt` pytest with the `pytest.ini` keys (pytest 9), and any other failure the workflow shows
   when run locally the same way (run the workflow's exact commands). Do not weaken lint rules or skip tests; mark tests
   that need Mac-only artifacts as skipped when the artifacts are absent only if they already do so on the Mac.
   Verify by running the workflow's commands locally; after pushing, check the GitHub Actions run with `gh run list`
   / `gh run view` and fix until green.
Never touch tmux sessions or processes you did not start. Git: branch t14-20261017; commit often; after each commit
`git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t14-20261017`.
Final message: files changed, CI status with the run URL, sha.
