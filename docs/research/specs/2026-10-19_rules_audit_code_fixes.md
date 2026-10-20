# Rules-audit code fixes — Sol spec (owner approved "yes all", 2026-10-19 ~23:15 PT)

Source: docs/research/advice/2026-10-19_rules_audit_fable_judgment.md (findings 2, 5, 9). Advice-level detail
in docs/research/advice/2026-10-19_rules_audit_astra_{ops,eng}.md.

1. **Qualified Colab transport (finding 2).** Live CLI acceptance PASSED on commit `ca249106` (branch
   origin/t31-colab-safety-20261018, profile `final-only`, receipt
   snake-dqn-artifacts/s1-mature-20261018/**/acceptance-result.json — find it). main lacks
   `research/colab_cli/admission.py` and differs in ~7 colab_cli files. Review every differing file
   (`git diff origin/main origin/t31-colab-safety-20261018 -- research/colab_cli docs/colab-cli.md tests/`),
   then bring main to the qualified transport with a reviewed merge (NOT a blind wholesale merge: list each
   file and why it is safe). Write `research/colab_cli/QUALIFICATION.md`: qualified commit, profile, limits,
   receipt path + sha256, and the rule "T34 manifests must run a colab_cli whose files hash-match this
   record; any colab_cli change requires re-running only the final-only acceptance (~1.5 u, operator runs
   it)". Add a check in the T34 manifest builders (research/t34_exit_b8/make_manifest.py and
   research/t34_exit_b8_fit equivalents) that refuses if colab_cli file hashes differ from QUALIFICATION.md.
   Update docs/colab-cli.md line ~8 ("not live-qualified") and scope "full-state resume" in
   research/colab_cli/ACCEPTANCE.md to jobs that claim exact continuation (final-only chunk jobs need verified
   outputs + priced lost-work recovery instead).
   CAUTION: S1 is live from the t31 worktree; do not modify that worktree or branch.
2. **Accounting (finding 5).** `research/colab_cli/accounting.py` (~L302-317) raises when provider assignment
   count/rate disagree; `supervisor.py` (~L1316) calls it in the poll loop unhandled, so a neighbour lane
   stopping between two provider reads kills a healthy lane. Change: when the mismatch is fully explained by
   a broker-known priced assignment that is starting/stopping, log WARNING for at most 2 consecutive polls,
   then fail closed exactly as today. Unknown endpoints, caps and rate-envelope violations still refuse
   immediately. Offline tests with the fake Colab fixture: delayed departure, delayed arrival, unknown
   endpoint, persistent mismatch (fails on poll 3). This is a colab_cli change -> note in QUALIFICATION.md that
   final-only acceptance must be re-run before T34 dispatch.
3. **Mac capacity (finding 9).** `research/perf_sim_20261007/with_slots.py` (~L39-52): move the AC/lid check
   to AFTER slot acquisition (and keep the existing one before). `Makefile` test target: default `-n 2` unless
   `PYTEST_WORKERS` is set.
Rules: Mac only; no live Colab/pods/tmux; do not touch S1, Phase 0 or other T34 worktrees; <=2 CPU threads;
commit/push often; merge to main; run the non-slow suite (-n 2); README-style summary in your final message.
