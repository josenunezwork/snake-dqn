# Spec: T3 screen probes + analysis (2026-10-15)

Implementer: Codex. Read AGENTS.md, then the pre-registration
`docs/research/specs/t3_screen_prereg_2026-10-15.md` (incl. Amendment 1) - it is binding - the
advisor review `docs/research/advice/2026-10-15_t3_launch_review.md`, and your T2 tooling
`research/t2_screen_20261015/` (probe_t2.py, watch_t2*.py, analyze_t2*.py) which you should reuse.

## Runs (launched by Claude; never touch tmux t3-lane0 / t3-lane1)
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t3-mac-20261015/`: LABELS_s{10..14},
LABELS_KEEPV8_s{10..14}, LABELS_KEEPV8_LO_s{10..14}; `status.txt` gets `NAME exit=RC`.
CTRL = `redesign-m6-mac-20261014/CTRL_s{10..14}/final.pth` (reused, not retrained).

## Task
1. Package `research/t3_screen_20261015/` (import/extend the T2 code, do not fork it blindly).
2. Bank audit for round_index 62 (and 63 for any extension): prove unused (seed audit like T2's).
3. Probes on round 62: 96 scripted / 32 frozen / 32 mixed per checkpoint, the T2 probe path. Probe
   the 5 CTRL finals and the incumbent frp3-s12 + v8 first, then each T3 final as it lands. Watcher
   in tmux `t3-probe`; at most 2 probe processes while t3 lanes are alive, up to 4 after.
4. Analysis per arm vs CTRL exactly as the pre-registration: primary, GO/KILL/EXTEND (with the
   automatic one-time extension at round 63 for arm + CTRL), attribution note across the three arms.
   Secondaries: alive-frame survival, mass vs incumbent per mix, death lengths, per-100k
   `greedy_boost_rate` curve and final value, `bank_agree` at 2M and 8M (from each run's log.jsonl).
5. Post-hoc re-roll (advisor risk 3): for each T3 final, take 200 decisive bank states (fixed sample,
   same for all finals, seeded), re-run the rollouts with that final as the continuation policy using
   `research/t3_rollout_labels_20261015/` engine, report the fraction of labelled surviving actions
   that still survive 300 frames and the fraction where the final's argmax is in the surviving set.
   Run after the probes; up to 4 workers once training lanes are done.
6. Results doc `docs/research/t3_screen_result_2026-10-15.md` (header: development screen, not gate
   evidence; disclosure: labels from simulator rollouts of B8, served artifact stays a bare network).
7. Tests for new analysis code; flake8 clean.

## Git (binding)
Work on branch `t3p-20261015` in this worktree. Commit often. After each commit:
`git fetch origin && git merge origin/main` (resolve only conflicts in your own files; otherwise
stop and report), then `git push origin HEAD:main` and `git push origin t3p-20261015`.
Never touch other branches.

Training takes ~4 h; keep polling (every ~10 min) until the results doc is pushed.
Final message: per-arm decision with primary deltas and SEs, mass deltas per mix, boost-rate and
bank_agree summaries, re-roll fractions, commit sha.
