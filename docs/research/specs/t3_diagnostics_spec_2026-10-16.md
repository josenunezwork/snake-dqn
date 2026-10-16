# Spec: T3 label-arm diagnostics (CPU only, 2026-10-16)

Read AGENTS.md and `docs/research/advice/2026-10-15_t3_interim_next.md` section 3. Development
diagnostics; they do not change the T3 decision.
Fact found by the planner: the bank margin is `margin.json` 0.0632 x `--label-bank-margin-scale`
1.0 = 0.063 Q units (T1 median fatal-vs-survivor gap 0.45), so "margin overshoot" is ruled out;
record this in the doc.

1. Separation on the bank: on a fixed seeded sample of 2,000 bank rows (decisive and counter in
   their bank proportion), compute Q(best surviving) - Q(B8's action) and argmax-in-surviving-set
   under each CTRL final (M6 CTRL_s10..14) and each finished T3 final (all three arms). Report
   per arm the distribution (quartiles) and agreement.
2. Mass decomposition (from the round-62 probe rows in `redesign-t3-probes-20261015/rows`, and
   replay if needed): per arm vs CTRL, paired by seed, scripted mix: death count and death frame
   distribution; on worlds where both arm and CTRL survive to 5000, mean hero length at frames
   2000/3000/4000/5000 (replay those worlds deterministically if the probe rows lack the series).
   Split the mass delta into "earlier/more deaths" vs "smaller while alive".
3. Where the arms disagree: replay CTRL_s{k}'s round-62 scripted trajectories and at every hero
   decision evaluate the matching LABELS_s{k} and LABELS_KEEPV8_s{k} finals' greedy action on the
   same observation; bucket disagreement rate by length (<400, 400-700, 700-1000, >=1000) and by
   CTRL's -max Q decile; note boost vs non-boost disagreements.
CPU only, at most 4 single-thread processes, never MPS; never touch tmux t3-lane*/t3-probe/
git-sync or processes you did not start. Wait for the last T3 runs to finish before including
seed 14 (status.txt). Doc `docs/research/t3_diagnostics_2026-10-16.md` with tables and a short
reading. Git: branch t4prep-20261015; commit often; after each commit `git fetch origin && git
merge origin/main`, `git push origin HEAD:main`, `git push origin t4prep-20261015`.
Final message: the three results in brief, commit sha.
