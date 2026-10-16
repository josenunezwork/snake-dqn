# Spec: T4 step 1b, tightened two-sided arm (TWO_SIDED_T) + drift baseline (2026-10-16)

Implementer: Codex. Read AGENTS.md, `docs/research/advice/2026-10-16_t4_tighten.md` (binding design),
`docs/research/t4a_screen_result_2026-10-16.md` (TWO_SIDED = KILL: survival -0.027, mass -24/-41/-15),
`docs/research/t4a_posthoc_diagnostics_2026-10-16.md` and your T4a code (`--label-bank-counter-*`).
Planner check (2026-10-16): on round 65, TWO_SIDED vs CTRL death overlap (both 96 / CTRL-only 106 /
TWO_SIDED-only 119 of 480) equals what independence predicts (91/111/124), so world-level overlap
cannot test whack-a-mole; TWO_SIDED deaths come ~300 frames later. Record this in the doc.

Part A, drift baseline (diagnostic, CPU, now): same-init drift = B8_s{k} final vs CTRL_s{k} on
CTRL_s{k}'s own round-62/65 states at L >= 700, matched exactly as readout (c); raw rate and
"consequential" rate (flip counted only when CTRL's Q gap between its argmax and the other model's
argmax > 0.3). Recompute TWO_SIDED, LABELS, KEEPV8 (c) with the consequential rate too.

Part B, code (default-off; default path byte-identical, prove with the existing traces/tests):
1. Decisive-row weighting by distance to death: rows with `death_decision - decision < 100` weight
   1.0, 100-300 weight 0.3 (derive from stored fields; offline filter producing a derived bank v3
   with a new manifest hash-bound to bank v2, or a trainer weight flag; your choice, document it).
2. Outside-bank counter rows: from CTRL and B8 round-62 probe trajectories (replay as needed), states
   at L >= 700 at least 300 decisions from any death, matched to decisive rows on length bucket and
   -max Q decile, count 1:1 with decisive rows; label = B8's own action; single-action margin with
   its own scale (`--label-bank-outside-margin-scale`, the arm uses 0.0632 x 1 = 0.063, i.e. ~0.06,
   within the advisor's 0.0-0.1) and weight 0.5. Store them as a separate bank part (own manifest).
   In-window counters unchanged (1:1, margin 0.30, weight 1.0).
3. Telemetry: agree per row family (decisive retained, in-window counter, outside counter).
4. Tests; CPU smoke with the real banks.

Part C, pre-registration `docs/research/specs/t4c_prereg_2026-10-16.md` (write before any run):
arm TWO_SIDED_T = TWO_SIDED recipe + Part B data; 5 seeds 10-14 from B8 finals, 8M; CTRL = M6 CTRL
finals; fresh round 66 (seed audit), 96/32/32; same GO/KILL/EXTEND rule. Readouts (advisor's):
(c) consequential-change rate within 10 points of the drift baseline; shared-survivor length delta
at 5000 >= -50; decisive agree >= 0.85 on retained rows; counter agree >= 0.95 on both counter
sets. The declared fork after it, verbatim from the advisor note section 3.
Launch scripts (do not start): `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4c-mac-20261016/lane{0,1}.sh`
running from THIS worktree.

CPU only, at most 4 single-thread processes, never MPS; never touch tmux sessions or processes you
did not start (t4b-build is running). Git: branch t4c-20261016; commit often; after each commit
`git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t4c-20261016`.
Final message: drift baseline numbers (raw + consequential) vs the three arms, Part B summary and
tests, prereg path, launch script paths, sha.
