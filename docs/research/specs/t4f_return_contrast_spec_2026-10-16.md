# Spec: RETURN_C, contrast return regression (new pre-registered arm, 2026-10-16)

Implementer: Codex. Read AGENTS.md, `docs/research/advice/2026-10-16_return_halt.md` section 3 (binding
design), `docs/research/t4e_bugcheck_2026-10-16.md` (no bug; decisive boost targets -4.7 vs normal
-35.8 with 87% vs 50% survival; outside boost-normal gap -0.16 Q), the RETURN prereg + amendment
(incl. Deviation 1), and the return tooling/code.

## A. Code (default-off; default path byte-identical, prove with existing traces/tests)
1. `--return-mode {level, contrast}` (default level = today). contrast: for each recorded branch a != a_B8,
   loss = Huber_delta( (Q(s,a) - stopgrad(Q(s,a_B8))) - (y(a) - y(a_B8)) ), weight
   w_a = min(1, |y(a) - y(a_B8)| / 10); branches with w_a = 0 contribute nothing. a_B8 = the
   bank's recorded B8 action at s.
2. `--boost-halt-rate R` (default off): halt the run with a clear reason if the greedy boost rate at
   any 100k log row exceeds R (pre-registered R = 0.05).
3. Telemetry: contrast rows used per update, mean |target contrast|, grad ratio, max|Q|, greedy
   boost rate (existing).
## B. Boost macro branches (collector + bank v2 of the return bank)
For every boost action in the existing return bank states (train, outside, holdout), re-roll the
branch as k consecutive boost frames (k = the trainer's boost cost period, i.e. the frames per
burned segment; read it from the sim config, document it) then B8 for the rest of D = 300, with the
same outcome recording; replace those branches' targets in a derived bank (new manifest, hash-bound
to the source). Non-boost branches unchanged. Use up to 12 single-thread CPU workers; report wall
time and how the boost-vs-normal target gap changes (decisive and outside).
## C. Calibration + pre-registration (before any RETURN_C run)
Recalibrate W for contrast mode (grad-norm ratio ~0.5 at init, as before, short MPS smokes from B8_s0
and B8_s2). Pre-register `docs/research/specs/t4f_prereg_2026-10-16.md`: arm RETURN_C = M6 CTRL recipe
+ `--return-mode contrast` + macro bank + calibrated W + `--return-huber-delta 100` +
`--boost-halt-rate 0.05`; 5 seeds 10-14 from B8 finals, 8M, `--device mps`; CTRL = M6 CTRL finals;
fresh rounds 70 (71 extension) with the bank-world exclusion; same GO/KILL/EXTEND rule; readouts as
the RETURN amendment (held-out decisive Q-gap <= -3, train gap within 2x, outside gap +/-0.5,
consequential change within 2 points of drift) + greedy boost curve; a halted run counts as KILL for
the arm. Launch scripts `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t4f-mac-20261016/lane{0,1}.sh`
(MPS, unstarted, `--check` prints full commands).
Never touch tmux sessions or processes you did not start. Git: branch t4d-20261016; commit often;
after each commit `git fetch origin && git merge origin/main`, `git push origin HEAD:main`,
`git push origin t4d-20261016`. Final message: code/tests, macro re-roll stats + wall time, chosen W,
prereg + script paths, sha.
