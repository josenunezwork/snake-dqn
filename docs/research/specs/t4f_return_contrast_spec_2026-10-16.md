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

## Planner addendum 1 (binding; from `docs/research/advice/2026-10-16_return_c_review.md`)
Supersedes the corresponding parts above:
1. Anchor: NOT a_B8. Use a_ref = argmax_a y(a) over the state's recorded branches (every legal action
   was rolled out), with stopgrad on Q(s, a_ref): loss on (Q(s,a) - stopgrad Q(s,a_ref)) - (y(a) - y(a_ref))
   for every recorded a != a_ref. Do not use the student's current greedy action.
2. Weighting: drop w_a = min(1, |dy|/10). Use an epsilon-insensitive Huber: zero loss while the
   contrast error |e| <= 2 Q, Huber (delta 100) on |e| - 2 beyond; uniform weight 1.0 on every row
   (decisive and outside). Flag `--return-contrast-epsilon` (arm: 2.0).
3. Tripwires (pre-registered, halt = KILL for the arm): greedy boost rate at L < 700 above 1% at any
   100k row; overall greedy boost rate above 2%. Replace the single 5% tripwire; log the boost rate
   per length bucket (<400, 400-700, 700-1000, >=1000) every 100k.
4. Early warning (pre-registered, binding stop): at 0.5M and 1M evaluate on the held-out decisive set
   the median Q(fatal) - Q(best surviving) and on held-out outside states the gap change; save
   checkpoints at 0.5M and 1M for this (`--checkpoint-every` or explicit). Stop the arm if, at 1M,
   the held-out decisive gap has not moved toward -3 (still >= 0) while the training-row contrast
   error has fallen, or the outside gap moved by more than 0.5 Q, or the L < 700 boost rate > 1%.
   Implement the 0.5M/1M evaluation as a small script the planner can run on those checkpoints.
5. Recalibrate W after these changes (far fewer active rows); report the active-row fraction.

## Planner addendum 2 (binding, before any RETURN_C run): fix the boost macro geometry
The macro repeated the same RELATIVE action for k=3 frames, so a turning boost became three turns in
a row (a tight U-turn into the own body). Result: boost-minus-normal gaps moved from -0.16 to -61 Q on
outside states and from +32 to -27 Q on decisive states: a geometry artifact, not boost cost, which
would teach "never boost" and erase the real escape signal. Fix: frame 1 = the branch's boost action;
frames 2..k = B8's greedy DIRECTION at that frame with boost forced on (the boost variant of B8's
relative action, when legal; if boost is illegal, B8's normal action), then B8 greedy for the rest.
Re-roll all boost branches with this macro into a new derived bank (non-boost rows unchanged,
byte-identical check), report the new gap table (expect outside gaps near the boost cost, about
-0.1 to -1 Q per burned segment, not tens), recalibrate W, update the prereg (section "Addendum 2
fix") and the launch scripts; keep everything else. Add a unit test that a boost-left macro does not
turn three times. Do not start training.
