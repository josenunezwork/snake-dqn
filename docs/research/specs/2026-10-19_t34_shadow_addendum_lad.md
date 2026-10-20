# Addendum (Claude, 2026-10-19 ~22:15 PT) — M2 must measure TWO LAD rules
Design v2 Amendment 1 (main 105debee, docs/research/final_run_20261019/DESIGN_V2_SHADOW.md): rescue
availability is non-monotone (T1: 47/75 deaths). In M2 replay the T3 dense bank under BOTH
(a) bisection + stride-1 refine (kept for Path B intervention) and
(b) the teacher label rule: stride-10 expansions over [t_live, t_doom], then stride-1 within +/-5 of every
    survive->die transition; LAD = latest surviving decision.
Report for each: row-steps/world, recovery of T3's interventions, LAD error vs the dense/stride-1 truth.
GO for the teacher rule = >= 90% of T3 interventions recovered.
