# Addendum (Claude, 2026-10-20 ~01:10 PT) — Amendment 2 changes the teacher label rule

DESIGN_V2_SHADOW.md Amendment 2 is on main (ca3fc731); merge main before finishing. Changes to your spec:
- DROP Amendment 1's stride-10 + edge rule. Search rule = bisection (radius 3) from the first detection decision.
- Labels: export the FULL ranked per-action outcome (T3 rule: bare if it survives; else horizon survivors; if all
  die, longest alive; ties B8 Q -> bare -> index) with horizon/censoring, alive frames, mass. Decisive rows =
  horizon rescues only; all-die "delay" choices tagged, weight ~0; shadow confirms = KL-anchor rows only.
  (Teacher v2 currently picks bare whenever no alternative survives: keep that as the executed action but export
  the ranked outcomes and the delay tag.)
- Restore M2's savings: reuse prepared snapshots and the free confirmed greedy control; do not reconstruct source
  prefixes or reroll greedy controls per query.
- Opponents with veto_slots must be dispatched exactly (do not inherit T33's rejection).
If you have already finished, I will relaunch with this addendum.
