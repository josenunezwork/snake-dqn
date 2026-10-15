# Guard-clause amendment (owner decision 2026-10-15, before any new gate data)

Owner: "both" (options B and C of the panel memo section 5.1), as two tiers. Recommended by the
panel and the cloud advisor (`docs/research/advice/2026-10-15_guard_clause.md`, whose clause text
and power figures this adopts). Applies to the next fresh-bank Phase R of any redesign candidate.
This is the one survival-guard amendment for the redesign line; no second amendment after a second
failure.

## Tier 1, release gate (binding): option B
Replaces clause c6 `guard_scripted_survival` (champion_a5 + v8 comparator). All on fresh banks under
a new round index with a seed audit; hero = candidate, no veto.
- c2 scripted survival: alive-frame surv5, hero minus frp3-s12 + v8 (served agent), pooled over the
  five seeds' banks, 96 scripted worlds each, >= -0.03 (SE ~0.017).
- c2 frozen / mixed survival: same metric and threshold, 64 worlds per bank (320 pooled).
- Per-seed floor: on each seed's bank, alive-frame surv5 hero minus served agent >= -0.15 (96 worlds).
- Absolute floor: hero episode survival >= 0.50 on every mix, pooled.
- All other M4 clauses (efficacy mass vs incumbent, mi10, guard mass point) unchanged.
- Champion_a5 + v8 is measured on all five banks with both metrics: diagnostic only.

## Tier 2, product bar (pre-registered, reported, not blocking release): option C
- Pooled scripted episode survival, hero minus champion_a5 + v8, 480 worlds (5 banks x 96), >= -0.05
  (SE ~0.032). Reported as PASS/FAIL in every result doc. The campaign's "done" criterion: a
  candidate that passes Tier 1 AND Tier 2.

## Discipline
Amend once, prospectively; fresh banks; never re-score the M4 look-2 data or the census bank as gate
evidence; publish the sensitivity table (B8 seed-0 look 2: -0.156 alive-frame, -0.188 episode;
census: -0.039 / -0.157; pooled vs served agent: -0.003 / -0.050); disclose the M4 PARTIAL and its
role in motivating this amendment; secondary: length-matched hazard, deaths per 1,000 decisions at
L >= 800.
