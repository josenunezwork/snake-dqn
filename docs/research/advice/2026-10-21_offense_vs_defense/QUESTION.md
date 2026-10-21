# Offense vs defense — owner question (2026-10-21 ~01:20 PT)

## Owner's text (verbatim, pasted by the owner; it is a prior assistant answer the owner wants tested)
> Probably not the best plan for a big jump. It's a reasonable plan for closing B8's remaining weakness, but that
> weakness isn't where most of the upside in slither.io is.
>
> How mass is gained in slither.io: steady eating of ambient pellets gives slow, linear growth. Kills are the big
> jumps — when a snake dies it drops its mass as a corpse, so cutting off a big snake and eating it can double you in
> seconds; good players grow mainly by encircling, cutting in front of heads, and harvesting corpses from fights.
> Survival is the precondition, not the source of growth. Boosting costs mass; it only pays when it wins a kill or a
> corpse race.
>
> What we're doing: the T34 search teacher mainly finds rescues (moves that avoid enclosure deaths), B8's remaining
> gap (B8 death census: scripted gap is self-enclosure at length ≥ 800). So T34 is a defensive lever; even if it works
> perfectly its upside is roughly "B8, but dies less", which T3 put at about +92 mass under ideal conditions.
>
> What the project notes say we're missing: Kills — eval notes record ~0 kills and ~89% boost abuse. Game rules — the
> July redesign blueprint's "mechanics v2" (pellets left when boosting, corpse drops, kill credit) was never
> implemented; if the sim under-models corpses and kills, no training plan will learn offense. Exploitable opponents —
> scripted or frozen policies; exact-sim search can predict them perfectly, so it could look for kills (cut-offs,
> encirclements), not just escapes.
>
> What I'd consider: (1) measure first (~1 h): in our sim, what share of B8's mass comes from pellets vs corpses, and
> how many kills per game; (2) point the search teacher at offense — score branches by mass gained incl. kills and
> corpses, not just survival; (3) check whether the sim's rules match the game; if corpse/kill payoffs are
> under-modelled, fix the sim first (mechanics v2). Caveats: from project notes + general slither.io knowledge, not
> verified; changing the sim's rules changes the benchmark, so any comparison to B8 must be re-run under the new rules.
>
> Fable's direction review (docs/research/advice/2026-10-21_direction/FABLE.md, adopted 6f0249bb) never addressed
> offense — it stayed inside "beat B8 on mass at equal survival via search".

## Goal
A clearly better served snake than bare B8 s3 (current Watch hero, released 2026-10-17), as large a jump as possible,
as soon as possible, with a "beats B8" claim the owner can trust. Colab credits (~1,145 u) are meant to be used.

## Open question
Where is the biggest credible mass gain over B8 s3 in *our* simulator and in the served game — defense (the current
T34 search-teacher rescue line) or offense (kills, cut-offs, corpse harvesting) — and is the claim above right? What
would you do, in what order, starting now, and what would change your mind?

## Facts to check (verify in the repo; cite file:line)
- Sim + rules: `src/simd_env/grid_sim.py`, `batch_sim.py`, `src/` game/snake/food code (corpse drop on death? boost
  pellets? kill credit?), `docs/ml_redesign_blueprint_2026-07.md` (mechanics v2), memory notes in
  `docs/research/` on eval pitfalls (~89% boost abuse, ~0 kills — `docs/slither_training_log_20260615.md`, tournament
  eval), B8 death census, `docs/research/final_run_20261019/{FINAL_PLAN.md,SUCCESS_CONTRACT.md,CAMPAIGN_LOG.md}`.
- Current campaign state: CAMPAIGN_LOG "Decisions" (direction reset 6f0249bb), T3-dense +92 mass result.
- Search machinery: `research/t34_*` (exact-sim teacher, shadow/branch rollouts), `research/t34_pathb/`.

## Rules for advisors
Read-only on code except your own answer file (and, for the measurement persona only, a scratch script + receipt).
Never open `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01/` (sealed). No Colab/RunPod, no tmux
or process changes. The Mac is shared with registered jobs: ≤ 2 threads, `nice -n 10`.
