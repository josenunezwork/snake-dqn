# Saved H64 misses point toward second-food experience

The saved-record census supports designing a targeted second-food curriculum.
Seven of the ten H64 growth misses ended with a repeating four-step input/action
pattern, two had stopped making strict distance progress, and one was still
approaching the pellet near the horizon. Every missed case collected its first
food on frame 2 and never collected a second. These failures therefore precede
boost availability.

This is new analysis of completed gameplay, with **zero new games, model
forwards, or optimizer updates**. It does not change the
[H64 two-of-three result](../apex_growth_h64_2026-09-22/README.md) or establish the
cause of the policy errors.

| Seed | Joint successes | Recurrent suffix | Stalled | Still progressing | Frame-64 second food | Other failure classes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026099201 | 45 | 3 | 0 | 0 | 0 | 0 |
| 2026099202 | 42 | 4 | 1 | 1 | 0 | 0 |
| 2026099203 | 47 | 0 | 1 | 0 | 0 | 0 |

Each row accounts for all 48 saved final-policy games. Death-before-joint,
no-first-food, geometry-blocked boost exposure, other no-second-food cases, and
unclassified exposure cases were all zero in every seed. Every recurrent suffix
matched both the exact observation/action token and the recorded world token;
there were no input-only or world-only recurrences among the ten misses.

## Every missed case

| Seed suffix | Case | Classification | Recurrent onset / period / repeats | Post-first Manhattan distance | Final target distance |
| --- | --- | --- | --- | ---: | ---: |
| 9201 | h1-b0-p2 | Cycle | 9 / 4 / 14 | 270 | 270 |
| 9201 | h1-b1-p0 | Cycle | 17 / 4 / 12 | 180 | 280 |
| 9201 | h3-b2-p0 | Cycle | 5 / 4 / 15 | 230 | 230 |
| 9202 | h0-b0-p1 | Cycle | 13 / 4 / 13 | 250 | 190 |
| 9202 | h0-b0-p3 | Stall | — | 390 | 270 |
| 9202 | h0-b1-p1 | Progress | — | 350 | 30 |
| 9202 | h0-b2-p0 | Cycle | 17 / 4 / 12 | 260 | 180 |
| 9202 | h1-b2-p2 | Cycle | 13 / 4 / 13 | 280 | 240 |
| 9202 | h3-b1-p2 | Cycle | 13 / 4 / 13 | 270 | 210 |
| 9203 | h2-b0-p0 | Stall | — | 290 | 70 |

Nine misses began their second-food pursuit more than 200 pixels away; the other
was 180 pixels away. Food was ahead in six and behind in four. These are
conditional descriptions of the misses, not comparisons proving that distance,
heading, or body length caused them. The original H8 drill used much shorter,
axis-aligned initial food placements. A broader experience distribution is a
reasonable next experimental treatment, with a matched continuation control.

## Frozen analysis and decision

The input population was exactly the 144 final H64 gameplay records. No initial
checkpoint games or model weights were loaded. The existing joint flags were
used to identify the ten misses; the completed H64 behavioral gate was not run
again.

For recurrence, the analyzer examined the final suffix after the first food and
before the second food or episode end. It tested periods 1–8 and required at
least three identical blocks. The input token combines the exact float32 state,
six-bit selection mask, and action. The world token includes head, current food,
direction, logical length, boost phase, exact safe mask, and action. A shared
period gives `BOTH`. This descriptor does not contain the complete body and RNG
state, so it is evidence of observed recurrence rather than proof of an infinite
physical loop.

Progress required a new strict Manhattan-distance minimum in the final eight
frames and a final distance below the post-first spawn distance. Stall required
no recurrence and no new strict minimum in the final sixteen frames. The full
mutually exclusive classification order, including zero-count alternatives, is
preserved in the frozen intent and output JSON.

A longer-horizon diagnostic required at least 75% censoring/progress failures
in **every** failing lineage, with no `BOTH` or `INPUT_ONLY` recurrence. That rule
failed. Curriculum design required cycle/stall cases to make up at least half
of failures in at least two lineages. All three lineages met that rule.

The selected branch authorizes a separately frozen design, not training on the
failed cases. The H64 bank is now adaptive development evidence. Any future
training must use separate cases; fresh confirmation must be kept out of
optimization. A matched-dose comparison should separate broader collection
experience from simply adding updates. The failed imitation-to-TD transfer
recipe remains closed, and Apex remains the operational incumbent until the
shared tournament gate supports replacement.

The [H64 report](../apex_growth_h64_2026-09-22/README.md) contains the behavioral
comparison, representative native paths, and the reused H8 learning curve. This
census introduces no new learning curve because no learning occurred.

## Execution and identities

The analysis was frozen before execution with a 90-second total allowance:
30 seconds for analysis, 30 for audit, and 30 for recovery, plus a two-minute
handoff reserve. It ran once under the shared Mac supervisor with two CPU
threads, 8-GiB RSS and 24-GiB MPS limits, and the 9.6-GiB available-memory reserve.
The child exited naturally after 0.210 seconds with no source drift. Sampling
was too sparse to treat the reported child RSS as a reliable peak estimate.

Sol reviewed the classifier and decision rules before the first execution. The
saved-record checks verified the 144 unique seed/case rows, all ten prior joint
misses, a complete mutually exclusive partition, and frozen input hashes.

Artifacts reside in
`snake-dqn-artifacts/ongoing-research-20260913/apex-growth-failure-census/`:

- Intent: `63be2c140c38fb490bc618f50b8557631d18f664857dfc1fb236b8019bdd4aae`.
- Report: `50d8df4850faf4dff288f3128209bc80ab07202a93631e40f85c15d1d4526fe7`.
- Closeout: `e2682fcc5a7abbd4a4efc2d52949ec04a36716f6e2b1d3cf2ab51aede7c50059`.

The source baseline was `03e0b69dd0e9503036878c94b871da5fa3a8acb6`.
