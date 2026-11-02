# Apex width/breadth results by seed

Audit status: tail-v3 **PASS**. These are the saved report metrics; the table does not reaggregate
raw games.

## New shared H64 HOLD

`food / joint / survival`; feasibility requires food ≥274, joint ≥92, first-food 96/96, and survival
96/96. Every row had first-food 96/96 and survival 96/96; the single failure is the 91/96 joint
cell shown below.

| Seed | Cell | Food | Joint | Survival | Fresh feasibility |
|---|---|---:|---:|---:|---|
| 2026101001 | w512_n96 | 292 | 95 | 96 | Pass |
| 2026101001 | w512_n384 | 289 | 92 | 96 | Pass |
| 2026101001 | w1024_n96 | 290 | 94 | 96 | Pass |
| 2026101001 | w1024_n384 | 292 | 91 | 96 | Fail |
| 2026101002 | w512_n96 | 301 | 96 | 96 | Pass |
| 2026101002 | w512_n384 | 296 | 95 | 96 | Pass |
| 2026101002 | w1024_n96 | 296 | 94 | 96 | Pass |
| 2026101002 | w1024_n384 | 295 | 96 | 96 | Pass |
| 2026101003 | w512_n96 | 293 | 94 | 96 | Pass |
| 2026101003 | w512_n384 | 303 | 96 | 96 | Pass |
| 2026101003 | w1024_n96 | 297 | 93 | 96 | Pass |
| 2026101003 | w1024_n384 | 302 | 96 | 96 | Pass |

## Legacy retention

Short-bank entries are `first-food / survival` out of 96; each requires 96/96 on H8 TRAIN, H8
held, and H16. H64 entries are `food / joint`; every H64 cell had first-food and survival 48/48.
H64 known requires food ≥152 and joint 48; H64 fresh requires food ≥154 and joint 48. ✓ means the
whole bank gate passed.

| Seed | Cell | H8 TRAIN | H8 held | H16 | H64 known | H64 fresh |
|---|---|---:|---:|---:|---:|---:|
| 2026101001 | w512_n96 | 96/96 | 96/96 | 96/96 | 145/47 | 151/48 |
| 2026101001 | w512_n384 | 96/96 | 96/96 | 96/96 | 147/47 | 146/47 |
| 2026101001 | w1024_n96 | 96/96 | 95/96 | 95/96 | 142/45 | 149/48 |
| 2026101001 | w1024_n384 | 96/96 | 95/96 | 95/96 | 147/45 | 150/47 |
| 2026101002 | w512_n96 | 96/96 | 96/96 | 95/96 | 153/48 ✓ | 155/48 ✓ |
| 2026101002 | w512_n384 | 96/96 | 96/96 | 96/96 | 148/45 | 153/48 |
| 2026101002 | w1024_n96 | 96/96 | 96/96 | 96/96 | 148/48 | 149/47 |
| 2026101002 | w1024_n384 | 96/96 | 96/96 | 96/96 | 146/47 | 152/48 |
| 2026101003 | w512_n96 | 96/96 | 96/96 | 96/96 | 148/48 | 155/47 |
| 2026101003 | w512_n384 | 96/96 | 96/96 | 96/96 | 149/47 | 150/48 |
| 2026101003 | w1024_n96 | 96/96 | 96/96 | 96/96 | 147/47 | 143/45 |
| 2026101003 | w1024_n384 | 96/96 | 96/96 | 96/96 | 147/48 | 148/48 |

Only 2 of 24 H64 bank/cell outcomes pass their full food-and-joint gates, both in the
2026101002-w512_n96 cell. Short-bank failures are missed first-food events; all short-bank
survival counts are 96/96.
