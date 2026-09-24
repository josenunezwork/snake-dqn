# Native pretrap counterfactual local probe

Branch: **ALL14_HAVE_TWO_FRAME_LOCAL_RESCUE**

| Seed | Arm | Root | Qualified | Alternatives | Two-frame rescues |
|---|---|---|---|---:|---:|
| 2026101001 | h64 | h0-b2-d4-x100-y100 | QUALIFIED | 1 | 1 |
| 2026101001 | h64 | h1-b2-d6-x300-y100 | QUALIFIED | 1 | 1 |
| 2026101001 | h256 | h0-b2-d6-x300-y100 | QUALIFIED | 1 | 1 |
| 2026101001 | h256 | h1-b2-d4-x100-y300 | QUALIFIED | 0 | 0 |
| 2026101001 | h256 | h2-b1-d6-x300-y300 | QUALIFIED | 1 | 1 |
| 2026101001 | h256 | h3-b2-d4-x300-y100 | QUALIFIED | 1 | 1 |
| 2026101002 | h64 | h1-b1-d6-x100-y100 | QUALIFIED | 2 | 2 |
| 2026101002 | h256 | h0-b0-d6-x100-y300 | QUALIFIED | 2 | 2 |
| 2026101002 | h256 | h0-b1-d4-x100-y100 | QUALIFIED | 2 | 2 |
| 2026101002 | h256 | h1-b2-d6-x300-y300 | QUALIFIED | 1 | 1 |
| 2026101003 | h64 | h1-b1-d4-x100-y300 | QUALIFIED | 2 | 2 |
| 2026101003 | h64 | h2-b0-d6-x300-y100 | QUALIFIED | 1 | 1 |
| 2026101003 | h256 | h0-b1-d4-x100-y300 | QUALIFIED | 2 | 2 |
| 2026101003 | h256 | h2-b2-d6-x300-y100 | QUALIFIED | 2 | 2 |
| 2026101003 | h256 | h3-b0-d6-x100-y100 | QUALIFIED | 1 | 1 |

## Cells

| Cell | Targets | Qualified | Alternatives | Cases rescued |
|---|---:|---:|---:|---:|
| 2026101001-h64 | 2 | 2 | 2 | 2 |
| 2026101001-h256 | 4 | 4 | 3 | 3 |
| 2026101002-h64 | 1 | 1 | 2 | 1 |
| 2026101002-h256 | 3 | 3 | 5 | 3 |
| 2026101003-h64 | 2 | 2 | 3 | 2 |
| 2026101003-h256 | 3 | 3 | 5 | 3 |

## Seeds

| Seed | Targets | Qualified | Alternatives | Cases rescued |
|---|---:|---:|---:|---:|
| 2026101001 | 6 | 6 | 5 | 5 |
| 2026101002 | 4 | 4 | 7 | 4 |
| 2026101003 | 5 | 5 | 8 | 5 |

Selected failures only; a rescue is the frozen alternative action followed by the lowest-index native-safe normal action.
