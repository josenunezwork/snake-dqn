# Saved H64 finite-return comparison

Branch: `NO_STRONG_FINITE_RETURN_DISCORDANCE`. Policy minus teacher; no bootstrap.

| Bank | Seed | Arm | Food Δ | Mean G64 Δ | Wins | Losses | Ties | Gate |
|---|---:|---|---:|---:|---:|---:|---:|---|
| h64_known_adaptive | 2026099201 | source | -17 | -1.671794 | 16 | 32 | 0 | no |
| h64_known_adaptive | 2026099201 | control | -14 | -0.584407 | 22 | 26 | 0 | no |
| h64_known_adaptive | 2026099201 | epsilon01 | -6 | 0.119676 | 27 | 21 | 0 | no |
| h64_known_adaptive | 2026099201 | mixture | -15 | -0.881415 | 19 | 28 | 1 | no |
| h64_known_adaptive | 2026099201 | uniform | -24 | -1.378207 | 15 | 33 | 0 | no |
| h64_known_adaptive | 2026099202 | source | -31 | -2.402877 | 10 | 38 | 0 | no |
| h64_known_adaptive | 2026099202 | control | -10 | -0.808323 | 17 | 31 | 0 | no |
| h64_known_adaptive | 2026099202 | epsilon01 | -1 | -0.040425 | 24 | 24 | 0 | no |
| h64_known_adaptive | 2026099202 | mixture | -12 | -0.874263 | 17 | 31 | 0 | no |
| h64_known_adaptive | 2026099202 | uniform | -11 | -1.243310 | 12 | 36 | 0 | no |
| h64_known_adaptive | 2026099203 | source | -18 | -2.607306 | 5 | 41 | 2 | no |
| h64_known_adaptive | 2026099203 | control | -13 | -1.003690 | 18 | 30 | 0 | no |
| h64_known_adaptive | 2026099203 | epsilon01 | -4 | -0.393011 | 17 | 30 | 1 | no |
| h64_known_adaptive | 2026099203 | mixture | -13 | -0.507114 | 21 | 27 | 0 | no |
| h64_known_adaptive | 2026099203 | uniform | -10 | -0.286387 | 24 | 24 | 0 | no |
| fresh | 2026099201 | source | -18 | -2.104163 | 12 | 36 | 0 | no |
| fresh | 2026099201 | control | -16 | -0.917337 | 14 | 34 | 0 | no |
| fresh | 2026099201 | epsilon01 | -7 | -0.368832 | 23 | 25 | 0 | no |
| fresh | 2026099201 | mixture | -16 | -1.076058 | 16 | 31 | 1 | no |
| fresh | 2026099201 | uniform | -24 | -1.345804 | 11 | 37 | 0 | no |
| fresh | 2026099202 | source | -28 | -3.017125 | 10 | 38 | 0 | no |
| fresh | 2026099202 | control | -15 | -1.105796 | 19 | 29 | 0 | no |
| fresh | 2026099202 | epsilon01 | -5 | -0.318111 | 27 | 21 | 0 | no |
| fresh | 2026099202 | mixture | -8 | -0.323167 | 25 | 22 | 1 | no |
| fresh | 2026099202 | uniform | -17 | -1.471845 | 13 | 35 | 0 | no |
| fresh | 2026099203 | source | -20 | -2.808068 | 4 | 40 | 4 | no |
| fresh | 2026099203 | control | -9 | -0.640083 | 20 | 28 | 0 | no |
| fresh | 2026099203 | epsilon01 | -10 | -0.573510 | 20 | 27 | 1 | no |
| fresh | 2026099203 | mixture | -20 | -1.066876 | 14 | 34 | 0 | no |
| fresh | 2026099203 | uniform | -14 | -0.739112 | 17 | 29 | 2 | no |

The three-reward post-first-food window is descriptive only. Finite policy-conditioned returns do not establish an infinite-horizon TD objective or a causal reward defect.
