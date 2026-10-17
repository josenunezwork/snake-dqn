# T13b six-snake roster results — 2026-10-17

**Descriptive measurement and pre-registered roster planning decision. Serving defaults unchanged.**

## Decision rule and outcome

The [decision rule](t13b_six_snake_roster_2026-10-17.md) was committed and pushed as `f36a20d30f5c53cff3d22978cceafab435666b4b` before running:

> At six snakes, keep all-B8 if the all-B8 hero's deaths per 1,000 are within 0.10 of the mixed roster's AND its p95 length is within 100 cells of the FRP3 hero's; otherwise the mixed roster (b8-s3 hero, frp3-s12 + v8 opponents) becomes the planned Watch default, subject to a separate roster qualification (not in this job).

- Deaths/1k: all-B8 **0.392**, mixed **0.296**; difference **+0.096**, absolute gap **0.096**: within 0.10.
- p95 length: all-B8 **596**, FRP3 hero **723**; difference **-127**, absolute gap **127** cells: outside 100.

**Outcome: `PLAN_MIXED_PENDING_ROSTER_QUALIFICATION`.** The conjunction fails. The mixed roster becomes the planned Watch default, subject to separate T12-protocol roster qualification. No mixed-roster serving change is made here.

This decision uses the registered point-estimate rule. Confidence intervals remain descriptive and do not replace the rule. B8 stays served without a veto. Play is unchanged.

The paired all-B8 minus mixed hero death-rate difference is +0.096/1k
[95% CI +0.008, +0.184]. The all-B8 minus FRP3 p95-length difference is −127
cells [−299, +68.025]. The latter interval includes zero; the planning outcome
follows the fixed point-estimate rule, not a claim of a conclusive p95 difference.

## Six-snake results

25 paired H5000 episodes per roster, at the default six snakes from `configs/mechanics_v2.yaml`. Deaths and food are per 1,000 actual decisions; boost is executed boosted movement as a percentage of decisions. Lengths are pooled pre-decision logical lengths. Kills and causes are totals. Mass is post-step live length over the full horizon with dead frames zero; arena mass is the sum across snakes.

| Roster | Scope | Decisions | Deaths / 1k | Length median / p95 / max | H5000 mass | Kills made / taken | Deaths head-on / body / wall / self | Boost | Food / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All B8 | hero | 125,000 | 0.392 | 189 / 596 / 1078 | 236.91 | 20 / 13 | 5 / 9 / 1 / 34 | 0.15% | 166.90 |
| All B8 | arena | 750,000 | 0.404 | 178 / 578 / 1078 | 1324.33 | 92 / 92 | 22 / 74 / 2 / 205 | 0.18% | 164.34 |
| All FRP3 (hero + v8) | hero | 125,000 | 0.184 | 249 / 723 / 1014 | 293.73 | 132 / 8 | 2 / 6 / 0 / 15 | 2.12% | 178.11 |
| All FRP3 (hero + v8) | arena | 750,000 | 1.159 | 90 / 450 / 1014 | 824.57 | 322 / 322 | 154 / 204 / 4 / 507 | 1.34% | 170.42 |
| All champion (hero + v8) | hero | 125,000 | 0.120 | 290 / 674 / 793 | 311.96 | 89 / 5 | 1 / 5 / 0 / 9 | 9.49% | 178.25 |
| All champion (hero + v8) | arena | 750,000 | 1.015 | 92 / 440 / 793 | 812.59 | 209 / 209 | 81 / 142 / 18 / 520 | 9.52% | 180.29 |
| B8 hero / FRP3 + v8 opponents | hero | 125,000 | 0.296 | 271 / 791 / 1019 | 316.76 | 19 / 16 | 2 / 14 / 0 / 21 | 0.17% | 202.06 |
| B8 hero / FRP3 + v8 opponents | arena | 750,000 | 0.348 | 161 / 547 / 1019 | 1222.94 | 98 / 98 | 23 / 79 / 1 / 158 | 1.96% | 148.95 |

Arena mass per snake: All B8 **220.72**; All FRP3 (hero + v8) **137.43**; All champion (hero + v8) **135.43**; B8 hero / FRP3 + v8 opponents **203.82**.

Unexpected death causes (other/starvation): none observed in any row.

## All-B8 at eight snakes

| Roster | Scope | Decisions | Deaths / 1k | Length median / p95 / max | H5000 mass | Kills made / taken | Deaths head-on / body / wall / self | Boost | Food / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All B8 | hero | 125,000 | 0.496 | 145 / 555 / 1037 | 195.05 | 16 / 21 | 2 / 19 / 0 / 41 | 0.25% | 165.12 |
| All B8 | arena | 1,000,000 | 0.498 | 143 / 515 / 1037 | 1475.06 | 149 / 149 | 42 / 111 / 3 / 342 | 0.23% | 161.18 |

Arena mass per snake: **184.38**. The Watch8 hero baseline is **0.496 deaths/1k**, with monitoring trigger **>0.992**.

The same seed labels are used at six and eight, but adding snakes changes the initial world. The supplemental eight-minus-six paired-seed intervals are in the full summary; they do not claim identical initial worlds across population sizes.

## Paired six-snake differences

Each entry is left minus right, estimate [95% paired-bootstrap CI]. The same 25 seed indices are resampled jointly across rosters; 10,000 draws, NumPy RNG seed 20261017. Pooled rates and length distributions are recomputed for every draw. Percentile intervals have no multiplicity correction. Quantiles use the nearest order statistic round((N − 1) × q), ties-to-even, matching T13/T12. Maximum means observed sample maximum, not a population bound.

### Hero: survival, size, action and food

| Left − right | Deaths / 1k | Median length | p95 length | Max length | H5000 mass | Boost (percentage points) | Food / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| All B8 − All champion (hero + v8) | +0.272 [+0.184, +0.360] | -101.000 [-159.000, -51.000] | -78.000 [-175.000, +94.000] | +285.000 [-93.000, +339.000] | -75.053 [-121.681, -23.689] | -9.335 [-9.903, -8.778] | -11.344 [-19.600, -3.184] |
| All B8 − All FRP3 (hero + v8) | +0.208 [+0.104, +0.312] | -60.000 [-114.000, -8.000] | -127.000 [-299.000, +68.025] | +64.000 [-315.000, +203.000] | -56.821 [-115.566, +2.082] | -1.966 [-2.269, -1.696] | -11.208 [-20.552, -2.760] |
| All B8 − B8 hero / FRP3 + v8 opponents | +0.096 [+0.008, +0.184] | -82.000 [-132.000, -31.000] | -195.000 [-331.000, +1.025] | +59.000 [-320.000, +148.000] | -79.852 [-138.962, -20.903] | -0.022 [-0.084, +0.037] | -35.160 [-47.856, -22.912] |
| All champion (hero + v8) − B8 hero / FRP3 + v8 opponents | -0.176 [-0.264, -0.088] | +19.000 [-38.000, +81.000] | -117.000 [-214.000, +11.000] | -226.000 [-280.000, -137.000] | -4.799 [-56.835, +45.622] | +9.314 [+8.774, +9.867] | -23.816 [-35.576, -13.072] |
| All FRP3 (hero + v8) − All champion (hero + v8) | +0.064 [-0.016, +0.144] | -41.000 [-117.000, +30.000] | +49.000 [-84.000, +188.000] | +221.000 [+82.000, +275.000] | -18.232 [-79.388, +44.804] | -7.369 [-7.954, -6.773] | -0.136 [-10.168, +9.528] |
| All FRP3 (hero + v8) − B8 hero / FRP3 + v8 opponents | -0.112 [-0.208, -0.024] | -22.000 [-75.000, +32.000] | -68.000 [-199.000, +110.000] | -5.000 [-140.000, +84.000] | -23.031 [-78.128, +32.096] | +1.945 [+1.678, +2.236] | -23.952 [-37.472, -10.080] |

### Hero: kills and death causes

| Left − right | Kills made | Kills taken | Head-on deaths | Body deaths | Wall deaths | Self deaths |
|---|---:|---:|---:|---:|---:|---:|
| All B8 − All champion (hero + v8) | -69.000 [-91.000, -49.000] | +8.000 [+3.000, +14.000] | +4.000 [+0.000, +9.000] | +4.000 [-1.000, +10.000] | +1.000 [+0.000, +3.000] | +25.000 [+16.000, +34.000] |
| All B8 − All FRP3 (hero + v8) | -112.000 [-151.000, -74.000] | +5.000 [-4.000, +14.000] | +3.000 [-1.000, +7.000] | +3.000 [-5.000, +11.000] | +1.000 [+0.000, +3.000] | +19.000 [+8.000, +29.000] |
| All B8 − B8 hero / FRP3 + v8 opponents | +1.000 [-14.000, +15.000] | -3.000 [-12.000, +5.000] | +3.000 [+0.000, +6.000] | -5.000 [-12.000, +2.000] | +1.000 [+0.000, +3.000] | +13.000 [+4.000, +22.000] |
| All champion (hero + v8) − B8 hero / FRP3 + v8 opponents | +70.000 [+43.000, +96.000] | -11.000 [-20.000, -3.000] | -1.000 [-4.000, +2.000] | -9.000 [-16.000, -2.000] | +0.000 [+0.000, +0.000] | -12.000 [-19.000, -4.000] |
| All FRP3 (hero + v8) − All champion (hero + v8) | +43.000 [+4.000, +82.000] | +3.000 [-5.000, +11.000] | +1.000 [-2.000, +4.000] | +1.000 [-6.000, +8.000] | +0.000 [+0.000, +0.000] | +6.000 [-1.000, +13.000] |
| All FRP3 (hero + v8) − B8 hero / FRP3 + v8 opponents | +113.000 [+77.000, +149.000] | -8.000 [-18.000, +1.000] | +0.000 [-4.000, +4.000] | -8.000 [-17.000, +0.000] | +0.000 [+0.000, +0.000] | -6.000 [-14.000, +3.000] |

### Arena: survival, size, action and food

| Left − right | Deaths / 1k | Median length | p95 length | Max length | H5000 mass | Boost (percentage points) | Food / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| All B8 − All champion (hero + v8) | -0.611 [-0.676, -0.551] | +86.000 [+74.000, +98.000] | +138.000 [+88.000, +187.000] | +285.000 [+161.000, +339.000] | +511.741 [+448.102, +578.688] | -9.347 [-9.673, -9.026] | -15.947 [-20.341, -11.627] |
| All B8 − All FRP3 (hero + v8) | -0.755 [-0.820, -0.688] | +88.000 [+77.000, +97.000] | +128.000 [+73.000, +180.000] | +64.000 [-30.000, +203.000] | +499.758 [+430.494, +575.001] | -1.163 [-1.243, -1.089] | -6.077 [-11.515, -0.728] |
| All B8 − B8 hero / FRP3 + v8 opponents | +0.056 [+0.004, +0.105] | +17.000 [+7.000, +29.000] | +31.000 [-28.000, +80.000] | +59.000 [-65.000, +148.000] | +101.392 [+36.747, +173.319] | -1.787 [-1.912, -1.660] | +15.395 [+10.347, +20.789] |
| All champion (hero + v8) − B8 hero / FRP3 + v8 opponents | +0.667 [+0.605, +0.725] | -69.000 [-76.000, -60.000] | -107.000 [-167.000, -48.000] | -226.000 [-280.000, -137.000] | -410.349 [-464.970, -356.204] | +7.561 [+7.153, +7.961] | +31.341 [+27.602, +34.972] |
| All FRP3 (hero + v8) − All champion (hero + v8) | +0.144 [+0.064, +0.223] | -2.000 [-8.000, +5.000] | +10.000 [-47.000, +69.000] | +221.000 [+82.000, +275.000] | +11.983 [-36.123, +62.169] | -8.184 [-8.512, -7.849] | -9.869 [-12.859, -6.389] |
| All FRP3 (hero + v8) − B8 hero / FRP3 + v8 opponents | +0.811 [+0.743, +0.881] | -71.000 [-78.000, -61.000] | -97.000 [-147.000, -51.975] | -5.000 [-140.000, +84.000] | -398.366 [-441.061, -354.353] | -0.623 [-0.747, -0.495] | +21.472 [+17.271, +25.439] |

### Arena: kills and death causes

| Left − right | Kills made | Kills taken | Head-on deaths | Body deaths | Wall deaths | Self deaths |
|---|---:|---:|---:|---:|---:|---:|
| All B8 − All champion (hero + v8) | -117.000 [-150.000, -87.000] | -117.000 [-150.000, -87.000] | -59.000 [-78.000, -39.000] | -68.000 [-99.000, -39.000] | -16.000 [-25.000, -6.000] | -315.000 [-354.000, -278.000] |
| All B8 − All FRP3 (hero + v8) | -230.000 [-277.000, -183.000] | -230.000 [-277.000, -183.000] | -132.000 [-161.000, -103.000] | -130.000 [-176.000, -87.000] | -2.000 [-8.000, +4.000] | -302.000 [-344.000, -257.000] |
| All B8 − B8 hero / FRP3 + v8 opponents | -6.000 [-34.000, +20.000] | -6.000 [-34.000, +20.000] | -1.000 [-16.000, +14.000] | -5.000 [-28.000, +17.000] | +1.000 [-3.000, +6.000] | +47.000 [+23.000, +71.000] |
| All champion (hero + v8) − B8 hero / FRP3 + v8 opponents | +111.000 [+78.000, +143.000] | +111.000 [+78.000, +143.000] | +58.000 [+40.000, +75.000] | +63.000 [+37.000, +88.000] | +17.000 [+8.000, +26.000] | +362.000 [+330.000, +394.000] |
| All FRP3 (hero + v8) − All champion (hero + v8) | +113.000 [+73.000, +153.000] | +113.000 [+73.000, +153.000] | +73.000 [+37.000, +110.000] | +62.000 [+27.000, +98.000] | -14.000 [-24.000, -3.000] | -13.000 [-67.000, +39.000] |
| All FRP3 (hero + v8) − B8 hero / FRP3 + v8 opponents | +224.000 [+177.000, +273.000] | +224.000 [+177.000, +273.000] | +131.000 [+98.000, +163.000] | +125.000 [+86.000, +165.000] | +3.000 [+0.000, +8.000] | +349.000 [+305.000, +394.000] |

## Monitoring erratum

The original release record is preserved. The [monitoring erratum](b8_s3_monitoring_erratum_2026-10-17.md) assigns Watch6 **0.424 / 0.848**, Watch8 **0.496 / 0.992**, and Watch12 **0.768 / 1.536** (baseline / trigger, deaths per 1,000 hero decisions). A rollback trigger requires strictly exceeding the matching roster threshold. Pooled traffic cannot fire or suppress a roster-specific death trigger. Existing latency qualification is unchanged; this measurement does not qualify Watch8 latency.

## Reproduction and evidence

Measurement source: `0fbb7c2c8462f0a93e3632a2a369ef4cfedc03bf`. CPU; four single-thread workers plus one coordinator (five measurement processes). 125 complete episodes, 625,000 arena frames, 4,000,000 snake decisions. Elapsed **22.35 minutes**; summed worker wall time **1.46 hours**. All worker exit codes were zero. No retries, tmux operations, or operations on unowned processes.

```bash
./venv/bin/python -m research.t13b_six_snake_20261017.run \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t13b-six-snake-20261017/run-v1 \
  --workers 4
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv/bin/python \
  -m research.t13b_six_snake_20261017.analyze \
  --run /Users/josenunez/Projects/ml/snake-dqn-artifacts/t13b-six-snake-20261017/run-v1 \
  --out research/t13b_six_snake_20261017/results
```

Outputs refuse overwrite; reproduction needs a fresh output directory.

- [Protocol](../../research/t13b_six_snake_20261017/protocol.md), [pre-run review](../../research/t13b_six_snake_20261017/REVIEW.md), and [five-cell fresh-seed H100 smoke](../../research/t13b_six_snake_20261017/SMOKE.json).
- [Seed audit](../../research/t13b_six_snake_20261017/SEED_AUDIT.json): 1,184,647 registered exclusions, zero registered or visible JSON/JSONL collisions; T7 final/rejected allocations, T12 banks, T13 worlds/smoke and historical banks included. No claim about opaque or unrecorded external experiments.
- [Summary and full CIs](../../research/t13b_six_snake_20261017/results/summary.json), [compressed raw episodes](../../research/t13b_six_snake_20261017/results/episodes.json.gz), [intent/source hashes](../../research/t13b_six_snake_20261017/results/intent.json), [completion receipts](../../research/t13b_six_snake_20261017/results/completion.json), and [evidence hashes](../../research/t13b_six_snake_20261017/results/evidence_sha256.json).
- [Machine-readable rule application](../../research/t13b_six_snake_20261017/RULE_RESULT.json) and [independent final review](../../research/t13b_six_snake_20261017/FINAL_REVIEW.md).
- Focused harness/analysis tests: 13 passed before launch. Final monitor tests and independent evidence checks are recorded in the final review.

Real GameSession step/snapshot and respawn paths were retained. Legacy homogeneous rows protect only their hero; mixed protects only its FRP3 opponents. The mixed comparison changes checkpoint and wrapping together, so it does not isolate those effects. All study telemetry is offline and excluded from product release monitoring. No policy math, network, checkpoint pin, gameplay rule, or serving default was changed.
