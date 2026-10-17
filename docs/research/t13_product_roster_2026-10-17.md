# T13 product-roster measurement — 2026-10-17

**Descriptive measurement, not a gate. Serving defaults are unchanged.**

## What a viewer gets

**For someone following the highlighted hero, all-B8 Watch got worse on survival and growth than the FRP3 configuration.** Hero deaths rose from 0.368 to 0.768 per 1,000 decisions (difference +0.400, 95% CI [+0.232, +0.552]); median length fell from 191 to 96, and H5000 mass from 239.73 to 121.51. The B8 hero also died more often and grew less than the champion configuration. These are comparisons of the specified product rosters, including their different veto scopes, not isolated policy-skill claims.

**For the whole arena, the result reverses.** All-B8 had fewer deaths (0.712/1k versus FRP3's 2.109 and champion's 1.389) and more total mass (1,646.70 versus 1,072.01 and 994.93). Its arena death-rate difference versus FRP3 was −1.397/1k [−1.482, −1.315], and mass difference +574.69 [+535.52, +612.97]. The legacy Watch configurations protect only the hero with v8, so a strong featured hero coexists with much less persistent unwrapped opponents. All-B8 also had fewer attributed kills and much less boosting; a more persistent arena is not the same viewing experience as a dominant hero or frequent fights.

**Roster composition is a useful lever for exhibiting B8.** Keeping the same B8 hero but using FRP3 + v8 opponents reduced hero deaths from 0.768 to 0.624/1k (18.75% lower; mixed minus all-B8 −0.144 [−0.264, −0.024]), raised median length from 96 to 143 (+47 [23, 73]), and raised hero mass from 121.51 to 190.67 (+69.16 [32.84, 109.26]). Hero kills did not clearly change: +3 [−17, +23]. The tradeoff was lower total arena mass (−137.13 [−174.81, −99.47]); the small arena death-rate change was unresolved (+0.017/1k [−0.021, +0.056]). The mixed B8 hero still died more often than either legacy protected hero.

The measured answer is therefore: **worse for following the all-B8 hero's growth, better for whole-arena persistence; the mixed roster improves the B8 hero exhibition.** That supports a deliberate product comparison of mixed Watch, not a serving-default change from this descriptive study. Mixed changes both opponent checkpoint and opponent wrapping, so their separate effects are not identified here.

## Roster results

25 paired episodes per roster, 5,000 frames each, twelve snakes. Deaths and food are per 1,000 actual decisions; boost is the percentage of decisions with executed boost movement. Length samples are taken immediately before each decision. Kills made/taken and causes are totals over all 25 episodes. H5000 mass is mean post-step live logical length, with dead frames zero. For the arena, mass is the sum across all twelve snakes; the per-snake mean is also shown.

| Roster | Scope | Decisions | Deaths / 1k | Length median / p95 / max | H5000 mass | Kills made / taken | Deaths head-on / body / wall / self | Boost | Food / 1k |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| All B8 | hero | 125,000 | 0.768 | 96 / 333 / 488 | 121.51 | 39 / 35 | 12 / 25 / 0 / 59 | 0.24% | 159.21 |
| All B8 | arena | 1,500,000 | 0.712 | 102 / 386 / 835 | 1646.70 | 378 / 378 | 118 / 272 / 5 / 673 | 0.27% | 158.46 |
| All FRP3 (hero + v8) | hero | 125,000 | 0.368 | 191 / 653 / 996 | 239.73 | 300 / 23 | 6 / 17 / 0 / 23 | 2.50% | 185.19 |
| All FRP3 (hero + v8) | arena | 1,500,000 | 2.109 | 53 / 293 / 996 | 1072.01 | 1428 / 1428 | 700 / 826 / 25 / 1612 | 1.60% | 174.75 |
| All champion (hero + v8) | hero | 125,000 | 0.360 | 150 / 476 / 625 | 183.57 | 143 / 14 | 4 / 10 / 1 / 30 | 12.63% | 176.34 |
| All champion (hero + v8) | arena | 1,500,000 | 1.389 | 58 / 242 / 625 | 994.93 | 728 / 728 | 353 / 427 / 48 / 1255 | 12.56% | 173.14 |
| B8 hero / FRP3 + v8 opponents | hero | 125,000 | 0.624 | 143 / 594 / 883 | 190.67 | 42 / 30 | 5 / 27 / 0 / 46 | 0.19% | 205.74 |
| B8 hero / FRP3 + v8 opponents | arena | 1,500,000 | 0.729 | 93 / 359 / 883 | 1509.57 | 480 / 480 | 90 / 400 / 5 / 598 | 2.28% | 151.99 |

Arena H5000 mass per snake: All B8 **137.22**; All FRP3 (hero + v8) **89.33**; All champion (hero + v8) **82.91**; B8 hero / FRP3 + v8 opponents **125.80**.

No starvation or unclassified death causes were observed.

## Paired differences

Every entry is **left roster minus right roster**, estimate [95% bootstrap CI]. The same 25 seed indices are resampled jointly across rosters, with 10,000 draws (NumPy RNG seed 20261017). Pooled rates and quantiles are recomputed on every draw. Intervals are descriptive percentile intervals without multiplicity correction. Counts refer to the complete 25-episode sample, not per-episode averages. The maximum is the observed sample maximum, not an estimate of a population upper bound.

### Hero: survival, size, action and food

| Left − right | Deaths / 1k | Median length | p95 length | Max length | H5000 mass | Boost (percentage points) | Food / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| All B8 − All champion (hero + v8) | +0.408 [+0.288, +0.528] | -54.000 [-82.000, -30.000] | -143.000 [-218.000, -53.000] | -137.000 [-199.000, -108.000] | -62.063 [-93.704, -33.481] | -12.387 [-12.930, -11.873] | -17.128 [-27.152, -7.152] |
| All B8 − All FRP3 (hero + v8) | +0.400 [+0.232, +0.552] | -95.000 [-141.000, -51.975] | -320.000 [-420.000, -174.000] | -508.000 [-562.000, -259.000] | -118.220 [-164.867, -73.064] | -2.262 [-2.498, -2.035] | -25.984 [-37.632, -15.191] |
| All B8 − B8 hero / FRP3 + v8 opponents | +0.144 [+0.024, +0.264] | -47.000 [-73.000, -23.000] | -261.000 [-375.000, -67.000] | -395.000 [-457.000, -258.000] | -69.162 [-109.264, -32.843] | +0.050 [-0.006, +0.107] | -46.528 [-56.368, -37.280] |
| All champion (hero + v8) − B8 hero / FRP3 + v8 opponents | -0.264 [-0.376, -0.152] | +7.000 [-24.000, +38.000] | -118.000 [-239.000, +59.000] | -258.000 [-287.000, -125.000] | -7.099 [-49.603, +31.837] | +12.437 [+11.929, +12.975] | -29.400 [-35.776, -23.312] |
| All FRP3 (hero + v8) − All champion (hero + v8) | +0.008 [-0.144, +0.168] | +41.000 [-7.000, +91.000] | +177.000 [+24.000, +294.000] | +371.000 [+122.000, +400.000] | +56.157 [-0.179, +111.779] | -10.126 [-10.690, -9.598] | +8.856 [-1.504, +19.192] |
| All FRP3 (hero + v8) − B8 hero / FRP3 + v8 opponents | -0.256 [-0.400, -0.104] | +48.000 [+4.000, +95.000] | +59.000 [-102.000, +255.025] | +113.000 [-33.000, +250.000] | +49.058 [-6.083, +102.532] | +2.311 [+2.078, +2.549] | -20.544 [-28.576, -12.088] |

### Hero: kills and death causes

| Left − right | Kills made | Kills taken | Head-on deaths | Body deaths | Wall deaths | Self deaths |
|---|---:|---:|---:|---:|---:|---:|
| All B8 − All champion (hero + v8) | -104.000 [-138.000, -68.000] | +21.000 [+11.000, +32.000] | +8.000 [+1.000, +16.000] | +15.000 [+7.000, +23.000] | -1.000 [-3.000, +0.000] | +29.000 [+16.000, +42.000] |
| All B8 − All FRP3 (hero + v8) | -261.000 [-323.025, -202.000] | +12.000 [-3.000, +26.000] | +6.000 [-2.000, +14.000] | +8.000 [-3.000, +18.000] | +0.000 [+0.000, +0.000] | +36.000 [+21.000, +51.000] |
| All B8 − B8 hero / FRP3 + v8 opponents | -3.000 [-23.000, +17.000] | +5.000 [-10.000, +21.000] | +7.000 [-1.000, +15.000] | -2.000 [-15.000, +11.000] | +0.000 [+0.000, +0.000] | +13.000 [-3.000, +30.000] |
| All champion (hero + v8) − B8 hero / FRP3 + v8 opponents | +101.000 [+68.000, +134.000] | -16.000 [-27.000, -4.000] | -1.000 [-4.000, +2.000] | -17.000 [-28.000, -6.000] | +1.000 [+0.000, +3.000] | -16.000 [-29.000, -3.000] |
| All FRP3 (hero + v8) − All champion (hero + v8) | +157.000 [+89.000, +227.000] | +9.000 [-4.000, +23.000] | +2.000 [-5.000, +9.000] | +7.000 [-5.000, +19.000] | -1.000 [-3.000, +0.000] | -7.000 [-18.000, +5.000] |
| All FRP3 (hero + v8) − B8 hero / FRP3 + v8 opponents | +258.000 [+196.000, +324.000] | -7.000 [-22.000, +9.000] | +1.000 [-5.000, +8.000] | -10.000 [-23.000, +3.000] | +0.000 [+0.000, +0.000] | -23.000 [-35.000, -10.000] |

### Arena: survival, size, action and food

| Left − right | Deaths / 1k | Median length | p95 length | Max length | H5000 mass | Boost (percentage points) | Food / 1k |
|---|---:|---:|---:|---:|---:|---:|---:|
| All B8 − All champion (hero + v8) | -0.677 [-0.731, -0.621] | +44.000 [+41.000, +48.000] | +144.000 [+129.000, +161.000] | +210.000 [+146.000, +239.000] | +651.769 [+606.669, +695.891] | -12.294 [-12.561, -12.031] | -14.680 [-17.604, -11.792] |
| All B8 − All FRP3 (hero + v8) | -1.397 [-1.482, -1.315] | +49.000 [+45.000, +53.000] | +93.000 [+79.000, +107.000] | -161.000 [-225.000, +88.000] | +574.691 [+535.524, +612.970] | -1.328 [-1.382, -1.273] | -16.291 [-19.138, -13.459] |
| All B8 − B8 hero / FRP3 + v8 opponents | -0.017 [-0.056, +0.021] | +9.000 [+4.000, +13.000] | +27.000 [+2.000, +53.000] | -48.000 [-112.000, +63.000] | +137.127 [+99.469, +174.809] | -2.009 [-2.081, -1.938] | +6.468 [+4.482, +8.492] |
| All champion (hero + v8) − B8 hero / FRP3 + v8 opponents | +0.660 [+0.615, +0.708] | -35.000 [-39.000, -33.000] | -117.000 [-141.000, -93.000] | -258.000 [-287.000, -147.000] | -514.642 [-557.800, -471.849] | +10.286 [+10.027, +10.552] | +21.148 [+18.746, +23.699] |
| All FRP3 (hero + v8) − All champion (hero + v8) | +0.720 [+0.645, +0.801] | -5.000 [-7.000, -3.000] | +51.000 [+34.000, +68.000] | +371.000 [+122.000, +400.000] | +77.077 [+41.314, +111.528] | -10.966 [-11.235, -10.694] | +1.611 [-1.339, +4.436] |
| All FRP3 (hero + v8) − B8 hero / FRP3 + v8 opponents | +1.380 [+1.308, +1.457] | -40.000 [-44.000, -38.000] | -66.000 [-93.000, -40.000] | +113.000 [-52.000, +224.000] | -437.565 [-483.116, -391.294] | -0.681 [-0.771, -0.595] | +22.759 [+20.512, +25.104] |

### Arena: kills and death causes

| Left − right | Kills made | Kills taken | Head-on deaths | Body deaths | Wall deaths | Self deaths |
|---|---:|---:|---:|---:|---:|---:|
| All B8 − All champion (hero + v8) | -350.000 [-424.000, -273.000] | -350.000 [-424.000, -273.000] | -235.000 [-294.000, -177.000] | -155.000 [-208.000, -100.000] | -43.000 [-57.000, -29.000] | -582.000 [-640.000, -519.000] |
| All B8 − All FRP3 (hero + v8) | -1050.000 [-1156.025, -951.000] | -1050.000 [-1156.025, -951.000] | -582.000 [-647.000, -517.000] | -554.000 [-639.000, -480.000] | -20.000 [-35.000, -8.000] | -939.000 [-995.000, -884.000] |
| All B8 − B8 hero / FRP3 + v8 opponents | -102.000 [-154.000, -51.000] | -102.000 [-154.000, -51.000] | +28.000 [-3.000, +58.000] | -128.000 [-174.000, -84.000] | +0.000 [-5.000, +5.000] | +75.000 [+26.000, +127.000] |
| All champion (hero + v8) − B8 hero / FRP3 + v8 opponents | +248.000 [+187.975, +308.000] | +248.000 [+187.975, +308.000] | +263.000 [+209.000, +318.000] | +27.000 [-21.000, +75.000] | +43.000 [+28.000, +58.000] | +657.000 [+596.975, +719.000] |
| All FRP3 (hero + v8) − All champion (hero + v8) | +700.000 [+599.000, +809.000] | +700.000 [+599.000, +809.000] | +347.000 [+270.000, +418.000] | +399.000 [+312.000, +495.000] | -23.000 [-40.000, -6.000] | +357.000 [+283.000, +432.000] |
| All FRP3 (hero + v8) − B8 hero / FRP3 + v8 opponents | +948.000 [+863.000, +1040.000] | +948.000 [+863.000, +1040.000] | +610.000 [+544.000, +674.000] | +426.000 [+348.975, +510.000] | +20.000 [+8.000, +35.000] | +1014.000 [+953.000, +1079.000] |

## What was measured

- The homogeneous B8 roster uses the released default, without a checkpoint override or veto. FRP3 and champion use their named serving environment selections and default v8 behavior: **only hero slot 0 is wrapped**. The other eleven snakes share the same checkpoint but are unwrapped. This is the actual named Watch behavior, not twelve independently veto-wrapped copies.
- The mixed roster uses B8 on hero slot 0 and FRP3 with the strict v8 wrapper on all eleven opponents. An explicit research-only hook attaches those policies and wrappers after normal registry resolution. Mixed changes both opponent checkpoint and opponent wrapping relative to all-B8, so this comparison does not isolate their individual contributions.
- The specification fixes twelve snakes. The current mechanics YAML has six; the harness uses the existing T12 constructor-local Watch12 configuration hook to build twelve without editing configuration files. The resulting twelve-snake measurements should not be equated with prior six-snake Watch measurements.
- Each frame runs the real `GameSession.step()` then `snapshot()`, including continuous Watch respawns. Offline release-monitor telemetry is suppressed so this descriptive experiment does not enter the product monitoring stream. No network, policy math, gameplay rules, serving defaults or checkpoint pins were changed.
- Decision lengths are observed at actual `AISnake.update()` calls after respawn. Deaths and kills come from authoritative per-frame events. Kills taken are attributed victim events and need not equal all deaths. Food is consumed items: the current live method allows at most one consumed item per snake/frame. Boost uses the executed action and actual boosting flag immediately after movement.
- Pooled length quantiles match T12: nearest sorted order statistic `round((N - 1) * q)`, using ties-to-even rounding. Hero and arena metrics use complete H5000 horizons; these are continuous-respawn Watch outcomes, not terminal single-life strict-gate scores.

## Reproduction and evidence

Measurement source commit: `56e20060e93e7bbc862f71c3efdff5ff802b28d7`. CPU; four single-thread workers plus one coordinator (five processes). 100 complete episodes, 500,000 arena frames and 6,000,000 snake decisions. Elapsed measurement time: 27.02 minutes; summed worker wall time: 1.74 hours. No automatic retries, no tmux interaction, no unowned processes operated on.

```bash
./venv/bin/python -m research.t13_product_roster_20261017.run \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t13-product-roster-20261017/run-v1 \
  --workers 4
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 ./venv/bin/python \
  -m research.t13_product_roster_20261017.analyze \
  --run /Users/josenunez/Projects/ml/snake-dqn-artifacts/t13-product-roster-20261017/run-v1 \
  --out research/t13_product_roster_20261017/results
```

Outputs refuse overwrite; use new output directories for a deliberate reproduction.

- [Protocol](../../research/t13_product_roster_20261017/protocol.md) and [spec](specs/t13_roster_spec_2026-10-17.md).
- [Seed audit](../../research/t13_product_roster_20261017/SEED_AUDIT.json): 25 fresh paired worlds plus a separate smoke seed; 1,184,621 registered exclusions, zero registered or visible JSON/JSONL collisions. Covers T7 final/rejected allocations, T12 banks, historical exclusions and mandatory artifact manifests; no claim about unrecorded or opaque external experiments.
- [Full summary and all metric CIs](../../research/t13_product_roster_20261017/results/summary.json), [raw compressed episode records](../../research/t13_product_roster_20261017/results/episodes.json.gz), [source/config/environment intent](../../research/t13_product_roster_20261017/results/intent.json), [completion receipts](../../research/t13_product_roster_20261017/results/completion.json), and [file hashes](../../research/t13_product_roster_20261017/results/evidence_sha256.json).
- [Pre-run independent review](../../research/t13_product_roster_20261017/REVIEW.md) and [four-roster H100 smoke](../../research/t13_product_roster_20261017/SMOKE.json). Nine focused tests passed, including exact T12 H100 hero trace parity. Full-run verification checks raw file hashes, exit codes, all horizons, per-snake accounting, mixed dispatch, stable identities, matching initial worlds/RNG states and frozen runtime source hashes.
- [Independent final evidence review](../../research/t13_product_roster_20261017/FINAL_REVIEW.md): PASS; independently reproduced every point estimate and bootstrap interval from compressed raw records and checked every rendered CI entry.
- Operational logs remain in the local artifact run directory. This experiment changes no serving defaults and authorizes no promotion or subsequent training arm.
