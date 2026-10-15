# T3 rollout-label ceiling test (2026-10-15)

Development diagnostic, not promotion or serving evidence. The shadow uses simulator rollouts; any future training labels would also come from simulator rollouts. The intended served artifact remains a bare network with no lookahead.

**Verdict: NO_GO.** The fixed rule is pooled scripted episode survival >=0.65 on the 64 required episodes: B8_s0 and B8_s3, each on the 32 original round-52 census worlds. The optional extension was omitted to bound CPU work while T2 lanes/probes were active. No outcome-dependent threshold tuning occurred.

## Rollout fidelity and throughput

The fixed random sample contains 20 T1 world/decision pairs and 99 legal continuations. Every continuation matched a fresh T1 single-world rollout on survival, death decision, transitions executed and alive transitions. Prepared-phase cloning, RNG independence, rowwise B8 inference and advisory masks preserve the census path.

Measured batched throughput: **972.1 env-frames/s** and **0.8429 decision batches/s**, including clone construction (23,067 valid continuation env-frames in 23.728s). Replay and single-world comparison costs are outside this throughput denominator. Single-thread CPU workers; at most three.

## Trigger and ceiling procedure

Trigger: length >=600 and **-max(raw Q) > 83.59117126464842**. Chosen prospectively as the most selective observed per-death maximum cutoff meeting 90% coverage after the five-decision cooldown. All six raw Q outputs enter the score. On the five B8 T1/M7 captures, 45/49 late deaths (91.84%) have a trigger in the last 150 decisions. Raw trigger rate 0.3845%; capped batch rate 0.0832%, both over all recorded alive decisions.

Each triggered batch tests every advisory-legal action, boosts included, for 300 transitions or the remaining H5000 horizon with the same B8 greedy continuation, eps 0, no veto. Select maximum alive transitions; ties prefer B8's current argmax, then highest Q and lowest index. There is at most one batch per five decisions; a state with no advisory-legal action retains B8's normal-only fallback. Before the first actual intervention each world must match recorded actions and lengths; unchanged episodes must reproduce census endpoint metrics exactly.

## Paired results

Episode survival is alive at H5000. Alive-frame survival is the mean post-step alive indicator over all 5000 frames. Mass is the mean post-step length over the same horizon with dead frames zero. Both runs use identical world seeds, rosters and checkpoint bytes. Baseline values are the census JSON records.

| Arm | Baseline episodes | Shadow episodes | Baseline alive-frame | Shadow alive-frame | Baseline mass | Shadow mass |
|---|---:|---:|---:|---:|---:|---:|
| b8_s0 | 21/32 (0.656250) | 23/32 (0.718750) | 0.899238 | 0.913506 | 652.259 | 669.074 |
| b8_s3 | 17/32 (0.531250) | 18/32 (0.562500) | 0.823175 | 0.830937 | 563.850 | 573.174 |
| pooled | 38/64 (0.593750) | 41/64 (0.640625) | 0.861206 | 0.872222 | 608.054 | 621.124 |

Ceiling wall time: 171.83s. 253 rollout batches, 49 actual changes from B8's current greedy action, 55,141 continuation env-frames. On 279,134 decisions (162,396 at length >=600), raw trigger rate was 0.4252% overall / 0.7309% among length-eligible states; capped rate was 0.0906% / 0.1558% respectively.

Death lengths below are actual pre-action lengths on the shadow trajectory. Complete death times, baseline identities, trigger decisions, candidate outcomes and paired endpoint deltas are retained per world.

- b8_s0: 9 deaths; lengths [1284, 724, 1012, 1317, 596, 1107, 1198, 1116, 340].
- b8_s3: 14 deaths; lengths [886, 1178, 1197, 1017, 972, 624, 907, 53, 1090, 559, 1113, 601, 1059, 664].

## Interpretation and next step

Part D did not run: no rollout label bank was generated. Part C is built and tested regardless of the ceiling verdict, as explicitly required. Part E supplies the unexecuted template requested by the owner; label arms remain unavailable without a separately eligible bank.

Coverage of a trigger in the last 150 decisions does not prove that it fired before the last avoidable decision. This result applies to the fixed trigger and greedy continuations on a reused development bank. It is not an impossibility result for other triggers, depths or continuation policies. No training experiment, promotion, serving change, upload or push occurred.

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015`. Receipts: `fidelity.json`, `trigger.json`, `ceiling/summary.json`, `ceiling-result.json`, `ceiling-pairs.csv`, `ceiling/<arm>_<env>.json`. Launch commands, logs, exit statuses and input hashes are retained alongside these files.

## Per-world paired changes

| Arm | Census env | World seed | Baseline alive at H | Shadow alive at H | Delta alive-frame | Delta mass |
|---|---:|---:|---:|---:|---:|---:|
| b8_s0 | 0 | 2515367783 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 1 | 539912944 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 2 | 3835332564 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 3 | 1229887956 | 0 | 0 | +0.000000 | +0.000 |
| b8_s0 | 4 | 4038340764 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 5 | 2976813830 | 0 | 0 | +0.000000 | +0.000 |
| b8_s0 | 6 | 4130786071 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 7 | 1401204162 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 8 | 3988821633 | 0 | 0 | +0.007000 | +7.072 |
| b8_s0 | 9 | 1996812937 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 10 | 2371429938 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 11 | 4046445356 | 0 | 0 | +0.002600 | +3.420 |
| b8_s0 | 12 | 3010307125 | 0 | 1 | +0.024600 | +36.499 |
| b8_s0 | 13 | 2570788724 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 14 | 2935400294 | 0 | 0 | +0.000000 | +0.000 |
| b8_s0 | 15 | 474182439 | 0 | 0 | +0.002200 | +2.427 |
| b8_s0 | 16 | 1142648366 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 17 | 1712899657 | 0 | 1 | +0.078000 | +105.793 |
| b8_s0 | 18 | 987234626 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 19 | 3774445511 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 20 | 191764384 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 21 | 1247816794 | 0 | 0 | +0.342200 | +382.877 |
| b8_s0 | 22 | 15900433 | 0 | 0 | +0.000000 | +0.000 |
| b8_s0 | 23 | 2687402297 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 24 | 1181537111 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 25 | 2493795855 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 26 | 1331460052 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 27 | 4114750927 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 28 | 1562232037 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 29 | 4051715387 | 1 | 1 | +0.000000 | +0.000 |
| b8_s0 | 30 | 1638785368 | 0 | 0 | +0.000000 | +0.000 |
| b8_s0 | 31 | 3273584622 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 0 | 2515367783 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 1 | 539912944 | 0 | 0 | +0.002200 | +1.951 |
| b8_s3 | 2 | 3835332564 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 3 | 1229887956 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 4 | 4038340764 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 5 | 2976813830 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 6 | 4130786071 | 0 | 0 | +0.000000 | +0.000 |
| b8_s3 | 7 | 1401204162 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 8 | 3988821633 | 0 | 0 | +0.002800 | +3.330 |
| b8_s3 | 9 | 1996812937 | 0 | 0 | +0.005400 | +5.458 |
| b8_s3 | 10 | 2371429938 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 11 | 4046445356 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 12 | 3010307125 | 0 | 0 | +0.000000 | +0.000 |
| b8_s3 | 13 | 2570788724 | 0 | 0 | +0.000000 | +0.000 |
| b8_s3 | 14 | 2935400294 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 15 | 474182439 | 0 | 0 | +0.001000 | +0.907 |
| b8_s3 | 16 | 1142648366 | 0 | 0 | +0.000000 | +0.000 |
| b8_s3 | 17 | 1712899657 | 0 | 0 | +0.003000 | +3.268 |
| b8_s3 | 18 | 987234626 | 0 | 0 | +0.000000 | +0.000 |
| b8_s3 | 19 | 3774445511 | 0 | 0 | +0.002400 | +2.647 |
| b8_s3 | 20 | 191764384 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 21 | 1247816794 | 0 | 0 | +0.000000 | +0.000 |
| b8_s3 | 22 | 15900433 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 23 | 2687402297 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 24 | 1181537111 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 25 | 2493795855 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 26 | 1331460052 | 0 | 0 | +0.000800 | +0.847 |
| b8_s3 | 27 | 4114750927 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 28 | 1562232037 | 1 | 1 | +0.000000 | +0.000 |
| b8_s3 | 29 | 4051715387 | 0 | 0 | +0.000000 | +0.000 |
| b8_s3 | 30 | 1638785368 | 0 | 1 | +0.230800 | +279.961 |
| b8_s3 | 31 | 3273584622 | 1 | 1 | +0.000000 | +0.000 |
<!-- t3-dense-ceiling-addendum:start -->
## Dense ceiling addendum (declared before running)

**Dense gate verdict: GO.** This is the ceiling of record: pooled scripted episode survival is 54/64 (0.843750) against the fixed GO threshold >= 0.65.

The dense trigger is every fifth episode-local decision beginning with the first decision at length >=600; its clock advances across later length dips and it has no Q threshold. Every batch evaluates advisory-legal actions for D=300 or to H5000, with B8 greedy continuation, eps=0 and no veto; ties prefer B8, then Q, then lowest index. All 64 dense receipts PASS, and every arm/env pair matches sparse and B8 baseline world seeds and checkpoint SHA256s.

| Arm | B8 episode survival | Sparse episode survival | Dense episode survival | B8 alive-frame | Sparse alive-frame | Dense alive-frame | B8 mass | Sparse mass | Dense mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b8_s0 | 21/32 (0.656250) | 23/32 (0.718750) | 28/32 (0.875000) | 0.899238 | 0.913506 | 0.949562 | 652.259 | 669.074 | 713.349 |
| b8_s3 | 17/32 (0.531250) | 18/32 (0.562500) | 26/32 (0.812500) | 0.823175 | 0.830937 | 0.935306 | 563.850 | 573.174 | 686.662 |
| pooled | 38/64 (0.593750) | 41/64 (0.640625) | 54/64 (0.843750) | 0.861206 | 0.872222 | 0.942434 | 608.054 | 621.124 | 700.005 |

Dense wall time: 16684.83s. 36,996 rollout batches, 75 changed actions, 36,997 dense triggers on 301,589 decisions (184,851 length-eligible): raw/capped trigger rates 12.2674%/12.2670% overall and 20.0145%/20.0140% among eligible decisions.

Raw triggers include terminal decisions with no advisory-legal action; those cannot execute a rollout batch.

Continuation core: 55,002,897 env-frames in 48843.84 summed worker-seconds (1126.1 env-frames/s and 0.7574 decision batches/s per worker). This includes clone construction; ordinary episode stepping and startup are outside that core timing denominator.

Dense death lengths are actual pre-action lengths: b8_s0: 4 deaths [1014, 596, 1086, 340]; b8_s3: 6 deaths [1218, 1200, 1020, 53, 559, 1067].

Part D status: COMPLETE, independent bank audit PASS; 10,885 decisive + 5,442 counter rows
from 1,400 fresh-namespace worlds. See [the final bank report](t3_label_bank_2026-10-15.md).
No training or push occurred.

### Dense per-world paired changes

| Arm | Env | World seed | B8 alive | Sparse alive | Dense alive | Dense-B8 episode | Dense-B8 alive-frame | Dense-B8 mass | Dense-Sparse episode | Dense-Sparse alive-frame | Dense-Sparse mass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| b8_s0 | 0 | 2515367783 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 1 | 539912944 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 2 | 3835332564 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 3 | 1229887956 | 0 | 0 | 1 | +1 | +0.127600 | +176.323 | +1 | +0.127600 | +176.323 |
| b8_s0 | 4 | 4038340764 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 5 | 2976813830 | 0 | 0 | 1 | +1 | +0.591800 | +668.221 | +1 | +0.591800 | +668.221 |
| b8_s0 | 6 | 4130786071 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 7 | 1401204162 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 8 | 3988821633 | 0 | 0 | 0 | +0 | +0.009600 | +9.847 | +0 | +0.002600 | +2.775 |
| b8_s0 | 9 | 1996812937 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 10 | 2371429938 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 11 | 4046445356 | 0 | 0 | 1 | +1 | +0.160400 | +229.968 | +1 | +0.157800 | +226.548 |
| b8_s0 | 12 | 3010307125 | 0 | 1 | 1 | +1 | +0.024600 | +35.346 | +0 | +0.000000 | -1.153 |
| b8_s0 | 13 | 2570788724 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 14 | 2935400294 | 0 | 0 | 0 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 15 | 474182439 | 0 | 0 | 1 | +1 | +0.164600 | +210.151 | +1 | +0.162400 | +207.724 |
| b8_s0 | 16 | 1142648366 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 17 | 1712899657 | 0 | 1 | 1 | +1 | +0.078000 | +107.817 | +0 | +0.000000 | +2.024 |
| b8_s0 | 18 | 987234626 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 19 | 3774445511 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 20 | 191764384 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 21 | 1247816794 | 0 | 0 | 0 | +0 | +0.192600 | +200.361 | +0 | -0.149600 | -182.516 |
| b8_s0 | 22 | 15900433 | 0 | 0 | 1 | +1 | +0.261200 | +316.851 | +1 | +0.261200 | +316.851 |
| b8_s0 | 23 | 2687402297 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 24 | 1181537111 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 25 | 2493795855 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 26 | 1331460052 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 27 | 4114750927 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 28 | 1562232037 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 29 | 4051715387 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 30 | 1638785368 | 0 | 0 | 0 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s0 | 31 | 3273584622 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 0 | 2515367783 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 1 | 539912944 | 0 | 0 | 1 | +1 | +0.422200 | +492.767 | +1 | +0.420000 | +490.816 |
| b8_s3 | 2 | 3835332564 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 3 | 1229887956 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 4 | 4038340764 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 5 | 2976813830 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 6 | 4130786071 | 0 | 0 | 0 | +0 | +0.145200 | +173.344 | +0 | +0.145200 | +173.344 |
| b8_s3 | 7 | 1401204162 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 8 | 3988821633 | 0 | 0 | 0 | +0 | +0.025200 | +30.077 | +0 | +0.022400 | +26.746 |
| b8_s3 | 9 | 1996812937 | 0 | 0 | 0 | +0 | +0.018800 | +18.992 | +0 | +0.013400 | +13.534 |
| b8_s3 | 10 | 2371429938 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 11 | 4046445356 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 12 | 3010307125 | 0 | 0 | 1 | +1 | +0.301600 | +363.973 | +1 | +0.301600 | +363.973 |
| b8_s3 | 13 | 2570788724 | 0 | 0 | 1 | +1 | +0.558400 | +574.338 | +1 | +0.558400 | +574.338 |
| b8_s3 | 14 | 2935400294 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 15 | 474182439 | 0 | 0 | 1 | +1 | +0.472800 | +526.729 | +1 | +0.471800 | +525.822 |
| b8_s3 | 16 | 1142648366 | 0 | 0 | 0 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 17 | 1712899657 | 0 | 0 | 1 | +1 | +0.093800 | +122.270 | +1 | +0.090800 | +119.002 |
| b8_s3 | 18 | 987234626 | 0 | 0 | 0 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 19 | 3774445511 | 0 | 0 | 1 | +1 | +0.012200 | +13.739 | +1 | +0.009800 | +11.093 |
| b8_s3 | 20 | 191764384 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 21 | 1247816794 | 0 | 0 | 1 | +1 | +0.603400 | +597.087 | +1 | +0.603400 | +597.087 |
| b8_s3 | 22 | 15900433 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 23 | 2687402297 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 24 | 1181537111 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 25 | 2493795855 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 26 | 1331460052 | 0 | 0 | 0 | +0 | +0.134000 | +142.501 | +0 | +0.133200 | +141.654 |
| b8_s3 | 27 | 4114750927 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 28 | 1562232037 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
| b8_s3 | 29 | 4051715387 | 0 | 0 | 1 | +1 | +0.569800 | +585.444 | +1 | +0.569800 | +585.444 |
| b8_s3 | 30 | 1638785368 | 0 | 1 | 1 | +1 | +0.230800 | +288.714 | +0 | +0.000000 | +8.753 |
| b8_s3 | 31 | 3273584622 | 1 | 1 | 1 | +0 | +0.000000 | +0.000 | +0 | +0.000000 | +0.000 |
