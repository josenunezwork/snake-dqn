# T2b Huber-delta sweep (2026-10-15)

**development screen, not gate evidence.**

## Decision rule

GO iff paired scripted episode-survival delta vs CTRL >= +0.06 AND mass delta vs CTRL >= -30 on every mix AND no other mix's episode survival down > 0.05; KILL iff primary <= 0; else EXTEND once (96 more scripted worlds, round_index 61, arm + CTRL; reuse the ANCHOR0 extension's CTRL rows if they exist on that bank).

A halted run (exit != 0) is dropped and blocks GO for that arm. This report records INCOMPLETE for a failed arm unless its available primary paired endpoint is <= 0, which is KILL. Holdout agreement is unavailable because the multi-actor trainer does not log it and is not a blocker.

## Decisions

| arm | initial / final decision | scripted episode Δ vs CTRL | scripted alive Δ vs CTRL | scripted mass Δ vs CTRL | failed runs |
|---|---|---:|---:|---:|---|
| HUBER10 | EXTEND / EXTEND | +0.072 (SE 0.017; world SE 0.022) | +0.037 (SE 0.017; world SE 0.012) | -14.001 (SE 15.555; world SE 11.614) | none |
| HUBER30 | EXTEND / EXTEND | +0.093 (SE 0.027; world SE 0.022) | +0.059 (SE 0.018; world SE 0.011) | -63.467 (SE 14.270; world SE 10.766) | none |

## Dose response

| arm | anchor floor | mix | episode Δ CTRL/inc | alive Δ CTRL/inc | mass Δ CTRL/inc |
|---|---:|---|---:|---:|---:|
| HUBER10 | 0.1 | scripted | +0.072 / +0.046 | +0.037 / +0.052 | -14.0 / +151.5 |
| HUBER10 | 0.1 | frozen | +0.000 / -0.037 | +0.030 / +0.011 | -28.2 / +55.4 |
| HUBER10 | 0.1 | mixed | +0.062 / +0.312 | +0.007 / +0.112 | -46.4 / +127.3 |
| HUBER30 | 0.1 | scripted | +0.093 / +0.023 | +0.059 / +0.046 | -63.5 / +80.8 |
| HUBER30 | 0.1 | frozen | +0.106 / +0.069 | +0.066 / +0.046 | -67.7 / +15.9 |
| HUBER30 | 0.1 | mixed | +0.075 / +0.325 | +0.024 / +0.129 | -99.6 / +74.1 |
| HUBER | 0.1 | scripted | +0.125 / +0.062 | +0.081 / +0.072 | -80.5 / +59.3 |
| HUBER | 0.1 | frozen | +0.133 / +0.078 | +0.072 / +0.045 | -95.9 / -17.4 |
| HUBER | 0.1 | mixed | +0.039 / +0.320 | +0.014 / +0.125 | -140.8 / +37.6 |
| HUBER_ANCHOR0 | 0 | scripted | +0.104 / +0.037 | +0.066 / +0.054 | -140.7 / -2.1 |
| HUBER_ANCHOR0 | 0 | frozen | +0.119 / +0.081 | +0.058 / +0.038 | -176.5 / -92.9 |
| HUBER_ANCHOR0 | 0 | mixed | +0.125 / +0.375 | +0.041 / +0.145 | -202.4 / -28.7 |

## Provenance and failures

All contrasts are paired by training seed where CTRL exists; seed SE is the decision SE and pooled-world SE is descriptive. Historical T2 Huber arms are round-60 only, so shared CTRL extension rows do not extend their dose-response contrasts.

Each initial cell has five paired seeds, 96 scripted worlds and 32 frozen/mixed worlds; an eligible extension adds 96 scripted worlds. H5000 uses rowwise ego2s forward, advisory mask, numba vector61 backend, fast anchors, and no veto for students.

Delta-100 historical HUBER has four available seeds because HUBER_s12 halted; its dose-response values are descriptive, not a five-seed decision.

Terminal statuses: HUBER10_s10=0, HUBER10_s11=0, HUBER10_s12=0, HUBER10_s13=0, HUBER10_s14=0, HUBER30_s10=0, HUBER30_s11=0, HUBER30_s12=0, HUBER30_s13=0, HUBER30_s14=0.

## Bank sizes and stopping

| arm | paired seeds | scripted worlds per seed | extension exhausted |
|---|---:|---:|---|
| HUBER10 | 5 | 192 | True |
| HUBER30 | 5 | 192 | True |

An exhausted EXTEND stops after round 61 without adopting the arm. Incumbent contrasts use round 60 only, including for extended arms; their CTRL scripted contrasts use 192 worlds per seed. Frozen/mixed retain 32. Episode survival means no death in H5000; alive-frame survival counts post-transition alive frames divided by 5000. Mass is the full-horizon mass integral with dead frames zero.

Preferred arm if both GO: not applicable. The preference compares the equal-weight mean mass delta vs CTRL across the three mixes.

## T2b secondary contrasts and uncertainty

| arm | mix | comparator | episode Δ (seed SE; world SE) | alive Δ (seed SE; world SE) | mass Δ (seed SE; world SE) |
|---|---|---|---:|---:|---:|
| HUBER10 | scripted | CTRL | +0.072 (SE 0.017; world SE 0.022) | +0.037 (SE 0.017; world SE 0.012) | -14.001 (SE 15.555; world SE 11.614) |
| HUBER10 | scripted | incumbent | +0.046 (SE 0.007; world SE 0.032) | +0.052 (SE 0.004; world SE 0.016) | +151.547 (SE 3.639; world SE 13.522) |
| HUBER10 | frozen | CTRL | +0.000 (SE 0.043; world SE 0.051) | +0.030 (SE 0.015; world SE 0.024) | -28.206 (SE 10.817; world SE 19.634) |
| HUBER10 | frozen | incumbent | -0.037 (SE 0.033; world SE 0.054) | +0.011 (SE 0.017; world SE 0.023) | +55.417 (SE 11.984; world SE 16.955) |
| HUBER10 | mixed | CTRL | +0.062 (SE 0.034; world SE 0.039) | +0.007 (SE 0.017; world SE 0.017) | -46.369 (SE 12.267; world SE 13.585) |
| HUBER10 | mixed | incumbent | +0.312 (SE 0.017; world SE 0.045) | +0.112 (SE 0.008; world SE 0.025) | +127.337 (SE 6.539; world SE 15.948) |
| HUBER30 | scripted | CTRL | +0.093 (SE 0.027; world SE 0.022) | +0.059 (SE 0.018; world SE 0.011) | -63.467 (SE 14.270; world SE 10.766) |
| HUBER30 | scripted | incumbent | +0.023 (SE 0.013; world SE 0.031) | +0.046 (SE 0.012; world SE 0.016) | +80.781 (SE 9.497; world SE 12.598) |
| HUBER30 | frozen | CTRL | +0.106 (SE 0.023; world SE 0.048) | +0.066 (SE 0.013; world SE 0.022) | -67.677 (SE 11.278; world SE 18.373) |
| HUBER30 | frozen | incumbent | +0.069 (SE 0.042; world SE 0.046) | +0.046 (SE 0.015; world SE 0.019) | +15.945 (SE 11.302; world SE 13.530) |
| HUBER30 | mixed | CTRL | +0.075 (SE 0.058; world SE 0.043) | +0.024 (SE 0.023; world SE 0.017) | -99.586 (SE 14.370; world SE 13.906) |
| HUBER30 | mixed | incumbent | +0.325 (SE 0.038; world SE 0.049) | +0.129 (SE 0.013; world SE 0.023) | +74.119 (SE 9.486; world SE 14.639) |

Probe failures: 0. Seed-bank audit passed: True. Checkpoint hashes, commands, row files and failure logs are retained in the existing redesign-t2-probes-20261015 artifact directory. This screen does not change serving or qualify a checkpoint for promotion.

## Final training telemetry

Holdout agreement is unavailable in this trainer and was not used as a blocker.

| run | last greedy boost rate | observation |
|---|---:|---|
| CTRL_s10 | 0.001069 | holdout agreement unavailable |
| CTRL_s11 | 0.000719 | holdout agreement unavailable |
| CTRL_s12 | 0.000629 | holdout agreement unavailable |
| CTRL_s13 | 0.000799 | holdout agreement unavailable |
| CTRL_s14 | 0.001279 | holdout agreement unavailable |
| HUBER10_s10 | 0.000879 | holdout agreement unavailable |
| HUBER10_s11 | 0.000290 | holdout agreement unavailable |
| HUBER10_s12 | 0.000889 | holdout agreement unavailable |
| HUBER10_s13 | 0.001019 | holdout agreement unavailable |
| HUBER10_s14 | 0.000380 | holdout agreement unavailable |
| HUBER30_s10 | 0.001908 | holdout agreement unavailable |
| HUBER30_s11 | 0.002538 | holdout agreement unavailable |
| HUBER30_s12 | 0.001938 | holdout agreement unavailable |
| HUBER30_s13 | 0.001568 | holdout agreement unavailable |
| HUBER30_s14 | 0.000569 | holdout agreement unavailable |
