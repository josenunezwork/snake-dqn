# T2 screen status — incomplete (2026-10-15)

**development screen, not gate evidence.**

The registered screen is **INCOMPLETE**: `HUBER_s12` hit the trainer action-collapse tripwire at 7,764,992 transitions and exited 1. It produced `aborted.pth`, not `final.pth`. No aborted checkpoint was substituted. All fourteen other T2 runs exited 0.

All eligible finals have been probed: **60/63 initial rows**, plus the ten required ANCHOR0/CTRL extension rows. Every available row passed checkpoint-hash, settings, world-set and survival-accounting verification. There were **zero failed probe attempts**.

The strict watcher remains running and idle in `tmux t2p-run`, waiting for the three unavailable HUBER seed-12 rows. Its all-63-rows barrier correctly prevents the registered `analysis.json` and `t2_screen_result_2026-10-15.md` from being written. This sibling status note and `analysis_available.json` report the available evidence without changing that completion requirement.

## Design and measurement

Five paired seeds (10–14), each initialized from corresponding B8_s0–4 finals; eight-million-transition fine-tunes. CTRL reuses M6 CTRL_s10–14. HUBER uses TD Huber delta 100 and anchor floor 0.1; ANCHOR0 uses delta 1 and floor 0; HUBER_ANCHOR0 uses delta 100 and floor 0. All student probes have no veto. The incumbent is frp3-s12 plus v8, checkpoint SHA prefix `eec144bf`.

H5000, rowwise ego2s forward, advisory mask, numba vector61 backend and fast scripted anchors. Initial round 60 uses 96 scripted, 32 frozen and 32 mixed worlds per checkpoint. The incumbent was evaluated once on the same worlds. The seed audit found no overlap with distillation rounds 0/1/2/10/11/12, earlier probe rounds 50–52, NI banks, or round 61. External distillation manifests were also checked against the recorded seed inventories.

Episode survival is no death within 5000 frames. Alive-frame survival counts post-transition alive frames divided by 5000; a death frame contributes zero. Death frames are one-based, and length at death is logical mass immediately before the fatal action. Mass is the full-horizon mass integral, with dead frames zero.

## Decisions and scripted endpoints

GO requires episode delta >= +0.06, scripted mass delta versus CTRL >= -30, no frozen or mixed episode drop beyond 0.05, and holdout agreement >= 0.6. Primary <= 0 gives KILL; holdout < 0.6 or a mass drop beyond 60 kills the arm. Otherwise EXTEND once by 96 scripted worlds and re-decide at 192. Mass-collapse checks use each mix versus CTRL.

SEs below are **across paired training seeds / pooled paired worlds**, respectively. Positive survival deltas favor the candidate.

| arm | decision | seeds; scripted worlds per seed | episode delta | SE seed / world | alive-frame delta | SE seed / world |
|---|---|---|---:|---:|---:|---:|
| HUBER | INCOMPLETE — exploratory numbers only | 4; 96 | +0.125000 | 0.031538 / 0.034984 | +0.081102 | 0.014094 / 0.017172 |
| ANCHOR0 | EXTEND — extension exhausted; stop | 5; 192 | +0.003125 | 0.021195 / 0.022328 | -0.010460 | 0.008239 / 0.012980 |
| HUBER_ANCHOR0 | KILL | 5; 96 | +0.104167 | 0.039938 / 0.031129 | +0.065612 | 0.015260 / 0.015766 |

ANCHOR0 initial 96-world episode delta was +0.047917 (seed SE 0.031354; world SE 0.031066), triggering EXTEND. Round 61 evaluated all five ANCHOR0 and five CTRL finals on 96 more scripted worlds. The combined primary stayed positive but below +0.06; no second extension is allowed.

HUBER_ANCHOR0 has a complete five-seed cell and is KILL under the mass-collapse rule. HUBER has only seeds 10, 11, 13 and 14; its available-case mass drops also exceed the cutoff, but no registered five-seed decision is claimed. The trainer halt itself is recorded separately from the endpoint-based decision rule.

## Secondary outcomes

| arm | mix | episode delta vs CTRL | alive-frame delta vs CTRL | mass delta vs CTRL | mass delta vs incumbent |
|---|---|---:|---:|---:|---:|
| HUBER | scripted | +0.125000 | +0.081102 | -80.455 | +59.258 |
| HUBER | frozen | +0.132812 | +0.072042 | -95.894 | -17.406 |
| HUBER | mixed | +0.039062 | +0.013667 | -140.815 | +37.620 |
| ANCHOR0 | scripted | +0.003125 | -0.010460 | +6.861 | +169.169 |
| ANCHOR0 | frozen | +0.012500 | -0.000221 | -16.632 | +66.990 |
| ANCHOR0 | mixed | -0.018750 | -0.024716 | -39.614 | +134.091 |
| HUBER_ANCHOR0 | scripted | +0.104167 | +0.065612 | -140.667 | -2.104 |
| HUBER_ANCHOR0 | frozen | +0.118750 | +0.057923 | -176.549 | -92.927 |
| HUBER_ANCHOR0 | mixed | +0.125000 | +0.040651 | -202.410 | -28.705 |

ANCHOR0 scripted contrasts versus CTRL use 192 worlds per seed; its incumbent contrast uses only the initial 96. All other scripted contrasts use 96. Frozen/mixed use 32. HUBER values remain four-seed available-case summaries.

## Interaction

The registered five-seed interaction is unavailable because HUBER_s12 failed. Exploratory common-case HA − H − A + CTRL episode interaction: -0.057292, seed SE 0.030813, pooled-world SE 0.049388, 4 seeds / 384 initial-bank observations. It is not the registered interaction result.

## Training telemetry

Holdout agreement was absent from all twenty run logs and null in their summaries. It was not invented or replaced with an on-policy measure. Missing agreement cannot satisfy GO and does not count as observed agreement below 0.6.

| run | holdout agreement | last logged greedy boost rate | observation |
|---|---|---:|---|
| ANCHOR0_s10 | missing | 0.047085 | successful 8M run |
| ANCHOR0_s11 | missing | 0.076773 | successful 8M run |
| ANCHOR0_s12 | missing | 0.060802 | successful 8M run |
| ANCHOR0_s13 | missing | 0.073669 | successful 8M run |
| ANCHOR0_s14 | missing | 0.066256 | successful 8M run |
| CTRL_s10 | missing | 0.001069 | successful 8M run |
| CTRL_s11 | missing | 0.000719 | successful 8M run |
| CTRL_s12 | missing | 0.000629 | successful 8M run |
| CTRL_s13 | missing | 0.000799 | successful 8M run |
| CTRL_s14 | missing | 0.001279 | successful 8M run |
| HUBER_ANCHOR0_s10 | missing | 0.036548 | successful 8M run |
| HUBER_ANCHOR0_s11 | missing | 0.037534 | successful 8M run |
| HUBER_ANCHOR0_s12 | missing | 0.011379 | successful 8M run |
| HUBER_ANCHOR0_s13 | missing | 0.019421 | successful 8M run |
| HUBER_ANCHOR0_s14 | missing | 0.016684 | successful 8M run |
| HUBER_s10 | missing | 0.004227 | successful 8M run |
| HUBER_s11 | missing | 0.002987 | successful 8M run |
| HUBER_s12 | missing | 0.014336 | before abort; not final 8M telemetry |
| HUBER_s13 | missing | 0.004636 | successful 8M run |
| HUBER_s14 | missing | 0.001399 | successful 8M run |

## Caveats and verification

The failed seed prevents overall completion and the registered interaction. No new training, checkpoint substitution, serving change or promotion was performed. Pooled-world SE treats the same world reused across training seeds as independent; it is descriptive and reported alongside the paired-seed SE. Optional deaths per 1000 decisions at L >=800 was not measured. Death diagnostics do not establish the proposed Huber mechanism; that requires the separate T1 readout.

`PYTHONPATH=$PWD ./venv/bin/python -m pytest -q tests/test_t2_screen.py`: **13 passed**, including two-world H5000 mass/death identity with dev_probe. New package and tests pass flake8. All seventy recorded rows were revalidated, including fatal-frame and alive-frame consistency. Trainer, src/, web/ and research INDEX were unchanged.

The independent ANCHOR0 extension used two nice single-threaded workers. Only our own idle watcher was paused during this work and resumed afterward; neither training session nor any process owned by another task received control input or a signal.

## Artifacts and continuation

Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t2-probes-20261015`.

- `analysis_available.json`: complete-cell results and labeled incomplete estimates.
- `probes_available.jsonl`: 70 eligible rows; `row_manifest.json` binds their bytes.
- `analyze_available.py`: replay of this separate incomplete-study handoff.
- `seed_bank_audit.json`, `implementation_receipt.json`, `verification-pytest.log`.
- `training_failure_receipt.json`, `watcher.jsonl`, `progress.json`, `logs/`.
- `extension_events.jsonl`, `anchor0_extension_complete.json`.

Unavailable rows: `HUBER_s12__r60__scripted`, `HUBER_s12__r60__frozen`, `HUBER_s12__r60__mixed`. They require a successful registered final or an explicitly amended study disposition. The strict watcher remains ready to observe a valid final; this handoff does not authorize a retry or change the study design.
