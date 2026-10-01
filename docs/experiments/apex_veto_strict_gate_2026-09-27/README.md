# Strict Tier-2 gate: Apex champion + free-space veto (run-v1), 2026-09-27

Status: **INVALID_STOP (`final supervision: heartbeat_stale`)**. Host sleep
stopped the run; it was not a code or science failure. Nothing is promoted,
and no retry is authorized under this admission.

## Admission

| Item | Value |
|---|---|
| Source | `apex-safety-and-eval-tooling` @ `213f38a` |
| Intent | `c0717256…b880` |
| Authorization | user: "confirm 45000" (final cap 30000→45000 s, before any final data) |
| Design | `research/apex_veto_strict_20260927/protocol.md`: N=235 per mix from the Tier-1 pilot at MDE 20 mass |
| Root | `snake-dqn-artifacts/apex-veto-strict-20260927/run-v1` |

## What happened (UTC)

- **18:50** run started.
- **18:50–18:58** calibration completed: 48 incumbent episodes, 2 workers, about 17.5 s each.
- **18:58–20:30** final stage: 148 of 1,410 episodes written. Episodes averaged about 70 s (max 212 s), roughly 4× slower than calibration. That is consistent with thermal throttling.
- **20:30** the power log shows the Mac on battery, then *Maintenance Sleep*,
  *Thermal Emergency Sleep* and *Dark Wake Thermal Emergency*. It slept
  about 16 min and woke on AC at 20:49 UTC. `caffeinate` cannot prevent
  thermal-emergency or lid-closed sleep.
- **20:49** both worker heartbeats were stale (more than 600 s), so the
  supervisor terminated the workers. The closeout is INVALID_STOP. Audit,
  decision and serving did not run.

## Preservation

- All stage records are preserved.
- The 148 partial final records are **not analyzed**. Looking at partial
  outcomes before deciding on a new admission would invite optional stopping.
- The final namespace `apex-veto-strict-final-v1` is partly consumed, so any
  new admission must use fresh namespaces.
- Apex/vector61 remains the incumbent.

## Next

A fresh admission (run-v2) with new dev, final and serving namespaces and the
same pre-registered design. The host must be on AC, with the lid open,
ventilated and not moved. That run needs explicit user authorization.

## run-v2 (2026-09-28): INVALID_STOP, closed-lid sleep on battery

| Item | Value |
|---|---|
| Source | `9e9a81f` |
| Intent | `bef9e80f…e0ea7` |
| Namespaces | fresh `-v2`, verified disjoint from every run-v1 bank |
| Authorization | "yes, it'll stay plugged in, run v2" |

The operator pre-check showed the host on battery, and the run was launched
anyway. That was a process error by the assistant.

| Time (UTC) | Event |
|---|---|
| 02:18 | Started |
| 02:18–02:26 | Calibration complete (48 episodes) |
| 02:26–02:51 | Final stage: 104 of 1,410 episodes, mean 29.5 s |
| 02:51 (19:51 local) | Power log: *Clamshell Sleep* on battery (70%), then repeated *Thermal Emergency Sleep* |
| 03:01 | Heartbeats stale; INVALID_STOP |

The 104 partial final records are not analyzed. No retry is authorized. The
v2 namespaces are now partly consumed.

**Lesson:** an 8-hour Tier-2 run on a laptop fails whenever the lid is closed
or the power is unplugged. Before any run-v3, do one or more of the following:

1. Make the launcher refuse to start on battery.
2. Make the runner tolerate host suspension. Base heartbeat staleness on
   monotonic time, which does not advance during sleep. Records are
   deterministic and write-once per episode, so a mechanically interrupted run
   could also resume safely.
3. Wire the fast SIMD engine so the gate takes much less wall time.
4. Run on an always-on machine.

## run-v3 (2026-10-01): **STRICT_PASS**

| Item | Value |
|---|---|
| Source | `c841726` (monotonic heartbeat, AC guard, fresh `-v3` namespaces) |
| Intent | `0159e26a…7e68` |
| Authorization | "i think we're good now continue" |
| Run | 00:26 → 04:42 UTC, on AC; calibration 48, final 1,410, serving 50 episodes |
| Mean episode wall time | 19.0 s |

The independent stdlib audit **PASS**ed. The producer self-check agrees, and
the decisions agree.

| Record | SHA-256 |
|---|---|
| `closeout.json` | `9df7a851…d122` |
| `receipt.json` | `18b65519…04c1` |
| `decision.json` | `88f5c4ef…f5` |

Final bank: 235 fresh worlds per mix, paired. Apex is the champion
`43d4e2c5…`; the candidate is the same checkpoint plus
`free-space-veto/v2-speed-preserving`.

| Mix | Apex mass | +veto mass | Δ (95% CI) | Holm-adj p | Survival Apex → +veto | Worlds better / equal / worse |
|---|---:|---:|---|---:|---|---|
| frozen | 49.2 | 125.4 | +76.2 [62.6, 89.8] | 5e-23 | 0.280 → 0.485 | 156 / 63 / 16 |
| scripted | 47.7 | 121.0 | +73.3 [59.8, 86.7] | 5e-22 | 0.247 → 0.428 | 147 / 75 / 13 |
| mixed | 61.8 | 144.5 | +82.6 [67.3, 97.9] | 5e-22 | 0.315 → 0.515 | 156 / 48 / 31 |

Every gate passed:

- **Superiority:** Holm superiority in all three mixes; at least two were required.
- **Scripted noninferiority:** the lower bound is +62.0, against a margin of −1.46.
- **Survival bands:** passed in all three mixes.

Mass is roughly 2.5× Apex's in every mix. Full-horizon survivals rose from 1
to 27 of 705. Most deaths are still self-collisions (196 / 192 / 205 with the
veto, vs 206 / 193 / 217 without). The veto delays self-traps far more often
than it prevents them, so a look-ahead veto remains the obvious next lever.

The Tier-1 screen estimate (+43 to +55 on 40 worlds per mix) was lower than
this result. Both are within noise of each other and confirm the same sign.

**What this does not do:**

- No champion file, default, config or deployment changed. A STRICT_PASS is a
  receipt only.
- `serving_path_qualified = false`. The web Watch/Play path has no veto hook,
  the serving receipt schema only accepts raster31v3, and a 50-episode web
  serving run with its own audit is still required.
- The study measures **one** wrapped hero against unwrapped opponents. Wrapping
  every AI snake in Play or Watch needs its own evaluation.
- Release is a separate, explicit user-approved action that cites this exact
  receipt and keeps the incumbent's rollback.
