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
