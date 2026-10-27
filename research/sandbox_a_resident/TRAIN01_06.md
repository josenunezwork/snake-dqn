# TRAIN-01-06: bounded preparation diagnostics and storage ownership

This sibling engineering update supersedes only the per-launch distribution-screen
requirement in TRAIN01.md. The owner requested an offline diagnosis, a much cheaper
startup screen, a no-progress watchdog, and packet-06 on 2026-10-27. The standing
A100 development sandbox permits this engineering change. The PPO recipe, natural
H5000 task, training/preparation quotas and row caps, rosters, seed allocations,
proxy horizon and release boundaries are unchanged. No runtime was dispatched.

## What attempt-05 proves

The actual runtime was b4225e2540c6b0cdb2e00e2ce84debc8296f5deb, in the
c7da28fc lineage. Its verified calibrated-boundary.pt contains a completed
three-mix CPU/device H5000 screen with no material gaps and a calibrated
10,000-row temperature bank. That callback runs after screen return and successful
calibration. The last mixed frame-4864 line was stale telemetry, not the stopping
location. First visible calibrated exports were at elapsed 5141.5 seconds;
SIGINT was at 16982.05 seconds. The supervisor receipt has no remote terminal
trace, so the exact remote interruption-time stack cannot be recovered.

Preparation formerly emitted its first refit line at 1,500,000 generated rows:
62,500 E24 task steps, plus critic fitting. The observed A100 mixed/frozen device
screen rates imply roughly 2.0-2.5 hours of collection before that line, even
without fitting/staging overhead. Slow bounded refit fits the available evidence;
a deadlock or infinite loop was not demonstrated. Screen aggregation compares
only twelve scalar pairs and writes no files. Screen, fit and holdout collection
all have finite bounds; the cross-life batch selection checks available capacity.

There is a confirmed storage defect: each retained observation row was a view of
its whole E24 CPU batch. A logical 12,443-byte row retained 298,632 bytes, including
discarded slots. The 160,000 logical row cap could therefore retain substantially
more memory than intended (a loose upper bound of 47.78 GB rather than 1.99 GB).
Actual remote RSS was not recorded. Retained rows now own their storage, preserving
uint8 planes, float32 scalars, numerical values, ordering and RNG calls.

## Startup cost and liveness

Fresh launches run E32 per mix on both engines for a 256-frame **early-life prefix**
of natural H5000 worlds, with an explicit v2 coverage label. This reduces the
maximum screen steps from 30,000 to 1,536 (19.53 times fewer). It does not replace
late-life distribution evidence, establish closed-loop CPU/device identity, or
qualify efficacy. The complete H5000 screen remains available with horizon=5000.
Small-E or partial-mix diagnostic calls require test_only=True; production calls
still require all three mixes and E32. The completed screen is exported separately
before preparation, including a failed material-gap report. Existing verified
full-trajectory resume still skips the screen; attempt-05's calibration-only
boundary is not imported as a PPO completion unit under a different source pin.

Preparation, proxy and rollout collection report completed work at least once per
60 seconds when the Python work loop is advancing, plus stage transitions.
The job arms a native faulthandler timer, default 900 seconds, reset by these
progress callbacks. Supervisor polls cannot refresh it. On expiry it dumps all
Python thread stacks and exits nonzero, including when the main thread blocks in
a native extension. The launcher seals the closed stack dump and last-stage
record as immutable SHA-bound exports after child exit, preserving the failure
code. Runtime termination and release remain the qualified supervisor's job.
The stack dump is Python-level diagnostic evidence, not a native CUDA backtrace.
