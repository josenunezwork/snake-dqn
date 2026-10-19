# Spec: L4 final-only acceptance dies on one transport stall (run 7)

Owner: Claude (plan/review). Implementer: Sol. Date: 2026-10-19 PT.

## Evidence (run 7, source 84eedcba, profile final-only)

Result dir: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance/acceptance-1792394444/`
(read `acceptance-result.json`, `outputs/l4/receipt.json`, `outputs/cpu/receipt.json`).

- CPU arm (lease 300 s): **PASS**. Every CPU-side gate OBSERVED, stop verified.
- L4 arm (lease 180 s): **INCOMPLETE** at 259 s of a 30 min cap, `stop_verified: true`,
  error `Refused('deadline reached during telemetry recovery')` (`supervisor.py` `poll_rpc`).
  The probe never reached `transfer_ready`; no download was attempted, so
  `final-artifact-download` and `mac-loss-watchdog-unassign` are INCOMPLETE.
- Call sequence tail (receipt `cli_calls` 36-40): `lease-heartbeat.json` upload (85 B) took
  41.5 s; following heartbeat uploads got shrinking timeouts (59 → 52 → 20 s); then one `exec`
  ran its whole 120 s and failed; the shared deadline was then already past.
- Earlier on the same arm, `acceptance_probe.sh` (236 B) took 39.5 s to upload. Colab Contents
  latency spikes of ~40 s and exec hangs of ~120 s are normal on this account (runs 3-7).

## Diagnosis to confirm (do not take on faith)

The L4 manifest uses `lease_seconds: 180` (acceptance.py:123) for the whole arm, not only for
the deliberate watchdog/lease-loss exercise. A single ~40 s Contents stall plus one 120 s exec
exceeds the lease-derived deadline, so a normal transport hiccup ends the arm before any
measured work. CPU survived because its lease is 300 s.

## Required outcome

1. A single transport stall of the size observed (≤ ~45 s Contents upload, one ≤ 120 s exec
   timeout followed by verified-idle recovery) must not end the L4 arm before the
   final-artifact download and the watchdog test. Keep every existing safety property:
   no exec resubmission while the kernel may be busy, bounded waits, no extra allocation,
   same unit caps, stop verified, lease-loss watchdog still proves VM-side checkpoint +
   unassign within 300 s.
2. Pick the mechanism yourself after reading the code (e.g. run-phase lease vs a short lease
   applied only for the watchdog exercise; or heartbeat timeouts that do not starve the shared
   deadline). Explain the choice in the commit message and ACCEPTANCE.md.
3. Fake-Colab regression test reproducing run 7's sequence (41 s heartbeat upload, then a
   120 s exec timeout, then idle) that previously refused and now completes the L4 final-only
   path; plus a test that a stall longer than the bounded budget still refuses cleanly.
4. ACCEPTANCE.md: add a short "run 7" evidence paragraph (CPU PASS, L4 INCOMPLETE as above).
5. `./venv`-equivalent: run `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q
   tests/test_colab_cli*.py -n 4` — all must pass.

## Rules

- Work only in `/Users/josenunez/Projects/ml/snake-dqn-colab-safety` (branch
  `t31-colab-safety-20261018`). Commit when green and push that branch. Do not touch tmux
  sessions, Colab, or `snake-dqn-artifacts`. Do not run any live Colab command.
- Do not weaken a gate to make it pass; do not change the S1 study code.
- Final message: what changed, why, test count, and anything you could not verify.
