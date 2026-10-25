# R30-1 timing calibration as a family probe

Prospective amendment, before calibration data: apply the exact
[TIMING_CALIBRATION_PROBE_AMENDMENT_20261025.md](TIMING_CALIBRATION_PROBE_AMENDMENT_20261025.md)
(SHA256 `447609b306bde15de4014640e1248a4349f8cb2feb811e455511a66886e943ff`)
alongside the unchanged original timing ruling. This supersedes the earlier
20/40-minute execution envelope and transport-blocker discussion in
TIMING_CALIBRATION.md. G5 remains HOLD; operator acceptance is still required.

The calibration itself is the family probe for `research/bl_round30/timing_colab.sh`.
There is no preliminary probe and no reuse of the scientific canary as timing-family
proof. The manifest declares CUDA, L4, 15-minute estimate, 30-minute whole-session
cap, the instrumented probe identity, and a 30-minute short-canary exemption.
The admitted full fit/audit plans, all training rows, 30 consecutive Adam updates,
closures at zero/10/30, precision and 600-second certificate cap remain unchanged.
Timeout or any incomplete phase leaves HOLD with no reduced-work replacement.

A single masked six-action decision uses B8 on predetermined admitted training
row zero. Its action and Q values are discarded. A separate probe phase measures
its overhead, including the installed observer's operational record. The observer
records count, shape, device and attributable caller metadata only, as specified
by the unchanged qualified observer. Timing metadata, terminal completion and
observer records cannot carry scientific values. The decision is not a training
update or a scientific look. Probe-only overhead is excluded from the main
extrapolation; all measured phases are removed before computing worker residual.

The worker has a 1,500-second maximum from launcher start. Production also reads
the existing supervisor's operational spec/state and uses the smaller remaining
runtime allowance minus a 300-second cleanup/export reserve. Bootstrap and staging
therefore consume the existing whole-session allowance. The unchanged supervisor
continues to enforce the 1,800-second manifest cap; forecast admission rejects
worker or supervisor elapsed times at or above their frozen limits. These are
failure guards, not runtime guarantees. The certificate is never shortened.

The new packet lives at `timing-calibration-probe-01`, with fresh runtime output
and supervisor state. It binds both ruling texts, the published source/config,
the tested dispatcher SHA and the qualified transport tree. The opaque private
bundle and old `timing-calibration-01` packet remain unchanged. Preserve the old
reservation and its full 40-minute liability until verified actual-debit/stop
reconciliation. The 10-minute unused difference is nontransferable; the combined
liability remains 472 minutes pending reconciliation. Main remains capped at
106 minutes with an estimate ceiling of 53 minutes. Verified L4 rate 1.54 units/h
and 20% technical reserve are frozen; they are reconciliation guards.

`tools/ops/dispatch.py` validates the optional timing-calibration no-retry flag
as a boolean. For this packet it is true: a retry-classified accounting failure
preserves the first attempt's evidence and refuses before creating a retry
manifest or starting a second allocation. Manifests without that flag retain
the prior wrapper behavior. The qualified `research/colab_cli` subtree and its
qualification registry are unchanged.

Offline tests use synthetic inputs only, nice 15, one numerical thread and at
most two test processes. Preparation performs only `dispatch.sh --dry-run`;
no Colab calls or live dispatch. Independent implementation review and a fresh
operator acceptance must precede execution. Account cap three, lease/watchdog,
transfer <=150 seconds, staging <=300 seconds, terminal receipt and independently
verified disappearance remain required. This probe cannot grant G4, G5 or release.
