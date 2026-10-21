# Authorized Mac measurements, 2026-10-21

Owner task: Y1 Mac ruler and L1′ diagnostics, about three Mac-hours, no Colab.
Owner authority records: `docs/research/advice/2026-10-21_big_lever/authority/`.
Artifacts: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l1-20261021/`.

The Y1 run uses the existing nine final frozen roots and unchanged ExactT3Operator,
one CPU process, one math thread, 30 actual dense H300 batches per root. No synthetic
batch is substituted. The guard runs between batches, outside the exclusive operator
timer. Each completed batch is exported before the next is admitted, with sidecars.
Nearest-rank p50/p95/p99/max retain all 30 samples; a separate warm-29 table drops
only the first batch per fixture. Guard/export overhead remains visible in the raw
ruler wall timer and is excluded from the separately recorded operator timer.
Source-bound roots are verified before measurement. No efficacy or release claim.

Phase-0 completion is owner-attested and its tmux absence is recorded. Quiet-host
snapshots retain load, system occupancy and processes; endpoint snapshots cannot
attest absence of transient contention throughout an interval. Any background
indexer observed is disclosed. AC/lid/thermal failure stops admission without killing
processes. Existing tmux sessions are read only.

L1′ production admission requires the registered reachable-root provider, frozen
five-decision offered trace, accepted clock seam, semantic adapter and independent
replay. The current tree has injected boundaries but no qualified production root
provider. Frozen Y1 roots cannot supply that missing proof. Any additional diagnostic
cells must be marked with that limitation, retain actual device placement, synchronize
MPS completion, and distinguish worker service, admission-to-visibility S, and valid
consumption latency C. Missing eligible consumptions cannot PASS. The serving grant
never substitutes for the scoped PROBE_ONLY file in probe transport.

CPU W1/K1 is mandatory. Expanded diagnostic cells are W1/K8, W2/K8, W3/K8,
W4/K8; the same device cells may be measured on MPS only with actual placement
and reference semantics. The explicit owner bars are S p99 W2 <= .83 s, W3 <=
1.25 s, W4 <= 1.458 s, with W1/K1 C p99 <= 1/2.4-2/24 or 1/1.6-2/24 s.
Coverage requires >=29 or >=20 valid consumptions in every 12-s burst; the 15-s
36/24 and complete mode/repetition inventory remain original-contract obligations.
2.4 roots/s is governing; 1.6 is sensitivity because X8 is not accepted.
Missing production evidence yields NOT_READY and nomination none, regardless of
micro-timing. No release, serving switch, benchmark optimization or scientific
threshold amendment is authorized here.
