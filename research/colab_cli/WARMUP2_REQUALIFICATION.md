# Warmup2 watchdog observation correction — 2026-10-22

Source under review: `bdb9a40a1dd114e0fd830b59f6a4aa853ea9698f`, branch
`supervisor-warmup-20261022`. This correction requires fresh live requalification;
offline tests do not qualify the transport. No Colab calls were made for this fix.

Evidence directory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/warmup2-requalification-20261022/requalify-puucow1s/acceptance-1792705484/`.
The original evidence remains unchanged.

## Failed gate and evidence

`acceptance-result.json` reports `INCOMPLETE`, `qualified=false`. Of eleven
required gates, ten are `OBSERVED`; only `mac-loss-watchdog-unassign` is
`INCOMPLETE`. The L4 arm reports
`Refused('watchdog checkpoint was not exported during grace')`.

`cold-kernel-metadata-recovery` passed on CPU and L4: the recovered kernel IDs
match their originals, tracked metadata was unchanged, and both first uploads
completed in one attempt. `final-artifact-download` passed: 67,108,864 bytes,
SHA-256 `8a38860428c868524d799977c95dfc245cce10ebcb03fc58816d9a138d242ce6`,
30.647 seconds, 2,189,724 bytes/s lower bound.

In `outputs/l4/receipt.json`, all 51 checkpoint-sidecar downloads failed. The
last call (`cli_calls[140]`) received a 26.533-second timeout and took 26.540
seconds. The observer's deadline is 195 seconds after the final renewal
acknowledgement (180-second lease plus 15-second grace), placing that request
at approximately 168.47 seconds after acknowledgement. That single request
occupied the entire remaining checkpoint window. Earlier checkpoint reads
also stalled for 40.187 and 34.457 seconds.

The observer allowed each read up to the general 120-second read budget,
clipped only to the end of grace. A request begun before the checkpoint exists
can therefore occupy the entire interval when a fresh lookup could retrieve
it. An offline replay with a checkpoint available during grace reproduces the
same refusal on the old code. This is an observer defect exposed by provider
latency; an unchanged rerun is not the prescribed fix.

The saved evidence cannot establish that the runtime actually published its
checkpoint or autonomously unassigned in this attempt. The observer raised
before polling assignment disappearance. `stop_verified=true`,
`cleanup_error=null`, and zero assignments afterward prove fallback cleanup,
not the watchdog gate. A real local probe signal check independently confirmed
SIGUSR1 produces a hash-valid checkpoint; it does not replace live proof.

## Correction and regression

`AcceptanceObserver._watchdog` now caps each small checksum/payload read at
three seconds. A stalled lookup leaves time for fresh requests and both files
within the existing 15-second grace. The lease, grace, session caps, and
300-second disappearance bound are unchanged. Exhaustion reports that the
checkpoint was **not observed**, with the last redacted download error; it no
longer claims that publication did not happen.

`test_watchdog_slow_pre_expiry_read_cannot_consume_checkpoint_window` replays
the late prepublication request with timeout and delayed-404 responses, and
one- and three-second successful file reads. It checks checkpoint integrity
and subsequent endpoint disappearance using the actual observer. The initial
regression failed on the old implementation with the reported refusal.

The operator must prepare and execute a fresh source-bound `final-only`
acceptance on the pushed correction SHA. No existing plan, receipt, or
qualification record is amended by this correction. Do not merge before that
requalification and review.

## Offline verification

Using `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`:

- `-m pytest -q -n 4 tests/test_colab_cli*.py tests/test_ops_qualification.py`:
  **428 passed** in 519.84 seconds.
- `-m pytest -q tests/test_colab_cli_latency_policy.py`: **10 passed**.
- Black (`--check --line-length 100`), isort (`--check-only --profile black`),
  and flake8 passed on both changed Python files; `git diff --check` passed.
- Independently verified the saved final artifact's size and SHA-256 on the Mac.

The initial serial broad test run was stopped to restart it with four workers;
the completed parallel run above is the full verification result.
