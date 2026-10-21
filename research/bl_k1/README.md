# K-1 — PLAN_v2 WP-2

Authority: the owner request, `snake-dqn-artifacts/big-lever/PLAN_v2.md`
§1 WP-2 / §2 Mac schedule / §3 D2 / §8, and round-11 `JUDGMENT.md`.
Builder Sol-K; independent reviewer Codex-V. Code due Thu Oct 22 16:00 PT.
Claude operates the timing Thu Oct 22 **20:00–22:00 PT**. No Colab dispatch here.

The stock path is the default. `configure("replay", stats)` or
`configure("kernel", stats)` selects native v8 only inside frozen/mixed
`bank_engine.rollout_all`. The ruler accepts `ExactT3Operator(native_mode=...)`.
Roots and shadows continue using the stock hook. Each branch owns its diagnostic
carry dictionaries. S2 source and all T34 campaign packages remain unchanged.

`replay` uses S2's exact diagnostic-counter replay. `kernel` calls `v8_decide`
without that replay; its only diagnostic exclusion is `live_vetoes` bookkeeping.
Both retain exact runtime memory, RNG and `last_veto`. Row receipts distinguish
kernel, replay, fallback, stock branch, unsupported configuration and stock root
rows. Native mode is never inferred from the old `s2_native_v8` fixture flag.

## Frozen decision before timing

The primary mode is **kernel**, with a **≥5.0× ratio of stock to kernel median
`stage_seconds.opponent_v8` on frozen-1500**. Each mode runs 30 batches; the first
sample is reported as cold and excluded from the warm29 median. All30 and warm29
quantiles are retained. Replay is measured descriptively, not selected afterward.
The stock and native runs alternate on this Mac, W1, all thread pools = 1.

Before timing, the full audit must reconcile **nine original Y1 fixtures × 30
batches plus 24 freshly seeded constructed long-body roots (three mixes × eight)**,
each through independent stock/replay/kernel executions at H300. The audit hashes
every float32 Q vector, action grid, post-step world array, RNG state, runtime carry
and last-veto record. Replay also compares diagnostic counters; kernel names their
exclusion explicitly. Object arrays compare serialized values rather than pointer
bytes. Frozen root reconstruction verifies its complete root hash before use.
These constructed roots and repeated fixtures are identity evidence, not efficacy.

Any action/Q/world/RNG/decision-carry mismatch makes that mode INVALID and stops
the audit. No automatic retry. A missing or partial identity receipt cannot admit
timing. A complete timing result below 5× closes exact served search for week 2
and cancels WP-4. At/above 5× plus action identity admits only the conditional
WP-4 build. No strict family, serving change or release is admitted.

## Operation

Use the primary venv, never a second environment. Prepare a fresh immutable packet:

```sh
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export RULER_B8_PATH=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_k1.prepare \
  --fixtures /absolute/original/ruler-fixtures.json --out /absolute/fresh/packet
```

Run the generated `DISPATCH.txt` commands in order. `run_mac.sh --check MANIFEST`
validates source closure and prints the frozen bar/forecasts without starting work.
The launcher sets CPU/thread/checkpoint paths, AC/lid checks, nice and caffeinate.
Audit admission logs power/lid/thermal/load; timing requires an idle receipt first,
logs foreign load before every batch, and retains completed batches on interruption.
It checks the night window and 7200-second cap before and after each batch.
No daemon or process is killed. The operator reserves the Mac exclusively.

Every artifact is create-only JSON or text with `.sha256` then `.complete`.
The manifest contains compact reconstruction rows (no fixture directories uploaded)
and preserves the nine original fixture/root hashes. **Z-9:** K-1 is Mac-only,
zero upload bytes/units. No real-transport Colab qualification is claimed. A future
Colab consumer needs WP-1's real-supervisor upload receipt at real bytes forecasting
≤150 seconds of its 300-second cap before dispatch; this package does not grant it.

Forecasts: identity 1.5–4 h (three independently instrumented modes), Mac timing
1–2 h, 0 Colab units. These are planning bounds, not measurements. If identity
cannot land before the timing window, or any dependent deadline slips, create a
sibling `stage0/slips/<item>-<gate>.json` with builder, reviewer, gate,
missing_evidence, blocked_descendants, remaining_hours and earliest_finish; keep
all prior artifacts. Never overwrite/relabel an incomplete attempt.
