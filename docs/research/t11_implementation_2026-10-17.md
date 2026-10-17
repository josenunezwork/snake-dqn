# T11 LOCAL30 implementation and MPS smoke — 2026-10-17

**Prepared; 8M training unstarted.** The bounded MPS smoke passed. No evaluation,
promotion, serving change, or remote compute was performed. No unrelated tmux
session or process was touched.

## Registration and immutable data

[Pre-registration](specs/t11_prereg_2026-10-17.md) was committed and pushed first
as `29f62870`, before bank derivation and smoke training. The implementation
follows [the LOCAL30 spec](specs/t11_local30_spec_2026-10-17.md) and preserves the
TWO_SIDED trainer/loss implementation unchanged.

Source v2 contains 10,885 decisive and 5,442 counter rows. LOCAL30 contains
**577 decisive + 577 unique matched counters = 1,154 rows**. Selected decisive
gaps span 1–30 decisions, with the <=30 endpoint included. Counters match 1:1
without replacement within the registered length bucket and decisive-population
negative-max-Q decile; there are no unmatched strata. All original arrays and
weights are exact source-row selections. Original survival provenance defines
counter identity, preserving prefer-nonboost labels independently.

Artifact: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t11-mac-20261017/bank-local30`.
Manifest SHA-256: `a7825a6096373c4e8c31da36e92e7d14e40d9c7de3d77d22b6a683efabd7bf18`.
Payload SHA-256: `8d5e29475189fa8311e4b65dcf27420bbcf086750a4b33f8308a786452e34907`.
The tracked [manifest](../../research/t11_local30_20261017/bank_manifest.json)
is byte-identical to the immutable artifact. A separate
[source-count supplement](../../research/t11_local30_20261017/source_counts.json)
adds the original per-stratum census without rewriting that manifest.

## Freshness and input binding

The [seed audit](../../research/t11_local30_20261017/seed_audit.json) passed:
rounds 74 and 75 each contain 96 unique worlds, mutually disjoint, with zero
registered, observed, or running-T7 collisions. It includes all five T7
round-76 banks (960 worlds), rejected T7 allocations, prior registered banks,
T4e collection, snapshot sources, 46 bounded training-namespace records, and
full visible JSON/JSONL scans including hidden/ignored paths. Seven legacy M3
summaries lack an explicit namespace field; independent review confirmed their
fixed `redesign-m3-train/v1` seeds 0–4 are covered by equal/larger prefixes in
the recorded namespace inventory. Unrecorded histories and opaque binary-only
world identities remain outside the audit's stated scope.

The audit source hash matches the executed file. Refresh into a new receipt
before future evaluation, preserving the frozen reservation evidence.
The [launch binding](../../research/t11_local30_20261017/launch_inputs.json)
freezes 37 input/source files, including B8 and CTRL finals, source and derived
banks, margin/demo inputs, registration, launcher, trainer and label loss.
Hash checks passed before and after the smoke.

## 200k MPS smoke

Executed from clean pushed commit `34dd6c19f4a62d5623e29ad939270ea3c88d7d2c` with:

```bash
./venv/bin/python -m research.t11_local30_20261017.smoke
```

The receipt records the fully expanded trainer command. The smoke uses seed 10
and B8_s0, MPS, two actors, 128 worlds, and the exact registered TWO_SIDED
settings with LOCAL30. Both boost guards remain enabled; only the transition
budget/output and isolated KILL receipt differ from a full run.

| Measurement | Result |
|---|---:|
| Actual transitions | 200,064 |
| Trainer wall time | 58.928196 s |
| Trainer throughput | 3395.046976 transitions/s |
| Whole wrapper wall time | 61.232983 s |
| Whole wrapper throughput | 3267.258763 transitions/s |
| Exit / actor exits | 0 / [0, 0] |
| Halt / diagnostic flags | null / [] |

Greedy boost at the two 100k windows was 0.1078% and 0.6731%; both overall and
low-length rates remain below the respective 2% and 1% tripwires. Lambda was
0.1, bank rows per update 128, and on-policy rows 128. Final-window decisive /
counter agreement was 0.9420 / 0.9993; these are smoke diagnostics, not an 8M
readout or gameplay evidence. The smoke completed and its two actors exited.

[Full smoke receipt](../../research/t11_local30_20261017/smoke_receipt.json),
[telemetry](../../research/t11_local30_20261017/smoke_telemetry.jsonl), and
[completion hashes](../../research/t11_local30_20261017/completion.json).
Smoke checkpoint SHA-256:
`402a355450a3d80beb9b85a7184f8627e271c4bb047be2579f114f830b14b921`.

## Prepared scripts and verification

Copied after successful smoke; both are executable, byte-identical to tracked
copies, and have been invoked only with `--check`:

- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t11-mac-20261017/lane0.sh` — seeds 10, 12, 14.
- `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t11-mac-20261017/lane1.sh` — seeds 11, 13.

`--check` validates the frozen inputs and exact bank shard inventory, then
prints all five full MPS commands across the two lanes. There are no LOCAL30_s*
training outputs, lane-start receipts, KILL receipts, or active lane locks.
Scripts refuse existing output/log paths, serialize the two owned lanes,
recheck hashes after waiting and before each seed, and stop the arm on failure.
Neither lane has been started without `--check`.

Focused validation: 27 T11 tests plus 26 shared label-bank/loss regression
tests passed across bounded invocations (53 total). These cover boundary
selection, original-side identity, exact matching, tamper/shortage rejection,
seed collisions and metadata handling, launch guards, and smoke acceptance.
Black, isort, flake8, shell syntax, and diff checks passed. Independent review
verified original-array preservation, matched pairs, seed rosters and source
hash, all launch hashes, both smoke windows, and successful actor completion.

The spec's fetch/merge/push sequence was followed after each commit, publishing
to `main` and `t11-20261017`. Full training remains a future authorized action.
