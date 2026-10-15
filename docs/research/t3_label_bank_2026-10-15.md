# T3 label-bank interface and hand-back (2026-10-15)

Development work, not gate or serving evidence. The [ceiling diagnostic](t3_ceiling_test_2026-10-15.md)
returned **NO_GO**: 41/64 scripted episodes survived (0.640625), below the fixed 0.65 rule.
Part D therefore did not run. No label worlds or namespace were selected, no rollout-label
shards were generated, and no training experiment was launched. Training-time labels, if a
later study qualifies, must disclose their simulator-rollout origin; serving remains a bare
network without lookahead.

| Bank statistic | Result |
|---|---|
| Decisive rows | 0; generation gated off |
| Counterweight rows | 0; generation gated off |
| Target >=20,000 decisive rows | Not applicable after NO_GO |
| Boost share in surviving sets | Not measured |
| Label density / agreement with B8 argmax | Not measured |
| Fresh world namespace / shard SHA256s | None; no label bank exists |

## Part C: implemented independently of the ceiling verdict

The multi-actor trainer has three new flags. All defaults leave the previous numeric path
unchanged; no bank loading, sampling or bank telemetry occurs at fraction zero.

| Flag | Default | Meaning |
|---|---|---|
| `--label-bank PATH` | unset | Directory of sorted `.npz` shards |
| `--label-bank-frac F` | `0` | Rounded share of the configured anchor rows drawn from the bank |
| `--label-bank-margin-scale S` | `1.0` | Multiplier of the run's existing DQfD margin constant |

An active bank requires `--actors >=1 --obs-version bv8 --anchor-mode v8margin`. Bank-only
anchors use `--label-frac 0 --label-bank-frac 1`; `--onpolicy-labels` can be off. Combined
on-policy/bank anchors use `--onpolicy-labels` with fractions summing to one, for example
`--label-frac 0.5 --label-bank-frac 0.5`. The ordinary label portion retains the existing
minimum-buffer-size and retained-ring-row rules, so early batches can have fewer ordinary
anchors while that buffer warms. The bank is loaded and validated before output allocation,
shared memory or actor startup. The synchronous trainer rejects active bank flags.

The shard schema is:

| Array | Shape | Dtype |
|---|---|---|
| `local` | `(N, 7, 31, 31)` | `uint8` |
| `global` | `(N, 4, 37, 37)` | `uint8` |
| `scalars` | `(N, 60)` | finite `float32` |
| `mask` | `(N, 6)` | bool |
| `surviving` | `(N, 6)`, nonempty set per row | bool |
| `action` | `(N,)`, B8's chosen action, 0 through 5 | integer |
| `weight` | `(N,)` | finite, nonnegative `float32` |

With `--obs-pack`, a shard may replace `local` and `global` with `packed`, a `uint8`
`(N, train_layout().nbytes)` array. The loader unpacks using the trainer's layout and verifies
canonical bytes by repacking. Unpacked shards also work with packed training rings. Empty
banks, malformed shapes/dtypes, nonfinite values, empty surviving sets and invalid packed
payloads fail before actors start.

Bank rows receive only the specified surviving-set margin:

```text
row_loss = max_a(Q[a] + margin * scale * (a not in surviving))
           - max_a_in_surviving(Q[a])
bank_loss = mean(weight * row_loss)
```

Both maxima consider the specified six-action sets without applying the observation mask.
The stored B8 `action` is provenance; it is never an additional supervised target. Ordinary
on-policy anchors keep their existing single-action loss. Ordinary and bank means are combined
by their actual row counts. Reference and fast learner paths share these semantics, including
TD Huber delta, the lambda schedule, target synchronization and finite-loss handling.

Every 100k transitions, active-bank logs add `bank_rows_mean`, `bank_margin_loss` and
`bank_agree`. Agreement is the fraction of sampled bank rows whose raw six-action Q argmax
belongs to their surviving set. Bank-only runs safely retain `label_buffer=0`.
`--demo-data` remains required for the run's margin constant and hold-out Q-scale metadata.

## Verification

- The exact spec-required command passed **32 tests**:
  `PYTHONPATH=$PWD ./venv/bin/python -m pytest -q tests/test_t3_rollout_labels.py tests/test_m4a_trainer.py tests/test_perf_learner.py tests/test_td_huber_delta.py`.
- `tests/test_t3_label_bank_trainer.py` passed **12 tests**, including hand-computed loss and
  gradients, the fatal-B8-action case, packed corruption, invalid flag rejection, bank-only
  and combined bank/on-policy reference/fast parity, and a synthetic **512-transition CPU**
  CLI smoke with a fast learner, packed rings, one actor, no on-policy buffer and verified
  actor cleanup. Parity fixtures each ingest 320 new synthetic transitions.
- Before/after default-off traces matched byte for byte in four cases (`b` and `bv8`, each
  reference/fast): online/target tensor SHA256s, losses, counters, NumPy RNG state and ring
  validity-call counts. Plain Adam was used for the exact CPU comparisons.
- Flake8 is clean on changed/new Python files. The 222 protected tracked files, including
  `src/`, `web/`, census, T1 and `docs/research/INDEX.md`, match their starting hashes.
- Independent read-only reviews found no remaining A/B or C source blockers. CUDA/AMP and
  compiled bank execution were not separately runtime-qualified by this CPU validation.

Receipts live in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/`:
`c-bank-tests.log`, `final-required-tests.log`, `default-before.json`, `default-final.json`,
their exit receipts and `protected-verification.json`. The only learning runs performed were
bounded validation tests; no 8M experimental arm was launched.

## Part E: unexecuted template

`t3_lanes.sh.template` is in the same artifact directory. It contains CTRL and LABELS recipes
for seeds 10–14 initialized from B8_s0–4 finals, 8M transitions, the M6 flags and literal
`{TD_FLAGS}` placeholder for T2's TD-loss/anchor-floor choice. LABELS uses bank-only anchors
and `--anchor-hold 2000000 --anchor-floor-at 2000001 --anchor-floor 0.1`.

The template runs one arm at a time with two actors, keeping its numerical worker count at
three. It checks all selected arms before starting any run. Under this NO_GO receipt, label
arms remain unavailable: a future qualifying ceiling receipt and real bank are required.
The optional LABELS_RETURN arm is documented as reserved; return targets/regression were
not part of the required bank schema. The template has been syntax-checked and not executed.
No push, upload, serving change or experimental training launch occurred.
