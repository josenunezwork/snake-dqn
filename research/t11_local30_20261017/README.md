# T11 LOCAL30 implementation

Authority: [`t11_local30_spec_2026-10-17.md`](../../docs/research/specs/t11_local30_spec_2026-10-17.md).
The [pre-registration](../../docs/research/specs/t11_prereg_2026-10-17.md) was committed
and pushed before bank derivation and smoke training. This package does not change
the shared trainer, the immutable v2 source, evaluation authority, or serving.

- `bank.py`: create-only LOCAL30 selection and deterministic 1:1 counter matching.
- `bank_manifest.json`: exact copy of the immutable external bank manifest.
- `source_counts.json`: supplementary source-population counts bound to that manifest.
- `seed_audit.py`: refreshable rounds 74/75 exclusion audit, including the running T7 namespace.
- `seed_audit.json`: frozen pre-smoke reservation evidence; do not overwrite it.
- `bind_launch.py` / `launch_inputs.json`: create-only historical/input/code hash binding.
- `launch.py` / `lane{0,1}.sh`: MPS commands with hash checks, output protection,
  sequential lanes, shared serialization, and shared KILL propagation.
- `smoke.py`: only the isolated 200k MPS smoke; keeps the registered boost thresholds enabled.

The two prepared 8M scripts are in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t11-mac-20261017/`.
`bash lane0.sh --check` and `bash lane1.sh --check` print full commands without
training. Running without `--check` starts training and is outside the completed
implementation task's authorization. Do not launch either automatically.

Future evaluation must refresh freshness into a new receipt, preserving the
frozen one:

```bash
./venv/bin/python -m research.t11_local30_20261017.seed_audit \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t11-mac-20261017/evaluation-seed-audit.json
```

A passing receipt is a bounded audit of recorded sources, not a guarantee against
future concurrent allocations or unrecorded experiments. A training halt or
nonzero exit is KILL. A stale owned lane lock requires manual diagnosis; the
launcher never kills a process or removes another invocation's lock.

The smoke is implementation/throughput evidence only. It is not an 8M final,
a gameplay screen, or a promotion result. See the implementation receipt in
`docs/research/t11_implementation_2026-10-17.md` for measured outcomes and hashes.
