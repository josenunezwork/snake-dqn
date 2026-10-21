# T4a post-hoc diagnostic evidence (2026-10-16)

Diagnostic only. These results do not change the registered T4a GO/KILL/EXTEND decision and are
not gameplay-gate evidence. The full NPZ captures stay at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4a-posthoc-20261016/`. The
[tooling README](../../../research/t4a_posthoc_20261016/README.md) documents the commands.

**Cited by:** [t4a_posthoc_diagnostics_2026-10-16.md](../t4a_posthoc_diagnostics_2026-10-16.md)
(main report) and [t3_diagnostics_2026-10-16.md](../t3_diagnostics_2026-10-16.md).

**What it supports:** two results, both for training seeds 10-14 on the 96 round-62 scripted
H5000 worlds:

- The matched spillover tables: argmax-change rate and cross-boost share for TWO_SIDED, LABELS and
  LABELS_KEEPV8 versus CTRL, on decisive-bank, counter-bank and outside-bank states.
- The growth comparisons.

| File | Contents |
| --- | --- |
| `posthoc-summary.json` | Complete summary (`t4a-posthoc-complete/v1`, ~2.9 MB): per-arm results, producer receipts, SHA-256 bindings, stratum counts, weights, per-seed growth. |
| `boost-summary.json` | Per-arm boost summary with receipts (`t4a-posthoc-boost-summary/v1`). |
| `part1-audit.json` | Independent Part 1 audit of the 25 required captures (CTRL action and own-prelength comparisons); not gate evidence. |
| `part2-ready.json` | Readiness receipt: all required finals had finished before Part 2 ran. |
| `part2-independent-audit.json` | Independent raw-artifact audit; status `PASS` (the report gives 85/85 checks). |
