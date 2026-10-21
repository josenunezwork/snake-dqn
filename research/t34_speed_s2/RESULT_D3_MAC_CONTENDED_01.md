# S2 D3 matched S0 result: Mac CONTENDED 01

**D3 passes on the excluded S0 workload: 1.8994x throughput,
+89.94% gain, 47.35% less wall time.**
Both definitions exceed the >=10% additional whole-workload bar. This compares native
v8 with stock digest-v3 execution; it is not a measurement of S1+S2 combined.

## Matched measurement

| Plain pair | Stock whole round (s) | Native whole round (s) | Throughput ratio |
|---|---:|---:|---:|
| 1 | 98.534327 | 51.877694 | 1.8994x |
| 2 | 100.901917 | 54.878870 | 1.8386x |
| 3 | 97.737206 | 49.006850 | 1.9944x |
| **Median** | **98.534327** | **51.877694** | **1.8994x** |

Each round contains all six S0 worlds, 300 decisions per world, H300/D300,
beta 1.0, CPU, one resident world, identical checkpoint and fixtures. Three
plain rounds per engine ran in AB/BA/AB order on the same contended Apple M5 Pro.
Warm-up and the two coverage rounds were excluded. One process, one thread,
`nice -n 10`, with actual niceness **10** in the receipt and process inspection.
The dev CPU slot was held, AC/lid checks passed, caffeinate covered the run,
and no thermal warning was recorded. Existing Phase 0 and other lanes continued.

Total wall time including warm-up and coverage was **596.968 s**
(9 min 57 s), under the 3,600 s deadline. Source remained fixed at
`fcb0b16ade6edcbcd5fef6b2b8a41ee0830e3ac8` for the entire measurement. The later prepared-respawn
counter fix changes only `identity.py` diagnostics and tests, not the measured
harness, native adapter, or benchmark. No engine was substituted into a registered study.

Whole-round work was **16,021 active row-steps**, including **10,139 search row-steps**;
the search-heavy long-mixed fixture accounts for most wall time. Results are
**CONTENDED-HOST engineering timings**, not quiet serving latency or a production
bank forecast. The fixture's single doom search determines its pooled weights.

| Case | Stock median wall (s) | Native median wall (s) |
|---|---:|---:|
| long-frozen | 11.476 | 7.719 |
| long-mixed | 70.810 | 27.615 |
| long-scripted | 3.714 | 3.649 |
| short-frozen | 5.087 | 4.600 |
| short-mixed | 4.428 | 4.484 |
| short-scripted | 3.548 | 3.308 |

Per-case medians are descriptive; D3 uses the median of complete rounds rather
than adding independently selected case minima or medians.

## Identity and coverage

All **eight** coverage/plain rounds, **18 files per round**, are byte-identical to
stock and the untouched `REFERENCE_MAC.json`: six world payloads plus six checksum
sidecars and six completion markers. The root independently rehashed **144 retained
files** against the committed reference after completion, with zero differences.
The stock source digest matches the frozen engine digest
`3b9f156400d07562cc623d0b02711f66259c7e04c9f869bc0e949f9745a40f2d`.

Each engine's coverage round recorded 10 root opponent deaths, 10 respawns, seven
shadow forks including a re-fork, one doom search, and all three mixes with
900/600/400/300/150/150 body lengths. Scored roots stayed on Python. The unchanged
stock collector wrote every world export.

Adversarial action/carry evidence supplements the S0 exports: the 33 passing focused
tests include exact-Q-tie and >=800-cell opponent rows, merged six-action branch
rollouts, audit corruption, staged fallback, and root routing. Pre-existing immutable
`receipts/2026-10-21_nonroot_qties_audit.json` contains 5,762 tied native rows and
26 tied-row overrides; `receipts/2026-10-21_nonroot_all_fixtures_audit.json` records
10,776 stock-audited native rows and 38 contested-food observations.

A fresh untimed corrected-counter S0 long-frozen audit at `1bb32a92` confirms
2,995 stock-audited native rows, three observed deaths and three respawns across
root/shadow worlds, four contested-food observations, and frozen-reference byte
identity. Its recipe also records CPU/one thread/actual nice 10. These observations
close the older diagnostic's prepared-respawn counting gap; old receipts were preserved.

This is D3 engineering evidence. The opt-in flag is available; default execution,
serving, and registered study engine identities remain unchanged.

## Reproduce and locate evidence

```bash
# Run from the worktree, on AC with lid open, outside quiet windows.
# The launcher in the artifact root supplies slot locking, nice 10 and caffeinate.
python -m research.t34_speed_s2.benchmark_s0 \
  --checkpoint /path/to/B8_s3_final.pth \
  --output-dir /fresh/absolute/output \
  --host-label mac-contended --deadline-seconds 3600
```

All paths below are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/speed-s0-20261021/`:

- `s2-matched-mac-contended-01/matched-s0.json` plus checksum/completion;
  SHA-256 `cf8320ecc0c6169a8fced044aba1995cfdbefe67c6ec718bff033bdfc671381d`.
- `s2-matched-mac-contended-01.sh` / `.log`, retained per-round receipts and raw exports.
- `s2-respawn-audit-01/respawn-audit.json`, SHA-256
  `2ac1eee5803ee708671f2ba48acaddab6a410b21b593c94096ba13555359e76a`, plus runner `.py` / `.sh` / `.log`.
- `s2-verification-01.log`: 33 passed in 18.78 s, no skips.
- `L4_MISMATCH.md`: saved Mac/L4 exports differ only in 4,443 stored Q scalars;
  all 1,800 root action entries and other exported fields agree. Repeat hashes are
  stable, with no observed tie flip or same-host nondeterminism. Exact backend cause
  is unproven; cross-platform byte identity remains failed. See
  `REVIEW_20261021.md` for review findings and the full report fingerprint.

The untouched frozen reference SHA-256 is
`2561df198872079ac59a1f0e55287a49cc38f3d3c4397f311550fe6d99cf113e`.
No Colab action, release change, or merge to main was performed. No access to the
restricted Round 0 study directory occurred. Existing tmux sessions observed and
untouched: `git-sync`, `phase0-selection`, `t34-claima-d0`, `t34-round0-autochain`,
`t34-round0-c001`. Both owned diagnostic sessions exited and released their slot.

Final main refresh: merged `origin/main` at `2bda7892a7ff1934249ea0bc8e76e14776115974`
into this branch. It adds unrelated O1 harvest files and a GPU-spike document;
the stock and audited-native source digests remain identical to their receipts.
