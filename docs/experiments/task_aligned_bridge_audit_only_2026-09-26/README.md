# Audit-only check of saved admission-v4 evidence, 2026-09-26

Status: **AUDIT_ONLY_PASS** on saved admission-v4 evidence. This is an
interface/accounting audit only.

Admission-v4 itself is unchanged and stays
`CLOSED_INVALID_STOP_AUDIT_FAILED_LIVE_RESULTS_PROVISIONAL` (final seal
`90bdbb35…dab5f`, re-verified after this run). Its failed audit and all of its
costs are preserved. Nothing was replayed, reloaded, retested or promoted.

## What failed in v4 and why

The v4 audit rejected the carried fixture results: 75 nodes but 82 passed
`call` rows.

pytest 9.0.1 logs each passing `unittest.subTest` as an extra `SubtestReport`
call with an integer `0` duration, before the outer call. The schema-1
producer did not keep the report class.

The saved counters exclude duplicate test execution: `test_cases = 75`, with
75 setup and 75 teardown rows. The subtest identity of the 7 extra rows is
**structural inference**, not recorded provenance. It is labelled that way in
the new evidence.

## Correction (branch `task-aligned-subtest-report-boundary`, isolated worktree)

The worktree is `/Users/josenunez/.codex/worktrees/task-aligned-subtest-report-boundary/snake-dqn`,
based on frozen `344a797`. The frozen checkout is untouched.

| Commit | Content |
|---|---|
| `4a4a21f`, `ae9c7b6` | Producer schema 2 records the report class and subtest context, uses `xfail_strict`, and derives status from the rows. Audit groups rows as setup, declared subtests, one outer call, teardown. Schema 1 accepts only int-zero calls at source-pinned counts (3/4). |
| `14ecdcb` | Protocol: `research/task_aligned_challenger_20260924/fixture-subtest-audit-protocol.md`. |
| `aa51560`, `9495dc7` | `audit_only.py` wrapper, described below. |

What the wrapper does:

- Authenticates its own source.
- Anchors every v4 stage receipt to the sealed failure attestation.
- Pins the v4 root and audit failure files by hash and content.
- Snapshots the v4 tree by stat only, before and after the run.
- Runs one supervised child with a 300 s cap, a separate 120 s handoff
  reserve, RSS ≤ 8 GiB, available RAM ≥ 9.6 GiB, and the shared CPU slot locks.
- Caps reads at 120 files / 4 MiB.
- Allows no retry.

Review:

- Two single-reviewer lanes checked the correction. One found no blocker; four
  fixes were applied.
- A workflow then ran one implementation lane, three independent review lenses
  (1 blocker and 7 should-fix items, all fixed), and a final verifier, who
  returned GO at `9495dc7`.
- 70 synthetic tests pass. No frozen qualification module was rerun.

## Execution

The user authorized it in these words: "you you continue without my
permission using workflows".

Root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-bridge-qualification-20260924/audit-only-v5`

| Record | SHA-256 |
|---|---|
| intent.json | `e4ca18b804fa89dcef531bf7fd612a3cb73d6e824ed91cde3b2b788c4605024c` |
| output/closeout.json | `02395b2530735353e6beb874b696d1689e688c70506edab8b22d5a656bc0d6b3` |
| output/report.json | `41d0b092ab32fdcac4dd78aae696f0f250c1a4e737fbe2b835b7efa15f490eee` |
| output/audit-evidence.json | `7a53434b6d3c83a71429d8f0349cc6879f9ac3eb38291602b5bbe0aa73dccf6a` |

- Prospective deadline: 2026-09-26 03:19:04 UTC.
- The child exited naturally with code 0 after 0.234 s. Its peak RSS was
  22.7 MB, and the minimum available RAM was 29.7 GB.
- It made 87 JSON reads totalling 2,712,920 bytes.
- It read no checkpoints, constructed no models, played no games and ran no
  pytest.
- The v4 tree was unchanged: 137 entries with an identical stat digest before
  and after.
- All 14 cells validated. Fixture subtests: 3 + 4, provenance
  `schema1-structural-inference`.

## What this does and does not establish

It establishes that the bridge's saved interface and accounting evidence passes
the corrected independent audit.

It does **not** establish:

- better gameplay;
- independent world coverage (shakedown worlds 2/3 are shared with Watch and
  Play);
- recorded subtest provenance;
- any promotion.

Apex/vector61 remains the incumbent. The original 6102/global35613 candidate
(`8d67915e…`) still needs a fresh strict tournament comparison through the
shared gate.
