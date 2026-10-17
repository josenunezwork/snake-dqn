# T13b independent final review — PASS

Independent read-only reviewer checked the completed artifacts and final documents.
This receipt is materialized by the root from the reviewer's reported evidence;
the reviewer did not edit product, experiment, or documentation files.

## Independently verified evidence

- 125 unique, successful H5000 episode receipts: 25 per each of four six-snake
  rosters and all-B8 at eight. 625,000 arena frames; 4,000,000 snake decisions.
- All 183 frozen source hashes match. Raw compressed records equal original
  episode files; each original hash matches its zero-exit receipt. Progress,
  intent/completion copies and evidence-manifest hashes match.
- Seed derivation and prior-bank bindings match the passing audit. Five fresh
  smoke-seed H100 cells have the requested counts, identities and wrapper scopes.
- Initial world and Python/NumPy RNG digests match across all four six-snake
  rosters per seed. Checkpoint/config identities are stable. Actual checkpoint
  bytes match their recorded hashes, including champion, whose registry pin SHA
  field is intentionally null but whose raw checkpoint identity is hashed.
- Homogeneous B8 is bare; homogeneous FRP3/champion wrap only the hero; mixed
  preserves bare B8 hero dispatch and wraps only FRP3 opponents with strict v8.
- Rebuilt hero/arena aggregates from per-snake counters and reconciled decision
  histograms, death causes, kills and all required horizons.

## Independent numerical reconstruction

A separate read-only Python calculation did not import the study analyzer. It
recomputed bootstrap sums by direct seed indexing and length quantiles by CDF
binary search, using the registered 10,000 paired resamples at RNG seed 20261017.
All **200 point estimates** and **280 metric intervals** matched: 240 intervals
over the six six-snake roster pairs and 40 supplemental eight-minus-six intervals.
All **10 rendered result rows** and **156 rendered CI entries** matched the report.

The fixed rule reproduces:

- All-B8 hero deaths/1k 0.392 minus mixed 0.296 = +0.096: within 0.10.
- All-B8 hero p95 596 minus FRP3 hero p95 723 = −127: outside 100 cells.
- **PLAN_MIXED_PENDING_ROSTER_QUALIFICATION**. Serving defaults remain unchanged.
- Watch8 hero 62 deaths / 125,000 decisions = 0.496/1k; strict trigger >0.992.

Summary SHA256:
`aac6619936dd4856e9be3d5e6908f347755db6a33aa0d6a021c86b08985b5e8d`.
Compressed raw SHA256:
`8724b98b7393516bf563de5d22e474ee118f798717c05db331c66b2fe0f8a7eb`.

The monitor's final baseline mapping is exactly Watch6 0.424, Watch8 0.496,
Watch12 0.768. Its latency dictionary is unchanged. The erratum preserves the
original release record and distinguishes descriptive death baselines from
latency qualification. No final review blockers remain.

## Root integration verification

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./venv/bin/python -m pytest -q \
  tests/test_b8_release_monitoring.py tests/test_t13b_six_snake.py \
  tests/test_t13_product_roster.py tests/test_t13_analysis.py
```

**37 passed in 3.75s**: 24 monitoring tests, four T13b contracts and nine original
T13 harness/analysis checks. Includes exact death-trigger equality and exceedance
at all three roster sizes, no dilution across rosters, unknown-roster fail-closed
behavior and preservation of Watch8's unqualified latency status.

Black check passed for all seven new/changed Python files. isort check passed for
the six harness/monitor/test files; the copied, already frozen `seed_audit.py`
retains an import-order-only warning. Its hashed bytes were deliberately preserved
after the run. This is a style exception, not a numerical or runtime test failure.
`git diff --check` passed.

Root confirmed no changes under `src/`, `web/`, or `configs/`, and no change to
`docs/research/b8_s3_release_2026-10-17.md`, against the initial `cd1801de` checkout.
The coordinator returned exit 0 after waiting for all of its child processes;
no experiment process remains running. No tmux or unowned process was operated on.
