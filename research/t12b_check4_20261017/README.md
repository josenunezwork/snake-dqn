# `t12b_check4_20261017`

**Purpose:** T12B Check-4-only latency and serving-speed ceiling rerun for `b8-s3` (`Watch6`/`Play6` 24 fps, `Watch12`/`Play12` 12 fps) with pacing diagnostics and independent stdlib audit.

**Status / Outcome:** **`SERVING_PASS`** for Check 4 (2026-10-17: `0 / 120` overruns across all default and ceiling probes after pre-batched inference fix, completing T12+T12B qualification).

## Key Files & Results

- **Package entrypoints / artifacts:** [`protocol_check4.md`](protocol_check4.md), [`run.py`](run.py), [`audit.py`](audit.py), [`diagnose.py`](diagnose.py), [`results/REVIEW.md`](results/REVIEW.md), [`results/summary.json`](results/summary.json)
- **Related docs / results:** [`docs/research/t12_b8_serving_qualification_2026-10-17.md`](../../docs/research/t12_b8_serving_qualification_2026-10-17.md)
