# Independent T15 qualification review

The independent audit reviewer reran the frozen audit in memory against
`run-v1-assembly` and obtained **SERVING_PASS**, with no failures in checks 1–6.
The review changed no source or evidence and launched no gameplay.

## Integrity and correction

- All 116 receipt entries exist and match their SHA256 hashes. No file other
  than the receipt and its subsequent audit was unlisted at review time.
- All 1,099 intent-bound source/config/frontend files exist and match.
- Both checkpoint hashes match the intent and registry. Released defaults remain
  checkpoint `b8-s3` and Watch roster `all_b8`.
- All 114 original `run-v1` files have byte-identical mapped copies in the
  assembly. The invalid original UI envelope is retained under `originals/`.
- The corrected UI envelope differs at exactly `/wall_seconds`, from null to
  38.903733. Independent subtraction of the retained startup and browser-capture
  timestamps gives exactly that value. No observations were rerun.
- All 33 final result envelopes are complete, with finite durations and no
  top-level errors. Earlier smoke/preflight failures remain preserved and do
  not qualify. The original UI envelope remains INVALID; the passing verdict
  applies only to the documented sibling assembly.

Receipt SHA256:
`8f6711affa4d23ee65fb6c727ef8e82da30f120b1c711460ef4338d362e4ba3f`.
Audit SHA256:
`e20530cb27b6dd5d0e692c37e100a89a508ff72413eec5b676efa184dd49c23b`.

## Results and reporting boundaries

The reviewer independently recomputed update and full-tick p95 from all 1,500
raw samples per case. Values match exactly: update p95 7.7922 / 9.5192 / 13.1413 ms,
tick p95 8.6353 / 10.3557 / 14.0731 ms for Watch6/8/12. All six paced probes have
200 warm-up frames, 120 ticks, zero overruns and passing pre/post quiet checks.

The product/evidence reviewer confirmed 20,000 exact parity world frames but
7,992 active hero packets because parity intentionally makes the hero terminal.
Continuous Watch instead uses the real all-respawn lifecycle: 125,000 hero
decisions, 34 deaths, 124,966 alive frames; 0.272 deaths per 1,000 decisions.
Pooled logical-length p50/p95 are 248/841, with range 1–1,153. Opponents retain
625,000 decisions and 3,288 applied vetoes. These are mixed Watch6 descriptive
measurements, not comparative safety or promotion evidence.

The screenshot visibly shows the opponent aggregate badge, with long text
ellipsized. Individual IDs/counters are retained in DOM/accessibility evidence;
the report does not claim they are all legible in the screenshot. Startup
backend identity, screenshot and later API state represent different instants.

Review of the draft release erratum caught and corrected its prospective rollback
wording: after a future default change to mixed, rollback requires an explicit
`SNAKE_SERVE_WATCH_ROSTER=all_b8` selection. Unsetting is a rollback only while
today's default remains all_b8. No default change is part of this qualification.

The root reconciled the source and evidence reviews with the final report. The
independent audit reviewer found SERVING_PASS defensible for this frozen assembly.

## Verification command

```bash
./venv/bin/python - <<'PY'
from pathlib import Path
from research.t15_mixed_roster_20261017 import audit
print(audit.audit(Path(
    '/Users/josenunez/Projects/ml/snake-dqn-artifacts/'
    't15-mixed-roster-20261017/run-v1-assembly'
)))
PY
```

Additional independent read-only scripts compared all original/assembly hashes,
the receipt inventory, checkpoint/source hashes, corrected-envelope JSON paths,
timestamp subtraction and raw latency percentiles. No foreign process or tmux
session was controlled.
