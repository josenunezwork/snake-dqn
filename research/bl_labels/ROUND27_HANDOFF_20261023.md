# Round 27 implementation and operator handoff

Authority: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-27/JUDGMENT.md`,
SHA256 `f3d4e6254771830625ec25938f9eb6dd5a9750e383cb12b34a7a04b687b8518c`, section (f).
Implementation begins at `2a0821a9f5c6cd5f7b45ba6b22352e48e349b451`.
This is an implementation handoff, **not a T-EQ/calibration PASS**.

The running D256 worktree, its `61900af28932e47e22e332840fc4406559adb062`
science pin, outputs, lease and tmux session were not opened or modified.
No collection, calibration, profile or heavy byte-identity run was launched.
The independent Codex-V triplet is `CODEX_V_ROUND27_FINAL_20261023.json`.

## Checklist against the ruling

- WP-3: D1024/D4096 W6↔W12 resume only; W12 owner flag and sealed PASS receipt;
  unchanged namespace/control/source/deadline/state identity; per-attempt widths;
  cumulative denominator `sum(width * elapsed)`; fresh width-segment slowdown
  baseline and admission reset/recent-thermal reduction. Fake-pool prefix and
  arithmetic proof passes. The genuine three-parent W1→W2 proof is prepared,
  defaults to skipped, and remains for the operator after D256.
- WP-4: only `world_fold != 0`, D256 1:3:3:2 streams, 3:3:2 gate rosters,
  append-only allocation indices, frozen OLD D256 and cross-tranche seed audits.
  Registered wall caps remain 40 h / 130 h. Finite parent reserves 2160 / 7020
  are prospective operational envelopes derived by the same cap/floor method
  as D256: six workers × registered hours × eight gate parents per worker-hour,
  plus Watch6 parents at one eighth of gate count. They are not new selected-life
  doses or judged throughput estimates. No refill after data; shortfall stays
  INSUFFICIENT. D4096 has no dispatch admission until its registered R6 gates.
- WP-2: immutable prepared packet; one raster/rowwise B8 forward per living
  natural decision; forced steps skip hero inference; sim/decision/generation
  validity; incremental ring trimming; mass-integral bare audit; explicit
  production 1/64 old-action/Q-byte assertion on every future parent.
- WP-1: fixed excluded 12-parent `f0-speed-equiv-v1` bank, sealed materialized
  gate plans and seed exclusions; separate clean detached OLD/NEW trees;
  real W1/W6 process widths; every life/session/unit retained; strict canonical
  allowlist; F3 survival/growth pair and float32 mass-gap identity; actual F3
  loss-helper synthetic crosscheck; frozen 64-root/all-schedule F1 hash identity;
  external call counters/advisory masks; origin/main publication before analysis.
- W12: the existing real runner is preserved at runner commit `2a0821a9`.
  It executes frozen `61900af2` science and produces
  `bl-labels-f0-w12-calibration/v1`; ≥1800 measured seconds, ≤2 h,
  ratio ≤1.30, no thermal/slowdown trip for PASS. Future admission verifies
  runner, owner, bank, checkpoints, host samples and cell provenance.
- WP-5 remains operator work: actual calibration, T-EQ and timing; publish
  metadata before comparisons, then publish receipt hashes in CAMPAIGN_LOG.
- WP-6 remains unbuilt. Profile mode measures OLD gate-mixed and Watch6 parents
  only, emits shares, and freezes gates at forward 20%, digests 8%, snapshot
  pickle 5%, cold start 4% of total parent wall. At most one builder day; every
  accepted change still needs a fresh T-EQ. No pruning, batching, reduced precision,
  model arena or persistent pool was introduced. The v8/flood diagnostic counts
  the outer native kernel entry once. Lazy-bookkeeping data is explicitly an
  upper bound and cannot authorize that redesign.

## Commands after D256 finishes

Use the published implementation SHA from the final handoff/Codex-V source hashes,
not a moving HEAD after publishing the phase ledger. All actual Mac runs below
use the pinned dispatcher and the workload's shared lease. Wait for D256's verified
terminal receipt and lease release before the first actual run. Wait for each
submitted job to terminate before its dependent command. A successful dispatch is
not workload completion. Never reuse a dispatcher job name after it is consumed.

```bash
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f0
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed
PIN=bf8ac97dc66d1b4515e4c5510db3ad5886fa695d # Immutable implementation/bank pin.
CAL_RUNNER=2a0821a9f5c6cd5f7b45ba6b22352e48e349b451
cd "$SOURCE"
```

Offline preparation is **already complete** at this exact pin. Verify and reuse
these sealed paths; the following are the preparation commands for reference,
not commands to rerun into those existing paths. Inventory:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/build-01/prepared-inventory.json`,
SHA256 `5930e7a64dda67ddfbf5b8571af2c23363ebac19f0d3baf86a534d3c7f862928`.

```bash
"$PY" -m research.bl_labels.mac_package --future \
  --out "$ART/d1024-d4096-pack-01" --commit "$PIN"
"$PY" -m research.bl_labels.speed_equiv prepare \
  --out "$ART/equiv-pack-01" --new-pin "$PIN"
```

Calibration first, as §d.2 requires. The old runner commit has the actual built
calibration script; the collector-only OLD commit does not contain that script.

```bash
"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-w12-check-r27 "$CAL_RUNNER" -- \
  /bin/bash research/bl_labels/RUN_W12_CALIBRATION.sh --check
# After successful check, run the real calibration; wait for its sealed receipt.
"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-w12-calibration-r27 "$CAL_RUNNER" -- \
  /bin/bash research/bl_labels/RUN_W12_CALIBRATION.sh
```

T-EQ collection, including OLD/NEW W1/W6, the actual frozen F1 workload at each
pin, and interleaved OLD/NEW/OLD/NEW W6 timing. This creates its own excluded
reference trees; it never uses the live D256 tree.

```bash
"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-speed-equiv-r27 "$PIN" -- /bin/bash research/bl_labels/RUN_SPEED_EQUIV.sh
```

After `COLLECTED_NOT_COMPARED` and lease release, publish the metadata record.
This step reads only paths/sizes/hashes. Analysis checks its exact bytes in
`origin/main` before opening science payloads.

```bash
"$PY" -m research.bl_labels.speed_equiv publication --output "$ART/equiv-01" \
  --out research/bl_labels/ledger/ROUND27_SPEED_EQUIV_PHASES.json
git add research/bl_labels/ledger/ROUND27_SPEED_EQUIV_PHASES.json \
  research/bl_labels/ledger/ROUND27_SPEED_EQUIV_PHASES.json.sha256 \
  research/bl_labels/ledger/ROUND27_SPEED_EQUIV_PHASES.json.complete
git commit -m "Publish round-27 equivalence phase ledger" \
  -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git fetch origin
git merge origin/main
git push origin HEAD:main
git fetch origin
"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-speed-analyze-r27 "$PIN" -- "$PY" -m research.bl_labels.speed_equiv analyze \
  --output "$ART/equiv-01" \
  --publication-path research/bl_labels/ledger/ROUND27_SPEED_EQUIV_PHASES.json
```

The sealed result is `$ART/equiv-01/receipt.json`, schema
`bl-labels-f0-speed-equiv/v1`. The timing table is `timing-table.json` beside it.
The calibration result is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-calibration-01/receipt.json`.
Record these paths, pins and SHA256s in CAMPAIGN_LOG before adopting the D1024 pin.

D1024 needs the actual immutable **fault-free D256 F4 fit metadata**. Set these to
its already prepared registered staging pack; no synthetic approval file is
accepted and no D1024 labels are required before collection. F4_CONTROL must
be `control.json` inside F4_INPUTS. Pass them as command arguments through `env`
after `--`, since the tmux server need not inherit the caller's environment.

```bash
: "${F4_INPUTS:?Set the immutable registered D1024 staging input directory}"
F4_CONTROL="$F4_INPUTS/control.json"
"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-d1024-check-r27 "$PIN" -- /usr/bin/env \
  F4_CONTROL="$F4_CONTROL" F4_INPUTS="$F4_INPUTS" \
  /bin/bash research/bl_labels/RUN_D1024.sh --check
# Only after CHECKED, submit the actual collection:
"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" \
  bl-f0-d1024-r27 "$PIN" -- /usr/bin/env \
  F4_CONTROL="$F4_CONTROL" F4_INPUTS="$F4_INPUTS" \
  /bin/bash research/bl_labels/RUN_D1024.sh
```

The launcher selects W12 only for T-EQ PASS + calibration PASS and supplies both
W12 authorization flags itself. A sealed **measured** calibration FAIL selects
W6 on NEW. Missing evidence, unsafe host, HOLD, or unknown state refuses launch.
If T-EQ FAILs, run the same check/launch commands with the additional `env` argument
`OLD_FACTORY_ROOT="$ART/equiv-pack-01/old-reference"`; this explicitly uses the
entire frozen OLD scientific import closure at W6, while the new operations
wrapper retains accounting. A PASS forbids this fallback option. One mechanical
fix and new T-EQ for D4096 remains the ruling's limit.

Post-switch operational sanity remains Claude's duty: first three W12 hours,
RSS ≤40 GiB, zero thermal trips, completed-parent worker-seconds/parent ≤1.30×
the same-stream D256 W6 baseline. If a bar trips, switch to W6 at the next boundary
and log it without reading outcomes. Resume must use the same admitted control,
source, state directory and wall cap; build a verified portable union ledger and
use a fresh output directory for each attempt. Direct `mac_driver` CLI exposes
`--workers`, `--owner-w12-ok`, and `--w12-calibration-receipt` for those continuations.

## Prepared profile and deferred real-width proof

Do not run either while D256 owns the Mac. Profile uses the same excluded bank
and a fresh output. Its collection also requires publication before analysis;
use the same `publication`/commit/push/analyze sequence with profile-specific paths.

```bash
CKPT=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
"$SOURCE/tools/ops/dispatch.sh" --mac --repo "$SOURCE" bl-f0-speed-profile-r27 "$PIN" -- \
  "$PY" -m research.bl_labels.speed_equiv collect --profile \
  --package "$ART/equiv-pack-01" --output "$ART/profile-01" \
  --b8 "$CKPT/b8_s3_t7_20261016.pth" \
  --frp3 "$CKPT/frp3_m3_s12_u60000_20261005.pth"
```

The real-width proof's exact dispatched/leased command is in
`test_width_resume_real.py`'s module docstring. It is opt-in via
`BL_RUN_REAL_WIDTH_PROOF=1`; routine tests must leave it skipped.
No serving pin, released checkpoint, F4 registration, strict gate or Colab
transport change is part of this package.
