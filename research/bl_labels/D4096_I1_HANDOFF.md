# D4096 I1 operator handoff

Authority: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/JUDGMENT_D4096_START_20261024.md`,
SHA-256 `ccf1e0cd8d60a2019c983ab2714d58663ca42e2a228a0ddf78c16216c2ef83fb`.
This package prepares collection only. No factory, provider query, allocation or dispatch is run
by implementation or verification. The existing D1024 factory and its live state remain untouched.

The new pack is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/d4096-old-equiv-02-pack-01`.
The exact two control source fields bind science pin `111c47818a097c57451386ef4f8a7009616f8f77`;
the collector remains OLD `61900af28932e47e22e332840fc4406559adb062`. The allocation hash remains
`74b1c5db8cd0c082311cf2c02666418847855298762174e0b9f721762cbff229` and the cumulative wall cap
remains 468,000 seconds. Read the count clarification in
`ledger/D4096_OLD_EQUIV02_ERRATUM_20261024.md`: 7,020 frozen fold-1--4 rows occupy the
8,792-index inclusive span; no allocation has been regenerated.

## Before sealing

Claude-V must accept I1 before dispatch. D1024 must be terminal COMPLETE with the factory exited and
lease released, and **every** D1024 fit canary/main/allowed redispatch must be terminal or HOLD.
Do not infer those facts from a projected finish time. Preserve the existing fit runbook unchanged.
Only after that gate, use terminal metadata to seal the throughput baseline, readiness and projection.
Do not read the labels, rescue counts, support or sealed rate payloads. The projection uses the fixed
1,024 selected-life denominator and 3,072 increment, never observed support.

Use this environment for every operator command below, from the clean published worktree:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-d4096-i1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export D4096_PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/d4096-old-equiv-02-pack-01
```

The baseline producer accepts the D1024 sealed control and **every** terminal receipt, in chronological
start order, through repeated `--terminal` arguments. These paths become known only after the current
factory ends. It verifies the start/terminal chain and unique complete parent coverage, then computes
same-stream W6 parent rates from the final cumulative active-stream wall metadata. See its CLI:

```bash
nice -n 15 ./venv/bin/python -m research.bl_labels.w12_runtime --help
nice -n 15 ./venv/bin/python -m research.bl_labels.d4096_sealed --help
```

After supplying the terminal metadata paths and the full fit-job inventory, seal readiness at
`$D4096_PACK/d4096-sealed-readiness.json` and the projection at
`$D4096_PACK/d4096-mac-h-projection.json`. Both are create-only and rechecked on admission and resume.
The explicit exit/release/job-inventory flags are operator attestations; the tool does not inspect or
mutate a lease or a live process. A projection above 130 h is HOLD and returns to the judge.

After the operator has independently confirmed B1, this exact metadata-only command derives the
paths, refuses any missing terminal/fit receipt, and seals the three receipts. **Do not run it while
D1024 or a fit job is live.** It does not itself establish factory exit or lease release; the three
explicit `True` arguments attest the operator's completed checks. It reads only control, start,
terminal, manifest and supervisor receipt metadata, including historical D1024 fit attempts.

```bash
nice -n 15 ./venv/bin/python - <<'PY'
import json
from pathlib import Path
from research.bl_labels import d4096_sealed, w12_runtime

art = Path('/Users/josenunez/Projects/ml/snake-dqn-artifacts')
pack = art / 'bl-labels-f0-speed/d4096-old-equiv-02-pack-01'
old = art / 'bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01'
def terminals(state):
    starts = sorted(state.glob('attempt-*.start.json'), key=lambda p: p.stat().st_mtime_ns)
    if not starts:
        raise ValueError('HOLD: no completed attempt chain')
    paths = [p.with_name(p.name.replace('.start.json', '.terminal.json')) for p in starts]
    if not all(p.is_file() for p in paths):
        raise ValueError('HOLD: attempt still live or terminal receipt missing')
    return paths
chain = terminals(old / 'd1024-state-mac-01')
baseline = pack / 'd1024-old-w6-throughput.json'
w12_runtime.seal_d1024_old_w6_throughput(old / 'd1024-control.json', chain, baseline)
jobs = {}
for manifest in sorted((art / 'big-lever/exp/f4').rglob('manifest.json')):
    metadata = json.loads(manifest.read_text())
    job = metadata.get('job_id', '')
    if job.startswith('f4-d1024-'):
        if job in jobs:
            raise ValueError('HOLD: duplicate fit job identity')
        jobs[job] = manifest.parent / 'receipt.json'
readiness = pack / 'd4096-sealed-readiness.json'
d4096_sealed.seal_readiness(readiness, chain[-1], jobs, baseline,
    factory_exited=True, mac_lease_released=True, all_fit_jobs_listed=True)
d256_terminal = terminals(art / 'bl-labels-f0/mac-pack-02/d256-state-mac-01')[-1]
projection = pack / 'd4096-mac-h-projection.json'
d4096_sealed.seal_mac_h_projection(projection, readiness, d256_terminal, chain[-1])
print(json.dumps({'readiness': str(readiness), 'projection': str(projection)}))
PY
```

## Exact first-start check and start

Run the check only after the preceding seals exist. A missing seal is a HOLD, not authorization to
substitute fixture data or bypass the check.

```bash
nice -n 15 ./venv/bin/python -m research.bl_labels.d1024_w12 \
  --stage d4096 \
  --control "$D4096_PACK/d4096-control.json" \
  --ledger "$D4096_PACK/d4096-verified-ledger.json" \
  --output "$D4096_PACK/d4096-output-mac-01" \
  --state-dir "$D4096_PACK/d4096-state-mac-01" \
  --admitted-control "$D4096_PACK/d4096-admitted.json" \
  --speed-equivalence-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/equiv-02/receipt.json \
  --calibration-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/w12-admission-01/receipt.json \
  --old-factory-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0-speed/equiv-pack-02/old-reference \
  --sealed-readiness "$D4096_PACK/d4096-sealed-readiness.json" \
  --projection-receipt "$D4096_PACK/d4096-mac-h-projection.json" \
  --b8 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  --frp3 /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/frp3_m3_s12_u60000_20261005.pth \
  --owner-w12-ok --check
```

After B1/B2, Claude-V ACCEPT, host guards and this check pass, execute the same command with
`caffeinate -dims` before `nice` and **omit only `--check`**. This is the operator's action; it was not
run by the implementer. Omit `--owner-w12-ok` to request W6. A verified runtime width policy may
require W6. No automatic completion retry is launched.

## Continuation and stop

After a terminal operational preempt, use `research.bl_labels.d4096_resume`, not the first-start
launcher. It selects a fresh numbered output and unions exactly all verified immutable exports into
its private resume directory. It charges every prior terminal attempt and preserves the 468,000-second
cumulative wall cap independent of width. Request W12 with `--workers 12 --owner-w12-ok` and the
same `--w12-calibration-receipt` as admission, or W6 with `--workers 6`. Always run its `--check` first.
Once a delegation directory has been attached, retain the same `--delegation-dir` on every
continuation; omission or a different path is refused before any resume union is materialized.
D4096 workers retain inherited nice >=15, so these niced commands do not attempt a priority escalation.

Registered archive census and R-A looks preempt collection at a life boundary. Do not run them
concurrently. W12 thermal/performance warnings, owned RSS above 34 GiB, or failed same-stream
three-hour sanity drain to a terminal attempt; the resume tool selects W6 at the next boundary.
The W12 sanity comparator is D1024 OLD W6 for the same admitted stream, at least 1.15×; rate drift
uses Canary-B × 1.36 and slowdown state stays split by width.

Use `d4096_sealed seal-not-run` for B6 after the operator preempts collection on D1024 KILL_FIT,
INSUFFICIENT, R-A STOP_FUNDING/INVALID, or an unrecoverable R-A HOLD. The canonical output is
`$D4096_PACK/d4096-not-run-by-rule.json`. This create-only seal permanently blocks collection,
public union publication and the fit job path. No non-F4 reuse is authorized here.

The delegated-index hook was already on main when I1 began. It is forwarded only with PASS funding;
L4 identity/delegation still requires all separate judgment gates, the identical OLD collector and
the prescribed calendar boundary. I1 starts Mac-only. No D4096 final union can be published without
PASS funding. Use `research.bl_labels.d4096_final` only after funding and collection complete, from
a clean pinned commit published on `origin/main`.
For a funded Mac/L4 split, `COMPLETE_EXCEPT_DELEGATED` is charged and accepted only with verified
OLD delegation records. If no Mac work is admissible, resume holds for import/reclaim instead of
launching an empty attempt. Use the existing `research.bl_labels.l4_lane close` with
`--funding-receipt` and every verified Mac attempt root for the complete imported union; it rechecks
funding, discard, the full Mac attempt chain, C5 identity and complete parent coverage before publishing.
The Mac-only final publisher refuses incomplete split evidence before publishing its final marker.
After a forced ten-minute drain timeout or mechanical worker exit, a width change is refused until
a subsequent same-width attempt provides a verified life-boundary drain.

## I3 integration after the rebase

`origin/main` now contains I3's receipt producers and strict zero-unit/funding/discard checks.
I1 preserves those checks. Its original pre-dispatch B2 seal remains immutable. The create-only
`seal-i3-projection` adapter binds that seal to the two I3 collection-ledger hashes and requires the
same cumulative wall times, formula and 130-hour bar. It produces I3's exact
`bl-labels-f0-d4096-mac-projection/v1` interface without a permissive schema fallback.

After I3's D256/D1024 collection receipts exist, set their actual paths and run:

```bash
: "${D256_COLLECTION_RECEIPT:?Set the sealed I3 D256 collection receipt path}"
: "${D1024_COLLECTION_RECEIPT:?Set the sealed I3 D1024 collection receipt path}"
nice -n 15 ./venv/bin/python -m research.bl_labels.d4096_sealed seal-i3-projection \
  --mac-projection "$D4096_PACK/d4096-mac-h-projection.json" \
  --d256-collection "$D256_COLLECTION_RECEIPT" \
  --d1024-collection "$D1024_COLLECTION_RECEIPT" \
  --out "$D4096_PACK/d4096-mac-projection-i3.json"
```

Only after canonical PASS funding and the complete funded final union, release that exact ledger to
I3. This writer rechecks B6, the clean published code pin, the complete charged attempt chain and
the immutable union. It writes I3's exact `bl-labels-d4096-collection-disposition/v1`,
`disposition: RELEASED`, binding the actual funding and final-ledger hashes:

```bash
: "${D4096_FINAL_LEDGER:?Set the complete funded final union ledger path}"
: "${D4096_FUNDING_RECEIPT:?Set the canonical PASS funding receipt path}"
nice -n 15 ./venv/bin/python -m research.bl_labels.d4096_sealed release-collection \
  --pack "$D4096_PACK" --ledger "$D4096_FINAL_LEDGER" \
  --funding-receipt "$D4096_FUNDING_RECEIPT" --code-commit "$(git rev-parse HEAD)" \
  --out "$D4096_PACK/d4096-collection-released.json"
```

The later fit-pack builder must bind and copy this release receipt as
`stage_receipts.D4096_COLLECTION_DISPOSITION`, bind `factory_union_ledger_sha256`, and propagate B6
discard presence before dispatch, as I3's handoff requires. I1 does not create or dispatch that fit
pack. Canonical PASS funding and every registered fit/probe/look gate remain mandatory.
