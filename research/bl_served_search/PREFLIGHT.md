# L2 / L0 verification handoff — 2026-10-21

Development only, excluded from registered verdicts. **L2 NOT DISPATCHED and
NOT_DISPATCHABLE. L0 INCOMPLETE under the frozen missing-data rule.**
No Colab calls or production-world evaluations occurred.

## Published identities

| Item | Git commit or SHA256 |
|---|---|
| Pre-data intent | `db8f698ba5c1b7041dcc9247ef55ef0c522b56f4` |
| Bank preparation commit | `53f4ff27` |
| L0 implementation commit | `7323bd68` |
| L2 implementation commit | `18f7622b4a03df1e86eed8657b8aa71a95e69417` |
| Final executable pin | `3b6f0bbf102d158085b2a975aa0fe42994de64d9` |
| B8 s3 checkpoint SHA256 | `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8` |
| Portable bank SHA256 | `0a8d3059e1feb7f9c954171093e0c21258b6cf89ea43e1c20f8e035776fcb8d3` |
| Seed audit SHA256 | `58a0f084180cab8dc9a03c33ffe69cd6ebcf28b3faafdf560ff92e1630967a20` |

The worktree is `/Users/josenunez/Projects/ml/snake-dqn-bl-l2`, branch
`bl-l2-dense-ceiling-20261021`. All implementation commits were pushed to
origin/main and the branch. One early integration merge, `d2c6ed0f`, omitted
the requested coauthor trailer; it was already published and was not rewritten.
All task content commits and subsequent integration merges have the trailer.

## Bank and frozen rule

Round 0 allocation CLEAR: 64 frozen + 64 mixed + 32 scripted H5000 worlds,
plus three burned preflight worlds. All 163 seeds unique; no collision in
the exact-token scan, four explicit historical bank checks, or seven concrete
FRP training-exclusion inventories. The scan pruned the sealed Round 0 output
directory; only the committed Round 0 bank supplied its identities.

The unchanged T3 rule remains L≥600, every fifth episode-local decision,
six advisory-legal actions × H300, B8 greedy continuation, eps 0, no veto.
The first eligible decision anchors the clock even with an empty advisory
mask; empty advisory masks skip search and retain B8 fallback. The ranking
uses alive transitions, excluding the fatal transition, then B8/Q/action ties.
World-level post-step mass has zero dead-frame contribution. The ledger splits
common-alive productivity, dense-only survival mass and baseline-only loss.

Frozen precedence requires complete identity and parity evidence before any
scientific readout. Pooled-equivalent UB90 <40 kills the served-search push.
Frozen and mixed LB90 >0 plus pooled point ≥40 funds Mac systems preparation.
The analyzer never accepts the three excluded fixtures as a scientific bank.

## Mac preflight

Final serial and parallel entrypoints used fresh, previously nonexistent output
parents. Each completed the same three burned H500 worlds:

| Mix | Burned seed | Decisions per arm | Dense batches |
|---|---:|---:|---:|
| frozen | 17222702 | 500 | 0 |
| mixed | 2901769088 | 500 | 0 |
| scripted | 3376427795 | 500 | 0 |

Serial (one worker): **9.285197542 s**. Parallel (three workers):
**4.595299708 s**. Numerical pools were capped at one, CPU only, `nice -n 10`.
After excluding timing fields only, canonical paired-row bytes match:
`9c2596dc4e2d3af0b73e1cb8ede9e745cb0070699b8cb595f74b5587158c852e`.
H500 fixtures do not reach L600, so these times are not dense-search forecasts.

A separate burned frozen-world probe naturally reached L600 at decision 2372.
The actual unchanged trigger evaluated all six H300 branches; all survived,
bare/chosen action both 0, batch **3.497740625 s**, total probe **10.415239834 s**.
No selection world was used. This is one functional probe on a contended Mac,
not a matched throughput study or an L4 qualification.

Final receipt:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/final-preflight.json`
SHA256 `3c0ecdd7360fde92745848d3f0ecabbb7cd54c1c4923ac3401c260d310b07e4e`.
It binds the final serial/parallel exports and the independent dense-probe receipt.

## Tests and commands

Final targeted suite: **18 passed in 0.72 s**. Black, isort, flake8,
shell syntax and diff checks passed. The independent reviewer closed all
findings for the validation-only preparation scope.

```bash
nice -n 10 env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q research/bl_served_search
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/bl_served_search
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/bl_served_search
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/bl_served_search
bash -n research/bl_served_search/run_l2.sh
git diff --check
```

Real fixture commands, with the same explicit CPU/thread/nice environment:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.ceiling \
  --fixture --bank research/bl_served_search/BANK_L2.json \
  --b8-path /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  --checkpoint-dir /Users/josenunez/Projects/ml/snake-dqn/saved_snakes --workers 1 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/final-preflight2-serial/fixture.json
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.bl_served_search.ceiling \
  --fixture --bank research/bl_served_search/BANK_L2.json \
  --b8-path /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  --checkpoint-dir /Users/josenunez/Projects/ml/snake-dqn/saved_snakes --workers 3 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/final-preflight2-parallel/fixture.json
```

These paths are used immutable evidence; rerunning requires fresh output paths.
Preserved development failures: initial missing `math` import; reused output
parent correctly rejected by immutable publication; fresh telemetry parent
initially missing (fixed in final pin with regression); manifest generation 01
refused a short commit spelling before allocation. Generation 02 was schema
valid but superseded by the fresh-output fix; generation 03 is authoritative.

## Manifest and forecast

Authoritative validation-only manifest:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/manifests-03/manifest-l2-validation-only.json`.
Its sibling `validation.json` records `SCHEMA_VALID_NOT_DISPATCHABLE`.
Manifest SHA256:
`5144ad2fa76f7f48d1e362f6940f32cca8fca314568366e4674e1af4d02b636e`.
The final pin is a full published 40-character commit. Inputs are uploaded-file
specifications with hashes, portable rosters and no runtime Drive dependency.
The qualified supervisor executable inventory was checked unchanged.

| Forecast | Allocation minutes | Units at 2 u/h |
|---|---:|---:|
| Historical scripted-only reference floor + 20 min setup/export | 189.675620138 | 6.322520671 |
| Conservative 3× L4 CPU slowdown + 20 min | 529.026860414 | 17.634228680 |
| Absolute envelope | 480 | 16 |
| Usable cap after 20% reserve | 384 | 12.8 |

The conservative forecast exceeds the absolute and usable caps; its required
2× timeout would be 1,058.053720828 minutes. Frozen/mixed long-body costs and
L4 per-core rates remain unmeasured. The manifest's estimate field is explicitly
the historical reference floor, never a claim that the conservative forecast
fits. The hard-false plan blocks both forecast and runtime parity admission.
No manifest in this handoff is authorized for dispatch.

The local launcher was tested against its exact generated input tree in both
`--check` and ordinary modes. Both must refuse at NOT_DISPATCHABLE before
gameplay. The sparse source fallback was built and imported offline; the
validation receipt is under `offline-verification-02/receipt.json` in the
artifact root. Both launcher modes exited 1 with the explicit missing-runtime-
parity refusal and created no gameplay output. The final sparse capsule had
268 files, 4,544,814 unpacked bytes, 1,128,982 compressed bytes, and passed
the isolated sparse import check. No Colab service calls occur in these checks.

## L0 evidence and untouched state

L0 audited 80 LAD + 80 linked N1 records, established 34 rescue-capable and
41 delay-only intervals (five M3 excluded), and found all 80 LAD records
missing the complete evaluation timeline. Its prescribed result is INCOMPLETE;
no k-grid delivery, 50%/80% crossing or k20 scientific kill is inferred.
The audit took 0.038157291 s in one process, one numerical thread, invoked
under `nice -n 10`, actual nice 15 from inherited priority.

`RESULT_L0.md` and `l0_receipt.json` are at the requested artifact root, with
verified `.sha256` and `.complete` sidecars. Receipt SHA256:
`324501f09130eead56d24cd9fa293012ab91e4ae2bd8c040d6265699c892456a`.

Phase 0 and existing tmux jobs were not modified or stopped. Sessions observed:
`colab-limit-test`, `git-sync`, `phase0-selection`, `t34-claima-d0r`.
No released checkpoints, serving pins, forbidden package sources, existing
hash-bound documents or receipts were edited. The sealed study was never opened.
