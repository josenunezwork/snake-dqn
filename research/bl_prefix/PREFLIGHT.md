# E6/E7 Mac preflight and dispatch preparation

**Built and Mac-preflighted. Scientific jobs NOT_DISPATCHABLE. No Colab calls.**
These are development experiments excluded from every registered campaign verdict.
No scientific E6/E7 bank was collected and no E2 endpoint was inspected.

## Frozen identities

- E6/E7 registrations: `c6d6bb5f`, published before gameplay.
- Collision allocator/terminal-prefix clarification: `9c175c56`.
- High-range seed-family addendum: `67529c3b`, before gameplay.
- Passing seed audit committed: `7387fa84`.
- Roster compatibility addendum: `31c2591d`, before successful world construction.
- Implementation: `4fd60693`; manifests pin published merge
  `264aff4025ff65107004adb84a25f879fac02d89`.
- B8 SHA-256: `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
- Seed audit SHA-256:
  `69d6a44a982e3884f431ccd14d2137bf62e610fa52b30b9d13798bf70f9d1a31`.

The audit binds 11,224 identities: source4096 / future3000 / exploration4000 /
burned preflight128. Historical FRP inventories, generated/published banks and
visible history were checked. The sealed live output tree was pruned before
traversal; published registered bank identities cover it. No sealed outputs were
opened. Opaque/unrecorded external runs remain an explicit coverage limitation.
Whole rejected v1 allocations and failed operational audits are retained outside
Git. The passing v2 world/RNG seeds retain all high bits through the real engine.

The legacy roster materializer validates uint32 labels even though roster choice
uses only allocation position. The package adapter materializes unchanged roster
rotation with ordinals, then supplies the actual audited high-range seeds to the
simulator. Ordinal labels never initialize gameplay. A full-rotation test covers
all three mixes. No protected source was edited to remove the uint32 guard.

## Real entrypoints and controls

All gameplay used `nice -n 10`, CPU, and one native math thread per worker:
`OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=NUMBA_NUM_THREADS=VECLIB_MAXIMUM_THREADS=1`.
The Mac was on AC. Preflight used one worker, then two workers sequentially;
scientific execution uses explicit spawn and exactly twelve workers.

The shared-bank entrypoint built three real B8 fixture worlds, using only the
burned preflight namespace. Candidate source admission is fixed to the first
48 burned seeds, first 528-frame survivor per mix. The selected first three
already survived; no source was skipped in the successful attempt. Roots were
captured at frames128/178/228 with actual128-decision histories. The remaining
burned seeds separately initialise R/F fixture weights, future RNG and E7 sampling.

Each mix's saved/restored root passed **300 independent continuation frames**
against unmodified T3 `build_world` / `choose_and_step`: actions, observation/Q
bytes, full simulator state, RNG and carried opponent state. Total:900 compared
frames. The source's actual128 preceding records were also checked.

| Real workload | One worker | Two workers | Work per run |
| --- | ---: | ---: | --- |
| Shared bank + three independent controls | 16.954828s | — | 3 mixes,9 roots,900 compared frames |
| E6 primary + secondary | 6.971951s | 4.988617s | 36 continuations,576 world-steps |
| E7 search + paired confirmation | 11.150488s | 7.939590s | 9 ancestor roots,117 continuations,1872 scheduled steps |

E6/E7 elapsed numbers are runner times including loading, recurrent history
reconstruction/calibration, and worker startup. Whole-entrypoint wall times for
the repeated runs were5.338477s (E6/two),11.495757s (E7/one),8.306308s (E7/two).
E7/two ran the actual `research/bl_prefix/run.sh` with a local supervised-runtime
fixture; other runs used the same dispatcher and real runner code. No provider
runtime or allocator was involved. R/F were genuine E2-format, zero-residual,
B8-equivalent **preflight fixtures**, never substituted scientific endpoints.
Every result is `PREFLIGHT_ONLY` with `NOT_SCORED_PREFLIGHT` analysis.

Serial/two-worker canonical result bytes were identical:

- E6: `f476f425875c6b9a0dcb0baa6ebb60ba21d69ac93ba71cc3d5c3b0d75cd68e3b`.
- E7: `4f9f3b4cbf2a8fa7fb2311c8e3dae82cb0e1e7c010c43e5b7d5a60652fa79390`.

Actual completed payloads were discovered and hash-checked by the supervisor's
`runtime._exports`. Snapshot, prefix, metadata, result and bank-bundle payloads
use create-only `.sha256` and `.complete` sidecars. Deterministic bank archives
are bounded at60MiB per chunk, below the64MiB manifest limit; large working
snapshot trees stay outside the runtime export directory. Bundle round-trip,
changed payload and missing-sidecar rejection tests passed.

Earlier failed attempts remain: bank01 refused the roster's uint32 label check;
bank02/03 exposed an off-by-one that omitted the final requested capture boundary.
The boundary regression is tested. No failed attempt supplied scientific data.

## Verification

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  research/bl_prefix tests/test_colab_cli_supervisor.py -k 'bl_prefix or validate'
```

**38 passed,64 deselected in26.15s**. Black100, isort/Black, flake8,
`git diff --check`, and `bash -n research/bl_prefix/run.sh` passed. Independent
read-only review found no remaining material issue. No broad suite was run.

## Forecasts and admission

The conservative extrapolation divides the measured one-worker tiny-entrypoint
time by its scheduled world-steps, assumes L4 cores are3x slower, twelve workers
scale linearly, and adds15min setup/export. It counts bank collection, selected
source replay and every secondary/control continuation omitted by the judgment's
headline counts. These early-root tiny fixtures include disproportionate fixed
loading/burn-in costs and are **not qualified long-body L4 throughput**.

| Experiment | Full work upper bound | Extrapolated allocation | Units at2u/h | Frozen cap |
| --- | ---: | ---: | ---: | ---: |
| E6 | 37.12M world-steps | 1887.098min /31.4516h | 62.9033u | 4u |
| E7 | 24.96M world-steps | 634.472min /10.5745h | 21.1491u | 20u |

Neither fits its frozen cap, even before the mandatory two-times timeout margin.
The caps remain unchanged: E6 .8u reserve /96min usable envelope; E7 4u reserve /
480min usable envelope. The runtime supervisor applies first-cap-wins to the
whole allocation, including collection, setup and export. No automatic retries.

The two manifest files are **blocked cap-envelope templates**. Local supervisor
structural validation passed, with `--colab /usr/bin/false` proving validation
needed no CLI call. Their nominal `est_minutes` describes the reservation
envelope, explicitly not the measured scientific forecast. `ready:false` makes
the dispatcher refuse before any gameplay. They are not scientific admissions.
A ready scientific manifest additionally checks actual bound inputs, final R/F
checkpoint/critic identities, audited RNG sets, twelve-worker assumptions,
bundle delivery inventory, and a recomputed fitting forecast before allocation.

E6 also awaits actual final R/F endpoints and exported critic receipts containing
`complete:true`, `training_complete:true`, `checkpoint_role:"final"`, matching
arm/checkpoint SHA, >=2M refit transitions, finite late-body EV>=.1 and lower
late-body MSE than the time/length baseline. E7 awaits the complete scientific
E6 bank and verified checkpoint relocation map; this preflight bank is excluded.

Concrete paths, manifest hashes, local-only commands, blockers and forecasts:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e6e7/DISPATCH.txt`.
No Colab calls, sessions, tmux jobs, released checkpoint writes, serving changes,
or registered T34 edits were performed by this work.
