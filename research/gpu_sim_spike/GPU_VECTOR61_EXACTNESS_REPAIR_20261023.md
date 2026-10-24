# GPU vector61 exactness repair — prospective mechanical canary

Owner-authorized 2026-10-23. Prepared engineering evidence only; operator dispatches.
Binding: round-28 JUDGMENT §7 and
`snake-dqn-artifacts/big-lever/exp/JUDGMENT_GPU_PARITY_STANDARD_20261023.md`.
This is a declared CPU-correctness repair, including the dense device baseline,
not a pure optimization or a waiver of its CPU contract. Original `42c84584`
INVALID evidence stays immutable. No native pass is claimed by Mac execution.

## Discrepancy and causal scope frozen before native data

Baseline pin `42c84584d5edef40009ad7278cb912b7f8bd4a9a`; failed fixture
`gpu-engine-sprint-20261023-02/sparse-cap160/canary-output/exports/recorded-states/scripted-static-e1024.npz`
under the external big-lever/exp root has SHA-256
`68e182b29e66a564743053dcc3c41b3fc0a8f55b2294a747c336a58a220028d2`.
Both reported outputs are float32; CPU reference is
`src/simd_env/vector61_featurizer.py` (float64 intermediates, final float32 cast).
Inputs preserve full bodies/rings, ordered food, directions, lengths, alive flags
and recorded nearest-enemy carry. These feature-only probes have no actions,
world transition or RNG draws; draw consumption is zero.

* Dense row 0 column 39: danger sector 15. The maximizing body point has pixel
  delta (-200,80), squared distance 46,400, distance 215.40659228538016,
  radius 300 and factor 1.5. There are 21 positive contributing points.
  CPU computes `(1 - sqrt(46400)/300)*1.5 + 0.2` in ordered float64;
  0.6229670385730992 casts to bits 1059027653. Old CUDA returned
  1059027652. Dense used float32 cell-distance sqrt/division/subtraction,
  multiplication and bonus addition. The evidence localizes this to the
  floating-point chain; it does not identify one contracted machine instruction.
* Static row 1 column 8: food sector 0 has count 11; expected count per sector is
  300/16 = 18.75. CPU float64 `11/18.75` casts to 1058418633.
  CUDA returned 1058418634, exactly reproduced by float32
  `11 * float32(1/18.75)`. The old static torch density reduction/division
  remained float32 even when distance intermediates were float64.

Declared affected fields: feature columns 4–46, 50–52 and 54–60 (normalization,
geometry, sector classification and dead-row positive zeros); nearest-enemy
`prev_dist` is now float64 pixels for both arms. Sector LUT changes may also
change which enemy/food geometric value is selected only within these columns;
nearest-enemy stable slot ties remain CPU ordered. Column 49 trend is causally
affected by repaired pixel-distance carry. Dead-row canonicalization covers
all 61 columns. Live discrete columns 0–3, 47–48, 53 and 57 retain CPU semantics.
All simulator tensor fields and runtime lifecycle/carry buffers outside the
explicit trend update remain unchanged by feature evaluation. Subsequent
closed-loop changes are under this new candidate identity, not baseline parity.
No reward, food/corpse ordering, mechanics, RNG, policy weights or serving repair
is included; those broader engine proof obligations remain separate.

## Operation inventory and repair

| Outputs | Integer work / ordered floating-point work |
|---|---|
| 0–3, 47–49, 53, 57 | One-hot/heading, integer comparisons and stable slot order; trend compares CPU float64 pixel distances; kill-distance threshold uses integer d2 plus RN double sqrt |
| 4 | logical length integer, RN float64 division, min, one final cast |
| 5–7 | first minimum ordered food d2 in integer pixels; RN double sqrt and coordinate/diagonal divisions; final cast |
| 8–23 | integer sector counts, CPU sector LUT, separately RN `food_capacity/sectors`, RN count/expected, min, cast |
| 24–39 | integer pixel d2 and multiplicity; RN sqrt/div/sub/mul, max; integer count, RN count/5 then times .2 then add and caps; final cast |
| 40–46, 50–52 | integer pixel deltas/logical lengths; ordered RN double divisions and ratio cap/divide-by-2; final cast |
| 54–56 | integer next cells and obstacle d2, RN sqrt, min, ordered RN `1-nearest/max_check`, cap, cast |
| 58–60 | integer occupancy/BFS/count/cap, indexed finite CPU float64-ratio-to-float32 table; no CUDA float division |

Both dense and static CUDA paths call the same native feature58 implementation.
It uses `__dadd_rn`, `__dsub_rn`, `__dmul_rn`, `__ddiv_rn`, `__dsqrt_rn`;
no torch CUDA floating-point geometry/division remains in this path.
Native builds use `--fmad=false --prec-div=true --prec-sqrt=true`, no fast math.
TF32, fp16/bf16 reduced-precision reductions are disabled and matmul precision
is highest. This does not establish CPU/GPU network Q/action identity.

`vector_exactness_probe` replays the failed full-capacity state through direct
and indexed runtimes, dense and static, checking all feature bytes, float64
trend carry and unchanged simulator tensors. The canary uploads the immutable
fixture by digest and must export `vector-exactness.json`. The normal all-roster
E1024 canary also requires independent CPU feature bytes from both backends.
The GPU comparison reports mismatch count and first row/raw bits for every
column, including signed zero. Neither a tolerance nor a baseline discrepancy
allowlist is used. Any mismatch is terminal INVALID.

## Frozen package and boundaries

Use a new published source pin, detached frozen checkout, job IDs, artifact root,
ledgers, outputs and state directories. The handoff records the exact identities
and hashes. Mac real Bash preflight and independent Codex-V LIGHT precede dispatch.
Canary remains 24 minutes/4 units (2/3 reserve), A100 >=12 vCPU, <=10 u/hour,
64 MiB exports. Main remains a template pending genuine same-pin canary PASS,
verified exports and release; 192 minutes/32 units. Shared series envelope remains
216 minutes/36 units including 6 reserve across attempts, never a fresh allowance.
Existing Q2/global balances, deadlines and the two-parity-INVALID rule still apply;
operator accounts for the old failure before dispatch. No automatic retry.

No labels, training, strict/product qualification, policy claim or release follows.
