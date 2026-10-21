# S1: opt-in flat full-state digest v4

This is excluded engineering work under
[`JUDGMENT.md` S1 and D2](../../docs/research/advice/2026-10-20_speed_vs_constraints/JUDGMENT.md).
It does not qualify a controller, amend a study, or change the default digest.
The unchanged v3 oracle is `research/t34_shadow/shadow.py` at
`d62070e51759ccf8bfbd3ee38c947132777aa392`, SHA-256
`9346b8ff447dd7322c78b0f87aa01e7603870e95b6a394e92104889880019058`.

## Encoding and opt-in seam

`digest.DigestV4` is a lane-local callable, with the v3 keyword arguments:

```python
from research.speed_s1_digest_v4.digest import DigestV4, FULL_DIGEST_SCHEMA

digest = DigestV4(checkpoint="immutable-checkpoint-sha256")
state_hash = digest(world, global_rng_state=isolated_rng_state)
assert FULL_DIGEST_SCHEMA == "t34-shadow-full-digest/v4"
```

V4 emits typed, length-delimited records in fixed order: live head-to-tail
bodies, food/corpse cells with environment boundaries, simulator fields in
lexical name order, environment RNG streams and wrapper metadata, runtime,
hero/opponent carry, checkpoint binding, and global MT state. Body-array shape
records delimit bodies; unused ring slots remain excluded. Lists retain order;
cell sets sort canonically. Every simulator field from v3 remains covered,
including padded grids, interior views, food mirrors and their unused tails,
optional lifecycle fields and extension fields.

Bulk arrays include dtype, dimensions, shape and C-order bytes. Large arrays
use SHA-256 content fragments; aliases share work only within one call. Equal
independent copies produce equal fragments. Runtime objects likewise use
per-call content fragments, with active-cycle context in the memo key and strong
references preventing temporary-object ID reuse. Stored object/simulator fields
and stored policy runtime values follow v3's dictionary traversal, including
values hidden by custom attribute getters.
Environment MT state is packed once, retaining the MT index and Gaussian cache;
non-RNG attributes of its wrapper remain encoded. Stored and observed RNG
streams both remain represented through per-call content references, preserving
mutations even when a getter exposes a different stream. Modules, callables, opaque
objects, tensors and nested metadata retain v3's semantic conventions. Timer
and address-derived cache exclusions follow v3.

Only exact, declared frozen dataclasses with deeply immutable contents are
cached between calls; undecorated subclasses and mutable children do not
qualify. Immutable config replacements get new fragments. Immutable checkpoint
overrides are cached; mutable overrides are encoded each call. A checkpoint
passed to the constructor is explicitly captured by value. Persistent caches
contain immutable fragments and layout/name metadata; mutable array/object
content memos are cleared each call. Instances are not shared across threads.

`full_digest_v4` is a stateless convenience function. Use `DigestV4` per lane for
static-fragment reuse. V3 and v4 digest strings are deliberately incomparable;
compare equality/divergence decisions and unchanged portable exports instead.

`validate.load_private_v4_modules()` exposes an excluded development seam for
S0: it loads private, source-pinned copies of shadow and teacher, redirects only
their digest references, and leaves normal imported modules at v3. The private
M1 identity loader uses the same isolated shadow. No environment variable or
automatic import enables v4 in a registered study.

## D2 acceptance and kill rule

The implementation target is approximately 4x digest speed and 1.36x whole-step
speed; these are targets, not measured claims. Adoption requires **all** of:

1. Byte-identical exports and sidecars on the S0 fixtures: three mixes, short
   and >=800-cell bodies, deaths/respawns, cold reforks.
2. At least 15,000 M1 live/Grid/shadow frames with zero physical, carry, v8,
   action or shadow differences.
3. Every mutation in `tests/test_t34_fast_digest.py` flips v4. The new test
   extracts the existing mutation assignment rather than maintaining a copy.
4. At least **1.25x matched whole-workload gain on the same host**, using the
   S0 ruler. An isolated digest ratio is insufficient.

The first recorded implementation start is **2026-10-20 21:32:28 PDT**
(`2026-10-21T04:32:28Z`); the D2 deadline is **2026-10-22 21:32:28 PDT**
(`2026-10-23T04:32:28Z`). At that deadline, incomplete or failing evidence means
retain v3 and close S1. No second attempt without a new profile. V3 remains the
default throughout review; a future study may opt in only at its registration.
There is no automatic release or adoption path here.

The local `validate bench` diagnostic is explicitly **not the S0/D2 adoption
receipt**. It supplies a matched three-round teacher comparison and byte
checks on excluded short/long fixtures, counting setup, cloning, search,
stepping and export while excluding JIT/import warmup. Death/respawn fixture
coverage must be established by S0; incomplete coverage must remain visible.
The host is contended by the owner's six-worker Phase 0 study, so these numbers
are not quiet-host timing/serving qualification.

## Reproduction

Run from `/Users/josenunez/Projects/ml/snake-dqn-speed-s1` with the primary
checkout's interpreter. Numerical threads are one; pytest workers are two;
every compute command runs at nice 10. Use fresh output paths. The validator
checks AC/lid and refuses identified quiet retention sessions. No Colab units
are spent by this lane; S0 owns its independent L4 run.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 10 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 2 tests/test_speed_s1_digest_v4.py tests/test_speed_s1_validation.py tests/test_t34_fast_digest.py tests/test_t34_shadow.py
```

Verification at implementation commit `2495ec0e6335153a2a64c73246a294d2cb3d6198`:
**92 passed** with the pytest command above. Black, isort and flake8 pass on all
new Python files. Repository `make lint` also passes (149 existing files checked
by Black). The independent reviewer found no remaining actionable findings.

Exact additional checks:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --line-length 100 research/speed_s1_digest_v4 tests/test_speed_s1_digest_v4.py tests/test_speed_s1_validation.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --check-only --profile black research/speed_s1_digest_v4 tests/test_speed_s1_digest_v4.py tests/test_speed_s1_validation.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/speed_s1_digest_v4 tests/test_speed_s1_digest_v4.py tests/test_speed_s1_validation.py
BLACK_NUM_WORKERS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 10 make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint
```

## Measured benchmark (excluded, contended Mac)

Receipt:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/speed-s1-digest-v4-20261021/bench-02.json`
(SHA-256 `0cde6f5df88c6947ae9e75c454f987399ac6cef3ce645a80a4f68490f2f358e7`).
Source hashes remained unchanged throughout the run. All 18 timed pairs had
byte-identical payloads and checksum/completion sidecars (54 files per arm).
This final benchmark ran alongside the one-core M1 identity diagnostic, keeping
the S1 lane at two cores. Neither run is a quiet-host pacing qualification.
Both arms executed exactly 14,598 active row steps over 18 completed excluded
fixture worlds; each fixture used 256 root frames and shadow depth 300.
Warmup/import/JIT work was excluded, while fresh teacher/world construction,
cloning, stepping and export remained inside each timed workload.

| Fixture | v3 wall s, 3 runs | v4 wall s, 3 runs | v3/v4 |
|---|---:|---:|---:|
| short, scripted | 8.969 | 7.235 | 1.240x |
| short, frozen | 14.335 | 12.528 | 1.144x |
| short, mixed | 11.717 | 10.230 | 1.145x |
| long, scripted | 10.401 | 8.629 | 1.205x |
| long, frozen | 30.206 | 28.558 | 1.058x |
| long, mixed | 22.864 | 20.344 | 1.124x |
| **Total** | **98.492** | **87.523** | **1.125x** |

The three complete paired-round gains were **1.1371x, 1.1179x, 1.1215x**;
their median is **1.1215x**. Aggregate active-row cost was **6.747 ms ->
5.996 ms**, including the workload costs described above. These are excluded
256-frame fixture numbers, not campaign H5000 worlds/hour or serving latency.

**The measured whole-workload gain fails the D2 >=1.25x bar and misses the
~1.36x target. V3 remains the default; v4 is not adopted.** Full S0 fixture
coverage (including forced deaths/respawns) remains with the independent S0
owner. The 48-hour kill rule above remains binding; no further optimization
attempt is justified without the required profile.

Benchmark command actually run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib nice -n 10 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.speed_s1_digest_v4.validate --mode bench --checkpoint /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth --frames 256 --seconds 900 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/speed-s1-digest-v4-20261021/bench-02.json
```

## M1 live/Grid/shadow identity

Receipt:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/speed-s1-digest-v4-20261021/identity-02.json`
(SHA-256 `13084145f7dd82987f02efdc61bd005b57992550ba0ab81684530b3ed8cdaf33`).
The corrected candidate at `2495ec0e6335153a2a64c73246a294d2cb3d6198`
completed in **415.730 seconds**, with unchanged source hashes throughout.
Candidate digest source SHA-256 is
`46423e2c8ad6ced990d8963b3e05bfae219253d5b7863bba741c7f5f93098fe4`.

| Mix | Frames | Physical diffs | Carry diffs | v8 diffs | Action diffs | Shadow diffs | Confirmed predictions |
|---|---:|---:|---:|---:|---:|---:|---:|
| scripted | 5,000 | 0 | 0 | 0 | 0 | 0 | 4,999 |
| frozen | 5,000 | 0 | 0 | 0 | 0 | 0 | 4,998 |
| mixed | 5,000 | 0 | 0 | 0 | 0 | 0 | 5,000 |
| **Total** | **15,000** | **0** | **0** | **0** | **0** | **0** | **14,997** |

The established M1 loop compares real offline Watch chronology against an
independent seeded Grid world and a rolling v4 shadow. All three initial
physical states matched. Frames without a pending hero prediction are not
counted as confirmations; physical/carry/v8/action comparisons still run.
This is same-host identity evidence, not cross-platform, paced, strict-gate,
or serving qualification.

Identity command actually run:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib nice -n 10 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.speed_s1_digest_v4.validate --mode identity --identity-frames 5000 --seconds 1200 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/speed-s1-digest-v4-20261021/identity-02.json
```

No S0/S2 files, v8 code, existing shadow/teacher files, ratified study files,
Colab tooling, released checkpoints, serving pins or existing tmux sessions
were changed. No Colab assignment was created. Only this package and its two
new test files belong to the branch. The earlier `bench-01.json` and
`identity-01.json` artifacts remain immutable historical receipts for
`04d7186f6e943f883cd56e598f4dc296b547a784`; final results above use the
corrected code and `*-02.json` receipts.
