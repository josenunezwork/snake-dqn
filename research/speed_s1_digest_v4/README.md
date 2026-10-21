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
per-call content fragments, with active-cycle context in the memo key.
Environment MT state is packed once, retaining the MT index and Gaussian cache;
non-RNG attributes of its wrapper remain encoded. Modules, callables, opaque
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

Measured receipts and final commands are recorded below after verification.
