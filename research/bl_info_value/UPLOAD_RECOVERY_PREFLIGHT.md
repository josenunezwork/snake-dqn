# E3 upload recovery preflight — 2026-10-21

Outcome: **SMOKE_COMPLETE / PREPARED, NOT DISPATCHED**. Development only;
no scientific GO/KILL, strict-gate evidence, serving change, or Colab calls.
This sibling supersedes only the upload/dispatch instructions in PREFLIGHT.md.
PROTOCOL.md, students.py and fit.py are unchanged.

The failed attempt receipt in `big-lever/exp/bl-e3/colab-output-01/receipt.json`
records INCOMPLETE, no terminal job output, and `stop_verified=true`. The
57,831,386-byte input archive exhausted the supervisor's **shared 300-second
initial upload deadline**. Bootstrap/source-helper transfers took about 59.15 s;
the input failed after 239.88 s. Splitting uploads cannot reset that deadline.
The elapsed session was 334.0868 s: at the registered 2 u/h ceiling its charge is
0.18560 u. The recovery reserves 0.2 u for this failure, leaving a 12.6 u session
cap plus the unchanged 3.2 u reserve inside the original 16 u experiment cap.

## Reconstruction and checkpoint identity

`compact.py` consumes the verified original v1 archive, never a new bank. The
v2 upload contains only its catalog: original world seeds, rosters, source
indices, root metadata, death controls and split assignments, plus the dtype,
shape and SHA-256 of each selected saved local/global/scalar/mask array. Replay
regenerates these arrays and verifies their exact bytes before generating any
D300 branch target. The original actions, frames, classes and controls remain
binding. No root is replaced after a reconstruction failure.

The ten exact checkpoint payloads move to a separate immutable GitHub input
asset. The existing supervisor public-input path verifies the archive size/hash
on the Mac before allocation and again on runtime. `compact.hydrate` also checks
archive size/hash, its exact unique member set and every member SHA-256 before
replay. The public input has a separate 300 s fetch budget; it is not uploaded
through Contents. This GitHub data asset is not a model/serving release.

Production contents: **256 worlds / 1,536 roots**, unchanged 192/32/32 world
splits, 768 roots/class, D300, 128-decision histories, A/B/C 270,522 parameters
each, 60 epochs and the frozen decision rule.

| Payload | Bytes / binding |
| --- | --- |
| Original archive | 57,831,386 |
| Compact Contents input | **360,881** (99.376% smaller) |
| Public checkpoint archive | 51,787,546 |
| Compact SHA-256 | `54eaa6a8f62fd427252d6468df720123e5c47a7c481b55e838c358be58a68fbc` |
| Checkpoint archive SHA-256 | `dfdef34f99aa73a21f5b9b3a23a7fd27ab4c8a4c388e5a73bfc4b5b13b520c37` |
| Frozen protocol SHA-256 | `4e84360fc1686ed4b9ad937f89fd1a72c9df8a8ef89c6c1e71f75283f9e8037d` |

Public asset:
<https://github.com/josenunezwork/snake-dqn/releases/download/bl-e3-inputs-20261021/checkpoints.tar.gz>.
It was fetched and size/hash-verified locally in 1.86235 s; this is Mac GitHub
network evidence, not a measurement of Colab transfer speed.

There is **no guaranteed safe byte limit** in the supervisor: time, including
RPCs/renewals and source fallback, governs. `manifests.py` now applies an E3-only
**1 MiB local input guard**. At the older documented 13 KB/s Contents rate this
leaves substantial room for helpers/RPCs; the actual compact upload is only
352.4 KiB. The source fallback is separately inventoried in the generated source
plan and must remain small. The old 39,691,705-byte smoke archive was also too
large for a conservative upload plan. No supervisor setting was changed.

## Mac verification

Real `job.sh`, `nice -n 10` (observed nice15 inherited from parent nice5), CPU,
one native math thread/process; serial replay followed by two-worker byte
parity, then two-epoch smoke fits. AC power/lid open, no thermal warning. Existing
`phase0-selection`, `bl-e4`, and `git-sync` tmux sessions were observed only.

| Measurement | Seconds |
| --- | ---: |
| Serial reconstruction, 3 worlds / 6 roots | 37.016180 |
| Two-worker replay parity | 23.275079 |
| Fits and batch benchmarks | 2.074325 |
| Whole entrypoint | 63.105532 |

All 13 dataset arrays (dtype, shape and bytes) equal the original
`preflight-01/dataset.npz`, including returns, geometry and histories. A separate
full-input audit checked **all 256 original task objects and all 1,536 selected
source rows** against the old archive, including every retained non-observation
field and independently computed observation hash. This is not a full production
replay. Every production root must still pass its runtime hash/control checks;
Mac/x86 float32 ties remain a fail-closed risk, without a waiver or changed policy.

The six-process forecast is **2.624129 h / 5.248258 u**: 2.206822 h replay,
0.083973 h fit CPU upper projection, 20 min setup/export. It retains the original
4x slower host assumption and credits no CUDA speedup. No L4 speed was measured.
Timeout **315 min** is at least twice the 157.447725 min estimate and costs at
most 10.5 u at the registered rate, below the remaining 12.6 u cap. Lease300 s,
L4 only, six single-thread workers, max export256 MiB. First wall/unit cap wins.
Unknown/current assignment prices still require operator review; no retries.

Targeted checks passed: **28 E3 tests**, **10 supervisor-validation/public-input
checks** (selected as part of the 36-pass/61-deselected run before two further
hydration refusal tests were added), Black/isort/flake8 and `bash -n job.sh`.
Independent read-only review found no remaining actionable defect.

Artifacts under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e3/`:
`full-input-02/{binding-audit,public-input-verification}.json`,
`preflight-02/{timing,terminal,legacy-parity}.json`, `preflight-02.log`.
The fresh `manifests-03/job.json` pins the published recovery implementation;
job/output attempt02 is unused. Final hashes and absolute local-validation and
operator dispatch commands live in **`recovery-02/DISPATCH.txt`**. The original
`DISPATCH.txt`, prior manifests and failed receipt remain intact as history.

No protected campaign/transport paths, released checkpoints, serving pins or
existing tmux jobs were edited. The embargoed artifact tree was never opened.
