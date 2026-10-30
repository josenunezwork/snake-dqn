# Spatial expert iteration (prospective T34 follow-on)

Build-only package. The concurrent pilot package is imported and never modified.
A scientific round requires its GO, a new independently ratified registration,
committed source/input hashes, a frozen allocation and Mac collision inventory.
Defaults are proposals. No round is ratified, dispatched, merged or released here.

## Components

| Module | Responsibility |
|---|---|
| `config.py` | Immutable per-round science/work/learning settings and separate smoke namespace; structural pilot UnitStore adapter |
| `data.py` | Frozen student drives fresh and training-anchor worlds; original-parent ancestry; batched sparse admission; natural mature reservoir; IPW query gate |
| `search.py` | Full legal-action search under frozen current student, versioned label keys, restore-once device fan-out, independent suffix assessment and exact duplicate job sharing |
| `fit.py` | Ordinary hard BC anchor + IPW queried CE; optional legal-action soft targets and centered advantage Huber; model/Adam/sampler/RNG continuation |
| `cache.py` | Optional train-only packed binary/uint16 raster cache; byte identity admission and bounded memory |
| `import_pilot.py` | Verified pilot final learner/Adam/RNG and natural training reservoir import after GO |
| `prepare.py` | Committed registration/source/input admission, pilot GO/lineage, seed audit, create-only freeze and non-dispatchable packet |
| `resume.py` | RoundConfig adapter to the pilot's outcome-blind Mac ledger builder |
| `run.py` | One round's collect → query/search → warm fit → fresh assessment/gameplay → sealed final endpoint |

Use one packet per round. A later packet consumes the preceding final full learner,
immutable reservoir and endpoint, not a selected intermediate checkpoint. Search
units retain a gate audit and labels only for queried candidates. Collection and
assessment units use the same deterministic IDs/publication envelope, SHA sidecars,
completion-last markers, accounting and original+two-completion cap as the pilot.
The `collect-NNNN` unit contains a training batch and corresponding fresh assessment
batch. Logical reservoir grouping remains independent of physics batch size.

The gate uses frozen current-student vs B8 disagreement, then hash-random ranking;
query the top fraction with probability 1 and all remaining candidates with the
registered positive random floor. The training target is the full candidate mean:
uniform queried sampling needs `n_query/(N*p)`, not just `1/p` with an empirical
queried-mean denominator. No unqueried action becomes an expert target. Ordinary
BC uses recorded B8 Q, and search rows retain all selection per-action/suffix mass
returns plus their mean teacher Q. Soft weights/temperature and normalized centered
advantage weights are explicit settings. No critic leaf is used in search.

## Tiny real-entrypoint Mac smoke

From this worktree, with the primary checkout's venv and canonical checkpoints:

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.spatial_expert_iteration.run --smoke \
  --checkpoint-dir /Users/josenunez/Projects/ml/snake-dqn/saved_snakes \
  --output /absolute/fresh/mac-smoke
```

This performs a real four-world/four-tick student collection, two queried training
roots, two learner updates, independent fresh assessment and paired gameplay using
canonical B8/frozen opponents. L1 is the smoke-only mature threshold; it proves no
mature support or efficacy. It automatically creates a smoke freeze before data,
then writes immutable units and a `SMOKE_COMPLETE_NO_SCIENTIFIC_RESULT` summary.

## Preparing a registered round (after pilot GO)

1. Import the GO-nominated pilot actor's full final verified trajectory and natural
   training roots with `python -m research.spatial_expert_iteration.import_pilot
   --pilot-go ... --actor ... --verified-ledger ... --input-dir ... --output .../learner.pt
   --reservoir-output .../reservoir.pt`. The ledger must be built by the existing
   pilot Mac verifier. This preserves Adam and RNG. Later rounds use `learner.pt`
   and `reservoir.pt` from the preceding sealed round endpoint directly.
2. Copy canonical teachers into an input directory under the pilot's destination
   names (`b8-s3.pth`, `frp3-s12.pth`, `champion.pth`). Include `pilot-go.json` and
   `initial-actor.pth` for round 1; subsequent rounds include `parent-endpoint.json`.
   The original pilot GO remains a required dependency on every round.
3. Copy `ROUND_CONFIG.template.json` and freeze all settings. After implementation
   is committed, compute the source identity with the primary venv:
   `python -c 'import subprocess; from research.spatial_expert_iteration.prepare import sources;
   from research.spatial_expert_iteration.config import identity;
   print(identity(sources(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(), tiny=False)))'`.
   Put it in `implementation_sha256` before hashing the complete config.
   Complete `REGISTRATION_TEMPLATE.md` and `REGISTRATION.template.json`; ratify and
   commit a new registration. Bind the complete config, every input hash, decision
   rules, look policy, seed inventories and base/effective envelopes. Have the
   independent judge review the science and Argon review Astra's implementation.
4. `python -m research.spatial_expert_iteration.prepare --config ... --registration
   /repo/path/to/committed/round-registration.json --inventory ... --input-dir ...
   --packet /absolute/fresh/packet`. Empty ledger is automatic on the first attempt;
   later attempts pass `--ledger ... --resume-roots ...` and copied verified units
   become packet inputs. New attempts use exactly the same scientific identities.
5. A local prepared runtime is `python -m research.spatial_expert_iteration.run
   --packet /absolute/packet --output /absolute/fresh/attempt --device cpu`. This
   allocates no remote runtime. PACKET.json is a preparation inventory, **not** a
   supervisor manifest. Colab transport/canary qualification remains separate.
6. Build a completion ledger on the Mac with `python -m research.spatial_expert_iteration.resume
   --config ... --roots ... --receipts ... --output /absolute/new/verified-ledger.json`.
   Scientific units require genuine stopped supervisor receipts under the reused
   pilot accounting contract. Local smoke uses its separate non-scientific mode.

## Deliberate boundaries

The implementation exports outcomes without a scientific GO/KILL analyzer. Register
that analyzer and thresholds before round 1. It does not dispatch, perform efficacy
looks, select checkpoints, release serving, or alter the current pilot. Held-out
assessment and fresh closed-loop gameplay are different measurements. Natural
reservoir anchors shift occupancy and retain parent correlation; fresh student
worlds remain a separately frozen stratum.

Patch-only decoding (ideas 01.3/09.4), lane refill, asynchronous streams and TF32 are
not enabled. They require independent numerical/performance qualification and
prospective settings. Dense fp32 pilot architecture/inference is retained; optional
integer caching addresses the exact training-input reuse direction. Neither CPU
smoke nor tiny parity fixtures qualifies CUDA batch numerics, throughput, fresh
L600 support or v2 Watch transfer.
