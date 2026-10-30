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
| `search.py` | Full legal-action search under frozen B8 (audited student opt-in), versioned label keys, restore-once device fan-out, independent suffix assessment and exact duplicate job sharing |
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

Registered search and assessment units require `search_chunk >= 512` (default 512).
At roughly 34% queried, four legal actions and two suffixes, a 512-root training
unit produces roughly 1,400 jobs, filling a 1,024-job slab before the unit tail.
All 128 default assessment roots stay in one unit. Final partial slabs remain
possible; no support is added or duplicated to fill them. Tiny fixtures are exempt.

Freeze refuses `len(reservoir) + candidate_roots > reservoir_max_roots`, an anchor
or reuse quota larger than the reservoir, or a byte forecast above the cap, before
creating the packet. The append-only policy never evicts parents. The byte forecast
reserves the larger of the largest serialized existing row and a maturity-sized
six-snake snapshot/compact row plus grid food and 64 KiB metadata per new candidate.
This is a forecast, not a maximum-body guarantee; exact serialized union sizing
also runs immediately after collection, before search/fitting, and at export.
The forecast and method are recorded in `FREEZE.json`.

Anchors are hash-ranked by round namespace, seed and UID, without replacement;
input order/age has no selection priority. Mature-support gates count only selected
fresh-L1 roots collected this round, excluding reused roots and anchor descendants.
Per-batch support checks refuse only when even perfect remaining worlds cannot
supply the registered counts; they do not refill or replace worlds.

The gate defaults to uniform hash-Bernoulli sampling at 0.34375, with no priority
top fraction. Frozen student/B8 disagreement ranking is an explicit opt-in.
Every candidate exports its disagreement rank and frozen logits; queried labels
export CE squared and inclusion probabilities for a later IPW top-quartile audit. The training target is the full candidate mean:
uniform queried sampling needs `n_query/(N*p)`, not just `1/p` with an empirical
queried-mean denominator. No unqueried action becomes an expert target. Ordinary
BC uses recorded B8 Q, and search rows retain all selection per-action/suffix mass
returns plus their mean teacher Q. Soft weight, separate `b8_temperature` and
`search_temperature`, and normalized centered
advantage weights are explicit settings. No critic leaf is used in search. Late/ordinary strata protect natural mature
support from ordinary-root eviction. Search targets and derived actors are
quarantined from product training pending exact-product replay. `guards.py`
admits an independently produced paired teacher-degradation audit for student
continuation; B8 is the default. `provenance.py` resolves/hash-audits the ancestor
DAG. `units.py` records actual GPU SKU within every immutable unit. The program
stops at five rounds. See the r50 section of `REGISTRATION_TEMPLATE.md` for exact
schemas, target options, support rules and replay boundaries.

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
   --reservoir-output .../reservoir.pt --mature-length <round-config.mature_length>`.
   The ledger must be built by the existing
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
