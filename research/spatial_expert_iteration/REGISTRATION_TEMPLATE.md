# T34 spatial expert-iteration round registration template

**DRAFT — NOT RATIFIED. No scientific round or dispatch is authorized by this template.**
Copy this template and `REGISTRATION.template.json` to a new round-specific, committed
registration. Never edit a registration after its frozen data exist. An independent
Fable judge (authorized stand-in if applicable) ratifies before preparation. A GO
from the spatial pilot is a dependency, not a strict pass or release authorization.

## Question, endpoint and continuation

- Program ID, round index, whole-program round schedule and final-only nomination rule.
- Pilot GO decision path/hash and nominated actor hash; import its final full learner
  via `import_pilot`. Pin model, Adam, sampler RNG, framework RNG and global update.
- For subsequent rounds pin the immediately preceding `summary.json`, final learner
  and reservoir bytes. No best-checkpoint selection after looking at held-out results.
- Comparator: released bare B8 s3, SHA
  `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
- Specify terminal-hero full-horizon mass integral (dead frames zero), horizon,
  roster mixes, survival/product guards, parent-clustered statistics, thresholds,
  INVALID/INCOMPLETE precedence, permitted labels and prospective decision rule.
  This package exports sealed observations; it deliberately has no new efficacy
  decision analyzer. Register and independently review that analyzer before round 1.

## Frozen collection and reservoir

Freeze the complete `RoundConfig` and its SHA. Its `implementation_sha256` must be
`identity(sources(committed_pin, tiny=False))` from `prepare.py`, computed after the
implementation is committed. This binds every resume unit to the entire runtime
source closure; changes require a fresh scientific identity, not reuse of old units. Pin the Mac inventory SHA in
`seed_inventory_sha256` and every runtime input in `input_sha256`. Enumerate all parent/world/seed
allocations and inventories, including every previous round, pilot assessment,
strict/product banks and the canonical historical exclusion registry. Bind a Mac
inventory with `scope=mac-historical-and-prior-rounds`, `excluded_seeds` and
`prior_round_freezes` listing the bound previous freezes. A collision rejects the
entire allocation; do not replace individual worlds.

Freeze fresh-L1 vs training-anchor world counts and horizons, natural maturity
threshold, root/demo schedules, logical reservoir group size, quotas and minimum
mature independent-parent support. The default first round starts fresh; a verified
pilot **training** reservoir may be imported, with explicit input hashes in the
ratified registration. C/T death-lookback or synthetic lengthened states are not
supported scientific inputs. Anchor descendants retain the original parent ID
and partition. Anchor cycling creates correlated descendants, never new parents.

Freeze number of reused reservoir roots, how their priorities are selected, the
maximum reservoir root/byte caps (overflow refuses the endpoint), and ordinary BC rehearsal. Collection is
entirely student-driven; querying never changes its actions. Failed maturation and
deaths remain in the allocation; there is no replacement/retry for scientific failure.

## Search, queries and fitting

- Freeze current student continuation identity, H300 (or the expressly registered
  horizon), all legal first actions, exact ties to current student then lowest ID,
  selection suffixes 0/1, independent scoring suffixes 4/5, namespace and opponents.
  Labels bind full root state + continuation + opponents + horizon + config.
- Gate: preceding frozen current student/B8 disagreement priority, top fraction,
  deterministic independent hash-Bernoulli random floor and gate seed. Record each
  inclusion probability. Every queried root gets full-action search; unqueried
  roots get **no expert target**. Cap maximal work at all candidate roots, not the
  random query count's expectation.
- Freeze ordinary/search batch proportions, IPW estimand, half-batch counts,
  microbatch, learning rate (current implementation Adam 3e-4), updates, checkpoints,
  BC weight, optional soft weight/temperature and centered-advantage weight.
  Uniform sampling over queried roots uses `n_query/(N*p)` with no clipping or
  self-normalization. Ordinary hard CE remains present; Q soft targets are optional.
- Full model/Adam/RNG state continues across rounds and mechanical retries.
  Nominate only the final update. Freeze sparse vs optional integer cache and its
  byte cap. Integer caching is train-only and byte-verified on admission.

## Fresh held-out integrity and look policy

Every round collects fresh assessment from L1 under the frozen pre-round student,
using a separate phase/seed namespace; assessment is never gated or fitted. Search
selection chooses a teacher first action on suffixes 0/1, then scores B8/current/
teacher/candidate first actions on 4/5 under the same pre-round student continuation.
Identical arm jobs share exact results. This mechanism assessment is distinct from
fresh paired **closed-loop candidate vs B8 gameplay**, also newly allocated each
round. Parent descendants stay clustered. Never reuse a sealed baseline as a fresh
candidate comparator.

Fix when each sealed round may be opened and how a round decision controls the
predeclared next round. Held-out data must not tune query/learner settings. Explicitly
state whether intermediate assessments are descriptive and remain sealed until the
program endpoint. Bind the Mac-side integrity/statistical analyzer and independent
review procedure. v1 simulator development evidence does not establish v2 Watch
transfer, Mac strict/product qualification or owner release approval.

## Budgets, freeze, mechanical recovery and transport

Record base and effective **2x** per-job timeout; useful-time estimates and N are
not doubled. Freeze per-round roots, branch horizon, updates, batch sizes, byte caps
and original + at most two outcome-blind completion attempts. Bind empty ledger on
attempt 1. Completion attempts use new output dirs/receipts, skip exactly Mac-verified
units, and resume only the latest full learner snapshot. No automatic retries.

`prepare` validates the committed registration and source closure, all input hashes,
pilot GO nomination, learner lineage, training reservoir, seed audit and ledger,
then creates `FREEZE.json` before any collection. The runtime verifies it before
loading data. PACKET.json is explicitly **NOT DISPATCHABLE**. Before any later Colab
operation, the operator must separately qualify a same-pin canary and prepare a
supervisor manifest/script with the full source/inputs and resume declaration under
`research/colab_cli` and `tools/ops/dispatch.sh`. No Colab dispatch is implemented or
performed in this build task. Strict/product work remains Mac-only; release remains
owner-only.
