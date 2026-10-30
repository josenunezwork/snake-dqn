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
and partition. Anchors are hash-ranked by namespace, seed and UID without
replacement; `anchor_worlds <= len(reservoir)` is required. Anchor descendants
remain correlated and do not count toward fresh-L1 mature support. Only this
round's selected fresh-L1 mature roots and their parents satisfy that gate.

Freeze number of reused reservoir roots, how their priorities are selected, the
maximum reservoir root/byte caps, and ordinary BC rehearsal. Freeze rejects when
`len(reservoir) + candidate_roots` exceeds the root cap or its recorded byte
forecast exceeds the byte cap. The policy is append-only, without eviction.
The forecast uses max(existing serialized root size, maturity-sized body/food row
plus 64 KiB allowance); exact union sizing runs after collection before search/fit
and at export. Forecasts do not guarantee maximum legal body sizes will fit.
Collection is
entirely student-driven; querying never changes its actions. Failed maturation and
deaths remain in the allocation; there is no replacement/retry for scientific failure.

## Search, queries and fitting

- Freeze B8 continuation identity (or the audited student opt-in), H300 (or the expressly registered
  horizon), all legal first actions, exact ties to current student then lowest ID,
  selection suffixes 0/1, independent scoring suffixes 4/5, namespace and opponents.
  Labels bind full root state + continuation + opponents + horizon + config.
  Freeze `search_chunk >= 512` for search and assessment (default 512) separately
  from `branch_batch_size` (default 1024); the final slab may be partial.
- Gate: uniform by default, optional frozen student/B8 disagreement priority, top fraction,
  deterministic independent hash-Bernoulli random floor and gate seed. Record each
  inclusion probability. Every queried root gets full-action search; unqueried
  roots get **no expert target**. Cap maximal work at all candidate roots, not the
  random query count's expectation.
- Freeze ordinary/search batch proportions, IPW estimand, half-batch counts,
  microbatch, learning rate (current implementation Adam 3e-4), updates, checkpoints,
  BC weight, optional soft weight, separate B8-Q `b8_temperature` and search-mass
  `search_temperature`, and centered-advantage weight.
  Uniform sampling over queried roots uses `n_query/(N*p)` with no clipping or
  self-normalization. Ordinary hard CE remains present; Q soft targets are optional.
- Full model/Adam/RNG state continues across rounds and mechanical retries.
  Nominate only the final update. Freeze sparse vs optional integer cache and its
  byte cap. Integer caching is train-only and byte-verified on admission.

## Fresh held-out integrity and look policy

Every round collects fresh assessment from L1 under the frozen pre-round student,
using a separate phase/seed namespace; assessment is never gated or fitted. Search
selection chooses a teacher first action on suffixes 0/1, then scores B8/current/
teacher/candidate first actions on 4/5 under the same frozen continuation (B8 by default).
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

## r50 follow-on defaults and mandatory admission (fix2)

Stopping is **the five-round cap**, with schedule `[1,2,3,4,5]` and
`stopping={"kind":"round-cap","max_rounds":5}`. There is no efficacy-based
"two small lower bounds" early stop. Integrity/support failures still terminate
as INCOMPLETE, without replacing natural worlds or extending the cap.

Collection now reserves disjoint late and ordinary strata: late every `root_every`
(default 32) at length >= `mature_length` (600), ordinary every
`ordinary_root_every` (128), ordinary taking precedence at shared ticks. Each
parent retains separate bounded pools; final selection enforces the
`roots_per_parent` cap per stratum, matching the pilot (at most four total with
the default two per stratum). `late_fraction` defaults to 0.25: 512 late and
1,536 ordinary candidates for the default training population; assessment
reserves at least 64 late roots to meet its 64-parent support gate. Fresh mature support and
independent parents are reserved first; anchors/reused roots cannot satisfy this
support. Insufficient natural support still refuses; stratification cannot make
an immature student reach L600. Freeze quotas with sufficient ordinary capacity.

The default query gate is **uniform**: `query_ranking="uniform"`,
`query_top_fraction=0`, `query_floor=0.34375` (the former expected query rate).
Disagreement ranking requires explicit `query_ranking="disagreement"` and a
prospectively chosen positive top fraction. Search units export every candidate's
UID/parent, disagreement rank/top-quartile flag, inclusion probability, queried
flag, frozen student logits and top-two logit margin. Queried labels also export
CE and CE squared against the hard expert action. Join by UID; compare inverse
probability weighted CE-squared sums for top quartile and whole population. That
ratio is an estimate when queries are sampled, exact only with full labels.
Unqueried roots have no invented targets. Show the retrospective CE-squared share
(and uncertainty, with parent clustering) before adopting a ranked gate.

Search continuation defaults to **B8**, including assessment suffixes. Collection
remains student-driven; first-action ties remain current-student then lowest ID.
Opt-in `continuation_mode="student"` requires `teacher_audit_sha256` and a pinned
`teacher-audit.json` input. Its schema is `spatial-exit-teacher-audit/v1`; bind
namespace, student tensor identity, opponents identity, branch horizon, selection
and scoring suffixes, and independent judge. `pairs` contain unique UID, parent
ID, `student_teacher_gain` and `b8_teacher_gain`, each computed from paired search
and independent scoring on identical roots. Parent-mean gain differences must
have a one-sided t LB90 >= `-teacher_max_degradation` (default zero), with at least
`teacher_audit_min_parents` (64) independent parents. Register and independently
review how that report is produced before using the hook; this build performs no
teacher audit or scientific validation.

Per-label exports include all legal action advantages relative to B8's first
action, selected-action advantage and selection top-two mass gap. Learning options
are off by default: `margin_threshold=0` leaves hard CE unchanged; a positive
threshold scales searched hard CE by `min(1, gap/threshold)`.
`soft_advantage_weight=0` disables optional masked softmax-advantage CE using
`search_temperature`. Existing Q-soft and centered-regression losses remain
separate options. Validate prospectively; these options do not establish benefit.

All search labels and derived actors remain quarantined and labelled
`product_training_eligible=false`. `quarantine_search_targets=true` is the
fail-closed default. Product training refuses searched rows. Disabling the flag
alone is insufficient: an independently verified exact-product-dynamics replay
and its admission implementation are required, and are deliberately not supplied
by this development-only package. These targets do not establish v2 Watch transfer.

Every completed unit uses a `spatial-exit-unit/v2` envelope with actual CUDA
`gpu_sku`, capability, memory, torch/CUDA version; CPU units explicitly record
`gpu_sku=null` and CPU architecture. This metadata is inside the hash-bound unit.
Fit completion resume across differing hardware/software identities refuses;
no cross-SKU bitwise-identity claim is made.

### Concrete ancestry DAG bindings

`prior_round_freezes` is a nonempty list of file descriptors, never strings.
Every descriptor has absolute `path` and exact `sha256`. A round descriptor has
`kind="round"`, points to the actual `FREEZE.json`, and resolves every sibling
config, input, seed-audit and ledger. Historical source files are hash-audited
against the recorded git commit. The bound seed-inventory recursively supplies
its parent descriptors. Exactly one scientific pilot and all same-program rounds
1 through current-minus-one must resolve, without cycles, conflicting allocations
or foreign programs. All derived ancestor effective seeds must be excluded.

The pilot uses `kind="pilot"` pointing to its actual supervisor manifest, plus
`inputs` mapping each manifest destination to a relocated Mac `{path,sha256}`
descriptor. Every manifest input must resolve and match, including `config.json`.
This permits relocation without trusting a truthy name. The Mac seed-audit embeds
the complete resolved file/hash DAG; the runtime verifies that bound report and
all packet inputs without needing access to Mac-only ancestor paths. For later
rounds include the immediate predecessor descriptor directly as well as any
recursive references: the imported parent endpoint's `freeze_sha256` must match
it, and its learner/reservoir hashes must match the imported bytes.
