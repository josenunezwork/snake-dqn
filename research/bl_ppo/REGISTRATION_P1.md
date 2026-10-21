# P1: one fixed 100M PPO endpoint — prospective registration

Status: CONDITIONALLY_PRE_RATIFIED, BEFORE P1 DATA, 2026-10-21.
Authority: owner implementation request; `big-lever/PLAN_v2.md` §1 WP-3,
§3 D1′ and §8; `round-11/JUDGMENT.md` §(f) P1. Owner → Sol-R;
independent reviewer: Codex-V; funding ruling: Fable judge. PLAN_v2 §1
governs the nomination filter where the judgment's advisor-summary row differs.
This is a new Tier-1 research registration, not an E2 admission or an E2 PASS.
No Colab calls are authorized to Sol-R; Claude dispatches. No release authority.

## Frozen identity and hypothesis

Exactly one feedforward F artifact, training seed **20261021**, **100,000,000
actor hero transitions**. No replacement seed, learning-rate change, endpoint
shrink, extension, intermediate-checkpoint selection, or best-checkpoint search.
Critic-only refit transitions are charged separately and never count toward
100M. Target is immutable; `pause_at` is an execution boundary, not a new target.
The qualification prefix belongs to this same trajectory and is resumed, not
discarded and restarted. Y3 branches are durability controls, excluded from
efficacy. Candidate/reference checkpoints, source closure, this registration,
recipe modules, bank/audit and authority documents are bound by SHA-256 in the
create-only preparation receipt before the corresponding gameplay.

Comparator: released **bare B8 s3**, public catalog identity
`research/t34_exit_b8/PUBLIC_INPUTS.json` (`B8_s3_final.pth`, SHA-256 beginning
`3e649783`). No decision-time hero veto. Question: does this frozen endpoint
increase terminal-hero H5000 full-horizon logical mass integral by at least 40
versus B8 on fresh paired worlds, while retaining the registered guards?

## F recipe and E8 critic gate

Copy the B8 ego2s-bv8 trunk; frozen B8 logits / positive temperature plus a
zero-initialized trainable residual head. Calibrate temperature once on separate
training states to 1% expected non-greedy actions, ties counted as greedy; persist
the fitted value. F is the existing feedforward parameter-matched core, no GRU.
128 worlds, 12 collector processes, rollout 128; one math thread per process;
fixed equal-mix frozen opponents, natural resets, no curriculum or self-play.
Use the existing S2 native hook consistently, bound in the run identity.

Reward = alive hero logical mass / 1000 per frame; gamma 1; GAE lambda .995;
H5000. Death and horizon bootstrap zero, rollout cuts bootstrap critic values.
Two PPO epochs; clip .1; KL stop .01; encoder LR 1e-5, new parameters LR 1e-4.
The actor has no clock. The critic receives `(5000-frame)/5000`, PopArt with
output-preserving rescaling. Refit critic only on at least **2M fit-world
transitions** before any actor/encoder update. Hold out complete worlds
(`world_id % 5 == 0`); fit the time/length baseline on training worlds only.
Require held-out complete-life late-body L>=800 **EV >= .1 AND MSE < fitted
time/length baseline MSE**, with nonempty support counts and full-life targets.
Missing support or gate failure => NOT_READY(P1), retaining INCOMPLETE critic
evidence. Neither tiny preflight nor a prior E2/E9 receipt bypasses this gate.

E8 diagnostics persist through every pause: fixed 10k ordinary-state bank,
frozen length deciles, KL(B8||policy), greedy disagreement, natural late-body
exposure, occupancy by mix/length/frame, consecutive non-greedy runs, life/mass
ledgers, and stochastic-versus-greedy probes every 1M total hero transitions.
Only these registered operational/safety diagnostics may be examined during
training; no endpoint evaluation bank or interim efficacy selection.
Nonfinite logits/value/loss/gradient, >90% one action in a complete matched
100k window, or sampled boost >3x matched B8 boost in that same window => KILL
the trajectory (zero B8 boost means any positive boost triggers). Nonfinite
fixed-decile drift => KILL; finite decile drift is measured, with no invented
post-data cutoff. Delivered endpoint/evaluation boost must lie in **[.005,.10]**;
outside => nomination rejected, never choose a different checkpoint.

## Y3 durability, then 1M complete-job qualification

Y3 source and replacement are distinct L4 assignments on slot A, sequentially.
Before leaving the source, freeze a qualified pre-actor boundary snapshot and
its **transitive** file set: model/frozen teacher/encoder, residual/critic,
optimizer and PopArt, critic phase/qualification, collector worlds/rosters/
opponent carry/RNG/reset state, adapter state, diagnostics/open lives,
ordinary/refit banks (including referenced immutable files), all RNGs and
counters, source/input identities. Every payload has `.sha256` and `.complete`.
On the source, record continuation through **>=3 actor updates, >=1 actual
episode boundary, >=1,000 delivered actions**. Restore the same snapshot on a
replacement L4, verify the complete transitive set, replay that continuation,
and compare action tape, model/optimizer/PopArt/world/carry/adapter/counters/RNG
exactly (wall-time counters excluded). Actual assignment identities must differ.
Missing state, action/state mismatch, missing episode boundary, or insufficient
updates/actions => Y3 not PASS and P1 NOT_READY. CPU unit tests are not Y3 PASS.

The 1M qualification resumes the source's original qualified boundary after Y3
PASS, with target still 100M and `pause_at=1,000,000`. Measure complete-job time:
critic refit, collection, updates, ordinary-state and gameplay probes, periodic
checkpointing, bootstrap/download, final export/transfer and release overhead.
The qualification report and supervisor terminal receipt bind these components;
missing final transfer/release timing blocks funding. Extrapolate the remaining
99M, explicit per-job overhead and final probes. Require **<=46h** for the whole
100M trajectory AND the **120u** invoice below. E4's measured ~1,440 active hero
transitions/s is a sizing input (100M collection alone 19.29h); E4 allocated
821/867/s and complete-job rate are distinct. Never call the collection estimate
a measured complete-training forecast. Failure => NOT_READY; no endpoint change.

## Sequential chunks, budgets, transport and deadlines

One L4 at a time for P1; account-wide cap three including notebook workers.
After qualification, pause at 20M, 40M, 60M, 80M and 100M (first tranche adds
19M). Exact continuation only; every job fresh output directory and unique
session. Per job **2× estimate <= timeout <=12h**, all costs charged; no paid
idle. Reprice from 1M evidence before materializing each dispatch manifest.
Planning forecast uses 1,440/s, then measured complete-job forecast replaces it.
Whole-session first wall/unit cap wins; cap expiry => INCOMPLETE, no automatic
retry. Mechanical recovery requires a new prepared attempt and transport receipt;
both attempts retained within the original envelope. Scientific failure is final.

Invoices: **Y3 <=8u; 1M <=4u; 100M <=120u including 2× liability plus reserve;
evaluation <=10u; total <=142u**. At the registered ceiling 2u/L4-hour, with 20%
of the 120u envelope reserved, usable liability <=96u, hence nominal full-job
forecast <=24h (2×48u +24u reserve =120u). This budget constraint is stricter
than the 46h timing gate; both apply. Retain each phase's debit, timeout liability,
failed-attempt cost, inventory and remaining reserve. No RunPod/A100/H100.

Z-9 transport: upload small catalogs/control manifests (~18kB family), never
checkpoint or fixture directories. Fetch checkpoints on the runtime and verify
registered sizes/hashes. Snapshot downloads have explicit HTTPS URLs, sizes,
hashes and safe flat names, including the transitive closure; no invented URLs.
Mac preflight exercises the **real pinned supervisor source/transport preparation
at actual bytes**, with forecast <=150s, half the shared 300s cap; receipt is
mandatory for each exact manifest, including fallback archive bytes. No live
Colab action is needed for this preparation. Before Claude assigns: `colab usage`,
fresh inventory, preceding terminal receipt and verified release, current
real-transport upload receipt. Missing one => NOT_DISPATCHABLE.

Registration + Y3 manifest due **Thu Oct 22 14:00 PT**; 1M qualification Thu PM;
100M first dispatch Thu night after slot A frees from E3. Endpoint forecast
Sat night/Sun; one evaluation Sun, print Mon. Handoff Sat Oct 24 16:00 PT (packet
15:00) reports actual state, including NOT_READY. Missed gate => create-only
`big-lever/stage0/slips/<item>-<gate>.json` using PLAN §2 schema; block only its
dependents. No deadline permits a gate bypass.

## Fresh paired evaluation and one-look decision

Freeze a new P1 namespace/round and collision-audited bank before gameplay:
**128 scripted +128 frozen +128 mixed**, each candidate and B8 on the same
world, roster, profile and worker/CPU model, H5000, terminal hero, dead frames
zero. Burned training/calibration/Y3/monitor worlds are excluded. Audit accessible
source and supplied historical inventories; reject the whole allocation on
collision, advance the consecutive round, never replace individual worlds.
Never open `snake-dqn-artifacts/t34-round0-study-01/`; require a supplied inventory
receipt if inaccessible history is not already covered. Record coverage limits.
Per-runtime E13 parity evidence is required and may explain only exact float32
Q ties under the registered parity rule; smoke tests do not establish identity.
No analysis before all 384 pairs and required controls are terminal and valid.

Unit = paired world, inference about this artifact, not a training-recipe claim.
Equal-mix mean of paired mass deltas; variance `sum(s_mix^2/128)/9`,
Welch-Satterthwaite df; LB90/UB90 are **one-sided 90%** bounds. No Holm or rank
selection: one artifact. Survival band v2: paired alive-fraction pooled lower
one-sided 95% bound >-.05, each mix lower bound >-.075, candidate alive-fraction
mean >=.30 in every mix (numeric tolerance 1e-9). Scripted no-death point delta
>=0, distinct from alive-fraction and mass metrics.

Precedence: missing/altered identities, incomplete rows, cap expiry, failed
parity/audit or unclean job => INVALID/INCOMPLETE (never scientific rejection).
Training tripwire => KILL and NOT_READY. Valid endpoint **UB90<40** =>
**ARTIFACT_REJECTED_AT_BAR** (one artifact, mechanism not killed). Otherwise
nominate only if **band v2 AND boost [.005,.10] AND scripted no-death point>=0
AND pooled mass point>=40 AND LB90>0**; label **NOMINATE_MAC_STRICT**.
Every other valid result => INCONCLUSIVE. No extra look or extension bank.
Nomination permits a fresh, X5-priced Mac-only strict registration in week 2;
never a strict pass, serving qualification or release. B8 remains served.

## Create-only receipts and handoff

`big-lever/exp/bl-p1/{registration,y3,qual-1m,tranche-100m,eval}/receipt.json`
with `.sha256` + `.complete`, completion marker last. Preparation receipts say
PREPARED_NOT_DISPATCHED and list missing prerequisites; they cannot stand in
for terminal/Y3/critic/qualification/efficacy receipts. Use attempt subdirectories
for later receipts, never overwrite an earlier receipt or hash-bound registration.
Every receipt binds full published executable commit, source/input digests,
manifest digests, forecast assumptions, estimate/timeout/unit/reserve invoice,
prerequisite evidence and actual release status. Each phase has `DISPATCH.txt`:
absolute preparation/validation commands and explicit dependency/refusal checks.
Sol-R supplies code and artifacts; Codex-V independently reviews; Claude supplies
live terminal receipts, verified release, and CAMPAIGN_LOG ruling/dispatch entries.
