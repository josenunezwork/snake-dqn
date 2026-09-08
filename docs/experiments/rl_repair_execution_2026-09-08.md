# RL repair execution — 2026-09-08

Status: the initial repair implementation and G0 qualification are complete;
the bounded X0 development screen closed inconclusive. The accepted plan is
preserved at `docs/plans/rl_repair_2026-09-06/`; its PLANNED fields describe the
delivered plan snapshot, not execution status. Conditional follow-up work has
not started. The incumbent and `main` remain unchanged.

- Plan base: `5d5e85d6da1be3af48845e233d826c9cdcec77ed`.
- Integration branch: `codex/rl-contract-repair`.
- Main at start: `950d583b9eb0c3e316edd0f651954d133334ac1a`, clean.
- Durable campaign ledger:
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/campaign.json`.
- Package worktrees:
  `/Users/josenunez/Projects/ml/snake-dqn-repair-worktrees/<package>/`.

The root integrates reviewed commits serially, records test evidence on the
integrated source, and starts only packages with accepted dependencies. Local
artifact paths below are accompanied by SHA-256 manifests in the campaign
folder. No package promotes a model or publishes main automatically.

## Resource baseline

Apple M5 Pro, 18 logical CPUs, 64 GiB unified memory; Torch 2.9.1 with MPS
available. The launch probe observed about 19.9 GiB available memory. This is a
point-in-time value. Keep at most two one-thread small CPU checks, or one
exclusive full-suite/evaluation/corpus/MPS job. MPS training uses two native
threads with the plan's 4 GiB RSS, 8 GiB driver and 6 GiB available-memory caps.

## Accepted packages

C0, A0, N0, S0 and E0 are accepted. The combined regression gate on
`572f089` passed **486 tests, 1 checkpoint-dependent skip** in 28.55s.
A subsequent import-lint comment correction leaves the tournament AST
unchanged; changed-file Black, isort and flake8 checks passed afterward.
An initial integration run caught the new `provided_fields` frozen set failing
JSON receipt serialization; `572f089` fixes the explicit receipt projection
and adds a regression. Failed and passing logs remain under `integration/`.

A1 and ENV are also accepted on integrated source `8b11f924`:

- The Apex union passed **323 tests in 12.12s**, covering the new coordinator
  lifecycle paths together with replay generation/epoch safety and the learner.
  The finalizer captures state while its owners are usable, attempts every
  cleanup step independently, and publishes the terminal checkpoint afterward.
  Wall-time enforcement is explicitly cooperative; resolved budgets and observed
  overshoot are recorded. It does not promise to preempt a hung learner step.
- The simulator union passed **52 tests in 143.92s**, including all three slow
  parity batteries (20 seeds × 10,000 frames each), repeated live resets, and
  inactive-world semantics. The exclusive run used about one CPU core; observed
  RSS was 345,440 KiB. This is a sampled resource observation, not peak memory.

OBS is accepted after an integrated **117 passed, 3 slow deselected** in 31.96s
on `cb3d03d1`. The slow cases had already passed in the ENV gate above. A later
import-only ordering fix gives the clean dispatch base `60747fb5`. The raster
tests compare a reconstructible v2 fixture from the pre-change producer and
independent v3 paint-order oracles, including complete enemy-slot permutations.
Legacy adapter defaults remain intact; v3 callers must supply semantic
normalization values explicitly. The Torch v3 surface currently delegates to
the canonical NumPy renderer; no CUDA throughput claim is made.

OFF and P1 are accepted and integrated through `64530856`. Their combined
regression run passed **668 tests in 14.83s**. OFF preserves full replay dataset
provenance across repeated copy-first migrations, validates both 58- and 61-D
rows, and labels the stored masks as advisory. Its complete live-world descriptor
preserves actual scaled pixel geometry without weakening the aligned SIMD contract.

P1 now freezes the actual world, reward, target, sampler and optimizer recipes.
Continuation rejects changed semantic descriptors even if their digests are
recomputed; it also validates actual Adam state. Weights-only loading requires
fresh optimizer, counters, pool and RNG state. Target and self-play descriptors
explicitly identify the remaining pre-P2 behavior rather than claiming it fixed.

S1 is accepted at source `0d88879`, integrated at `03336522`. Its final package
union passed **229 Python tests, 5 conditional skips**, with **32 frontend tests**
and a production build. The integrated P1/S1 union passed **271 tests, 2 skips**.
A checkpoint serialized on that same integrated source loaded in Watch and Play
with nondefault normalization values and passed the executable serving spot check.
These are initialization, serialization and serving checks; no model was trained.
The detached serving manifest describes manual reset, no live episode horizon,
the external evaluator's separate 5,000-frame horizon, and the live world's
unbounded storage adapter. Real GameState oracles cover pre-human-move observation
bytes, sparse IDs, mixed policies, reorder, deaths and respawns. Raw checkpoint
world, runtime and model-head descriptors must be complete before construction;
tests remove and re-sign each field to prove defaults cannot fabricate source truth.

The P1-WORLD amendment at `f85e4ea` serializes all 23 effective-world fields,
including inactive circle geometry for a rectangular world. Its focused gate
passed 96 tests. This is a provenance completion, with no numerical recipe change.

E1 preparation identified a further nontraining food-replenishment mismatch:
actual GameState maintains ambient food after each eater, while the previous
SIMD path spawned unconditionally. ENV-WATCH is accepted at `eee93701`: the
same actual-live oracle produced four expected failures on the preceding source,
then the integrated union passed 53 tests with 3 slow cases deselected. E1 will
inherit this amendment.

A frontend collection failure also exposed eight required source files present
on the main checkout disk but hidden from Git by a broad `lib/` ignore pattern.
Commit `0e441cdb` preserves their exact local bytes and adds a narrow exception.
The source manifest records their origin and hashes. With this prerequisite,
the S1 frontend passed 166 tests and its production build. Main was not edited.

Independent preparation produced a 70-digit mpmath Student-t reference fixture,
cross-checked with higher-precision density quadrature and analytic cases, plus
exact rational Holm boundary cases. The binary64 extension uses 80-digit exact-input references, with 422 independent
assertions. A bounded pure-Python numerical prototype passed 12 tests and independent
review against these fixtures, including direct quadrature and exact Holm boundaries.
These are E2 implementation oracles, not evidence that the inferential gate exists.

The campaign ledger records exact package/integration/test SHAs and durable
evidence manifests. G0 later qualified the repaired source for a bounded
development screen. X0 then retained five incomplete wall-stopped arms; it did
not produce a model-quality estimate, strict authority, release authorization,
or incumbent change.

P2 is accepted at source `3692585`, integrated through `c080f6a`. The integrated
PQN union passed **291 tests in 21.80s**. Hero-only updates now use actual episode
completion and event counts; immutable opponent leases are released before new
admission. Fixed opponents do not consume phantom exploration randomness. Target
recurrence, sampler descriptors, and fresh continuation preconditions have actual
consumer oracles.

A2 is accepted at source `847bc07`, integrated through `d208afb`. The combined
Apex, PQN and serving union passed **1,241 tests in 32.50s** on that source.
Mask semantics survive actors, n-step returns, replay, persistence and TD targets.
Recipes record actual optimizer and target behavior. Verified continuation starts
fresh RNG/replay streams while preserving validated optimizer clocks; weights-only
starts keep all training state fresh. Streamed checkpoint snapshots bind saved
parent lineage to the exact loaded bytes without duplicating a large checkpoint
in memory. Snake episode rewards remain separate from policy cumulative rewards.

E1 and the ACTION-RNG amendment are accepted on integrated source `0896caa`.
The combined evaluator, serving, simulator and PQN regression run passed
**252 tests, 1 conditional checkpoint skip in 12.86s**. An actual zero-update
PQN checkpoint from that same source loaded and stepped in Watch and Play,
and both evaluation engines received identical tactical, strategic and scalar
tensors over three frames including a food replacement. The checkpoint and
receipt are preserved under `integration/p2-e1-roundtrip/` in the evidence root.

Two runtime counterexamples explain the additional amendments. The SIMD evaluator
previously selected actions before Watch food maintenance and respawn; the new
callback puts selection at the same phase as live Watch. The live v3 attachment
also referenced classes imported only in the v2 branch; the actual producer test
caught and now covers that path. Finally, AISnake consumed the shared Python RNG
even when epsilon was zero. This changed later food replacements only in live
evaluation. The zero-epsilon guard fixes that coupling while retaining positive-
epsilon draw order. This intentionally corrects zero-epsilon live trajectories;
it does not claim legacy trajectory identity.

The additional hermetic regression at `c079970` builds its own checkpoint and
asserts all three input tensors, all anchor contexts/actions, and zero inference
RNG draws. Full diagnostic traces and failing attempts remain in durable evidence.
No learner update or skill qualification is implied by these checks.

## Qualification and development screen

E2 and H0 started from the accepted `0896caa` source. The E2 statistical slice is
accepted through integrated `cd82c09`: the statistics and tournament union passed
**74 tests, 1 conditional checkpoint skip in 4.71s**. The one-sided paired tests,
three-mix Holm correction, separate scripted noninferiority test, and conservative
pilot sizing were independently reviewed against the frozen numerical oracles.
Pilot power is a marginal approximation, not a claim of joint power across mixes.

E2 is accepted at source `abb77d3`, integrated as `94304cd`. Review rejected the
first prototype because it could accept claimed statistics without opening world
and readiness evidence. The accepted artifact boundary now recomputes decisions
from frozen requests and actual hashed files, and the tournament runtime consumes
that API. The integrated artifact, serving, runtime and H0 union passed
**198 tests, 1 conditional checkpoint skip in 18.31s**. Legacy gate and sweep
results explicitly remain diagnostic; they cannot authorize release.

H0 is accepted at source `4eb8f21`, integrated as `b673c46`. Its first runner was
rejected for manifest reuse, stale-heartbeat supervision and time-projection gaps.
The replacement passed independent review and **43 focused tests**; the root's
integrated runner, evaluator-trace and schema union passed **72 tests in 9.46s**.
It freezes source, seeds, configuration and checkpoint lineage, supervises each
learner and evaluator under separate resource and wall limits, and confirms child
cleanup before continuing. Failed and unrun arms retain their original positions;
there are no automatic retries.

The initial G0 CPU launch exposed a nested `RunProvenance` seed-lineage mismatch
before an initial checkpoint or update. `810d877` repairs checkpoint seed
validation, `3a9e7a2` covers the nested provenance metadata, and `7cb66a9`
formats the final assertions. An earlier builder log was overwritten and is not
an immutable receipt. The G0-R3 runner/source hashes and retained receipts below
are the authoritative evidence; `b673c46` remains the original H0 implementation
identifier rather than the executed source revision.

The serving evidence producer is accepted at source `ff63255`, integrated as
`95d376d`. Its combined serving union passed **79 tests in 8.96s**. The previous
validator could accept handwritten success records without showing that 50 seeded
sessions had executed. The producer observes real S1 Watch/Play execution, seed
application, AI identity dispatch and human updates. An actual terminal Play
receipt now passes through the strict consumer, including its checkpoint source
runtime and deployment manifest. The frozen later
readiness schedule contains one Watch session at the external 5,000-frame horizon
and 49 Play sessions ending at actual human death or that horizon. These mode
counts carry no statistical-power claim. Checkpoint source-runtime contracts and
actual deployment differences remain intact. Implementation uses bounded serving
oracles; the later 50-session readiness run is not part of the development screen.

### G0-R3: qualified for development screening

G0-R3 qualified clean source `7cb66a9437f63b89cd5dc8731b5f8577ce1c9765`
for a bounded development screen. The canonical receipt is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/g0-r3/qualification.json`
(SHA-256 `11482e88e5835cbd7b2f08995f4523d7b5822fa9f8f307a3f449cd96bdc98983`),
with independent review at `g0-r3/readiness-review.md` (SHA-256
`dbdd47f10d9310a5ec1da6c6efec21ed2b64fe7dac483cc178b36360cd227bcd`).

The receipt covers **2,519 non-slow passes, 6 conditional skips and 3 slow
tests deselected**; **3 slow simulator-parity passes**; **167 frontend tests**
and a production build; browser/serving checks; 300 bounded incumbent frames;
bounded Apex success and invalid-configuration controls; H0 CPU success and
retained wall-stop controls; a 5,000-frame Watch; and one MPS shakedown. The MPS
shakedown completed 100,901 actual hero steps against 100,000 requested in 73
rollout updates and 292 optimizer steps. It is a diagnostic operability receipt,
not a confidence interval or model-selection result. The G0 receipt explicitly
sets `strict_authority: false` and `release_action_authorized: false`.

A preliminary full non-slow regression on `c633135` completed in 85.32s with
**2 failures, 2,411 passes, 6 conditional skips and 3 slow cases deselected**.
Its sampled process-tree RSS reached 1.28 GB and available system memory stayed
above 18.8 GB. This early run is not the final-source G0 qualification.
Both failures are resolved: `010a0b8` corrects the schema parity test's treatment
of constructor-injected policy provenance, and `ce169b4` supplies explicit resolved
masks in the positive audit fixture. The latter also fixes a real report label:
legacy mask presence is no longer called exact coverage. Focused checks passed
25 and 22 tests respectively. The raw failed run and its interpretation remain
in the evidence folder.

The parser-only E2 addition at `f143865` passed 34 integrated tests. It registers
request/receipt paths and carries no decision authority. Only the net safe patch
was integrated; the rejected prototype's validators were removed in its worktree.

G0 command recording covered normal exit, error exit, and a detached child
surviving its parent; the latter was detected and terminated. This recorder is
separate from H0's learner executor. The final source and seed namespaces were
frozen by the accepted runner before the retained X0 screen began.

The campaign's `cpu_test_slot.py` now enforces at most two concurrent one-thread
focused test processes. Owned-file repair/rerun attempts are continuously authorized
through that wrapper. Full-suite, evaluation, corpus and MPS work remains exclusive.
The recorded event stream has observed at most two concurrent test children.

Two ambiguous A2 card phrases are resolved explicitly in the execution ledger:
50/100 are local/distributed TD-target clipping values, while gradient clipping
must record its actual separate value. Historical checkpoints remain loadable for
inference or weights-only starts; restoring optimizer/counters requires verified
continuation, with no unverified optimizer-continuation escape hatch.

### X0: executed, inconclusive development screen

The X0 launch bound exactly five `corrected-v3` arms to the qualified `7cb66a9`
source, 500,000 requested hero transitions per arm, one MPS learner, and later
paired CPU evaluation. Every original arm is retained as `wall_stop`; none was
retried, extended, replaced, or evaluated. The campaign closed with no active
learner or remaining campaign process. The final analysis is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/x0-final-analysis.json`
(SHA-256 `dd65903b68a1aceb6e0e1d180343c6e913c85efdfd0720bd4e1a6c877cb0db09`),
and its independent review is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/x0-final-analysis-review.md`
(SHA-256
`cb10eb415bc503d86ce80bca1ae351108d0e09e6beca31887e8f367e2871c616`).

| Arm | Last recorded hero steps | Requested | Final checkpoint / evaluation |
| --- | ---: | ---: | --- |
| screen-1 | 416,687 | 500,000 | absent / absent |
| screen-2 | 404,516 | 500,000 | absent / absent |
| screen-3 | 431,780 | 500,000 | absent / absent |
| screen-4 | 424,379 | 500,000 | absent / absent |
| screen-5 | 455,227 | 500,000 | absent / absent |

The 2,132,589 aggregate is a lower bound from each interrupted arm's final
telemetry row, which may omit unflushed work. With zero complete arms, accepted
checkpoint pairs, or evaluations, the five-seed interval is null. The analysis
reports no integrity error. Its generic `observed_world_count: 12` and mix labels
describe the frozen evaluation plan, not executed worlds: every arm has no
evaluator record and `world_pairs_by_mix` is null.

All checked loss/Q fields were finite and every arm reached the wall cap rather
than a numerical or resource incident. Maximum recorded process RSS was
1,559,150,592 bytes (1.452 GiB), maximum MPS-driver allocation was 1,195,245,568
bytes (1.113 GiB), and minimum available memory was 15,994,765,312 bytes
(14.896 GiB). The runtime summary is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/rl-repair-20260908/x0-runtime-summary.md`.
The frozen per-arm wall cap was 435.61 seconds; observed detection was
435.77–436.51 seconds, including less than 0.90 seconds of monitor overshoot.
This censored screen does not show that PQN, the sampler, the representation, or
an algorithm alternative won or lost.

The short calibration sampled only the early all-hero phase. In screen 1, the
first 73 updates produced 1,513.5 useful hero transitions per update-wall-second;
subsequent recorded updates averaged 859.7. Completed environments remained
inactive until the batch reset, and frozen opponents first received exposure
after that reset. This is expected under the accepted batch-barrier contract,
but the timing projection did not represent those later phases. The retained
`x0-episode-utilization.md` and `x0-autoreset-research.md` in the evidence root
explain the raw telemetry and primary-source comparison.

## Conditional follow-up work

The campaign ledger marks X0 `EXECUTED_INCONCLUSIVE`; H1, L0, L1, G1, X2 and REL
are `DEFERRED_CONDITIONAL`. Follow the tracked
[post-X0 follow-up plan](../plans/rl_repair_2026-09-08-followup.md) rather than
altering this campaign. First calibrate steady-state cost across early all-hero,
sparse-tail, reset, and frozen-opponent phases. A per-environment autoreset
comparison is a separately versioned semantic experiment: it must retain actual
death targets, final post-step truncation bootstraps, return boundaries,
per-environment RNG, and opponent leases before testing a matched valid-transition
budget. PPO and state-representation changes remain optional and require their
own qualified paths.

After all runs and the independent analysis review closed, the learner docstring
was corrected: `PQNTelemetry.completed_episodes` counts environments newly completed
in a rollout, while `episode_reset_count` counts actual batch resets. The
docstring-stripped AST is identical before and after. All qualification and
training evidence remains bound to `7cb66a9`; the final documentation commit
does not claim a new training or qualification run.
