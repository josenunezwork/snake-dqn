# After T23: buy policy improvement, then throughput

2026-10-17 · ML systems / compute-efficiency advice

**I would stop the unchanged long league run, spend a small amount on diagnosing the B8 restart, and make a demonstrably stronger training-only teacher the next substantial bet.** Use L4 for the current actor-bound online recipe, CPU runtimes for simulation and development evaluation, and A100 for large, batched offline fits. Distill only after the teacher wins on mass with survival guards. Keep the final product a single bare network, and reserve the Mac for the independent strict gate.

This is a proposed queue and experiment design, not a launch registration, spending authorization, or result. No training, evaluation, pod, or tmux job was launched for this note. Numerical thresholds below are prospective investment rules. Implementation, tests, concrete seed audits, immutable manifests and an accepted registration must precede execution. Repository evidence was read at `56d59d27`; later publication may rebase this document without changing that evidence snapshot.

## What T23 changes

[T23's result](../../../research/t23_r0_screen_20261017/RESULTS_T23.md) and [full verdict](../../../research/t23_r0_screen_20261017/results/t23_verdict.md) are unusually clear: five 8M fine-tunes, mean equal-mix mass difference −80.7846 versus B8, upper one-sided 90% limit −65.5103, and zero positive seeds. Scripted/frozen/mixed differences were −101.3/−61.2/−79.9. Frozen episode survival improved from 0.681 to 0.763 while mass fell. These are development results from Colab evaluation, not Mac strict evidence; nevertheless they decisively reject this recipe's registered investment test.

The conclusion is **do not extend R0 + N=3 + this static pool to 20M or 100M, and do not add PFSP/exploiters merely to keep workers occupied**. T23 does not establish that every opponent curriculum or larger model fails. Nor does it isolate the cause: opponent exposure, restart schedules, optimizer/replay reset and the learning objective were bundled. The intermediate 2M/4M evaluations were skipped. “More cautious, less productive” describes the outcome, not a proven gradient-level mechanism.

The restart is particularly worth one bounded diagnostic. The [earlier recipe analysis](2026-10-17_h100_preruns_rl.md) and `Schedules` in `research/redesign_m3c_20261008/train_m3c.py` show gamma restarting at .99 and reaching .998 at 8M, epsilon restarting at .05, and anchor weight remaining 1 through 8M. Adam and replay were fresh. In the BV8 path, the training anchor is on-policy v8 margin supervision, not a frozen B8 behavior/Q anchor. Preserving v8 labels therefore does not preserve B8's productive policy. This is a plausible explanation to test, not an excuse to relabel T23.

The [campaign retrospective](../redesign_campaign_retrospective_2026-10-17.md) closes the repeated late-death reward/loss/label/restart arms. T3 learned labels without the desired gameplay improvement; return-label arms hit boost problems. A larger GPU does not repair those findings. Earlier [global-observation advice](2026-10-17_open_next.md) was qualified by the subsequent [divergence analysis](2026-10-17_after_t9.md): weak long-horizon prediction does not by itself prove missing observation information. I would not immediately fund another death predictor, blanket pessimism loss, or broad architecture sweep.

Our comparator is released **B8-s3**, SHA256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`, [released bare with v8-derived inputs](../b8_s3_release_2026-10-17.md). Fixed-opponent H5000 mass is primary here. Product self-play results, league Elo, food, kills and boost style are useful diagnostics, not substitutes. This changes the priority in the older [Lane A design](../specs/lane_a_league_design_2026-10-17.md), which also targeted crowded product play.

## Compute economics: L4 first for this online workload

Use the owner's approximately 1,250 available units, A100 6.8 units/hour and L4 2 units/hour as planning inputs, not guaranteed prices or capacity. Capture actual rates at runtime admission. Colab's [official FAQ](https://research.google.com/colaboratory/faq.html) says hardware availability and resource limits vary; a subscription does not reserve several particular accelerators.

The [T16 record and T22/T24 appends](../t16_colab_smoke02_results_2026-10-17.md) distinguish N=3 actor-bound throughput from isolated learner capacity. A100 N=3 is about 7.7k learner transitions/s; the task supplies L4 about 7.6k and Mac about 5.2k. The approximately 14k A100 figure is a projection after host optimizations, not a new measured full-system result. T24's long-body kernel improvement is not a universal end-to-end speedup.

| Placement / rate | One 8M seed: pure training | Five seeds / 40M | Units for 40M | Status |
|---|---:|---:|---:|---|
| Mac, 5.2k/s | 25.6 min | 2.14 h | 0 Colab units | Approximate local reference |
| L4, 7.6k/s | 17.5 min | 1.46 h | 2.92 | Owner-supplied measurement |
| A100, 7.7k/s | 17.3 min | 1.44 h | 9.81 | Approximate measured N=3 rate |
| A100, 14k/s | 9.5 min | 0.79 h | 5.40 | Conditional projection only |

These are `transitions / valid transitions per second`, excluding setup, compilation, checkpoint export, evaluation and cleanup. Five minutes of extra runtime costs 0.167 L4 units or 0.567 A100 units. Report aggregate runtime-hours, not just elapsed time across parallel workers.

At the measured rates L4 delivers about **3.36 times as many transitions per unit**. Even 14k/s on A100 would still cost about 1.85 times as many units per transition as 7.6k/s on L4. At these prices A100 must exceed 3.4 times L4 throughput to win on variable unit cost; it can still win on a separately valued deadline. Measure the optimized L4 too: host fixes are not exclusive A100 benefits.

At 83 isolated updates/s and 256 new transitions/update, A100's current learner capacity is only about 21.2k transitions/s before actor/transfer overhead. A fourfold end-to-end gain over 5.2k is already near that ceiling. H100 is worth testing only when a useful workload is GPU-bound. It is not the prerequisite for the next scientific result. RunPod retains its separate ≥4× per-step rule; this proposal allocates it no work.

Maintain fp32, bs2 and the 256-ingested-transition update cadence for online comparisons. Batch-scale, replay reuse, learner-seat count and policy lag change learning; they cannot be silently traded for a nicer throughput number. Bf16 and larger batches belong to a later paired quality test once there is a winning recipe. Offline supervised fits can use independently qualified batching without pretending they are R0-equivalent.

## Queue order and hard envelopes

The first tranche is **at most 75 Colab units**, including billed setup/export time, plus **60 aggregate CPU-runtime hours** for diagnostics, teacher work and development evaluation. CPU runtime usage is charged to the same unit ledger at its observed rate. Use at most three Colab runtimes concurrently: two L4 online workers plus one CPU worker initially; substitute one A100 for dense fits after teacher admission. These are aggregate study caps, not per-worker allowances. Unspent capacity remains unspent if dependencies fail.

| Priority | Job and dependency | Placement | Hard budget / stop | Required decision |
|---|---|---|---|---|
| Q0 | Recover T23 curve and retention evidence; no new training | CPU runtime | 4 CPU-h, ≤2 units | Diagnostic report; cannot overturn KILL |
| Q1 | Matched current-stack N=3 L4/A100 qualification | One runtime of each type, serial scored rows | 45 min each; ≤6.6 units combined | Choose placement from valid work/unit and lag |
| Q2 | Three-arm restart test, five seeds, 8M each | L4, one complete seed per worker | 120M transitions; ≤10 L4-h / 20 units, plus development evaluation | Can one narrow repair beat B8, not merely R0? |
| Q3a | Existing five-final B8 mean-Q teacher screen | CPU; depends on identity/adapter checks, independent of Q2 | ≤8 CPU-h / 4 units | Establish teacher headroom or stop this teacher |
| Q3b | Mass-ranked rollout teacher, only if Q3a fails | CPU simulation; batched GPU inference only if measured cheaper | ≤24 CPU-h / 12 units, including pilot and label export | Establish headroom for training-only planning |
| Q4 | Small/large imitation and student aggregation; only after Q3 GO | Profile L4/A100 on fixed data; ≤2 A100-h equivalent | ≤14 units for fits; ≤12 CPU-h / 6 units for collection/evaluation | A bare student must beat B8 with guards |
| Reserve | Remaining development evaluation, transfer failures, accounting margin | CPU first | ≤10 CPU-h / 10 units | No automatic retry or new scientific arm |

The row ceilings total 74.6 units and 58 explicit CPU-hours; the global caps win if measurements or billing differ. Q2's 10 hours are a conservative ceiling, not a forecast: raw 120M training is about 4.39 L4-hours / 8.77 units at 7.6k/s. Its development evaluation uses the reserve, whose feasibility must be checked before training. Q3 and Q4 have no credible measured branch-simulation/fit rates yet; their numbers are spending caps, not promised completion times.

Use a small, disjoint calibration shard to project each whole job before admission. If its complete frozen workload will not fit its row and global cap, mark the plan infeasible and design a new smaller registration **before** scientific outcomes. Do not silently drop seeds, shorten horizons or omit comparisons. Mac strict evaluation gets its own later time budget and feasibility calculation; these Colab caps do not buy a strict result.

### Q0: recover information already purchased

Inventory the five T23 runs' saved 0/2M/4M/8M artifacts and telemetry by SHA. If intermediates exist, evaluate all available registered intermediates on a fixed fresh **16 scripted / 8 frozen / 8 mixed worlds per seed** diagnostic bank, paired with B8. Missing checkpoints are reported; no retraining to reconstruct them. Complete five-seed coverage means up to 480 candidate episodes plus 160 shared B8 episodes; the 0 endpoint is the B8 identity control, not another independent comparison.

Plot mass, alive fraction and episode survival, alongside food per alive decision, executed/greedy boost by length, mean length while alive, action agreement with B8 and v8-margin agreement on a separate 4,096-state corpus. Label denominators explicitly. Rising survival with falling food or alive-conditional mass supports productivity loss; early disagreement supports restart drift. Neither proves which schedule component caused it. T23's boost floor was unevaluable because eligible-support counts were absent; do not infer zero eligible boosting.

All curve inference is retrospective and exploratory. No early checkpoint becomes a new candidate because its curve looks good. Q0 can proceed while Q1 is prepared, and its artifacts inform a separately frozen Q2 registration.

### Q1: price the implementation that will actually run

Freeze the tested T22/T24 source, N=3 six-policy pool, fp32/bs2, raw local rings, 128 worlds, six actors and queue depth two from T23. On each device run two disjoint matched fixtures: ordinary exposure and a declared long-body exposure. Each row gets at least two minutes warm-up and five minutes scored time, twice; allow the remaining 17 minutes for setup/export/cleanup. Record distributions and reset rules so short bodies cannot stand in for late-game actor cost.

Require complete rows, clean actor shutdown, valid hashes, zero corrupt/duplicate replay packets, ±2% update-cadence accounting, and behavior publication-lag mean/p95 no worse than the corresponding pre-optimization same-device fixture; p95 wall age may increase at most 10%. Include the pre-change fixtures' receipts; missing comparable evidence means no freshness admission. Require differential parity tests for the optimized paths independently of training performance. Pick the lower total units per completed seed, unless a deadline premium is explicitly recorded. A slower but valid L4 is acceptable. Passing Q1 establishes no learning advantage.

### Q2: one repair test, not a hyperparameter search

Test the hypothesis that restarting a mature B8 under pretraining schedules and a non-B8 anchor erodes useful behavior. Use **three arms × five new, paired training seeds × 8M learner transitions**, all N=3 and the same frozen pool:

| Arm | Schedule and additional objective | Interpretation |
|---|---|---|
| C | Reproduce T23 R0, fresh Adam/replay | Contemporaneous control; historical T23 cannot replace it |
| M | Fix gamma=.998, epsilon=.01 and v8-anchor weight=.1 from the start; otherwise C | Test the mature-schedule package |
| MB | M plus frozen B8 masked policy KL | Test behavior retention conditional on M |

Before registration verify those final schedule values against the pinned B8 training configuration. A discrepancy blocks this proposed numerical recipe; it is resolved before outcomes. All arms remain weights-only restarts, not resumed B8 runs. Keep LR, batch size, target cadence, reward and n-step unchanged. This nested comparison identifies a schedule package and the added retention term, not individual gamma/epsilon/anchor causal effects or their interaction.

For MB, use a fixed 100k-state B8 on-policy bank, equal world-family sampling, excluding every evaluation bank. Define `p_B8 = softmax(masked Q_B8 / 1.0)` and the student distribution identically; apply mean `KL(p_B8 || p_student)` with coefficient .1 relative to the existing per-update loss. Freeze exact reduction and mask semantics in tests. Supply 256 B8-bank rows per optimizer step **in addition to** unchanged TD/v8 batches. This is a new learning term, not a throughput optimization. Log its gradient contribution and price its additional forward/backward work. No adaptive coefficient or LR sweep.

Save 2M/4M diagnostics; **8M alone decides**. Apply the common five-seed screen below to M and MB versus B8 and report paired differences versus C. A repair qualifies only if it meets the B8 GO rule and its lower90 difference versus C is positive. If both qualify choose M, the simpler intervention; MB wins selection only if M does not qualify. If neither qualifies, close this restart-repair tranche. Mere recovery from −81 to −5 mass is not a GO. Do not use an R0 failure to censor its comparisons.

This test targets productive behavior under a new training distribution; it is not a reopening of the closed death-window curriculum. The particular repair remains speculative. Its low cost and direct counterfactual justify one test, not indefinite rescue of the old recipe.

### Q3: demonstrate an expert before paying to imitate it

Start with the existing five B8 finals from [T7](../../../research/redesign_t7_phase_r_20261016/PREREGISTRATION_T7.md). Q3a uses their unweighted mean fp32 Q, the existing legal/advisory mask and deterministic lowest-index tie rule. It is a **training-only ensemble**, never the proposed served network. Verify all five checkpoints and evaluate it against B8 on five disjoint development world blocks of 64/32/32 worlds. Those blocks estimate world uncertainty for one fixed teacher; they are not five independently trained teachers.

Require mean equal-mix mass gain **at least +40**, a one-sided 90% paired-world bootstrap lower bound above zero, nonnegative scripted mass, positive effects in at least four of five world blocks, and the common survival guards. Resample worlds jointly across mixes where identities overlap, with 10,000 draws and a pinned analysis RNG. The +40 target leaves room for imperfect distillation; it is a proposed headroom margin, not a predicted result. A cautious ensemble that only survives longer fails. If Q3a passes, skip Q3b and spend on its student. If it fails, one different teacher is allowed below.

Q3b is a **mass-ranked rollout policy**, not the old survive-versus-die label bank. At hero decisions 0,16,32,…, branch every legal action, then follow B8 for a total horizon of 256 hero decisions; use two independent simulation RNG continuations with common randomness across root actions, separately seeded from the real episode's future. Frozen opponents retain their pinned policy identities and state; clone all simulator and policy carry state. Rank the average undiscounted sum of hero length, with zero after death; no survival bonus, death-window selection or leaf-Q bonus. On ties choose B8's action, then lowest index. Between query points follow B8. The finite horizon and sparse intervention are explicit approximations; H5000 gameplay, not branch score, decides their value.

This needs a new qualified adapter: the existing T3 rollout engine accepts stateless scripted anchors and cannot simply repeat arbitrary frozen/mixed opponent state. Require scalar-versus-batched clone/restore oracles for every policy family, root-action legality, RNG isolation, terminal zeros, boost burn and turning geometry. A boost turn is the first action only, followed by ordinary B8 continuation; do not repeat a relative-turn macro. T4's old return labels already used counterfactual mass rollouts. The differences here are broad visitation, a fixed explicit policy, no moving leaf backup, and demonstrated whole-game teacher improvement before imitation. Do not add raw mass sums to B8 Q, whose training-reward units differ.

First run a disjoint **256-query capacity/clone-integrity pilot**, excluded from all skill estimates. Maximum branching work is `queries × 6 actions × 2 continuations × 256` = 786,432 simulated world frames for that pilot. For 640 full H5000 teacher episodes, the upper bound is about **615M branch world frames**, before ordinary episodes and labels. This is why a reported 7.7k *learner transitions/s* cannot price this job. Measure world frames/s inclusive of clone, features and all policy inference, and compute the full projection. If it exceeds Q3b's cap, stop at infeasibility; do not assume A100 or parallel workers make it fit.

If admitted, apply Q3a's exact headroom screen on new world blocks. No teacher wins by using the actual episode RNG future or training on screen worlds. Simulation state is allowed for offline teaching, but future randomness is independently sampled. A teacher pass supplies a new, mass-oriented policy-improvement mechanism; it does not prove its decisions can be compressed into the current observation. A teacher failure closes the teacher tranche rather than triggering an unbounded search-depth sweep.

### Q4: test compression and capacity separately

The reason to use a large teacher is to amortize expensive improvement into cheap repeated inference. [Policy Distillation](https://arxiv.org/abs/1511.06295) demonstrates that policy compression can work; [Expert Iteration](https://arxiv.org/abs/1705.08439) separates planning from generalization. Those primary results motivate this experiment, not a prediction of success in this six-snake game.

From the admitted teacher collect **100k observations**, equally across the three opponent families, on fresh all-length trajectories. Half come from B8 visitation and half from the teacher; select frames uniformly within each source, with at most 200 records per world. Store full snapshot provenance, teacher action probabilities/branch scores, legal masks and normal BV8 inputs. No death-window oversampling. Split by whole world into 80% train / 10% validation / 10% test; descendants of one snapshot stay together. Teacher-screen and strict worlds are excluded entirely.

Fit three controls on exactly the same inputs and source states:

1. **S0:** B8-sized student, B8 action targets; checks that the new pipeline can retain B8.
2. **S1:** B8-sized student, admitted teacher targets; the first serving candidate.
3. **L1:** twice the channel and hidden widths of the B8 network, same observation; tests capacity. This is a new architecture requiring implementation and a frozen parameter count, not an existing CLI switch.

Use five independent fit seeds each, B8 initialization for S0/S1 and a preregistered function-preserving widening for L1, verified before fitting. A failed widening check blocks L1. Train masked policy cross-entropy, Adam LR 1e-4, batch 512, ten full passes, fp32; use the final pass only. Ensemble labels are its masked softmax Q at temperature 1. Rollout labels are a one-hot mass-ranked action at query frames and B8's action between queries. Fix equal source sampling and loss reduction. No TD fine-tuning, reward changes or checkpoint hunting in this phase.

Before gameplay investment require finite outputs, valid masks, and **at least 20% reduction** in teacher-disagreement rate relative to B8 **on the held-out teacher/B8 disagreement subset** for at least four of five fit seeds in the arm: student disagreement must be at most 0.8 times B8 disagreement on that subset. Apply this eligibility rule separately to S1 and L1; L1 may proceed when S1 fails, so the capacity question remains testable. S0 proceeds as the retention control. Report subset support and a paired-world interval; fewer than 100 such test observations makes the compression result inconclusive. Overall agreement can hide all valuable changes. Sparse rollout interventions also create a compression challenge because the student is not given the teacher's query clock; disclose this rather than secretly adding that input. Report held-out teacher-action regret from branch scores where available, but do not equate short-horizon regret with H5000 improvement.

Run one fixed aggregation round only for eligible arms and S0: each fit seed collects 10k new on-policy observations per arm, the same admitted teacher relabels them, and each model receives five further passes on a 50:50 mixture of its new rows and the original training bank. S0 uses B8 labels. This is an explicit distribution-shift test, not permission to keep collecting until a curve wins. Its maximum 150k additional observations, label costs and evaluation must fit Q4's cap; Q3b's expensive labels may make that route infeasible. Freeze five resulting finals per eligible arm and apply the common gameplay screen on new worlds. Do not drop an individual seed because its imitation score was weaker.

S0 must retain B8: lower90 mass difference >−10 and per-mix alive/episode point differences ≥−.02. Otherwise the distillation pipeline is not qualified. S1 must meet the full B8 GO rule; L1 must meet it too to count as a capacity success. If S1 passes, prefer it. If only L1 passes, the next bounded proposal is L1→S1 compression, not an automatic large-model release. Price its Mac forward latency before that next proposal.

If the teacher wins but both students fail, distinguish poor imitation from rollout distribution shift using the held-out and aggregated results. Do not immediately run longer TD training. A new exact-body/global input branch is justified only by an independent action-value/teacher-disagreement learnability test showing a real improvement over the same-input capacity control, with explicit observation parity and Mac latency costs. The old long-horizon AUC results alone do not provide that justification. No enriched-input branch is funded in this first queue.

## Shared scientific rules and the Mac endpoint

For Q2 and Q4, register five fresh training/fit seed IDs after collision audit. Each seed receives **64 scripted / 32 frozen / 32 mixed fresh H5000 worlds**, terminal hero, fixed opponent identities and seats, candidate versus B8 paired on the same world. Only the hero changes. Banks are distinct across seed replicas and studies. Reuse B8 outputs across arms only within the same frozen study and exact bank/engine/roster identity. Each of Q2's three arms has five independent runs. Q4's five fits estimate optimization/visitation variability conditional on one fixed teacher and shared initial corpus; they do not replicate teacher training or the initial corpus.

Let each seed effect be its equally weighted average of the three mix mass differences. Compute mean G and one-sided 90% Student-t bounds over five seed effects (df=4), retaining paired-world variances and a hierarchical bootstrap sensitivity. More frames, seats or checkpoint snapshots do not increase the independent seed count. These are development selection intervals, not multiplicity-corrected strict evidence.

Apply in order:

1. **INVALID/INCOMPLETE:** missing/corrupt artifacts, cap expiry, failed parity/identity, sample-count mismatch or unclean execution. No scientific verdict and no automatic replacement seed.
2. **KILL:** numerical/illegal-action tripwire; >90% one-action share over a full 100k action window; greedy boost >2% overall or >1% below length 700; or complete-screen survival guards fail: pooled alive difference <−.03 in any mix, any seed's scripted alive difference <−.15, or pooled no-death episode survival <.50 in any mix. Also KILL an efficacy arm when upper90(G)≤0. Emit boost-eligible support counts; do not impose an unvalidated minimum-boost reward or infer safety from a desired boost rate.
3. **GO:** lower90(G)>0, mean G≥20, scripted mass point≥0, at least four of five seed effects positive, and all guards pass. Q2 additionally requires superiority over its fresh C; Q3 has its separate fixed-teacher bootstrap and +40 headroom rule. For offline fits, behavior tripwires are checked on the declared on-policy collection/screen windows; a dataset action histogram is not a behavior window.
4. **INCONCLUSIVE:** everything else. No extension in this tranche, extra training, seed replacement, best-intermediate swap or threshold revision. A later study needs a new registration.

A Q2 winner goes to fresh Mac strict planning at its registered 8M endpoint; it need not consume 20M simply because a queue can. A Q4 winner goes as a **single bare network**. The strict plan must compare against released B8, retain the accepted survival/Tier 2 requirements, H10000 endurance/prefix checks where required, and use fresh banks excluded from every development/label/selection stage. Use the standing sequential Phase R process unless a new fixed-one-look protocol is explicitly ratified; do not copy T7's incumbent identity or one-look exception by habit. Retain the historical champion guard wherever the protocol requires it.

Size the strict bank from development paired variances and between-seed variation before accessing strict outcomes. Do not count a borderline five-seed screen as a power guarantee. The [B8 release](../b8_s3_release_2026-10-17.md) had a narrow Tier 2 margin; survival precision is a real resource cost. Reserve Mac time for one frozen candidate recipe at a time, then independent audit and separate serving qualification. Cross-platform screening parity on four worlds, a teacher win, imitation accuracy, or a successful fit cannot substitute for that chain.

## Queue mechanics that preserve value per unit

The queue should maximize completed, decision-bearing experiments per unit. When no admitted useful job is ready, release the paid runtime instead of filling it with an optional sweep. Parallelism changes time-to-answer; it does not create additional units or independent evidence.

- Keep each online learner, actors and live replay/rings inside one runtime on local scratch. Parallelize independent seeds and immutable CPU evaluation/label shards. Drive is an artifact exchange and control channel, not a live replay bus.
- Pre-create a canonical directory tree and use stable Drive file/folder IDs. T23's concurrent `mkdir` produced duplicate `evaluation` folders. A central dispatcher must grant exclusive job leases with monotonically increasing fencing tokens; Drive folder names/rename semantics alone are not an atomic queue. An expired worker's stale token cannot publish an accepted result. Completion needs exact shard coverage, hashes, clean exit and durable manifest-last publication.
- Bind each job to source/recipe/data/opponent/model hashes, seed and bank IDs, hardware class, observed unit rate, transition/update budgets, wall cap and dependency verdict. A screen GO opens only its declared child jobs; KILL/INCOMPLETE never quietly launches a retry or larger budget.
- Distinguish current checkpoint capabilities from T23's weights-only start. The present [checkpoint contract](../../../research/redesign_m3c_20261008/CHECKPOINTS.md) supports full state; replay mode `none` is a refill continuation, not exact resume. Raw replay snapshots can be multi-GB and durable drain takes time. Register snapshot frequency from measured interruption risk and export overhead; avoid repeated huge Drive exports for an approximately 18-minute seed. Validate restoration of optimizer, target, replay, actors, schedules and RNG before claiming exact continuation.
- Batch offline observations from local cached shards, overlap validated input preparation with GPU work, and measure samples/s through durable output. Keep fp32 reference sentinels. A100 earns its place when the larger fit's total units beat L4 or a declared deadline warrants it; report the comparison, not GPU utilization alone.
- Collect per-stage time: setup, simulation/features, opponent inference, learner/gather, checkpoint drain, export, evaluation and cleanup. Include completed-unit cost, wall-age/policy lag, length distribution and actual sampled rows. Idle accelerator time and failed attempts stay in the ledger.
- Keep CPU development evaluation off expensive GPU allocations where CPU runtimes suffice. Pin its evaluator and pass the required portability checks, while recording it as non-authoritative. Mac strict and serving qualification remain Mac-only. Do not touch existing tmux work to make room.

## Commitment after the first tranche

If Q2 finds a real B8 improvement, take that small checkpoint through strict qualification before adding league complexity. If a teacher has headroom and S1 captures it, invest the next tranche in independently replicated teacher/student improvement and its Mac gate. If only a large learner captures it, treat compression or a measured richer observation as the next hypothesis. If neither route produces a candidate, publish the negative results and keep B8.

I would retain at least **1,175 of the approximate 1,250 units** until this tranche produces a useful decision. The bottleneck after T23 is finding a learning intervention with positive mass return under survival constraints. Cheap parallelism can test that faster; it cannot make an unchanged negative recipe worth scaling.
