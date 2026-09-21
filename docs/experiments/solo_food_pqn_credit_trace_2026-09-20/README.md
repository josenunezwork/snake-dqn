# CL: matched credit-trace continuation

**Qualified; serialized scientific comparison running. No learning result or promotion yet.**

CK's completed body-input comparison retained short-horizon food seeking but did not establish reliable survival at H256. Body input helped two seeds and hurt the third. This experiment tests whether a longer return trace improves the already body-aware policy more than an equal additional dose of the current learner.

All three CK body-aware mark-2560 checkpoints are common parents. Fresh seeds 2026095601–5603 each receive two runs: λ=.65 and λ=.95. Both restore the complete network, Adam moments, ages, agent-step odometer and tripwire counters. Initial simulator, action and SGD RNG states match within each pair. There is no repeated body-channel initialization. Each run makes exactly 512 additional native updates; mark 2816 is descriptive and only final mark 3072 determines success.

The fresh evaluation bank has 32 worlds, four headings and three balanced reachable-food placements: 384 lanes per role. Seventeen greedy CPU evaluations cover teacher, random-safe, three initial parents, and both learned arms at midpoint and final. Exact H64/H128 prefixes come from each H256 trace. Training never receives evaluation arrays.

For every training seed, the λ=.95 final must pass the unchanged 16 absolute food/survival checks, including at least 348/384 H256 endpoint survivors. It must also have a strictly positive paired endpoint CI over λ=.65, retain food within the declared 5% noninferiority bound versus λ=.65, and retain food versus its shared initial parent. Intervals use 32 paired world means, df31; seeds and lanes are not pooled. The control's absolute results are reported independently. No midpoint or historical teacher-label fit can rescue a failed final criterion.

The science cap is 5,940 seconds for six serialized MPS learners, seventeen CPU evaluations and one saved-array analysis. Qualification has a separate 180-second aggregate cap, zero CPU optimizer updates and two discarded MPS updates. The existing two-slot lock, two CPU/BLAS threads, one interop thread, 4 GiB RSS cap, 12 GiB available-memory floor, 8 GiB MPS-driver cap and progress watchdog remain in force.

The original qualification invocation stopped during test collection because a new test imported shape constants from the wrong module. It used 1.263 seconds and made no forward call, optimizer update or simulator step. Revision r1 corrected the import and passed 19 checks; one reducer fixture failed on exact floating-point equality (1.299 seconds, two planned constant-Q forwards, no optimizer or simulator work). Revision r2 corrects only that assertion and reuses the passing checks. Its eight reducer tests pass in 1.194 seconds. All attempts remain preserved and count toward the same budget. A separate pre-numeric preparation failure referenced a replay study's nonexistent bank; the incomplete directory is preserved and no intent or numerical work existed in that attempt.

The root agent owns design, integration and all numerical execution. Sol reviewed the implementation and scientific criteria, Terra implemented the bounded analysis and documentation slices, and Luna checked bank freshness against 559 historical metadata files and is preparing independent result verification.

Apex remains incumbent until the shared tournament gate supports replacement. Its much larger historical training dose prevents treating incumbency as inherent architectural superiority. CL is a continuation experiment on existing trained body-aware parents, not a from-scratch λ comparison. A benefit would not isolate terminal credit as its cause because λ changes all valid returns.

- [Frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-credit-trace-r3/intent.json)
- [Prospective design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-credit-trace-r3/design.md)
- [Qualification plan](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-credit-trace-r3/qualification-plan.json)
- [Completed CK comparison](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_pqn_body_access_2026-09-20/README.md)


## Admission evidence

The MPS smoke passed in 20.243031 seconds. Full parent network and Adam restoration
and initial masked actions match exactly within each pair; CPU/MPS Q parity
uses rtol 1e-5 and atol 1e-6. The first native rollout and bootstrap tensors are
identical between arms. The two λ target tensors differ by up to 0.247149, while
each matches the independent scalar recurrence within tolerance (maximum absolute
errors 2.86e-6 and 8.58e-6). Both actual rollouts contain 16 nondeath edge rows and
no death row; the already completed synthetic canonical tests cover terminal
returns and invalid-successor handling. Both discarded updates have finite
gradients, with preclip norms 0.192290 and 0.498653.

The smoke also verifies full model/Adam round trips at marks 2560 and 2561,
nonzero body input through the actual native selector, and exact H64/H128
prefixes for both H256 smoke policies. Total qualification cost is
**23.999031 / 180 seconds**, including both failed test attempts. Exactly two MPS
optimizer updates were discarded, alongside 10,764 gameplay lane-frames and two
16-step × 16-environment training rollouts. No qualification checkpoint enters
science. Peak smoke RSS was 1,401,438,208 bytes and available memory stayed above
35,132,702,720 bytes.


## Preserved teacher-report recovery

The first teacher job completed and archived all 98,304 gameplay lane-frames,
then hit its 30-second limit during the final hash check of an unnecessarily
broad inherited read set. The guard stopped it at 30.1011495 seconds; no learner
training had started. The original complete raw14 archive, launch, receipt and
heartbeats are preserved.

Operational revision r3 recovers the teacher report from that archive without
replaying actions. Starting-food metadata is reconstructed with the same initial
placement routine while stepping, learned forward calls and checkpoint loading
are forbidden. The original full input check is completed once. Future
evaluations bind the exact 21 historical-fit paths they read, plus seven
explicit recovery files, instead of carrying 23,005 unrelated inherited paths.

All numerical learner, evaluator, reducer, smoke, test and bank files remain
byte-identical to qualified r2, so completed qualification is reused. Remaining
job reservations sum to 5,900 seconds; with the preserved failed attempt, the
reservation is 5,930.1011495 seconds within the original 5,940 cap. No scientific
threshold or training dose changes.


Recovery completed with a natural zero exit in 12.543018 seconds. The original
teacher archive was reused unchanged: H256 food 43.5390625, survival time
0.957143148, endpoint survivors 325/384. H128 food is 23.75 with 374 survivors;
H64 food is 12.447917 with all 384 alive. These anchors are descriptive and do
not relax the learner's unchanged 348/384 endpoint criterion.
