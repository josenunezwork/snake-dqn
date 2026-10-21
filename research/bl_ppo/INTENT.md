# BL E4 then E2 development intent — frozen before gameplay

Authorized by the owner's implementation request, 2026-10-20. Source authority:
`big-lever/round-01/JUDGMENT.md` §(f), E4 and E2; `big-lever/CHARTER.md`.
This development experiment is excluded from EVERY registered verdict. It makes no recipe,
strict-gate, serving, or release claim. No Colab dispatch is authorized to this builder.

## Order, endpoints, caps

1. Commit this intent before any collection, including the tiny Mac preflight.
2. Tiny real-entrypoint CPU preflight: nice 10, at most two single-thread processes.
   Timing is operational and contended by Phase 0; never product latency evidence.
3. E4: one L4, 128 total worlds, frozen equal mixes, long-body fixtures,
   6 versus 12 single-thread collector processes, report active AND allocated hero
   transitions/s (one hero decision, never six snake rows). At most 30 allocation
   minutes and 2 units including setup/export. Stock v8 unless S2 native is on main.
   Pick the faster configuration by allocated rate; ties choose 6 processes.
4. Before E2 data, bind the immutable E4 receipt: below 1000/s -> 4M transitions
   per arm; 1000 through <4000/s -> 8M; >=4000/s permits the predeclared 20M
   variant, only when its full forecast including evaluation fits the caps.
   Default is 8M. No outcome-based extension or checkpoint selection.
5. E2: R and F, one training seed each, fixed equal opponent mixes, final endpoint
   only; combined first-cap-wins 20 units, 10 L4 allocation-hours, 48 elapsed hours
   from build handoff. Charge bootstrap, calibration, critic refit, training,
   screening, checkpoints, transfer, failed attempts and reserves. E4 has its own cap.
   No automatic retry. A cap-expired/missing/unclean run is INCOMPLETE, not a loss.

## Recipe fixed before data

Copy the B8 ego2s-bv8 encoder. Frozen B8 Q logits divided by a positive temperature
calibrated on separate training observations to 1% expected non-greedy probability
(ties treated as greedy). Trainable residual policy head starts identically zero.
R uses GRU 256; F uses a feedforward module matched in trainable parameter count.
R hidden resets at every hero death and horizon; learner replays 32-step burn-in
without loss before sequence gradients. Critic refit runs before actor/encoder
unfreeze. Gamma 1; H5000 terminal with zero bootstrap at death AND horizon;
reward is alive hero logical mass/1000 per frame. GAE .995, two PPO epochs,
clip .1, KL stop .01, LR 1e-5 encoder and 1e-4 new parameters. Frozen opponents
never learn; hero collection is on-policy. Every complete rollout is reconciled.

## Fresh screen and frozen decisions

Paired H5000 terminal-hero worlds: 96 scripted /32 frozen /32 mixed, B8 and both
final endpoints on the same worlds and CPU model. Separate training/calibration
namespaces. Collision audit against available complete historical inventories and
all accessible source banks before screen gameplay; reject an entire colliding
allocation, never substitute individual worlds. Sealed paths MUST NOT be opened;
if existing inventories do not cover a bank, fail closed pending a supplied receipt.
Pooled mass is the equal-weight mean of mix means, dead frames zero. Fixed-world
one-sided 90% bounds use stratified paired-world variance and Welch df; no recipe
inference from one training seed. Report R minus F separately.

Precedence: invalid identity, incomplete rows, counters, cap or parity -> INCOMPLETE.
Nonfinite loss/logits/value/gradient, >90% one action in a complete 100k-decision
window, or sampled boost share >3x matched frozen B8 share in that same window
-> KILL that arm (zero B8 boost means any positive hero boost triggers).
Otherwise final paired pooled mass UB90 <=0 -> KILL that arm.
Otherwise GO to scale planning only if pooled mass >=+10, scripted paired no-death
point delta >=0, AND survival band v2 passes: equal-mix pooled alive-fraction
one-sided 95% lower bound >-.05, every mix bound >-.075, every candidate mix
alive-fraction mean >=.30. One fixed look; no repeated-look spending needed.
Every other valid result is INCONCLUSIVE. KILL precedes GO. No replacement seed,
intermediate checkpoint substitution, tuning after outcomes, or second look.

## Durability and isolation

Immutable periodic exports use unique names, payload + `.sha256` + `.complete`;
completion marker is last. Export model, optimizer, critic/actor phase, counters,
RNG, recurrent and collector state, plus source/input identity. Interrupted-rollout
recovery must not claim exact resume unless all those states are restored.
Manifests use only the unchanged qualified `research/colab_cli` supervisor.
Never edit registered T34 packages or supervisor code, serving pins/checkpoints,
or inspect `snake-dqn-artifacts/t34-round0-study-01/`.
