# P1 amendment 04 C2-prime and replay qualification

Status: **HOLD_RETURN_TO_JUDGE — NO DISPATCH**, 2026-10-22 PT.
Authority: `snake-dqn-artifacts/big-lever/exp/JUDGMENT_P1_E8_04.md`,
sections 4–6 and 8. Executable replay pin:
`35104156c44cefd59f1113a976400821e6029a47` (implementation `da262a3a`,
then origin/main merged and both remotes pushed before replay).

C2-prime PASS: FIRST_STEP_FIXTURE_04 has installed sigma4.9824724197387695,
three cumulative raw drifts2.0773143768310547/.658843994140625/.3901634216308594,
registered-control drifts.40287870168685913/.1446724534034729/.09396742284297943,
actual bias shift3.814697265625e-6, analytic sensitivity below1, and finite
values/losses/gradients in both arms. D1 repeats the unchanged original fixture;
D2 stays below.050147 raw units. Initial execution raised a dtype error in the
control before a receipt existed; the fixture now passes fp32 features/targets
to the frozen registered fit_completed_critic method (called fit_complete_targets
in the judgment). No learner/model/refit/run/optimizer/recipe edits were made.

C3 audit v3: P1_E8_04_ALLOCATION_V3_INDEX.json binds external immutable audits.
All four seed/ranges enumerate e0..127 and a0..1:131072 tuples,131069 unique
family hashes,131466 burned exclusions. Each selected lane has16384 allocated
roots. The initial offset pair was rejected on a burned collision; BOTH advanced
by exactly10000 to canary3010000/source3011000. Burned and cross-lane
intersections are zero. Source04 non-run reconciliation and eval-seed
classifications are cited. Claude's collection-history-inventory.json is still
absent, explicitly pending; Sol did not create a Claude receipt.

The single qualifying C1 replay was run at the executable pin above, with
SNAKE_DQN_DEVICE=cpu and OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=
VECLIB_MAXIMUM_THREADS=NUMBA_NUM_THREADS=1. Replay_02 records all four requested
variables as1 and torch_threads1. Every source digest matches that pin.
P1_E8_04_REPLAY_02_QUALIFICATION.json applies the NEW judgment tolerances:

| Check | Actual | Required | Result |
|---|---:|---:|---|
| Rank |257|257|PASS|
| Raw FIT MSE |896460.1646345354|896460.1646345354, rel1e-9|PASS|
| Installed FP32 FIT MSE |896460.1739480429|896460.1603503079, rel1e-9|FAIL, rel1.5168253542086063e-8|
| Maximum prediction delta from old head |8.7606201171875|11.07476806640625, abs1e-6|FAIL, error2.31414794921875|
| Absolute common-mode sensitivity |2.955857780762017e-16|<=1e-9|PASS|

The legacy tool's FIT_ACCEPTANCE_PASS does not implement the new reproduction
tolerances and is NOT a qualifying pass. Section5 orders a stop and return on
these deviations. No rerun, tuning, solver variant, -03 pack construction,
allocation or provider call followed. No held-out number selected a change.

Section6 disposition: (1) implementation merged, independent review HOLD;
Claude log check pending; (2) replay published but qualification FAIL;
(3) zero-collision v3 audits published, Claude inventory pending;
(4) current CLI tree553af3f692515b2bf66173356fe4d099acc96942 equals qualified
registry and verified PASS acceptance, but no -03 manifest pin exists;
(5) fresh packs NOT BUILT due mandatory stop;
(6) live zero-assignment/rate/debit reconciliation NOT performed, prior
WARNING_UNEXPLAINED_DEBIT retained;
(7) compute/testing ended before Thu20:00PT X5, no timing/K-1 job modified;
(8) no dispatch, B8 serving unchanged, protected study closed, l2mac untouched.
"dispatched pin == replayed pin" is NOT ESTABLISHED: nothing was dispatched
and no new manifest was built. Only the replayed executable identity is proven.

Verification:102 focused runtime/replay tests passed in19.26s; allocator5 and
pack-preparation3 tests passed, all single process(-p no:xdist). Black100,
isort Black profile, flake8 and diff-check passed. The broader unchanged
supervisor/canary test selection was interrupted to finish promptly:17 passed
in134.98s before KeyboardInterrupt, not a full-suite PASS. No tests left running.
No full suite, CUDA qualification or Colab run occurred.

Dispatch commands: NONE. The judgment's stop forbids a dispatchable pack or
command on this failed qualification. Return the immutable replay and independent
review to the judge. Historical review packs remain admission=false and unchanged.
