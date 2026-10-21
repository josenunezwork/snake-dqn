# E3 recovery-04: verified all-seat action replay

Preparation only; no Colab calls or dispatch. This sibling describes the
owner-requested reconstruction repair. It changes no root allocation, split,
history length, D300 endpoint, student, fitting recipe or decision rule.
PROTOCOL.md remains byte-identical, SHA-256
`4e84360fc1686ed4b9ad937f89fd1a72c9df8a8ef89c6c1e71f75283f9e8037d`.

Recovery-03 terminated INCOMPLETE with saved local identity mismatch at decision
2916 on Colab x86. Its saved receipt reports 2318.7428601669963 seconds and
`stop_verified=true`. Known Mac/x86 float32 Q differences can flip action ties;
re-evaluating the original policies is therefore unsuitable for reconstructing
the Mac-recorded trajectory. The failed attempt and all earlier evidence remain
untouched. Prior attempt costs are conservatively charged 2.4 units in total:
the previous 1.1 units plus 2318.7428601669963 seconds at 2 units/hour, rounded up.
The original 16-unit envelope therefore retains 10.4 usable session units and
3.2 reserve units. First wall or unit cap wins.

The original unsealed T3 bank saves only selected-root hero actions, not complete
six-seat action traces. `trace_bank.py` regenerates the frozen worlds on the Mac
CPU and records every hero/opponent action up to the final selected root. It
checks all selected original raster/scalar/mask/action/frame bindings and the
unchanged B8 branch against actual Mac post-transition D300 mass prefixes,
including terminal zero suffixes. When traversed, original death controls remain
binding. No alternative-action target or student fit is produced by this full
Mac preparation. Mac processes use nice >=10, one native math thread each and
at most three workers plus their coordinator.

The resulting portable v3 root bank is a compact action representation, not a
pickle of full simulator objects. Its create-only archive is capped at
2,000,000 bytes. Checkpoints remain in the existing size/hash-bound public
archive. Every task binds its action file and SHA-256 hashes of each prepared
root's simulator state, RNG and Vector61 selection/carry/trend memory. Those
full-state hashes are newly recorded on the Mac after original selected-row
checks; the original T3 bank did not independently save full-state hashes.

Runtime root reconstruction replays the recorded actions for all six seats
through the normal prepare/observe hooks, without dispatching any policy.
It verifies all root hashes plus original observations, action and frame.
Histories contain only preceding decisions. A root mismatch is INCOMPLETE;
no root substitution, tolerance or numerical waiver is introduced.

For each resolved legal first action, D300 targets are computed afresh on the
runtime CPU: force that first hero action, then follow the original B8 and
opponent policies reactively for the remaining transitions. They use living
post-transition mass/300 and a zero-dead suffix. Recorded actions are never
continued after an intervention. The recorded-first-action branch is checked
against an independently advanced same-host reactive control, not against the
Mac future trajectory. Fresh runtime continuations may differ from Mac futures
on Q ties; the starting roots must be byte-identical. This is the owner's
explicit runtime-target allowance, not cross-architecture policy identity or
strict/serving qualification.

Real Mac `job.sh` preflight used the v3 smoke bank, three original source
worlds (including a mixed-policy roster), six roots, D300, two-epoch smoke fits,
one math thread per process and nice15. Serial reconstruction took
13.846653750 seconds; two-worker parity took 9.541679542 seconds; fits and
batch benchmarks took 1.735764250 seconds; the complete entrypoint took
25.738519250 seconds. All 13 dataset arrays match recovery-03's Mac baseline
by dtype, shape and exact bytes. The new path's serial/two-worker arrays also
match exactly. These smoke fits produce no scientific GO/KILL.

The conservative six-worker L4 forecast is 1.310004307 hours / 2.620008615
units at 2 units/hour: 0.904692579 hours reconstruction, 0.071978395 hours CPU
fit upper projection and 20 minutes setup/export. It retains a 4x slower-host
assumption and credits no CUDA speedup. Estimate 78.600258436 minutes; timeout
158 minutes (at least 2x), 300-second lease and 256 MiB export cap. The timeout
ceiling costs 5.266666667 units, inside the remaining 10.4-unit session cap.
No recovery-04 x86 throughput or target identity has been measured.

Targeted verification: 54 passed, 64 deselected across the E3 package and
supervisor validation/public-input selection. Changed Python passes Black100,
isort and flake8; `bash -n job.sh` and `git diff --check` pass. Independent
read-only source review found no remaining material issue. Protected source,
frozen protocol, prior attempts, checkpoints, serving pins and tmux sessions
were not modified.

Exact measured Mac preflight, bank size/digests, final published source pin,
Z-9 receipt, budget forecast, offline validation and commands are recorded in
`big-lever/exp/bl-e3/recovery-04/DISPATCH.txt`. That file is the operator handoff.
The manifest is validated locally and remains NOT DISPATCHED. Account inventory
must be reviewed by the future operator; no live account state is inferred here.

Full Mac root-bank preparation passed all 256 worlds / 1,536 roots and all
1,536 unchanged B8 controls. The archive is 1,954,150 bytes, below 2,000,000,
SHA-256 `9878576dadd291668c8c809343b05422c7291de1050b3338d106105a16e2bb8c`.
The original 192/32/32 splits, 768/768 classes and every original task field
match exactly. Every tape member hash and portable checkpoint path verifies.
Three overlap worlds match serial and three-worker recording by every action
prefix, shared root hash and B8 control hash. All observed workers were nice15.
Evidence: `recovery-04/{bank-verification,preflight-comparison}.json`, full-input
archive/catalog/timing and `trace-bank.log`. The frozen-opponent world that
failed at decision 2916 passed Mac source reconstruction and controls.

The decision-2916 regression also passed actual v3 replay versus the original
Mac live-policy reconstruction for all six frozen-opponent roots (decisions
2916, 2936, 3286, 4476, 4481, 4486): all 13 arrays match by exact byte binding.
Replay took 18.804049375 seconds; original reconstruction 22.737304625 seconds.
Evidence: `recovery-04/decision-2916-verification.json`. Full Mac bank recording
completed in 588.083684708 seconds. These are Mac checks, not x86 execution.
