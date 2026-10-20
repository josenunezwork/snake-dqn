# T34 teacher v2 — batch the shadow across worlds (Sol spec, 2026-10-20 ~13:10 PT)

A5 v2 on 2 L4 lanes (snake-dqn-artifacts/t34-a5-20261020/l4-{0,1}-output/exports/benchmark-lane*.json):
~50 active row-steps/s per lane vs the 3,000 go-rule; 525 s of 600 in host_stage `shadow_s` filling D=300 shadows
for 256 roots (14,896 shadow row-steps, ~35 ms each), 0 decisions, GPU util ~2%, policy_s 65 s. Mac preflight
(2 worlds): 95 row-steps/s. Diagnosis: the v2 shadow/continuation advances each world serially with batch-1 hero
forwards; the root/shadow path never uses GridBatchSim's batch dimension.

Fix (same semantics, byte-identical outcomes):
1. Advance ALL root worlds' shadows in lockstep in one GridBatchSim batch (rows = worlds), one batched hero forward
   per step on the device (GPU on L4, fp32), opponents batched (vector61 nets batched on device; scripted on host;
   exact v8 dispatch preserved), active-row compaction as worlds die/finish; same for the real-world root step and
   for search branches (bucketed, as already for branches).
2. Cold fill: fill shadows for all worlds together (300 lockstep steps), not world by world.
3. Identity: exports/labels/hashes identical to the current serial implementation on the Mac fixtures (A1), incl.
   doom/search/rescue cases and ties; replay harness 0 mismatches.
4. Profile receipt on the Mac (<=2 threads, CPU): 32 worlds x 1,000 frames, stage breakdown, row-steps/s, vs
   current serial; then project L4 (state assumptions). Also make the A5 benchmark report time-to-first-decision
   and a steady-state window separately from cold fill, and let the operator set the window (e.g. 1,200 s).
5. Rebuild the two L4 A5 manifests at the new commit (validate only); leave dispatch commands for the operator.
Rules: Mac only; no live Colab/tmux; <=2 CPU threads (Phase 0 running); commit/push often; merge to main;
run test_t33*/test_t34* (-n 2); report commit SHA and the profile numbers.
