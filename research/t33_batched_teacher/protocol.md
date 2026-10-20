# T33 implementation and calibration protocol

Authorized by the owner's 2026-10-19 instruction to implement the T33 spec.
This protocol covers development fixtures and descriptive calibration, with
no promotion, training, deployment, or remote allocation authority.

Work in the handed `t33-batched-teacher-20261019` worktree only. Phase 0 code
and its existing receipts remain read-only. No other worktree or its artifacts
are accessed. Local commands use at most four single-thread workers; smoke
runs finish within a few minutes. The full ablation is left for the operator.

1. Reuse only the three calibration worlds in Phase 0's `WORLD_BANK.json`;
   validate its allocation digest, passing seed-audit binding, unique seeds,
   and separation from all 160 selection worlds before collection.
2. On each bare-B8 calibration trajectory, collect eligible prepared decision
   indices under L>=600 OR margin<0. Uniformly sample without replacement with
   fixed seed 20261019, balanced over the three mixes (default total 300).
   Replay the same bare trajectory to those exact decisions. Never change
   visitation according to a search variant. Refuse insufficient samples.
3. Compare H32/H64 x N4/N8 against N8/H128, epsilon .02, leaf_scale 1,
   with masked-max Q leaf and dueling V leaf (mean over all six Q outputs).
   Use common random numbers across first actions for the proposed teacher;
   record both the full CRN reference and original independent-action reference.
   Regret is max full-reference score minus the full-reference score of the
   cheaper teacher's selected action. Report both reference conventions.
   An optional margin gate may skip search; skipped choices still receive
   full-reference agreement/regret accounting. No recipe is selected here.
4. Small synthetic timing smoke is explicitly separate from the 300 actual
   trajectory decisions. All outputs are create-only and bind source, bank,
   audit, checkpoint hashes, parameters, platform and commands.
5. Implement the host teacher after the ablation smoke. Qualify its fixed
   shape observations and trajectories against Phase 0 fixtures, determinism
   and source immutability. Batch/fp16 ties can differ; no oracle reruns.
6. CUDA inference uses fp16 autocast and reduce-overhead compile, with >=1000
   real states and >=99.5% masked argmax agreement required before benchmark.
   CPU/MPS results cannot stand in for that CUDA qualification.
7. Prepare one L4 benchmark only, with 256/1024/4096 source worlds and about
   20 minutes of measurement. Host >=60% -> consider three L4s and step 5;
   GPU >=60% -> one separately operated 20-minute A100 probe. Allocation,
   uploading/publishing checkpoints, and step 5 remain outside this execution.
