# T34 ExIt-B8 engine prep (P1, P2 entrypoint, P3, P6 labelling) — Sol spec

Owner approved 2026-10-19 (evening PT): start prep for the panel's final plan and run the ~2-unit L4
benchmark. The plan is `docs/research/final_run_20261019/FINAL_PLAN.md` (read all of it; sections 3, 4, 5,
7 are binding). `MERGED_PLAN.md` is the pre-red-team draft, background only. A sibling job (worktree
`../snake-dqn-exit-fit`, branch `t34-exit-fit-20261019`) does P4 fit harness + P5 tower; do not touch
those files.

Base: T33 batched teacher (`research/t33_batched_teacher/`, branch tip eeb46b72). REUSE its engine,
admission, runtime, observation kernels and export code; do NOT continue T33's N8/H128 stochastic-search
host optimisation (stopped by the plan). T3 dense (`research/t3_rollout_labels_20261015`) is the mechanism
reference for the deterministic H=300 verify/expand teacher.

## Deliverables (new package `research/t34_exit_b8/`)

1. **P1 teacher engine** (plan §3 steps 1–4):
   - Triggers: confirm pass every 8th decision at L>=600; death window = 150 decisions before any student
     death, replayed from a snapshot at interval 1, teacher acts where it differs, <=2 rewinds/world;
     3% uniform sample below L600. All configurable.
   - VERIFY: one deterministic H=300 continuation of the student greedy action (eps 0, real opponents,
     cloned food RNG). EXPAND on failure: all other legal actions in lockstep at H=300; a* = highest
     student-Q survivor (ties -> student action), none survive -> longest-surviving; forbidden set F.
     Record per action (survive, alive_steps, mass).
   - Teacher-acts beta in {1, 0.5, 0}.
   - Active-row compaction: dead/finished rows leave the batch; report active vs resident row-steps.
   - Hero (and rollout-policy) forward on GPU when available, **fp32** by default; frozen vector61
     opponents batched on the device; scripted bots on host.
   - Rollout policy is pluggable: bare B8 now; the student-with-tower checkpoint from the fit job later
     (loader arch id comes from the sibling job — define a small interface/protocol and a stub that loads
     B8 so you are not blocked).
2. **Exports** (plan §5): per world seed, all six snakes' action traces, food/scripted RNG stream ids,
   sim-state hashes every 250 frames; per searched decision: index, a*, F, 6 x (survive, alive_steps,
   mass) fp16. Plus a **replay regenerator** that rebuilds the exact trajectory (and B8 observations) from
   an export, verifying hashes and dropping+reporting diverging worlds.
3. **A1 sim-identity test**: same seeds -> identical outcomes and identical exports across two runs and
   two processes (pytest, small). Report (not gate) agreement with T3 dense receipts if cheap.
4. **P2 A5 benchmark entrypoint** (plan §7 P2): the real H=300 kernel on real root worlds, all three mixes
   (scripted / frozen frp3-s12+v8, champion_a5+v8 / mixed), exact 2 processes x 5 numba threads layout,
   rollout policy = B8 (and a flag for the tower student once it exists), W=256 root worlds per process.
   Emits JSON: active row-steps/s, active/resident fraction, expansion fraction per mix, RAM peak,
   export bytes/world, worlds/h, decisions/s, GPU util samples. Hard wall-time bound (~25 min) and
   `--max-worlds`. Go-rule printed but not enforced: >=3k active row-steps/s per lane.
5. **P3 A2 cross-runtime replay harness**: label on runtime X, regenerate on runtime Y, compare hashes;
   CLI that takes an export directory and reports mismatches per world. (The live L4-vs-Mac run is mine.)
6. **P6 Colab entrypoints + manifests** for: (a) A5 benchmark, (b) labelling chunk (round, shard,
   attempt namespaces; chunk worlds from measured worlds/h; est<=180 min, timeout 360, export per chunk;
   2-min telemetry lines), using the existing `research/colab_cli` supervisor conventions (load the
   `snake-dqn-colab-jobs` skill notes / colab-jobs skill docs in ~/.claude/skills/colab-jobs if readable;
   follow what S1/T33 manifests do). Inputs: B8 s3 + six frozen checkpoints come from a public GitHub
   release that the owner has NOT published yet — keep T33's `INPUTS_REQUIRED.json` pattern, list exact
   files + SHA-256, and make the manifest builder refuse until URLs exist. Also write, as an alternative,
   whether the CLI's bounded upload path can carry these inputs (sizes vs limits) — report, don't use it.
7. **A6 Mac preflight** of each entrypoint: 2 worlds, 500 frames; record timing/progress; no /Users paths
   leak into the job payload.

## Rules
- Mac only. NO live Colab, RunPod, pods, dispatch, tmux, or paid services. Do not touch the running
  Phase 0 (tmux phase0-selection, worktree ../snake-dqn-search-teacher) or S1 jobs, and keep CPU use to
  <=2 threads for tests/probes while Phase 0 runs (6 workers busy).
- Launch scripts must use the recipe device (CUDA on L4; CPU numba threads per layout), even though your
  own Mac tests stay on CPU.
- Commit often with clear messages; push the branch; then merge to main and push (resolve conflicts
  carefully; main moves). Run the full non-slow test suite before the final merge.
- Finish with `research/t34_exit_b8/README.md` containing the EXACT commands for me to run: Mac preflight,
  manifest build, A5 two-lane dispatch, A2 replay check, and a status table of P1/P2/P3/P6 acceptance.
