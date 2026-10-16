# Spec: RunPod plan for T4 label fine-tunes (dry-run only, 2026-10-15)

Implementer: Codex. Read AGENTS.md; `research/redesign_m3c_20261008/gpu_run_v2.py` (plan keys incl.
`extra_inputs`, per-run `init`, `preflight_transitions`, `gpu_count`, `image`, `--pod-prefix`,
stock polling) and the last real plan `research/redesign_m6_20261014/plans/m6_pod_plan.json`; the
T3 pre-registration and the T4 tooling doc on main. Owner allows RunPod "if it helps"; the project
rule is a >= 4x speed-up per step vs the Mac and ~$10 per step.

NO SPENDING: never create a pod, never call the runner without its dry-run mode, never upload.

Task:
1. Write `research/redesign_t4_20261015/plans/t4_pod_plan.json`: 5 runs (seeds 10-14, init B8_s0..4
   finals via extra_inputs + per-run init), each an 8M fine-tune with the M6 recipe
   (`--td-huber-delta 1.0`) and a label bank shipped as extra_inputs (every file of the derived
   prefer-nonboost bank v2 directory, sha256-bound, under one dst dir) with
   `--label-bank /r/in/<dst> --label-bank-frac 0.5 --onpolicy-labels --label-frac 0.5
   --anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0.1` as the PLACEHOLDER arm (the real arm is
   chosen after T3 reports; make the arm flags one clearly marked block). Pinned GPU type RTX 4090,
   no fallbacks; size RAM/vCPU from the M6 plan lessons (1-GPU hosts with >= 32 vCPU; RAM per run
   with --obs-pack ~6-7 GB plus the bank); no compile on pods.
2. Check the trainer on the pod's torch (2.5.1; a local venv may exist at /private/tmp/torch251)
   accepts the bank flags (run the label-bank unit tests under that venv if present, else say so).
3. Run the runner's plan validation / dry-run, fix any plan problems, and report: projected wall
   time per run and per pod from the measured pod rates in docs/research or the M6 logs, projected
   cost, and whether the step clears the 4x rule vs the Mac (Mac: ~27-43 min per 8M run, 2 lanes).
4. Doc `docs/research/t4_pod_plan_2026-10-15.md` with the dry-run output and the launch command.
Git: branch t4prep-20261015; commit often; after each commit `git fetch origin && git merge
origin/main`, `git push origin HEAD:main`, `git push origin t4prep-20261015`.
Never touch tmux t3-lane0/t3-lane1/t3-probe/git-sync/runpod-stock-monitor or processes you did not
start. Final message: dry-run verdict, 4x verdict, cost projection, commit sha.
