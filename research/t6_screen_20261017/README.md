# T6 RESTART development screen

Implements `docs/research/specs/t6_probe_spec_2026-10-17.md` under the immutable
T6 pre-registration. This screen is not promotion or serving qualification.

The task runs from `t4p-20261016` and owns only its `t6-probe` tmux session and
children. It never inspects or manipulates other sessions or controls training.
CPU rowwise inference and one numerical thread per child are mandatory. The
watcher always caps gameplay probes at two, including after T7 finishes.

Round 72 uses 96 scripted, 32 frozen and 32 mixed H5000 worlds per checkpoint.
All CTRL and incumbent rows finish before RESTART rows start. Round 73 is the
sole conditional extension, adding 96 scripted worlds for RESTART and CTRL.
A refreshed audit must exclude the snapshot store's 319 source worlds, prior
banks and the T7 Phase R namespace before any evaluation starts.

The five RESTART finals are fixed; smoke checkpoints are excluded. Readouts
preserve per-seed counts and paired statistics, telemetry at every 100k, shared
survivor lengths, and the immutable T4e holdout's decisive Q gaps versus the
corresponding B8 initialization. None overrides the registered decision rule.

Portable results are stored in `results/`; raw probe outputs and execution logs
are under `snake-dqn-artifacts/redesign-t6-probes-20261017`. The final report is
`docs/research/t6_screen_result_2026-10-17.md`.

After each task commit: fetch origin, merge origin/main, push HEAD:main, and push
t4p-20261016. Pre-existing untracked T4e work is not part of this task.
