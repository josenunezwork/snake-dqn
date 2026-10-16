# Spec: T6 RESTART arm, late-game snapshot-restart curriculum (2026-10-16)

Owner suggestion (2026-10-16): "start from large mass states and continue from there since that's the
hard points". Implementer: Codex. Read AGENTS.md, `docs/research/t4_snapshot_restart_tooling_2026-10-16.md`
(the built, default-off `--restart-bank` / `--restart-frac`; restore parity exact), the snapshot store
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4b-snapshot-restart-20261016/full` (10,885 death-window
states of B8, L >= 700 at the window), the strategy note `docs/research/advice/2026-10-16_strategy.md`
(it ranks this lever 4th and expects it to inherit the perception limit; say so in the prereg), and the
M6 CTRL recipe (`redesign-m6-mac-20261014/CTRL_s10/summary.json`).

1. Pre-registration FIRST: `docs/research/specs/t6_prereg_2026-10-16.md`. Arm RESTART = the M6 CTRL recipe
   exactly (`--device mps --actors 2 --worlds 128`, on-policy v8 anchor, `--label-frac 1`, lambda 0.1,
   `--td-huber-delta 1.0`, no bank, no return loss) + `--restart-bank <store> --restart-frac 0.2`, plus the
   boost tripwires `--boost-halt-rate 0.02 --boost-halt-low-length-rate 0.01` (halt = KILL). 5 seeds 10-14
   from B8_s0..4 finals, 8M. CTRL = M6 CTRL finals. Fresh rounds 72 (73 extension) with a seed audit that
   also excludes every world in the snapshot store; 96/32/32; same GO/KILL/EXTEND rule as T3-T4.
   Readouts: deaths per 1,000 decisions at L >= 800 in training (restarted vs fresh worlds, from the
   telemetry), fuse-free Q readout: held-out decisive Q-gap on the T4e holdout states (fatal minus best
   surviving) at 8M vs B8; shared-survivor length delta at 5000; boost curve.
2. Confirm in a 200k smoke with the real store on MPS: restart_frac_actual ~0.2, rate (expect >= 1.5k/s per
   lane), no tripwire, telemetry fields present.
3. Launch scripts (do NOT start): `/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t6-mac-20261016/lane{0,1}.sh`,
   `--device mps`, no CPU device override for training, running from this worktree; `--check` prints the
   full command.
Never touch tmux sessions or processes you did not start. Git: branch t6-20261016; commit often; after each
commit `git fetch origin && git merge origin/main`, `git push origin HEAD:main`, `git push origin t6-20261016`.
Final message: prereg path, smoke results (rate, restart fraction), script paths, sha.
