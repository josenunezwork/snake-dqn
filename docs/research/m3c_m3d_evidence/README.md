# M3-C evidence (2026-10-09)

Small JSON receipts copied from the M3-C training run (ego2s-b raster DQN, arms A / B / L x seeds 0-4
x 20M transitions). The full artifact trees stay outside the repo, under
`snake-dqn-artifacts/redesign-m3c-*` and `snake-dqn-artifacts/redesign-autonomous-20261010/`.

**Cited by:** [m3c_m3d_learnings_2026-10-12.md](../m3c_m3d_learnings_2026-10-12.md).

**What it supports:** the M3-C bottom line in that doc. No arm beat the incumbent frp3-s12 + v8
(434.93 mass) without a veto, so nothing was promoted and neither selection entered Phase R.
The 10M/15M probes are descriptive only. They show learning still improving on seeds 0/1.

| File | Contents |
| --- | --- |
| `10m-probe-summary.json` | `DESCRIPTIVE_ONLY` probe of seeds 0 and 1 at 10M (16 worlds x 3 mixes); identity and paired-world checks PASS. |
| `15m-probe-summary.json` | The same probe at 15M; descriptive only, no selection or promotion. |
| `final-noveto-five-seed-summary.json` | No-veto results for all five seeds at 20M (32 worlds per mix); incumbent v8 434.93; registered no-veto bar 384.93. |
| `m3c-selection-final.json` | Registered selection scoring at the final checkpoint: scores, primary/secondary results, contrasts and rules. |
| `retry-recovery-closeout.json` | Experiment closeout: status `CLOSED`, primary and secondary `NOT_ENTERED`, with source commit and preregistration hashes. |

The learnings doc summarizes the M3-D diagnostic series; this folder holds no M3-D receipts.
