# Q0a mass-loss audit result (2026-10-18, descriptive only)

Job `q0-audit-t23-mass-a100r5` (Colab A100-40GB, 10 workers, 480 H5000 development episodes on 160 fresh
collision-audited worlds; receipt `q0_mass_audit_receipt.json`, status COMPLETE). Compares released B8 with the T23
2M and 4M snapshots (8M was the T23 decision; missing snapshots stay missing). No efficacy claim.

| mix | metric | B8 | T23 2M | T23 4M |
|---|---|---|---|---|
| scripted | mean mass per frame | 538.1 | 425.1 | 483.8 |
| scripted | food per alive frame | 0.286 | 0.252 | 0.246 |
| scripted | deaths / episode | 0.49 | 0.55 | 0.44 |
| frozen | mean mass per frame | 524.1 | 470.0 | 472.4 |
| frozen | food per alive frame | 0.241 | 0.203 | 0.202 |
| frozen | deaths / episode | 0.30 | 0.20 | 0.17 |
| mixed | mean mass per frame | 572.6 | 461.1 | 446.8 |
| mixed | food per alive frame | 0.237 | 0.204 | 0.198 |
| mixed | deaths / episode | 0.15 | 0.23 | 0.23 |

Executed boost per eligible decision rose from 0.0003-0.0018 (B8) to 0.001-0.007 (T23), still tiny.

**Reading:** the T23 mass loss is already present at 2M and is a foraging-rate loss (food per alive frame -12% to
-17% in every mix) with survival roughly unchanged or better - consistent with, but not proof of, the schedule-restart
hypothesis that Q1 arms B/D (mature schedule) test directly.
