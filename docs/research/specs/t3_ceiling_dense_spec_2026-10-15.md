# Spec addendum: dense ceiling test (declared before running, 2026-10-15)

Why: the first ceiling test (docs/research/t3_ceiling_test_2026-10-15.md, NO_GO at 41/64 = 0.641
vs 0.65) used a -max Q trigger that fired on 0.09% of decisions (253 batches, 49 changed actions in
64 episodes). T1 found the last avoidable decision a median ~100 decisions before death, so a
sparse trigger mostly fires after the fate is sealed. That run measured the trigger, not the
ceiling of B8 + lookahead. This addendum is declared before the dense run; the first result stays
reported as run.

Variant DENSE (the ceiling of record for the T3 gate):
- Same worlds (round-52 scripted, census worlds), same checkpoints (B8_s0, B8_s3), same 64
  episodes, same continuation (B8 greedy, eps 0, no veto, advisory mask), depth D = 300, same
  selection rule (longest survival; ties to B8's argmax, then highest Q, then lowest index).
- Trigger: every 5th decision with L >= 600 (decision index mod 5 == 0 counted from the first
  L >= 600 decision of the episode), no Q threshold.
- GO iff pooled scripted episode survival >= 0.65 (unchanged rule). Report alive-frame survival,
  mass, deaths/lengths, batches, changed actions, wall time, and the paired per-world change vs B8
  alone and vs the sparse variant.
- If GO, continue with Part D (label bank) of t3_rollout_labels_spec_2026-10-15.md, using the SAME
  dense trigger for label collection (every 5th decision, L >= 700, frame >= 2000) instead of the
  -max Q trigger, then Part E. If NO_GO, stop and report.
- Budget: up to 3 worker processes while tmux t2-lane0/t2-lane1/t2p-run are alive (split by world
  or seed). Same guardrails as the main spec.
