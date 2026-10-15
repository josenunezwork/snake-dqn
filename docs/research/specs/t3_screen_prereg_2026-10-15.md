# T3 screen pre-registration (declared before any T3 training run, 2026-10-15)

Development screen, not gate evidence. Adopts the cloud advisor's review
(`docs/research/advice/2026-10-15_t3_launch_review.md`) sections 3-5.

## Arms (8M fine-tunes, seeds 10-14 from B8_s0..4 finals, the M6 recipe = template `common`)
- CTRL: the existing M6 CTRL_s10..14 finals (recipe verified identical flag for flag; Huber delta
  unset = 1.0, bit-identical loss). Not retrained.
- LABELS: `--td-huber-delta 1.0 --anchor-hold 2000000 --anchor-floor-at 2000001 --anchor-floor 0.1
  --label-frac 0 --label-bank BANK --label-bank-frac 1 --label-bank-margin-scale 1.0`.
- LABELS_KEEPV8 (attribution): same schedule, `--onpolicy-labels --label-frac 0.5 --label-bank BANK
  --label-bank-frac 0.5 --label-bank-margin-scale 1.0`.
- BANK = `snake-dqn-artifacts/t3-rollout-labels-20261015/bank` (10,885 decisive + 5,442 counter rows,
  audit PASS, manifest-bound). Labels came from simulator rollouts of B8; the served artifact stays a
  bare network (owner ruling 2026-10-15).

## Evaluation
Fresh dev bank: round_index 62 (60 = T2, 61 = T2 extensions; verify unused before probing).
Per checkpoint: 96 scripted + 32 frozen + 32 mixed worlds, no veto, rowwise ego2s forward, advisory
mask, numba vector61, fast anchors (the T2 probe path). CTRL finals and the incumbent frp3-s12 + v8
are probed on the same worlds.

## Decision rule, per arm vs CTRL (paired by seed)
Primary: paired scripted episode-survival delta, 5 seeds.
- GO: primary >= +0.06 AND mass delta vs CTRL >= -30 on every mix AND no frozen or mixed
  episode-survival drop beyond 0.05.
- KILL: primary <= 0, or any halted run (exit != 0) in the arm.
- Otherwise EXTEND once: 96 more scripted worlds at round 63 for the arm and CTRL; re-decide at 192;
  stop.
- GO means: 20M retrain with the bank, then a fresh-bank Phase R under the guard rule fixed before
  it. It promotes nothing.
- Attribution: if both arms GO or both move similarly, the labels carry the effect; if only LABELS
  moves, the anchor regime does.
Secondary (reported, not deciding): alive-frame survival; mass vs the incumbent per mix; death
lengths; final greedy boost rate; `bank_agree` at 2M and 8M (ignore 2.0-2.3M transient); post-hoc
re-roll of 200 bank states under each final (fraction of labelled survivors that still survive).
