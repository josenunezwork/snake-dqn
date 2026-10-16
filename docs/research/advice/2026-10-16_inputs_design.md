# Input experiment design (advice, 2026-10-16)

Advisory note from the cloud session, twelfth request, after RETURN_C hit the boost tripwire at
0.2M: four arms in which teaching the decisive boost escape generalised into broad boosting.
Not gate evidence. Everything below computes from the observable state the way the 30 v8
per-action features in `src/simd_env/v8_nb.py` do, so the served artifact stays a bare network.

## 0. Why inputs, and why these

The v8 area features use `cap = min(160, max(32, 2L))` and `need = min(L, cap)`. For any snake
over 80 long the cap is 160 cells, so `area_g` reads 1.0 for a 200-cell pocket and for an open
1,800-cell field alike, and `spacious` calls a 160-cell pocket safe for a 1,000-long snake. T1
measured the head-side region at trap onset at a median of 160 (the cap) and one frame earlier
at 230 with an IQR of 160-1,872. The pre-seal shrink from 1,800 to 300 is invisible to the
network by construction. That, not tail topology, is the missing perception; T1's R3 tested
whether the surviving action differed on tail features, not whether region scale was visible.

## 1. Per-action features (6 actions each, appended to the 60 scalars: 60 -> 84)

All floods are the existing tail-aware breadth-first search (`tail_aware_reachable`) after the
hypothetical move, same passability rule, no cap (early exit at `gw * gh`). Normalisation
`log2(min(area, 4096) + 1) / 12`, so 160 cells reads 0.61 and 1,800 reads 0.90.

1. **`area_log`**: uncapped post-move reachable area, normalised as above. The pre-seal signal.
2. **`boost3_area_log`**: advance the head three boosted frames straight along the action's
   direction (two cells per frame, own body advanced, boost cost applied, opponents static),
   0 if any frame collides; else the uncapped area from the landing cell. This is the v5
   landing flood taken three frames deep, and it is the boost-escape feasibility test: large
   when a short burst crosses a closing gap, zero or small when it does not.
3. **`core_area_log`**: the post-move reachable area when cells with two or fewer free
   four-neighbours are impassable (an opening by one cell). A pocket joined to the field by a
   one-wide neck loses the neck and the field; this is the cheap bottleneck signal and needs
   no min-cut.
4. **`tail_reachable`**: 1 if the flood touched an own-tail cell (TTL <= 1), else 0. The
   existing flood already visits it; it is the one bit that says the pocket keeps opening.

Most likely to separate boost-escape states from look-alikes: features 2 and 1, then 3; feature
4 is for survival once enclosed and costs nothing. Everything else from the earlier notes (time
to seal under opponent motion, narrowest gap as a true min-cut) is deferred: the first needs
opponent models, the second is expensive and feature 3 approximates it.

## 2. Separability test before any training

States: the return bank, 9.5k decisive and 9.5k matched outside states, restored from the
snapshot store so the features can be computed; held out by world with grouped folds.
Labels per (state, action): branch died within 300; and per state, boost-only escape (some boost
branch survives, every normal branch dies). Fit a gradient-boosted classifier three ways: the 60
scalars; 60 + 24; the 24 alone. Report AUC on both labels. Proceed if the 60 + 24 AUC on
per-action death beats the 60-scalar AUC by at least 0.05 and the 24-alone AUC is at least
0.80. Second check on T1's 46 late LADs: the share where the surviving action has higher
`area_log` or `boost3_area_log` than the chosen action, expecting 70% or more, the bar R3 used.
Compute the features in plain NumPy for this test; the numba port and parity tests come only
after it passes. About half a day including the port, minutes for the fits.

## 3. Training plan if separability improves

- **Widen** 60 -> 84 scalars with zero-initialised columns, as `load_init` widens b to bv8, so
  the network starts bitwise as B8.
- **Arm INPUTS**: the M6 CTRL recipe, TD plus the on-policy v8 anchor, no bank, 8M from B8,
  five seeds, fresh round, the T2 rule. The hypothesis is that with the pre-seal shrink visible,
  plain TD learns the value difference from the deaths it already sees. One change per arm.
- **Arm INPUTS+LABELS** only if INPUTS moves survival but misses GO: add the TWO_SIDED_T bank
  loss, which the inputs may now let the network fit without spillover. Not RETURN_C: its
  failure was a boost push from the targets, which inputs do not cure.
- Keep both boost tripwires and the held-out decisive Q-gap readout. Add the fuse readout:
  -max Q AUC against death within 100 decisions, expecting it to rise above B8's 0.85.
- 20M retrain from scratch with the widened input only after a GO, then Phase R under the
  two-tier guard. The serving path must compute the same features; the v8 featurizer already
  runs there, so this is an extension of an existing contract, not a new one.
