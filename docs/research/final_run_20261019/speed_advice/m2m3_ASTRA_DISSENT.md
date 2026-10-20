# Astra vs Fable after M2/M3 (2026-10-20 ~00:45 PT)
Astra full text: /Users/josenunez/Projects/ml/snake-dqn/docs/research/advice/2026-10-20_t34_m2m3_astra_perf.md
Fable full text: /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-m2m3-advice/fable_search.md

AGREE: keep exact shadow; drop serial LAD search; Path B must publish the first fully evaluated improvement at a
still-future frame (earliest-first, batched alternatives, prepared-snapshot reuse) instead of searching for the
LAD; do not fund label rounds yet; the deciding test is paired gameplay under real deadlines vs bare B8.

ASTRA DIFFERS ON:
1. Label target. Astra: keep T3's ranked rule (bare if it survives; else horizon survivors; if ALL die,
   maximize alive frames; ties by B8 Q) and distil preferences between evaluated actions — restoring the class
   of the 33 alive-extending changes. Teacher v2 currently picks bare when no alternative survives. Fable:
   decisive labels only where an alternative survives the full horizon; no longest-alive labels (the 33 sit
   in the 6 worlds that died anyway, gains 1/2/4 frames).
2. The deciding test. Astra: FRESH disjoint banks, 16 H5000 worlds/mix + 16 continuous-respawn mixed-Watch6
   sessions, three arms (bare B8 / offline-unlimited successor / same successor paced at 24 fps), all contract
   skill+product+timing bars (+40 mass LB90>0, zero overruns), P7 inference, cap 16 h elapsed / 32 CPU-h, after
   Phase 0 frees the Mac. Fable: the 64-world T3 dense bank, Path B v2 online vs bare B8 at real 24/12 fps,
   GO >= 50/64 survival, ~1.5 h per cadence.
3. Smaller: Astra says ONNX saves only 3-5% on this path (not 2.5x), and that shadow catch-up swaps
   process-global RNG (must be isolated before threading); confirm+advance ~42% of a core at 24 fps (mixed).
