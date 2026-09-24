# Native solo256 imitation of the spacious teacher (AG)

All three fresh training seeds learned substantial food collection and fit the selected training examples perfectly. None met the final survival thresholds or held-out action-fit criteria. The experiment remains **INCOMPLETE / INCONCLUSIVE** under its declared rule because seed2026093001's midpoint evaluation stopped at the watchdog. Its completed final evaluation is retained, with all known gates reported. No completed or partial scientific run was repeated.

Apex remains incumbent. This diagnostic is not a shared-tournament promotion result.

## Frozen question and design

Can the unchanged full-raster network learn the qualified spacious teacher and repeat its food-seeking and survival behavior in native256 solo games? This is expert-trajectory behavioral cloning, not policy-state aggregation. No external spacious-teacher veto was applied to learned gameplay.

The existing native simulator, prepared-decision observation pipeline, raster31v3 triple, six-action resolved mask, and RasterDuelingNetwork were retained. Native mechanics use a145×83 cell arena, one initial body segment,300 ambient food, body capacity400, terminal hero death, and unchanged16/500/150 observation normalizers. Each world has12 poses: four starting headings by three relative food placements.

Training seeds are2026093001,2026093002,2026093003. Training worlds2026101200–1215 and held-out worlds2026101300–1307 are disjoint; their native world seeds are disjoint too. A world's native seed is intentionally shared by its12 poses, while lane IDs are unique. The excluded qualification seed/world are2026093009/2026101399. The106-file prior-seed inventory found no collisions.

Training used3072 selected rows,192 per world. Selection stratified teacher-vs-old-greedy disagreements across time quarters/actions, then filled from ordinary teacher states. There were122 disagreement rows across15 training worlds. Held-out evaluation used6144 natural sampled rows plus29 disagreement-only extras; its union contains36 disagreement rows across five worlds. Both splits include left and right disagreement labels and have zero conflicting labels for byte-identical full observations and masks. Training fit and held-out natural/disagreement fit are reported separately.

Each seed received500 Adam updates at learning rate0.0005, epsilon0.00015, batch256, gradient clip10:128000 presentations and41 complete shuffled epochs plus eight batches. Checkpoints were saved at0/125/250/500; actual greedy gameplay was planned at0/250/500 on all96 held-out lanes for256 frames.

## Criteria declared before data collection

All three seeds were required to pass jointly. Training accuracy≥98%, macro recall over normal actions≥95%, disagreement accuracy≥95%. Held-out natural accuracy≥98%, macro recall≥95%, each world and each time-quarter accuracy≥90%, and disagreement accuracy≥90%.

The seven gameplay checks were time alive≥95%; endpoint survival≥90% (at least87 of96); food≥75% of the teacher; food in every heading/placement cell≥50% of its teacher cell; paired eight-world95% t-interval lower bounds above zero versus random and versus the seed's initialization; and final teacher-normalized food no more than0.05 below its midpoint. These thresholds were not relaxed.

Calibration passed all six declared checks. The teacher collected47.6354 food/game and survived96/96; random-safe collected4.71875 and survived96/96. Food collection therefore offers substantial headroom while random survival alone is easy.

## Every training seed

| Training seed | Initial / midpoint / final food per game | Final time alive | Alive at256 | Train action fit | Held-out natural fit | Held-out disagreement fit | Final behavior |
|---|---:|---:|---:|---:|---:|---:|---|
|2026093001|1.7917 / unavailable /32.6354|91.215%|71/96|100%|86.458%|25.000%|Three known failed gates; retention unavailable|
|2026093002|6.5208 /32.1250 /30.6875|93.738%|81/96|100%|86.833%|58.333%|FAIL: food fraction and both survival gates|
|2026093003|0.1250 /35.9688 /35.8021|89.071%|56/96|100%|85.433%|44.444%|FAIL: both survival gates|

All three pass every training-fit check and fail every held-out-fit check. All three final models improve food versus initialization and random with positive paired-world confidence intervals. Final-minus-random food deltas are27.9167 [25.7752,30.0582],25.9688 [20.0354,31.9021],31.0833 [26.0817,36.0849]. Final-minus-initial deltas are30.8438 [28.0917,33.5958],24.1667 [18.0474,30.2859],35.6771 [30.9852,40.3690]. These intervals use eight worlds, not96 nominally independent lanes.

Every final death was classified by the native simulator as self-collision:25,15,40 by seed. This report does not yet attribute which earlier choices caused the deaths. The teacher's perfect survival on its own trajectory does not establish that it can recover every state visited by a learner.

## Receipts and limitations

All three training runs and all three final evaluations completed naturally. Eight of nine planned checkpoint gameplay evaluations completed. The original step250-seed2026093001 job stopped after20.113386 seconds with `watchdog_timeout`, confirmed SIGTERM, and no report or frame archive. Four partial fit Q archives exist and are preserved but were not salvaged into this analysis. The missing125/250 fit points and250 gameplay point remain absent; retention is null, not a pass or failure.

A versioned evaluator added resource heartbeats around fitting before the five unused evaluation jobs. One guarded qualification test compared its Q values and all14 native gameplay fields against the already completed excluded smoke, with exact equality and unchanged network weights. This instrumentation amendment did not change learning, observations, action selection, budgets, or criteria. The original evaluator and partial artifacts remain immutable.

The final reducer revalidated selected datasets, raw gameplay arrays, checkpoint hashes, saved Q predictions and fit calculations, receipts, and source/input closures. The learning curve below uses a separate render-only version to make missing-point gaps explicit; the original analysis and figures are retained. Representative paths use the first held-out world's fixed left/straight/right lanes, rather than selected best games.

## Compute and evidence

The budget was675 scientific seconds and120 qualification seconds, declared before execution. Actual scientific wall time was271.288449 seconds including the stopped job, calibration, final analysis and rendering. Qualification consumed9.498626 seconds:25 tests plus three excluded smoke jobs, with no failed qualification attempts. There were17 complete scientific jobs and one preserved partial, serialized under the existing two CPU-slot locks. Peak scientific process RSS was1,430,241,280 bytes; minimum available system memory was28,408,741,888 bytes. The limits remained4GiB process RSS,12GiB minimum available memory,8GiB MPS driver allocation, two intra-op threads and one inter-op thread,20-second heartbeat watchdog. The Mac's64GiB/18CPU configuration was checked; macOS reported no recorded thermal/performance warning, which is not a temperature measurement.

Frozen simulator source: `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.
Intent SHA256: `79c46d819741d60e99fc6dcd6285a3c5ce6a8ee2724ee8bb657c509e00cb0e17`.
Bank SHA256: `3a2f6f170a4d3415993a10405f1b10bce7eebef0769d37139cd738359ea2edaf`.
Analysis SHA256: `0524a58dda2bf8ee100b5880795ee57d87fdbd06faf0569243fb3cbc8da367dd`.
Apex checkpoint remains `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93`.

[Full analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-bc/analysis/report.json) · [Receipts and resources](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-bc/resource-reconciliation.json) · [Qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-bc/qualification-runtime-complete.json) · [Runtime amendment](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-bc/runtime-amendment.json)

![Learning and greedy gameplay curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-bc/render-v2/learning-curve.png)

![Representative recorded native gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-bc/analysis/representative-gameplay.png)

Independent closeout audit: **PASS_AUDIT_STUDY_INCOMPLETE**. Its seven checks verify source/intent/analysis identity, both final frozen maps, all18 receipts and resource totals, qualification aggregates, per-seed scalar/gate summaries, and the fixed heartbeat amendment. The auditor did not decode raw NPZ or model tensors or inspect pixels. The initial audit artifact is preserved as FAIL_AUDIT because its script misread the seed-inventory and qualification schemas; the versioned correction fixes those audit predicates without changing study evidence. Qualification totals are21 initial tests and25 cumulatively after instrumentation qualification, not46 tests. [Corrected audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-bc/independent-review-v2.json) · [Audit correction record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-bc/independent-review-correction.json).

## Next decision

Replay the three completed final trajectories using their saved actions, verify exact simulator parity, and query the teacher on each prepared learner state. Separate ordinary food-direction disagreement from the spacious veto, and measure disagreement before the safe action set disappears. This tests whether corrective examples exist in states the learner actually visits. It does not establish the effect of taking a different action; any later policy-state aggregation trial needs fresh training data and fresh evaluation worlds. The distinction follows the state-distribution issue studied in [DAgger](https://proceedings.mlr.press/v15/ross11a.html), without claiming this BC experiment implements that algorithm.
