# Controlled-encounter world expansion — rejected — 2026-09-21

The planned 256-world data expansion was rejected during collection. A
four-case exact-input alias in shard 4 has incompatible realized early-H4
teacher labels, so the frozen admission rule requires preserving the evidence
and rejecting the whole expansion. The completed 80 worlds cannot be selected
as a smaller corpus, and no qualification, learner training, policy inference,
held replay, confirmation, promotion, or Apex change followed.

This follows the completed [world-breadth learning comparison](../controlled_encounter_world_breadth_learning_2026-09-21/README.md), which remains the source
of the earlier learning curves and representative policy paths. This rejected
cycle has no new learning curve because it made zero learner updates.

## What completed and what did not

Five guarded collection jobs completed naturally: shards 0–3 passed their
collection gates and shard 4 completed with a scientific admission rejection.
They collected 80 of the planned 192 new worlds, 1,920 four-frame teacher games,
7,680 state rows, and 7,680 native frame facts. Every collected teacher game
survived; together they collected 2,276 food contacts.

The remaining 112 planned worlds were not run. The [closeout](#source-records)
retains the status of every planned world, including those marked unexecuted;
they were not removed or treated as negative examples. The combined 256-world
corpus was never admitted.

The proposed comparison was archived as
[`deferred-learning-not-executed`](#source-records). Its three planned learner
lineages never ran: there are no qualification or scientific checkpoints, no
new model inference, and no learned-policy training or gameplay result. The recorded Apex
incumbent SHA-256 remains
`43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93`;
there is no promotion decision beyond the existing incumbent. Apex remains in
place under the shared tournament gate; these results do not establish inherent
superiority, and the historical training doses differ.

## The rejected alias

The only failed collector gate was the frozen requirement that every
within-shard exact-input group retain a nonempty raw-target intersection. Cases
3416–3419 are all from world `2026115071`, `cross_right`, headings 0–3, at
zero-based step 2 after the action prefix left then straight (`0, 1`). Their
observation arrays and action mask are byte-identical, with exact input hash
`9aadef521a436883ee2fd1a7c5fa41e6a976c2590bcd7c428c464f03551829d4`.

All normal first actions—left, straight, and right—remained viable for the
four-frame horizon. The teacher's raw target is nevertheless right for cases
3416–3418 and straight for case 3419. The maximal total food counts for
left/straight/right are respectively `0/1/2` for the first three cases and
`0/2/1` for the fourth. Each target is a singleton, so this is not an early
tie: the raw-label intersection is empty (`{right} ∩ {straight} = ∅`).

The independent audit recomputed the NPZ and native joins, typed input bytes,
and early-H4 path summaries. It confirmed that the four rows and masks are
identical and the conflicting target facts match the saved collector record.
This is a four-member counterexample to a consistent realized early-H4 label
under the retained observation. It does not measure conflict prevalence in the
112 unexecuted worlds, establish an RNG cause, or demonstrate a learning
algorithm failure.

## Frozen decision boundary

The [frozen intent](#source-records) required all 192 planned worlds to pass
admission, including nonempty raw-label intersections, with the explicit rule:
preserve a scientific failure, reject it, and never select or drop worlds or
rows by teacher outcome or relax the threshold. The [planned learning
criteria](#source-records) were declared before collection and required complete
corpus admission plus a separate immutable learning launch before the matched
repeat-64 versus expand-256 comparison could start.

Those criteria therefore remain guidance for a future valid study, not results
from this cycle. The narrow next question is a prospective four-by-three
RNG-only probe of this witness. At this closeout, that probe had not run.

The frozen research note drew motivation from separate procedural
train/test-environment work in [CoinRun](https://proceedings.mlr.press/v97/cobbe19a.html)
and [Procgen](https://proceedings.mlr.press/v119/cobbe20a.html): training-world
performance and unseen-world evidence should remain separate. That motivation
does not show that 256 worlds would have been sufficient for this imitation
task, nor does it transfer a result from those environments to Snake.

## Execution record

The five completed receipts charged 764.5210806240793 seconds of the
3,120-second collection budget. There were zero execution failures and one
scientific rejection. Peak RSS was 2,007,810,048 bytes; minimum available
memory was 29,527,769,088 bytes, above the 9.6-GiB reserve. The independent
audit used only stdlib JSON/ZIP/AST/struct/hashlib processing; it ran no fresh
native replay, model inference, NumPy/Torch import, or numerical update.

## Source records

- [Frozen expansion intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-expansion/intent.json)
  — SHA-256 `254170bd0936ecc6cab6710d4179692ee0e3bb14ad1312b339fabf969a95d8ea`
- [Frozen prospective learning criteria](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-expansion/learning-criteria.json)
  — SHA-256 `69814e5ded5fda4ba8adfc8fca3f00aae0d41f0fd1aa64ef654877a8e1896fd4`
- [Rejected audited closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-expansion/closeout.json)
  — `REJECTED_AUDITED`, SHA-256 `906bbc75f7a297f50d432b99e98629c1fb5315d77e526fe915b29d3a9c96f313`
- [Independent rejection audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-expansion/independent-rejection-audit.json)
  — `COMPLETE_AUDITED`, SHA-256 `f26cf45ca0197d35b9f38db97d29b9c40c6f491f04fde56940c5a27f0adff322`
- [Frozen research note](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-expansion/research-note.md)
  — SHA-256 `ec73810dbf5cebdddab6edf8a860c6ecf6c3f841057d7cf78d0ee90739e1ecc4`
- [Deferred, unexecuted learning archive](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-world-expansion/deferred-learning-not-executed/README.md)
