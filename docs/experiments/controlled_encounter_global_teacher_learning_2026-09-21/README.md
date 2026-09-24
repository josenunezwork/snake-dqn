# Controlled-encounter global-teacher learning — 2026-09-21

The admitted global-teacher expansion produced a positive paired held-food
comparison in all three warm-started lineages, but it did not meet the fixed
absolute or all-three early-held requirements. Fresh confirmation was therefore
not permitted, and this study does not promote or replace the incumbent Apex
policy.

This is the matched continuation of the admitted
[global-teacher union](../controlled_encounter_global_teacher_2026-09-21/README.md).
Both arms began from the same stored mark-512 parent network **and full Adam
state** in each lineage, then continued to mark 1536. `repeat64` sampled the
7,184-row, 64-world parent corpus; `expand256` sampled the 25,616-row,
256-world admitted union. The intervention is the complete expanded-targets
package, including its target-construction change. It does not isolate world
count from that change, and it is not training from fresh initialization.

## Final behavior at mark 1536

All six continuations reached 100% membership on their own training corpus.
The fixed held evaluation has 192 games across the eight reused development
worlds, so it is adaptive development evidence, not fresh confirmation.

| Lineage | Held food: repeat64 → expand256 | Held survivors: repeat64 → expand256 | Early-held membership: repeat64 → expand256 | Expand256 early-held fit | Absolute package |
| --- | ---: | ---: | ---: | --- | --- |
| 2026098001 | 152 → 220 | 148 → 180 | 88.02% → 93.23% | Fail | Fail |
| 2026098002 | 144 → 224 | 168 → 176 | 84.90% → 95.83% | Pass | Fail |
| 2026098003 | 160 → 220 | 160 → 172 | 85.55% → 93.75% | Fail | Fail |

The [learning curves](learning-curves.png) begin at the shared mark-512
parents and follow marks 768, 1024, and 1536. They show the two fixed-dose
continuations diverging under their respective sampling packages; they do not
show a fresh-from-scratch comparison. The [representative paths](representative-paths.png)
are the corresponding saved native gameplay traces, not evidence beyond the
fixed eight-world development set.

## Paired comparison and unchanged decision

The paired result is `expand256 − repeat64` food contacts per case over the
eight held worlds. Every lower 95% bound is positive, and all three paired
relative packages pass. Their benign and threat survival checks also pass.

| Lineage | Mean paired food change | 95% paired-world interval | Paired relative package |
| --- | ---: | --- | --- |
| 2026098001 | +0.3542 | [+0.1973, +0.5111] | Pass |
| 2026098002 | +0.4167 | [+0.1481, +0.6852] | Pass |
| 2026098003 | +0.3125 | [+0.1556, +0.4694] | Pass |

The fixed absolute package was not relaxed. It still requires train and held
teacher-membership floors, food retention and directional/exact-random food-gap
checks, and benign, overall-threat, and per-threat-family survival floors. Each
expanded continuation fails that complete package. Although 2026098002 passes
its expanded early-held fit check, 2026098001 and 2026098003 do not; the
all-three early-held condition is false. Thus neither arm is confirmation
eligible and the eight reused development worlds must not be described as a
fresh generalization result.

## Execution record and limits

The three lineages each ran 1,024 updates and 131,072 examples per arm. Across
six scientific arms, that is 6,144 updates and 786,432 examples. The
qualification procedure discarded 16 updates: scientific arms reloaded the
unchanged stored mark-512 parents before continuing.

Nineteen physical attempts produced 17 complete jobs and two preserved failed
attempts. The repairs addressed tuple-versus-list metadata and a fit-report
heartbeat only; successful training outputs were reused. Charged runtime was
926.920379558 seconds of the 3,600-second budget. Peak RSS was 6,859,849,728
bytes and minimum available memory was 28,636,266,496 bytes. The repository
revision for the completed run was `47edd2f`; the frozen source revision was
`1f32d2de1987d97441fa74c1898ed9039b535f24`.

The saved analysis is `COMPLETE_AUDITED`. A separate saved-report audit
independently recomputed all six endpoints, three paired-world intervals,
training-dose and parent joins, and the 19-attempt accounting, with zero issues.
It did not rerun native gameplay, model inference, or NPZ fit calculations.
The consolidated closeout is `COMPLETE_AUDITED_RELATIVE_GAIN_ABSOLUTE_FAIL`;
the root rehashed its complete 9,296-file input closure. No new
confirmation-world gameplay ran.

The next bounded question is where the expanded policies first go wrong in
their 48 remaining fatal development games. Replaying their 168 saved decisions
against cached stationary H4 targets can distinguish an uncovered first mistake
from an earlier allowed alternative followed by a later mistake. That temporal
association alone would not prove the earlier alternative caused the death.

## Source records

- [Frozen global-teacher learning intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-global-teacher-learning/intent.json)
  — SHA-256 `4948dd7ac5a0e6b8beca6adb6a49f3e7699f0ea006e47f130864e34f367ba956`
- [Completed saved analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-global-teacher-learning/analysis/report.json)
  — SHA-256 `b47c06e53f3a759bdf062eedd2b2d3d30ac8beb3435ee87e5384558fe9af3dc8`
- [Saved learning curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-global-teacher-learning/analysis/learning-curves.png) and [representative paths](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-global-teacher-learning/analysis/representative-paths.png)
- [Saved analysis receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-global-teacher-learning/supervisor-runs/analysis/receipt.json)
- [Independent behavior audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-global-teacher-learning/independent-audit.json)
  — SHA-256 `0364349539fbd8fdbfdc758fe3d9cdf54d4bc0f3d70fcb1775e905ae5c4f81e4`
- [Consolidated closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/controlled-encounter-global-teacher-learning/closeout.json)
  — SHA-256 `6fabf1770ed635c71bdf269d14aa214f67e40422ed35a9f0cd046c7a85ff824a`
