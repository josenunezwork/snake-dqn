# Documentation — start here

Read in this order. Each page has one job; none repeats another.

1. **[STATUS.md](STATUS.md)** — one page: goal, what is being built, what is running, what is served, binding rules.
2. **[FINDINGS.md](FINDINGS.md)** — what the project has learned, in plain language.
3. **[INDEX.md](INDEX.md)** — what the project is (model stacks, source trees) and how to run things.
4. **[MAP.md](MAP.md)** — one line for every file and folder under `docs/`.
5. **[TIMELINE.md](TIMELINE.md)** — dated milestones from June 2026 to now.

Code: live research packages are listed at the top of [../research/INDEX.md](../research/INDEX.md); product code (served Watch/Play) is described in [PRODUCT_CODE_GUIDE.md](PRODUCT_CODE_GUIDE.md). Closed research lines: [research/history/INDEX.md](research/history/INDEX.md). Vocabulary: [research/GLOSSARY.md](research/GLOSSARY.md), [notes/GLOSSARY.md](notes/GLOSSARY.md).

Dated records keep their original outcome; they are evidence, not permission to resume an old plan. The per-document status tables that used to be on this page duplicated [MAP.md](MAP.md) and were removed on 2026-11-11 (see git history).

## Reconciled discrepancies and source precedence

The newest decision record and `STATUS.md` describe current state. A branch launch note,
an artifact namespace, a campaign's initial registration, or an older external memory may
preserve an earlier state without changing the recorded outcome.

| Discrepancy | Current interpretation |
| --- | --- |
| Older release and redesign descriptions | Champion+v8 was replaced by **frp3-s12+v8 on 2026-10-06**, then by **b8-s3 without a veto on 2026-10-17**. The July raster/PQN blueprint is historical; the October main line is **GridBatchSim + ego2s-b + distilled DQN**, with PQN deferred. See [STATUS.md](STATUS.md). |
| Old 5x RunPod rule and “always serverless” preference | The threshold was amended to **at least 4x wall-clock per step**. Waivers below 4x are granted by the user case by case and listed in [STATUS.md](STATUS.md#operating-rules-current). When RunPod is used, cheap pods are checked first; serverless is only a fallback because its real cost has run at about 6–10x the catalog rate. The Mac remains available on AC. See the external `~/.claude/projects/-Users-josenunez-Projects-ml-snake-dqn/memory/runpod-5x-rule.md` and `runpod-cheap-pods-first.md`. |
| Strict gates on RunPod | The [strict-on-RunPod amendment](research/governance_amendment_strict_on_runpod_2026-10-05.md) is **NOT RATIFIED and was superseded on 2026-10-08**. It was never used: the FRP-v3 strict gate ran on the Mac. Mac↔x86 bitwise identity fails at random on exact float32 ties. The amendment is hash-bound by template v3 `prepare`, so its "Pending" footer stays unedited; the [status note](research/governance_amendment_strict_on_runpod_2026-10-05_status_2026-10-08.md) governs. Future strict-on-pods use needs a new amendment with a tie-explaining identity check and a per-world single CPU model. Strict gates and serving qualification stay on the Mac. |
| FRP-v5-S versus FRP-v5-S2 | They are different studies. **v5-S** did not pass fidelity and did not train or enter Phase R; fresh-world **v5-S2** recorded `GO_S_RECIPE`, not a gate or promotion. |
| S2 date versus its `20261009` artifact namespace | The namespace is not the result date. The recorded S2 merge/result date is **2026-10-08**. The suffix is a namespace label; see the note in the [research index](research/INDEX.md). |
| LH-1 appears as `DRAFT` in an opening document | The amendment was **ratified 2026-10-05 (`ecddfab`)**; LH-1 recorded `CLEAR` for FRP-v3. The document is hash-bound by the FRP-v3 LH-1 screen intent, so its header stays unedited; see the [erratum](research/history/colab-and-process.md#governance). Do not relabel the result from an opening-state label. |
| Campaign `training_bet_*` entries still say `REGISTERED_NAMESPACES` | Those `status` fields are initial campaign metadata and are never rewritten. On 2026-10-08 a `final_outcome` field (status, date, summary and pointers) was **added** to `training_bet_frp_v4`/`_v5h`/`_v5s`/`_v5s2`, `governance_strict_on_runpod`, `sequential_strict_gate` and `sequential_strict_runner`. Read `final_outcome` for the result. |
| “CNN variants lost” in older prose | Retracted by git archaeology: the **June CNN code** was never trained and produced no checkpoints. Only the older GRU/DRQN result exists; this does not describe later raster/PQN or October CNN-student work. See [the blueprint correction](ml_redesign_blueprint_2026-07.md#corrected-historical-record). As of 2026-10-08 every repository occurrence on main is already a retraction or carries a correction (README, AGENTS.md, `ml_algorithm.md`, `h100_training_recipe.md`, `project_history_and_findings.md`). |

The external current-state source is
`~/.claude/projects/-Users-josenunez-Projects-ml-snake-dqn/memory/` (especially
`standing-decisions-2026-10-08.md`, `redesign-scope.md`, and the dated result notes).
Its entry point is `MEMORY.md`. The read-only artifact evidence is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/HANDOFF_2026-10-05.md` and
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/campaign.json`;
the JSON is campaign metadata. Its existing fields must not be rewritten; additions such as `final_outcome` are allowed. The companion RunPod material is
`~/.claude/skills/runpod/{SKILL,NOTES,OPTIMIZATION,HARDWARE,CPU_PROVIDER_COMPARISON}.md`.
These external locations are not versioned in this repository; use the linked repository
records for reviewable, commit-pinned history.

## Status labels

- **Current** describes the maintained release, contract, or decision record in its stated
  scope. It does not turn an old study package into a new run authorization.
- **Historical** preserves a dated result, audit, plan, or handoff. Its numbers and verdicts
  remain attributable to the recorded protocol.
- **Superseded** means a later record replaced the document as an active plan or status source;
  its provenance remains useful.
- **Conditional** means a record has stated prerequisites or ratification conditions. Read the
  record before applying it.
