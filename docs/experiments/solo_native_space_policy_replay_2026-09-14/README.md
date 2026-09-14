# Spacious-teacher labels on saved learner trajectories (AH)

All three completed AG final policies repeatedly missed spacious-teacher turns while an advisory-safe action was still available. The predeclared diagnostic decision is **ADVANCE_TARGETED_AUGMENTATION**. This supports testing training on learner-visited states; it is an association on recorded trajectories, not proof that any particular alternate action would rescue a game.

AG remains INCOMPLETE / INCONCLUSIVE because of its preserved missing midpoint. AH uses only the three independently completed final-500 endpoints and does not fill or rescore that missing result. Apex remains incumbent.

## Method and frozen decision

For seeds2026093001,2026093002,2026093003, rebuild the exact96 held-out worlds and replay all256 frames using archived student actions. Require initial food and all14 native archive fields to match exactly. Query frozen old greedy and AF SpaceTeacher only at live prepared decision states; step neither teacher's action. A recording delegate exposes the mask actually filtered by SpaceTeacher without implementing a second flood-fill calculation.

The first valid decision with an empty `legal & advisory` action set is F. If it never occurs, effective F is the256-frame horizon and the observed boundary remains null. A decision is actionable only when valid and strictly before F. D is the first valid death transition. Ordinary rows have unchanged masks; latent-veto rows have a changed mask but unchanged greedy direction; behavior-changing-veto rows have different spacious and old-greedy actions. Student direction is `action % 3`; same-direction boost is counted separately.

The primary window is the16 decisions before F in death lanes;4- and64-decision windows are descriptive. Survivors contribute to overall pre-F rates, but not death windows. Every death lane remains in the missed-veto fraction denominator, including deaths without F. Zero denominators produce null rates.

Advancing requires every seed to have at least32 behavior-changing-veto rows before F, at least four worlds, both teacher turn labels0 and2, missed behavior-changing vetoes in at least50% of death lanes within16 decisions, and veto directional error at least10 percentage points above ordinary error in the same windows. These conditions were fixed before replay.

## Every seed

| AG training seed | Behavior-changing veto rows before F | Worlds / teacher labels | Directional errors on those rows | Death lanes with missed veto in preceding16 decisions | Veto / ordinary error in primary windows |
|---|---:|---|---:|---:|---:|
|2026093001|163|8 / left, straight, right|141/163 (86.50%)|15/25 (60%)|100% /27.47%|
|2026093002|196|8 / left, straight, right|144/196 (73.47%)|9/15 (60%)|100% /21.78%|
|2026093003|73|8 / left, straight, right|58/73 (79.45%)|28/40 (70%)|97.30% /36.69%|

All five advance checks pass for each seed. The primary-window veto-minus-ordinary error gaps are72.53,78.22,60.61 percentage points. All80 recorded deaths occur on the first fallback transition: F equals D, with zero earlier fallback events, zero fallback survivors, and zero deaths without fallback. The last missed veto before these selected deaths occurs1–6,1–2,1–12 decisions earlier by seed. This establishes a pre-fallback label opportunity for many deaths; it does not establish recovery success or describe every failure.

The original learned food and survival outcomes are unchanged. The cumulative-food plot below is reconstructed from the same existing native archive; AH contains zero additional optimizer updates and zero new student action decisions. Teacher label queries cover22442,23052,21930 valid decisions. The replay validates all14 fields, including stale values on invalid dead-lane rows, while diagnostic denominators use only valid decisions.

## Qualification, audit and compute

Nine guarded qualification tests passed in2.267 seconds of wall time, within60 seconds. They include an excluded12×16 replay, separate action/state corruption rejection, direct-vs-recorded teacher output parity on a prepared native state, invalid teacher action rejection, and synthetic window/boost/censoring/decision cases. An unexecuted first input freeze was preserved when static review identified stricter endpoint, receipt and censoring checks. There were no failed numerical qualification attempts.

Scientific budget: three serialized15-second CPU replay jobs plus a10-second analysis/render job,55 seconds total. Actual replay times were7.637773,7.812962,7.212316 seconds; analysis/render took1.437543 seconds. Total24.100594 seconds. All four jobs completed naturally; no reruns or partials. Peak process RSS was291,274,752 bytes and minimum available system memory24,479,956,992 bytes. Existing4GiB RSS/12GiB available/8GiB MPS limits, two CPU threads, one inter-op thread, shared compute locks, and heartbeat watchdog remained active; no MPS computation was used.

The reducer rechecked original archive identity, diagnostic shapes/dtypes/actions/masks, raw validity/action/death agreement, report provenance, frozen source/teacher code, exact launch commands and resource receipts, serial ordering and budget. Root inspected both rendered PNGs. Representative lanes follow a fixed selection rule: first lane with a primary-window missed veto, falling back to first death then first lane only if needed. Orange markers show a missed teacher label and do not depict an alternate trajectory.

Intent SHA256: `ec24b3adf27f2f8f612666ce757618ccdb65070fb3af99513355c6c0aa51dcf6`.
Analysis SHA256: `c15b23f37d42549df6ffb21531a535c127d5fb358dfc8b737b21e1735a4261e8`.
Frozen source: `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`.

[Full diagnostic and each seed](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-policy-replay/analysis/report.json) · [Resource reconciliation](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-policy-replay/resource-reconciliation.json) · [Qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-policy-replay/qualification-complete.json) · [Event cases](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-policy-replay/analysis/cases.jsonl)

![Diagnostic and recorded food curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-policy-replay/analysis/diagnostic-curve.png)

![Representative saved gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-policy-replay/analysis/representative-gameplay.png)

Independent closeout audit: **PASS**. The reviewer verified frozen input hashes, all four receipts and budget, qualification evidence, endpoint provenance, and JSON-derived per-seed/pooled counts and gates. It did not independently decode NPZ payloads or inspect pixels; those checks belong to the executed reducer and root visual review. [Audit record](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-native-space-policy-replay/independent-review.json).

## Next experiment

Use a small paired continuation: original expert data versus a fixed mixture containing teacher labels on fresh learner-visited training states, equal optimizer budgets, all three parent seeds. Measure actual greedy survival and food on fresh evaluation worlds, plus fit to original, new, and unseen teacher states separately. The AH held-out worlds are diagnostic evidence and must not become training data. This is one bounded aggregation round motivated by the state-distribution issue studied in [DAgger](https://proceedings.mlr.press/v15/ross11a.html); repeated imitation alone is not evidence that a complete DAgger procedure has been implemented.
