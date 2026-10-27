# T34 ruling: ordinary desktop applications during the bare served-tick panel

2026-10-27 · Astra ruling judge · source snapshot: main `8c727cefaa3f9daecb8ea8ffc75f4922e8d6309b`.

**Decision: CONDITIONALLY ADMIT an explicitly enrolled `desktop_app` class for ordinary owner desktop activity. Count it inside the existing shared background ceiling: strictly >150% of one core for strictly >60 consecutive observed seconds. Retain the ordinary >=20% per-process guard for local compute and unidentified processes. Implement and verify this narrow prospective amendment now; prioritize the first natural supervisor-free Mac window for the full bare panel. The unfinished remote-supervision collector is a future capability, not this panel's critical path. Attempts 07 and 08 remain INVALID.**

This is the requested prospective operational ruling for the engineering timing diagnostic. It conditionally approves the rule below, not an unbuilt implementation or a panel launch. It changes no scientific threshold, panel exposure, serving qualification or release authority. Repository code and running jobs were read-only; the sole requested write is this artifact.

## 1. What the evidence establishes

Attempt `panel-argon-classified-08` used source `e7f1f18fa13a7777c6af5cef56a9cf2e7fd83d16`, completed `cell-00.json` through `cell-02.json` of the fixed 12-cell panel, and became INVALID after **259.075595375034 seconds**. Its last CPU interval records Outlook PID **12374**, PPID **1**, birth `Sat Oct 24 11:38:49 2026`, executable `/Applications/Microsoft Outlook.app/Contents/MacOS/Microsoft Outlook`, kind `unknown`, at **26.211639347734064% of one core**. The existing >=20% rule correctly rejected it. Its identity as the owner's ordinary mail application is owner-supplied; the export does not verify a code signature or prove the activity performed.

That interval also records **77.69878806451274%** service/renderer background, **31.828419207984737%** unknown CPU and zero workload CPU, with complete sampled coverage. It already exceeds the 20% low-background threshold. Changing Outlook's prospective classification would not make that interval low-background or establish a valid completed panel.

Attempt 07 records INVALID after **24.379662374965847 seconds**; its final point screen identifies `(ps)` PID **29517**, PPID **55697**, kind `unknown`, at **28.0%**. Attribution to the ops renicer, and the subsequent pausing of renicer/watchd during panels, are owner-supplied. An unrelated ops helper receives no desktop-app exemption.

Sources: `served-tick-20261026/panel-argon-classified-{07,08}/{host.json,attempt.json}` beneath `/Users/josenunez/Projects/ml/snake-dqn-artifacts/`. I inspected operational evidence and cell-file inventories, without opening cell timing outcomes. No historical cell may be salvaged, rescored, or combined with a later attempt under this ruling.

## 2. Identification rule: explicit software identity and permitted role

Use a sealed registry, frozen before the next attempt, rather than a blanket `/Applications/*.app` rule. An executable receives `desktop_app` only when **all** of these conditions hold:

1. **Enrolled installed application.** The registry identifies a specific owner-used application under canonical `/Applications/` or `/Users/<owner>/Applications/`. Record permitted process UID, canonical bundle path, bundle identifier, version/build, verified signing identity and designated requirement, Team Identifier where applicable, and code/executable digest. Require a valid trusted signature; an ad-hoc signature or an unverified bundle does not establish admission. Resolve symlinks and reject path traversal, copies outside the enrolled installation, and identity mismatches. The display name, `.app` suffix, argv text, PPID 1, or an owner-UID match alone is insufficient.
2. **Exact executable role.** Enroll the main GUI executable and each permitted UI helper separately by canonical bundle-relative executable path and identity. Eligible roles include ordinary mail, browsing, remote-chat presentation, notifications, UI rendering and ordinary application synchronization. No recursive whole-bundle trust; no inheritance by descendants, shells, plugins, XPC services, generic runtimes or arbitrary helpers. A helper needs its own enrolled UI role and verified identity.
3. **Running-process match.** Match the OS-observed executable to the enrolled identity and require its process UID to equal the registry's permitted UID: the owner's session UID, or a separately justified enrolled UI-service UID. Retain UID, PID, process birth identity, executable identity and role evidence. Cache expensive verification outside scored ticks and revalidate on new process identity, exec or executable replacement. A restart may qualify under the same frozen entry after fresh verification, with churn recorded. An application update cannot inherit the previous version's admission; invalidate an attempt if its admitted identity binding changes. Missing evidence gives no background exception.
4. **Workload exclusions take precedence.** Apart from the existing exact probe-owned exclusion, known local workloads remain workloads regardless of bundle location, signature, parent or GUI launcher. This includes Python, node, CLI agents, pytest/uv, repository/research workers, supervisors, builds, tests, training, local inference, simulations, benchmarks, bulk data processing and deliberate compute/export/transcode jobs. An IDE or chat frontend cannot confer permission on its local agent, language-server, indexing/build worker or embedded compute engine. Known substantive in-process work receives ordinary-work treatment for the whole hosting process when it cannot be separated.

The first implementation should enroll Outlook's exact main GUI executable after verification, plus the actual ordinary browser/chat/UI executables needed for the owner's working environment. The old renderer path/suffix heuristic must also become explicit identity-and-role enrollment in this new policy; otherwise a fabricated installed `(Renderer)` or `(GPU)` remains a bypass. Unenrolled executables stay `unknown` or `agent_workload` as appropriate. Enrollment is based on role and identity before exposure, never on which process happened to trip a panel or which enrollment yields a favorable timing result.

**Browser and Electron limitation:** a genuine signed GUI executable can itself run local computation. Signing authenticates software identity, not harmless activity. Ordinary browsing/remote chat is eligible; deliberate compute jobs implemented in JavaScript/WebAssembly/WebGPU, browser-hosted local ML or automation/benchmark work are not. Record the operator's permitted-use declaration and known active workload exclusions before admission. If the actual role is ambiguous, withhold the exception. Detection of prohibited compute under an admitted desktop role during exposure invalidates the attempt; do not relabel it after seeing timing. This controls accidental contamination on a cooperative owner's Mac; it cannot prove the absence of concealed in-process compute. No CPU threshold or executable-name rule can provide that stronger guarantee.

## 3. Class, ceilings and coverage

All percentages refer to **one core**, without dividing by the Mac's core count. These are prospective engineering admission limits, not measured harmlessness thresholds or Colab budget envelopes.

| Category | Prospective rule |
| --- | --- |
| Verified `desktop_app` | Background contribution; no ordinary per-process 20% abort in either point or interval path. Retain its raw CPU evidence. |
| macOS services + enrolled desktop renderers + enrolled desktop apps | **One shared aggregate pool**, invalid only when strictly **>150%** continuously for strictly **>60 seconds** under the existing observed-interval rules. No separate allowance per application or category. |
| Ordinary workload or unknown | Preserve **>=20% per process** in both historical point screens and interval CPU deltas. A GUI parent grants no exception. |
| Remote supervision | Remains ordinary workload unless admitted under the separate, fully implemented and qualified remote-supervision ruling. Desktop enrollment cannot admit Python polling. |

Exactly 150% or exactly 60 seconds does not cross the shared ceiling. A sub-ceiling interval resets its duration; gaps cannot establish continuity. Keep nominal 1-second sampling, the existing 5-second maximum coverage interval, CPU-counter/birth/exec checks and churn handling. Churn must not erase measured high background CPU. Missing coverage is never zero CPU. This ruling does not add an aggregate ordinary-work abort or otherwise silently revise the existing ordinary-work predicate.

Do not stop, renice, hide or disable the owner's desktop applications to manufacture a quiet result. Record their actual interference. The whole panel still ends INVALID if the combined background ceiling or another operational guard is crossed. A valid panel can still FAIL its unchanged timing bars.

## 4. Reporting and implementation acceptance

Create a sibling amendment and a distinct policy identifier, for example `classified-desktop-v1`; do not change the meaning of `classified-v1` or claim desktop admission through `classified-supervised-v2`. Any later combined policy must explicitly adopt both sets of rules and preserve supervision's separate accounting and limits.

Bind this ruling, the registry, amendment, actual execution-worktree source and policy into the intent and immutable payload/SHA/completion chain. Propagate identical classification through preflight, historical point screens, interval monitor, raw export and independent reducer. Relevant current boundaries are `probe_host.foreign_load`, `host_load.process_kind`, `host_load.cpu_interval`, `host_load.LoadGuard`, and `background_report`'s accepted kinds and independently reconciled totals. Changing only `BACKGROUND_KINDS` is insufficient.

Report per-process and per-application identity/role, admission or rejection reason, CPU, coverage and classification changes. For each cell and overall, report service, renderer and desktop-app CPU separately, their **combined background** total, ordinary-work/unknown CPU, sustained-ceiling duration/maxima, and affected tick counts. Preserve raw point CPU and interval evidence, aborts, registry mismatches and all prior-attempt dispositions. Do not publish sensitive mail/chat contents or credentials to establish an app's role.

Retain **every measured tick and client receipt** in authoritative timing distributions. No CPU subtraction, removal of app-overlap ticks, favorable-cell repeats or timing-threshold adjustment. The descriptive low-background subset requires complete original coverage and combined background **<=20% in every overlapping interval**, workload and unknown totals each **<20%**, and all existing churn/gap exclusions. Keep consecutive original receipt identities for client-gap analysis; never join sparse retained receipts. Empty subsets retain count zero and null statistics. This subset cannot turn FAIL into PASS.

Before a new full panel, require source-bound offline tests and independent implementation review covering: genuine enrolled Outlook/UI identities; same-name or copied executables, wrong signatures/paths/UIDs, traversal/symlinks, helpers and PID reuse/exec; local compute beneath or inside apps; enrolled versus fabricated renderers; both >=20% abort paths; shared >150%/>60-second exact boundaries; churn and missing coverage; reducer reconciliation/policy mismatch; and all-tick/low-subset/client-gap behavior. Add a bounded host-only integration observation with the actual enrolled apps before measured exposure. These checks establish implementation conformance, not causal timing neutrality. Tests and this observation were **not run in this read-only ruling**.

## 5. Priority: desktop amendment, then the next supervisor-free window

**Take the next natural supervisor-free window before waiting for the collector.** The current `REMOTE_SUPERVISION_AMENDMENT_20261027.md` explicitly states that the complete live observer is unimplemented, offline validity is not live admission, and the existing root cannot be repaired by longer observation without unavailable/prohibited instrumentation or restart. The reported Argon `ACCEPT-WITH-FIXES` is owner-supplied review status; it does not remove that source-recorded blocker.

| Priority and dependency | Placement and stop boundary | Decision enabled |
| --- | --- | --- |
| 1. Implement, test and independently accept the narrow desktop policy | Bounded Mac engineering work outside a panel; no live job changes | Removes the ordinary-GUI blocking condition prospectively |
| 2. At TRAIN-01's natural terminal boundary, finish export, verification, cleanup and qualified stop; then confirm local supervision/transfer work has ended | Existing qualified operator; keep lease, watchdog, heartbeat and safety polling intact throughout | Establishes an actual supervisor-free Mac window |
| 3. Once 1 and 2 hold, run one fresh, complete, unchanged bare panel | Mac CPU, nice 15, five thread caps at 1, AC/open lid, lease/thermal/heartbeat guards; existing **45–60-minute forecast**, **120-minute lifetime cap** | First complete engineering timing verdict under the declared ordinary-working-Mac environment |
| 4. Separately complete the collector for future instrumented roots | Independent engineering lane outside scored Mac timing; original supervision tests and **10-minute live qualification** still required | Future panels may coexist with admitted useful remote jobs |

An idle GPU alone is insufficient: an allocated runtime with an active polling supervisor, or terminal downloads/hashing/cleanup, can still block the Mac. Do not suspend supervision while TRAIN-01 runs, stretch safety cadences, restart its root or terminate useful training to obtain a panel. At the natural job boundary, the operator should reserve the panel window before the next discretionary dispatch where campaign obligations permit. No provider query, runtime stop or scheduling change was performed here.

The 45–60-minute value is the existing README forecast, not a measured duration recovered from these partial attempts. Current TRAIN-01 remaining time and collector completion time were not independently measured; no clock-time completion promise follows. If no supervisor-free window can be scheduled, collector work becomes the prerequisite for overlap with a **future fully instrumented root**, not a bypass for the current root. Preserve useful remote work and its safety obligations meanwhile. This recommendation is about elapsed time to valid evidence, not rationing Colab credits; newer no-unit-budget standing instructions supersede older credit ceilings.

## 6. Evidence bindings and read-only closeout

Inspected repository main at the SHA above; initial dirt was only untracked `exp/mac-lease.json.live` and `exp/mac-lease.json.lock`. Reviewed campaign authority in `AGENTS.md`, `docs/STATUS.md` and `SUCCESS_CONTRACT.md`; the served-tick README, guard, point monitor, reducer, focused test sources, both amendments, previous supervision ruling and implementation handoff. An independent read-only reviewer challenged executable-versus-activity identity, renderer admission, both guard paths, reducer consistency and current-root feasibility; those findings are incorporated here.

Key SHA256 bindings:

- Attempt 08 `host.json`: `6e5bd793e1b1dd07113276439185962f90ad8160b4563914c57d9592450f47dd`.
- Attempt 07 `host.json`: `0406193476e36c1de4bb152f01c8a7a2efdc0e0b616fdc67a3525631bae0db15`.
- `host_load.py`: `b8720e1d87e389b736d010aafbb9265fb2536f754f8931539ae7f1c56a0b91f1`.
- `HOST_LOAD_AMENDMENT_20261027.md`: `ef6469cee5968ca1be3d53fb698594f018497190860acce86f99ab88bacf3709`.
- `REMOTE_SUPERVISION_AMENDMENT_20261027.md`: `5aefbbd0fef00241e755a64e246d9ed6939ca5312c0f3fd7c4fb928917c499df`.
- Prior `astra-guard-supervisor/RULING.md`: `83e47ff2b3cfddba0d9ee4a6ec2bab48efb645259857d7e81ea5fd23459f7426`.

Verification here consisted of read-only Git/source inspections, standard-library JSON parsing and SHA256 calculation, and independent review. No gameplay, test suite, app signature qualification, live provider query, process control, source edit, Git mutation or release action was performed. This ruling grants no retroactive validity to incomplete attempts and no authority to change the served B8 baseline.
