# F5 kernel provenance amendment: ADOPTED, PRE-DATA

2026-10-26 · Astra, independent F5 judge · inspected clean detached source `ad207314bcccc037121b6da03c3aa296d3e1c0ac`.

**ADOPT the narrow prospective v6 amendment below.** Adoption takes effect now as a rule; admission of any new execution remains conditional on implementation acceptance and fresh qualification. This is not a finding that either kernel has already passed workload parity or timing, nor authorization to dispatch, fit, evaluate, change serving, or release. The owner's statement that no F5 fit/evaluation data exist is the pre-data basis; I did not inspect scientific payloads to establish it.

## Authority, evidence and limits

This sibling supersedes only the cross-assignment invariance of `os_kernel.release`, `os_kernel.version`, and the release component embedded in `platform` in the prior L4 assignment-identity amendment and v5 environment policy. Every other rule remains binding.

Rehashed authority:

- `F5_REGISTRATION_ADOPTED_20261025.md`: `4e218263f169334415968f02eb2d767e1ff136dfc3ccf470090dd474b31dd6a8`.
- `F5_L4_ASSIGNMENT_IDENTITY_AMENDMENT_ADOPTED_20261025.md`: `a1dc92ffb8fa341fb51738a56fe0097d75e15b0aff7194940c12979adbb04d62`.
- `ACCEPT_F5_FAMILY_CANARY_02.md`: `9bf076600686523e2d4c8fe1bb0c115474074731dacbde908a2b0ec5861e40fd`; its prospective repeat-refusal rule and fresh-canary gate continue to apply.

Outcome-blind diagnostic inspected:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f5-timing-recovery-offline-20261026/canary-02/output-02/exports/environment/cross-assignment-refusal.json`, SHA256 `ae157f89cce650d73bab192acb324987f97a50cc29fed44ee83bc8a09faf48b9`.
It reports exactly `/os_kernel/release`, `/os_kernel/version`, `/platform`, with `admissible=false`, `fit_usable=false`, `scientific_use_authorized=false`, and nested `capture_status=REFUSED`. It records 69 locked wheels and driver `580.82.07`. No package/numerical/CPU/GPU difference is reported by that diagnostic. This is operational diagnostic evidence, not newly executed compatibility proof.

The sealed canary-02 control was read only as environment/control metadata, SHA256 `deffdc47b62ead71245a57c08e82fe68ac630889d573877c9fdd643ed54e7e51`; its reference is SUCCESS capture-05. Observed tuples:

| Receipt | release | version | platform |
|---|---|---|---|
| SUCCESS capture-05 reference | `6.6.122+` | `#1 SMP Thu Apr 30 18:17:14 UTC 2026` | `Linux-6.6.122+-x86_64-with-glibc2.39` |
| REFUSED canary-02 diagnostic | `6.6.159+` | `#1 SMP Thu Oct 22 03:07:54 UTC 2026` | `Linux-6.6.159+-x86_64-with-glibc2.39` |

These observations justify testing a host-build provenance classification. They do not establish numerical identity or equal performance across kernels. Canary-02 remains HOLD/INCOMPLETE under v5. Its approximately 556-second exposure and verified stop are owner-reported here, not independently re-audited. No refused object, timing unit, terminal status or consumed dispatch authorization is rehabilitated.

## Exact v6 comparison rule

1. Create source-pinned `f5_environment_schema_v6.json`, receipt schema `bl-residual-f5-environment-receipt/v6`, exhaustive leaf classifications and a new policy digest binding this amendment's SHA256. Preserve v5 bytes and past seals.
2. Across assignments ONLY, classify `os_kernel.release` and `os_kernel.version` as assignment provenance. Retain their complete original strings, hashes and assignment linkage. No removal of the `os_kernel` object: `system`, `machine`, `processor` remain invariant; the already adopted classification of `node` stays unchanged.
3. Classify `platform` as **verified raw-to-canonical evidence**, never wholly assignment provenance. For this Linux/x86_64/glibc profile, validate its complete raw structure against raw `os_kernel.system`, exact `os_kernel.release`, invariant machine and captured libc identity. Replace only the single structurally identified release component with a fixed sentinel in the cross-assignment projection. Preserve and compare the entire remaining string, including `Linux`, `x86_64`, `with-glibc2.39`, separators and any other retained component. No global string replacement, substring stripping, loose regex, fallback parse or blanket platform exclusion. Ambiguous, inconsistent, empty or malformed representations refuse. Independently capture/check libc identity on the running host so a mutually forged platform/kernel pair cannot satisfy acquisition checks.
4. Within each assignment, **all raw fields remain exact**, including release/version/platform, UUID, roots, driver and affinity, at the existing capture, boundary, unit, prepare and fit checks. Kernel drift within an assignment refuses before a unit is accepted or continuation occurs. This is not a cross-assignment C/W fit-resume permission.
5. CPU model/ISA/vCPU, OS family/architecture, glibc, Python, driver `580.82.07`, GPU/capability/memory, all locked wheels and installed contents, build evidence, source and numerical controls retain their existing checks. Unknown/missing/extra/multiply classified leaves refuse. No driver, CPU, package, precision or hardware waiver is granted.

The new field classification is prospective, not a claim that arbitrary kernels are interchangeable. No search for a preferred kernel is authorized. Qualification may land on either observed build. A different newly observed build requires explicit disclosure and independent review before affected timing/science admission; do not silently enlarge the qualified kernel set.

## Exact Sol implementation scope and required offline evidence

Sol owns the bounded environment-policy repair and its consumers: `research/bl_residual/f5_environment.py`, the new v6 schema, `f5_job.py`, `f5_qualification.py`, `f5_timing_resume.py`, source/package policy bindings and focused tests. Change other control/descriptor/receipt generators only as necessary to bind the same v6 policy and amendment. Keep the qualified CLI/ops transport implementation unchanged; if transport changes prove necessary, HOLD for separate review/requalification.

A new reference seal must bind the original SUCCESS capture-05 bytes plus the v6 comparison policy. If a legacy-reference adapter is needed, permit only an explicit versioned comparison-only projection of that pinned SUCCESS v5 reference, validated under frozen v5 rules first. Preserve its original schema/hash/status and record the derivation; never stamp old raw evidence as a newly captured v6 SUCCESS. Require native fresh v6 runtime receipts. The adapter must reject REFUSED diagnostics, nested refused raw receipts, arbitrary v5 runtime inputs and unpinned references through every prepare/timing/fit consumer. No implicit schema fallback or generalized v5 acceptance.

Before acceptance, demonstrate:

- The two exact kernel tuples above differ under v5 and compare equal under the v6 cross-assignment projection only after the three enumerated components are handled; raw objects remain untouched. Same-assignment comparison still refuses each individual kernel/platform mutation.
- Platform/kernel raw consistency, architecture/glibc/OS-family drift, driver/CPU/package/wheel/content/numerics drift, ambiguous parsing, extra fields, missing fields and tampered provenance/hash links all refuse. Test each changed component separately and show all non-kernel invariants survive the projection.
- Actual capture-to-control-to-qualification-to-producer-to-ledger/scanner and prepare/fit admission paths enforce v6, exact same-assignment identity and diagnostic isolation. Do not substitute mocked admission or fabricated CUDA events.
- Publish a new immutable source/descriptor/reference/control/ledger/package seal, empty first-attempt ledger, fresh namespace/output/receipt paths, source archive and amended policy digest. No canary-01/02 units enter that ledger; old artifacts and accounting remain immutable. The new canary proof must bind the exact new source and manifest.

## Required compatibility, workload-parity and timing evidence

**Compatibility gate — fresh canary-03, separately reviewed before one operator dispatch.** On whichever of the two observed kernels it receives, require native v6 capture SUCCESS, comparison to the sealed reference, exact before/after raw identity and actual CUDA execution using the unchanged pinned L4/driver/cu126/Python/package stack. Require finite six-action fp32 B8 Q values, the source-native observation/action-mask path, CPU/CUDA Q parity with unchanged `rtol=2e-4`, `atol=2e-4`, and exact masked argmax action equality on every executed canary fixture. No CPU fallback, skipped CUDA proof, unrelated warmup or fabricated observation qualifies.

Require the actual CANARY_ONLY sequence: publish tiny 672, build and scan its immutable verified ledger, revisit and skip 672, publish new 673. Bind source, inputs, raw assignment environment and kernel provenance to both units. Verify required payloads, SHA sidecars, completion markers, refusal of these tiny records by full-unit/fit consumers, terminal PASS, transfer <=300 seconds, independently staging <=300 seconds, and verified release/backend-account disappearance. Preserve the prior base15/effective30-minute canary envelope and other guards; any proposed envelope change needs a separate pre-data ruling. A canary PASS proves only the exercised compatibility/resume path; it does not qualify the full workload or the unobserved kernel. No extra allocation to chase the other kernel is required or authorized here.

**Workload parity and timing gate — separately sealed long timing operation after canary acceptance.** Preserve the registered 12 disjoint timing fixtures (three per stream) plus timing-only calibration, all source-native workload seams, observer checks including applicable argmax/multinomial evidence, RNG/continuation checks, existing parity thresholds and feasibility bars. All required full units must complete; tiny canary units cannot substitute. Every full unit, including calibration, must carry or hash-link its assignment UUID, raw kernel release/version/platform and verified before/after environment receipts; the Mac scanner must recompute those links and refuse within-assignment drift. Re-run genuine CUDA parity preflight in each new assignment before producing full units, even if that kernel appeared previously.

Freeze timing forecasts/envelopes from available lawful measurements before dispatch, disclose setup/export and prior failed exposure, and retain existing timeouts, 2x-rule accounting, retry triggers and maximum two outcome-blind completion attempts. The prior 120-minute base/240-minute effective engineering allowance is not made an empirical forecast by this amendment. Fresh package review must confirm its adequacy. Kernel labels are recorded regardless of performance; never select units by kernel speed or restart a slow valid assignment to obtain another build.

Timing from different kernels is not silently exchangeable. Report per-kernel/per-unit measurements and use the slowest observed qualifying rate for the forward feasibility forecast, including the existing setup/export allowances; do not average a faster kernel into a slower stratum to pass a bar. Preserve registered unit counts and one-look points, and verified-unit skipping. A previously unseen kernel must pass its own assignment-local parity gate and receive prospective review before its units can enter the registered result set. This rule does not claim cross-kernel bitwise identity; qualification is workload-specific and limited to observed executions. Prior v5 refusals/partial canaries supply no full timing units.

**Fit gate — still separate.** C and selected W must run in the same verified physical assignment with one constant raw kernel tuple and the same frozen plan/recipe. Check exact raw boundaries around both arms and continuation. No C/W contrast may span kernels/assignments, and no kernel-dependent recipe, seed, endpoint, allocation, loss, optimizer, RNG or checkpoint substitution is admitted. Scientific fitting still requires the existing complete timing/feasibility proof and independent final operational acceptance. Mac strict/serving gates and the owner's release authority are unchanged.

## Next steps and decision precedence

1. Sol implements/tests only the bounded v6 scope and submits the new immutable packet with this amendment bound by SHA256.
2. Independent implementation and exact-package acceptance precede operator canary-03. This document itself authorizes no dispatch or automatic retry.
3. Review canary-03's actual terminal/CUDA/export-skip/cleanup evidence. Failure remains HOLD/INCOMPLETE; a scientific or integrity failure is never a mechanical retry trigger.
4. Only after accepted canary PASS may the separately reviewed long timing operation proceed; only its complete accepted proof may support subsequent separate C/W fit review.

No thresholds, N, exclusions, label access, statistical look, scientific endpoints or release rules are changed. No label/outcome payloads, D4096 outputs or `t34-round0-study-01/` content were opened. No commits, source edits, dispatch, Colab/provider calls, tmux or scientific execution occurred. Only this create-only ruling and its SHA256 sidecar were written under the authorized Astra root.

Created UTC: 2026-10-26T17:59:07.750241+00:00
