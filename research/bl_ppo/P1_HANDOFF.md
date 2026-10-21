# WP-3 P1 operator handoff — 2026-10-21

**Prepared and independently reviewed; not dispatched.** Sol-R made zero Colab
calls. Claude dispatches. B8 s3 remains served; there is no release authority.

Executable pin: `4bb1e32a7b6c960a2e23c3bab19715dee3d21bbc` (published on main).
Registration commit: `464b21f8`; forecast/input helpers: `ad4ee761`; paired
evaluation/bank pin: `d54e4a04`; durable runtime/admission: `7a986c9d`.
Transport sidecar rename correction: `4bb1e32a`. It supersedes preparation
against `f18cf1df`; the original registration bytes remain unchanged.
Later documentation commits do not change this executable pin.

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-p1/`.
The root `DISPATCH.txt` indexes exact phase instructions and all preparation
receipts. Every new receipt, manifest, bank and instruction has `.sha256` and
`.complete`; they are create-only. A preparation receipt is never a live PASS.

## Frozen contract and initial invoice

F seed 20261021, target exactly 100,000,000 actor hero transitions; E8 critic
gate before actor updates. Qualification pauses at 1,000,064 (requested 1M,
overshoot 64), and its prefix is resumed. Natural resets, 128 worlds, 12
processes, 128-step rollouts; no checkpoint selection or silent recipe changes.

| Job | Estimate min | Timeout min | Unit cap | Reserved units |
| --- | ---: | ---: | ---: | ---: |
| Y3 source | 65 | 130 | 6 | 1.2 |
| Y3 replacement | 20 | 40 | 2 | .4 |
| Complete-job 1M qualification | 40 | 80 | 4 | .8 |
| 1M prefix →20M | 254 | 508 | 24 | 4.8 |
| 20M→40M | 267 | 534 | 24 | 4.8 |
| 40M→60M | 267 | 534 | 24 | 4.8 |
| 60M→80M | 267 | 534 | 24 | 4.8 |
| 80M→100M | 267 | 534 | 24 | 4.8 |
| 128/mix paired H5000 evaluation | 90 | 180 | 10 | 2 |

One P1 L4 at a time, account-wide maximum three assignments. All timeouts are
at least twice their estimates and below 12h. The tranche collection-only
forecast is 19.29h at ~1,440 active hero transitions/s. Initial remaining-job
forecast is 22.033h with 10% provisional overhead +12min/job; 2x liability at
2u/h plus 24u reserve =112.133u, within 120u. These are sizing forecasts, not
measurements of full PPO execution. Qualification must replace them and pass
both <=46h complete-trajectory time and the 120u invoice. With reserve inside
the envelope, the budget is stricter than the 46h ceiling. All phases sum to
142u maximum, with no automatic retry or purchase.

## Dispatch sequence and evidence

1. Use `y3/source-02/DISPATCH.txt` after Codex-V review, fresh account usage and
   inventory, preceding terminal receipt and verified release. Regenerate a
   **new** attempt with `p1_manifests --inventory` if inventory changes. The
   source reconstructs public checkpoints and qualifies E8 before creating a
   pre-actor origin and a source continuation through >=3 updates, an actual
   episode boundary and >=1,000 actions.
2. Release source and verify disappearance. Publish its listed immutable
   snapshot payloads through an operator-approved HTTPS route, with exact
   sizes/hashes. Deduplicate identical immutable siblings by name/hash. The
   small runtime-download catalog must contain the origin snapshot, every
   referenced immutable bank, source continuation snapshot, action trace and
   source receipt (give the receipt a unique flat transport name). Never upload
   checkpoint directories through Contents. Prepare a fresh replacement attempt
   with `--downloads`, `--resume-name` and `--expected-trace-name`. The last name
   identifies the **source receipt**, which binds its trace and final state.
3. Release replacement. Its Y3 PASS must bind different source/replacement
   endpoints, the full continuation tape and complete state. Prepare the 1M
   job using `--evidence <Y3 receipt>` and the **original** qualified origin,
   not the source continuation's endpoint.
4. After 1M terminal export and verified release, run
   `python -m research.bl_ppo.p1_qualification --runtime <qualification receipt>
   --supervisor <1M supervisor receipt> --y3 <Y3 receipt>
   --source-supervisor <source supervisor receipt>
   --replacement-supervisor <replacement supervisor receipt> --output <fresh receipt>`.
   This binds actual terminal exports/assignment identities to runtime timing,
   measures transfer/release overhead and seals PASS or NOT_READY. The unchanged
   supervisor's own receipt is raw create-only JSON; its bytes are hash-bound
   by the new P1 receipt without editing it.
5. Only a PASS complete-job receipt can materialize the five tranche jobs.
   Use `--phase tranche --chunk 1..5 --evidence <qualification receipt>` plus
   each verified runtime-download catalog and exact `--resume-name`. Chunks
   2–5 also require `--previous-receipt <preceding tranche receipt>`; the runtime
   checks its digest and exact actor boundary before creating workers. Each job
   must release before its successor allocates. Caps expire as INCOMPLETE.
6. Evaluation requires exact 100M endpoint and E13 parity receipts before any
   bank gameplay. Use `--phase eval --bank <bank> --audit <audit>
   --endpoint-receipt <training-status.json> --parity <E13 receipt>
   --downloads <catalog> --resume-name <endpoint payload name>`. The candidate
   download hash must equal the endpoint receipt. No partial analysis or
   intermediate checkpoint can enter. The frozen bank's seeds are excluded
   from training/calibration; E8 monitors use separate excluded worlds.

The executable manifest preparer is
`python -m research.bl_ppo.p1_manifests --phase <phase> --root <fresh directory>
--commit 4bb1e32a7b6c960a2e23c3bab19715dee3d21bbc --bank <bank.json>
--audit <seed_audit.json> [phase-specific evidence above]`.
Run from `/Users/josenunez/Projects/ml/snake-dqn-wp3` with the primary venv
`/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`.
Each generated `DISPATCH.txt` supplies absolute readiness/validate/run commands.
Refresh inventory in a new preparation, never hand-edit a frozen manifest.

## Frozen evaluation and transport

Namespace `bl-ppo-p1-eval-20261021/v1/round-0`; 128 scripted +128 frozen +128
mixed, paired vs bare B8 s3, H5000, terminal hero, one look. Bank payload SHA256
`8223a16b70165b23f86c5cc40f7195149b1b009bc4f79e6f0d049798489b11ba`;
pre-data audit SHA256
`81df75c16b2fe15168962425381c7510265caceea903e004ad8e92bbeb4bcedf`.
The audit is CLEAR against accessible history and seven FRP inventories; sealed
study data were pruned before descent. Coverage limits remain in the receipt.

Nomination: survival band v2 AND delivered boost [.005,.10] AND scripted
no-death point >=0 AND equal-mix mass point >=40 AND one-sided LB90>0.
Rank n/a: one artifact. Valid UB90<40 => ARTIFACT_REJECTED_AT_BAR; missing
evidence/caps/parity => INVALID/INCOMPLETE. Nomination only permits registration
of fresh Mac strict work in week 2; strict/product/serving and owner release
are separate.

Z-9 stages two small local files: the public-checkpoint catalog and one bounded
compressed input bundle containing control/admission sidecars. Checkpoints are
downloaded and hash-verified on the runtime. Preflight calls the real pinned
supervisor source preparer at actual bytes, prices the fallback archive, and
requires <=150s staging forecast (plus source-fetch bound within 300s). Every
manifest has its own bound upload receipt and offline supervisor validation.
This does not claim a live Colab upload or timing measurement.

## Verification and remaining gates

Root final P1/runtime/evaluation/admission/recovery/bank suite: **64 passed**
in 17.08s, including the transport rename regression. Codex-V independently
accepted that correction with **24 passed**.
Separate unchanged transport/snapshot/export selection: **39 passed**, with
overlap in recovery coverage. Codex-V independently ran 53 focused tests and a
real frozen-opponent snapshot fingerprint reproduction. Changed Python passed
Black/isort/flake8; shell syntax and `git diff --check` passed.

Real tiny shared CPU entrypoint: 2 worlds, 1 process, 16 critic +16 actor
transitions, 8 updates, 13.585s; endpoint/snapshot/status sidecars verified.
It is explicitly `scientific=false`, critic BYPASSED_PREFLIGHT_ONLY, and is
neither a Y3 nor a critic/qualification PASS.

No live Y3, 1M complete-job rate, 100M endpoint or efficacy result exists here.
Attempt-01 preparation artifacts are retained and superseded because their
renamed input sidecars did not verify. Use only attempt-02: Y3 `source-02` and
`replacement-02`; qualification `prepare-02`; tranche `prepare-02/chunk-01..05`;
evaluation `prepare-02`. No attempt-01 job ran.
Registration/Y3 due Thu Oct22 14:00 PT; 1M Thu PM; tranche Thu night when slot A
frees. No deadline has elapsed at preparation. A missed gate needs PLAN's
create-only slip record and blocks only its descendants. No tmux/process kill
commands, protected-package edits, sealed-study access or serving changes.
