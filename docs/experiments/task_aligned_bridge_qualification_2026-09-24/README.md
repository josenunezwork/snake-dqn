# Original 6102 bridge qualification: preflight failure

**Closed: `CLOSED_INVALID_STOP_CONTROLLER_PREFLIGHT_IDENTITY_MISMATCH`. No numerical qualification or scientific audit ran. No promotion.**

The first controller invocation exited with code 1 on 2026-09-24 at approximately 18:27 UTC. The correctly bound intent named the original lineage6102 checkpoint. A manually copied acceptance literal in the frozen runner omitted one character: it contained 63 hexadecimal characters instead of 64. The controller therefore rejected the correct intent at `runner.py:129–133`.

This error escaped the independent source and exact-plan reviews. Root owns integration responsibility. Authenticating a source hash establishes the reviewed code's identity; it does not prove that the code's guard agrees with the plan.

The exception occurred in `validate_intent`, before the run-root boundary check, source authentication, directory creation, supervisor loading and child dispatch. The `run` directory does not exist. No stage was admitted, so no child receipt or normal runner failure JSON could be produced. The [external launch traceback](controller-launch-traceback.txt) and [administrative attestation](failure-attestation.json) preserve that distinction.

| Work | Recorded outcome |
| --- | --- |
| Controller launch | One attempt, exit 1 |
| Qualification stages admitted | 0 |
| Checkpoint payload reads/loads, models, forwards | 0 by pinned control-flow inference |
| Games/native frames/optimizer updates | 0 by pinned control-flow inference |
| Independent scientific audit | Not run |
| Guarded child runtime | 0; controller runtime was not separately measured |

The zero-operation conclusion comes from the exact frozen source and traceback, not direct I/O instrumentation: instrumentation had not started. The earlier validation prefix has no separate durable success seal.

## Closed authority and identity

The frozen policy had zero recovery. The failed controller, source commit, intent, reviews and window remain immutable. There was no repair run, re-admission, deadline extension, resampling or repeated qualification. The original 6102 challenger is unchanged, and this failure says nothing about its policy quality.

- Isolated source commit: `aa3d0dd0e08ba7905707e3ea59acf85a38f909e0`.
- Intent SHA-256: `82f0e99a9fda82effc36d04cfe8a23ff254880694904ed50a20f7248685c84b0`.
- Plan SHA-256: `4cd3de85456ab8b762d3fb5a339c63915ff048095482c01da218aca532ee9c35`.
- Exact-plan review SHA-256: `4a596d532e3d1acad0ca04c1244190354e635bf97cd4ca1ca2e43c3065e735aa`.
- Failure attestation: `e9cfa4dbf1c750a96e526cd902cb71f74e37e1d4a996d8e4080172a3d27e4de5`.
- Closeout: `e1f0bbcdc3aa7eef140fbea38d86a9d0417fdbbeda4b0443cc66b18206b87706`.

The correct bound candidate identity is `8d67915ef1ea5dbb0804ffd8bf9874dedb27c95878bacdd5b5e5ae77682aa46f`. The [versioned identity corrigendum](identity-corrigendum.json) records the same malformed transcription in two administrative narrative fields, preserving their original files and seals. The admitted intent and actual source-checkpoint metadata used the correct identity. No checkpoint was reopened to produce that correction.

## Decision needed before any replacement

A separate Astra [policy recommendation](../../research/task_aligned_bridge_preflight_replacement_decision_2026-09-24.md) proposes one explicit exception for this supported preflight-only failure. It would require user authorization, a canonical validated identity guard, a new immutable source/plan/review/admission/deadline, unchanged original candidate and scientific settings, and no automatic second retry. It is a proposal only; no replacement source was implemented and no new admission is authorized.

The preceding [fresh-game continuation](../task_aligned_continuation_2026-09-24/README.md) remains a completed, audited negative screen. Its higher food and survival did not satisfy mass-retention criteria. That result neither repairs this serving failure nor substitutes the child for the original challenger.
