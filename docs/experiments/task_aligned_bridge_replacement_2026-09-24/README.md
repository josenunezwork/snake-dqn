# Original 6102 serving-bridge replacement: capture failure

**CLOSED_INVALID_STOP_CAPTURE_JSON_ROUNDTRIP_MISMATCH.** The explicitly authorized single replacement failed during its first capture stage. It did not qualify the bridge and supplied no new gameplay or learning result. Both the original preflight-only attempt and this replacement remain closed; there was no further retry.

The first attempt had stopped before any child because its controller repeated a malformed 63-character candidate digest. After the user approved one separately reviewed replacement, a new isolated checkout removed that duplicate and derived the guard from the canonical capture declaration. Two independent GPT-6 Astra source reviews passed, including the actual guard and the complete new plan. Those reviews did not establish runtime qualification, and the later serialization defect escaped them.

## What ran and what stopped

The replacement controller passed its repaired identity guard. Its capture child completed the declared four-file historical metadata read and namespace materialization, then copied and authenticated five fixed checkpoint files. It deserialized candidate metadata once while constructing the saved input freeze. The next check failed at `capture.py:580`:

```python
json.loads(input_bytes) == inputs
```

The source returns tuple-valued observation descriptors, including `(9, 31, 31)`, in the in-memory object. JSON stores that as an array and reloads it as `[9, 31, 31]`. Direct equality therefore fails. This is a sufficient structural explanation from the frozen source, conditional on that authenticated source being executed. The original runtime object was not retained, so the review does not enumerate every differing leaf or claim it was the only mismatch. See [the source review](failure-source-review.md).

The saved `freeze/inputs.json` and earlier prefix records remain preserved. The top-level `output/inputs.json`, successful capture report and completed-stage receipt were never published. Local PASS markers on the namespace and byte-copy substeps are preserved prefix evidence; they do not turn the failed capture stage or whole qualification into a PASS.

| Item | Preserved observed count |
|---|---:|
| Controllers / admitted children | 1 / 1 |
| Successfully completed stages | 0 |
| Historical metadata files / bytes | 4 / 6,339,153 |
| New namespace IDs materialized | 206 |
| Checkpoint snapshots copied | 5 |
| Checkpoint opens / read calls | 16 / 42 |
| Checkpoint bytes read | 86,308,029 |
| Candidate metadata deserializations | 1 |
| Model constructions / forwards | 0 / 0 |
| Tests / episodes / native frames / optimizer updates | 0 / 0 / 0 / 0 |

The one child exited naturally with code 1 after **1.026056833914481 guarded seconds**. The enclosing controller also returned code 1. Sampled peak child RSS was **316,325,888 bytes**, with minimum available RAM **32,320,765,952 bytes**. Root confirmed process exit, acquired and released both shared locks, and checked the unchanged clean source. No other controller was launched.

Reported operation totals are the instrumented child's preserved counters. Root's administrative receipt closure is not an independent scientific audit. The planned synthetic fixtures, 12 shakedown games, Watch, Play and saved-output audit never ran. Root did not reopen checkpoint payloads, replay games, re-read historical metadata or reconstruct the failed runtime object.

## Identity, scope and interpretation

- New source commit: `b22d59c537a1d0a30c8cc9b2c503aa8d8d7f86b5`, isolated from main and the first failed checkout.
- Exact plan SHA: `900d8ed407eeb93965e30d37273f8b638193cac4b5c67bcb67fc4020c41b83e3`.
- Independent implementation review SHA: `508c9d74ca4e1eb8f04e6041e948ef6d1b419a41b9a490352fef2424c71bf07e`.
- Intent SHA: `883c2c4351b368c9e537c8a0fa4ccdd212615669deb1277b418c14309b8d4d40`.
- User authorization SHA: `8ee4f8318ca4be724faf702a188dcff6958c1a26ebd41cae291fdb7c48105d03`.
- Original candidate SHA: `8d67915ef1ea5dbb0804ffd8bf9874dedb27c95878bacdd5b5e5ae77682aa46f`.
- New window: `2026-09-24T20:24:28.067831+00:00` through `2026-09-24T22:24:28.067831+00:00`, now **CLOSED**.
- Evidence root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/task-aligned-bridge-qualification-20260924/admission-v2`.

The original 6102 candidate remains unchanged and unpromoted. The separately completed continuation remains a negative mass-retention result; neither its child nor another checkpoint replaces the nominated candidate here. This infrastructure failure says nothing new about candidate-versus-Apex utility. There is no new learning curve for zero updates.

The authorization covered one replacement only. Further execution needs a new explicit user decision and separately reviewed finite scope. Any such design must preserve and account for this completed metadata/capture prefix, avoid repeating it, qualify the serialization boundary with synthetic data before any new live work, and run only previously unexecuted work. Neither failure is relabeled as success or erased.

See [administrative closeout](closeout.json) and [preserved failure attestation](failure-attestation.json). This documentation and source commit remain local; unrelated source history was not pushed.
