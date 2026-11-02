# T35 two-deviation rescue probe: primary result

**Registered verdict: EXTEND (W = 2/7 rescued worlds, U = 0)** (2026-10-20, Pacific). The rule is
PREREGISTRATION.md Amendment 1 §A1.3 (frozen in `321ba7c7` before the run): W ≥ 3/7 → REAL,
W ≤ 1/7 → KILL, W = 2/7 → EXTEND. U = 0, so f(W) = f(W + U) and the result is not INCONCLUSIVE.
The verdict is not relabelled.

**The EXTEND run is NOT launched.** Per approval condition C6 it is a separate launch under
§A1.3/§A1.8 (the 26 remaining all-die events, same shape, cap ≤ 2 h on the Mac). It needs its own
approval check first, including a mid-run lid/thermal check. Until then the pooled decision is open.

- Summary: `snake-dqn-artifacts/t35-two-deviation-20261020/primary/summary-primary.json`
  (sha256 `69acca9d11f76cb11337c91df60110424b3f8af9348a11ab2f52b1782c428a90`,
  schema `t35-two-deviation-summary/v1`). Receipts: `.../primary/events/*.json` (7). Plan:
  `.../primary/plan-primary.json`. Launch log: `.../t35-two-deviation-20261020/run.log`.
- Approval: `snake-dqn-artifacts/fable-panel-2dev/APPROVAL.md` (Fable approval judge, APPROVE WITH
  CONDITIONS, ~18:50 PT), after the design/engine/ops panel in the same folder.
- Compute: Mac, 3 single-thread processes, nice 10, from the detached frozen worktree
  `snake-dqn-t35-2dev-frozen` at `321ba7c7`, main venv interpreter. Wall 18:59:32–19:11:19 PT
  (~12 min, cap 3 h). 2,048,954 row-steps in total. **Colab: 0 units.** Every event exited rc 0.

## Per-world results (primary set, one event per distinct world)

Rescue = (t1, a1) first deviation → (t2, a2) second deviation, beam rank of the first deviation, then
the depth-600 confirmation (A1.5; an episode end before 600 frames counts as survival).

| Event | Status | Rescue found by the registered beam | Rescued ranks | Layer-2 candidates alive at t2+300 | Unconfirmed survivors | Cap expired | Row-steps | Search s |
|---|---|---|---|---:|---:|---|---:|---:|
| `b8_s0_21@3707` | NOT_RESCUED | — | — | 0 | 0 | no | 370,039 | 345 |
| `b8_s0_8@4318` | NOT_RESCUED | — | — | 0 | 0 | no | 326,941 | 312 |
| `b8_s3_19@4693` | RESCUED_DEPTH2 | (4813, 3) → (4965, 0), rank 1; confirm 35 frames to episode end, survives | 1, 2, 3, 4 | 4 | 0 | no | 141,293 | 159 |
| `b8_s3_26@3647` | NOT_RESCUED | — | — | 0 | 0 | no | 380,981 | 351 |
| `b8_s3_6@4562` | NOT_RESCUED | — | — | 0 | 0 | no | 156,174 | 155 |
| `b8_s3_8@4246` | NOT_RESCUED | — | — | 0 | 0 | no | 285,910 | 284 |
| `b8_s3_9@4188` | RESCUED_DEPTH2 | (4198, 0) → (4204, 3), rank 3; confirm 600 frames, survives | 3 | 1 | 0 | no | 387,616 | 370 |

Totals: rescued 2, not rescued 5, unknown 0; rescued at rank > 4: 0; row-steps per rescue 1,024,477.
**Secondary read-out (not in the verdict):** worlds with ≥ 1 layer-2 candidate alive at t2 + 300 =
**2** (the same two worlds).

## Known-path caveat (Amendment 1 §A1.7; approval condition C5)

Both rescues are in the **only two primary worlds where T3's own history already showed a
multi-deviation escape**, disclosed pre-data in §A1.7:

- `b8_s3_19@4693`: T3 changed (4697, 2), then (4727, 4), and survived to 5000. A two-deviation rescue
  meeting the 600 bar was therefore known to exist. The beam contained T3's first deviation
  (`beam_contains_known_first_deviation: true`), but the rescue it confirmed starts from a
  different first deviation, (4813, 3); all four beam ranks rescued.
- `b8_s3_9@4188`: T3 changed (4198, 0), and the M2 case from 4199 had a single-deviation 300-step
  survivor at 4204 (a* = 3), a known two-deviation *candidate* whose 600 bar was unknown. The beam
  found exactly that path, (4198, 0) → (4204, 3), and confirmed it to 600 frames.

In the five worlds without a known multi-deviation path, the registered beam found **no rescue and no
layer-2 candidate alive at t2 + 300**. The registered count is 2/7 = EXTEND, as the approval judge
agreed (counting only rescues found by the beam, outcome-blind). But read it with the asymmetry: on
this sample, the cheap beam reproduced known escapes. It has not yet found a two-deviation rescue
where none was already visible. The EXTEND set (26 events) is where that question is answered.

KILL, had it fired, would have meant "the registered beam heuristic found ≤ 1 two-deviation rescue,
so do not fund GPU search now". It would not have meant "one deviation is the ceiling" (§A1.3). The
same restraint applies to this EXTEND: it is not evidence for REAL.

## Provenance check (approval condition C2)

| Check | Result |
|---|---|
| Receipts present | 7 / 7 (`primary/events/*.json`) |
| `provenance.commit` on every receipt | `321ba7c7e7a44e866ffb64592f316c1fa9855b67` (7/7) |
| `provenance.dirty` on every receipt | `""` (7/7) |
| `plan-primary.json` commit / root | `321ba7c7…` / `/Users/josenunez/Projects/ml/snake-dqn-t35-2dev-frozen` |
| `plan-primary.json` interpreter / procs / nice / cap | main venv absolute python / 3 / 10 / 10,800 s |
| Summary commit | `321ba7c7…` (`summarize` refuses receipts bound to more than one commit) |
| Retries, replacement events, re-runs (C4) | none |
| Any `cap_expired` | none |

## Next step

EXTEND: one approval check (an Opus stand-in while Fable is at its usage limit; see CAMPAIGN_LOG,
2026-10-20 ~19:05), then launch the 26 EXTEND events under §A1.8 in a fresh `--out`. The launch must
fit around the Path B quiet window. The pooled 33-event decision (7 + 26) is then made with the
approval judge, as the PREREGISTRATION "Verdict" section states (pooled 33 decided with the owner/Fable).
