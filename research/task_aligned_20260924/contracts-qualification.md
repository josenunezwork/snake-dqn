# Finite qualification contract

Status: implementation prepared; no fixture, model load, game, or timing probe has been executed. Static compilation is not runtime qualification. `stage_cli.py` calls `qualify.run_qualification(intent, out, heartbeat)` only after the new create-only admission. A failed check, cap, subprocess, or fit ends this package; retries are not part of this contract.

## Inputs and outputs

The input is the same frozen intent used by the controller: all three explicit `parents` bindings keyed by full CZ lineage IDs, `qualification_seed`, `device`, embedded `evaluation` template, and the qualification stage's physical caps. `out` is the controller-created empty output directory. The heartbeat object publishes progress while the root-owned thread maintains the physical-resource heartbeat.

The function returns the common report fields `counters` and `artifacts`, plus `runtime_contracts_pass`, `evaluation_fits`, `fixed_science_fits`, the fixed dose and its canonical hash, and timing forecasts. It creates `qualification-evidence.json` once. Model state, synthetic receipt trees and synthetic reporter fixtures are disposable, held in temporary directories and removed after checking. The probe's scalar timing and accounting telemetry is saved as qualification evidence; none of its learner state or worlds enters scientific training.

## Exact finite work

| Work | Bound |
| --- | --- |
| Parent deserializations | Exactly one for each of 2026096101, 2026096102, 2026096103 |
| New checkpoint roundtrip | One additional deserialization; total at most four |
| Synthetic wrapper parity | Six observations per parent, two network invocations per parent |
| Physical lifecycle fixture | One E16/S6 fixture batch, 16 initial worlds, exactly 17 active lane frames, no optimizer step |
| Disposable learner timing | Source 6101 only; exactly 30 complete `[solo,s2,s2,s6,s6]` cycles, 150 rollouts, E16/T16 |
| Probe hero sample draws | At most 38,400; equal to actual distinct valid hero rows |
| Probe optimizer steps | Exactly 150, one exact-coverage epoch and at most 256 valid rows per rollout |
| Learned serving timing | E32 × 128 frames × three fixed profiles: at most 12,288 active lane frames |
| Scripted anchor timing | E32 × 32 frames × three profiles × two anchors: at most 6,144 active lane frames |
| Batched/individual serving parity | E32 × 2 plus 32 E1 × 2: at most 128 active lane frames |
| All qualification lane frames | At most 56,977, beneath the frozen 60,000 cap |
| Qualification elapsed wall time | At most 1,800 seconds, enforced by the external supervisor |

The frozen shared cap additionally bounds hero transitions to 60,000, network invocations to 6,000, forwarded rows to 150,000, initial/reset episode roots to 4,096, prepared-preview calls to 256 and SGD draws to 38,400. Counts are actual operations, and the evidence separates the learner probe, physical lifecycle fixture, and evaluation probe. A physical fixture's 17 hero transitions are not SGD sample draws. World roots count new procedural initialization and resets; a disposable preview clone is counted as a preview, not a new procedural world. Artificial geometry edits belong only to the explicitly labelled fixture, never the science stream.

The runtime initialization is 48 worlds: 16 each for solo, S2 and S6. At most 16 worlds reset per rollout, giving a conservative runtime root bound of 2,448. The three restoration checks do not initialize environments. The separate physical lifecycle fixture initializes 16 more. All serving roots and individual parity roots are also counted. The actual probe need not reach these ceilings.

## Restoration and numerical contracts

Each parent's frozen report and checkpoint hashes are authenticated by the new adapter, which checks the exact CZ enemy-visible mark-4608 metadata, source parameter layout, restricted `raster31v3` inputs, coefficients, half-MSE optimizer contract and complete Adam state. Qualification compares restored network and Adam value digests to the parent, checks learner clocks and checks that numeric-recovery snapshots contain the restored parent. No old experiment module or executable driver is imported.

Six deterministic synthetic observations per parent compare the adapter to an independent `RasterDuelingNetwork` with copied weights, explicitly restricted channels/scalars and the inherited 0.1 output scale. Both Q values and externally masked actions must agree, and original observations must remain unchanged. A direct two-element loss-and-gradient fixture distinguishes half-MSE from Huber without performing an optimizer step.

One new checkpoint save/load verifies model, optimizer and clock roundtrip. After the 150 timing updates, the same model and optimizer objects must have served every task, the shared update/Adam clocks must have advanced from 4608 to 4758, and per-rollout sampling evidence must establish exactly one draw of each valid hero row. The timing endpoint is discarded.

## Physical lifecycle fixture

The fixture constructs an actual six-snake simulator with all six bodies and six masks, while the learner tensors contain one hero per world. One lane deterministically dies on its first move; one lane survives sixteen moves to the live cap; fourteen lanes are already capped and remain frozen. A scripted opponent's respawn timer expires during Watch preparation. The fixture checks terminal hero death, opponent respawn, actual valid-row/frame counts, inactive bodies/clocks/RNG preservation, no reset inside the rollout, and an immutable disposable preview.

The live cap's final observation and mask must equal a separately prepared disposable successor. Native target computation must use reward alone for death and reward plus the masked successor value for live truncation. These are tested through the new collector's actual returned rollout. No optimizer update occurs in this fixture.

## Hardware-only dose and serving fit

Accelerator timing synchronizes MPS before and after each complete five-rollout cycle. Let `C` be the slowest of the 30 complete cycles. Let `F` be the slowest measured parent setup plus fresh runtime initialization plus three measured checkpoint-save durations. The per-lineage forecast for dose `D` is:

`2 × (F + (D / 5) × C)`.

Select the largest multiple of five at most 39,060 whose forecast fits 25,200 seconds. A selection below 7,815 is resource infeasibility and stops the package. All three lineages use the same 6101 hardware projection; there is no claim of measuring three learners' speeds. Three times the per-lineage forecast is `estimated_science_seconds`. No loss, return, survival, reward, encounter or parameter value is used to choose dose.

The canonical dose hash binds only `selected_rollouts`, `conservative_cycle_seconds`, `fixed_overhead_seconds`, `shared_probe_lineage: 2026096101` and profile rollout counts (`solo: D/5`, `S2: 2D/5`, `S6: 2D/5`). The controller and independent auditor recompute this binding.

The CPU serving probe reconstructs pristine weights from the source's saved native tensor state, because the disposable learner shares and mutates the original model object. It performs no additional checkpoint deserialization. Batched/individual parity and the real batched timing must pass before scientific training starts. Twice the three-profile learned timing extrapolated to H3000 must fit one 900-second model-evaluation stage; twice all six anchor profile timings extrapolated to H3000 must fit the 900-second anchor stage. Three saved synthetic E32/H3000 frame-column artifacts measure representative serialization time. The learned forecast is twice gameplay plus three-cell serialization, with another 60 seconds reserved; the anchor forecast doubles six-cell gameplay plus twice the three-cell serialization cost, also reserving 60 seconds. These synthetic artifacts are explicitly labelled and hashed. Any serving fit failure stops before the first science update.

## Saved-evidence schema fixtures

Reporter fixtures build 24 purely algebraic cells, each with 32 labelled synthetic rows and 17 per-frame records. These are not games or procedural world constructions. They exercise all five decision branches, create-only output, missing/duplicate cells, mismatched initial pairing, optimizer activity in evaluation, nonfinite facts and reordered worlds.

A complete temporary receipt tree contains qualification, three training reports and their independently derived world-ID ledgers, all 24 evaluation cells, saved analysis, and the final audit reservation. The production runner and production auditor check this tree. Negative variants reject a forced exit, changed report bytes, a counter violation and a nonmaximal dose. Fixture mutations never touch real run evidence. These tests execute only as part of admitted qualification.

## Remaining evidence boundary

A successful qualification demonstrates the bounded implementation and hardware fit; it does not demonstrate scientific gain, all-lineage throughput at long horizons, promotion, or general reliability. The external wall and memory guards still stop later work if actual costs exceed forecasts. Fixed science endpoints and the saved-output comparison remain required. Before admission, independent source review must verify this contract, the adapter's serving helper, the stage CLI and the controller/auditor schemas agree.
