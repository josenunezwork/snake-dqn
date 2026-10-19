# S1 final-only export design and admission status (2026-10-18)

Status: **DRAFT_PENDING_OWNER_REVIEW; ADMISSION REFUSED; NO DISPATCH.**
This pre-data design replaces the S1 full-state export path described by
`BARRIER_EXPORT_STATUS_20261018.md` and the previous handoff. Those measurements,
receipts and original refusal records remain historical, unchanged evidence.
The prospective change is recorded in `PREREGISTRATION.md`'s change log.

## Export contract and measured size

S1 creates no mid-run checkpoint or full-state/replay snapshot. Only terminal
online `final.pth`, terminal `target.pth`, summary/counter/provenance/parity
receipts, `manifest.jsonl`, `log.jsonl` and `telemetry.jsonl` enter its immutable
archive. No optimizer, replay, actor state, RNG state or aborted/intermediate
weights enter the archive. These weights cannot provide exact resume.
The trainer's full-state barrier correction is retained for other consumers.
An opt-in `--final-target-weights` flag saves the actual target network after the
learner flush; its default is off. The S1 weights checkpoint interval is 9M,
beyond 8M plus the largest registered 2,303-decision overshoot.

`FINAL_ARTIFACT_SIZE_FIXTURE.json` binds the real local CPU entrypoint fixture:

| Item | Measured bytes |
| --- | ---: |
| Online weights | 6,566,741 |
| Target weights | 6,562,677 |
| Summary + provenance + manifest | 30,370 |
| Compressed archive + transport index | **12,180,957** |

This is a 12,000-decision, two-actor, eight-world fixture with reduced replay,
not an 8M L4 measurement. It finishes before periodic telemetry/log windows;
production final validation requires those files and all parity/CUDA receipts.
CPU compile falls back to eager; no CUDA qualification is claimed. The adapter
completed in 120.3 seconds (including its polling interval), with zero overshoot,
clean actor exits, zero ring overruns, reconciled n-step/update counters and no
full-state events. Reassembled archive members exactly matched the source bytes.
See `TRAIN_PREFLIGHT_FINAL_ONLY.json` for the source and evidence hashes.
Earlier development fixtures 01/02 are retained and are not current admission
proof; fixture 03 is the final adapter source measurement.

The enforced archive-plus-index cap is **64 MiB per attempt**, **640 MiB** across
ten primary runs, or **768 MiB** across twelve attempts including the two allowed
reruns. Oversize output refuses export; it is never silently truncated. This
replaces about **867,602,770,220 bytes** of planned full-replay snapshots across
ten runs in `CHECKPOINT_SIZE_MEASUREMENT.json`, before their final archives.

## Operational reruns

At most one replacement run per recipe (N3 and N1) is allowed across its five
seed blocks. `rerun.py` prepares a create-only manifest and review record locally;
it performs no allocation. An explicit `run.py --continue --retry-record ...`
uses the same immutable admission plan and ledger. It reserves the arm's single
`retry-n3` or `retry-n1` liability atomically, preserves seed/recipe/input/source
identity, starts from B8 with fresh optimizer and replay, recovers prior settled
finals by verified bytes, and continues unstarted primary jobs. No automatic
retry or resume exists. Both dispatch paths use the same Mac evaluation watchdog
and measured two-worker accounting.

Before preparation and launch, the guard rejects efficacy artifacts, a complete
verified final, scientific tripwires and ambiguous/integrity failures. A reviewed
infrastructure loss requires a stopped/disappeared runtime and complete usage
and elapsed evidence. Human review delay is elapsed study time rather than paid
GPU time. Unknown spend/release intervals block further allocations. A second
failure cannot acquire another liability and remains INCOMPLETE.

## Conditional forecast versus unchanged caps

`ADMISSION_STATUS_FINAL_ONLY.json` was regenerated locally from the current CPU
adapter receipt and the unchanged disjoint ten-world Mac timing receipt.
The latter reserves **15.8656728896 Mac worker-hours** for 1,920 policy episodes,
startup, audit and preflight, versus **19.2 usable**.

| Quantity | GPU allocation hours | Units at 2/hour |
| --- | ---: | ---: |
| Ten primary training runs, 30/68-minute bases | 8.1666667 | 16.3333333 |
| Two possible from-scratch reruns | 1.6333333 | 3.2666667 |
| CLI acceptance allowance | 2.0 | 4.0 |
| Twelve five-minute cleanup tails | 1.0 | 2.0 |
| Known conditional planning subtotal | **12.8** | **25.6** |
| All twelve hard timeouts + cleanup + acceptance | **22.6** | **45.2** |
| Usable caps (20% withheld) | **20.0** | **48.0** |

Setup, final-artifact preparation and final transfer remain conditional; no
complete planning total is emitted until their transport evidence exists.
Transport liability is the per-arm final-bundle size bound divided by the
measured throughput lower bound. The field remains
`s1_final_artifact_transfer.bytes_per_second_lower_bound` in the hash-bound CLI
acceptance receipt. `final_artifacts.py` requires current adapter/trainer and
transport source identities, verified L4-to-Mac bytes, and bound production-size
receipts for both arms. A CPU fixture or assumed home-connection speed cannot
fill that field. A disjoint acceptance fixture may validate the fixed weight
shape and enforced telemetry/artifact bounds without producing S1 efficacy
data. Full-replay throughput qualification is no longer required.

Hard job estimates/timeouts remain N3=30/60 minutes and N1=68/136 minutes. The
hard envelope fits units and elapsed time (30.5328364448 hours versus 38.4), but
**exceeds usable GPU allocation by 2.6 hours**. Admission continues to refuse even
if transport is instantaneous. No cap, reserve, timeout or estimate was changed
to conceal that conflict.

## Remaining admission prerequisites

1. Complete live acceptance of the independently owned CLI upload-deadline
   repair, including measured final-only transfer rate and N3/N1 final-size
   receipts bound to the final published sources. The repair was integrated
   from upstream commit `dd74d5fe`; no live acceptance was run here.
2. Resolve the 22.6 versus 20 GPU-hour hard reservation conflict through a
   prospective reviewed execution design that fits the existing caps.
3. Claude's prospective registration approval, fresh required runtime/parity
   and initializer/collector evidence, immutable published input/source pins,
   a clean published admission checkout, and a fresh account snapshot with all
   assignments and rates reconciled. DRAFT review manifests cannot authorize work.
4. Generate a fresh create-only PASS pricing receipt and admission plan only
   after all prerequisites pass. No failed evidence may be overwritten.

Offline forecast command (choose a fresh output path):

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.s1_mature_20261018.price \
  --eval-preflight /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/EVAL_PREFLIGHT_FINAL.json \
  --training-preflight /Users/josenunez/Projects/ml/snake-dqn-s1-mature/research/s1_mature_20261018/TRAIN_PREFLIGHT_FINAL_ONLY.json \
  --out /absolute/fresh/path/pricing.json
```

This command is expected to refuse until transport and hard-budget admission
are satisfied. It never calls Colab. No live runtime, tmux session, serving pin,
released checkpoint, Q1 evidence or protected transport source was changed.


## Published review manifests and verification

Ten `jobs_review/*.json` manifests are repinned to published implementation
commit `f9bb1055d3e7fdbc74689b94a49e8547e4f34c15`, including the independently
published CLI deadline fix. All ten pass local `supervisor.validate` with zero
allocations, DRAFT owner/registration tokens, L4 only, at least 12 vCPUs, a
180-second lease and 64 MiB export cap. Their input SHA inventories, fresh
output paths and unchanged 30/60 and 68/136 minute estimate/timeouts are in
`MANIFEST_VALIDATION_FINAL_ONLY.json`. They remain review manifests.

Verification commands and final counts are recorded in
`FINAL_EXPORT_VERIFICATION.json`; logs are under the external
`snake-dqn-artifacts/s1-mature-20261018/final-only-verification/` directory.
The independent review found no remaining blocking implementation findings.
Admission refusal is an intentional budget/evidence result, not a test failure.

The final post-merge supervisor/S1/trainer selection passed **228 tests**, with
2 slow tests deselected. Black/isort/Flake8, core `make lint`, shell syntax and
`git diff --check` passed. The wider repository run finished **7,900 passed,
21 failed, 26 skipped**, including **two native worker segmentation faults**.
Its failures include compiler/backend tests, old actor-oracle telemetry
assertions, missing built frontend output and a serving fallback assertion.
No S1 test failed in that run. The wider repository gate is not green; the
current affected sources are covered by the subsequent post-merge selection.
These broader failures were retained and not repaired outside this task's scope.

Upstream native-runner fix `cf5df857` was integrated after that full-suite run.
The full repository gate was not rerun after this independently authored fix;
its inclusion does not turn the recorded failure into a pass.

After integrating `cf5df857`, the final S1/trainer/native-runtime selection
passed **113 tests**, with 2 slow tests deselected, in 23.04 seconds.
