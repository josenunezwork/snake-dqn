# Round-27 speed harness repair

The failed `bl-f0-speed-equiv-r27` dispatch used
`bf8ac97dc66d1b4515e4c5510db3ad5886fa695d`. NEW's `GateDriver.step` forwards
`forced, packet` positionally through `super()`. The external instrumentation
wrapper accepted only one positional argument after `self`, so NEW failed before
the equivalence comparison. OLD workers ran; their partial evidence is retained.

`speed_worker.py` now forwards the original positional and keyword arguments
unchanged while retaining its forced/natural context and counters. It also retains
an inherited nice value above 10 rather than trying to raise process priority
(macOS denied that operation in this agent's subprocess environment).

The scientific factory, comparator, normalization, bank construction, T-EQ rules
and thresholds are unchanged. OLD remains
`61900af28932e47e22e332840fc4406559adb062`.

## Verification

Using `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`, CPU and
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1`:

- `-m pytest -q research/bl_labels/test_speed_run.py research/bl_labels/test_speed_equiv.py`:
  12 passed in 10.30 seconds, including actual isolated OLD/NEW worker processes.
- `-m pytest -q research/bl_labels/test_prepared_packet.py`: 5 passed in 0.54 seconds.
- `-m black --check --line-length 100 research/bl_labels/speed_worker.py research/bl_labels/test_speed_run.py`:
  passed.
- `-m isort --profile black --check-only research/bl_labels/speed_worker.py research/bl_labels/test_speed_run.py`:
  passed.
- `-m flake8 research/bl_labels/speed_worker.py research/bl_labels/test_speed_run.py`:
  passed.
- `bash -n research/bl_labels/RUN_SPEED_EQUIV.sh research/bl_labels/RUN_SPEED_EQUIV_REPAIR.sh`
  and `git diff --check`: passed.

The Mac integration regression archives the OLD scientific closure from Git,
uses the current NEW closure in a separate process, and collects one two-step
scripted-gate parent on each side using the actual pinned checkpoints. It verifies
two living decisions and both audit-tape entries, verifies immutable science
exports, and requires PASS from the unchanged canonical comparison. The synthetic
regression also covers positional `super().step(forced, packet)` and direct keyword
forced calls. This tiny functional proof does not constitute registered T-EQ,
timing, W6, F1 or release evidence.

Independent Codex-V light review (`/root/codex_v_light`): PASS, no remaining
actionable findings. The reviewer independently checked transparent argument
forwarding, regression coverage, unchanged comparison/rules, fresh paths and shell
syntax. Its publication preflight finding was corrected before commit.

## Operator continuation

Use the full published repair pin supplied in the final handoff. Offline preparation
creates `bl-labels-f0-speed/equiv-pack-02` at that pin. `equiv-pack-01` and `equiv-01`
remain untouched. The repaired launcher targets only `equiv-02` and refuses an
existing collection output. The failed partial run contains 151 files totaling
43,528,679 bytes; it is not resumed or analyzed by these commands.

The new runbook accepts exactly one phase per invocation:

```bash
RUNBOOK=/Users/josenunez/Projects/ml/snake-dqn-speedfix/research/bl_labels/RUN_SPEED_EQUIV_REPAIR.sh
# PIN must be the full published repair pin from the final handoff.
/bin/bash "$RUNBOOK" "$PIN" speed
# Wait for COLLECTED_NOT_COMPARED and lease release before publication.
/bin/bash "$RUNBOOK" "$PIN" publish-speed
/bin/bash "$RUNBOOK" "$PIN" analyze-speed
# Wait for the analysis dispatch to finish before recording.
/bin/bash "$RUNBOOK" "$PIN" record-speed
```

Collection dispatch: `bl-f0-speed-equiv-r27-02`.
Analysis dispatch: `bl-f0-speed-analyze-r27-02`.
Publication: `research/bl_labels/ledger/ROUND27_SPEED_EQUIV_PHASES_02.json`.
Recorded receipts: `research/bl_labels/ledger/ROUND27_SPEED_RECEIPTS_20261024_02.json`.
The publication and recording phases commit/push their create-only records and
campaign-log entries; preflight preserves existing staged work or campaign-log edits.
The collection/analysis phases use the existing Mac dispatch and lease controls.
No phase is chained automatically.

No equivalence, timing, F1, calibration, training, dispatch or serving job is run
by this repair. A later D1024 operator must explicitly select the newly sealed
`equiv-02/receipt.json` (the existing D1024 launcher's `EQ` override) after the
registered analysis; this repair grants no admission or release authority.
