# F5 V5 output-layout correction — final verification

Outcome: ENGINEERING ONLY; V5 preliminary qualification attempt 03 prepared and
published; no dispatch, Colab allocation, genuine CUDA qualification, numerical
look, fit acceptance, serving change or release performed.

## Change and sweep

`f5_step3_driver.run` accepts an absent output or the supervisor-created empty
real directory. Files, symlinks (including dangling links), hidden entries and
child directories refuse before qualification. Immutable producer exports retain
create-only publication. Seven new parameterized regression cases exercise the
non-dry-run path with only CUDA qualification stubbed, verify synthetic exports,
and prove rerun refusal without changing prior export bytes.

The read-only sweep confirmed `runtime._supervise` creates `root/output` before
Popen, and `f5_step3.sh` passes it directly. No other F5 runtime has this bug:
`f5_qualification.run` reserves fresh producer children; `f5_colab.sh` passes
`output/prepare` and `output/paired` to `f5_job`, whose absent-child checks remain
correct. Local package builders retain absent-target protection.

Attempt-02 evidence is preserved in
`research/bl_residual/ledger/F5_STEP1_ATTEMPT02_20261026/`: 38 exact metadata
copies with create-only SHA/completion sidecars. Its original receipt SHA256 is
`b4861b1a203538b929b96dc5c10fbaa61d70b3e92cf2917064e9baff0a5fa144`.
INCOMPLETE / CHILD_NONZERO_EXIT, transfer 93.9016244992381 s, staging
183.79829629103187 s, elapsed 625.7038177910727 s, stop_verified=true, no exports.
Saved usage changed from 1058.66 to 1058.42 units, with final assignments/rate 0.
No fresh provider query was made. Preserve both failed attempts in accounting.

## Verification

All commands execute locally at nice 15 with host numerical thread limits 1.
The full exact test command and all results are stored in the create-only
`F5_ENGINEERING_SEAL_20261026_V5/TEST_VERIFICATION.json`.

- F5, ops, supervisor, runtime, canary and observer suite: **1016 passed,
  3 skipped in 960.42 s**, exit 0. The three skips require a real CUDA runtime.
- Independent reviewer: driver/package/engineering seal tests, **40 passed in
  61.84 s**; no actionable source or actual-artifact findings. This is engineering
  review, not independent Fable scientific authority.
- Black `--check --line-length 100`, isort `--profile black --check-only`, flake8
  on the three changed Python modules and driver tests: PASS. `git diff --check`
  and `bash -n research/bl_residual/f5_step3.sh`: PASS.
- Actual `python -m research.bl_residual.f5_step3_driver --dry-run`, using the
  V5 manifest env and real uploaded inputs: DRY_RUN_VALID for fixtures 672–683;
  runtime_calls=false; qualification output still absent. Exact argv and env
  are in `CLI_DRY_RUN.json`.
- Actual `python -m research.bl_residual.f5_step3_package dry-run PACKAGE`: PASS.
  `offline_verify.py` independently verifies current engineering seal/package,
  all 563 retained historical evidence paths, the exact V4 machine diff, and
  supervisor offline validation. Qualification validates; measurement correctly
  refuses `dispatch requires a SHA-bound family canary receipt`.
- The initial 33-test exploratory run overlapped source edits and produced one
  source-inventory mismatch (32 passed); both stable-source runs above passed.

Reproducible sealing/dry-run command bodies, after the common nice/thread prefix:

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
V5=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v5
SEAL=/Users/josenunez/Projects/ml/snake-dqn-f5rawenv/research/bl_residual/ledger/F5_ENGINEERING_SEAL_20261026_V5/ENGINEERING_SEAL.json
"$PY" -m research.bl_residual.f5_engineering_seal seal "$SEAL"
"$PY" -m research.bl_residual.f5_step3_package seal "$V5/step3-package"
PYTHONPATH=/Users/josenunez/Projects/ml/snake-dqn-f5rawenv "$PY" "$V5/offline_verify.py"
"$PY" -m research.bl_residual.f5_step3_package dry-run "$V5/step3-package"
```

Sealing and offline report publication are create-only commands: rerunning them
against already sealed destinations refuses rather than replacing evidence.

## Frozen identities and machine diff

Source/manifest commit: `a2574377194b1688722eab4003ec9b697bde812d`, published on main.
The metadata-only publication and final verification commits do not alter its
233-file executable inventory.

| Artifact | SHA256 |
| --- | --- |
| Source inventory | `531892ad1d002cecfbd15283442f531e255eaf945c2dd6cdc291a10e6ca5e073` |
| Engineering seal | `cf4cb17ea7c02d62a5d30cb3740f39ecf7e51276ee32af98ceb4cc56f0be7093` |
| Package seal | `6c04949bea0306a009123baccb234a08c5b2d95646d5a41969551d514c8874d8` |
| Qualification manifest attempt03 | `8feb0c1380230c13fc0698a487cac78f560bd93202b1852ad564650e18a07d17` |
| Machine diff vs V4 | `4b0cbcba570c5170dd86aecfdc9ffeeeb93856a094851835a3d73636ef02a59b` |

The full JSON-pointer diff is `step3-package/MACHINE_DIFF_VS_V4.json`. Compared
with V4 qualification attempt02: 18 manifest rows change, limited to commit,
job ID, output/state paths and input references. Input paths relocate; only
control, empty ledger and ledger SHA sidecar change payload hashes. Control
changes namespace, three source hash entries, source digest and provenance source
digest. Ledger changes only identity namespace/source digest. Exactly three of
233 executable paths change: driver, package builder, engineering seal builder;
230 are byte-identical. The source patch is `SOURCE_DIFF_VS_V4.patch`.

Allocation, fixtures, workload payload hashes/sizes, 48,645,269 upload bytes,
environment contract, science/thresholds/look discipline, historical maxima,
thirteen retained completions, six host workers, threads=1, retry maximum,
archive/shard bounds, export bound and required exports retain V4 meaning.
Envelope bytes are identical. Preliminary qualification retains 15/30-minute
base/effective timeout, the 8-unit legacy guard, 64-MiB export limit, and normal
150-second transfer / independently 300-second staging limits. No transfer
exception is added. Measurement remains a HOLD template.

## Operator handoff and boundaries

The exact unexecuted attempt-03 command is saved at
`$V5/DISPATCH_COMMAND.sh` and copied to the tracked V5 package metadata. Its job,
manifest, broker, cwd and fresh state path were independently checked. The prior
attempt-01 disposition authorized attempt 02 only; no authority for attempt 03 is
inferred from it. No independent acceptance, CUDA result or family receipt is
manufactured. Existing acceptance/order and normal admission remain in force.

All V1–V4 seals, failed receipts and earlier evidence bytes remain unchanged.
No edits to `tools/sandbox/`, `research/sandbox_*`, transport code, tmux sessions,
released checkpoints, serving pins, protected docs or another worktree.
