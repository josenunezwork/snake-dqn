# F1 amendment A1 — CPU-only pack-10a

Create-only CPU pack:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/f1-pack-10a/`.
Authority: `big-lever/round-24/JUDGMENT_F1_REFUND.md`, including A1, SHA256
`86e9ca69383d21a45ede31fd721d315ff39ae09736ad538496ce42b1abf1ac72`.
Unrestated terms retain pre-A1 authority
`f2f35631e65e4234341a5404e3f512a740b43241c9060f4cf8a467673091b6bc`.
The pack's own `DISPATCH.txt` releases only its CPU probe and CPU ruler,
conditional on the independent Codex-V delta PASS recorded in
`CODEX_V_F1_A1_DELTA_20261023.json` beside this handoff.

Worker pin remains `2af48cd0b98a2063f46871cf4d320a5541b789af`; qualified
transport tree remains `d0d92bdb572b770284a1fea6152417ffa5f06bea`.
There are no runtime, transport or generator source changes. The Mac-side
analyzer accepts the exact ratified A1 ruling-file hash while still requiring
the original immutable census authority hash. It also accepts the original
ruling-file hash; every other ruling hash is refused. This narrow compatibility
fix is necessary because A1 requires unchanged census bytes while appending to
the authority file at the census's existing path. No decision rule changes.
Pack-10, including its original frozen dispatch instructions, is untouched.
L4 remains an independent operator lane; this package grants no L4 dispatch.

The CPU probe changes exactly A1's four time fields to 30 minutes / 1800 seconds,
adds `f1.amendment` with the full post-A1 authority SHA, uses job ID
`bl-f1-cpu-probe-10a-2af48cd0` and output `f1-output-10a/cpu-probe`.
Estimate 10 minutes, finalize 120 seconds, cap 0.07 u and reserve 0.014 u inside
the cap are unchanged. The CPU ruler template changes its job ID to
`bl-f1-cpu-ruler-10a-2af48cd0`, output to `f1-output-10a/cpu-ruler` and adds
the same amendment field. Its 180-minute / 0.30 u cap, 0.06 u reserve,
45-minute estimate and 300-second finalize allowance are unchanged.

Bank, CPU ledger, refund census and all input references/hashes remain exact
pack-10 values. Their supporting triplets are also copied unchanged into 10a.
Bank logical SHA256:
`a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d`.
`budget-plan.json` records the corrected CPU probe effective 30 minutes and A1;
`envelope-account.json` binds that plan and records A1's effective time and
the new package preparation status. Every unit figure is preserved, including
the original floating-point serialization of the probe reserve.
Fresh F1-R remains 4.04 u; historical 0.153 u is carried separately, giving
4.193 u conservative liability within 4.20 u and the 5.0 u ruling ceiling.
Original plus at most one manual mechanical completion per stage shares the
same cap. Any new INVALID requires a new ruling; no automatic completion retry.

The index-6 carve-out remains the initial generation-bound
`lease-heartbeat.json` Contents upload by `CLI._put_lease`, before runtime-root
creation, witnessed in the qualified PASS. Other failed calls still count.
Both pack-08 packaging INVALIDs remain in the census. Rates come only from the
main rulers; the 25% placement rule is unchanged. F0 admission, its Mac audit,
strict gates and serving authority remain separate.

## Verification

- Actual supervisor CLI probe validation: exit 0; evidence in
  `f1-pack-10a/cpu-probe-supervisor-validate.log`.
- Probe supervisor API validation: PASS. Ruler template refuses the missing
  SHA-bound family-canary receipt, as required; no fabricated receipt is used.
- Both full CPU upload qualifications: PASS; forecast 149.7382728764186 seconds
  below 150, source-fetch-inclusive bound 269.7382728764186 seconds below 300.
- Initial package/family-canary regression selection: 153 passed in 9.66 seconds,
  single-process, CPU, with all numerical thread caps set to 1. The final run
  adds five authority-gate cases covering original/A1 acceptance, foreign-file
  refusal and refusal of a changed census authority. Exact final results are
  bound in the independent delta receipt and pack inventory.
- Actual unchanged census loading passes with the post-A1 ruling file and
  both historical bindings, without opening new outcomes; evidence in
  `a1-analyzer-authority-verification.json` in the pack.
- Pack-10 baseline inventory is frozen in `pack-10a/pack-10-baseline.json`;
  the independent delta receipt records final integrity checks and exact hashes.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu F1_REAL_ARTIFACT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist -p no:cacheprovider research/bl_labels tests/test_colab_cli_canary.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.colab_cli.supervisor --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/f1-state-10a-cpu-probe --colab /Users/josenunez/.local/bin/colab validate /Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/f1-pack-10a/f1-cpu-probe.json
```

## Operator commands

Preparation only: no CPU job was allocated or dispatched by this task.
The probe and ruler commands deliberately use their own fresh state paths.
Run the ruler block only after the actual same-pin CPU probe PASS, verified
exports and verified release; `bind` enforces that proof.

```bash
BROKER=/Users/josenunez/Projects/ml/snake-dqn-acctfix
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f0
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0
PACK="$ART/f1-pack-10a"
OUT="$ART/f1-output-10a"
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
cd "$SOURCE"
"$PY" "$BROKER/tools/ops/qualification.py" check --repo "$BROKER" --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json

"$BROKER/tools/ops/dispatch.sh" bl-f1-cpu-probe-10a-2af48cd0 "$PACK/f1-cpu-probe.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir "$ART/f1-state-10a-cpu-probe" --dispatch-txt "$PACK/DISPATCH.txt"

# Only after CPU probe PASS, verified exports and release:
"$PY" -m research.bl_labels.f1_manifests bind --template "$PACK/f1-cpu-template.json" --receipt "$OUT/cpu-probe/receipt.json" --out "$PACK/f1-cpu.json"
"$PY" -m research.bl_upload.qualification --manifest "$PACK/f1-cpu.json" --receipt "$PACK/cpu-bound-upload.json"
"$BROKER/tools/ops/dispatch.sh" bl-f1-cpu-ruler-10a-2af48cd0 "$PACK/f1-cpu.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir "$ART/f1-state-10a-cpu-ruler" --dispatch-txt "$PACK/DISPATCH.txt"
```

After both rulers finish and release (or a platform is excluded), use the
pack's `ANALYZE.txt`. It retains both pack-08 probes and pack-10 L4 attempts,
replacing only the pack-10 CPU entries with pack-10a CPU probe/ruler entries.
Include every additional manual completion as another `--attempt` input.

No existing tmux session, released checkpoint, serving pin, legacy worker,
qualified transport file or existing hash-bound artifact was modified.
