# A2 large-input transfer repair — operator handoff

Use **pack-20261022-05**. Prepared only; Claude/operator dispatches. Full source
pin: `cb67a4b90ec2577ed398d76b59c18427d6ae764a`. The final documentation commit
may be newer; dispatch the frozen pin in these manifests.

The failed main receipt is INCOMPLETE / UPLOAD_CAP, zero exports, stop verified.
Its initial input transfer ran into the shared 300-second staging deadline.
The preceding pack-03 canary actually uploaded the complete original 91,491,883-byte
archive in 62.451 seconds (77.624 seconds for all uploads); the canary workload
was synthetic and small. The main's transfer/renewal channel was much slower.

The repair projects the 360 T4e NPZ shards to the three columns A2 reads:
`source_decision`, `state_packed`, and `state_scalars`. It retains original dtypes,
first-seen order, and exact bytes. Duplicate decisions are removed only after all
three retained values compare byte-for-byte; conflicting duplicates refuse.
103,264 repeated observations become 19,976 unique observations. No float16,
label conversion, source-row selection, fold change, or training change occurs.
All A5 JSON, T5 tables/folds, opponent checkpoints and original manifests retain
their original bytes. XZ preset 9 compresses the archive. Original bundle receipt,
source manifests and projection lineage remain inside the archive and are verified
before use. Both the synthetic canary and main exercise decompression/lineage.

Original scientific archive SHA remains
`6572e096f1e6c08b1b5dab1cc9d829b6931dfe3287491ae3d91c81e32360d68c`;
allocation remains
`8045413220ac81e2ef7fc06435dabe6d83af4f8b09287c95c61b99943d88cc3b`.
The original empty verified ledgers are copied unchanged. New transport archive:
**34,180,148 bytes**, SHA
`b9774b034f03a562c88b5055e74ab84aeadc17191ac60526244494d9be600bc6`.
All five manifest inputs total **40,743,637 bytes**, versus about 98.06 MB before.

Forecast from actual sizes: **135.81 seconds** for all inputs at the reported slow
0.3 MB/s effective rate; **32.26 seconds** at the preceding canary's aggregate
measured upload goodput. Bootstrap/source add 17,755 bytes plus small-file RPC and
renewal overhead. Allow roughly **150–160 seconds** for staging at the slow rate;
this is a forecast, not a new remote measurement. The existing family gate still
requires **<=150 seconds measured transfers** and **<=300 seconds total staging**.
A slow canary may refuse at that stricter transfer gate; never waive it or reuse
old proof. The reduced payload gives the main substantial staging headroom, but
cannot remove RPC/network variability.

No `research/colab_cli` bytes changed. Qualification check returns
`553af3f692515b2bf66173356fe4d099acc96942`; no full supervisor requalification
is needed. Drive is prohibited on this CLI path, and this supervisor has no
per-manifest upload-cap override. Sharding manifest inputs would not reset its
shared staging deadline. New code/payload requires a **new same-pin family canary**.
The actual validation rejects both an unadmitted main and the old canary receipt.

Prior allocations are all charged at the conservative 10u/hour ceiling:
pack-02 failed canary 792.488650s + pack-03 passed canary 527.832287s + pack-03
failed main 352.604430s = **1,672.925366s / 4.647014907u**. Reserve the new canary
18min/3u; main gets **242min/40.35u**, including its 8u reserve. Maximum sum:
287.88209min/47.997014907u, inside original 288min/48u. The owner 2x rule was
already applied to that series envelope and is not applied again. Forecast stays
120min; 242min satisfies its 2x timeout requirement. This is completion attempt 1
of at most 2. Complete receipt ancestry is enforced before pack preparation.
No cap reset, N extension, automatic retry, or scoring look is authorized here.

The pack accounts for the one other endpoint observed at preparation:
`gpu-a100-s-usc1c0-780pn6afwnq5`, 5.3u/hour (one assignment and total 5.3u/hour).
If account assignments/rates change, build a fresh sibling with reviewed rates;
the supervisor refuses an unaccounted endpoint. Preserve immutable packs.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-a2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
PACK=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a2/pack-20261022-05
python3 "$REPO/tools/ops/qualification.py" check --repo "$REPO" \
  --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
"$REPO/tools/ops/dispatch.sh" a100-a2-canary-20261022-03 "$PACK/canary.json" \
  --broker "$REPO" --cwd "$REPO" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-canary-20261022-03 \
  --dispatch-txt "$PACK/DISPATCH.txt"
# Wait for genuine PASS, <=150s transfers, verified exports and stop_verified=true.
cd "$REPO"
"$PY" -m research.a100_a2.make_pack admit --template "$PACK/main-template.json" \
  --canary-receipt "$PACK/canary-output/receipt.json" --out "$PACK/main.json"
"$REPO/tools/ops/dispatch.sh" a100-a2-main-20261022-03 "$PACK/main.json" \
  --broker "$REPO" --cwd "$REPO" \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-main-20261022-03 \
  --dispatch-txt "$PACK/DISPATCH.txt"
```

Offline evidence:

- All 24 A2 tests passed in 1.67s with `-p no:xdist`, single native thread.
- 15 focused existing canary tests passed, 60 deselected, 0.53s, same process limit.
- Actual Bash entrypoint, complete compact inputs, CPU synthetic workload: 8/8 fit
  units, 6.555s including archive verification. No historical fits or replay ran.
- Every one of the 12 canonical T5 arrays has identical shape, dtype and SHA256
  before/after projection; all unprojected members match their original hashes.
  `transfer-probe-20261022-01/FINAL_BYTE_PARITY.json` records this comparison.
- Pinned clean Python 3.12.15 dependency preflight: PASS, 545 import statements,
  180 local modules, system/user site packages excluded.
- Real supervisor canary validation: PASS, `pack-20261022-05/CANARY_VALIDATE.txt`.
  Main refusal evidence: `MAIN_ADMISSION_REFUSALS.json` in that pack.
- Black, isort, flake8 and Git whitespace checks passed for changed Python files.
- Independent Codex-V receipt is recorded with this handoff and beside pack-05.

Reproduce the narrow suite and style checks:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-a2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu "$PY" -m pytest -q -p no:xdist \
  tests/test_a100_a2_data.py tests/test_a100_a2_fit.py tests/test_a100_a2_protocol.py \
  tests/test_a100_a2_aggregate.py tests/test_a100_a2_job.py tests/test_a100_a2_transfer.py
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
"$PY" -m pytest -q -p no:xdist tests/test_colab_cli_canary.py \
  -k 'receipt_cannot_admit_an_uncovered_dispatch or probe_retains_real_input_inventory or staging_accepts_slow_source_delivery'
"$PY" -m black --check --line-length 100 research/a100_a2/{transfer,make_pack,job,data}.py tests/test_a100_a2_transfer.py
"$PY" -m isort --profile black --check-only research/a100_a2/{transfer,make_pack,job,data}.py tests/test_a100_a2_transfer.py
"$PY" -m flake8 research/a100_a2/{transfer,make_pack,job,data}.py tests/test_a100_a2_transfer.py
git diff --check
```

Pack-04 is preserved as an undispatched diagnostic: the invocation used a short
commit pin; `SUPERSEDED.txt` marks it unusable. The builder now resolves all pins
to 40 characters before publication. Only pack-05 is covered by this handoff.
All old receipts, ledgers, registrations and packs remain untouched. No live
runtime, tmux session, released checkpoint, serving pin, or qualified supervisor
source was modified. New remote transfer timing/CUDA family proof remains for
the operator; no Colab allocation or scientific look was performed by this task.
