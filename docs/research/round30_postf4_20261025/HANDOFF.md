# R30 post-F4 preparation — 2026-10-25

**PREPARED_NOT_DISPATCHED. G0/G1/G2 accepted; G3 offline preparation accepted.
G4 native L4 execution is pending. G5 main admission is HOLD. Scientific decision:
NOT_RUN.** No Colab/provider call, allocation, dispatch, tmux operation, Mac-lease
file access/change, serving change or checkpoint modification occurred.

Runtime source: `a7893b97c73fcf3e7d7cef4bb3990636aa147f6b`, published on main.
Worktree `/Users/josenunez/Projects/ml/snake-dqn-r30post`, branch
`sol-r30-post-f4-20261025`. CLI tree remains
`c148431277f4e33c3f589251df892ea158407e25`. This handoff is a documentation-only
successor; keep the runtime pin and exact private input bytes frozen.

Authority: original ruling `f69c44863a8331028ade0dff20ee69eb9dc5959a92e8480a7329cb39d2b58d5c`
and amendment `df62897bf2624b64f701db60a1eb1725f06bc5b0acd017e57f54db3aae8d109d`.
The exact amendment is sibling RULING.md; prospective intent was published in
2eec1bbe before materialization. F4 stays STOP_FUNDING / R9 label 5; exact H was
not exported and was not recovered. Published terminal/seal metadata was checked
against cached origin/main. No D4096 tranche or protected study was opened.

## Delivered evidence

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1`.
Every listed JSON has SHA/completion sidecars except the clean-import checker’s
native receipt. PACKET_INDEX.json and CHECKSUMS.sha256 bind all delivered evidence.
Private label arrays and bundles remain outside Git.

| Ruling deliverable / gate | Artifact relative to root | Result / SHA256 |
|---|---|---|
| 1 / G0 authority publication | G0-publication.json | PASS; 9a699d7cb0b8e4ec8e770d8fabdc1304ab1401a85f49f22990db370db74f5e95 |
| 1 exact inventory/intent | metadata-v3/corpus-inventory.json; CORPUS_REUSE_INTENT.json | inventory 1dd5b4e42940d9f03cab8d735f8e66645de82fe1a6c5734b0a742540c125528e |
| 2 / G1 metadata ACCEPT | metadata-review-03/corpus-approval.json | e801de402a3a48ae5d9c8adf132d371081435bc5e92624013cbec900f85bf0e8 |
| 2 / G1 partition coverage ACCEPT | coverage-review-01/coverage-approval.json | 5b019a2494d8c80a42abb9c80b7965a06f0ed66e0bb25f2bcab39d5b9d273dd3 |
| 2 immutable partitions/map | partitions-02/partition-manifest.json | 358532112ab30e8ce51db41043be0236d83257377ef1814535e597dcc124c55d |
| 3 admission/control | admitted-02/control.json | Exact current source, rulings, inventory, coverage and sampler pins; see index |
| 4 exact scientific batch lists | admitted-02/sampler-plan.npz; sampler-plan.json | NPZ 9faeb8c592a58b4cd164f88381dd8c344df2b9b284d31f2f97bb70eae6b1b1ee |
| 5 transport + empty ledger | native-canary-01/private-inputs.tar.gz; admitted-02/verified-ledger.json | bundle af82651cec44d3878b91b7410c3b844bf890d779e0bceb305ac3d93a70ef9455 |
| 6 / G2 full current-source preflight | verification-03/preflight.json | PASS 106 tests; 0b654ff99262f397b8582437bfe9f63867dde908bd04cdcc762d55af500ef953 |
| 6 / G2 clean sparse import | clean-import-03.json | PASS; 8918f9c4049fb0d9582befdf2407156dfef1e85672afbf25f98ecf971eeb9ad9 |
| 6 / G2 independent code ACCEPT | code-review-01/review.json | f32cac9fbdbfce4a5fd48c5233acd7576c396b117eea9ff8667169d53e3a0678 |
| 6 / G4 exact-input canary preparation | G4_EXACT_INPUT_CANARY_PACKET.json | cbf8f2efd8ff41661eed1ccd9eca412c5d22faf859577d6fc17bc01cc8992763 |
| 6 / G5 final main packet | G5_FINAL_MAIN_ADMISSION_PACKET.json | HOLD; 840b9ff0051c8bfb918aec6a364f264a0ec8c826308d71240a44438983614842 |
| 7 frozen diagnosis preparation | DIAGNOSIS_BINDING_TEMPLATE.json; analyze.py | One canonical training-only look; execution binding/result pending verified main |
| Complete mapped delivery | FINAL_PACKET.json | 614168af6f04574da069863361f87844933658f13b33e94ea8088d52b90df427 |
| All evidence hashes | PACKET_INDEX.json | 6c5b99f79305b0afa29125ff19081a21d0a0337713eec152ed7b2469fc7faa8d |

## Exact admitted corpus

The 960 D1024-selected fold-1–4 lives include the identical 192 D256 prefix once.
Reserve, fold-0 ancestry, natural archive, unused tails and D4096 are excluded.
The unchanged salt produces 730 train lives/291 parents and 230 validation
lives/81 parents. All 81 validation parents are rescue-bearing. Train has 65,825
rows (ordinary 5,613, rescue 60,212); validation has 20,698 rows (ordinary 1,754,
rescue 18,944). No outcome-based trimming occurred. Validation is sealed locally.
The split is newly withheld from R30 fitting, not historically fresh evidence.

Global source-life strings and integer maps are bound throughout sampling and
metrics. The prospective plan freezes 2,000 seed-3403 batches of 128 rescue plus
128 ordinary samples and a separate 2,000-batch seed-3404 training audit, with row
IDs, multiplicities, recipe/source/partition identities. No features or fit are
constructed on the Mac by this plan. Actual frozen feature cache construction
and its numerical SHA remain L4 work, exported in the indivisible fit unit.

The resolved optimizer determines conversion control: Adam for ADEQUATE,
certified L-BFGS for MATERIAL, otherwise HOLD. MATERIAL initializer/procedure,
recipe, cache, sampler, control parameters and fused endpoint are reproduced and
checked for any successor. Certificate and conversion thresholds are unchanged.
Complete conversion below 0.10 explicitly closes ranking. Every diagnosis reports
archive status and successor blockers. R30-2 is HOLD without a separately admitted
natural archive and a passing resolved R30-1 diagnosis; no H=0 is fabricated.

## Verification and transport

Full current-source command (run at effective nice 15; CPU, all pools 1, no xdist,
shared serial slot, at most two processes; no Mac lease touched):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-r30post
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  ./venv/bin/python -m research.bl_round30.offline_check \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/bl-r30-postf4-20261025-v1/verification-03
```

The destination already exists; this is the recorded command, not a rerun target.
106 tests passed in 3.63 seconds (whole preflight 5.3763 seconds). Static Black
(workers 1), isort, flake8 (jobs 1), bash syntax and git diff --check passed. The
fresh sparse import used Python 3.12.15 and only declared dependencies: 102 import
statements, 35 local modules, no user/system site packages. The actual prepare
entrypoint passed and generated the current exact private package.

Private bundle: 36,189,672 bytes, 29 files, 39,344,799 unpacked bytes. Existing
public B8 route/hash retained. Recent measured L4 receipt model predicts transfer
118.4704/150 seconds (31.5296 seconds margin), staging 175.9621/300 seconds
(124.0379 seconds margin). These are forecasts; no native measurement is claimed.
No D1024 exception token or changed shared transport constant is used.

History charges all three R30 canaries/failures: 1,070.2578 whole supervisor seconds,
rounded up to 18 minutes, independently checked through verified stop. The new
canary reserves 20 minutes inside R30-1’s 144-minute envelope, leaving at most
106 main minutes. The historical 60-minute summed reservation explanation is
superseded as an operative charge by HISTORY_CLARIFICATION.json, without editing
old bytes. No unit ceiling/balance gate is imposed by R30; positive wall-derived
operational guard and separate technical reserve remain. Actual current rate,
capacity and qualification checks belong to the operator before execution.

Dry-run stdout, preserved under native-canary-01:

```text
PASS_DRY_RUN r30-postf4-native-a7893b97-01 allocations=0 live_accounting=NOT_CHECKED
REFUSE: manifest scientific_admission=false
```

The second line is the required refusal of the blocked main template.
OPERATOR_COMMANDS.md contains exact dry-run, later operator canary/main and frozen
Mac diagnosis commands. No cloud command without --dry-run was executed.

## Preserved preparation failures and next gate

partitions-01 remains HOLD: the first selector differed from its review SHA at
launch. The independent difference review is retained, the exact source guard
was added before access, fresh metadata-review-03 prospectively approved it, and
partitions-02 was rebuilt and independently approved. Both 77.5767 and 68.0734
seconds count against the R30-0 allowance. No scientific or validation look occurred.

admitted-01 remains INVALID after preflight caught manifest reserialization;
exact raw-byte/sidecar copying fixes it in admitted-02. The first clean import
failed on an unwanted transitive collection dependency; a hermetic local fixture
removed it, and current clean-import-03 passes. Prior artifacts were not rewritten.

Next: operator pre-canary review/admission, then G4 exact-input same-pin L4 proof.
Require actual CUDA observation, standard 150/300 clocks, verified exports,
terminal PASS and runtime disappearance. Bind a separately reviewed Mac-only
runtime forecast without changing private/series bytes, then obtain independent
G5 main-packet ACCEPT. Main and the one Mac training-only diagnosis remain NOT_RUN.
