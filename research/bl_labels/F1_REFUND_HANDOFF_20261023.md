# F1 refund pack-10 — qualified-canary cap conflict

**HOLD.** Pack-10 preserves the exact ruling and source, but cannot obtain both
probe validation passes under the required qualified transport. CPU probe 40 min
is refused by `research/colab_cli/canary.py`'s 30 min family-canary maximum.
Independent Codex-V review is recorded in the sibling receipt. No Colab job was
allocated or dispatched. This sibling supersedes neither old manifests nor their bytes.

Worker pin: `2af48cd0b98a2063f46871cf4d320a5541b789af`.
Qualified CLI tree: `d0d92bdb572b770284a1fea6152417ffa5f06bea`.
Authority: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-24/JUDGMENT_F1_REFUND.md`,
SHA256 `f2f35631e65e4234341a5404e3f512a740b43241c9060f4cf8a467673091b6bc`.
Authority was already logged on origin/main at 58967388. Mac implementation at df90e5fa;
worker runtime and qualified CLI files are unchanged.

| Stage | Minutes | Cap units | Reserve units inside cap | Supervisor |
| --- | ---: | ---: | ---: | --- |
| CPU probe |40|0.07|0.014|REFUSED: qualified canary maximum 30 min|
| CPU ruler |180|0.30|0.060|Await family PASS binding|
| L4 probe |20|0.65|0.130|PASS|
| L4 ruler |94|3.02|0.604|Await family PASS binding|

Fresh F1-R 4.04 u plus historical 0.15260654936326223 u = 4.192606549363262 u,
within 4.20 u account and 5.0 u effective ruling ceiling. The supplemental `envelope-account.json` carries the ruling's rounded 0.153 u
as spent (conservative lifetime liability 4.193 u). Historical spend is not
charged again against fresh stages. Estimates 10/45/5/23.5 min and finalize 120/300 s
are unchanged. Reserves are 20% of the rounded ruled cap and lie inside it.
The owner multiplier is recorded once; caps are not doubled again.
Original plus at most one outcome-blind manual completion within each stage cap.
The standard resume declaration retains its mandatory retry maximum 2; the study's
stricter completion maximum 1 controls. Any new INVALID requires a new ruling.

`refund-census.json` binds both exact pack-08 manifests, receipts and INVALID
exports, plus the qualified canary-05 PASS witness. Both old attempts stay in the
analyzer output as `PACKAGING_INVALID_EXCLUDED (JUDGMENT_F1_REFUND)` without rates
or platform-exclusion faults. The sole call carve-out is the zero-based index 6
initial Contents upload of generation-bound `lease-heartbeat.json`, before the
runtime directory exists. Other pack-10 failed or exhausted calls still count.
The 25% rule, main-only rate selection and kill logic remain intact.
A100 is explicitly `NOT_RUN by ruling` in `f1-a100-NOT_RUN.json`.

Artifact pack: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0/f1-pack-10`.
Outputs and all four state paths are fresh and absent. Bank logical SHA remains
`a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d`.
No Watch6 result, F0 admission, Mac-worker yield, strict gate or release is implied.

## Verification

- 153 package/family-canary tests passed in 8.86 s, single process `-p no:xdist`.
- Black 100/isort/flake8, Bash syntax and `git diff --check` pass.
- Fresh declared-only Python 3.12.15 import preflight: 8 roots, 679 statements,
225 local modules; user and system site packages disabled.
- All four full upload qualifications PASS. Maximum fallback forecast 149.738273 s
< 150 s; source-fetch-inclusive 269.738273 s < 300 s.
- Actual supervisor API validation plus both probe CLI validations: CPU refuses
40 min vs 30 min; L4 PASS; templates refuse missing family receipts as required.
- Actual native `f1_ruler run --probe --workers 1` with manifest-staged inputs:
COMPLETE in 15.2643165 s; five verified export triplets. Both ledgers pass at exact
runtime destinations. Evidence: `bl-labels-f0/f1-functional-14`. Functional only,
not a live family proof or timing benchmark.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-f0
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu F1_REAL_ARTIFACT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist research/bl_labels tests/test_colab_cli_canary.py
```

The pack's `verify_pack.py` contains the exact four qualification/API-validation
operations. It publishes create-only evidence and requires fresh receipt paths
when repeated. Probe CLI invocations and return statuses are retained in
`cpu-probe-supervisor-validate.log` (exit 1) and `l4-probe-supervisor-validate.log` (exit 0).

## Reserved operator sequence — currently blocked

`DISPATCH.txt` is frozen HOLD; wrapper refuses both probes. The command sequence
below is retained for review, not an instruction to change a cap or bypass a guard.
A resolution must preserve old evidence and produce fresh reviewed instructions.
Once admitted: probes in parallel, each ruler immediately after its own PASS,
then analyze all pack-08+pack-10 attempts (and any completion attempts).

```bash
BROKER=/Users/josenunez/Projects/ml/snake-dqn-acctfix
SOURCE=/Users/josenunez/Projects/ml/snake-dqn-f0
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/bl-labels-f0
PACK="$ART/f1-pack-10"
OUT="$ART/f1-output-10"
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
cd "$SOURCE"
"$PY" "$BROKER/tools/ops/qualification.py" check --repo "$BROKER" --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json

# Reserved parallel probes. Current DISPATCH.txt is HOLD and refuses both.
"$BROKER/tools/ops/dispatch.sh" bl-f1-cpu-probe-10-2af48cd0 "$PACK/f1-cpu-probe.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir "$ART/f1-state-10-cpu-probe" --dispatch-txt "$PACK/DISPATCH.txt" &
"$BROKER/tools/ops/dispatch.sh" bl-f1-l4-probe-10-2af48cd0 "$PACK/f1-l4-probe.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir "$ART/f1-state-10-l4-probe" --dispatch-txt "$PACK/DISPATCH.txt" &
wait

# CPU ruler: only after its actual same-pin CPU probe PASS, verified exports/release.
"$PY" -m research.bl_labels.f1_manifests bind --template "$PACK/f1-cpu-template.json" --receipt "$OUT/cpu-probe/receipt.json" --out "$PACK/f1-cpu.json"
"$PY" -m research.bl_upload.qualification --manifest "$PACK/f1-cpu.json" --receipt "$PACK/cpu-bound-upload.json"
"$BROKER/tools/ops/dispatch.sh" bl-f1-cpu-ruler-10-2af48cd0 "$PACK/f1-cpu.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir "$ART/f1-state-10-cpu-ruler" --dispatch-txt "$PACK/DISPATCH.txt"

# L4 ruler: independently, immediately after its actual L4 probe PASS and release.
"$PY" -m research.bl_labels.f1_manifests bind --template "$PACK/f1-l4-template.json" --receipt "$OUT/l4-probe/receipt.json" --out "$PACK/f1-l4.json"
"$PY" -m research.bl_upload.qualification --manifest "$PACK/f1-l4.json" --receipt "$PACK/l4-bound-upload.json"
"$BROKER/tools/ops/dispatch.sh" bl-f1-l4-ruler-10-2af48cd0 "$PACK/f1-l4.json" --broker "$BROKER" --cwd "$SOURCE" --state-dir "$ART/f1-state-10-l4-ruler" --dispatch-txt "$PACK/DISPATCH.txt"

# After both rulers finish and release, include all pack-08 + pack-10 attempts.
# Add every further manual mechanical completion as another --attempt input.
"$PY" -m research.bl_labels.f1_analyze --bank "$PACK/f1-root-bank.json.xz" --refund-census "$PACK/refund-census.json" --attempt "$ART/f1-output-08/cpu-probe" --attempt "$ART/f1-output-08/l4-probe" --attempt "$OUT/cpu-probe" --attempt "$OUT/cpu-ruler" --attempt "$OUT/l4-probe" --attempt "$OUT/l4-ruler" --out "$ART/f1-decision-10/decision.json"
```

No existing tmux session, released checkpoint, serving pin, legacy worker,
qualified CLI code, old pack or existing hash-bound receipt was modified.

## Prepared artifact hashes

| File | SHA256 |
| --- | --- |
| `DISPATCH.txt` | `344c821d3aaa4da112d47f7491cc0297bbcc97e416e533bc414937beea16bee1` |
| `bank-transport.json` | `1a79cc6bdfd9da9c69f7dc5b8448fc09cdeaa6a4278edce6c1c0247af57eac79` |
| `budget-plan.json` | `3b3aa321c619928304cc040ccfeeb762e32e654194c8233a747776a426fe1341` |
| `clean-import-preflight.json` | `3c1b318fda5f9a1de9a4b9369c5150c28109ea11d5abbc3b01eb9871b68e7a3c` |
| `cpu-probe-supervisor-validate.log` | `7a610b5756bdb228726f2184fc9931eeade1a8c6b05c735f8bce61c9b246767e` |
| `cpu-probe-upload.json` | `0253f4a620eed150dfd231959b6283e94885095cb141d27d30332bdec9e09c0b` |
| `cpu-template-upload.json` | `bda7f053c0fc8dff79a7ab6f1bf5a78d06ccf427b0d3686dcb97132dcb4a464e` |
| `f1-a100-NOT_RUN.json` | `3f5c2dcd61f60a88ae32eb2d5a2275660d1e3c4e2c0b96f5a5e7cf95ef3296be` |
| `f1-cpu-probe.json` | `112398201db76ccd6f9ecc90b3cc26c3571ba4481efcee8bd69224d68a30d22d` |
| `f1-cpu-template.json` | `e0949b95a410d799fd1399568dc00724f28340860729061ef4ca4046847f4b46` |
| `f1-l4-probe.json` | `bc27084314cffe2e26c3a78e13b4db4fb38434815c67e603f54a3ff25bfcc1e3` |
| `f1-l4-template.json` | `047acd843040a3b4dd051bc4b6921df757f524f0795697f5749e3071d96a4ee5` |
| `f1-root-bank.json.xz` | `f626e2510d1c2f7620d2463c3eece089f9d11e72b3c37d8c873f0b68b1f2b995` |
| `l4-probe-supervisor-validate.log` | `b46be970de5c0d4fde3db04b1dfa26481f91eef4caa09b130f4871cdc7adf4fb` |
| `l4-probe-upload.json` | `d69b858bdfb39aade4769c773b8fffea559bd7d5d4ca1023cc8d8b39eb6efd69` |
| `l4-template-upload.json` | `cf1943251eff697da5795a4cd96578b119a2ec1f74e4d09cd684cbfd0b0f440e` |
| `refund-census.json` | `248e5091ea1992eb045bf411557394ff4779ad04bf53e35da84befa618449b6f` |
| `validation.json` | `35d4d76f2057033fd14d400db0dbbfaa1e31782d62c8d85a3d18529144aa73ea` |
| `verify_pack.py` | `37d806c00726d0d954c6f4a57bc16e324b214cdf23b17e7e15510e7c3509f5b3` |
| `envelope-account.json` | `6603253bd0e6b9a96070dfa9650b4ce3c5655eca810ae044e5f48ef50849e776` |
