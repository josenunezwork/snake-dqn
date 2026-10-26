# F5 step-1 attempt-01 disposition — 2026-10-26

Astra, independent F5 judge. Read-only review of clean `sol-f5-rawenvelope-20261026` at `8f9444483aa198f033e15d7cfa49850c9e8e2fe1`; executable pin remains `00af09a16913eb8ef3eab60003891657858dff69`.

**MECHANICAL NON-ATTEMPT for qualification/scientific completion; preserve the terminal label INCOMPLETE / UPLOAD_CAP. Authorize exactly one manually dispatched, otherwise identical preliminary qualification attempt 02 at transfer ≤150 seconds and independent whole staging ≤300 seconds. No further Astra pre-dispatch review is needed if Sol seals the exact three-field successor below and the operator verifies that machine diff and normal admission. The existing 300-second exception cannot be borrowed by this package.**

This is an affirmative bounded successor authorization, not permission for automatic retries. The original allocation, wall time and debit remain in append-only operational accounting. It produced no qualification or timing completion, consumes no scientific unit/look, and does not establish CUDA execution. Preserve attempt 01 and its seals unchanged. Another failure requires a new disposition; this ruling grants no attempt 03.

## Evidence and failure phase

All artifact paths in this section are below `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v4/step3-package`.

- `qualification-output-attempt01/receipt.json`: SHA256 `0a8e758c996348f5923e5bbc8f2502c486c4d726a8f031e49b56b4e7f94640a6`; one allocation; INCOMPLETE / UPLOAD_CAP; elapsed 247.70002616604324 seconds; error `transfers must complete within 150 seconds`; stop_verified=true; cleanup_error=null; terminal=null; exports=[]. Saved final usage has assignments=0 and rate=0. These are saved observations, not a fresh provider query.
- The receipt binds normalized `qualification-output-attempt01/manifest.json`, SHA256 `de2a5eeaa2878f444291ae8c6588e187d7218e2e4a640bf8b1ef36e112ad3470`. It equals the approved original plus the supervisor default `lease_seconds: 180.0`; this is normal serialization/defaulting, not unexplained manifest drift.
- `qualification-output-attempt01/upload-measurement.json`: whole staging 215.50497454195283 seconds; transfer clock 171.80560220812913 seconds; source delivery 41.273925042012706 seconds; setup 2.4232255830429494 seconds. The 13 transfer rows sum to 171.8046890432015 seconds; the small clock difference is enclosing bookkeeping. Largest row: best_apex_fs.pth, 3,451,633 bytes, 74.81394687504508 seconds. control.json: 17,922,579 bytes, about 18.289 seconds.
- `research/colab_cli/supervisor.py:1712` applies canary staging admission before the ready.json release and hardware/child launch. `research/colab_cli/canary.py:349` independently enforces whole staging ≤300 and `:354` obtains the transfer bar from the validated policy. The observed failure is thus before workload execution. Uploading all inputs and delivering source is not executing the qualification.

The owner's report of intermittent latency is consistent with the saved timings but is not needed to classify this failure. No outcome-dependent choice is involved.

## Exact immediate successor for Sol to seal

Create `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v4/step3-package/qualification-manifest-attempt02.json` from the original `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v4/step3-package/qualification-manifest.json`, original SHA256 `dd390beace928c4a15c39e4ba314bf61155a0c3575a87b561b7fde60406741b4`.

The complete permitted JSON semantic diff is:

| Field | Required successor value |
| --- | --- |
| `/job_id` | `f5-step3-qualification-v4-attempt02` |
| `/output_dir` | `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v4/step3-package/qualification-output-attempt02` |
| `/supervisor_state_dir` | `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v4/step3-package/qualification-state-attempt02` |

No other manifest field may change. In particular preserve commit, script, source_paths, every input path/hash/size/destination, empty-ledger identity, env including F5_STEP3_MODE=qualify, canary, required_exports, device, requirements, timeout, budget and resume_exemption. Do not add either transfer-exception field, substitute the old entrypoint, change upload to HTTPS, compress/remove inputs, or turn this into measurement. Normal supervisor default normalization remains permitted.

For the repository's canonical JSON encoding (`json.dumps(value, sort_keys=True, allow_nan=False) + "\n"`), the exact successor manifest SHA256 is **`b64a33b3cac214fa323766ea7332c2321f3def38003a3d27e93b422905f2de69`**. This hash was calculated in memory; the successor was not created or dispatched by this judge. Sol must create its sidecar and an append-only mechanical-successor seal binding original package/engineering seals, original and normalized manifests, failed receipt, this ruling and its sidecar, successor hash and the exact semantic diff. The old package seal is retained as ancestry, not rewritten to pretend it binds new bytes.

All three successor locations were absent at review. Operator must recheck freshness at admission. Original and current CLI subtree both equal qualified `5833e54e9c5396121cbbf99a5cd4e52006e564fd`; unchanged source therefore needs no transport requalification for this 150-second successor. The wrapper must still perform its normal current qualified-record, account/rate/assignment/lock and cleanup checks. No live state was checked here.

Once Sol seals and the machine diff passes, the exact next operator dispatch is:

```bash
nice -n 15 env \
  PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH" \
  PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn-f5rawenv/tools/ops/dispatch.sh \
  f5-step3-qualification-v4-attempt02 \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v4/step3-package/qualification-manifest-attempt02.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-f5rawenv \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-f5rawenv \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/sol-20261026/f5-hold-corrections-v4/step3-package/qualification-state-attempt02
```

This remains the V4 step-1-only 15/30-minute base/effective L4/CUDA preliminary qualification, 8-unit legacy guard and 64-MiB export bound. It authorizes no timing fixture execution. Apply the full subsequent order in ACCEPT_F5_HOLD_ENGINEERING_V4.md after genuine qualification PASS, hash-verified required exports and verified release. Timing collection and steps 4–8 retain their separate admission.

## Why 300 seconds is a different change

A prospective 300-second transfer / 300-second staging allowance is operationally reasonable for the current step-3 family, but it is **not already qualified or manifest-only**:

1. `QUALIFIED_TRANSPORT_F5ONLY_57a89e7d.md` is explicitly scoped. `research/colab_cli/transfer_policy.py:235` admits only `research/bl_residual/f5_colab.sh` with F5_DEVICE=cuda:0 and F5_MODE=qualification or qualification-canary, not `f5_step3.sh` with F5_STEP3_MODE=qualify/family-canary/measure. Changing only a context token does not admit this driver.
2. `transfer_policy.py:165` refuses uploaded inventory above 45,397,643 bytes. The current manifest uploads **48,645,269 bytes**, including B8 as an upload, not the older public HTTPS role. Thus the current package also exceeds that ceiling by 3,247,626 bytes. The earlier qualification's uploaded synthetic inventory was 37,603,798 bytes, with public B8 separately counted. Similar total size is not exact-inventory authority.
3. Later step-3 final/resume inventories add qualification, acceptance and/or `units/` roles. The existing exception admits only its base roles plus specific `timing-units/` triples; later packages cannot inherit an allowance merely by claiming the same approximate MB count.

Sol may separately prepare a narrow prospective step-3 policy successor naming this ruling, the exact driver/modes, L4/CUDA and exact inventories (including actual upload/HTTPS roles, byte ceilings and resume derivatives), preserving transfer≤300 and independently staging≤300. Preserve every scientific rule, historical maxima, thirteen completions, fixed fixtures/allocation, environment contract, cumulative execution envelope, retry maximum and sealed lineage. Do not alter global defaults or weaken unrelated families/D1024 gates.

That path requires negative scope/size/role/mode tests, renewed source/seal/canary identities, and independent re-review before any changed dispatch. A transport subtree change invalidates reuse of the old qualified-tree record: follow the normal source-bound requalification and independent qualification review; never copy/overwrite a registry to imply compatibility. This document does not authorize a 300-second attempt or treat a hypothetical relaxed pass as qualification. An alternative engineering design may be submitted, but changing entrypoint or inventory is outside the three-field authorization above.

## Review boundary and governing identities

- ACCEPT_F5_HOLD_ENGINEERING_V4.md SHA256 `e92d9caad66be313c127ff53ba614c82aae7b80c7dc106f630390c7b8ac5356d`.
- QUALIFIED_TRANSPORT_F5ONLY_57a89e7d.md SHA256 `eb58c9c28bdd2f572117c34b8e32d7cff2677ab490c726e64d527020852db752`.
- Original V4 engineering seal `e73b2d24144cd3e375aeffca1706dd5c677c3b23da95dce49db30d15ba3f9038`; package seal `72e739989e13a1adbee19cbf40e13f280092fe1d11bca43464a0cf0a4417ffa4`, as bound by the V4 acceptance. No replacement engineering acceptance is inferred.

Reviewed control-plane documents, manifests, saved receipt/transfer metadata and source only. No real label/outcome payload, checkpoint payload, control.json contents, D4096 output or t34-round0-study-01/ was opened. No Colab/provider call, dispatch, tmux operation, repository edit or commit occurred. Only this create-only disposition and SHA256 sidecar were written under the authorized Astra root. Verification used metadata/hash/normalization comparisons and source ordering; no numerical tests or workloads ran. Historical memory supplied only evidence-boundary context; all decisive facts were checked from current saved evidence.
