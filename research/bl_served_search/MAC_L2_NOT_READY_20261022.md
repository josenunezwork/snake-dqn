# L2 Mac prerequisites: NOT_READY, 2026-10-22

Outcome: **NO BANK LAUNCH. P3 is INCONCLUSIVE; independent review is REFUSED.**
This status sibling supersedes the earlier implementation-review PASS receipts
as launch authority. The registration, Mac amendment, bank, audit, chunk order,
scientific rules, B8 checkpoint and serving configuration remain untouched.
No bank world was started. No Colab call or unit expenditure occurred. The
sealed Round 0 artifact directory was never opened.

## Blocking production/ruler contract

The production path calls historical `bank_engine.build_world`. Its evaluator
constructs vector opponents with `veto_slots=()` and `veto_variant="v2"`:
`bank_engine.py:429`, `src/simd_env/eval_engine.py:904`. Their action selector
only invokes the veto when slots are nonempty (`vector61_policy.py:798`).
`bl_k1.native.configure` routes existing veto-hook calls; it cannot enable a
missing hook. The ruler separately calls `_configure_veto(1..5, "v8", 8.0,
None)` (`ruler_operator.py:140`). That configuration is absent in production.

Consequently, the unchanged production opponents cannot produce the native-v8
activity required by P1(c). Enabling the ruler's opponent veto in production
would change the registered opponent behavior. Relaxing the counter rule would
violate the judgment. Neither was done. Mode B also requires the missing kernel
activity and cannot bypass this contradiction. A prospective independent judge
correction is required before further admission: resolve the historical
opponent estimand and execution qualification explicitly. No bank retry or new
scientific protocol is authorized by this handoff.

## Implementation and publication

- `bbb4ee79b12ae21d2baa93394212827bdae78673`: pre-data Mac amendment.
- `1ae30c695ca72ea7cfb024320fd3e11d8f9a84bb`: production mode/counter wiring,
  fail-closed analyzer, Mac parity helper, sequential leased launcher, P3 harness.
- `a8c4e6c4c07a358b99b181cc43c89f04d7585fa8`: preparation-only materializer fix.
- `61ae9afb02ad5029b95e0e30699b27c0b45c37a5`: published P3 run checkout.

The launcher is fixed at two spawned workers, one numerical thread, nice 10,
`caffeinate -is`, eight sequential 8/8/4 chunks, 142-minute chunk timeouts and a
28.3-hour owned lease. It records a fresh quiet start receipt and host samples,
checks typed review/P3 receipts and Mac-local runtime parity, and consumes only
operational attestations plus opaque shard hashes before the one look. It does
not retry. None of the bank execution machinery was started.

## Immutable evidence

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/mac-prerequisites-01/`.
Every JSON receipt has its `.sha256` and `.complete` export markers.

| Evidence | Status | SHA256 |
|---|---|---|
| `review-03.json` (superseding independent review) | REFUSED / NOT_READY | `4c3e58d104ed97b5aa44ad5d7430aede34ed6bb9da1b22d80a72792ea54b05d3` |
| `p3-identity-02/receipt.json` | INCONCLUSIVE | `8995f839aa98bac46f32a20e8467cc5fe62ee7833939764425ab7970c7cb8d3b` |
| `P3_SUMMARY.json` | INCONCLUSIVE / NO LAUNCH | `af76a54b538a7b3fdeeed2e0d6c4c4f7f5cd3007f20647c751bfe6ddd0e7aa40` |
| `runtime-parity-02/runtime-parity.json` | VERIFIED, H500 fixture-only | `62a7bc3edc54c19c996fb48c0d4883a71fad9100163bacaf9648114e06a69508` |

Tracked copies: `MAC_REVIEW_20261022_03.json`, `MAC_P3_SUMMARY_20261022.json`,
`MAC_RUNTIME_PARITY_20261022.json`. Earlier review/parity files are retained;
they do not establish production-path identity or authorize launch.

The P3 run allocated five fresh H5000 development worlds (2 frozen, 2 mixed,
1 scripted), whole-allocation audit round 1 CLEAR. A first materialization
attempt failed before any world; its zero-world receipt and closed round-0
identities were retained before the fresh whole-round audit. The corrected
materializer received a new independent review and source-bound parity run
before gameplay. The runtime source-manifest digest is
`36b5f99b675e12d7fe88d57a1b171779b0115fcaf60f0944c2acc2f508257049`.

P3 ran under nice 10 / caffeinate with at most two one-thread workers. After
independent confirmation of the contract fault, the owner of the run sent
SIGTERM to its verified coordinator PID, triggering owned-worker cleanup and
an immutable INCONCLUSIVE receipt. Elapsed: 2014.6695320840226 seconds. Only two
`all` frozen worlds completed; no exact-stop rows completed. This is neither a
five-world identity PASS nor an observed action mismatch.

| Completed dev world | Fired batches | Evaluated branches | Kernel rows | Native calls |
|---|---:|---:|---:|---:|
| round-1/frozen/1 | 491 | 2542 | 0 | 0 |
| round-1/frozen/0 | 575 | 3085 | 0 | 0 |

The actual analyzer returns `missing native branch rows` for both receipts.
The independent reviewer verified the exports, source hashes, allocation/audit,
counters and analyzer refusal. Coordinator PID 29875 and workers 29955/29956
are absent; their shared lease is absent. `bank-admission-refusal.json` records
that the real P3 admission gate refuses this receipt before creating a lease or
bank worker. No outcome statistics or bank look were computed.

## Verification

- 45 focused tests passed, including mode forwarding, worker native scope,
  counter/frame/thread refusal, exact-stop decisions, no-peek operational
  validation, receipt-type admission, lease cleanup and roster materialization.
- Black, isort, flake8, shell syntax/check, diff check and `make lint` passed.
- Broader research-package run: 149 passed, 1 legacy transport-fixture failure
  (manifest missing the current required `budget`; those files are outside this
  task's diff).
- Repository non-slow run was interrupted after failures: 3337 passed,
  3 skipped, 13 failures in `test_perf_compile.py`, all reporting missing
  `@rpath/libc++.1.dylib` in PyTorch Inductor. Full-suite completion is not
  claimed. No environment/library repair was attempted.
- Exact commands and check evidence: artifact-root `verification.json`.

Primary focused command:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  research/bl_served_search/test_mac_identity.py \
  research/bl_served_search/test_mac_execution.py \
  research/bl_served_search/test_mac_l2.py \
  research/bl_served_search/test_mac_parity.py \
  research/bl_served_search/test_exact_stop.py \
  research/bl_served_search/test_ceiling.py
```

## Claude operator command: currently refused, not cleared to launch

The exact invocation below binds the current receipts and deliberately fails
admission. A corrected judgment and replacement reviewed runtime evidence are
required before a bank-ready command can be supplied. Do not substitute older
PASS receipts, change opponent behavior, relax counters or retry the bank.
The `.noindex` option was not selected; the registered output root is unchanged.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-l2mac
bash research/bl_served_search/run_l2_mac.sh --check

tmux new-session -d -s l2prime-mac-20261022-01 \
  'cd /Users/josenunez/Projects/ml/snake-dqn-l2mac && set -C && exec bash research/bl_served_search/run_l2_mac.sh --run \
    --bank /Users/josenunez/Projects/ml/snake-dqn-l2mac/research/bl_served_search/BANK_L2.json \
    --b8-path /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
    --runtime-parity-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/mac-prerequisites-01/runtime-parity-02/runtime-parity.json \
    --review-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/mac-prerequisites-01/review-03.json \
    --p3-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/mac-prerequisites-01/p3-identity-02/receipt.json \
    > /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l2/mac-amendment-01.launch.log 2>&1'
```

No pre-existing tmux session, released checkpoint, serving pin, historical
registration or hash-bound receipt was modified. One automatic merge commit,
`9a20cb020ed1fb2afb4dc80de0f7274e47c3c691`, omitted the requested co-author
trailer; shared published history was preserved. Authored step commits and
subsequent merges carry the requested trailer.
