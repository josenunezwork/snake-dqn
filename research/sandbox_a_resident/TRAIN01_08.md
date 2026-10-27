# TRAIN-01-08 — detached diagnostics and update-16 proxy

Prepared for simulator-D development under Astra's 2026-10-27 ruling, A items 1–5.
No runtime has been dispatched. The learning recipe, training/task seeds and state,
finite 384-update / 100,663,296-row endpoint, original nomination set and 8-hour
base / 16-hour effective runaway guard are retained. Packet train01-07 remains
separately sealed with its original bytes and startup-screen behavior.

The new candidate-monitoring schedule is 16,64,128,192,256,320,384, each on the
same cached B8 bank with 96 worlds/mix and H5000. Update 16 is diagnostic only:
it adds at most 1,440,000 candidate world-ticks and introduces no new stop band,
nomination candidate or outcome-based retry. The initial B8 agreement/cache
check remains in place. Proxy execution preserves RNG streams and every module's
train/eval mode.

Per-update diagnostics report actor optimizer steps/possible steps (256 in
production), first old-KL stop location, tested old-KL mean/max, ratio clipping,
reference KL, entropy, pre-global-clip trunk/head gradient L2 norms, raw and
globally normalized advantage mean/std per mix, pre-update raw critic MSE/EV
against detached GAE returns, and old/new PopArt moments. GAE EV is explicitly
bootstrap-dependent and distinct from held-out MC EV. Behavioral observations
carry counts and denominators; greedy disagreement from reused PPO forwards is
labelled as repeated minibatch observations. Length bands use logical pre-action
lengths, including <700 and >=800, rather than scalar0. Stochastic rollout boosts
are separate from greedy proxy boosts. Calibration reports actual departure and
entropy; prep support distinguishes observed rows, complete-life rows, fitted
rows and censored rows/lives by mix/length.

A new empty ledger is shipped: a prior snapshot is optional. Historical diagnostic
fields absent from old snapshots remain historical unknowns; they are not
reconstructed by changing learning/task state. Resume at exactly update16 can run
its missing diagnostic once before the next update. A later predecessor snapshot
cannot reconstruct update16 weights and records that the pre-adoption check was
unavailable; it does not roll back or label later weights as update16.

## Frozen identities and verification evidence

- Runtime source pin: `c8ab2e6feefc63ce866e1493bfd2f7f1bf3f64dd`.
- Dispatch identity: `sandbox-a-train01-20261027-08`.
- Manifest SHA-256: `404e1e07198164cac4306690da932703d5bc973c88e8f33422360e7408e8d96f`.
- Runtime checkout: `/Users/josenunez/Projects/ml/snake-dqn-train01-08-runtime.noindex`.
- Broker checkout: `/Users/josenunez/Projects/ml/snake-dqn-train01-08-broker.noindex`.
- [Packet seal](packet-train01-08/SEAL.json) binds all 186 runtime source paths,
  the prior packet's exact bytes, current control/ledger/manifest and ruling hash.
- [Verification receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-08-diagnostics-20261027/verification.json)
  records the frozen broker HEAD, test counts, exact commands, log hashes,
  preflight, admission, dry-run, protected-file checks and limits.
- [Operator dispatch](packet-train01-08/DISPATCH.txt) is prepared only. No live
  accounting, provider allocation, CUDA kernel, timing or Inductor qualification
  follows from CPU/CUDA-semantics tests. These are API-contract fixtures.

The original regular monitoring/nomination rules remain as registered. The
`nominates: false` field means a monitoring result never itself selects or
releases a checkpoint. Old-KL mean/max summarize the tested minibatch guards;
behavioral disagreement and entropy/reference-KL summarize repeated pre-step
minibatch observations, including the minibatch that first latches the stop.
Gradient norms summarize actual actor optimizer steps. Zero-support denominators
are explicit and must not be interpreted as a measured zero rate.

## Optional continuation from packet07

No prior snapshot is needed for the default empty-ledger packet08 dispatch.
If a verified packet07 full-state snapshot does exist, use the existing manual,
outcome-blind completion process. It requires the exact prior executed manifest
and a receipt proving the registered released mechanical stop (`INCOMPLETE`,
`RPC_EXEC_POLL_TIMEOUT`, verified stop, no cleanup error). A reward curve, proxy
result or preparation-only/calibrated payload cannot justify a completion.

Create a **separate fresh completion broker checkout** at the frozen packet08
broker seal from the verification receipt; preserve both original detached
checkouts. In that new checkout, run the following with actual verified receipt
and export paths. The example uses a fresh completion identity and output root;
none of these completion files are created by sealing the default packet08.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_THREAD_LIMIT=1 \
TORCHINDUCTOR_COMPILE_THREADS=1 nice -n 15 \
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
-m research.sandbox_a_resident.training_completion \
--original /Users/josenunez/Projects/ml/snake-dqn-train01-07-broker.noindex/research/sandbox_a_resident/packet-train01-07 \
--receipt /absolute/path/to/verified-07-state/receipt.json \
--exports /absolute/path/to/verified-07-state/artifacts/training \
--successor-template /Users/josenunez/Projects/ml/snake-dqn-train01-08-broker.noindex/research/sandbox_a_resident/packet-train01-08 \
--packet /Users/josenunez/Projects/ml/snake-dqn-train01-08-completion01-broker.noindex/research/sandbox_a_resident/packet-train01-08-completion01 \
--output /Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-attempt-08-completion01/output \
--index 1 --job-id sandbox-a-train01-20261027-08-completion01
```

The helper verifies completion-last payload/sidecar/marker triples by path, size
and hash, reads metadata only on the Mac, selects the latest eligible full-state
boundary and uploads only that snapshot. It writes a new control-bound ledger
with an explicit source migration: predecessor source must be exactly
`152af938acc32da6aeea9fcb5a300f470d96c53c`, and the only control difference may be
the successor source pin. Runtime validation checks the old full control,
allocation, metadata and cached B8 identity before restoring task, optimizer and
RNG state. Valid cached B8 measurements are reused, with explicit new identity
metadata; they are never silently recomputed. Existing packet07 bytes remain
immutable. Missing/corrupt/foreign/unsafe snapshots fail closed.

Create a DISPATCH.txt for the completion packet with its fresh identity/state
paths. Commit and publish that file together with the newly created packet,
seal its broker checkout clean, then run frozen training preflight, broker
admission tests and dispatch dry-run.
The existing original-plus-two-manual-completion limit remains; index must
advance exactly once. This helper never dispatches or retries automatically.
