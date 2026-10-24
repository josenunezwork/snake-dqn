# A1-PROF canary-03 operator handoff

Prepared only; no allocation or dispatch. Worktree:
`/Users/josenunez/Projects/ml/snake-dqn-a1-prof`, branch
`t34-a1-prof-20261023`. Published runtime/manifest pin:
`b39604af519b59e1c1048809fdc38037ed060d36`, containing repair `44c57573`.
The later handoff commit changes documentation only; it does not change the
runtime source closure. Both source and handoff are published to main and branch.

Attribution gaps no longer abort, including 100% unmatched kernel time. Every raw
kernel contributes once to the full name/provenance inventory (count, total/mean
device seconds, stream/device, launch op/API and stack when available). Top 30
unattributed names rank summed time across streams and provenance groups. Explicit
scope correlation, native-name suggestions and synchronized exclusive event
measurements remain distinct. Ambiguous CPU correlation-ID collisions stay
unattributed. Synchronized exclusive CUDA-event intervals are authoritative for
stage splitting; uninstrumented full-tick throughput and engine bytes are unchanged.
See `A1_PROF_GAP_DIAGNOSTICS_20261023.md` for the prospective correction.

## Fresh immutable package

Root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-03`.

| Artifact | SHA256 |
| --- | --- |
| `a1-prof-canary.json` | `caa9d37ed2808d8335e19fef34933b941486f475967d800dda5b3ab58778147e` |
| `a1-prof-main-template.json` | `135220af8473c0510f405c1681f0c15ab87cfe31b7774a636b3746719c8624ed` |
| `DISPATCH.txt` | `724f2df1848bea9bba1a5aad95f3d58c8bfc05b12f3e412286be349eab2f13e6` |
| `dispatch-main-after-canary.sh` | `cea73793e48969051fe1cd26cd21ded3720756543b0c99413ebc2402e600f8d6` |
| `mac-preflight/mac-preflight.json` | `67100fe024c5e9a1c24191286027d65262aad8fa2a5bcac5aafa93d04dbb5c98` |
| `mac-preflight-analysis.json` | `50df36b115e66e76b49365dd62924efeda0bc1c810d10bd7415d20c6f59c985d` |
| `codex-v-light.json` | `df8cb634be88b2cea43a9688776d81eedfbbe9e3d551ae99b666dd2de8d86eff` |

Both manifests bind the same seven input files, fresh empty pin-bound ledgers,
qualified transport and successful current-pin Mac preflight. Canary/main outputs
and state directories and the bound main manifest are absent at handoff. The
helper validates/binds the canary receipt and immediately dispatches main; missing,
failed, foreign/wrong-pin, unverified-export or unverified-release receipts cannot
reach dispatch. It preserves pin, suffix, rate and preflight across binding.

Existing envelopes retained: A100 only, >=12 vCPU, positive rate ceiling 10 u/hour;
canary estimate 10 min, wall 30 min, cap 5 u, reserve 1 u, workload 900 s;
main estimate 30 min, wall 90 min, cap 20 u, reserve 4 u, workload 2400 s.
These are already-doubled envelopes. No old cap or historical accounting is changed.
Live account assignments/rates are refreshed by ops dispatch.

## Verification

From the worktree, with the shared venv:

```bash
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 ./venv/bin/python -m pytest -q \
  research/gpu_sim_spike/test_a1_prof.py \
  research/gpu_sim_spike/test_a1_prof_units.py \
  research/gpu_sim_spike/test_a1_prof_pack.py \
  tests/test_colab_cli_canary.py tests/test_colab_cli_schema_failures.py \
  tests/test_ops_dispatch.py
```

**198 passed, 1 target-CUDA skip in 42.47 s.** Tests cover the reported
17842/17910 and 77.4804054% gap, full missing coverage, raw inventory conservation,
full inventory versus top-N, cross-stream name aggregation, captured provenance,
CPU ID collisions in both orders, raw-data errors, event segment fences, and main
helper refusal/success paths. Black 100-column check, isort Black-profile check,
flake8 passed on the four changed Python files. `bash -n` passed both runtime
entrypoints and the emitted helper; `git diff --check` passed. AST comparison to
`3568adaa` confirmed sync, instrument, metrics, run and the nested tick unchanged.
Timing segment fences and the separate profiler pass intentionally changed.

```bash
./venv/bin/python -m black --check --line-length 100 \
  research/gpu_sim_spike/a1_prof.py research/gpu_sim_spike/a1_prof_pack.py \
  research/gpu_sim_spike/test_a1_prof.py research/gpu_sim_spike/test_a1_prof_pack.py
./venv/bin/python -m isort --check-only --profile black \
  research/gpu_sim_spike/a1_prof.py research/gpu_sim_spike/a1_prof_pack.py \
  research/gpu_sim_spike/test_a1_prof.py research/gpu_sim_spike/test_a1_prof_pack.py
./venv/bin/python -m flake8 \
  research/gpu_sim_spike/a1_prof.py research/gpu_sim_spike/a1_prof_pack.py \
  research/gpu_sim_spike/test_a1_prof.py research/gpu_sim_spike/test_a1_prof_pack.py
bash -n research/gpu_sim_spike/a1_prof.sh research/gpu_sim_spike/colab.sh
git diff --check
./venv/bin/python tools/ops/qualification.py check --repo "$PWD" \
  --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json \
  --ref b39604af519b59e1c1048809fdc38037ed060d36
```

Qualification returned transport tree `d0d92bdb572b770284a1fea6152417ffa5f06bea`.

```bash
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-03
PIN=b39604af519b59e1c1048809fdc38037ed060d36
./venv/bin/python -m research.gpu_sim_spike.a1_prof_preflight \
  --commit "$PIN" --root "$ART/mac-preflight"
./venv/bin/python -m research.gpu_sim_spike.a1_prof_pack \
  --commit "$PIN" --artifact-dir "$ART" --dispatch-suffix 03 --validate
./venv/bin/python -m research.gpu_sim_spike.a1_prof_analyze --smoke \
  --commit "$PIN" --exports "$ART/mac-preflight/runtime/output" \
  --output "$ART/mac-preflight-analysis.json"
bash -n "$ART/dispatch-main-after-canary.sh"
```

All passed. Real Mac entrypoint elapsed **20.322627958026715 s**; all three CPU E1
rosters completed with verified sidecars and complete smoke union. Baseline
seconds/tick: scripted 0.22867133363615721, frozen 0.41149998634743196, mixed
0.3508964587041798. These are portability measurements, not A100 throughput.
Independent Codex-V LIGHT review is in the artifact root's `codex-v-light.json`
with checksum and completion sidecars. CUDA behavior, native kernel identities,
production-shape overhead, and auxiliary-stream event behavior require the new
operator-run A100 canary. No live GPU qualification is claimed here.

## Exact operator commands — not executed

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-a1-prof
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-03
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" a1-prof-a100-canary-03 "$ART/a1-prof-canary.json" \
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-canary" \
  --dispatch-txt "$ART/DISPATCH.txt"
```

After same-pin canary PASS, verified exports and verified release, invoke the
prepared helper directly. It binds the exact receipt, validates the immutable
main manifest and executes `a1-prof-a100-main-03` through normal ops dispatch:

```bash
bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-03/dispatch-main-after-canary.sh
```

No tmux sessions, live jobs, released checkpoints, serving pins, legacy queue,
qualified transport, prior logs/manifests/receipts or hash-bound documents were
modified. Unrelated incoming main factory/ops/documentation changes were merged
without editing their files; they are outside this mechanical repair's light review.
