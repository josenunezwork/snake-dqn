# P successor preparation verification

Prepared only. No Colab allocation, scientific look, successor run, serving edit
or release. Existing P 09e is INVALID/INCOMPLETE; its 21 imported sessions and the
entire predecessor bank remain closed. No old outcome payload was opened.

Runtime source pin: `8fb91f6de19ec6055fb1e7f09ea8508be32e371b`.
Branch: `astra-argon-falsifier-p-20261028`; branch-only publication.
The final documentation commit does not change this runtime closure.

## Independent review

Argon (`agy --model gemini-4-argon-high --mode plan --continue --print`, bounded
900-second print / 960-second outer guard) statically reviewed source at the pin
above in its own detached worktree
`/Users/josenunez/Projects/ml/snake-dqn-p-successor-review.noindex`.
**ACCEPT; no unresolved blocking issues; process EXIT=0.** It invoked no tools,
ran no commands, changed no files, and accessed no P outcome data. Native family
canary admission, pricing and unchanged qualified transport were accepted in
both review rounds. No final scientific registration was ratified by this review.

Evidence base:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-p-successor/`.

- Final review: `review-r3/answer.md`, SHA256
  `a03bc039ec0f5ea39441034d69f33b666c9e034472eef99fcc85e66e7bfe36f6`.
- Final bindings and retained evidence: `verification-final.json`, SHA256
  `4c782bbb955d613e235e13d40a7c667d1a9929c0642893b790edbe177a5d55d1`.
- Earlier review and sealed packet remain intact. `packet-v1.SUPERSEDED.md`
  records their disposition; only `packet-v2` is the current preparation.
- Review worktree remained clean at the reviewed source commit afterward.

Review fixes: separate warmup from work; measure real active concurrency and
homogeneous search pressure; count actual H300 calls; require local engineering
root/search support before sealing; replace executor internals with fixed owned
spawn processes and tested signal reaping; add mid-session original-engine tests.
The frozen P zero-baseline statistic was retained and clarified rather than
changed. Nonblocking notes: startup error ordering can report a generic worker
exit first; the exclusion audit's prose coverage label omits profile-v1 even
though its four seeds are explicitly excluded by executable code.

## Verification run by Astra

The interpreter is `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`.
All numerical commands used CPU and all seven single-thread environment caps.
The tool environment already runs at nice 15, so final local jobs used `nice -n 0`
to preserve 15; workers lower themselves to 19. An earlier additive `nice -n 15`
smoke correctly refused an already-maximal parent priority; its log is retained.

- `python -m pytest -q tests/test_colab_cli_supervisor.py
  tests/test_colab_cli_canary.py tests/test_colab_cli_source.py
  tests/test_colab_cli_runtime.py research/watch_lead_p_successor/test_profile.py`:
  **215 passed in 622.16 s** before the review revisions. Qualified CLI bytes
  were unchanged throughout. Log: `tests-v1.log`.
- Final `python -m pytest -q research/watch_lead_p_successor/test_profile.py
  tools/sandbox/tests/test_source_delivery.py`: **33 passed in 4.65 s**.
  Seventeen probe tests and sixteen existing sparse-source tests. Log:
  `tests-final-revision2.log`.
- Real final `python -m research.watch_lead_p_successor.profile --smoke --bank
  .../dev-inputs-v2/profile-bank.json --output .../smoke-output-v5`: **PASS**.
  Three arms × two ticks at each of w1/w2, original constructor vs exact restore,
  matching opaque trajectory/initial-state hashes. No maturity or rate claim.
- Real `--check-roots` under `research/perf_sim_20261007/with_slots.py --slots 1
  --timeout 20 -- /usr/bin/caffeinate -is ...`, with AC/lid checks and the dev CPU
  lock: **PASS**, `roots-preflight-v4/profile.json`. All four fixed v2 seeds
  naturally reach L600; P0 and P48 each actually call H300 search twice per seed.
  These are local plumbing/workload-support checks, never Colab timing rates.
  Internal root caches are outside Git and never uploaded.
- `python -m black --check --line-length 100 research/watch_lead_p_successor
  tools/ops/dispatch.py`, `python -m isort --check-only --profile black` on the
  same paths, `python -m flake8` on those paths, `bash -n` on both new shell
  scripts, `bash tools/ops/tests/test_dispatch.sh`, `git diff --check`: **PASS**.

The first v1 engineering allocation had an unsupported natural root, detected
locally before dispatch. The whole allocation was retired; the fixed v2 key was
created before its checks. Both engineering allocations are barred from successor
inference. This is not a retry, replacement or look at predecessor P data.

## Sealed current packet and operational limits

Packet: `.../astra-p-successor/packet-v2/`.

- `seal.json`: `74fd2d6fa4da80b4f63e3146ed2b21d0972be987ba3d13e3e60fb1896a33e0a6`.
- `l4-manifest.json`: `a55701d4e704ed512e847e9344789546235f7ee1ffc67a4d463795bbe5cd6adb`.
- `a100-manifest.json`: `8708a58fdc301930bf5bcf659745487c214b59f61b71b08ade7cd528d4f767d3`.
- Appending `--dry-run` to the exact single-line `canary-command.sh` and
  `compare-command.sh` passed the real ops wrapper: **PASS_DRY_RUN,
  allocations=0, live_accounting=NOT_CHECKED** for each.
  Logs: `dry-run-l4-v2.txt`, `dry-run-a100-v2.txt`.
- Runtime source closure and packet seal were independently recomputed locally.
  Both output/state paths remain absent; no name was reserved or dispatched.
- Source archive: **1,424,953 B compressed / 6,708,074 B unpacked, 315 files**;
  three private uploads total **13,468,539 B**. Log: `source-size-v2.json`.
  Sizes satisfy the offline source cap; live staging speed remains unmeasured.
- Whole-session base/effective guards: **15/30 min**. L4 1.54 u/h, base/effective
  caps **0.39/0.78**, reserves **0.078/0.156**; 30-min exposure **0.77 units**.
  A100 5.30 u/h, caps **1.33/2.66**, reserves **0.266/0.532**; exposure **2.65**.
  Rates are prepriced ceilings checked by live calibration. Work guard is
  base/effective **450/900 s**. No automatic allocation or completion retry.
- CLI 0.7.4 help/version are sealed. No vCPU selector; `--high-mem` is ignored
  for L4 and is not forwarded by the qualified supervisor. Optional A100 compares
  its actually assigned CPU host; it is not a guaranteed higher-vCPU allocation.
- Native qualification subtree remains
  `5833e54e9c5396121cbbf99a5cd4e52006e564fd`. The host adapter preserves native
  canary admission and uses the existing bounded background sparse-source route.

Operator: inspect `packet-v2/canary-command.sh` for the exact L4 dispatch line;
after a passing receipt, verified exports and stop/disappearance, optionally use
`packet-v2/compare-command.sh`. Live accounting, balance and the account-wide
three-assignment limit are rechecked at actual admission; no occupied slot is
assumed or stopped. This preparation does not execute either command.

Full-session long-body/late-search uplift, residual contention, host choice,
shards and successor envelopes remain unfilled. No executable successor manifest
exists. `DRAFT_REGISTRATION.md` awaits target measurements and prospective review.
Original P code, source-bound packets/receipts, released checkpoints, serving
pins, pre-existing `saved_snakes`, artifact `prompt.txt`/`retry.out`, all tmux
sessions and other jobs were preserved.
