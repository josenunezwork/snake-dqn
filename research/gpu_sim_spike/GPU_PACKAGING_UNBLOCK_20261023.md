# GPU sprint packaging unblock

Prepared only; no Colab allocation, dispatch, stop, or engine semantic change.
Mechanical repair under the owner standing rule and round-28 JUDGMENT.

Sparse/profiler runtime pin: `42c84584d5edef40009ad7278cb912b7f8bd4a9a`,
published and contained in origin/main. Frozen checkout:
`/Users/josenunez/Projects/ml/snake-dqn-gpu-packaging-frozen-02`.
Its parent is original `313758255ee5b9f1422a0202c14a80cc32554857`; only seven
packaging/reporting/test files change. Engine, native kernel and policy bytes
remain those of the original package. Qualified transport tree:
`d0d92bdb572b770284a1fea6152417ffa5f06bea`.

Chrome trace export is optional and off by default (`A1_PROF_EXPORT_CHROME_TRACES=0`).
The separate profiler pass still emits complete per-kernel names/counts/times,
top-N summaries and native-launch diagnostics in JSON. Export ceiling is 64 MiB.
The Mac analyzer accepts trace-free units and still verifies any supplied trace.
No duplicate profiler GPU run is needed to admit sparse work.

Sparse helpers no longer require a successful v4 profiler receipt. They may bind
prior-pin recovered unit exports as optional diagnostic provenance, or omit them.
The prepared packet binds five complete recovered JSON units (15 files including
SHA/complete sidecars), 3,157,803 payload bytes, from the old v4 canary. These are
not sparse resume units or successful native-canary evidence. Historical receipts,
trace failures and partial outputs remain unchanged. Sparse main still requires
its own genuine same-pin canary PASS, verified exports and verified release.

Sparse artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-02/sparse-cap160`.
Fresh job IDs `sparse-bench-a100-{canary,main}-42c84584`; separate fresh
`state-canary`, `state-main`, `canary-output`, `main-output` under that root.
Canary/main limits remain 24/192 minutes and 4/32 units, including reserves;
series envelope remains 216 minutes/36 units. A100, >=12 vCPU, <=10 u/hour.
The same registered correctness and speed decisions apply.

V8 artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-sprint-20261023-03`.
Runtime remains exactly `85180521` (full SHA in the manifest), using frozen broker
`/Users/josenunez/Projects/ml/snake-dqn-gpu-v8-frozen-02`.
Fresh dispatch/job names end `-85180521-n03`; state/output directories end `-n03`.
Only package identity and paths change. Main is a template until its helper binds
and validates the successful fresh canary receipt into `v8-main.json`.
No large traces are produced by this lane.

Verification: source-slice Codex-V LIGHT ACCEPT; 183 packaging/transport regressions
passed in 60.48 s; trace-focused suite 35 passed/1 CUDA-only skip. Black (100), isort,
flake8 and diff whitespace passed. Real frozen-pin Mac Bash entrypoints completed:
sparse all tiny rosters 8.414107125 s; profiler all tiny rosters 22.201563083 s,
zero trace files and 43,538 total export bytes. The real Mac analyzer accepted the
trace-free smoke exports. These are CPU portability checks, not GPU timing proof.
Final independent artifact review and hashes are in the sibling
`gpu-engine-sprint-20261023-02/CODEX_V_LIGHT.json` and `VERIFICATION.json`.

Exact operator order (not executed):

```bash
export PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH"
SPARSE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-02/sparse-cap160
V8=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-sprint-20261023-03

# (a) sparse-cap160 canary
nice -n 15 bash "$SPARSE/dispatch-canary.sh"
# (b) V8 fresh-namespace canary
nice -n 15 bash "$V8/dispatch-canary.sh"

# Each following main only after its own canary PASS, verified exports/release.
nice -n 15 bash "$SPARSE/dispatch-main-after-canary.sh"
nice -n 15 bash "$V8/dispatch-main-after-canary.sh"
```

Helpers use qualified `tools/ops/dispatch.sh`; live assignment/rate evidence is
refreshed by ops at actual dispatch. Prepared names do not reserve or launch work.
All existing series/global caps and mechanical retry limits remain binding.
No tmux/live jobs, released checkpoints, serving pins or immutable past evidence
were edited. Main follows current upstream; dispatch uses the frozen pins above.
