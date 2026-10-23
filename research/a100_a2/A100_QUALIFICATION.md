# A2 A100-40 transport qualification

This note records the transport boundary for A2. It does not authorize an
allocation. The registration is [PREREGISTRATION.md](PREREGISTRATION.md); the
operator alone dispatches reviewed manifests through `tools/ops/dispatch.sh`.

## Qualified broker boundary

The current qualified `research/colab_cli` tree is
`553af3f692515b2bf66173356fe4d099acc96942`, verified locally with:

```bash
python3 tools/ops/qualification.py check \
  --repo /Users/josenunez/Projects/ml/snake-dqn-a2 \
  --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
```

No A2 change may modify a tracked Python or shell file under
`research/colab_cli`. The supervisor already accepts `A100` as an explicit
requested and allowed GPU, sends `colab new --gpu A100`, and checks the runtime
reported GPU kind against `requirements.allowed_gpus` before the job starts.
See `research/colab_cli/supervisor.py` in `validate()` and the ready RPC.

The qualified source acceptance is CPU plus L4. It does **not** itself prove
that an A100 family entrypoint has exercised this A2 workload. A2 therefore
requires the smaller same-pin family canary below. Because the broker source
tree is unchanged, it does not require a new CPU/L4 final-only acceptance.
Changing any executable transport byte changes that conclusion: Claude must
run `tools/ops/requalify.sh plan`, independently review its generated plan, run
`tools/ops/requalify.sh execute` with the review token, and only then run the
A2 same-pin family canary again.

## A100-40 admission

The manifest must request and allow only `A100`. The generic broker's `A100`
kind is deliberately not a memory-size assertion. Before bundle materialization
or input access, A2's own entrypoint must fail closed unless it observes an
NVIDIA A100 with the registered 40 GiB device-memory capacity. It must publish
the detected name and byte capacity into the canary/main hardware evidence.
An A100-80, unknown GPU, no CUDA GPU, or a failed query is a refusal, never a
substitution. This guard belongs in A2 package code so the already-qualified
generic transport inventory stays unchanged.

## Same-pin family canary

`research.colab_cli.canary.FAMILIES` is only a convenience allowlist for its
manifest generator. `canary.check()` admits any package whose probe manifest is
correctly formed. Do not add `a100_a2` to that allowlist: doing so would alter
the qualified transport tree without adding a runtime safety property.

The generated A2 canary manifest must retain the exact production entrypoint,
source paths, inputs, `requirements.gpu: "A100"`, and
`requirements.allowed_gpus: ["A100"]`. It changes only the registered
synthetic workload controls and includes all of the following:

```json
{
  "canary": {
    "schema": "colab-family-canary/v1",
    "mode": "probe",
    "family": "research/a100_a2/colab.sh"
  },
  "resume": {
    "unit_kind": "final-fold-fit",
    "ledger_path": "inputs/canary-ledger.json",
    "skip_verified": true,
    "max_outcome_blind_retries": 2
  },
  "env": {
    "SNAKE_COLAB_CANARY": "1",
    "SNAKE_COLAB_CANARY_PACKAGE": "research/a100_a2"
  }
}
```

It is capped at 18 minutes and 3 units (within the plan maximum of 30 minutes), within the registered 288-minute /
48-unit canary-plus-main series envelope. Its terminal receipt must show the
same pinned commit and the qualified transport identity, verified immutable
exports, a CUDA six-action decision attributed to `research/a100_a2`, the
recorded A100-40 hardware observation, and backend session disappearance.
A kernel RPC stall is `INCOMPLETE`, not a scientific result; only the
pre-registered outcome-blind mechanical path may prepare a fresh replacement
attempt.

After Codex-V has accepted the generated manifest and receipt plan, Claude uses
the package's concrete generated paths with this sequence; do not invoke
`colab` directly:

```bash
tools/ops/dispatch.sh a100-a2-canary /absolute/generated/a100-a2-canary.json \
  --broker /Users/josenunez/Projects/ml/snake-dqn-a2 \
  --cwd /Users/josenunez/Projects/ml/snake-dqn-a2 \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/a100-a2-canary \
  --dispatch-txt /absolute/generated/A2_DISPATCH.txt
```

Claude must inspect the terminal receipt and independently verify release
before the main manifest is dispatched through the analogous `a100-a2-main`
command. A successful launcher exit, a screen observation, or a prepared
manifest is not canary proof.
