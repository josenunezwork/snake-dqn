# F4 round-26 implementation handoff

Worktree `/Users/josenunez/Projects/ml/snake-dqn-f3`, branch `data-f3-20261023`.
Authority: round-26 **RATIFY-WITH-CHANGES**, judgment SHA256
`0fe1a103384035839cb143a6df9193abf8269197e6d02f44bf7dceb000ab8362`.
The exact 129-line extracted registration is `F4_REGISTRATION.md`, SHA256
`66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65`.
It has zero `UNRATIFIED` tokens. The immutable registration bytes were checked
before implementation. Earlier F3 artifacts remain historical and unchanged.

The code implements the ruling's eight changes: fixed stratified life-uniform
training and exact-only pair objectives; reserve parent cap and allocation
reports; fold-zero natural archive, life-macro retention and zero-event admission;
joint create-only funding look; trace-aware three-arm gameplay, probe tripwires
and Watch6; R8/R9 one-look analyzer and synthetic boundary fixtures; staged fault,
funding and budget refusals; exact registration plus approval binding.

The new Codex-V receipt is `CODEX_V_F4_REVIEW.json`, with completion marker and
SHA sidecar. It covers each edit, the synthetic fixtures, source hashes and
verification commands. It is implementation verification, not a scientific look.
No real factory labels, reserve outcomes or gameplay outcomes were opened.
No Colab runtime was allocated, no units spent, no Mac lease/tmux state changed,
and no serving checkpoint/pin or existing hash-bound research receipt was edited.

The qualified Colab CLI subtree is
`d0d92bdb572b770284a1fea6152417ffa5f06bea`, brought in from `origin/main`.
Future packs bind that tree and current published executable source. Any prior
immutable pack remains historical. Operator dispatch stays through
`tools/ops/dispatch.sh`; an offline prepared HOLD pack is not live qualification.
Current-source same-pin family canary proof remains required before main dispatch.

Claude must write the exact `judgment.json` printed in the final handoff, with
the new recipe digest, only after accepting the Codex-V receipt. The program
requires the exact basename `F4_REGISTRATION.md` and its judged bytes. Approval
must precede every factory-label fit, reserve look or screen allocation. The
approval/logging step belongs to Claude and was not synthesized as real approval
by this implementation. Record judgment and registration hashes in CAMPAIGN_LOG.

D256/D1024 unconditional fit faults terminate the path. D4096 additionally
requires passed funding (or a published F2 non-strike disposition) and a bound
cost projection before dispatch. Scientific terminal stops produce
`NOT_RUN_BY_RULE` for later units, preserving fixed N. A GO authorizes only a
separate fresh Mac strict-gate plan; B8 s3 stays served. Serving changes still
require the owner's OK after strict pass and serving qualification.

After verification, Claude writes `judgment.json` with exactly these five fields:

```json
{
  "schema": "bl-residual-f4-judgment/v1",
  "decision": "APPROVE",
  "judgment_sha256": "0fe1a103384035839cb143a6df9193abf8269197e6d02f44bf7dceb000ab8362",
  "registration_sha256": "66388c795b3bec5bd014fa03d700994527b67763479143f7c3e5f35d1bb2fd65",
  "recipe_sha256": "49d12f486bffa455adcedbc318e88f1524f5c502894d0c1ace47daa84b9d3a5a"
}
```

Recipe SHA256 uses `protocol.canonical_sha(dataclasses.asdict(FitRecipe()))`.
The approval is prospective; the JSON shown here is documentation and grants no
execution authority until Claude creates the actual file after verification.

## Verification on the final implementation

All tests were light, single-process runs with `-p no:xdist`, CPU device and
OMP/MKL/OPENBLAS/NUMBA/VECLIB thread caps of one. The final combined selection
passed **151 tests in 18.60 seconds**:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist \
  tests/test_bl_residual_*.py tests/test_ops_qualification.py \
  tests/test_colab_cli_schema_failures.py tests/test_colab_cli_canary.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m black --check --workers 1 --line-length 100 research/bl_residual tests/test_bl_residual_*.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m isort --profile black --check-only research/bl_residual tests/test_bl_residual_*.py
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m flake8 research/bl_residual tests/test_bl_residual_*.py
bash -n research/bl_residual/colab.sh
git diff --check
```

The actual job CLI completed three synthetic fit units at
`/tmp/f4-round26-smoke-20261023-a/`. The actual gameplay CLI completed the H3
B8-as-synthetic-candidates fixture and sealed
`/tmp/f4-gameplay-final-fixture-20261023.json`, with three delivered decisions
per arm. Neither is registered scientific evidence. Six additional integration
cases mock expensive worlds and exercise real probe/Watch/admission/analysis
wiring, including 128 tiny fake paired Watch sessions, exact clone separation,
controller installation before session construction, dead-frame mass, known
fault KILL_FIT propagation, and cleanup without reclassifying infrastructure
errors. Synthetic fixtures also cover all ten R9 labels and equality cases.
No full suite, training-rate forecast, live family canary, or timing/serving
qualification is claimed.

For a later main dispatch, run `job --input-dir <main-inputs> --preflight-main`
before any allocation. The package's `make_pack` remains a synthetic HOLD
preparer and refuses to create a main manifest. Production probe/Watch modes
consume actual sealed producer files through `--heldout-receipt` and
`--go-receipt`; they construct the hash-bound consumer envelopes themselves.
Canonical reserve claims persist after failure and prevent another look via a
new output filename. Claude owns any later admitted manifest and dispatch.
