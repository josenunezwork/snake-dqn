# Round-09 Z-2 pre-data implementation

Implementation only, 2026-10-21. Authority: the owner's task instruction and
`snake-dqn-artifacts/big-lever/round-09/JUDGMENT.md` §c4, Z-2 and Z-3.
No paced runs, timing measurements, Colab, bank allocation, serving changes or
release are authorized by this file. No probe authorization is issued here.
The earlier `PREFLIGHT.md` and its hash-bound exports remain historical evidence
for their original source closure; they do not verify the changed implementation.

Requests now carry two required integer-nanosecond fields:

- `publish_cutoff_ns`: target tick minus two ticks; coordinator visibility must
  be strictly earlier. Worker completion alone does not establish visibility.
- `consume_cutoff_ns`: target tick plus one tick; validation and the final
  post-recorder consumption commit must be strictly earlier.

`PacedController._matches` is unchanged: the session, life, exact target frame,
checkpoint, shadow epoch and root digest must match. Equality at either cutoff
falls back to bare. EDF uses publication cutoff, target frame, then request ID.
The independent reducer implements the same dispositions without importing the
controller. Single-cutoff journals are not reusable as new-source proof.

`adapter.py` constructs these deadlines from an absolute integer tick schedule,
selects eligible immutable reachable roots on the five-decision cadence and
binds the complete request and input payload. Candidate-specific root production
and planner math remain injected dependencies requiring separate qualification.

The four fake-clock cases are on-time acceptance, publication-late rejection,
consumption-late rejection and wrong-frame rejection. Synthetic fixture journals
retain each decision and independent scheduler replay. L3 receipt admission
requires `clock_seam` with all four correct dispositions and PASS replays, bound
by `source_binding_sha256` to the receipt's complete `source_hashes` map. Missing,
wrong or stale proof blocks L3 prerequisites even if the other oracles pass.
A passing synthetic clock proof grants no paced or semantic qualification.

Probe and serving authority are separate. `CandidateIdentity.require_authorized`
and `ProcessTransport` default to `mode="serving", device="cpu"`. Serving admits
W1/K1 unless `owner_multi_worker=True` represents the owner's explicit serving
clause; a PROBE_ONLY file never substitutes for that clause. MPS is probe-only.

For L1′ measurements use `mode="probe"`. CPU/W1/K1 remains the baseline.
Expanded W/K or MPS requires a regular, non-symlink authorization file passed as
`probe_authorization_file`. Its exact JSON fields are:

```json
{
  "schema": "bl-paced-identity-probe-authorization/v1",
  "authorized": true,
  "scope": "L1_PRIME",
  "mode": "PROBE_ONLY",
  "allowed_workers": 4,
  "allowed_max_pending": 8,
  "allowed_devices": ["cpu", "mps"]
}
```

This is a schema example, **not an authorization**. Actual files must be
canonical JSON (`canonical_bytes(value)`: sorted keys, compact separators,
trailing newline). The requested cell must fit the explicitly recorded limits
(1≤W≤4, W≤K≤8) and device list. Malformed/missing/wrong-scope authorization
fails before worker allocation. The serving flag cannot authorize expanded
probes. Without the file, expanded/MPS cells remain NOT_MEASURED.

The transport journals mode, device and authorization path/SHA before process
allocation. Spawned workers keep one math thread and receive the selected device
through `SNAKE_DQN_DEVICE`; an MPS worker checks availability before computation.
The injected planner worker must honor that device. Test coverage uses mocked
MPS availability and process allocation; real MPS transport and L1′ performance
remain unmeasured until separately authorized execution.

## Targeted verification

The final package test selection passed **72 tests in 2.76 seconds** on the Mac
with `SNAKE_DQN_DEVICE=cpu` and the primary checkout's venv interpreter:

```bash
SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  tests/test_bl_paced_identity_adapter.py tests/test_bl_paced_identity_audit.py \
  tests/test_bl_paced_identity_authorization.py tests/test_bl_paced_identity_clock_seam.py \
  tests/test_bl_paced_identity_controller.py tests/test_bl_paced_identity_journal.py \
  tests/test_bl_paced_identity_receipt.py tests/test_bl_paced_identity_replay.py \
  tests/test_bl_paced_identity_semantic.py
```

Black (`--check --line-length 100`), isort (`--profile black --check-only`),
flake8 and `git diff --check` passed on the package and changed tests.
Independent read-only review covered clock/reducer agreement, source-bound L3
proof, probe/serving separation and adapter identity/selection. The adapter's
root tie order is target frame, root digest, horizon checkpoint, then canonical
planner-input SHA; reversing provider order preserves the selected request.
Sealed synthetic controller/reducer integration passes at the target tick after
the publication cutoff. No full preflight entrypoint or paced run was executed.
