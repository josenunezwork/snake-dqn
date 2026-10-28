# Sandbox v2 dispatch wiring and prospective combined policy

Status: **PREPARATORY IMPLEMENTATION ONLY. Live co-running panel admission and
the 600-second qualification remain unavailable.** This sibling document does
not change the accepted amendments, ruling, historical aborts, timing bars,
scientific manifests, Colab controls, or release authority.

The combined policy ID is `classified-desktop-supervised-v2`, host schema
`served-tick-host/v5`. Its components retain their identities and rules:
`classified-desktop-v1` and `classified-supervised-v2`. The offline composition
replays desktop identity/history, supervision identity/lifecycles, exact CPU
accounting and both ordinary >=20% paths. Desktop apps remain in the same shared
background pool with the strictly >150% for strictly >60-second ceiling.
Supervision retains its separate 50%, 1/2/3 CPU-second engineering ceilings.
No new aggregate ceiling is invented. The sensitivity report requires combined
background plus supervision <=20%, complete coverage, and operation-free windows
expanded one second each way; every tick remains in authoritative timing.

The original evidence is not rewritten. Derived validation views keep each
component's accepted semantics. Source-bound reduction checks both intents,
registries, rulings, source closure and the independently replayed summary before
the live gate. An `OFFLINE_VALID` predicate is never a panel or qualification PASS.

## Explicit packet opt-in

For a **new, unbound future sandbox packet**, include this top-level field in
its `dispatch-manifest.json` before generating the ten-line sandbox manifest's
`outputs.broker_manifest.sha256`:

```json
"sandbox_supervision": "collector-v2"
```

The ordinary `tools/ops/dispatch.sh` entrypoint delegates to `dispatch.py`, which
selects `tools/sandbox/launch_v2.sh` only with this exact opt-in and existing
`sandbox_admission`. Without it both the old sandbox launcher and ordinary
qualified launcher keep their previous selection. Unknown selectors and foreign
brokers refuse. Validation creates no collector. Run derives a create-only
`<output_dir>.supervision` sibling and passes it through a host environment
variable, never the VM's `env`. Allocation retries derive distinct journal paths.
Existing/dangling journal paths and competing launch adapters refuse before launch.
The run entrypoint independently crosschecks the packet-derived path.

The packet and ten-line manifest must still pass existing committed preflight,
source publication, resume, input, account and timeout rules. Do not modify any
previously bound packet or add the selector after admission. A local implementation
commit is not yet a published dispatch pin. Do not run the following dispatch
commands until the operator has published/reviewed that source and committed a
fresh valid opted-in packet. Nothing in this handoff dispatches a job.

```bash
ROOT=/Users/josenunez/Projects/ml/snake-dqn-supv2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
# Set these to the actual NEW committed packet and unused job identity.
PACKET=/absolute/path/to/new/dispatch-manifest.json
JOB=new-v2-sandbox-job
OPS_DIR=/Users/josenunez/Projects/ml/snake-dqn-artifacts/supv2-ops
export OPS_DIR SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1
cd "$ROOT"
nice -n 15 "$ROOT/tools/ops/dispatch.sh" "$JOB" "$PACKET" --broker "$ROOT" --cwd "$ROOT" --dry-run
nice -n 15 "$ROOT/tools/ops/dispatch.sh" "$JOB" "$PACKET" --broker "$ROOT" --cwd "$ROOT"
```

## Live dispatch declaration

`supervision_dispatch` reads the immutable startup context from the output's
collector sibling, checks its actual root identity and loaded-route declaration,
crosschecks the live lane assignment/session/generation, and seals a registry.
It does not discover roots by name, inspect credentials/environments, contact
Colab, attach instrumentation, or change a job. It rechecks the assignment and
manifest after source capture. A restart, terminal root, altered manifest or
source mismatch refuses. The registry remains a **declaration**, with no live
accounting admission. A metadata flag cannot change that.

Use the **live refreshed manifest** recorded in the dispatch accounting JSON's
`manifest` field, not the original packet. An allocation retry has its own live
manifest, root, state and journal; capture that attempt's declaration separately.

```bash
LIVE_MANIFEST=/absolute/path/from/dispatch-accounting/manifest-field.json
EVIDENCE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-supv2-new
DESKTOP_REGISTRY=/absolute/path/to/accepted/sealed-desktop-registry.json
nice -n 15 "$PY" -m research.bl_served_search.supervision_dispatch \
  --dispatch-manifest "$LIVE_MANIFEST" --output "$EVIDENCE/supervisor-registry.json"
```

## Ruling section 6 item 5: required qualification procedure

The operator interface below fixes the requested host-only duration at 600 seconds
and binds the intended live dispatch. **It currently refuses before observation or
output creation. It is not an implemented 10-minute capture harness.** `--check`
reports `NOT_ADMISSIBLE`, exit 2, without publishing a qualification. Without
`--check` it raises the retained live-observer blocker. Supplying a capability
flag, synthetic transcript or qualification filename cannot open the panel.

```bash
nice -n 15 "$PY" -m research.bl_served_search.supervision_qualification \
  --dispatch-manifest "$LIVE_MANIFEST" --desktop-app-registry "$DESKTOP_REGISTRY" \
  --host-only-seconds 600 --output "$EVIDENCE/qualification" --check

# Prospective execution interface; also refuses with the current collector.
nice -n 15 "$PY" -m research.bl_served_search.supervision_qualification \
  --dispatch-manifest "$LIVE_MANIFEST" --desktop-app-registry "$DESKTOP_REGISTRY" \
  --host-only-seconds 600 --output "$EVIDENCE/qualification"
```

Before that interface can execute qualification, a separate implementation must:

1. Prospectively capture all helpers, including CLI grandchildren and reaping
   inclusions, plus exact native counters and their resolution/timebase provenance.
   Direct wait4 timeval totals have 1,000-ns resolution and cannot be relabelled
   the accepted exact counter stream. Missing descendants remain incomplete.
2. Produce continuous independent root/helper observations, root-phase coverage,
   ongoing CPU samples, operation payload coverage, and a verified live journal
   prefix. The accepted reader currently requires the terminal seal; it cannot
   be used to read an active job by sealing its journal early.
3. Bridge those observations to both accepted reducers without synthetic
   completeness flags. Enforce online ceilings with no resets and preserve remote
   cleanup/safety if evidence becomes INVALID. Ordinary staging, exports and
   unrecognized work remain ordinary host work.
4. Freeze a fresh source-bound intent and observe 600 seconds of the actual v2
   control path, including naturally overlapping heartbeat/foreground calls and
   final accounting within qualified call deadlines. Export immutable host and
   qualification evidence and independently reduce it. Do not induce live faults;
   retries/timeouts belong in offline fixtures. This establishes operational
   envelope/observer overhead only, not causal timing neutrality.

Only a successful, independently reviewed, source-bound qualification of that
complete observer could admit a fresh full panel. No live qualification was run.

## Combined panel interface (currently blocked)

This interface preserves the full panel manifest/order/seeds/exposure/parity and
bars. It refuses at `read_registry(..., live=True)` before output creation,
lease acquisition or product imports. The reducer also refuses synthetic-only
combined evidence. These are target commands, not ready execution authorization.

```bash
"$ROOT/research/bl_served_search/run_tick_probe.sh" \
  --load-policy classified-desktop-supervised-v2 --classified-host-load \
  --desktop-app-registry "$DESKTOP_REGISTRY" \
  --remote-supervision-registry "$EVIDENCE/supervisor-registry.json" \
  --qualification-receipt "$EVIDENCE/qualification/report.json" \
  --output "$EVIDENCE/panel"
```

## Independent review disposition

Argon (`gemini-4-argon-high`, source-only plan review) rejected **live readiness**:
the missing observer/bridge cannot be supplied by wiring or metadata. Its source
closure and intent-binding findings were addressed in this preparatory package.
Its requested combined aggregate ceiling was not adopted: the ruling specifies
separate background/supervision ceilings and combined <=20% only for sensitivity.
Removing authenticated supervision from a desktop validation view is intentional;
that CPU remains in the exact supervision ledger and in the combined report.
`source_delivery.py` has no host subprocess dependency to replace; its Popen text
is remote code. Both accepted policy implementations and qualified transport
source remain authoritative; no live gate was weakened.
