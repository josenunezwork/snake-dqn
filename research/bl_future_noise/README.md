# Round-31 A0 preparation

**STOP_AT_A0: no separately cleared archive was supplied or identified in the
permitted repository metadata.** This is a preparation result, not a scientific
test of leakage or a statement that historical physical support is zero.
See [INTENT.md](INTENT.md) for the adopted downstream rules and caps.

Implemented: a metadata-only immutable census gate. A1's adapter, A2's producer,
reducer and Colab manifests are deliberately not generated across the A0 stop.
No Colab/Mac rollout job is ready. Preparing a synthetic scientific bank or a
dispatchable manifest would evade the ruling's order.

## Run the actual A0 entrypoint

Only run lightweight work in an admitted factory gap. The full companion route
remains ACTIVATION HOLD; the subsequently published nice-interim serial-test route
in `tools/ops/agent_nice.sh` acquires the shared test slot and Mac lease and refuses
factory/timing overlap. Use its reviewed manifest interface for focused tests,
with inherited nice 19. Keep inherited nice ≥15 for read-only A0 work. Do not
renice upward. All numerical thread counts stay one. The A0 entrypoint preflight
held the existing Mac lease; it did not activate a coexistence route.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-r31
env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.bl_future_noise.census \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-31/a0-check-02.json
```

With no `--clearance`, expected exit **2**, `BLOCKED_MISSING_CLEARANCE` and zero
admitted parents. Every output path must be fresh; never remove/reuse an existing
receipt. This takes no gameplay steps and uses **0 units / 0 allocated wall**.
The development preflight uses the same entrypoint on empty input, not a fake
claim that a real reconstruction fixture has passed.

## Supply an archive only after explicit metadata clearance

Do not search scientific artifact directories for candidates. The historical
E3/T3 notes document unsealed development material but not round-31 clearance;
they also report missing original complete trajectories/state hashes and later
recovery artifacts. No original bank was opened by this implementation.

Claude's next step is to identify and record a separately cleared source and
have Codex-V certify exclusion coverage and reconstructability. Clearance
documents cannot substitute for true inputs or actual original replay identity.
The tool's success label is always pending independent review.

Invoke the same command with `--clearance /absolute/cleared-r31-metadata.json`
and a fresh `--out` path. The metadata contract is:

- Clearance: `schema: bl-r31-clearance/v1`, ruling SHA256 from INTENT,
  `source_kind: historical-development`, `already_unsealed: true`,
  `current_factory_output: false`, nonempty `cleared_by`, `clearance_basis`,
  absolute `archive_root`, and member bindings `census`, `ancestry_audit`.
- Every member binding: archive-relative `path`, integer `size`, full `sha256`.
  No traversal or alias escape. Only these named metadata members are read.
- Census: `schema: bl-r31-source-census/v1`, `source_namespace`, full 40-character
  `source_commit`, full checkpoint digest map with `b8` equal to served B8 s3,
  `parents`. It is the entire eligible source census, not 64 cherry-picked rows.
- Parent: unique `parent_id`, complete `ancestry_ids` including itself, nonempty
  `stream`, ordered `roster` identity, and all eligible `roots`.
- Root: globally unique `root_id`, `prepared: true`, `decisive_doom: true`,
  `bare_action`, `a_star`, distinct integer lists `legal`, `served_selectable`,
  six-key `original_outcomes`, `reconstruction_receipt` member binding.
- Each outcome: `status` EXACT/UNKNOWN/NOT_EVALUATED. EXACT needs `horizon: 300`,
  boolean `survived`, finite nonnegative full-horizon `mass_sum`.
- Reconstruction evidence: `schema: bl-r31-reconstruction-evidence/v1`,
  exact parent/root/source/checkpoint/outcome bindings, `original_stream_identity:
  true`, `prepared_once: true`, `root_state_sha256`, nonempty `inputs` bindings.
  These must be existing evidence, not fabricated assertion flags. This A0 tool
  checks their metadata bindings; it does not open the referenced raw payloads,
  rerun reconstruction, or certify original replay on Colab CPU.
- Ancestry audit: `schema: bl-r31-ancestry-audit/v1`, `census_sha256`, `protected`
  with every category in `census.PROTECTED_CLASSES`. Each category has nonempty
  `inventory_basis`, `inventory_sha256`, and complete `ancestry_ids`, `namespaces`
  lists. Empty lists require independent confirmation that coverage is complete.

Receipts freeze the whole supplied census/audit/clearance and selected allocation.
They never print outcomes to the console. Insufficient support, invalid identity
or missing clearance stops A0. Passing metadata produces only
`CENSUS_PREPARED_PENDING_CODEX_V`, with `dispatch_allowed: false`.

## After A0 (future preparation, not a dispatch command)

Only after independent A0 clearance, implement A1 and current-pin CPU acceptance;
the three excluded real timing fixtures determine whether A2 fits the **8 h /
.64 u forecast**. The already-doubled all-attempt envelope is **19.2 h / 1.536 u**,
including **3.2 h / .256 u reserve** and ≤30 allocated minutes for the canary.
The ruling does not double that envelope again. CPU rate must be positive and
≤.08 u/h; only the spare third account assignment may be used. No L4/A100 fallback.

Claude logs ruling SHA, owner routing, bank clearance, stage and reservations in
CAMPAIGN_LOG. Sol does not edit that Claude-owned record. A1/A2 need reviewed,
published pins, Mac tiny preflight, CPU-family qualification, disjoint native
canary, empty verified ledger, immutable exports and exact Codex-V acceptance
before any `tools/ops/dispatch.sh` command exists. No funded job was dispatched.
