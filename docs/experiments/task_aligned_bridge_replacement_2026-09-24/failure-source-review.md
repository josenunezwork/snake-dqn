# Replacement capture failure: bounded source-only review

The replacement failed at `capture.py:580`, which compares
`json.loads(input_bytes) == inputs` after `freeze_inputs` saved `freeze/inputs.json`.
A tuple-versus-list mismatch is a sufficient deterministic explanation in the
frozen source; this is a serialization contract failure, not a learning result.

Source checkout inspected:
`/Users/josenunez/.codex/worktrees/task-aligned-serving-replacement/snake-dqn`.
Git HEAD was read as `b22d59c537a1d0a30c8cc9b2c503aa8d8d7f86b5`.

Evidence chain:

1. `research/task_aligned_challenger_20260924/public_artifacts.py:243-246`
   places `RASTER31V3_CONTRACT.semantic_dict()` directly into
   `contract['observation']['descriptor']`.
2. `src/model/obs_spec.py:137-140` constructs the canonical v3 contract with
   `RASTER31V2_SHAPES.tactical_shape`. That property returns the tuple
   `(9, 31, 31)` at `obs_spec.py:88-90`. Other shape/channel/origin fields are
   tuples too.
3. `src/core/runtime_contract.py:116-125` returns those original tuple-valued
   fields in `semantic_dict()` without converting them to JSON lists.
4. `public_artifacts.py:287-297` puts this contract in the returned result,
   writes the result, and returns the original Python object. Its `_write`
   helper (`public_artifacts.py:31-40`) serializes via `json.dumps` and writes
   bytes; it does not canonicalize or replace the caller's object.
5. JSON serializes a tuple as an array and `json.loads` reconstructs an array as
   a list. Therefore the saved-and-parsed leaf
   `serving_contract.observation.descriptor.tactical_shape` is `[9, 31, 31]`,
   while the returned in-memory leaf is `(9, 31, 31)`. These compare unequal,
   so the enclosing dictionary comparison at `capture.py:580` is false.

This proves a sufficient structural mismatch conditional on the inspected frozen
source being the executed source. It does not require reconstructing the actual
runtime object. Without that object, this review cannot enumerate every unequal
leaf or establish that this was the only mismatch. No claim is made about
checkpoint validity beyond what the earlier successful stage statements imply.
The failure occurs after the explicit input-freeze physical-count checks and
before publication of the top-level `output/inputs.json` and successful capture
report; the saved `output/freeze/inputs.json` is an attempted-stage artifact,
not a completed qualification result.

The root reports natural child/controller exit 1 and saved physical totals of
16 checkpoint opens, 42 reads, 86,308,029 bytes and one metadata deserialization,
with zero models, tests, games, forwards or optimizer updates. These runtime
facts are attributed to the root's closeout, not independently re-audited here.

Review scope: read only the named source and Git revision metadata. No saved
evaluation/training data, checkpoint payload, raw game, log or runtime object
was opened. No project module was imported, and no test, fixture, preflight,
model, simulator, audit or retry ran. No repository file was changed. This note
does not propose or authorize an automatic retry; the failed attempt remains
failed and preserved.
