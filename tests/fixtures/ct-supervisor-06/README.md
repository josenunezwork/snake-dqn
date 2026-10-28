# Raw qualified producer corpus, packet 06

Captured before admission edits by `tests/ct_supervisor_goldens.py`, using the
existing subprocess fake CLI/OAuth provider and the unmodified qualified broker.
Only offline provider state, virtual time, observer fault points and runtime
local-release exemptions are injected. No real provider/account is accessed.

Each capture contains the actual supervisor stdout/stderr, complete output tree
(including staging, polls, receipt and exports), fake-provider call/state records,
and, for child exits, real `runtime.supervise` output. `capture.json` describes the
observed classification. Do not rewrite these files when parser expectations change.

This first corpus exposed admission incompatibilities before fixes. The sibling
`ct-supervisor-06-exports` corpus adds the actual nonterminal C/T export inventory
to runtime.read_state and the locked-account priced-unit guard / busy RPC paths.
`ct-supervisor-06-ct-bound` uses actual C/T canary manifests and canonical inputs
with tiny C/T controls. Earlier failed tiny-canary driver attempts are preserved
under the task artifacts, not substituted for successful evidence.

Example reproduction (new output and scratch paths required):

```
<CPU-wrapper> -m tests.ct_supervisor_goldens --destination <new-golden-dir> --scratch <private-scratch-dir>
<CPU-wrapper> -m tests.ct_supervisor_goldens --destination <new-ct-dir> --scratch <private-scratch-dir> --ct-manifest --scenario child-timeout --scenario wall-cap
```

The CPU wrapper sets nice 15, all thread caps to one, CPU device selection and
private temporary directories. Original absolute paths in raw receipts/logs are
historical fixture values. Playback explicitly relocates and reseals paths in its
private copy; it never silently trusts an old scratch directory.
