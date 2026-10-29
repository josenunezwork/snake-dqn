# C/T reseal-03: Mac continuation compatibility

This correction keeps runtime **af2bba1c8e7e44031a50a78cd282f7aec1d5b6c7** and the
existing reseal-03 seal intact. It changes only the Mac continuation/admission code.
Run continuation from a broker checkout containing this correction. Never modify
the live broker/runtime checkout or attempt-14-r3 state to install it.

`ct_resume` admits the exact H100 bootstrap literal
`2.9.1+cu126 12.6 (9, 0)` alongside the qualified A100 literal. Other capabilities
remain outside the closed grammar.

For the original preferred dispatch, pass `--gpu-preference`, `--accounting`, and
`--gpu-receipt` together. The writer verifies both admitted pure variants, selected
job/output identity, accounting source/live hashes, observed hardware, GPU and
supervisor receipt bindings, and the retained normalized manifest. A fallback
also requires the dispatcher's bound pre-endpoint reconciliation evidence. The
only permitted manifest differences are `budget.other_assignments` from the
accounting evidence and the supervisor's missing-field default `lease_seconds=180`.
No arbitrary lease change is admitted.

The resulting continuation remains bound to the original canonical manifest and
seal, and additionally records `completion_attempt.dispatch_binding`. This binds
all external evidence files by absolute path, size and SHA256. Ledger admission
reopens and checks these files again, across both continuation hops. Fault scanning,
verified stage union, source commit, scientific inputs and the two manual attempt
limit are retained. A fixed A100 successor drops its parent's `prefer` field; its
c2 uses `--accounting` alone, binding its ordinary dispatch source/live/retained
manifests, receipt and hardware. Transient account membership is refreshed again
for each successor.

The generic sandbox source check has a separate committed compatibility record:
`tools/sandbox/ct_mac_compatibility.json`. It permits exactly the old/new hashes of
`ct_resume.py` and `ct_cuda_observed.py`, only for this runtime and the original
reseal-03 manifest/seal (at most two bound lineage hops). All other source
comparisons still require equality. Original packet components in the original
broker are verified read-only against its exact `6c7ff7ea` commit and this seal;
no arbitrary external packet is accepted and successor packets stay in the new
continuation broker. Runtime staging continues to use the frozen
commit/archive, including the old versions of these two Mac grammar modules.

Exact mutually exclusive H100/A100-fallback c1 commands are retained in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-impl-ct18/OPERATOR_COMMANDS.md`.
They are **NOT RUN** and require a terminal, verified mechanical stop with at least
one eligible verified stage. Independent review is required before continuation.

## Next reseal note: review-17 MINOR-1

Keep this runtime unchanged. In the next reseal, derive `TORCH_CUDA_ARCH_LIST` from
`torch.cuda.get_device_capability(0)` and assert membership in `{(8, 0), (9, 0)}`.
Retain the adapter fault twin. This avoids compiling both architectures inside
the unchanged 900-second startup work window. No `ct_job.sh`, runtime, watchdog,
or architecture setting is changed by this correction. Measure the provisional
H100 rate after its first run; leave this seal's ledger namespace intact.
