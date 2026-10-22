# Native L4 paired ruler preparation — 2026-10-22

Owner scope: prepare, validate locally, publish source and dispatch commands;
no Colab calls or dispatch. Same nine burned fixtures, native kernel, H300,
`all` and `exact_stop`, 30 repeats per fixture per cell, twelve one-thread
workers. The existing Mac receipt remains immutable. This is timing and
placement preparation, with no selection-bank execution or release authority.

`native_manifests prepare` verifies the passed Mac receipt, its exact fixtures
and K1_PASS digest before creating the new family. Z-9 transport uploads one
deterministic archive containing the original 30 fixture export components,
three name-correct K1 export components, and the existing public checkpoint
catalog. Canary and main upload inventories are identical. Public checkpoint
bytes are downloaded and hash-verified by the existing runtime materializer.
Local transport qualification exercises the real pinned sparse-source builder
and supervisor upload primitive, including fallback bytes, the 150-second
forecast limit and shared 300-second deadline. It makes no provider call.

The new `run_native_l4.sh` family uses the unchanged native CPU batch runner.
Each invocation first executes its own one-repeat paired native canary. The
main invocation then executes the full ruler using that just-created receipt;
no generated native-canary receipt is added to the upload inventory. A real
B8 CUDA masked-action audit on a restored frozen root runs outside the CPU
timing cells in both invocations. The audit establishes the family gate's CUDA
action evidence; its CUDA action/timing does not feed any native ruler choice,
rate or placement result. It is not CPU/CUDA parity qualification.

Family probes are submitted through the existing `canary` schema and unchanged
supervisor validation. The shared helper's `FAMILIES` list is a convenience
registry for its generic preparation CLI; this package constructs the new
script's explicit probe manifest directly. No shared transport executable or
canary gate is changed. Source/commit, exact uploads, actual CUDA decisions,
real scanner exports and verified release remain required by the broker.

Canary: 18-minute whole-session cap at 2 units/hour, 1-unit cap with 0.2 reserve.
Main: 72-minute whole-session cap, 3-unit cap with 0.6 reserve. Together they
reserve at most 4 units; worst-case wall liability is 3 units plus 0.8 reserve.
Native entrypoint caps are 600 seconds for the family canary and 1,800 seconds
for the main invocation including its internal native canary. Setup, public
fetches, compilation, exports, failures and stop consume the whole-session
caps. Estimates (9/36 minutes) are prospective allocation envelopes, with a
2x wall margin, not measured L4 throughput and not transferred Mac timing.
First cap wins; preserve failed attempts; no automatic retry or allocation.

Full main admission cannot exist before an undispatched family canary has
produced real evidence. Offline preparation records `PASS_PRECANARY_STRUCTURE`
only after `supervisor.validate` reaches its exact missing-canary final gate;
all earlier checks must pass. The canary manifest passes complete local broker
validation. After the real canary finishes and release is verified,
`native_manifests bind` re-derives the broker proof and creates a separate
immutable `main-admitted.json`; the prepared template is never overwritten.
Do not treat local preparation as live canary execution or a release receipt.

The process-gate merge added a Mac-only CLI lease to `ruler.py` after the passed
Mac native ruler. `native_provenance` admits that exact historical ruler hash
and the former native-validator hash, plus omission of the new compatibility
helper only for that former native hash. Every other source dependency must
match. AST regression checks compare the computational functions against the
historical Git blobs. The original receipt bytes and source inventories stay
intact, and placement reports retain both original source inventories. The
known hash transitions do not permit another operator, fixture, rule, branch
mode, kernel or timing change.

At dispatch, refresh the live inventory and price every other assignment in
`budget.other_assignments` as a map for each admission. If inventory changes,
create fresh manifest copies and rerun validation; never edit a dispatched
manifest. Run canary first, bind its actual released broker receipt, then run
main. Only `native_manifests publish` may place the recovered full ruler receipt
at `big-lever/exp/bl-l2/y1-native-l4/ruler-receipt.json`: it checks clean broker
PASS, verified release, scanner digest, paired native validity and exact Mac/L4
semantics before publishing the payload and sidecars create-only. Then call
`placement_stop --l4-receipt` for the prospective placement recommendation.
The exact paths, source SHA, local validation records and ordered commands are
in the prepared artifact family's `DISPATCH.txt`.
