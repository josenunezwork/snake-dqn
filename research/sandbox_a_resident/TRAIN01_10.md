# TRAIN-01-10: bounded kernel source launch

Prepared only; no dispatch. This execution-only successor preserves TRAIN-01's
learning recipe, runtime pin `c8ab2e6feefc63ce866e1493bfd2f7f1bf3f64dd`, allocations,
empty ledger, frozen inputs, monitoring/nomination rules, endpoint and envelopes.
Packets 07–09, their detached checkouts and attempt evidence remain immutable.
This note supersedes the packet-only-remedy and transient diagnosis in the older
TRAIN01_09 note; it does not revise that note's bytes.

## Failure boundary

The post-upload source control calls lasted 24.263 seconds (07), 30.014 seconds
(08, failed), and 30.009 seconds (09, failed). Both refusals then exhausted three
kernel-idle recovery probes; stop was verified. Zero assignments before/after
is distinct from zero allocations: each refusal allocated and released a runtime.

The exact kernel payload is assembled by `Broker.execute.deliver_source` in
`research/colab_cli/supervisor.py`. It imports the uploaded, pinned `source.py`
and calls `mod.fetch(root, plan)`. `source.fetch` synchronously initializes a Git
repository, runs `git fetch --depth=1 https://github.com/josenunezwork/snake-dqn.git
<full-pinned-sha>`, checks commit/tree, and checks out the full tree. Its internal
fetch allowance is 120 seconds, while the enclosing staging control RPC is capped
at 30 seconds. `source_paths` limits only the fallback archive, not GitHub fetch.

The already-backgrounded runtime child is waiting for `ready.json` throughout
this operation. Bootstrap, `train_job.sh`, Torch imports/compilation, proxy-cache
construction and update-16 diagnostics cannot cause this particular timeout:
none runs until source delivery and hardware admission finish. The two pinned
`research/colab_cli` trees are identical (`5833e54e9c5396121cbbf99a5cd4e52006e564fd`).

Between `152af938` and `c8ab2e6f`, diagnostics and update-16 monitoring changed
training code, but the delivery algorithm did not change. The full repository
grew from 12,934 files / 571,025,163 bytes to 12,943 / 571,117,470: only 9 files /
92,307 bytes more. Packet 08 and 09 have identical runtime inputs and source
plans; their dispatch identity and local paths differ. Packet 07's sparse archive
is 918,666 bytes (3,784,137 unpacked); 08/09 are 925,149 (3,812,943 unpacked), all
191 files. The receipts contain no Git subprocess output, so attributing the
remote delay to a specific GitHub/cache/network mechanism would be inference.

## Adapter fix and launch boundary

The packet explicitly selects `sandbox_source_delivery: background-sparse-v1`.
Only TRAIN-01 sandbox admission permits this value. The host sandbox adapter
selects the qualified, precomputed sparse archive directly, avoiding the
unbounded-for-RPC full-tree GitHub path. The archive still passes the qualified
upload verification and exact `source.install_fallback` inventory, member-mode,
size, path and SHA checks; no synthetic successful delivery is accepted.

The install control only writes the small worker and passes its plan to a subprocess,
then returns its PID. It performs no source unpack, source hashing, model import,
compilation or learning. The worker prints start/completion/error progress,
imports the unchanged pinned installer and publishes its result atomically.
The host waits using Contents file reads under the existing session/staging
caps, then validates the verified commit/tree result before the qualified broker
can publish `ready.json`. Lease renewal, accounting, workload telemetry and
cleanup remain qualified. No source operation is replayed after an uncertain RPC.

Normal jobs and sandbox packets without the explicit option retain their old
route. No file under `research/colab_cli` changes, including the 30-second timeout.
Regression tests exercise a deliberately slow worker, error and timeout paths,
source corruption, identity verification, route scope and hook restoration.
The frozen TRAIN-01 preflight now includes these launch-boundary tests.

## Evidence and operator handoff

Create-only evidence lives under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-10-engineering-20261027`.
Local installs of the actual 07/08/09 sparse archives took 0.025–0.035 seconds
across nine independent temporary directories. Correctly timed full-source fetch
of 07 took 32.379 seconds; 09 exceeded the independent 35-second bound and its
process group was terminated. Thus even the older source pin can breach the
kernel cap on a fresh fetch. The initial scout timing files under `root-cause/`
contain a misplaced inner timer and are retained as superseded diagnostics;
use only `fetch-07-corrected.json` and `fetch-09-corrected.json` for fetch timing.
Corrected full-fetch timing,
launch timing, exact verification commands, frozen identities and preservation
hashes are recorded in that directory and `VERIFICATION_TRAIN01_10.json`.
Local evidence does not qualify a live Colab acknowledgement time.

The fresh detached broker is
`/Users/josenunez/Projects/ml/snake-dqn-train01-10-broker.noindex`; the fresh detached
runtime is `/Users/josenunez/Projects/ml/snake-dqn-train01-10-runtime.noindex`.
State and output under `train01-attempt-10` remain absent until operator dispatch.
See `packet-train01-10/SEAL.json` and `packet-train01-10/DISPATCH.txt` for the frozen
identity and exact command. `dispatch.sh --dry-run` verifies the prepared packet;
it does not allocate, check live account inventory, or reserve the identity.
