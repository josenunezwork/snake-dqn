# R33 excluded verification packet

This is the bounded R33 rows 2–3 adapter. It uses the qualified Colab CLI transport;
it cannot allocate, dispatch, authenticate, mount Drive, or admit science. Existing
factory, F4, P1, serving and A1-PROF packages are not converted to this adapter.

The frozen inventory has three resumable shards: durable ops/proof checks, package
contracts, and the shared process/slowdown/fake-transport checks. Every required
node must be collected and pass. Missing nodes, skips, xfails, and unexpected passes
under an xfail declaration are INCOMPLETE. Native Darwin ownership, actual A100
correlation/attribution, Mac timing, strict/product and serving remain named pending
obligations. Linux CI is independent of the frontend job and has a ten-minute bound.

The real entrypoint is `colab.sh`. Inputs are uploaded relative to `COLAB_INPUT_DIR`:
`verified-ledger.json` and `source-inventory.json`, both including their destination
SHA sidecars and last-published completion markers. `resume.ledger_path` is runtime
root-relative `inputs/verified-ledger.json`. The runtime checks the complete uploaded
source map, exact pin, declared environment and frozen packet before work. Payloads
are immutable, atomic, and namespace/allocation/index-bound. Remote attempt terminals
are diagnostics; only the Mac union consumer can return `PASS_LINUX_CONTRACTS`.

The Mac ledger reads only filenames, sizes and hashes. Its separate consumer reads
all canonical unit identities and requires a current terminal receipt. Interrupted
prefixes remain usable without fabricating a job terminal. Duplicate or foreign units
are INVALID. A complete unit whose acknowledgment was lost is skipped, never rerun.
A final pilot PASS additionally needs real supervisor, accounting, exported-byte and
independent account-release evidence. Cleanup/RPC uncertainty remains HOLD.

## Frozen funding and retries

| Envelope | Base | Effective (2x once) |
| --- | --- | --- |
| Shared series | 144 min / 0.192 u | 288 min / 0.384 u |
| Included reserve | 24 min / 0.032 u | 48 min / 0.064 u |
| Original | 120 min / 0.160 u | 240 min / 0.320 u |
| Each of at most two completions | 12 min / 0.016 u | 24 min / 0.032 u |

Planning rate 0.08 u/hour, original forecast allowance 120 min / 0.160 u, including
setup/export. Estimates, N, rates and retry count are not doubled. Reserves are
included. No cap redistribution or mid-series raise. Smaller live/account/global
limits win; existing F4/R28/P1 liabilities keep priority. A different measured CPU
rate needs a pre-dispatch reforecast within these ceilings. No L4 comparison is funded.

Completions are operator-owned, never automatic. They require a same-pin interrupted
child (registered exit -15/-9/124/137/143), verified stop, missing shards, matching
immutable manifest/receipt history, per-attempt and cumulative caps, and a fresh
manifest/output. Contract-test failures close the packet. RPC/cleanup uncertainty,
timeout after success, foreign identities, exhausted caps, noncontiguous history or
no missing work never authorize a replacement. Keep every prior receipt, including
any existing exact accounting retry; history that this adapter cannot reconcile is
HOLD for full review. It does not change that retry's existing authority.

## Preparation and operator path

Use a full published pin and fresh absolute paths. The native preflight acquires the
existing shared test slot and Mac lease, sets a nice floor of 15 without raising
priority, and has a 120-second complete deadline. It installs the declared dependencies
in a fresh scratch environment, archives the pinned sparse closure, scrubs credentials
and ambient imports, runs the actual Bash entrypoint with three tiny excluded shards,
verifies the Mac consumer, and writes an immutable receipt. It refuses an active lease.

```bash
python -m research.r33_verification.preflight --commit FULL_PUBLISHED_PIN --out FRESH_PREFLIGHT
python -m research.r33_verification.pilot --commit FULL_PUBLISHED_PIN --root FRESH_PACK \
  --preflight FRESH_PREFLIGHT/receipt.json --independent-receipt CODEX_V_RECEIPT
```

`pilot` validates the exact pin/closure in both receipts, runs the existing isolated
import preflight, writes the empty first ledger, input inventory, proof dependencies,
manifest, preparation receipt and dispatch instructions. It prepares only. Claude uses
`tools/ops/dispatch.sh` after fresh qualification, assignment/rate/balance/liability,
authority and host checks under the existing allocation lock.

For an eligible completion, repeat `pilot` with fresh `--root`, the original
`--previous-manifest`, Mac `--verified-ledger`, and every `--prior-receipt` in order.
It refuses uncertainty and cannot launch the completion. After exports are downloaded:

```bash
python -m research.r33_verification.verify ledger --manifest PACK/manifest.json \
  --attempt OUTPUT/exports --out FRESH_LEDGER --decision FRESH_DECISION
python -m research.r33_verification.verify observe-release --receipt OUTPUT/receipt.json \
  --out FRESH_RELEASE_OBSERVATION
python -m research.r33_verification.verify accept --manifest LIVE_MANIFEST \
  --receipt OUTPUT/receipt.json --decision FRESH_DECISION \
  --release-evidence FRESH_RELEASE_OBSERVATION --out FRESH_ACCEPTANCE
```

Repeat `--attempt` for prior immutable attempts in the union, and supply all prior
`--prior-receipt` values to `accept` on completion. Account release observation is
read-only. Exact frozen operator paths and pins are in `research/r33_ops/HANDOFF.md`.
