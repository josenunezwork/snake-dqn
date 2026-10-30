# Watch replay source (T34, Astra)

Opt-in playback of Colab-precomputed, complete Watch6 sessions. Default Watch
and Play remain live. Nothing here dispatches a job or changes release pins.
This is a product recording pipeline, not a fresh efficacy look or a promotion.

## World and controller

`product.py` and `engine.py` adopt the constructor, live/Grid mirror and exact-stop
implementation from `astra-watch6-search-efficacy` at
`881f7bf675a765f3ba27a1dae26c14939b811bc4`. They use actual mechanics-v2 Watch,
continuous respawn, B8 s3 hero and five FRP3 s12+v8 opponents. Production units
contain exactly 6,000 game frames. The generator audits live physical/RNG/carry/v8
mirror state every frame. No inference, search, or resimulation runs in playback.

`exact-stop-l2-prime` retains L>=600, every fifth eligible decision, H300,
bare-first/descending-Q alternatives, first-survivor stop, unchanged all-die
ranking, and trigger reset on respawn. `bare-b8` is a separate optional bank.

## Format and controls

`snake-watch-replay/v1` is deterministic gzip (`mtime=0`) over canonical JSON
lines: metadata header, then ordered existing protocol-v2 frame DTOs. Header
contains controller, checkpoint SHA256s, source commit, frozen allocation digest,
namespace/index unit ID, seed, horizon, fixture flag and whole-session stats.
Frames include full bodies/heads/boost/length/food, inspector/netviz and roster
counters, plus death causes and delivered hero decision events. Runtime-only v8
wall-clock diagnostics are omitted; browser-consumed counters are retained.

Frame recording avoids an unsupported cross-platform seed+action reconstruction
claim: the existing mirror checks live RNG and policy carry at each frame, but
an action log alone does not export that complete initial/respawn state. Frame
decoding is tested byte-for-byte, independent of x86/Mac inference ties.

Each complete recording is published atomically with destination-bound `.sha256`
and `.complete` last. Producer validates before completion. Consumer verifies
all frames and the final count before exposing frame one, pins compressed bytes
to prevent a file replacement during playback, and refuses unknown version,
world/controller/checkpoint, corrupt/truncated/oversized/fixture recordings.
Bounds: 128 MiB compressed, 512 MiB decoded, 2 MiB per line, 1,024 bank paths.
One recording is held at a time, including during selection.

Set `SNAKE_WATCH_SOURCE=replay` and `SNAKE_WATCH_REPLAY_DIR=/absolute/bank` only
when ready to opt in. The server first builds the normal released live fallback.
Path B live search is disabled in replay-source Watch. The bank shuffles without
reusing a unit in a cycle or immediately repeating it. Empty/bad/exhausted
single-session banks use live B8. Selection retries only at reset or re-entry to
Watch; it does not repeatedly hash bad files on every live tick.

The header visibly says replay and “Recorded assisted hero” for exact-stop.
Pause/speed operate on playback. Inspector follows recorded hero #0; roster
remains visible. Hero selection reports that limit. Food/epsilon or a valid
checkpoint load return to live operation. Play always clears replay and uses
the live game/input/scoring path. Explicit checkpoint constructors bypass replay.
Playback statistics are recorded session statistics, not live B8 release-monitor
samples. Fresh viewer count, pause/speed and errors overlay the stored DTO.

## Preparation and placement

Use the project interpreter, CPU inference and all seven one-thread environment
caps. `preflight` runs three excluded real tiny sessions plus three excluded
long-body H300 branches. `smoke` separately proves W1==W2 recording SHA256 and
uses FastAPI's real WebSocket handler for recorded DTO, live fallback and Play.
All output directories are fresh; no evidence is overwritten.

`prepare --commit FULL_PUBLISHED_BRANCH_SHA --benchmark ABSOLUTE/benchmark.json
--out ABSOLUTE/FRESH_PACKET` defaults to N=4 (1..12 per packet). It uploads both
main and excluded canary contracts/empty ledgers, the two exact checkpoints,
and portable pinned release evidence. Sparse source includes the static research
closure plus product source/config. No runtime `/Users` dependency or Drive.

Use L4 with >=12 vCPUs, four independent one-thread CPU session workers; GPU is
unused. This is CPU singleton policy/v8/simulator/search work. L4 supplies the
known host/core configuration; A100/H100 adds idle accelerator capacity. CPU
runtime remains an alternative only after its actual vCPU count and useful rate
are qualified. Planning L4 rate=1.54 u/h is refreshed by dispatch, never assumed
to be a guaranteed current provider rate.

Forecast (explicit assumptions): 1,200 possible roots/session ×1.2 branches/root
×slowest measured H300 singleton seconds ×3 cloud slowdown ×1.5 W4 contention.
The physical maximum is six branches/root. `forecast.json` reports expected
sessions/runtime-hour and setup/export allowance. Actual full-session branch
frequency, CPU contention and transport remain unmeasured. Counts, estimates and
rates are not doubled. Base wall/cap/reserve and effective 2× guards are frozen
in manifests before data. Jobs have <=60-second progress and a 900-second
completed-work watchdog. SIGUSR1 aborts partial work; prior completed session
units remain eligible. There is no partial-trajectory resume.

Use only `tools/ops/dispatch.sh NAME /absolute/canary.json --broker ABSOLUTE_REPO
--cwd ABSOLUTE_REPO --state-dir ABSOLUTE_FRESH_STATE --dry-run`, then the same
line without `--dry-run` for operator dispatch. This assignment does not dispatch.
Main refuses until an actual same-pin, same-inventory L4 canary PASS with verified
exports and `stop_verified` is bound. Prepare a fresh packet with
`--canary-receipt /absolute/output-canary/receipt.json` after that proof exists.
These commands remain subject to independent Argon review before live execution.

## Resume and bank installation

Frozen unit identity is namespace + `sessions` + allocation index; seed derives
from namespace/index. Contract binds N/controller/horizon/commit. The uploaded
Mac-verified ledger skips exactly its validated IDs. Foreign namespace/allocation,
out-of-range and duplicate IDs refuse. `resume --contract CONTRACT --roots
OUTPUT/exports/sessions ... --out FRESH_LEDGER` reads paths/size/hash and exported
contract binding, not outcomes. Retain the broker receipts too.

One original plus at most two outcome-blind completion attempts are allowed after
verified mechanical timeout/lease/transfer/upload/staging/interruption/runtime
loss, verified stop, and missing units. No retry on outcomes, mirror failure,
nonfinite Q or malformed data; no automatic completion retries. `prepare` accepts
`--ledger FRESH_LEDGER --previous-receipts RECEIPT1 [RECEIPT2]`, checks stopped
mechanical failure lineage and matches the entire verified ledger to prior broker
export inventories. Use fresh job/packet/state/output/receipt identities. Changed
ledger bytes require another same-pin canary with that exact inventory. Preserve
all prior manifests and results. Incomplete units are discarded, never spliced.

`install --contract MAIN_CONTRACT --roots OUTPUT/exports/sessions ... --bank
ABSOLUTE_FRESH_BANK` is a Mac-only complete-union check. It refuses missing,
duplicate, corrupt, short-fixture or foreign sessions before publishing the bank.
No frame/outcome aggregation occurs remotely beyond per-session recording stats;
no efficacy verdict or release authority is inferred. Enabling the flag is an
operator action and was not performed by this implementation task.
