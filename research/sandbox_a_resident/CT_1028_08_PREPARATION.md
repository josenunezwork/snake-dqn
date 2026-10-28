# C/T packet 08 preparation — CUDA capture pending

Astra implements; Opus reviews. This is admission and fixture-canary preparation,
not a reseal of the science packet. Do not dispatch C/T science until the operator's
A100 capture has been folded into the observed-line goldens and Opus accepts the build.
`packet-ct-1028-08` deliberately does not exist yet. Existing packets and evidence remain immutable.

## Priced-unit continuation and account arithmetic

The operator's 2026-10-28 instruction restores the exact registered baseline
9d32f804 priced-unit causes, without a new ruling:

- `priced session unit cap reached`
- `priced session unit cap reached during calibration`
- `priced session unit cap reached while awaiting account broker`

Only whole exact causes and the existing exact SupervisorError/Refused repr wrappers
are admitted. Appended text, alternate spellings, ambiguous provider stderr and the
other removed telemetry-recovery/busy-kernel causes remain ineligible.

Packet 07's frozen estimate/base/effective wall is 720/720/1440 minutes. Its base/effective
unit cap is 96.5/193, reserve 0/0, priced own-session rate 8.0 u/h. TRAIN-01 attempt 15's
local active record reports 5.3 observed and 8.0 priced u/h. These are Colab compute units,
with no conversion. The qualified supervisor prices its own elapsed session, not the
raw account debit from other assignments (`accounting.py`); `dispatch.py` registers
L4=1.54 and CPU=0.08, and uses the calibrated broker record for A100 rather than a static
public A100 price. Allocation rate calibration remains binding at actual dispatch.

193/8*60 = 1447.5 minutes; 96.5/8*60 = 723.75 minutes. Thus neither base nor effective
priced guard precedes its corresponding wall; the 193-unit effective cap also exceeds
2*(720/60)*8 = 192. At 5.3 observed u/h the effective cap lasts 2184.90566 minutes.
No cap change or budget ruling is required. Account balance, concurrent assignment
inventory, rate drift, release proof and CUDA throughput are not verified by these local facts.

## Admission corrections

`ct_progress.py` derives every emit stage pattern and exact key inventory from the AST
of ct_run.py and its collection/scoring/fit callbacks, including fit metrics, dynamic
stage numbers and resumed-ledger records. The coverage test independently walks every
emit call and callback dict. Finite fit loss diagnostics may be signed. Unknown keys,
phase strings and non-finite numbers still refuse.

ct_run retains its complete progress stream before printing. ct_job seals that history
on exit. For the qualified runtime's byte-sliced 4096/65536-byte tails, admission accepts
a first partial line only when it is an exact suffix of a complete, schema-validated
retained history line. It never silently drops that line. Missing witnesses or unknown
CUDA output refuse; real CUDA notices can only enter observed allowlists after capture.
Refusals identify the field/surface and omit raw progress values.

The 15-second SLA is not relaxed. A false marker or a claimed-true marker with elapsed
>=15 seconds refuses. The canary signals during actual publication of a CUDA-backed
255 MiB fixture tensor via ct_run's normal stage writer, records full checkpoint size,
checks completion-last stage publication and latest durable stage, and independently
requires signal-to-child-exit <15 seconds, including serialization, hashing, marker and
watchdog teardown. Live CUDA qualification is pending; CPU timing proves only the harness.

The exception scanner also rejects bare Error/Exception/Interrupt/Exit classes,
message-less exceptions, traceback headers, thread exception headers and custom
`Class: message` lines. Retained JSON/repr recursion and malformed source paths become
clean refusals. Source-plan tree is bound to the sealed tree or the source-commit tree
resolved at lineage admission, while archive-member admission remains Git-free.

Lineage carries each verified manifest's original bytes forward for retirement checks;
job identities come from ancestor manifests, not from a self-consistent receipt being
scanned. Generation agreement is tied to the hash-bound provenance receipt because the
qualified manifest has no generation before allocation. Record-local generation fallback
is removed. Every export root, regardless of basename, must lie under a bound output
root, and that entire root is scanned. Historical temporary paths exist only in tests.
Supersession checks OR hash, job and exact packet-directory criteria. No current
production caller lacking original bytes was found; the API also supports job/path
retirement without bytes. Tests require the full 13-scenario mechanical corpus, pin
refusal reasons, test a SHA-correct first edge into a two-node cycle, exercise matching
hash retirement, and dry-run fresh c1 before its outputs exist as well as fresh c2.

## Canary limits and follow-up

The separate packet is `packet-ct-cuda-smoke-1028-08`, with native canary instrumentation,
short-canary resume exemption, A100 only, estimate/base/effective wall 15/15/30 minutes,
base/effective guard 2.05/4.1 units and reserve 0.05/0.1 at 8 u/h. There are no automatic
completion retries. The qualified transport subtree is unchanged. Background sparse
source delivery uses the existing host adapter behind a narrowly checked fixture route;
no provider action or qualified-supervisor edit is introduced.

All model inputs are synthetic checkpoints initialized with fixture seed 31; fit/bootstrap
use fixture seeds 31/131. World ranges start at 950000000 and 951000000 (plus the fixed
validation/assessment offsets), disjoint from C/T science ranges and each other. The
preparation audit checks their combined low-63-bit collisions and protected-bank
exclusions alongside the full frozen C/T allocation, without reading numerical outcomes.
Two real ct_run CLI subprocesses capture all normal collection/scoring/fit/completion
stages, then an interrupted large-checkpoint run. This does not exercise the resumed-only
or failed-support branches; those remain covered by AST and CPU continuation tests.

Colab-limits walk: background sparse archive and every input are pinned and counted;
source compressed/unpacked sizes stay <32 MiB; source install remains outside the 30 s
foreground RPC; staging nominal/worst projections and upload count are sealed and must
stay <150 s, below the 300 s hard cap. Bootstrap uses Python 3.12/cu126 and the qualified
15-second checkpoint grace/900-second work watchdog. The roughly 267 MB interrupted
snapshot downloads in about 60 s nominal / 162 s worst at the documented rates, plus
split overhead, under both VM post-terminal and host session clocks. Individual exports
are <1 GiB and complete-last. Actual compile/setup, VRAM, transfer durations, account
inventory/balance/calibration and stop_verified/session disappearance remain operator
capture requirements. An incomplete or slow capture is a mechanical qualification
failure; do not infer a scientific verdict or retry it automatically.

After operator dispatch: preserve receipt, all polls/transport records, terminal CUDA
observations, raw child logs, complete progress histories and marker/stage SHA triplets.
Verify actual A100 execution, finite fits, complete stage inventory, measured SLA, input
and source pins, and verified release. Then add only observed CUDA lines plus negative
fault twins to the golden admission tests; rerun admission and numerical continuation
checks, obtain Opus review, and only then reseal C/T as packet-ct-1028-08.
