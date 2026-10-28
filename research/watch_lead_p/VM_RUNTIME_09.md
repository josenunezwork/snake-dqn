# Packet-09 scheduling, construction and completion verification

Astra implementation; Opus reviews. Prepared locally only; no push, allocation,
Colab/account call, live dispatch or release is authorized by this handoff.
Original packet/evidence bytes and all scientific sources remain unchanged.

The operator's 2026-10-28 decision under owner delegation makes the existing
480-minute / 12.32-unit effective envelope governing for this attempt under the
owner's standing 2x rule. The 240-minute / 6.16-unit base is informational. Neither
is raised. RETRY_09.json and the sealed retry-accounting.json bind this direction,
the refreshed forecast and the unchanged mechanical completion accounting.

The newest contended fixture measurement (engineering-07/benchmark.json, original
12-worker forecast 274.54148 minutes) supplies every per-arm cost. Reprojecting the
same 96 tasks through 11-worker cohorts yields 284.90198 minutes (7.31248 units at
1.54 units/hour). Setup/export allowance remains 1200 seconds; no speedup is
credited for reserved CPU headroom or worker priority. Effective margin is
195.09802 minutes (40.6454%). This is a contended-Mac projection; L4 throughput
remains unqualified. The seal requires at least 20% effective-envelope margin.

A separate qualified transport rule still requires timeout >= 2 * est_minutes.
The truthful estimate is 285 minutes, so transport requires >=570 minutes. The
frozen effective guard is 480 minutes. Without an authorized resolution, this
packet is blocked at transport admission even though its forecast fits the
operator's governing envelope. Never reduce the estimate to evade this check.

Workers are scheduled in a spawn pool of 11. A worker initializer sets its nice
value to min(19, parent nice + 10), refusing if it cannot be strictly lower priority
than the parent. This is one setting per process, never an increment per task.
Each session retains W1/K1, seed allocation, action/branch rules and fixed horizon.
The real guarded fixture compares all six immutable unit components byte for byte
between pool sizes 11 and 12, with fixed source/input paths; records include
initial/final state, action trace, counts, roster and outcomes. The short fixture
covers 60 real ticks per arm on the excluded seed. Existing separate natural L600,
P0/P48, death/respawn and W1/K1 tests cover deeper controller behavior; this is not
a full H6000 or cross-platform parity claim.

SNAKE_MECHANICS_V2 has one executable read: web/backend/session.py
_mechanics_v2_enabled(), called per construction by _config_for(). No import-time
read was found. The adapter applies the intended environment before construction.
Inside each worker, after the real arm, construction metadata records that actual
reader result and get_config().game.mechanics_version, and refuses a mismatch.
The metadata binds the reader's source hash, adapter bytes and intended resolved
configuration. VM tests start with SNAKE_MECHANICS_V2=0; the adapter must still
construct mechanics-v2, report it accurately, and restore its environment.

Aggregation validates every payload's hash, canonical filename, unique index and
outcome-free construction metadata in one complete first pass. Only then does it
parse any scientific payload. The second pass requires the embedded allocation
index to equal the filename index. A bad later metadata file prevents opening
any earlier outcome. Ledger construction remains outcome blind.

Soft signals during metadata, metadata-sidecar and metadata-marker publication
finish/account the critical section. A separate injected mechanical interruption
at each phase leaves the session missing, keeps orphan metadata out of completion
accounting, and proves clean republication in a fresh completion output. Production
runner accounting does not glob metadata as session payloads.

Both qualified download routes are tested: runtime._exports -> collect_exports ->
individual export_file, and the real small-export batch archive and download path.
Only provider Contents GET/RPC IO is replaced by local file operations. The fixture
recovers all 32 synthetic units and their metadata, verifies all six components,
then runs make_ledger/read_ledger and the real Mac-only union aggregate on the
downloaded tree. Synthetic outcomes never use campaign seeds for real gameplay.

Optional findings addressed: adapter bank invalidation is explicit in RETRY_09;
vm_guard's docstring now accurately says it is archived as test support and only
activated via sitecustomize; worker guard proof requires at least one actual worker
and a startup record for every working PID rather than assuming two start equally.
Other optional findings remain for later review; no independent review is claimed.
