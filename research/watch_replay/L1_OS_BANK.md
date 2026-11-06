# L1-OS packet completion and merge

`prepare_l1_os.py` prepares three disjoint 44-session packets. A stopped runner
prints its result JSON followed by `INTERRUPT: session budget` or an explicit
signal/stop-grace reason. The supervisor's `CHILD_NONZERO_EXIT` receipt preserves
that final line, so completion-retry admission recognizes the mechanical stop.
Admission still requires verified stop, matching manifest/contract lineage, an
exact completed-unit ledger, missing units, and at most two completion attempts.

Bank resume archives retain the newest **two** verified checkpoints per unfinished
unit. If reconstruction rejects the newest, the runner can try the older snapshot.
The existing 128 MiB archive cap still applies. Screen callers retain their default
of one checkpoint unless they explicitly request more.

After all three packets finish, merge their explicit `sessions` directories:

```bash
SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.watch_replay.merge_l1_os \
  --roots /absolute/packet1/exports/sessions \
          /absolute/packet2/exports/sessions \
          /absolute/packet3/exports/sessions \
  --bank /absolute/fresh-l1-os-bank
```

Run from this worktree. For a packet completed across attempts, list every attempt's
`sessions` directory. The tool requires all 132 allocated sessions, one source
commit, matching packet contracts, complete validated recordings, unique units,
and unique seeds. Duplicate units are refused even if their payloads match; supply
only each attempt's newly exported units. An existing destination is refused.

The destination contains `000000.watch.gz` through `000131.watch.gz`, each with a
new `<sha256>  <destination filename>` sidecar and a `complete\n` marker. Gzip bytes,
header unit IDs, original allocation indices, and source provenance stay unchanged.
`merge.json` records the source-to-destination mapping and counts. The resulting
directory can be read by `web.backend.watch_replay` and the controller extraction
mode of `research.watch_replay.install`; do not feed it back into the single-packet
`install --contract --roots` mode, which expects original allocation filenames.

To exercise the merger on the local job-shell smoke output, use
`--roots /absolute/smoke/output/sessions --bank /absolute/fresh-smoke-bank --fixture`.
This mode accepts only known smoke/packet namespaces with horizons at most 300 and
marks the report as a fixture. It does not provide release or serving qualification.
Merging never changes the active bank or serving configuration.
