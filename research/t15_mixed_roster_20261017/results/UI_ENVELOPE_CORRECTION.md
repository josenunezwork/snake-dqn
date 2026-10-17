# T15 UI envelope correction — sibling evidence assembly

The first UI result envelope in `run-v1/records/ui-000.json` contains
`wall_seconds: null`. Its SHA256 is
`f16028a0c2ba774951777dde0e478f984484df15fb914c0000f33fdb57d9aa10`.
The frozen independent audit correctly rejects this envelope as INVALID. The
original file remains byte-identical; this memo does not retrospectively validate it.

The UI was observed once, in the owned Chrome tab at
`http://127.0.0.1:8815/#controls`. The startup backend record, two screenshots,
two DOM snapshots, browser observation and later backend state are preserved.
The test server and tab have been stopped/closed. No gameplay or UI observation
is repeated to repair this clerical metadata omission.

The separate `run-v1-assembly` evidence directory copies the unchanged intent and
all original observations and valid results byte-for-byte. It retains the invalid
envelope at `originals/run-v1-ui-000-invalid.json`. Its replacement UI envelope
changes only `wall_seconds`; every other key and value, including the complete
`data` object and intent binding, is identical. This is an evidence assembly,
not a new experiment, seed allocation, producer or source revision.

The duration is the recorded UTC interval from session construction to the final
browser evidence capture:

- Start: `ui/backend.json` field `created_utc`,
  `2026-10-17T13:59:19.243267+00:00`;
  SHA256 `204282f4339bd782f5c85bdb23a777a3b03c5234e84523b58745f625acfe0d3b`.
- End: `ui/browser-observation.json` field `created_utc`,
  `2026-10-17T13:59:58.147Z`;
  SHA256 `82eb602a75574b3d143dfd3f40a446f1033d8aa24c782c31750750aace8fc7c2`.
- Formula: end minus start = **38.903733 seconds**. This is a duration derived
  from retained timestamps, not a monotonic performance measurement or a latency
  sample. The later backend API response is supporting identity evidence and is
  outside this browser-capture duration. Filesystem mtimes are not used.

Only the assembly receives the final qualification receipt and six-check audit.
The original run's UI envelope remains INVALID. SERVING_PASS for the corrected
assembly is conditional on its independent frozen audit passing all six checks.
The original invalid envelope, this memo, both timestamp source files and the
corrected envelope are all included in the assembly's receipt. The publication
must disclose this correction and distinguish the original from the assembly.

This correction does not change the pinned protocol, producer, audit, source,
checkpoint bytes, decisions, seeds, screenshots, DOM, counters, serving defaults
or qualification thresholds. Independent audit review recommended this separate,
provenance-bound assembly rather than overwriting any create-only result.
