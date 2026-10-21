# Z-9: E4 upload path

Prepared locally only. No Colab allocation or dispatch is part of this change.
The supervisor needs no modification.

`research/colab_cli/transfers.py` sets `UPLOAD_TOTAL_TIMEOUT_SECONDS = 5 * 60`.
`supervisor.py` creates one deadline before uploading `bootstrap.sh`, `source.py`,
and all local inputs. Its source-archive fallback uses that same deadline.
The 2 MiB chunk size limits memory; splitting checkpoints into more files does
not create another five-minute allowance.

Evidence, preserved outside Git:

- E3: `big-lever/exp/bl-e3/colab-output-01/receipt.json` and `run-01.log`.
  The 57,831,386-byte input archive failed. Its partial transfer has no measured
  whole-file goodput. Bootstrap and source uploads took 59.148 seconds combined.
- E4: `big-lever/exp/bl-e2/e4-e8-output/receipt.json` and `run-e4.log`.
  B8 completed: 6,562,005 bytes / 147.511215541 seconds =
  **44,484.7869766 bytes/s**. Champion completed at 316,563.43 bytes/s and
  best_apex_fs at 116,479.39 bytes/s. The next checkpoint hit the shared cap.
  All six intended checkpoint files total **30,720,004 bytes**; at the slowest
  successful large-file rate they alone need about **690.6 seconds**.

The new package readiness entrypoint is `python -m research.bl_ppo.upload_readiness`.
It uses the shared experiment helper `research/bl_upload`, outside the protected
supervisor. Every generated E4/E2 draft receives a fast byte/hash check. Full
qualification calls the supervisor's real sparse-source preparer at the pinned
commit, measures its actual compressed fallback, and writes a create-only
`upload-preflight.json` binding the exact manifest and source identities.

The prospective forecast uses the slowest completed large upload above, a
65-second setup allowance (above the observed 59.148 seconds), and an extra
2 seconds per additional local input. The upload forecast must be at most
150 seconds, half the supervisor cap. The fallback forecast includes the
compressed source archive; adding the source fetch's 120-second timeout must
also fit the 300-second hard cap. These are conservative forecasts from saved
observations, not a guarantee about a future network connection. The supervisor
retains its independent hard limits.

For the registered six-checkpoint roster, generation checks local bytes against
the existing `research/t34_exit_b8/PUBLIC_INPUTS.json` identities and replaces
the six uploads with that small catalog. The runtime wrapper calls the existing
public-input materializer into a fresh `public-checkpoints` tree. It verifies
both public release asset sizes/hashes and all six extracted checkpoint hashes
before the E4 ruler starts. Unpublished custom inputs remain visible to the
budget guard; large inputs require an explicitly hash-bound download route.

Use a new artifact directory for each preparation. The E4 recovery command
supports `--e4-only --validate`; it writes the manifest, Z-9 receipt, and
`DISPATCH.txt` without invoking Colab. A manifest edit, including refreshed
assignment inventory, requires a new qualification receipt before dispatch.
E2 still requires its independent immutable E4 and E9 evidence. Upload
qualification does not supply scientific admission or serving authority.
