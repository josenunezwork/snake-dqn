# Native ruler scheduling correction — 2026-10-22

Owner instruction supersedes the fixed Friday calendar placeholder in the
historical STOP amendment/handoff: admit any single run once K1_PASS and the
existing host checks pass. The exclusive Mac lease expires two hours from
launch. The 7200-second cap includes the canary; the full run still deducts
canary elapsed time. Existing lease/output refusal, idle and foreign-load
checks, rules, bars, fixture bytes, and output paths are unchanged.

`run_stop_mac.sh --check` is read-only. `--run` acquires the canonical lease
and uses the existing canary/full sequence. No timing was run for this change.

No prepared manifest under the relevant bl-l2/bl-k1 artifact directories pins
`run_stop_mac.sh` or `native_ruler.py`; no manifest regeneration is required.
New native receipts automatically bind both updated source hashes through
`native_ruler.source_identity()`. Historical fixtures and K-1 pins stay intact.
