# R30 timing upload allowance: prospective NON-ATTEMPT repair

Authority: Astra `JUDGMENT_R30_TIMING_UPLOAD_CAP_NONATTEMPT_20261025.md`,
SHA256 `de404ed2ec595ed60f506ca403e142ae0ee7e90666b2a6c49b64c457c103a51b`,
read fully and verified against its `.sha256` sidecar on 2026-10-25.
This sibling amendment implements only that ruling. G5 and main admission remain HOLD.

Probe-01 at `big-lever/round-30/bl-r30-postf4-20261025-v1/timing-calibration-probe-01`
was refused locally before allocation: receipt UPLOAD_CAP, null generation/session/endpoint,
zero elapsed time and no CLI calls, transfers, control operations or terminal workload.
It is a **NON-ATTEMPT**. Its packet, local staging/output, receipt, operator log and
accounting records remain unchanged. The single allocated timing attempt remains available;
no extra allowance or automatic retry is created. Its `stop_verified=false` is not an
orphan-runtime indication for this pre-allocation refusal.

The timing packet builder now fixes the qualified shared `max_export_bytes` transport
field at **67,108,864 bytes (64 MiB)**. Every prepared source/control file and declared
private input is checked offline by size and hash before operator dispatch. This changes
no qualified transport implementation or other lane and grants no transfer-time exception.
The exact opaque bundle stays
`af82651cec44d3878b91b7410c3b844bf890d779e0bceb305ac3d93a70ef9455`
(36,189,672 bytes), without unpacking or reading private contents during preparation.

The producer separately enforces **524,288 bytes (512 KiB)** on canonical `timing.json`
before publication. Admission checks stat the payload before parsing, verify SHA/completion
sidecars, require exactly timing.json and those sidecars, and reject extra scientific exports.
The forecast independently rejects native receipt export sizes above 512 KiB or an export
inventory other than one timing.json. Native receipts use the qualified runtime's `size`
field. Existing exact typed metadata redaction and numerical log suppression remain binding.

Retained unchanged: probe route `research/bl_round30/timing_colab.sh`; all frozen training
cache rows; exact fit/audit plans and full 2,000-batch objective; continuous Adam updates
0–10/10–20/20–30; closures at 0/10/30; full update-30 certificate with 600-second cap;
transfer <=150 seconds and staging <=300 seconds; base 15 / effective 30-minute session;
worker <=1,500 seconds and actual remaining runtime minus 300-second cleanup/export reserve;
`no_automatic_retry=true`; original 40-minute accounting liability pending reconciliation;
native terminal receipt and independently verified release. No reduced surrogate, relabeling,
preliminary canary, scientific admission or serving change is authorized.

Changed sources and bindings must be pinned to a published commit in a fresh create-only
`timing-calibration-probe-02` packet with unique job/output/supervisor-state paths. Focused
synthetic tests, every prepared/private upload size, and dispatch `--dry-run` must pass.
Supply that packet for independent predispatch acceptance. Sol prepares; the operator alone
may dispatch once after acceptance. This implementation performs no Colab calls or dispatch.
