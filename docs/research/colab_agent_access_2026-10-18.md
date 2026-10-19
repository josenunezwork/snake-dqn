# Using Colab compute units from coding agents (researched 2026-10-18)

Verified on PyPI/GitHub: `google-colab-cli` 0.7.4 (official, googlecolab/google-colab-cli, Apache-2.0, pushed 2026-10-15).

## Ranked setups
1. **Google Colab CLI** (`uv tool install google-colab-cli`; macOS/Linux). Agent-usable from Bash:
   `colab new -s NAME --gpu H100|A100|L4|G4 [--high-mem]`, `colab exec -s NAME -f script.py` (kernel state persists),
   `colab upload/download`, `colab status|sessions|usage` (units balance + burn), `colab stop` (real unassign),
   `colab run --gpu A100 job.py` (allocate, run, always release). Uses Pro+ units. Ships `skills/colab-operator/SKILL.md`
   (targets Claude Code, Codex, Antigravity). One-time OAuth (`--auth oauth2` copy-paste or gcloud ADC with the
   colaboratory scope). Limits: `colab drivemount`/`colab auth` still need a human browser click; idle VMs are reclaimed
   (keep a process running) and burn units until stopped; allocation retries via `COLAB_CLI_ASSIGN_RETRIES`.
2. **Take Drive off the critical path**: replace the Drive queue + worker notebooks with CLI upload/exec/download from
   the Mac; durable checkpoints to a GCS bucket or Hugging Face Hub with credentials we upload (untested on consumer
   Colab). Removes the per-runtime Drive OAuth click.
3. **Colab VS Code extension** (`google.colab`, Marketplace + Open VSX): interactive notebooks on Colab GPUs from
   VS Code; GPU is picked in UI (no API); Antigravity/Cursor via Open VSX unverified; Drive mount still needs OAuth.
4. **Colab MCP server** (`googlecolab/colab-mcp`): agent edits/runs cells in a notebook open in the browser; no runtime
   allocation or GPU choice. Better than typing into cells, weaker than the CLI.

Not useful here: Colab Enterprise/Vertex (billed per GCP VM-hour, not Pro+ units); third-party mcp-server-colab-exec
(inactive, T4/L4 only); third-party tunnels (grey area under the "remote proxies" ban). Pro+ FAQ: up to 24 h continuous
execution with units; idle timeout applies only after code stops; SSH/bypassing the UI is banned only for free
zero-balance runtimes (`colab ssh` is an official feature).

Sources: github.com/googlecolab/google-colab-cli; developers.googleblog.com (CLI 2026-06-05, VS Code 2025-11-13,
MCP 2026-03-17); research.google.com/colaboratory/faq.html; cloud.google.com/colab/pricing.
