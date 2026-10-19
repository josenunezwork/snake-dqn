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

## Verified facts and corrections (2026-10-18 evening)

- Owner signed in: `colab --auth oauth2 whoami` = josemnunez.jn@gmail.com (bundled public OAuth client, copy-paste
  code; scopes openid/profile/email, cloud-platform, colaboratory, drive.file). The "Expires in 59m" is the access
  token; the stored refresh token renews it. Refresh tokens end on revocation, ~6 months of non-use, or >100 newer
  tokens per client; password changes only revoke Gmail-scoped tokens (none here).
- `colab usage`/`sessions` also see notebook-started runtimes (balance 1178.30, 6.84 u/h for one A100 + one L4).
- Unit rates (no official table; mccormickml.com 2026-03-10, fluctuate): T4 1.19, L4 1.71, A100-40GB 5.40,
  A100-80GB 7.52, G4 8.71, H100 unmeasured. Our planning keeps L4 = 2.0.
- Correction to the ranking above: a background process alone does not keep a runtime; liveness follows kernel
  execution (CLI keep-alive daemon removed 2026-09-25), so the supervisor keeps a kernel-side supervised job and
  stops the VM when done (Astra: docs/research/advice/2026-10-18_colab_cli_operating_model.md).
- Colab FAQ (fetched 2026-10-18): Pro+ continuous execution up to 24 h with sufficient units; free-tier-only bans on
  SSH/remote control/bypassing the notebook UI lift with a positive balance; GPU types vary over time; units expiry
  not stated.
