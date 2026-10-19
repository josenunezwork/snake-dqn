---
name: colab
description: Operate the owner's Google Colab Pro+ (A100/H100/L4/G4 GPUs, ~1,180 compute units on 2026-10-18) from Claude Code. DEFAULT = the official Colab CLI (`~/.local/bin/colab`, signed in, no owner clicks) driven through the Mac supervisor research/colab_cli; LEGACY = browser-pane notebooks + Google Drive queue (only for already-running work such as Q1). Use for any Colab / GPU / "run it on Colab" task. Load colab-jobs too when launching or monitoring jobs.
---

# Colab: operating the owner's compute

For the project-specific supervisor contract, manifest schema, evidence, and failure handling, read
`docs/colab-cli.md`. Do not bypass it with direct mutating CLI commands for a new job; the Drive path below remains
only for the already-running Q1 work.

Owner instruction 2026-10-18: **all new Colab work goes through the Colab CLI**; leave already-running Drive-queue
work (Q1, the notebook workers) alone until it finishes. Design and guardrails:
`docs/research/advice/2026-10-18_colab_cli_operating_model.md`. CLI docs snapshot: `docs/research/colab_cli_ref/`.

## Colab CLI (default)

- Installed: `google-colab-cli==0.7.4` in `~/.venvs/colab-cli`, symlink `~/.local/bin/colab`.
- Auth: always pass `--auth oauth2` FIRST (`colab --auth oauth2 <cmd>`). The owner signed in once (2026-10-18,
  josemnunez.jn@gmail.com). Token `~/.config/colab-cli/token.json` (chmod 600) holds a refresh token; the
  "Expires in 59m" in `whoami` is only the access token and refreshes automatically. Re-login is needed only after
  the owner revokes access, ~6 months of NON-use, or >100 newer tokens for the same client (a password change only
  revokes Gmail-scoped tokens; ours has none): run `colab --auth oauth2 whoami` in a Terminal-panel tab
  (`mcp__terminal__run_in_terminal`) and ask the owner to open the URL, approve, and paste the code into the
  TERMINAL (never into chat). Never print or copy the token file.
- Read-only checks (safe anytime): `colab --auth oauth2 usage` (balance, burn/h, active assignments),
  `colab --auth oauth2 sessions` (all assignments incl. notebook ones, shown `[?]`), `status`, `log`.
- Run jobs through the supervisor `research/colab_cli/` (`docs/colab-cli.md` has exact commands): one broker serializes
  allocation, max 3 concurrent assignments account-wide (notebook runtimes count), `COLAB_CLI_ASSIGN_RETRIES=0`,
  unique `-s` names, per-lane `--config`, pinned bundles uploaded with hashes, job runs as a supervised subprocess,
  polling exec reads progress files, checkpoints downloaded to the Mac (hash, atomic rename) then copied to Drive
  for desktop as backup, runtime lease watchdog unassigns if the Mac goes away, `stop` then verify the assignment
  disappeared in `sessions` and write a per-job account-usage-after-stop report. `allowed_gpus` defaults to the
  requested GPU; only an explicitly pre-priced substitution is admitted. The runtime model/capability check writes
  `hardware.json` before ready/bootstrap.
- Pitfalls (from upstream source): `exec` kernel errors may exit 0 -> trust receipts, not the exit code; never
  resubmit `exec` to a busy kernel; client timeout != remote stop; `run` cleanup can swallow unassign failures
  -> verify release; transfers buffer whole files -> bound sizes/timeouts; idle kernels are reclaimed (no
  keep-alive daemon since 2026-09-25) but still bill until stopped.
- Never use `colab drivemount` / `colab auth` (human-interactive). No Drive on runtimes for CLI jobs.
- `colab stop` only works for sessions the CLI created; notebook runtimes show as `[?]` in `sessions`. To release
  one, the owner runs the library unassign (`colab_cli.common.state.client.unassign(endpoint)`, script pattern in
  research/colab_cli) — Claude's own direct unassign was blocked by the auto-mode classifier ("Interfere With
  Workloads") on 2026-10-18, so ask the owner (or a permission rule) instead of retrying. Measured 2026-10-18: one
  L4 alone = 1.54 u/h.
- `colab url -s NAME` attaches the web UI to a CLI session (for the owner to look) without a new allocation.
- The <=4-unit acceptance plan is expected after Q1 and independent review (CPU <=20 min, then L4 <=30 min).
  `final-only` is the default profile: it is required for S1 and terminal-only
  exports, keeps every operational safety gate, and adds an exactly 64 MiB
  hash-verified final download with a recorded bytes/second lower bound. It
  records the 600 MiB busy snapshot, exact resume, and checkpoint-overhead rows
  as `NOT_REQUIRED`. S1 needs a current-source-bound `final-only` PASS uploaded
  as hash-bound `cli-acceptance-result.json`; no plan or different-profile
  result admits it. `full-state` is required before mid-run full snapshots or
  exact-resume claims, retains shared operational gates, and adds the three
  full-state gates; the 64 MiB download is `NOT_REQUIRED` there. Run the reviewed
  profile through `launch.sh acceptance --profile final-only`; until it passes,
  no new research allocation is qualified. Run 6 at `fe0fd684` was `INCOMPLETE`
  and cannot qualify either profile.

## GPUs (2026-10-18)

| GPU | units/h | notes |
|---|---|---|
| L4 | ~1.7 (we plan with 2.0) | 12 vCPU; N=3 ~7.5k decisions/s; default for Q-series training |
| A100-40GB / A100-80GB | ~5.4 / ~7.5 | N=3 ~13.5k decisions/s; learner fp32 bs2 ~83 upd/s isolated |
| H100 | unmeasured (expect > A100-80) | scarce; when obtained, chain approved queued work, never idle indefinitely |
| G4 = RTX PRO 6000 Blackwell, sm_120, ~96 GB | ~8.7 | substitute when H100 is out; needs torch **cu128** (cu126 = "no kernel image") |
| T4 | ~1.2 | not used |

Google publishes no rate table: rates are third-party measurements (mccormickml.com, 2026-03-10) and fluctuate; our
2026-10-18 `colab usage` read 6.84/h for one A100 + one L4, consistent. The supervisor treats a manifest rate as a
conservative pricing ceiling: observed account debit includes other notebook spend and provider lag, not exact job
billing. Trust fresh `colab usage` over this table.
After allocation the supervisor subtracts reconciled other-assignment rates and requires three stable positive
provider-rate reads five seconds apart within 120 seconds (stability 0.01 units/hour). Observed must be at most
the priced ceiling plus 0.05 units/hour; unknown, unstable, or over-ceiling calibration stops as INCOMPLETE.
Later drift uses that observed rate; admission, reservations, and caps keep the priced ceiling. Acceptance uses
CPU 0.5 and L4 2.0 units/hour pricing defaults, overridable before allocation; measured rates are outputs.
Pro+ FAQ: continuous execution up to 24 h with sufficient units; idle runtimes time out; with a positive balance the
free-tier bans on SSH/remote control/bypassing the notebook UI do not apply (abuse bans always apply). Units expiry
is not stated in the FAQ.

Pick the wheel from `nvidia-smi --query-gpu=compute_cap` (>= 10 -> cu128). Colab's kernel is Python 3.13; jobs
build a uv Python 3.12 venv for the numpy 1.26.4 parity pins.

## Credit hygiene

- Check `colab usage` before allocation and after verified stop; the per-job receipt records the latter. Report
  balance and burn to the owner in Pacific time.
- Release anything idle with no queued approved work; ask before releasing a runtime the owner said to keep.
- No automatic purchases, no allocation loops, no retries on allocation failure (report instead).

## Legacy: browser-pane notebooks + Drive queue (running work only)

Used by the 2026-10-18 Q1 run (research/colab_queue + notebooks/colab_queue_worker*.ipynb). Facts still needed
to babysit it:
- Drive for desktop on the Mac: `$HOME/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive`
  (= /content/drive/MyDrive). Shared dirs are pre-created on the Mac by `q enqueue` (Drive duplicates concurrent
  creates as "x (1)").
- Each NEW notebook runtime needs the owner's "Connect to Google Drive" click (popup opens only from their click;
  check `tabs_context`, ask for Cmd+Shift+B if the pane is hidden). The prompt times out after ~2 min with
  "mount failed": just re-run the mount cell; a recent grant usually mounts without a new click.
- Colab disconnects notebooks when no CELL executes; worker notebooks end with a blocking keep-alive cell. Cells
  run in notebook ORDER; run cells individually via `document.querySelectorAll('colab-run-button')[i].click()`.
- A disconnected tab shows "Reconnect" at top right (gives a NEW VM: re-mount, re-run cells).
- Never type multi-line code into cells; commit scripts and fetch by SHA.
- Release: `from google.colab import runtime; runtime.unassign()` or Runtime > Disconnect and delete runtime;
  then confirm with `colab --auth oauth2 sessions`.

See `colab-jobs` for job manifests, monitoring and recovery.
