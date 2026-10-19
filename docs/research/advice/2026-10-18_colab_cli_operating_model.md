# Colab operations: adopt CLI control after Q1, retain governed dispatch

2026-10-18 · Astra operations/ML-infrastructure advice · repository snapshot `017b589b76b5a051a00365c9067e548f5623d4c0`; upstream v0.7.4 `66d7963611bcce2d272488e982ad38e519c062f0`

**Adopt the official CLI behind one owner-controlled Mac supervisor. Claude plans/reviews; Sol builds adapters; neither independently allocates workers. Keep tonight’s Drive workers unchanged. Subsequently remove runtime Drive mounts through CLI uploads/downloads, retaining admission, fencing and receipts.** This supports the existing goal: a bare network stronger than released B8-s3 on terminal-hero H5000 full-horizon mass, with survival guards; strict gates and serving qualification remain Mac-only ([STATUS](../../STATUS.md)).

Advice only: no launch, registration, queue mutation or spending authorization. Acceptance thresholds below are prospective.

**What changes now.** CLI authentication is reusable; runtime `drivemount` and `auth` remain human-interactive, with 600-second CLI timeouts. A longer timeout does not remove consent. Correct the [earlier findings](../colab_agent_access_2026-10-18.md): background processes alone are insufficient; execute a kernel supervisor that waits for actual work. Backend kernel activity maintains liveness; no keep-alive daemon is needed ([official authentication/liveness documentation](https://github.com/googlecolab/google-colab-cli/blob/main/docs/04_automation_and_utility.md)).

**Transition, in dependency order**

| Order | Owner and deliverable | Admission/stop rule |
|---|---|---|
| 0: tonight | Claude verifies Q0b receipts and Q1 admission; preserve current notebook pins, mounts and claims. | No CLI migration or new test VM during Q1. |
| 1: prepare | Claude freezes adapter contract, budgets and fault tests; Sol implements a single supervisor plus CLI transport. | Mac real-entrypoint preflight, shipped input hashes, deterministic parallel evaluation; no live spending. |
| 2: qualify | Sol runs the isolated tests below; Claude independently audits artifacts and release. | After Q1; separate approval; one runtime, no automatic retry. |
| 3: switch | Route only newly admitted jobs through CLI; retain Drive fallback until completion/recovery passes. | Never let both transports own the same job; rollback between jobs. |

**Budget discrepancy must be resolved before dispatch.** Owner-supplied 20 × 85 minutes = **28.33 GPU-hours, 56.67 units at 2 u/h**, excluding setup/export and 100 CPU evaluation jobs. This exceeds the registered 24 GPU-hour combined ceiling and its 19.2-hour usable allocation; parallelism does not fix aggregate hours. [Q0/Q1 Amendment A3](../../../research/q0q1_20261017/PREREGISTRATION.md#amendment-a3--inclusive-q0-envelope-and-operational-lag-tolerance-2026-10-18) preserves those limits and Q1’s 60-unit training cap. It sets combined units to T+135, usable T+100, where T is terminal inclusive Q0 cost. Require a measured compliant forecast or an explicit prospective amendment, never silently drop seeds. Measure CPU debit; include every failure and idle minute.

**Owner setup once, in Mac Terminal**

1. Install an isolated pinned CLI using the [official uv installer](https://docs.astral.sh/uv/getting-started/installation/):

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh
"$HOME/.local/bin/uv" tool install --python 3.12 'google-colab-cli==0.7.4'
umask 077
mkdir -p "$HOME/.config/colab-cli/snake"
"$HOME/.local/bin/colab" version
"$HOME/.local/bin/colab" --auth oauth2 whoami
```

2. Expect version **0.7.4**, then an authorization URL and code prompt. Open that URL, choose the Google account holding Pro+, complete Google’s consent/2FA, copy the displayed authorization code, and paste it **into Terminal**, not chat. The [bundled OAuth client](https://github.com/googlecolab/google-colab-cli/blob/66d7963611bcce2d272488e982ad38e519c062f0/src/colab_cli/auth.py) needs no Cloud project or billing setup. Confirm the returned email matches your paid account.

3. Verify reusable access without allocating anything:

```sh
chmod 600 "$HOME/.config/colab-cli/token.json"
"$HOME/.local/bin/colab" --auth oauth2 usage
"$HOME/.local/bin/colab" --auth oauth2 sessions
```

Expect current balance, aggregate burn and existing assignments; compare with Colab’s resource panel, not the stale 1,240-unit starting balance. Repeat `usage` in a new Terminal: no browser prompt should appear. Authentication expiry/revocation can require another owner login; “once” is not a lifetime guarantee.

4. Keep the Mac plugged in, lid open and online while unattended jobs run. In **System Settings → Battery → Options**, [prevent automatic sleep on power adapter when the display is off](https://support.apple.com/en-ie/guide/mac-help/mchle41a6ccd/mac). Sol should install a supervised local service with sleep prevention. Both agents use that local service under your macOS account; credentials never enter prompts, Git or runtime uploads. No per-VM Drive clicks are needed for the proposed transport.

**What Sol builds first**

Use explicit `--auth oauth2`, `COLAB_CLI_ASSIGN_RETRIES=0`, unique session names and per-lane `--config` files. These isolate session metadata, **not credentials/settings/history** ([source](https://github.com/googlecolab/google-colab-cli/blob/66d7963611bcce2d272488e982ad38e519c062f0/src/colab_cli/common.py)). One broker serializes mutations. Preserve manifest/source/input hashes, generation fencing, 15-second heartbeats and create-only receipts. `exec -f` sends local Python into a persistent kernel; set an explicit execution timeout, reset job state, and verify terminal receipts: [kernel errors need not produce nonzero CLI exit](https://github.com/googlecolab/google-colab-cli/blob/66d7963611bcce2d272488e982ad38e519c062f0/src/colab_cli/commands/execution.py#L258).

Upload pinned bundles; run in local scratch with Python 3.12 and the qualified device wheel: cu126 for L4/A100/H100, cu128 for substituted Blackwell. Save consistent optimizer/replay/actor/target/RNG state plus counters; weights-only recovery is not exact resume. Use bounded asynchronous export of immutable snapshots, hashes and completion markers to Mac storage, then Drive backup; measure overhead. Do not serialize live mutable state concurrently. No acknowledged export means no durable checkpoint. If Mac connectivity fails, a runtime deadline/lease watchdog must stop spending; recovery loses work since the last verified export. Independent cloud storage is a later project if this limitation is unacceptable.

[Transfers buffer whole files](https://github.com/googlecolab/google-colab-cli/blob/66d7963611bcce2d272488e982ad38e519c062f0/src/colab_cli/contents.py); enforce subprocess timeouts and download-to-temporary/hash/atomic-rename. Reconnect through file telemetry, never blindly resubmit `exec` behind a busy kernel. Client timeout is not proof of remote termination.

**Cheap acceptance tests before trust**

Propose **4 units maximum including 20% reserve; 3.2 usable** (0.32% of owner-supplied 1,240): CPU ≤20 minutes, then L4 ≤30 minutes (~1 unit), first cap wins. Price CPU before admission; unknown pricing blocks allocation. Test fresh-VM credential reuse/refresh; busy-kernel full-size snapshot upload/download with matching hashes; >60-second silence with explicit timeout; browser attachment via `url` without a second allocation; client loss/reconnect without duplicate execution; deliberate child failure and corrupt export rejection; exact resume against uninterrupted execution; and watchdog VM unassignment while the Mac is disconnected, not merely trainer exit. Require ≤2% measured checkpoint overhead, verified artifacts, and backend session disappearance after stop. Missing evidence means INCOMPLETE, not rollout.

**Credit and policy guardrails**

Count account-wide assignments, including notebooks; cap concurrency at three. Maintain positive balance, study reserves and per-session wall/unit limits; forbid automatic purchases, allocation loops and retries. Default L4/CPU. Retain an obtained H100 by chaining approved useful work within a priced retention lease; scarcity cannot justify indefinite idle billing. `run` suits disposable probes; use `new/exec/download/stop` for artifact-bearing work, verifying release independently: [cleanup can swallow unassign failures](https://github.com/googlecolab/google-colab-cli/blob/66d7963611bcce2d272488e982ad38e519c062f0/src/colab_cli/commands/run.py#L461).

The [FAQ](https://research.google.com/colaboratory/faq.html) offers Pro+ continuous execution up to 24 hours, not guaranteed hardware/lifetime. Paid positive-balance accounts avoid the free-tier SSH/distributed-worker restrictions; all-tier proxy, account-limit evasion and abuse prohibitions remain. The [Terms](https://research.google.com/colaboratory/tos_v5.html) prohibit account sharing/reselling and unauthorized access. Keep automation owner-controlled; no credential distribution, fake activity or capacity hoarding. Official CLI access does not waive these rules.
