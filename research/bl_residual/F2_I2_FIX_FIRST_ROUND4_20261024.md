# I2 FIX-FIRST round 4 — fixed-host HTTPS authority

This sibling implementation handoff addresses the P1 in
`snake-dqn-artifacts/t34-sol/astra-review-i2-04/snake-dqn.last-message.md`,
reviewed at `2b277ae82be119b485f3fcd2ef8a1cf7da196a63`. Work started at that
commit in the clean `snake-dqn-f2-i2` worktree on
`sol-f2-i2-fix4-20261024`. This is transport hardening and synthetic verification;
it does not execute or authorize a real F2 sealed run or R-A look.

## Review-to-fix mapping

- **Inherited GitHub CLI transport:** authority and time GETs now use stdlib
  `http.client.HTTPSConnection` directly to `api.github.com:443`. No authority
  request uses `gh api`, Git credentials, an enterprise host, or a Unix socket.
  The client does not discover proxies or follow redirects, including same-host
  redirects. Every non-200 response is unavailable evidence.
- **Override detection:** present proxy environment variables (including lowercase
  forms and `NO_PROXY`), GitHub config/host/API routing variables, and CA bundle
  environment variables cause refusal before credentials or network access.
  Standard GitHub CLI `config.yml` and `hosts.yml` are inspected for nonempty
  `http_unix_socket` settings, but never configure the HTTPS client. Unreadable,
  malformed, cyclic, excessively deep, or oversized configuration is unavailable.
  Values, credentials and raw configuration are never logged.
- **Credentials:** one process-local cached read uses `GITHUB_TOKEN` if present,
  otherwise only `gh auth token --hostname github.com`. Its environment permits
  only normal executable/home/user/temp/config-directory locations and disables
  prompts; inherited `GH_*`, `GIT_*`, proxy, trust and debug settings are absent.
  Authentication errors are generic and do not include captured output. The
  previous Git credential-helper fallback has been removed.
- **TLS and observations:** a new `SSLContext(PROTOCOL_TLS_CLIENT)` requires
  certificate verification and hostname checking. It loads explicit compiled
  default OpenSSL root paths, falling back to the OS `/etc/ssl/cert.pem` bundle
  where Python's compiled CA file is missing. It never loads env-selected roots,
  custom bundles, or requests/certifi configuration. The connected TLS peer must
  match the fixed host/context and provide a certificate and TLS version before
  the authorization header is sent. Observations record `verified_peer_host`,
  `peer_port`, `tls_verified`, `tls_version` and GitHub `request_id`, alongside
  the existing Date, local clock and elapsed-time evidence.
- **Earlier protections:** publication ancestry, exact-commit ledger and both
  sidecar bindings, activity pagination restrictions, fresh revalidation, Date
  freshness/skew/cache checks, packet identity and once-only claims remain.
  Responses are size bounded. A main-thread Unix wall-clock alarm bounds the
  entire request, including DNS/TLS/header processing; it restores the prior
  handler and refuses a pre-existing timer or non-main-thread invocation.
  Unavailable publication returns `None` (strikes); unavailable trusted time
  refuses before claim. There is no local-time or cached-evidence fallback.

## Frozen statistic

`research/bl_residual/f2.py` has no changes in this round. Both the full source
and the statistic block must match the reviewed baseline. The statistic block
from `def selected_survival(` up to `def _validate_plan(` retains SHA256
`23c86e2b3086d55a2fc26bc6a9bc5be4d0b46dced9cf11683ea12a2c466c39d1`.
Full unchanged F2 source SHA256:
`d36eaa778dceca49e46a69f8103d6ee745087ee5d0ea9526e5cab06c0a4142f8`.
No existing hash-bound registration, ledger, sidecar or receipt was edited.
Existing source-closure enforcement still applies to any future execution.

## Verification

All Python checks use `./venv/bin/python`, nice 15, CPU, serial execution and
the six thread limits below. The final validation preloads the repository's
OpenMP path to avoid the macOS conftest relaunch wrapper, leaving one pytest
process and at most one serial helper subprocess. The independent security
reviewer only inspected source.

An initial broad validation passed 449 tests (7 deselected) in 146.44 seconds,
but used the existing macOS conftest relaunch wrapper. After noticing that extra
launcher, a redundant validation attempt was stopped by terminating only this
task's two pytest PIDs, then restarted with `env DYLD_LIBRARY_PATH=...` after
`nice`. Setting DYLD before `/usr/bin/nice` does not survive macOS environment
sanitization. The final validation has one pytest PID and serial helpers, as
confirmed by process inspection; no campaign process was stopped.

```bash
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export SNAKE_DQN_DEVICE=cpu
nice -n 15 env DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib \
  ./venv/bin/python -m pytest -q tests/test_bl_residual*.py \
  -k 'not sealed_runner and not success_writes and not no_payload_or_fallback and not unpublished_input_refused'
nice -n 15 ./venv/bin/python -m black --check --line-length 100 \
  research/bl_residual/github_authority.py tests/test_bl_residual_github_authority.py
nice -n 15 ./venv/bin/python -m isort --check-only --profile black \
  research/bl_residual/github_authority.py tests/test_bl_residual_github_authority.py
nice -n 15 ./venv/bin/python -m flake8 \
  research/bl_residual/github_authority.py tests/test_bl_residual_github_authority.py
git diff --check
nice -n 15 ./venv/bin/python -m research.bl_residual.github_authority check-time
```

Regressions cover the real installed CLI writing a disposable
`http_unix_socket` config, with an actual Unix listener proving zero connections;
proxy and custom-CA environment overrides; alternate host routing; all five
redirect statuses to another host; TLS/certificate and peer-host failures;
sanitized/cached token acquisition and output secrecy; system root loading;
cyclic and ordinary two-file configuration; and deadline cleanup after stalled
headers. Earlier synthetic ancestry, ledger, funding, timing and claim tests
remain included. Seven tests that invoke the sealed F2 runner are excluded.

The live read-only `check-time` invocation returned exit 1 with
`AuthorityUnavailable: GitHub credential unavailable`: this environment has no
`GITHUB_TOKEN`, and sanitized `gh auth token --hostname github.com` returned no
credential. No trusted time or publication observation was fabricated, and no
claim was made. An operator environment with one of those supported credentials
is needed for a successful authenticated live observation; no interactive login
or credential modification was performed here.

Final direct-process validation: **449 passed, 7 deselected in 150.56 seconds**.
Black, isort, flake8 and `git diff --check` passed. Full F2 byte equality against
`2b277ae8` and the statistic-block SHA256 above passed. Independent read-only
security re-review found no remaining blocker. An earlier focused authority
selection passed 82 tests before the last normal-configuration regression was
added; that final regression is included in the broad result.

## Protected state

Untouched: live D1024 factory; tmux; `exp/mac-lease.json*`;
`t34-round0-study-01/`; `research/bl_labels/*`; `research/colab_cli/*`;
released checkpoints; serving pins; existing hash-bound artifacts. No Colab
action, real F2 sealed run, R-A execution, protected payload read or release
change occurred. No dispatch command is needed for this fix.
