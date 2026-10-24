#!/usr/bin/env bash
# The runtime watchdog is armed before this lease-fenced dependency setup.
set -euo pipefail
ROOT=${1:?usage: bootstrap.sh ROOT REPO_RELATIVE_SCRIPT}
SCRIPT=${2:?usage: bootstrap.sh ROOT REPO_RELATIVE_SCRIPT}
case "$ROOT" in /content/snake-cli/*) ;; *) echo "unsafe root: $ROOT" >&2; exit 2;; esac
case "$SCRIPT" in /*|*'..'*) echo 'script must be repo-relative' >&2; exit 2;; esac
REPO="$ROOT/repo"
VENV="$ROOT/.venv"
REQUIREMENTS=${COLAB_CLI_REQUIREMENTS:-requirements.txt}
case "$REQUIREMENTS" in /*|*'..'*) echo 'unsafe requirements path' >&2; exit 2;; esac
[ -f "$REPO/$SCRIPT" ] || { echo 'pinned job script missing' >&2; exit 2; }
SETUP_STARTED=$(python3 -c 'import time; print(time.monotonic())')
RETENTION_CACHE=$(python3 - "$ROOT/spec.json" <<'PY'
import json,sys
value=json.load(open(sys.argv[1])).get('retention_cache')
print(json.dumps(value,sort_keys=True,separators=(',',':')) if value else '')
PY
)
CACHE_STATUS=DISABLED
if [ -n "$RETENTION_CACHE" ]; then
  [ "${SNAKE_COLAB_CANARY:-}" != 1 ] || { echo 'canary roots cannot retain caches' >&2; exit 2; }
  CACHE_KEY=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["key"])' <<<"$RETENTION_CACHE")
  [[ "$CACHE_KEY" =~ ^[0-9a-f]{64}$ ]] || { echo 'invalid cache key' >&2; exit 2; }
  CACHE_ROOT="/content/snake-cli-immutable-cache/$CACHE_KEY"
  if [ -e "$CACHE_ROOT" ]; then
    python3 "$REPO/research/colab_cli/cache_environment.py" restore "$CACHE_ROOT" "$VENV" "$RETENTION_CACHE"
    CACHE_STATUS=HIT
  else
    CACHE_STATUS=MISS
  fi
fi
if [ "$CACHE_STATUS" != HIT ]; then
  python3 -m pip install -q uv
  python3 -m uv python install 3.12
  python3 -m uv venv --python 3.12 --managed-python --seed "$VENV"
fi
PY="$VENV/bin/python"
INDEX=https://download.pytorch.org/whl/cpu
CUDA_EXPECTED=cpu
if command -v nvidia-smi >/dev/null 2>&1; then
  CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  INDEX=https://download.pytorch.org/whl/cu126
  CUDA_EXPECTED=12.6
  if "$PY" -c "import sys; sys.exit(0 if float('$CAP') >= 10 else 1)"; then
    INDEX=https://download.pytorch.org/whl/cu128
    CUDA_EXPECTED=12.8
  fi
fi
if [ "$CACHE_STATUS" != HIT ]; then
  "$PY" -m pip install -q --index-url "$INDEX" torch==2.9.1 numpy==1.26.4
  printf 'torch==2.9.1\nnumpy==1.26.4\n' > "$ROOT/.bootstrap-constraints.txt"
  if [ -n "$REQUIREMENTS" ]; then
    [ -f "$REPO/$REQUIREMENTS" ] || { echo 'requirements file missing' >&2; exit 2; }
    "$PY" -m pip install -q -c "$ROOT/.bootstrap-constraints.txt" -r "$REPO/$REQUIREMENTS"
  fi
fi
if [ "$CUDA_EXPECTED" = cpu ]; then
  "$PY" -c "import torch; assert torch.version.cuda is None; print(torch.__version__, 'cpu')"
else
  "$PY" -c "import torch; assert torch.cuda.is_available(); assert torch.version.cuda.startswith('$CUDA_EXPECTED'); print(torch.__version__,torch.version.cuda,torch.cuda.get_device_capability(0))"
fi
# Cold qualification probes report their measured pins too. Other GPU families
# keep the existing fallback/source-delivery behavior and do not enter this path.
if [ -n "$RETENTION_CACHE" ] || [ "$SCRIPT" = research/colab_cli/retention_acceptance_probe.sh ] || [ "${COLAB_RETENTION_CAPTURE_PINS:-}" = 1 ]; then
  "$PY" - "$ROOT" "$REPO" "$REQUIREMENTS" "$RETENTION_CACHE" "$CACHE_STATUS" "$SETUP_STARTED" <<'PY'
import hashlib,importlib.metadata,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path
root,repo,requirements,raw,status,started=sys.argv[1:]
import torch
sys.path.insert(0,repo)
from research.colab_cli.retention import runtime_source_identity
source_sha256,source_origin=runtime_source_identity(Path(root),Path(repo))
contract=json.loads(raw) if raw else {}
toolchain={}
for key,command in [('nvcc',['nvcc']),('cxx',shlex.split(os.environ.get('CXX','c++')))]:
    toolchain[key]={'path':shutil.which(command[0]),'version':subprocess.check_output([*command,'--version'],text=True)}
actual={
 'python_abi':sys.implementation.cache_tag,
 'torch_cuda':str(torch.__version__)+'|'+str(torch.version.cuda),
 'installed_distributions_sha256':hashlib.sha256(json.dumps(sorted((str(d.metadata['Name']).lower(),d.version) for d in importlib.metadata.distributions())).encode()).hexdigest(),
 'dependency_lock_sha256':hashlib.sha256((Path(repo)/requirements).read_bytes()).hexdigest(),
 'source_sha256':source_sha256,
 'compiler':hashlib.sha256(json.dumps(toolchain,sort_keys=True).encode()).hexdigest(),
 'compiler_flags':json.dumps({'CXXFLAGS':os.environ.get('CXXFLAGS',''),'TORCH_CUDA_ARCH_LIST':os.environ.get('TORCH_CUDA_ARCH_LIST','')},sort_keys=True,separators=(',',':')),
 'sm':'%d.%d'%torch.cuda.get_device_capability(0),
}
if contract and actual != contract['pins']:
    raise RuntimeError('retention cache runtime pins differ from frozen canary')
value={'schema':'snake-dqn-retention-bootstrap/v1','cache_key':contract.get('key'),'cache_status':status,'pins':actual,'source_origin':source_origin,'setup_seconds':time.monotonic()-float(started)}
Path(root,'cache-evidence.json').write_text(json.dumps(value,sort_keys=True)+'\n')
PY
fi
if [ -n "$RETENTION_CACHE" ]; then
  if [ "$CACHE_STATUS" = MISS ]; then
    python3 "$REPO/research/colab_cli/cache_environment.py" publish "$CACHE_ROOT" "$VENV" "$RETENTION_CACHE"
  fi
  mkdir -p "$ROOT/torch_extensions"
  export TORCH_EXTENSIONS_DIR="$ROOT/torch_extensions"
  SITE_PACKAGES=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
  { printf '%s\n' "$REPO"; printf '%s\n' "import importlib; importlib.import_module('research.colab_cli.extension_cache').install()"; } > "$SITE_PACKAGES/snake_retention_extension_cache.pth"
  export COLAB_RETENTION_CACHE_KEY="$CACHE_KEY"
  # Include snapshot publication and hook setup in the setup comparison.
  python3 - "$ROOT/cache-evidence.json" "$SETUP_STARTED" <<'PY'
import json,sys,time
from pathlib import Path
p=Path(sys.argv[1]); value=json.loads(p.read_text()); value['setup_seconds']=time.monotonic()-float(sys.argv[2]); p.write_text(json.dumps(value,sort_keys=True)+'\n')
PY
fi
if [ "${SNAKE_COLAB_CANARY:-}" = 1 ]; then
  SITE_PACKAGES=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
  { printf '%s\n' "$REPO"; printf '%s\n' "import importlib; importlib.import_module('research.colab_cli.canary_observer').install()"; } > "$SITE_PACKAGES/snake_colab_canary.pth"
  "$PY" -c "import importlib.util; from pathlib import Path; s=importlib.util.find_spec('research.colab_cli.canary_observer'); assert s and s.origin and Path(s.origin).resolve().is_relative_to(Path('$REPO').resolve())"
fi
cd "$REPO"
export COLAB_CLI_ROOT="$ROOT"
export COLAB_INPUT_DIR="${COLAB_INPUT_DIR:-$ROOT/inputs}"
export PATH="$VENV/bin:$PATH"
exec bash "$REPO/$SCRIPT"
