import json
import os
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[3]
child = Path(__file__).with_name("seed_repro_child.py")
env = os.environ.copy()
env.update({
    "PYTHONPATH": str(root),
    "SNAKE_DQN_DEVICE": "cpu",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
})

def run(mode):
    out = []
    for _ in range(2):
        p = subprocess.run(
            ["/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python", str(child), "123", mode],
            cwd=root,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        out.append(json.loads(p.stdout))
    return out

result = {
    "pqn_seed_without_torch_seed": run("random"),
    "pqn_seed_with_explicit_torch_seed": run("manual"),
}
result["without_torch_seed_hashes_equal"] = (
    result["pqn_seed_without_torch_seed"][0]["initial_model_tensor_sha256"]
    == result["pqn_seed_without_torch_seed"][1]["initial_model_tensor_sha256"]
)
result["with_torch_seed_hashes_equal"] = (
    result["pqn_seed_with_explicit_torch_seed"][0]["initial_model_tensor_sha256"]
    == result["pqn_seed_with_explicit_torch_seed"][1]["initial_model_tensor_sha256"]
)
print(json.dumps(result, indent=2, sort_keys=True))
