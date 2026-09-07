import hashlib
import json
import sys

import torch

from src.training.pqn_trainer import PQNConfig, PQNTrainer

seed = int(sys.argv[1])
manual_torch_seed = sys.argv[2] == "manual"
if manual_torch_seed:
    torch.manual_seed(seed)
cfg = PQNConfig(num_envs=1, num_snakes=2, rollout_len=1, pool_capacity=0, seed=seed)
trainer = PQNTrainer(cfg, device=torch.device("cpu"))
blob = b"".join(t.detach().cpu().contiguous().numpy().tobytes() for t in trainer.network.state_dict().values())
print(json.dumps({
    "seed": seed,
    "manual_torch_seed": manual_torch_seed,
    "initial_model_tensor_sha256": hashlib.sha256(blob).hexdigest(),
    "pqn_seed": trainer.cfg.seed,
    "torch_initial_seed": torch.initial_seed(),
}))
