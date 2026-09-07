import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch

from src.core.config_loader import resolve_yaml_device
from src.scripts import train_pqn
from src.training.pqn_trainer import PQNConfig, PQNTrainer

config_path = Path(__file__).with_name("input_hardware_params.yaml")
overrides = train_pqn._load_config_overrides(str(config_path))
resolved = PQNConfig(**overrides)
trainer = PQNTrainer(resolved, device=torch.device("cpu"))
sim_cfg = asdict(trainer.sim.cfg)

# A mocked auto detector makes the device precedence observable without using GPU.
orig_get_device = train_pqn.DeviceManager.get_device
try:
    train_pqn.DeviceManager.get_device = classmethod(lambda cls: torch.device("cuda"))
    mocked_auto = str(train_pqn._resolve_device(None))
finally:
    train_pqn.DeviceManager.get_device = orig_get_device

cli_cpu = str(train_pqn._resolve_device("cpu"))

result = {
    "input_yaml": str(config_path),
    "hardware_device_from_yaml": resolve_yaml_device(str(config_path)),
    "train_pqn_overrides": overrides,
    "resolved_pqn_config": asdict(resolved),
    "effective_batchsim_config": sim_cfg,
    "mapped_shared_game_fields": {
        k: sim_cfg[k]
        for k in ("num_snakes", "mechanics_version", "arena_type")
    },
    "unmapped_shared_game_fields_observed_defaults": {
        k: sim_cfg[k]
        for k in (
            "game_width",
            "game_height",
            "segment_size",
            "wall_thickness",
            "initial_food",
            "max_food",
            "min_boost_length",
            "boost_length_cost_frames",
            "frame_rate",
        )
    },
    "hardware_precedence_probe": {
        "yaml_requests": resolve_yaml_device(str(config_path)),
        "mocked_auto_without_cli": mocked_auto,
        "explicit_cli_cpu": cli_cpu,
    },
}
print(json.dumps(result, indent=2, sort_keys=True))
