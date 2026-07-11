"""Vectorized, cell-exact batch snake simulator (blueprint §4.1).

Dynamics-only NumPy core: ``E`` environments x ``S`` snakes advanced one frame
per :meth:`BatchSim.step`, cell-exact and parity-faithful to the live game's v1
and v2 mechanics (default v2). The featurizer/raster is a separate task; this
package exposes world state via read-only accessors.
"""

from src.simd_env.batch_sim import (
    DEATH_BODY,
    DEATH_HEAD,
    DEATH_NONE,
    DEATH_SELF,
    DEATH_WALL,
    BatchSim,
    BatchSimConfig,
)
from src.simd_env.featurizer import (
    ObsInputs,
    build_observations,
    expand_tactical,
    obs_inputs_from_batch_sim,
    to_network_input,
)
from src.simd_env.rng import EnvRng, make_env_rngs

__all__ = [
    "BatchSim",
    "BatchSimConfig",
    "EnvRng",
    "make_env_rngs",
    "DEATH_NONE",
    "DEATH_WALL",
    "DEATH_SELF",
    "DEATH_HEAD",
    "DEATH_BODY",
    "ObsInputs",
    "build_observations",
    "expand_tactical",
    "to_network_input",
    "obs_inputs_from_batch_sim",
]
