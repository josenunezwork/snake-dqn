"""Reproduce advisory-mask escape and unrealizable horizontal augmentation."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from src.model.raster_network import raster_tensors_from_obs
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.featurizer import ObsInputs, build_observations
from src.training.pqn_trainer import flip_augment

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
started=time.monotonic()
rng=np.random.default_rng(9147)
sim=BatchSim(BatchSimConfig(num_envs=16,num_snakes=6,mechanics_version=2,gamma=.997),seeds=range(100,116),train_mode=True)
found=None
for step in range(1500):
    mask=sim.get_action_mask()
    actions=np.zeros((sim.E,sim.S),dtype=np.int64)
    for env in range(sim.E):
        for slot in range(sim.S):
            choices=np.flatnonzero(mask[env,slot])
            actions[env,slot]=int(rng.choice(choices)) if choices.size else int(rng.integers(6))
    sim.step(actions)
    trapped=sim.get_alive() & ~sim.get_action_mask().any(axis=2)
    if trapped.any():
        env,slot=map(int,np.argwhere(trapped)[0]); found=step+1,env,slot;break
assert found is not None
frame,env,slot=found
mask=sim.get_action_mask()
other_actions=np.zeros((sim.E,sim.S),dtype=np.int64)
for e in range(sim.E):
    for s in range(sim.S):
        choices=np.flatnonzero(mask[e,s])
        other_actions[e,s]=int(rng.choice(choices)) if choices.size else 0
outcomes=[]
for action in range(6):
    branch=copy.deepcopy(sim)
    acts=other_actions.copy();acts[env,slot]=action
    branch.step(acts)
    reward=float(branch.get_reward()[env,slot])
    outcomes.append({'action':action,'reward':reward,'alive':bool(branch.get_alive()[env,slot]),
        'done':bool(branch.get_done()[env,slot]),'next_mask':branch.get_action_mask()[env,slot].astype(int).tolist(),
        'current_target_if_marked_trapped':reward if branch.get_done()[env,slot] else reward+.997*(-3)})
assert any(r['alive'] and all(r['next_mask']) for r in outcomes)
trap={'frame':frame,'env':env,'slot':slot,'length':int(sim.get_lengths()[env,slot]),
      'head':sim.get_heads()[env,slot].tolist(),'direction':int(sim.get_directions()[env,slot]),
      'current_mask':mask[env,slot].astype(int).tolist(),'previous_reward':float(sim.get_reward()[env,slot]),
      'outcomes':outcomes}

def observation(mirror):
    col=lambda x:144-x if mirror else x
    bodies=np.array([[[[col(30),40],[col(30),41],[col(30),42]]]],dtype=np.int64)
    food=np.array([[[col(20),20],[col(100),20]]],dtype=np.int64)
    inputs=ObsInputs(heads=bodies[:,:,0],bodies=bodies,body_len=np.array([[3]]),lengths=np.array([[3]]),
        alive=np.array([[True]]),heading=np.array([[0]]),boost_frames=np.zeros((1,1),dtype=np.int64),
        frames_since_food=np.zeros((1,1),dtype=np.int64),boosting=np.zeros((1,1),dtype=bool),
        food_cells=food,food_mass=np.ones((1,2)),food_is_corpse=np.zeros((1,2),dtype=bool),
        grid_w=145,grid_h=83,max_snakes=1,starvation_max=500,max_length=150,min_boost_length=5,
        boost_cost_frames=3,frame=np.array([10]),max_frames=5000)
    return raster_tensors_from_obs(build_observations(inputs,mask=np.ones((1,1,6),dtype=bool)))

original=observation(False);real_mirror=observation(True)
tactical,strategic,scalars,_=flip_augment(original['tactical'],original['strategic'],original['scalars'],torch.tensor([1]))
indices=torch.nonzero((scalars-real_mirror['scalars']).abs()>1e-6,as_tuple=False)[:,1].tolist()
flip={'tactical_equal':bool(torch.equal(tactical,real_mirror['tactical'])),
      'strategic_equal':bool(torch.equal(strategic,real_mirror['strategic'])),
      'scalar_diff_indices':indices,'augmented_world_x':float(scalars[0,23]),
      'real_mirrored_world_x':float(real_mirror['scalars'][0,23])}
assert flip['tactical_equal'] and flip['strategic_equal'] and indices==[23]
files=['src/training/pqn_trainer.py','src/simd_env/batch_sim.py','src/simd_env/featurizer.py','src/model/raster_network.py']
result={'status':'reproduced','command':sys.argv,'elapsed_seconds':time.monotonic()-started,
        'source_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in files},
        'advisory_mask_escape':trap,'flip_realizability':flip,
        'scope':'existing behavior unchanged by sampler campaign; target value is explicit current trapped-branch formula, not a new training arm'}
(OUT/'audit_reproductions.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
