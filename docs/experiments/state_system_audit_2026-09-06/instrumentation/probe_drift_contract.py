"""Pure, bounded probes of the observation-drift instrument's acceptance scope."""
import hashlib
import json
from pathlib import Path
import sys
import warnings
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.scripts.obs_histogram_diff import compare_observation_sets

cases={
    'identical_finite_control':(np.tile([[0.,0.],[1.,1.]],(10,1)),np.tile([[0.,0.],[1.,1.]],(10,1))),
    'changed_joint_same_marginals':(np.tile([[0.,0.],[1.,1.]],(10,1)),np.tile([[0.,1.],[1.,0.]],(10,1))),
    'both_nan':(np.full((20,2),np.nan),np.full((20,2),np.nan)),
    'both_inf':(np.full((20,2),np.inf),np.full((20,2),np.inf)),
}
results={}
for name,(a,b) in cases.items():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        r=compare_observation_sets(a,b)
    results[name]={'passed':r['passed'],'max_ks':r['max_ks'],'flagged':r['flagged'],
        'input_all_finite':bool(np.isfinite(a).all() and np.isfinite(b).all()),
        'warnings':sorted({str(w.message) for w in caught})}
    if name=='changed_joint_same_marginals':
        results[name]['fraction_x_equals_y_a']=float((a[:,0]==a[:,1]).mean())
        results[name]['fraction_x_equals_y_b']=float((b[:,0]==b[:,1]).mean())
result={'source_sha256':hashlib.sha256((ROOT/'src/scripts/obs_histogram_diff.py').read_bytes()).hexdigest(),
    'results':results,'scope':'Synthetic acceptance probes, no training or live nonfinite observation claimed. Marginal KS cannot validate joint relations, and the current function does not reject nonfinite observations.'}
p=Path(__file__).with_suffix('.json');p.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps(result,indent=2,allow_nan=False))
