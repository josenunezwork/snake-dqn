"""Create-only prospective F5 manifest templates; no admission or dispatch authority."""
import copy
import hashlib
import json
from pathlib import Path
from research.bl_ppo.exports import export_json
from tools.ops.f5_unit_guard import repair_guard

ROOT = Path('/Users/josenunez/Projects/ml/snake-dqn-f5w')
OUT = Path(__file__).parent
rows = []
for source in sorted((ROOT / 'research/bl_residual/ledger').glob('F5_*/*manifest*.json')):
    value = json.loads(source.read_text())
    if value.get('budget', {}).get('unit_cap') != 9999:
        continue
    if value.get('f5', {}).get('technical_guards'):
        successor = repair_guard(value)
    else:
        draft = value.get('draft_only', {})
        assert value.get('scientific_admission') is False
        assert draft.get('no_scientific_fits') is True and draft.get('no_outcome_look') is True
        assert draft.get('status', '').startswith('DRAFT_ONLY_NOT_DISPATCHABLE')
        assert value['budget']['reserve_units'] == 0
        successor = copy.deepcopy(value)
        successor['budget']['unit_cap'] = 100.0
    target = OUT / 'prospective-templates' / source.parent.name / source.name
    export_json(target, successor)
    rows.append({'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'successor': str(target), 'successor_sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'status': 'TEMPLATE_ONLY_REQUIRES_FRESH_IDENTITIES_PREFLIGHT_RESEAL_REVIEW'})
export_json(OUT / 'PROSPECTIVE_F5_TEMPLATES.json', {'status': 'CORRECTED_PROSPECTIVE_COPIES_NO_DISPATCH', 'templates': rows, 'historical_originals_unchanged': True})
print('prospective templates', len(rows))
