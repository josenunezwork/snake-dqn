"""Lock the AppConfig dataclass <-> pydantic schema field correspondence.

_schema_to_appconfig() maps each *Schema into its dataclass via model_dump(), so
the field names MUST match 1:1 per section. If someone adds a field to one side
only, this fails loudly here instead of silently dropping a YAML value at load
time (the dead-config failure mode this project has hit before).
"""

import dataclasses as dc

import pytest

from src.core import config_loader as cl
from src.core import game_config as gc

SECTIONS = [
    ("game", gc.GameSettings, cl.GameSettingsSchema),
    ("network", gc.NetworkSettings, cl.NetworkSettingsSchema),
    ("training", gc.TrainingSettings, cl.TrainingSettingsSchema),
    ("rewards", gc.RewardSettings, cl.RewardSettingsSchema),
    ("apex", gc.ApexSettings, cl.ApexSettingsSchema),
    ("curriculum", gc.CurriculumSettings, cl.CurriculumSettingsSchema),
    ("checkpoint", gc.CheckpointSettings, cl.CheckpointSettingsSchema),
]


@pytest.mark.parametrize("name,dataclass_cls,schema_cls", SECTIONS)
def test_config_schema_dataclass_field_parity(name, dataclass_cls, schema_cls):
    dataclass_fields = {f.name for f in dc.fields(dataclass_cls)}
    schema_fields = set(schema_cls.model_fields.keys())
    assert dataclass_fields == schema_fields, (
        f"{name}: dataclass/schema field drift — "
        f"only in dataclass: {sorted(dataclass_fields - schema_fields)}, "
        f"only in schema: {sorted(schema_fields - dataclass_fields)}"
    )
