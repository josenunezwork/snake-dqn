"""The owner-authorized A1 exception must not widen generic acceptance jobs."""

from copy import deepcopy

import pytest

from tools.ops.dispatch import A1_AUTHORITY_SHA256, Refused, require_resume_policy


def a1() -> dict:
    return {
        "resume_exemption": {
            "kind": "acceptance",
            "reason": "PLAN_A100 section3",
            "authority_sha256": A1_AUTHORITY_SHA256,
        },
        "timeout_minutes": 120,
        "script": "research/gpu_sim_spike/colab.sh",
        "env": {"GPU_SIM_SPIKE_ENTRYPOINT": "a1-ruler"},
        "requirements": {"gpu": "A100", "allowed_gpus": ["A100"]},
        "budget": {"unit_cap": 20},
        "a1": {
            "shapes": [6, 256, 1024],
            "mixes": ["scripted", "frozen", "mixed"],
            "repeats": 3,
            "ticks_per_repeat": 8,
        },
    }


def test_specific_a1_acceptance_envelope() -> None:
    require_resume_policy(a1())
    manifest = a1()
    manifest["timeout_minutes"] = 121
    with pytest.raises(Refused, match="<= 120"):
        require_resume_policy(manifest)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("resume_exemption", "authority_sha256", "0" * 64),
        ("resume_exemption", "kind", "short-canary"),
        ("env", "GPU_SIM_SPIKE_ENTRYPOINT", "benchmark"),
        ("requirements", "gpu", "H100"),
        ("requirements", "allowed_gpus", ["A100", "H100"]),
        ("budget", "unit_cap", 21),
        ("a1", "shapes", [6]),
        ("a1", "ticks_per_repeat", 9),
    ],
)
def test_a1_exception_refuses_wrong_identity(section: str, key: str, value: object) -> None:
    manifest = deepcopy(a1())
    manifest[section][key] = value
    with pytest.raises(Refused):
        require_resume_policy(manifest)


def test_generic_acceptance_remains_sixty_minutes() -> None:
    with pytest.raises(Refused, match="<= 60"):
        require_resume_policy(
            {
                "timeout_minutes": 120,
                "resume_exemption": {"kind": "acceptance", "reason": "generic"},
            }
        )
