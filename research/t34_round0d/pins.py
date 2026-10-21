"""SHA-256 pins of every imported or copied non-Round-0-D rule and module (reviewed bytes).

Round 0-D executes Round 0's bars, sampling, sizing, bare arm and integrity helpers by
import, plus the owner-approved relative boost floor (``research.t34_round0_amend.rule``).
Any byte change to these files after review would otherwise be inherited silently;
``check_pins`` refuses it (ratify, require_registration, tests) unless a sibling
amendment re-pins with approval.
"""

from __future__ import annotations

from pathlib import Path

PINNED: dict[str, str] = {  # noqa: E501
    "research/t34_round0/__init__.py": "b306000a7ac6a0c9a81816f0b75d770d5ea61d7b8a69b18390a691f64afae9da",  # noqa: E501
    "research/t34_round0/analyze.py": "03c704c02bfaf23492567c83010e52b95ef915f505e815b272b0c479ff64be30",  # noqa: E501
    "research/t34_round0/bank.py": "303f10e00ac124ba58e761c9ae71c0e087d9224918a46eb014ffbc31575f0024",  # noqa: E501
    "research/t34_round0/common.py": "13a1906941ccb3f180b18a031ea19c833183b0990b6bbeef14f741cba7a40514",  # noqa: E501
    "research/t34_round0/dispatch.py": "2212d24524df37647b871a7acc79c1e49d74afad956c6998e6bcdcc8ba180288",  # noqa: E501
    "research/t34_round0/history.py": "56167e875bc825ca0b82d5a1f1c814d60334b6022c7d27d8482416f8ff5c761c",  # noqa: E501
    "research/t34_round0/make_manifest.py": "a955ae70b78a6fa8d0b83f6a00c8aeccbd830601df0d85ba66e9b40c70ad2d72",  # noqa: E501
    "research/t34_round0/runner.py": "2122f70fa402f953a95929130602fdf4d9d7b79fc233295a01fe492e9c1d2b7f",  # noqa: E501
    "research/t34_round0/PREREGISTRATION_ROUND0.md": "2e7b90ddec8b406691f3cef546c8db105253170493da6fb126c75cde303e8c1c",  # noqa: E501
    "research/t34_round0/PREREGISTRATION_ROUND0_FIXES1.md": "6788ffb977a3c082a4bc754a58e6fe73ef62752b3ccba60a01a7c87385fbc5bd",  # noqa: E501
    "research/t34_round0/ERRATUM_FIXES1.md": "edb4ec787c7f5768e132168c863c1653a2b5a0ef9fdc2cb1f1e83307261ee0a0",  # noqa: E501
    "research/t34_round0/ERRATUM_INTEGRITY_SIZING.md": "99a25c202ae79c87cf7d4fad51779a54262d620703a9b7c29f15d58e9983ff5c",  # noqa: E501
    "research/t34_round0/OC_ROUND0.json": "9c9981f27a9e1421d4629b17a97c1bc58922cdbf8302a750014ecab0f1c96b47",  # noqa: E501
    "research/t34_round0/BANK_ROUND0.json": "76863477c4e47bee40c19980ca0dd01983f1bf49ea473c8b2fd4018958d0c2ae",  # noqa: E501
    "research/t34_round0_amend/__init__.py": "b163871a006000269d8828b0c300e5e62f265a5c38a93743fde7936518f48e40",  # noqa: E501
    "research/t34_round0_amend/rule.py": "e3c0d1ab9328b589539c7d7b3325d98b30acf361ac7fcc9d397a5bb244a5a135",  # noqa: E501
    "research/t34_round0_amend/AMENDMENT_BOOST_FLOOR.json": "bc20181c43fc5aaf44c3656487ba58d3c48c375cff1d8110f768bb0a8e2a8aed",  # noqa: E501
    "research/t34_exit_b8/engine.py": "ad1324eeffc0b82c6b8b8eca6fc1ffd328e8a0f8213d0d24952506cf161feb05",  # noqa: E501
    "research/t34_exit_b8/exports.py": "7475130b190f6d2a845da5e9051bb15d21ea8808faf5944b6a24aec6251d232e",  # noqa: E501
    "research/t34_exit_b8/faults.py": "f7a2e08ae05831420f3b92baba4d729894e6e0dc5362e433c75e5b58ee8739de",  # noqa: E501
    "research/t34_exit_b8/label.py": "9e9b42fd113e348df338253fe17fc7d1fc61aca7e06cced2fe82c4ca69317d75",  # noqa: E501
    "research/t34_exit_b8/policy.py": "11cb3334b7711c8d0150c68ac869db380472bd2dd293b013e3e96665f584e8e3",  # noqa: E501
    "research/t34_exit_b8/qualification.py": "ebac40768df267f3de0d2924b880d6dc9fbc641f14e3348e234db49e586ae691",  # noqa: E501
    "research/t34_exit_b8/teacher_v2.py": "53d392a65e77ae748a2c778d35e48005daedfa4755ca2f4497441f72f9116d55",  # noqa: E501
    "research/t34_exit_b8/trajectory.py": "390752220963124f2c0ce99f05d2cc5c1b47b4fe040bb486c6d69521c5bf50b9",  # noqa: E501
    "research/t34_pathb/precheck_v2.py": "277742640e81373c33bf2ef80886c85e017f5423f9d2bbc3baf2ab305971c53e",  # noqa: E501
    "research/t34_pathb/retention_history_v2.py": "47c4afec2300eafb5c85dec1c7735547660d521f09e3734fa44801b3dbb043b6",  # noqa: E501
    "research/t34_pathb/source_binding_v2.py": "e328a9a285c7ccb03f9267c12d57abdb2e95f88e992e1c648793c2e438a7068c",  # noqa: E501
    "research/t3_rollout_labels_20261015/trigger.py": "7dcc7293226a62fc89e7542625937bca90cd0a754224ef9ca77501afd00ff252",  # noqa: E501
    "src/scripts/eval_stats.py": "9329b02058dfba2852fa8fe8fdc51a378bf8766c566c3b214468cf20d0429e69",  # noqa: E501
}


def check_pins(repo: Path) -> None:
    """Refuse any drift from the reviewed bytes of an imported rule or module."""
    from research.t34_round0d.common import sha256

    drift = [name for name, digest in PINNED.items() if sha256(repo / name) != digest]
    if drift:
        raise ValueError(f"imported Round 0/amendment bytes differ from the reviewed pins: {drift}")
