"""Guard the calibration-only entry point against campaign-length execution."""

from pathlib import Path

from research.redesign_m3c_20261008.train_m3c import build_parser
from research.t4e_return_20261016.calibration_smoke import UPDATE_LIMIT, recipe


def test_calibration_recipe_preserves_return_arm_with_short_ceiling() -> None:
    args = build_parser().parse_args(recipe(12, 0.02, Path("unused"), "mps"))
    assert UPDATE_LIMIT == 300
    assert args.transitions == 200_000
    assert args.device == "mps"
    assert (args.actors, args.worlds, args.batch_scale) == (2, 128, 2)
    assert args.init.endswith("B8_s2/final.pth")
    assert args.return_bank.name == "train"
    assert (args.label_frac, args.return_frac, args.return_weight) == (1, 0.5, 0.02)
    assert args.return_huber_delta == 100 and args.td_huber_delta == 1
    assert args.return_gradient_telemetry
    assert args.learner_fast and args.onpolicy_labels and args.obs_pack
    assert args.anchor_floor == 0.1 and args.gamma_start == args.gamma_end == 0.998
