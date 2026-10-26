"""Independent mechanism checks for the B-2 loss, heads, and metrics."""

import pytest
import torch

from research.sandbox_b_heads.core_b2 import Config, _pair_loss, masks, training_loss
from research.sandbox_b_heads.job_b2 import metrics, threshold_actions
from research.sandbox_b_heads.model_b2 import make_head


def _decisive_row_table() -> dict[str, torch.Tensor]:
    """Build one decisive row with unknown, illegal, and masked actions."""
    return {
        "action_mask": torch.tensor([[True, True, True, True, False, False]]),
        "status": torch.tensor([[True, True, False, True, True, True]]),
        "legal": torch.tensor([[True, True, True, False, True, True]]),
        "died": torch.tensor([[True, False, False, False, False, False]]),
        "bare": torch.tensor([0]),
        "ordinary": torch.tensor([False]),
        "life_id": torch.tensor([0]),
    }


@pytest.mark.parametrize("loss_name", ["kl_pairs", "hinge_top"])
@pytest.mark.parametrize("bad_action", [2, 3])
def test_selectable_unknown_and_illegal_high_q_actions_get_downward_gradient(
    loss_name: str, bad_action: int
):
    """Both survival objectives must lower each selectable unknown/illegal action."""
    table = _decisive_row_table()
    assert masks(table)[3].tolist() == [True]

    q_values = torch.tensor([[-2.0, -1.0, -3.0, -3.0, 100.0, 99.0]])
    with torch.no_grad():
        q_values[0, bad_action] = 5.0
    q_values.requires_grad_()
    base_q = torch.zeros_like(q_values)
    config = Config("normed_linear", loss_name, 2.0, 500, "row_uniform")

    training_loss(q_values, base_q, table, config).backward()

    # Gradient descent moves a positive-gradient Q downward.  The selected
    # unknown/illegal action is the unique top bad action for hinge_top.
    assert q_values.grad[0, bad_action] > 0
    assert q_values.grad[0, 1] < 0


@pytest.mark.parametrize("loss_name", ["kl_pairs", "hinge_top"])
def test_masked_actions_have_no_survival_loss_pressure(loss_name: str):
    """High-Q actions outside action_mask must not enter either survival loss."""
    table = _decisive_row_table()
    q_values = torch.tensor([[-2.0, -1.0, 3.0, 2.0, 100.0, 99.0]], requires_grad=True)
    config = Config("normed_linear", loss_name, 2.0, 500, "row_uniform")

    training_loss(q_values, torch.zeros_like(q_values), table, config).backward()

    assert q_values.grad[0, 4] == 0
    assert q_values.grad[0, 5] == 0


def test_gated_mlp_first_step_trains_body_and_gate_via_separate_losses():
    """The supervised gate must escape the old zero-output saddle safely."""
    torch.manual_seed(123)
    model = make_head("supervised_gated_normed").train()
    features = torch.randn(32, 256)
    length = torch.rand(32)
    margin = torch.rand(32) * 10

    output, _, logits = model.forward_components(features, length, margin)
    output.square().sum().backward()
    assert model.body[0].weight.grad is not None
    assert model.body[0].weight.grad.norm() > 0
    assert all(parameter.grad is None for parameter in model.gate.parameters())
    torch.nn.functional.binary_cross_entropy_with_logits(logits, torch.ones(32)).backward()
    assert model.gate[0].weight.grad.norm() > 0

    model.eval()
    with torch.no_grad():
        model.gate[-1].weight.zero_()
        model.gate[-1].bias.fill_(-100.0)
        closed = model(features, length, margin)
    assert torch.equal(closed, torch.zeros_like(closed))


def test_pair_loss_ignores_nondecisive_rows_and_averages_each_decisive_row():
    """Different pair counts give equal row weight; ordinary/safe rows give none."""
    q = torch.tensor(
        [[1.0, 0.0, -1.0], [3.0, 2.0, 0.0], [100.0, -100.0, 20.0], [-100.0, 100.0, 0.0]],
        requires_grad=True,
    )
    survivor = torch.tensor([[1, 0, 0], [1, 1, 0], [1, 1, 0], [0, 1, 0]]).bool()
    bad = ~survivor
    bad[0, 1] = False  # One pair on row 0 versus two pairs on row 1.
    decisive = torch.tensor([True, True, False, False])
    row0 = torch.nn.functional.softplus(q[0, 2] - q[0, 0])
    row1 = torch.nn.functional.softplus(q[1, 2] - q[1, :2]).mean()
    loss = _pair_loss(q, survivor, bad, decisive)
    assert loss.item() == pytest.approx(((row0 + row1) / 2).item())
    loss.backward()
    assert torch.equal(q.grad[2:], torch.zeros_like(q.grad[2:]))
    assert torch.isfinite(q.grad).all()

    empty_q = q.detach().clone().requires_grad_()
    empty_loss = _pair_loss(empty_q, survivor, bad, torch.zeros(4, dtype=torch.bool))
    empty_loss.backward()
    assert empty_loss.item() == 0
    assert torch.equal(empty_q.grad, torch.zeros_like(empty_q))


@pytest.mark.parametrize("logit", [-4.0, -0.1, 0.0, 0.2, 4.0])
def test_eval_gate_equals_train_gate_on_identical_inputs(logit: float):
    model = make_head("supervised_gated_normed")
    features, length, margin = torch.randn(3, 256), torch.rand(3), torch.rand(3)
    with torch.no_grad():
        model.gate[-1].weight.zero_()
        model.gate[-1].bias.fill_(logit)
    model.train()
    trained, body, logits = model.forward_components(features, length, margin)
    model.eval()
    evaluated = model(features, length, margin)
    assert torch.equal(trained, evaluated)
    assert torch.equal(trained, body * logits.sigmoid().unsqueeze(1))


@pytest.mark.parametrize("loss_name", ["kl_pairs", "hinge_top"])
def test_ordinary_regularization_masks_unselectable_actions_without_nan(loss_name: str):
    table = _decisive_row_table()
    table["ordinary"][:] = True
    table["died"][0, 0] = False  # Safe base, hence no survival pressure.
    q = torch.tensor([[0.0, 1.0, 2.0, 3.0, 100.0, -100.0]], requires_grad=True)
    base = torch.zeros_like(q)
    config = Config("normed_linear", loss_name, 2.0, 500, "row_uniform")
    loss = training_loss(q, base, table, config)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(q.grad).all()
    assert torch.equal(q.grad[0, 4:], torch.zeros(2))
    assert q.grad[0, :4].abs().sum() > 0
    changed = q.detach().clone()
    changed[0, 4:] *= -20
    assert training_loss(changed, base, table, config).item() == pytest.approx(loss.item())


@pytest.mark.parametrize("sampling", ["row_uniform", "decisive_life_balanced"])
def test_gate_bce_balances_supervised_classes_and_excludes_unknown_rows(sampling: str):
    table = {
        key: value.repeat((11,) + (1,) * (value.ndim - 1))
        for key, value in _decisive_row_table().items()
    }
    table["died"][1:10, 0] = False
    table["status"][10] = False  # Neither decisive, ordinary, nor safe-base.
    gate = torch.zeros(11, requires_grad=True)
    q = torch.zeros(11, 6)
    config = Config("supervised_gated_normed", "kl_pairs", 32.0, 500, sampling)
    loss = training_loss(q, q, table, config, gate, q)
    loss.backward()
    assert gate.grad[0].item() == pytest.approx(-0.45)
    assert gate.grad[1:10].tolist() == pytest.approx([0.05] * 9)
    assert gate.grad[10] == 0
    assert gate.grad.sum().item() == pytest.approx(0.0, abs=1e-7)


@pytest.mark.parametrize("decisive", [True, False])
def test_gate_bce_is_finite_when_only_one_supervised_class_exists(decisive: bool):
    table = _decisive_row_table()
    table["died"][0, 0] = decisive
    gate = torch.zeros(1, requires_grad=True)
    q = torch.zeros(1, 6)
    config = Config("supervised_gated_normed", "kl_pairs", 32.0, 500, "row_uniform")
    loss = training_loss(q, q, table, config, gate, q)
    loss.backward()
    assert torch.isfinite(loss)
    assert gate.grad.item() == pytest.approx(-0.5 if decisive else 0.5)


@pytest.mark.parametrize("loss_name", ["kl_pairs", "hinge_top"])
def test_ordinary_body_regularization_is_independent_of_gate_magnitude(loss_name: str):
    """Small gates cannot hide an unregularized body; Q losses cannot train gates."""
    model = make_head("supervised_gated_normed")
    table = _decisive_row_table()
    table["ordinary"][:] = True
    table["died"][0, 0] = False
    features, length, margin = torch.randn(1, 256), torch.ones(1), torch.ones(1)
    base = torch.zeros(1, 6)
    config = Config("supervised_gated_normed", loss_name, 32.0, 500, "row_uniform")
    gradients = []
    for logit in (-8.0, 4.0):
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.gate[-1].weight.zero_()
            model.gate[-1].bias.fill_(logit)
        residual, body, _ = model.forward_components(features, length, margin)
        training_loss(base + residual, base, table, config, ordinary_q=base + body).backward()
        gradients.append(model.body[-1].weight.grad.clone())
        assert all(parameter.grad is None for parameter in model.gate.parameters())
    assert gradients[0].norm() > 0
    assert torch.equal(*gradients)


def test_all_b2_heads_are_invariant_to_positive_per_row_feature_scaling():
    """L2-normalized features prevent magnitude-only changes from changing Q."""
    torch.manual_seed(7)
    features = torch.randn(5, 256)
    scaled = features * torch.tensor([0.25, 2.0, 10.0, 1.5, 5.0]).reshape(-1, 1)
    length, margin = torch.rand(5), torch.rand(5) * 10

    for name in ("normed_linear", "cosine_fastgain", "supervised_gated_normed"):
        model = make_head(name).train()
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(mean=0.0, std=0.05)
        original = model(features, length, margin)
        rescaled = model(scaled, length, margin)
        assert torch.allclose(original, rescaled, atol=1e-6, rtol=1e-6), name


def _uneven_life_metrics_fixture() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Return three-vs-one decisive and four-vs-one ordinary life rows."""
    n = 9
    status = torch.ones(n, 6, dtype=torch.bool)
    legal = torch.ones_like(status)
    died = torch.zeros_like(status)
    base = torch.zeros(n, dtype=torch.long)
    selected = torch.tensor([1, 1, 0, 0, 1, 2, 3, 0, 4])

    # Three decisive rows in life 0 and one in life 1.
    for row in (0, 1, 2, 3):
        died[row, 0] = True
    # Ordinary rows: known survivor (safe), known death (harm), illegal (harm),
    # base (safe), and unknown (harm).
    died[5, 2] = True
    legal[6, 3] = False
    status[8, 4] = False
    table = {
        "status": status,
        "legal": legal,
        "died": died,
        "bare": base,
        "action_mask": torch.ones(n, 6, dtype=torch.bool),
        "ordinary": torch.tensor([False] * 4 + [True] * 5),
        "life_id": torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 1]),
    }
    return table, selected


def test_metrics_report_uneven_life_micro_macro_and_ordinary_harm_denominators():
    """Retention/harm use row micro averages and equal-life macro averages."""
    table, selected = _uneven_life_metrics_fixture()
    report = metrics(torch.zeros(9, 6), table, selected)

    # Decisive retention: 2/3 in life 0 and 0/1 in life 1.
    assert report["rescue_decisive_count"] == 4
    assert report["rescue_retention"] == pytest.approx(0.5)
    assert report["life_macro_rescue_retention"] == pytest.approx(1 / 3)
    # Ordinary harm: 2/4 in life 0 (known death + illegal), 1/1 in life 1
    # (unknown), while known-survivor and unchanged base rows are safe.
    assert report["ordinary_count"] == 5
    assert report["ordinary_harm_count"] == 3
    assert report["ordinary_harm_proxy"] == pytest.approx(0.6)
    assert report["life_macro_ordinary_harm"] == pytest.approx(0.75)


def test_tau_requires_strictly_more_than_threshold_and_preserves_base_on_ties():
    """A candidate wins only when its Q advantage is strictly greater than tau."""
    q = torch.tensor(
        [[1.0, 1.4, 0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.5, 0.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    table = {
        "bare": torch.tensor([0, 0]),
        "action_mask": torch.tensor([[True, True, False, False, False, False], [True] * 6]),
    }

    assert threshold_actions(q, table, 0.0).tolist() == [1, 2]
    assert threshold_actions(q, table, 0.4).tolist() == [0, 2]
    assert threshold_actions(q, table, 0.5).tolist() == [0, 0]


def test_nonempty_b2_resume_bundle_passes_registry_and_runtime_recipe_checks(tmp_path, monkeypatch):
    """A real partial ledger remains admissible and matches the shipped B-2 recipe."""
    import json

    import numpy as np

    from research.sandbox_b_heads import job_b2, package_b2, resume_b2
    from research.sandbox_b_heads.core_b2 import CONFIGS
    from tools.sandbox import preflight, registry

    parents = np.array([f"synthetic-{index}" for index in range(40)])
    data = tmp_path / "projection.npz"
    np.savez(
        data,
        row_id=np.array([f"life/{index}" for index in range(len(parents))]),
        life_id=np.array(["life"] * len(parents)),
        parent_world=parents,
    )
    b8 = tmp_path / "b8.pth"
    b8.write_bytes(b"frozen-b8")
    registry_path = tmp_path / "registry.json"
    registry_path.write_text("{}")

    frozen = resume_b2.recipe(data, b8)
    exports = tmp_path / "exports"
    exports.mkdir()
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint-bytes")
    unit = resume_b2.identity(
        0, resume_b2.NAMESPACE, resume_b2.ALLOCATION, resume_b2.fingerprint(frozen)
    )
    row = {
        "config": CONFIGS[0].id,
        "status": "COMPLETE",
        "sha256": resume_b2.sha256(checkpoint),
    }
    bundle = exports / f"00-{CONFIGS[0].id}.tar"
    job_b2._publish_bundle(bundle, checkpoint, row, unit)
    ledger = tmp_path / "ledger.json"
    built = resume_b2.build_ledger(exports, ledger, frozen)
    assert len(built["units"]) == 1

    # This is the real registry branch under test; the package's broader input
    # admission is stubbed so the fixture need not impersonate the closed D1024.
    registry.validate_input(
        {
            "kind": "verified-unit",
            "path": str(bundle),
            "sha256": registry.digest(bundle),
        },
        {},
    )

    monkeypatch.setattr(registry, "load_registry", lambda _path: {})
    monkeypatch.setattr(registry, "validate_npz", lambda *args, **kwargs: None)
    monkeypatch.setattr(registry, "validate_rows", lambda *args, **kwargs: None)
    monkeypatch.setattr(package_b2, "validate_projection", lambda _table: None)
    monkeypatch.setattr(package_b2, "_git_commit", lambda: "1" * 40)

    packet_path = tmp_path / "packet.json"
    sandbox_path = tmp_path / "sandbox.json"
    output_dir = tmp_path / "fresh-output"
    package_b2.build(
        data,
        registry_path,
        b8,
        ledger,
        packet_path,
        output_dir,
        60.0,
        sandbox_path,
    )
    packet = json.loads(packet_path.read_text())
    sandbox = json.loads(sandbox_path.read_text())
    runtime_ledger = json.loads(
        packet_path.with_name(packet_path.stem + "-runtime-ledger.json").read_text()
    )
    assert runtime_ledger["units"]
    upload = next(
        record for record in packet["inputs"] if record.get("dest", "").startswith("resume-")
    )
    assert upload["path"] == str(bundle.resolve())
    assert runtime_ledger["units"][0]["bundle"]["path"] == upload["dest"]

    rows = [record for record in sandbox["inputs"] if record["kind"] == "closed-f0"]
    b8_inputs = [record for record in sandbox["inputs"] if record["kind"] == "frozen-checkpoint"]
    preflight._validate_b_heads_runtime_recipe(sandbox, packet, runtime_ledger, rows, b8_inputs)

    runtime_ledger_path = packet_path.with_name(packet_path.stem + "-runtime-ledger.json")
    packet_bytes = packet_path.read_bytes()
    sandbox_bytes = sandbox_path.read_bytes()
    runtime_bytes = runtime_ledger_path.read_bytes()
    with pytest.raises(ValueError, match="fresh"):
        package_b2.build(
            data,
            registry_path,
            b8,
            ledger,
            packet_path,
            output_dir,
            60.0,
            sandbox_path,
        )
    assert packet_path.read_bytes() == packet_bytes
    assert sandbox_path.read_bytes() == sandbox_bytes
    assert runtime_ledger_path.read_bytes() == runtime_bytes
