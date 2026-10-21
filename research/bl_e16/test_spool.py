from pathlib import Path

import torch

from research.bl_e16.spool import batches, write_life
from research.bl_e16.training import E16Batch


def test_complete_life_spool_reads_bounded_batches(tmp_path: Path) -> None:
    n = 65
    batch = E16Batch(
        local=torch.zeros(n, 1, 7, 3, 3),
        global_=torch.zeros(n, 1, 4, 3, 3),
        scalars=torch.zeros(n, 1, 60),
        frame=torch.arange(n).view(n, 1),
        actions=torch.zeros(n, 1, dtype=torch.long),
        logp=torch.zeros(n, 1),
        rewards=torch.ones(n, 1),
        dones=torch.zeros(n, 1, dtype=torch.bool),
        values=torch.zeros(n, 1),
        last_value=torch.zeros(1),
        action_masks=torch.ones(n, 1, 6, dtype=torch.bool),
        critic_targets=torch.arange(n, dtype=torch.float32).flip(0).view(n, 1),
    )
    path = write_life(tmp_path / "life.npz", batch)
    chunks = list(batches(path, 32))
    assert [x.actions.shape[0] for x in chunks] == [32, 32, 1]
    assert chunks[-1].critic_targets.item() == 0


def test_streaming_geometry_never_materializes_whole_life(tmp_path: Path) -> None:
    from research.bl_e16.spool import StreamingLife

    writer = StreamingLife(tmp_path / "stream", rows=32)
    for index in range(65):
        record = {
            "local": torch.zeros(1, 7, 3, 3, dtype=torch.uint8),
            "global_": torch.zeros(1, 4, 3, 3, dtype=torch.uint8),
            "scalars": torch.zeros(1, 60),
            "action": torch.zeros(1, dtype=torch.long),
            "logp": torch.zeros(1),
            "value": torch.zeros(1),
            "mask": torch.ones(1, 6, dtype=torch.bool),
            "geometry": torch.zeros(1, 7, 83, 145),
            "history": torch.zeros(1, 128, 43),
        }
        writer.append(record, 1.0, index == 64, index)
        assert len(writer.buffer) < 32
    writer.finish()
    chunks = list(writer.batches())
    assert [len(chunk.actions) for chunk in chunks] == [32, 32, 1]
    assert chunks[0].critic_targets[0].item() == 65.0
    assert chunks[-1].critic_targets[-1].item() == 1.0
    assert all(chunk.geometry.shape[0] <= 32 for chunk in chunks)


def test_pending_spool_resume_preserves_chunks_targets_and_minibatch_boundaries(
    tmp_path: Path,
) -> None:
    import copy

    from research.bl_e16.spool import StreamingLife

    def record():
        return {
            "local": torch.zeros(1, 7, 3, 3),
            "global_": torch.zeros(1, 4, 3, 3),
            "scalars": torch.zeros(1, 60),
            "action": torch.zeros(1, dtype=torch.long),
            "logp": torch.zeros(1),
            "value": torch.zeros(1),
            "mask": torch.ones(1, 6, dtype=torch.bool),
            "geometry": None,
            "history": None,
        }

    original = StreamingLife(tmp_path / "control", rows=4)
    for i in range(6):
        original.append(record(), float(i + 1), False, i)
    saved = copy.deepcopy(original.snapshot())
    restored = StreamingLife.restore(tmp_path / "restored", saved)
    for writer in (original, restored):
        for i in range(6, 9):
            writer.append(record(), float(i + 1), i == 8, i)
        writer.finish()
    first, second = list(original.batches()), list(restored.batches())
    assert [len(b.actions) for b in first] == [4, 4, 1]
    assert [len(b.actions) for b in second] == [4, 4, 1]
    for left, right in zip(first, second):
        assert torch.equal(left.critic_targets, right.critic_targets)
        assert torch.equal(left.rewards, right.rewards)
