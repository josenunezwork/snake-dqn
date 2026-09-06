"""Checkpoint catalog policy shared by the web listing and loader."""

from web.backend.checkpoints import list_checkpoint_catalog, resolve_checkpoint_name  # isort: skip


def _roots(tmp_path):
    repo = tmp_path / "repo"
    saved = repo / "saved_snakes"
    runs = repo / "runs"
    saved.mkdir(parents=True)
    runs.mkdir()
    return repo, saved, runs


def test_catalog_names_are_exactly_the_resolvable_client_namespace(tmp_path):
    repo, saved, runs = _roots(tmp_path)
    (saved / "champion.pth").write_bytes(b"checkpoint")
    run = runs / "pqn_local"
    run.mkdir()
    (run / "latest_pqn.pth").write_bytes(b"checkpoint")
    (run / "other.pth").write_bytes(b"checkpoint")

    catalog = list_checkpoint_catalog(str(repo), str(saved), str(runs))
    assert [entry.name for entry in catalog] == ["champion.pth", "runs/pqn_local/latest_pqn.pth"]
    for entry in catalog:
        assert resolve_checkpoint_name(entry.name, str(repo), str(saved), str(runs)) == entry


def test_catalog_and_resolver_reject_traversal_and_outside_symlinks(tmp_path):
    repo, saved, runs = _roots(tmp_path)
    outside = tmp_path / "outside.pth"
    outside.write_bytes(b"checkpoint")
    (saved / "outside.pth").symlink_to(outside)
    linked_run = runs / "linked"
    linked_run.mkdir()
    (linked_run / "latest_pqn.pth").symlink_to(outside)

    assert list_checkpoint_catalog(str(repo), str(saved), str(runs)) == []
    for name in ("../outside.pth", "runs/../saved_snakes/outside.pth", "/tmp/outside.pth"):
        assert resolve_checkpoint_name(name, str(repo), str(saved), str(runs)) is None


def test_run_alias_keeps_latest_name_while_loading_the_canonical_target(tmp_path):
    repo, saved, runs = _roots(tmp_path)
    payload = runs / "payload"
    payload.mkdir()
    target = payload / "model.pth"
    target.write_bytes(b"checkpoint")
    job = runs / "job"
    job.mkdir()
    alias = job / "latest_pqn.pth"
    alias.symlink_to(target)

    catalog = list_checkpoint_catalog(str(repo), str(saved), str(runs))
    assert catalog == [
        resolve_checkpoint_name("runs/job/latest_pqn.pth", str(repo), str(saved), str(runs))
    ]
    assert catalog[0].name == "runs/job/latest_pqn.pth"
    assert catalog[0].path == str(target)


def test_symlinked_repo_root_keeps_lexical_run_name(tmp_path):
    physical = tmp_path / "physical-repo"
    (physical / "saved_snakes").mkdir(parents=True)
    runs = physical / "runs"
    run = runs / "job"
    run.mkdir(parents=True)
    (run / "latest_pqn.pth").write_bytes(b"checkpoint")
    repo = tmp_path / "repo-link"
    repo.symlink_to(physical, target_is_directory=True)

    catalog = list_checkpoint_catalog(str(repo), str(repo / "saved_snakes"), str(repo / "runs"))
    assert [entry.name for entry in catalog] == ["runs/job/latest_pqn.pth"]
    assert (
        resolve_checkpoint_name(
            "runs/job/latest_pqn.pth", str(repo), str(repo / "saved_snakes"), str(repo / "runs")
        )
        == catalog[0]
    )
