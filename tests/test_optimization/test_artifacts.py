import argparse

import pytest


@pytest.mark.parametrize('value', [
    '/tmp/output.json',
    '../work_dirs/optimization/output.json',
    'work_dirs/optimization/../escape.json',
    'results/output.json',
])
def test_optimization_output_rejects_absolute_traversal_and_wrong_root(
        tmp_path, value):
    from mambapose_opt.artifacts import optimization_output_path

    with pytest.raises(argparse.ArgumentTypeError):
        optimization_output_path(value, repository_root=tmp_path)


def test_optimization_output_rejects_root_and_descendant_symlink_escapes(
        tmp_path):
    from mambapose_opt.artifacts import optimization_output_path

    outside = tmp_path.parent / f'{tmp_path.name}-outside'
    outside.mkdir()
    work_dirs = tmp_path / 'work_dirs'
    work_dirs.mkdir()
    (work_dirs / 'optimization').symlink_to(outside, target_is_directory=True)
    with pytest.raises(argparse.ArgumentTypeError):
        optimization_output_path(
            'work_dirs/optimization/result.json', repository_root=tmp_path)

    (work_dirs / 'optimization').unlink()
    artifact_root = work_dirs / 'optimization'
    artifact_root.mkdir()
    (artifact_root / 'linked').symlink_to(outside, target_is_directory=True)
    with pytest.raises(argparse.ArgumentTypeError):
        optimization_output_path(
            'work_dirs/optimization/linked/result.json',
            repository_root=tmp_path)


def test_optimization_output_returns_containment_checked_effective_path(tmp_path):
    from mambapose_opt.artifacts import optimization_output_path

    expected = tmp_path / 'work_dirs/optimization/run/result.json'
    assert optimization_output_path(
        'work_dirs/optimization/run/result.json',
        repository_root=tmp_path) == expected.resolve()


def test_optimization_output_rejects_symlinked_root_even_when_target_is_in_repo(
        tmp_path):
    from mambapose_opt.artifacts import optimization_output_path

    work_dirs = tmp_path / 'work_dirs'
    work_dirs.mkdir()
    target = tmp_path / 'internal-artifacts'
    target.mkdir()
    (work_dirs / 'optimization').symlink_to(target, target_is_directory=True)

    with pytest.raises(argparse.ArgumentTypeError):
        optimization_output_path(
            'work_dirs/optimization/result.json', repository_root=tmp_path)


def test_candidate_result_rejects_symlink_candidate_root_inside_repository(
        tmp_path, monkeypatch):
    import mambapose_opt.evaluation as evaluation_module
    from mambapose_opt.evaluation import CandidateResult, MetricError

    repository = tmp_path / 'repo'
    actual = repository / 'work_dirs/optimization/candidates/actual'
    actual.mkdir(parents=True)
    linked = repository / 'work_dirs/optimization/candidates/linked'
    linked.symlink_to(actual, target_is_directory=True)
    monkeypatch.setattr(
        evaluation_module, '_TRUSTED_REPOSITORY_ROOT', repository)

    with pytest.raises(MetricError, match='symlink'):
        CandidateResult.from_artifacts(linked)
