import hashlib
import json


def test_completion_requires_nonempty_hashed_in_repository_artifacts(
        tmp_path, monkeypatch):
    from tools.reproduction import run_campaign

    monkeypatch.setattr(run_campaign, 'REPO_ROOT', tmp_path)
    work_dir = tmp_path / 'run'
    work_dir.mkdir()
    provenance = {'config_sha256': 'a' * 64}

    (work_dir / 'completion.json').write_text(json.dumps({
        'provenance': provenance,
        'artifacts': [],
    }))
    assert not run_campaign._completion_valid(work_dir, provenance)

    artifact = work_dir / 'metrics.json'
    artifact.write_text('{}')
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    (work_dir / 'completion.json').write_text(json.dumps({
        'provenance': provenance,
        'artifacts': [{
            'path': 'run/metrics.json',
            'sha256': digest,
        }],
    }))
    assert run_campaign._completion_valid(work_dir, provenance)

    (work_dir / 'completion.json').write_text(json.dumps({
        'provenance': provenance,
        'artifacts': [{
            'path': '../outside.json',
            'sha256': digest,
        }],
    }))
    assert not run_campaign._completion_valid(work_dir, provenance)


def test_campaign_children_enable_trusted_mmengine_checkpoint_loading():
    from pathlib import Path

    campaign_source = Path('tools/reproduction/run_campaign.py').read_text()
    gates_source = Path('mambapose_repro/gates.py').read_text()
    assert "'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'" in campaign_source
    assert "'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD': '1'" in gates_source
