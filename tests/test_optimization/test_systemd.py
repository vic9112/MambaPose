from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
SYSTEMD = ROOT / 'systemd'


def test_campaign_unit_has_durable_restart_and_group_shutdown():
    unit = (SYSTEMD / 'mambapose-optimization.service').read_text()

    assert 'Restart=on-failure' in unit
    assert 'RestartPreventExitStatus=78' in unit
    assert 'StartLimitIntervalSec=0' in unit
    assert 'StartLimitBurst=' not in unit
    assert 'RestartSec=30s' in unit
    assert 'RestartSteps=' not in unit
    assert 'RestartMaxDelaySec=' not in unit
    assert 'KillMode=control-group' in unit
    assert 'PYTHONNOUSERSITE=1' in unit
    assert 'CUDA_VISIBLE_DEVICES=0' in unit
    assert 'tools/optimization/run_campaign.py --run' in unit


def test_controller_owned_stage_retries_cannot_consume_a_global_unit_burst():
    unit = (SYSTEMD / 'mambapose-optimization.service').read_text()

    assert 'StartLimitIntervalSec=0' in unit
    assert 'StartLimitBurst=' not in unit
    assert 'RestartSec=30s' in unit


def test_observer_timer_has_no_controller_authority():
    service = (
        SYSTEMD / 'mambapose-optimization-observer.service').read_text()
    timer = (SYSTEMD / 'mambapose-optimization-observer.timer').read_text()

    assert 'Type=oneshot' in service
    assert 'tools/optimization/observe.py' in service
    assert 'run_campaign.py' not in service
    assert 'OnUnitActiveSec=60s' in timer
    assert 'Persistent=true' in timer


def test_installer_fixture_verifies_units_without_installing(tmp_path):
    rendered = tmp_path / 'rendered'
    result = subprocess.run(
        ['bash', 'tools/optimization/install_user_service.sh',
         '--fixture-smoke', str(rendered)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert 'fixture systemd verification passed' in result.stdout
    assert f'rendered_repo_root={ROOT}' in result.stdout
    for name in (
            'mambapose-optimization.service',
            'mambapose-optimization-observer.service'):
        unit = (rendered / name).read_text()
        assert f'WorkingDirectory={ROOT}' in unit
        assert f'{ROOT}/tools/optimization/' in unit
    assert not (tmp_path / '.config/systemd/user').exists()


def test_pwl_installer_renders_exact_offline_conditional_campaign(tmp_path):
    rendered = tmp_path / 'rendered-pwl'
    result = subprocess.run(
        ['bash', 'tools/optimization/install_pwl_user_service.sh',
         '--fixture-smoke', str(rendered)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    service = (rendered / 'mambapose-pwl.service').read_text()
    expected_root = (
        '/home/vicchen/workspace/MambaPose/.worktrees/'
        'algo-ssm-quant-pwl-frozen')
    expected_command = (
        f'ExecStart={expected_root}/.venv/bin/python -B '
        f'{expected_root}/tools/optimization/run_campaign.py --run '
        f'--manifest {expected_root}/optimization/candidates.json '
        f'--campaign-root {expected_root}/work_dirs/optimization '
        '--candidate pwl-silu-s-v1 '
        '--candidate pwl-gelu-s-v1 '
        '--candidate pwl-softplus-s-v1 '
        '--candidate pwl-exp-s-v1 --admit-conditional')
    assert expected_command in service
    assert 'PrivateNetwork=yes' in service
    assert 'IPAddressDeny=any' in service
    assert 'RestrictAddressFamilies=AF_UNIX' in service
    assert 'network-online.target' not in service
    assert 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD' not in service
    assert 'PYTHONDONTWRITEBYTECODE=1' in service
    assert 'CUBLAS_WORKSPACE_CONFIG=:4096:8' in service
    observer = (rendered / 'mambapose-pwl-observer.service').read_text()
    assert (
        f'--campaign-root {expected_root}/work_dirs/optimization'
        in observer)
    assert 'run_campaign.py' not in observer
    timer = (rendered / 'mambapose-pwl-observer.timer').read_text()
    assert 'Persistent=true' in timer
    installer = (
        ROOT / 'tools/optimization/install_pwl_user_service.sh').read_text()
    assert 'SOURCE_COMMIT' in installer and 'RUNTIME_COMMIT' in installer
    assert 'status --porcelain' in installer
    assert 'campaign root must be fresh' in installer
    assert not (tmp_path / '.config/systemd/user').exists()


def test_pwl_stage_environment_scrubs_unsafe_checkpoint_fallback(monkeypatch):
    from tools.optimization.run_campaign import SubprocessStageRunner

    monkeypatch.setenv('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    environment = SubprocessStageRunner(
        ROOT / 'work_dirs/optimization',
        ROOT / 'optimization/candidates.json').environment()

    assert 'TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD' not in environment
    assert environment['PYTHONDONTWRITEBYTECODE'] == '1'
    assert environment['CUBLAS_WORKSPACE_CONFIG'] == ':4096:8'


def test_subprocess_runner_refuses_symlinked_attempt_log_before_child(
        tmp_path, monkeypatch):
    from tools.optimization.run_campaign import SubprocessStageRunner

    stage_dir = tmp_path / 'profile'
    stage_dir.mkdir()
    outside = tmp_path / 'outside.log'
    outside.write_text('sentinel\n', encoding='utf-8')
    (stage_dir / 'attempt-1.log').symlink_to(outside)
    runner = SubprocessStageRunner(
        tmp_path, tmp_path / 'optimization/candidates.json',
        heartbeat_interval=0.01)
    monkeypatch.setattr(
        runner, '_command', lambda *_args: ['/bin/echo', '/bin/true', 'bad'])
    monkeypatch.setattr(runner, '_relative', lambda path: Path(path).as_posix())

    with pytest.raises(FileExistsError, match='attempt log|symlink'):
        runner(SimpleNamespace(id='fixture'), 'profile', stage_dir, 1)

    assert outside.read_text(encoding='utf-8') == 'sentinel\n'
