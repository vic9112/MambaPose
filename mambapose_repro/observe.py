"""Read-only campaign health derivation and observer-owned evidence output."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _gpu_telemetry() -> dict[str, Any]:
    try:
        output = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
            '--format=csv,noheader,nounits',
        ], text=True, timeout=10)
        values = [part.strip() for part in output.splitlines()[0].split(',')]
        return {
            'index': int(values[0]),
            'name': values[1],
            'memory_used_mib': int(values[2]),
            'memory_total_mib': int(values[3]),
            'utilization_percent': int(values[4]),
        }
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return {'status': 'unavailable'}


def observe(
        campaign_dir: Path | str,
        *,
        heartbeat_max_age: int = 180,
        startup_grace: int = 600) -> dict[str, Any]:
    """Derive health without changing controller-owned state or processes."""
    campaign_dir = Path(campaign_dir)
    state = _load(campaign_dir / 'state.json')
    heartbeat = _load(campaign_dir / 'heartbeat.json')
    previous = _load(campaign_dir / 'status.json')
    now = _now()
    heartbeat_time = _parse_time(heartbeat.get('timestamp'))
    heartbeat_age = (
        (now - heartbeat_time).total_seconds()
        if heartbeat_time is not None else None)
    pid = heartbeat.get('pid')
    process_alive = (
        isinstance(pid, int) and pid > 0 and Path(f'/proc/{pid}').exists())
    stage = state.get('stage', 'not_started')
    state_time = _parse_time(state.get('updated_at'))
    state_age = (
        (now - state_time).total_seconds() if state_time is not None else None)

    if stage == 'complete' and state.get('current_run'):
        health = 'complete'
    elif stage in {'blocked', 'exhausted'}:
        health = 'failed'
    elif stage in {'running', 'retry_wait'}:
        if heartbeat_age is not None and heartbeat_age <= heartbeat_max_age:
            health = 'running' if process_alive else 'failed'
        elif state_age is not None and state_age <= startup_grace and not heartbeat:
            health = 'starting'
        else:
            health = 'stalled' if process_alive else 'failed'
    else:
        health = 'not_started'

    disk = shutil.disk_usage(campaign_dir)
    status = {
        'schema_version': 1,
        'observed_at': now.isoformat(),
        'health': health,
        'stage': stage,
        'current_run': state.get('current_run'),
        'generation': state.get('generation', 0),
        'heartbeat_age_seconds': heartbeat_age,
        'process_alive': process_alive,
        'pid': pid,
        'phase': heartbeat.get('phase'),
        'log_bytes': heartbeat.get('log_bytes'),
        'progressed_since_previous_observation': (
            heartbeat.get('log_bytes') != previous.get('log_bytes')),
        'disk_free_bytes': disk.free,
        'gpu': _gpu_telemetry(),
    }
    _atomic_json(campaign_dir / 'status.json', status)
    history = campaign_dir / 'health.jsonl'
    if history.exists() and history.stat().st_size > 10 * 1024 * 1024:
        rotated = campaign_dir / 'health.jsonl.1'
        os.replace(history, rotated)
    with history.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(status, sort_keys=True) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    return status

