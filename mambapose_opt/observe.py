"""Strictly read-only health projection for an optimization campaign."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .gpu_guard import (
    ExternalGpuContention,
    controller_process_tree,
    query_compute_processes,
)


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value)
        return parsed if parsed.tzinfo is not None else parsed.replace(
            tzinfo=timezone.utc)
    except ValueError:
        return None


def observe(
        root: Path, heartbeat_max_age: float,
        *, gpu_lock_path: Path | None = None) -> dict[str, Any]:
    """Derive campaign and GPU health without writing or controlling work."""
    root = Path(root)
    state = _load(root / 'state.json')
    heartbeat = _load(root / 'heartbeat.json')
    lease = _load(Path(gpu_lock_path) if gpu_lock_path else root / 'gpu.lock')
    plan = _load(root / 'campaign-plan.json')
    now = datetime.now(timezone.utc)
    heartbeat_time = _parse_time(heartbeat.get('timestamp'))
    heartbeat_age = (
        (now - heartbeat_time).total_seconds()
        if heartbeat_time is not None else None)
    pid = heartbeat.get('pid')
    process_alive = (
        isinstance(pid, int) and pid > 0 and Path(f'/proc/{pid}').exists())
    stage = state.get('stage', 'not_started')

    run_ids = plan.get('run_ids')
    expected = (
        tuple(run_ids)
        if isinstance(run_ids, list)
        and all(isinstance(run_id, str) for run_id in run_ids)
        else ())
    runs = state.get('runs') if isinstance(state.get('runs'), dict) else {}
    completed_runs = sum(
        runs.get(run_id, {}).get('status') == 'complete'
        for run_id in expected)
    campaign_complete = bool(expected) and completed_runs == len(expected)

    if campaign_complete:
        health = 'complete'
    elif stage in {'blocked', 'exhausted'}:
        health = 'failed'
    elif stage == 'complete':
        health = 'incomplete'
    elif stage in {'running', 'retry_wait'}:
        if heartbeat_age is None:
            health = 'starting' if stage == 'running' else 'waiting'
        elif heartbeat_age <= heartbeat_max_age:
            health = 'running' if process_alive else 'failed'
        else:
            health = 'stalled' if process_alive else 'failed'
    else:
        health = 'not_started'

    device_index = lease.get('device_index', 0)
    allowed = {
        pid for pid in lease.get('allowed_pids', [])
        if isinstance(pid, int) and pid > 0
    }
    try:
        allowed = set(controller_process_tree(allowed)) if allowed else set()
        owners = query_compute_processes(int(device_index))
        external = sorted(owner.pid for owner in owners if owner.pid not in allowed)
        gpu_status = 'available'
    except (ExternalGpuContention, TypeError, ValueError):
        owners = ()
        external = []
        gpu_status = 'unavailable'

    return {
        'schema_version': 1,
        'observed_at': now.isoformat(),
        'health': health,
        'stage': stage,
        'current_run': state.get('current_run'),
        'generation': state.get('generation', 0),
        'heartbeat_age_seconds': heartbeat_age,
        'process_alive': process_alive,
        'pid': pid,
        'stage_id': heartbeat.get('stage_id'),
        'completed_runs': completed_runs,
        'expected_runs': len(expected),
        'gpu_status': gpu_status,
        'gpu_processes': [
            {
                'pid': owner.pid,
                'used_memory_mib': owner.used_memory_mib,
                'command': owner.command,
            }
            for owner in owners
        ],
        'gpu_contention': bool(external),
        'external_gpu_pids': external,
    }
