"""Crash-safe mutable campaign state and append-only evidence records."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Mapping


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _boot_id() -> str:
    try:
        return Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    except OSError:
        return 'unavailable'


class StateStore:
    """Single-writer persistence used only by the campaign orchestrator."""

    def __init__(self, directory: Path | str):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.state_path = self.directory / 'state.json'
        self.events_path = self.directory / 'events.jsonl'
        self.results_path = self.directory / 'results.jsonl'

    def read(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                'schema_version': 1,
                'generation': 0,
                'stage': 'not_started',
                'current_run': None,
                'runs': {},
                'updated_at': None,
                'boot_id': _boot_id(),
            }
        return json.loads(self.state_path.read_text(encoding='utf-8'))

    def _atomic_json(self, path: Path, value: Mapping[str, Any]) -> None:
        temporary = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
        try:
            with temporary.open('w', encoding='utf-8') as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
        payload = json.dumps(value, sort_keys=True, separators=(',', ':'))
        with path.open('a', encoding='utf-8') as stream:
            stream.write(payload + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def transition(
            self, run_id: str, status: str, *, attempt: int,
            **details: Any) -> dict[str, Any]:
        state = self.read()
        now = _timestamp()
        run_state = dict(state.get('runs', {}).get(run_id, {}))
        run_state.update({
            'status': status,
            'attempt': attempt,
            'updated_at': now,
            **details,
        })
        runs = dict(state.get('runs', {}))
        runs[run_id] = run_state
        next_state = {
            **state,
            'schema_version': 1,
            'generation': int(state.get('generation', 0)) + 1,
            'stage': status,
            'current_run': run_id,
            'runs': runs,
            'updated_at': now,
            'boot_id': _boot_id(),
        }
        self._atomic_json(self.state_path, next_state)
        self._append_jsonl(self.events_path, {
            'timestamp': now,
            'boot_id': next_state['boot_id'],
            'generation': next_state['generation'],
            'run_id': run_id,
            'status': status,
            'attempt': attempt,
            'details': details,
        })
        return next_state

    def record_result(
            self, run_id: str, result: Mapping[str, Any],
            **provenance: Any) -> None:
        self._append_jsonl(self.results_path, {
            'timestamp': _timestamp(),
            'boot_id': _boot_id(),
            'run_id': run_id,
            'result': dict(result),
            'provenance': provenance,
        })

