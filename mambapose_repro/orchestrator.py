"""Finite-retry single-owner control primitives for the GPU campaign."""

from __future__ import annotations

from dataclasses import dataclass
import fcntl
import hashlib
from pathlib import Path
import time
from typing import Callable, Iterable

from .state import StateStore


TRANSIENT_EXIT = 75
PERMANENT_EXIT = 78


class PermanentFailure(RuntimeError):
    pass


class RetryExhausted(PermanentFailure):
    pass


class ConcurrentCampaign(PermanentFailure):
    pass


@dataclass(frozen=True)
class AttemptOutcome:
    exit_code: int
    fingerprint: str
    artifacts_valid: bool = True


class CampaignLock:
    """Exclusive advisory lock held for the complete orchestrator lifetime."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._stream = None

    def __enter__(self) -> 'CampaignLock':
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open('a+')
        try:
            fcntl.flock(
                self._stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            self._stream.close()
            self._stream = None
            raise ConcurrentCampaign(
                f'another campaign owns {self.path}') from error
        self._stream.seek(0)
        self._stream.truncate()
        self._stream.write(str(Path('/proc/self').resolve().name) + '\n')
        self._stream.flush()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._stream is not None:
            fcntl.flock(self._stream.fileno(), fcntl.LOCK_UN)
            self._stream.close()
            self._stream = None


def failure_fingerprint(text: str) -> str:
    normalized = ' '.join(text.strip().split())[-4096:]
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def run_until_terminal(
        run_id: str,
        runner: Callable[[], AttemptOutcome],
        state_store: StateStore,
        *,
        max_attempts: int,
        delays: Iterable[float] = (30, 120, 600)) -> AttemptOutcome:
    """Run one stage with persistent, bounded, fingerprinted recovery."""
    delay_values = tuple(delays)
    previous = state_store.read().get('runs', {}).get(run_id, {})
    if previous.get('status') == 'complete':
        return AttemptOutcome(
            0, previous.get(
                'completion_fingerprint', 'validated-existing-completion'))
    starting_attempt = int(previous.get('attempt', 0))
    last_fingerprint = previous.get('failure_fingerprint')
    for attempt in range(starting_attempt + 1, max_attempts + 1):
        state_store.transition(run_id, 'running', attempt=attempt)
        outcome = runner()
        if outcome.exit_code == 0:
            if not outcome.artifacts_valid:
                state_store.transition(
                    run_id, 'blocked', attempt=attempt,
                    failure_fingerprint='artifact-validation-failed')
                raise PermanentFailure(
                    f'{run_id} artifact validation failed after exit 0')
            state_store.transition(
                run_id, 'complete', attempt=attempt,
                completion_fingerprint=outcome.fingerprint)
            return outcome
        if outcome.exit_code == PERMANENT_EXIT:
            state_store.transition(
                run_id, 'blocked', attempt=attempt,
                failure_fingerprint=outcome.fingerprint)
            raise PermanentFailure(
                f'{run_id} permanent failure: {outcome.fingerprint}')
        if outcome.exit_code != TRANSIENT_EXIT:
            state_store.transition(
                run_id, 'blocked', attempt=attempt,
                failure_fingerprint=outcome.fingerprint,
                exit_code=outcome.exit_code)
            raise PermanentFailure(
                f'{run_id} unexpected exit {outcome.exit_code}: '
                f'{outcome.fingerprint}')
        last_fingerprint = outcome.fingerprint
        if attempt >= max_attempts:
            state_store.transition(
                run_id, 'exhausted', attempt=attempt,
                failure_fingerprint=last_fingerprint)
            raise RetryExhausted(
                f'{run_id} exhausted retries for {last_fingerprint}')
        delay = delay_values[min(attempt - 1, len(delay_values) - 1)]
        state_store.transition(
            run_id, 'retry_wait', attempt=attempt,
            failure_fingerprint=last_fingerprint,
            retry_delay_seconds=delay)
        if delay:
            time.sleep(delay)
    raise RetryExhausted(f'{run_id} has no remaining attempts')
