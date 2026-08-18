"""Resumable verified acquisition and traversal-safe archive extraction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path, PurePosixPath
import random
import shutil
import stat
import tarfile
import time
import zipfile

import requests


class DownloadError(RuntimeError):
    pass


class PermanentDownloadError(DownloadError):
    pass


class UnsafeArchiveError(PermanentDownloadError):
    pass


@dataclass(frozen=True)
class DownloadResult:
    path: Path
    bytes: int
    sha256: str
    resumed: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _quarantine(path: Path) -> Path:
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    target = path.with_name(f'{path.name}.bad.{stamp}')
    os.replace(path, target)
    return target


def download_verified(
        url: str,
        destination: Path | str,
        *,
        expected_sha256: str | None = None,
        expected_bytes: int | None = None,
        max_attempts: int = 8,
        timeout: tuple[int, int] = (30, 120),
        chunk_size: int = 8 * 1024 * 1024) -> DownloadResult:
    """Resume into ``.part`` and publish only after content validation."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def validate(path: Path) -> tuple[int, str]:
        size = path.stat().st_size
        if expected_bytes is not None and size != expected_bytes:
            raise PermanentDownloadError(
                f'byte-size mismatch for {destination}: '
                f'expected {expected_bytes}, got {size}')
        digest = sha256_file(path)
        if expected_sha256 is not None and digest != expected_sha256:
            raise PermanentDownloadError(
                f'sha256 mismatch for {destination}: '
                f'expected {expected_sha256}, got {digest}')
        return size, digest

    if destination.is_file():
        size, digest = validate(destination)
        return DownloadResult(destination, size, digest, resumed=False)

    part = destination.with_name(destination.name + '.part')
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        offset = part.stat().st_size if part.exists() else 0
        headers = {'Range': f'bytes={offset}-'} if offset else {}
        try:
            with requests.get(
                    url, headers=headers, stream=True, timeout=timeout,
                    allow_redirects=True) as response:
                if response.status_code in {401, 403, 404}:
                    raise PermanentDownloadError(
                        f'HTTP {response.status_code} for {url}')
                response.raise_for_status()
                honored = (
                    offset > 0
                    and response.status_code == 206
                    and response.headers.get('Content-Range', '').startswith(
                        f'bytes {offset}-'))
                mode = 'ab' if honored else 'wb'
                with part.open(mode) as stream:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            try:
                size, digest = validate(part)
            except PermanentDownloadError:
                _quarantine(part)
                raise
            os.replace(part, destination)
            directory_fd = os.open(
                destination.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
            return DownloadResult(
                destination, size, digest, resumed=honored)
        except PermanentDownloadError:
            raise
        except (requests.RequestException, OSError) as error:
            last_error = error
            if attempt == max_attempts:
                break
            delay = min(3600.0, 30.0 * (2 ** (attempt - 1)))
            time.sleep(delay * random.uniform(0.8, 1.2))
    raise DownloadError(
        f'download exhausted {max_attempts} attempts for {url}: {last_error}')


def _safe_member(name: str) -> PurePosixPath:
    member = PurePosixPath(name)
    if member.is_absolute() or '..' in member.parts:
        raise UnsafeArchiveError(f'unsafe archive member: {name}')
    return member


def _merge_tree(source: Path, destination: Path) -> int:
    published = 0
    destination.mkdir(parents=True, exist_ok=True)
    for child in source.iterdir():
        target = destination / child.name
        if child.is_dir():
            if target.exists():
                published += _merge_tree(child, target)
                child.rmdir()
            else:
                os.replace(child, target)
                published += sum(1 for item in target.rglob('*') if item.is_file())
        else:
            os.replace(child, target)
            published += 1
    return published


def extract_archive(
        archive: Path | str, destination: Path | str, *,
        strip_components: int = 0) -> int:
    """Validate all members, extract to a staging tree, then publish."""
    if strip_components not in {0, 1}:
        raise ValueError('only zero or one stripped component is supported')
    archive = Path(archive)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    staging = destination / f'.extract-{archive.name}-{os.getpid()}'
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir()
    try:
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as stream:
                for info in stream.infolist():
                    _safe_member(info.filename)
                    mode = info.external_attr >> 16
                    if stat.S_ISLNK(mode):
                        raise UnsafeArchiveError(
                            f'unsafe archive member symlink: {info.filename}')
                bad_member = stream.testzip()
                if bad_member is not None:
                    raise PermanentDownloadError(
                        f'zip CRC failure at {bad_member}')
                stream.extractall(staging)
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as stream:
                for info in stream.getmembers():
                    _safe_member(info.name)
                    if info.issym() or info.islnk() or info.isdev():
                        raise UnsafeArchiveError(
                            f'unsafe archive member link/device: {info.name}')
                stream.extractall(staging)
        else:
            raise PermanentDownloadError(
                f'unsupported or corrupt archive: {archive}')
        publish_root = staging
        if strip_components == 1:
            roots = list(staging.iterdir())
            if len(roots) != 1 or not roots[0].is_dir():
                raise UnsafeArchiveError(
                    'strip_components=1 requires one wrapper directory')
            publish_root = roots[0]
        return _merge_tree(publish_root, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
