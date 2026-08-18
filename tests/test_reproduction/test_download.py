import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import threading
import zipfile

import pytest


@pytest.fixture
def range_server():
    payload = (b'MambaPose-range-test-' * 4096) + b'end'

    class Handler(BaseHTTPRequestHandler):
        ranges = []

        def do_GET(self):
            range_header = self.headers.get('Range')
            self.__class__.ranges.append(range_header)
            start = int(range_header.removeprefix('bytes=').split('-')[0]) \
                if range_header else 0
            body = payload[start:]
            self.send_response(206 if range_header else 200)
            self.send_header('Content-Length', str(len(body)))
            if range_header:
                self.send_header(
                    'Content-Range', f'bytes {start}-{len(payload)-1}/{len(payload)}')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (
            f'http://127.0.0.1:{server.server_port}/asset', payload, Handler)
    finally:
        server.shutdown()
        thread.join()
        server.server_close()


def test_resume_appends_only_when_server_honors_range(range_server, tmp_path):
    from mambapose_repro.download import download_verified

    url, payload, handler = range_server
    destination = tmp_path / 'asset.bin'
    part = destination.with_suffix('.bin.part')
    part.write_bytes(payload[:137])
    result = download_verified(
        url,
        destination,
        expected_sha256=hashlib.sha256(payload).hexdigest(),
        expected_bytes=len(payload),
        max_attempts=1)
    assert result.resumed is True
    assert handler.ranges[-1] == 'bytes=137-'
    assert destination.read_bytes() == payload
    assert not part.exists()


def test_checksum_mismatch_is_quarantined(range_server, tmp_path):
    from mambapose_repro.download import PermanentDownloadError, download_verified

    url, _, _ = range_server
    destination = tmp_path / 'asset.bin'
    with pytest.raises(PermanentDownloadError, match='sha256 mismatch'):
        download_verified(
            url, destination, expected_sha256='0' * 64, max_attempts=1)
    assert not destination.exists()
    assert list(tmp_path.glob('asset.bin.part.bad.*'))


def test_safe_zip_extraction_rejects_parent_traversal(tmp_path):
    from mambapose_repro.download import UnsafeArchiveError, extract_archive

    archive = tmp_path / 'bad.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('../escape.txt', 'unsafe')
    with pytest.raises(UnsafeArchiveError, match='unsafe archive member'):
        extract_archive(archive, tmp_path / 'out')
    assert not (tmp_path / 'escape.txt').exists()


def test_safe_zip_extraction_publishes_files(tmp_path):
    from mambapose_repro.download import extract_archive

    archive = tmp_path / 'good.zip'
    with zipfile.ZipFile(archive, 'w') as stream:
        stream.writestr('annotations/example.json', '{}')
    extracted = extract_archive(archive, tmp_path / 'out')
    assert (tmp_path / 'out/annotations/example.json').read_text() == '{}'
    assert extracted == 1
