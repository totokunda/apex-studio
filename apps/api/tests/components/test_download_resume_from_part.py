import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from src.mixins.download_mixin import DownloadMixin


def test_download_resumes_from_part_when_server_ignores_range(tmp_path):
    # Simulate a server that always returns 200 OK (ignores Range),
    # and ensure we do NOT restart locally from scratch / duplicate bytes.
    payload = (b"apex-" * 16384) + b"done"  # ~80KB
    prefix_len = 25_000

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args, **_kwargs):  # pragma: no cover
            return

        def do_HEAD(self):  # noqa: N802
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()

        def do_GET(self):  # noqa: N802
            # Intentionally ignore any Range header and always serve the full payload.
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    httpd = HTTPServer(("127.0.0.1", 0), Handler)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    try:
        url = f"http://127.0.0.1:{httpd.server_port}/file.bin"

        dest = tmp_path / "file.bin"
        part = tmp_path / "file.bin.part"
        part.write_bytes(payload[:prefix_len])

        DownloadMixin().download_from_url(
            url=url,
            save_path=str(tmp_path),
            dest_path=str(dest),
            expected_size=len(payload),
        )

        assert dest.read_bytes() == payload
        assert not part.exists()
    finally:
        httpd.shutdown()
        httpd.server_close()
