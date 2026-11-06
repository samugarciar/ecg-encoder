"""Sirve la interfaz web/ en local con las mismas cabeceras que Vercel (web/vercel.json).

Uso: python scripts/serve_web.py [--port 8000]   y abrir http://localhost:8000
Con http://localhost:8000/?ejemplo la página carga y analiza el ECG sintético de ejemplo.
"""

from __future__ import annotations

import argparse
import json
import re
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

from ecg.paths import REPO_ROOT

WEB_DIR = REPO_ROOT / "web"
# En Windows, mimetypes lee el registro y puede servir .js como text/plain; los módulos ES fallan
MIME = {".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
        ".onnx": "application/octet-stream", ".csv": "text/csv", ".txt": "text/plain",
        ".css": "text/css", ".html": "text/html"}


def _vercel_headers() -> list[tuple[re.Pattern[str], list[dict[str, str]]]]:
    rules = json.loads((WEB_DIR / "vercel.json").read_text(encoding="utf-8"))["headers"]
    return [(re.compile("^" + r["source"].replace("(.*)", ".*") + "$"), r["headers"])
            for r in rules]


class Handler(SimpleHTTPRequestHandler):
    extensions_map = {**SimpleHTTPRequestHandler.extensions_map, **MIME}
    rules = _vercel_headers()

    def end_headers(self) -> None:
        path = self.path.split("?")[0]
        for pattern, headers in self.rules:
            if pattern.match(path):
                for h in headers:
                    self.send_header(h["key"], h["value"])
        super().end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(description="Sirve web/ en local.")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=WEB_DIR))
    print(f"Interfaz en http://localhost:{args.port}  (Ctrl+C para detener)")
    server.serve_forever()


if __name__ == "__main__":
    main()
