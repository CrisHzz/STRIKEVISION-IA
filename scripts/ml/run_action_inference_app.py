"""Launch the local Gradio tester for an action-recognition candidate."""

from __future__ import annotations

import argparse
import socket

from ufc_tracker.ui.action_inference_app import build_app


def available_port(preferred: int, attempts: int = 20) -> int:
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"No hay puertos disponibles entre {preferred} y {preferred + attempts - 1}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7865)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    port = available_port(args.port)
    if port != args.port:
        print(f"El puerto {args.port} está ocupado; se usará http://127.0.0.1:{port}")
    build_app().queue(default_concurrency_limit=1).launch(server_name=args.host, server_port=port, inbrowser=not args.no_browser)


if __name__ == "__main__":
    main()
