"""Launch the strike-only action annotation UI."""
import argparse
import socket

from ufc_tracker.detection.weights import project_root
from ufc_tracker.ui.action_annotation_app import build_app


def available_port(preferred: int, attempts: int = 20) -> int:
    """Return the preferred local port, or the next available port after it."""
    for port in range(preferred, preferred + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port
    raise OSError(f"No hay puertos disponibles entre {preferred} y {preferred + attempts - 1}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=7864)
    parser.add_argument('--no-browser', action='store_true')
    parser.add_argument('--annotation-dir', default=None, help='Optional separate output directory for annotation or UI testing.')
    args = parser.parse_args()
    port = available_port(args.port)
    if port != args.port:
        print(f"El puerto {args.port} está ocupado; se usará http://127.0.0.1:{port}")
    root = project_root()
    build_app(annotation_dir=args.annotation_dir).queue(default_concurrency_limit=1).launch(
        server_name='127.0.0.1',
        server_port=port,
        inbrowser=not args.no_browser,
        allowed_paths=[str(root)],
    )


if __name__ == '__main__':
    main()
