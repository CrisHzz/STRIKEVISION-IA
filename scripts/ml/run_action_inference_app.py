"""Launch the local Gradio tester for an action-recognition candidate."""

from __future__ import annotations

import argparse
import socket
from pathlib import Path

from ufc_tracker.tracking.device import resolve_tracking_device
from ufc_tracker.ui.action_inference_app import ENGINE, build_app


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
    parser.add_argument("--candidate-dir", type=Path, help="Select a specific trained candidate for this session.")
    parser.add_argument("--tracking-device", help="Override tracking hardware: auto, cpu or cuda:N.")
    parser.add_argument("--pose-python", type=Path, help="Run tracking and pose in this separate Python environment.")
    args = parser.parse_args()
    if args.pose_python is not None and not args.pose_python.is_file():
        parser.error(f"No existe el Python de poses: {args.pose_python}")
    if args.pose_python is None and args.tracking_device is not None:
        try:
            resolve_tracking_device(args.tracking_device)
        except ValueError as error:
            parser.error(str(error))
    port = available_port(args.port)
    if port != args.port:
        print(f"El puerto {args.port} está ocupado; se usará http://127.0.0.1:{port}")
    build_app(args.candidate_dir, tracking_device=args.tracking_device, pose_python=args.pose_python).queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=port, inbrowser=not args.no_browser,
        allowed_paths=[str(ENGINE.output_root)],
    )


if __name__ == "__main__":
    main()
