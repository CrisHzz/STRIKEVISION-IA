"""Launch the combined fighter pose and strike/no_strike tester."""

import argparse

from ufc_tracker.ui.pose_strike_inference_app import build_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, inbrowser=not args.no_browser,
    )


if __name__ == "__main__":
    main()
