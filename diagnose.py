#!/usr/bin/env python3
"""
Diagnose why an RTSP stream can't be opened, with the real FFmpeg errors.

Usage:
    python diagnose.py "rtsp://admin:pass@192.168.100.5:554/ch=1&subtype=0"
    python diagnose.py "rtsp://..." --timeout 15

Quote the URL — an unquoted '&' breaks the command line.
Tip: close VLC and any other viewer first; many cameras allow only one client.
"""

import argparse
import json
import sys

from rtsp_backend.diagnostics import probe
from rtsp_backend.errors import RTSPBackendError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="RTSP URL to diagnose (quote it!)")
    parser.add_argument("--timeout", type=float, default=10.0,
                        help="per-attempt open timeout in seconds (default 10)")
    args = parser.parse_args()

    try:
        report = probe(args.url, open_timeout=args.timeout)
    except RTSPBackendError as exc:
        print(json.dumps(exc.to_dict(), indent=2))
        return 2

    print(json.dumps(report, indent=2))
    print("\nVERDICT:", report["verdict"], file=sys.stderr)
    return 0 if any(a["frame_decoded"] for a in report["attempts"]) else 1


if __name__ == "__main__":
    sys.exit(main())
