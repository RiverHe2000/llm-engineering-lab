"""Download the Tiny Shakespeare corpus (~1.1 MB, public domain) used for the demo run.

Usage: python scripts/download_data.py [--out data/tinyshakespeare.txt]
"""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
EXPECTED_BYTES = 1_115_394


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/tinyshakespeare.txt")
    args = parser.parse_args()
    out = Path(args.out)
    if out.exists() and out.stat().st_size == EXPECTED_BYTES:
        print(f"{out} already present ({EXPECTED_BYTES} bytes)")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    print(f"downloading {URL}")
    with urllib.request.urlopen(URL, timeout=60) as resp:
        data = resp.read()
    if len(data) != EXPECTED_BYTES:
        raise SystemExit(f"unexpected size {len(data)} (expected {EXPECTED_BYTES})")
    out.write_bytes(data)
    print(f"saved {out} sha256={hashlib.sha256(data).hexdigest()[:16]}")


if __name__ == "__main__":
    main()
