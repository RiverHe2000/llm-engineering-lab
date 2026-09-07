"""`python -m sftdpo`, so the package runs without its console script being installed.

The whole module is one call, on purpose. `cli.main` returns an exit code rather than calling
`sys.exit`, which keeps every command testable as a function; turning that number into a
process status is the one thing that has to happen at the process boundary, and this is the
boundary.
"""

from __future__ import annotations

from sftdpo.cli import main

__all__ = ["main"]

if __name__ == "__main__":
    raise SystemExit(main())
