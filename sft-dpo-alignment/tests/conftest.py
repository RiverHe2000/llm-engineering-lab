"""Shared test configuration.

One rule: a test marked ``network`` is skipped whenever the environment says the Hugging Face
hub is off limits. CI sets ``HF_HUB_OFFLINE=1`` and has no cached weights, so without this
hook the marker's own description -- "skipped in CI" -- was a promise nothing kept, and the
corpus-budget test failed there on a tokenizer download it could never make. Locally, where
the tokenizer is cached and the variable is unset, the test runs.
"""

from __future__ import annotations

import os

import pytest

OFFLINE = os.environ.get("HF_HUB_OFFLINE") == "1" or os.environ.get("TRANSFORMERS_OFFLINE") == "1"


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Skip ``network`` tests when the hub is unreachable by policy."""
    if not OFFLINE:
        return
    skip = pytest.mark.skip(reason="needs the Hugging Face hub; HF_HUB_OFFLINE is set")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
