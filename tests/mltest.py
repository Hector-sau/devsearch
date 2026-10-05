"""Helpers for tests that need torch/transformers/tokenizers: skip cleanly where they are absent.

The dependency-free CI job runs the whole suite without these libraries (these tests then *skip*,
and the skip is reported as a skip, never as a pass). The ML CI job installs them and asserts that
nothing was skipped.
"""

import importlib.util
import unittest

HAVE_ML = all(importlib.util.find_spec(name) is not None for name in ("torch", "transformers", "tokenizers", "safetensors"))
requires_ml = unittest.skipUnless(HAVE_ML, "torch/transformers/tokenizers/safetensors are not installed")

# The HTTP service and its UI need the `serve` extra; without it those tests skip, never pass.
HAVE_SERVE = all(importlib.util.find_spec(name) is not None for name in ("fastapi", "uvicorn", "httpx", "streamlit"))
requires_serve = unittest.skipUnless(HAVE_SERVE, "fastapi/uvicorn/httpx/streamlit are not installed")
