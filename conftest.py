"""
conftest.py — pytest root configuration.

Adds the project root to sys.path so that `server.*` and `worker.*`
imports resolve without requiring a full `pip install -e .`.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
