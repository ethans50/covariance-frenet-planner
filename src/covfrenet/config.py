"""YAML config loading with attribute-style access (cfg.cost.safety_radius)."""
from __future__ import annotations

from types import SimpleNamespace

import yaml


def _to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_namespace(v) for v in obj]
    return obj


def load_config(path):
    with open(path) as f:
        return _to_namespace(yaml.safe_load(f))
