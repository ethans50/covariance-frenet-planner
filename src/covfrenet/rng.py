"""Deterministic seeding -- never use the built-in `hash()` (varies run to
run under PYTHONHASHSEED) or the global `np.random`.
"""
from __future__ import annotations

import hashlib


def trial_seed(scenario_id: str, params: tuple, idx: int) -> int:
    """Derive a deterministic seed from (scenario, sweep params, trial index)."""
    key = f"{scenario_id}|{params}|{idx}"
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)
