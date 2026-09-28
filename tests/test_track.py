"""Stage 0 exit criterion: (x,y) -> (s,d) -> (x,y) roundtrip error < 1 mm."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from covfrenet.track import Track

TRACK_DIR = Path(__file__).resolve().parents[1] / "configs" / "track"
TRACK_NAMES = ["T_oval", "T_mixed", "T_tight"]


def load_track(name: str, lane_gap: float = 1.8) -> Track:
    with open(TRACK_DIR / f"{name}.yaml") as f:
        data = yaml.safe_load(f)
    return Track(np.array(data["waypoints"]), lane_gap)


@pytest.mark.parametrize("name", TRACK_NAMES)
def test_frenet_roundtrip(name):
    track = load_track(name)
    rng = np.random.default_rng(0)
    s_samples = rng.uniform(0, track.length, size=30)
    for s in s_samples:
        # Stay well inside the tightest curvature radius on this track.
        max_d = min(2.0, 0.5 / (abs(track.curvature(s)) + 1e-9))
        d = rng.uniform(-max_d, max_d)
        x, y = track.frenet_to_xy(s, d)
        s2, d2 = track.xy_to_frenet(x, y)
        x2, y2 = track.frenet_to_xy(s2, d2)
        err = np.hypot(x2 - x, y2 - y)
        assert err < 1e-3, f"{name}: s={s:.2f} d={d:.2f} roundtrip error={err * 1000:.3f} mm"


@pytest.mark.parametrize("name", TRACK_NAMES)
def test_lane_center_symmetric(name):
    track = load_track(name)
    assert track.lane_center_d(0) == -track.lane_gap / 2
    assert track.lane_center_d(1) == track.lane_gap / 2


def test_track_lengths_in_reported_range():
    # Paper: lap length ~700-1000 m (Sec 1/4.2); exact geometry unspecified.
    for name in TRACK_NAMES:
        length = load_track(name).length
        assert 650 <= length <= 1000, f"{name}: length={length:.1f} m out of range"
