"""Generate track waypoint files (configs/track/*.yaml). Run once; commit output.

Track shapes are not specified in the paper -- only lap length (700-1000 m)
is reported. These are simple rounded-polygon shapes chosen to hit that
range and to give each track a distinct character (oval / mixed curvature /
tight curves). Run this script from the repo root:
`python scripts/gen_tracks.py`.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covfrenet.track import Track  # noqa: E402


def stadium(straight_len: float, radius: float, n_arc: int = 24) -> np.ndarray:
    """Two straights + two semicircular ends (a running-track shape)."""
    half = straight_len / 2
    bottom = [[-half, -radius], [half, -radius]]
    right_arc = [
        [half + radius * np.sin(a), -radius * np.cos(a)]
        for a in np.linspace(0, np.pi, n_arc, endpoint=False)[1:]
    ]
    top = [[half, radius], [-half, radius]]
    left_arc = [
        [-half + radius * np.sin(a), radius * np.cos(a)]
        for a in np.linspace(0, np.pi, n_arc, endpoint=False)[1:]
    ]
    return np.array(bottom + right_arc + top + left_arc)


def fillet_polygon(vertices: np.ndarray, radii, n_arc: int = 16) -> np.ndarray:
    """Closed polygon with each corner rounded by a circular fillet.

    Args:
        vertices: (P, 2) simple closed polygon vertices.
        radii: length-P fillet radius at each vertex (< half the adjacent
            edge lengths, or fillets overlap).
    """
    V = np.asarray(vertices, dtype=float)
    P = len(V)
    pts = []
    for i in range(P):
        prev_v, v, next_v = V[i - 1], V[i], V[(i + 1) % P]
        u_in = (v - prev_v) / np.linalg.norm(v - prev_v)
        u_out = (next_v - v) / np.linalg.norm(next_v - v)
        theta = np.arctan2(
            u_in[0] * u_out[1] - u_in[1] * u_out[0], u_in[0] * u_out[0] + u_in[1] * u_out[1]
        )
        r = radii[i]
        t = r * np.tan(abs(theta) / 2)
        tangent_in = v - u_in * t
        tangent_out = v + u_out * t
        sign = 1.0 if theta >= 0 else -1.0
        rot = np.array([[0, -sign], [sign, 0]])
        n_in = rot @ u_in
        center = tangent_in + n_in * r
        a0 = np.arctan2(tangent_in[1] - center[1], tangent_in[0] - center[0])
        a1 = a0 + sign * abs(theta)
        angles = np.linspace(a0, a1, n_arc, endpoint=False)
        pts.append(tangent_in)
        for a in angles[1:]:
            pts.append(center + r * np.array([np.cos(a), np.sin(a)]))
        pts.append(tangent_out)
    return np.array(pts)


def report_length(name: str, waypoints: np.ndarray, lane_gap: float = 1.8) -> float:
    length = Track(waypoints, lane_gap).length
    print(f"{name}: {length:.1f} m ({len(waypoints)} waypoints)")
    return length


def main():
    out_dir = Path(__file__).resolve().parents[1] / "configs" / "track"
    out_dir.mkdir(parents=True, exist_ok=True)

    tracks = {}

    # T_oval: two long straights + two semicircular ends. Target ~750 m.
    tracks["T_oval"] = stadium(straight_len=255.9, radius=27.4)

    # T_mixed: rounded rectangle alternating a gentle (large-radius) and a
    # sharp (small-radius) turn on each side. Target ~900 m.
    rect = np.array([[169.0, -67.6], [169.0, 67.6], [-169.0, 67.6], [-169.0, -67.6]])
    tracks["T_mixed"] = fillet_polygon(rect, radii=[45.1, 9.0, 45.1, 9.0])

    # T_tight: small rounded rectangle, sharp corners throughout. Target ~700 m.
    rect_tight = np.array([[124.3, -56.5], [124.3, 56.5], [-124.3, 56.5], [-124.3, -56.5]])
    tracks["T_tight"] = fillet_polygon(rect_tight, radii=[13.6, 13.6, 13.6, 13.6])

    for name, waypoints in tracks.items():
        report_length(name, waypoints)
        with open(out_dir / f"{name}.yaml", "w") as f:
            yaml.safe_dump(
                {"name": name, "waypoints": [[float(x), float(y)] for x, y in waypoints]},
                f,
                sort_keys=False,
            )


if __name__ == "__main__":
    main()
