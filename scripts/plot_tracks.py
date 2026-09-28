"""Plot the 3 tracks (centerline + both lanes) for a visual sanity check.

Usage: python scripts/plot_tracks.py [out.png]
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from covfrenet.track import Track  # noqa: E402

TRACK_DIR = Path(__file__).resolve().parents[1] / "configs" / "track"
NAMES = ["T_oval", "T_mixed", "T_tight"]


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tracks.png")
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, name in zip(axes, NAMES):
        with open(TRACK_DIR / f"{name}.yaml") as f:
            waypoints = np.array(yaml.safe_load(f)["waypoints"])
        track = Track(waypoints, lane_gap=1.8)
        s = np.linspace(0, track.length, 400)
        for d, style, label in [
            (0.0, "k--", "centerline"),
            (track.lane_center_d(0), "b-", "lane 0"),
            (track.lane_center_d(1), "r-", "lane 1"),
        ]:
            x, y = track.frenet_to_xy(s, d)
            ax.plot(x, y, style, label=label, linewidth=1)
        ax.set_title(f"{name} ({track.length:.0f} m)")
        ax.set_aspect("equal")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
