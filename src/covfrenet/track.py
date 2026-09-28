"""Centerline, lane geometry, and Frenet <-> Cartesian conversion.

Paper: Shin et al. 2025 Sec 2.2 defines the planner in Frenet frame but gives
no track geometry or lane gap `l` -- both are assumed and loaded from
config, never hardcoded here.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


class Track:
    """Closed-loop centerline, arc-length parameterized by a cubic spline.

    Frenet frame: s = arc length along the centerline (wraps at `length`),
    d = signed lateral offset, positive to the left of the direction of travel.
    """

    def __init__(self, waypoints: np.ndarray, lane_gap: float, n_lut: int = 20000):
        """
        Args:
            waypoints: (P, 2) closed-loop control points, first point not repeated.
            lane_gap: l (Eq. 8): distance between the two lanes.
            n_lut: resolution of the internal s<->u lookup table.
        """
        waypoints = np.asarray(waypoints, dtype=float)
        assert waypoints.ndim == 2 and waypoints.shape[1] == 2 and len(waypoints) >= 3
        wp = np.vstack([waypoints, waypoints[0]])
        chord = np.linalg.norm(np.diff(wp, axis=0), axis=1)
        assert np.all(chord > 0), "duplicate consecutive waypoints"
        u = np.concatenate([[0.0], np.cumsum(chord)])
        self.u_total = u[-1]
        self._sx = CubicSpline(u, wp[:, 0], bc_type="periodic")
        self._sy = CubicSpline(u, wp[:, 1], bc_type="periodic")

        # u (chord-length parameter) != true arc length s. Build a lookup table
        # by integrating speed(u) = |d(x,y)/du| over u, then invert s <-> u by
        # interpolation. n_lut controls roundtrip precision (Stage 0 exit: <1mm).
        uu = np.linspace(0.0, self.u_total, n_lut, endpoint=False)
        speed = np.hypot(self._sx(uu, 1), self._sy(uu, 1))
        assert np.all(speed > 1e-9), "zero-speed point on centerline spline"
        ds = speed * (self.u_total / n_lut)
        s_of_u = np.concatenate([[0.0], np.cumsum(ds)])
        self.length = s_of_u[-1]
        self._u_lut = np.concatenate([uu, [self.u_total]])
        self._s_lut = s_of_u
        self.lane_gap = lane_gap

    def _u_to_s(self, u):
        return np.interp(np.mod(u, self.u_total), self._u_lut, self._s_lut)

    def _s_to_u(self, s):
        return np.interp(np.mod(s, self.length), self._s_lut, self._u_lut)

    def curvature(self, s):
        """Signed curvature kappa(s) of the centerline (1/m)."""
        u = self._s_to_u(s)
        dx, dy = self._sx(u, 1), self._sy(u, 1)
        ddx, ddy = self._sx(u, 2), self._sy(u, 2)
        speed = np.hypot(dx, dy)
        return (dx * ddy - dy * ddx) / speed**3

    def heading(self, s):
        u = self._s_to_u(s)
        return np.arctan2(self._sy(u, 1), self._sx(u, 1))

    def frenet_to_xy(self, s, d):
        """(s, d) -> (x, y). Asserts |d| stays inside the local curvature radius
        (Frenet mapping is not unique once d exceeds 1/|kappa|)."""
        u = self._s_to_u(s)
        px, py = self._sx(u), self._sy(u)
        tx, ty = self._sx(u, 1), self._sy(u, 1)
        speed = np.hypot(tx, ty)
        nx, ny = -ty / speed, tx / speed
        kappa = self.curvature(s)
        assert np.all(np.abs(np.asarray(d) * kappa) < 0.999), (
            f"Frenet mapping not unique: d={d}, kappa={kappa}"
        )
        return px + d * nx, py + d * ny

    def xy_to_frenet(self, x, y):
        """(x, y) -> (s, d): nearest point on the centerline, via coarse LUT
        search + local scalar refinement.

        ponytail: O(n_lut) coarse search + a scalar minimize per query point.
        Fine for tests/offline analysis; revisit if plan_time_us (Stage 1 exit,
        40 Hz budget) shows this is a hot-loop bottleneck once wired into
        planner.py -- vectorizing the refine step would be the upgrade.
        """
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        y_arr = np.atleast_1d(np.asarray(y, dtype=float))
        s_out = np.empty_like(x_arr)
        d_out = np.empty_like(x_arr)
        lut_u = self._u_lut[:-1]
        lut_x = self._sx(lut_u)
        lut_y = self._sy(lut_u)
        h = 2 * self.u_total / len(lut_u) + 1e-9
        for i, (px, py) in enumerate(zip(x_arr, y_arr)):
            d2 = (lut_x - px) ** 2 + (lut_y - py) ** 2
            u0 = lut_u[np.argmin(d2)]
            res = minimize_scalar(
                lambda u: (self._sx(u) - px) ** 2 + (self._sy(u) - py) ** 2,
                bounds=(u0 - h, u0 + h),
                method="bounded",
                options={"xatol": 1e-12},
            )
            u_star = res.x
            tx, ty = self._sx(u_star, 1), self._sy(u_star, 1)
            speed = np.hypot(tx, ty)
            nx, ny = -ty / speed, tx / speed
            s_out[i] = self._u_to_s(u_star)
            d_out[i] = (px - self._sx(u_star)) * nx + (py - self._sy(u_star)) * ny
        if np.isscalar(x):
            return float(s_out[0]), float(d_out[0])
        return s_out, d_out

    def lane_center_d(self, lane_id: int) -> float:
        """lane 0/1 -> signed d offset of that lane's centerline, +-lane_gap/2."""
        assert lane_id in (0, 1)
        return -self.lane_gap / 2 if lane_id == 0 else self.lane_gap / 2
