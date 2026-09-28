"""Ego vehicle kinematics: kinematic bicycle model with first-order actuator lag.

Paper: Shin et al. 2025 Sec 2.1 defines EKF state [x, y, theta, v, delta]; the
simulated ego vehicle reuses the same layout. Sec 2.2/5 note the planner
itself ignores vehicle motion constraints (an acknowledged limitation) -- the
physical constraints applied below are not in the paper; chosen as plausible
limits for this vehicle class. Kanayama control (Sec 2.2) commands [v, delta]
directly; the accel / steer-rate limits model how fast the real vehicle can
reach that command.
"""
from __future__ import annotations

import numpy as np


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def step(state, command, dt, cfg):
    """Integrate one step toward a commanded [v, delta], subject to actuator limits.

    Args:
        state: (5,) [x, y, theta, v, delta].
        command: (2,) [v_cmd, delta_cmd] from the controller.
        dt: integration timestep (s).
        cfg: needs `wheelbase`, `accel_max`, `decel_max`, `steer_max_rad`,
             `steer_rate_max` (assumed; not specified by the paper).
    Returns:
        (5,) next state.
    """
    x, y, theta, v, delta = state
    v_cmd, delta_cmd = command

    a = np.clip((v_cmd - v) / dt, cfg.decel_max, cfg.accel_max)
    v_next = v + a * dt

    delta_rate = np.clip(
        wrap_to_pi(delta_cmd - delta) / dt, -cfg.steer_rate_max, cfg.steer_rate_max
    )
    delta_next = np.clip(delta + delta_rate * dt, -cfg.steer_max_rad, cfg.steer_max_rad)

    theta_next = wrap_to_pi(theta + v / cfg.wheelbase * np.tan(delta) * dt)
    x_next = x + v * np.cos(theta) * dt
    y_next = y + v * np.sin(theta) * dt

    return np.array([x_next, y_next, theta_next, v_next, delta_next])
