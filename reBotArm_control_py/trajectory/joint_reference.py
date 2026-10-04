"""Continuous joint paths with an independent, analytic time law.

Path knots are sampled in space, not in wall-clock time. Quintic segments
match position and two path derivatives; minimum-jerk time scaling gives
continuous position, velocity and acceleration, including the two endpoints.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .sampler import TrajProfile, profile_state


@dataclass(frozen=True)
class JointReferenceSample:
    q: np.ndarray
    qd: np.ndarray
    qdd: np.ndarray


def joint_vector(value, n: int, label: str, *, positive: bool = False):
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        arr = np.full(n, float(arr))
    if arr.shape != (n,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{label} must be a finite scalar or {n} values")
    if positive and np.any(arr <= 0):
        raise ValueError(f"{label} must be positive")
    return arr.copy()


def _extrema(coeff):
    """Exact polynomial extrema on one normalized segment [0, 1]."""
    derivative = np.polynomial.polynomial.polyder(coeff)
    roots = np.polynomial.polynomial.polyroots(derivative)
    inside = [r.real for r in roots if abs(r.imag) < 1e-9 and 0 < r.real < 1]
    return np.polynomial.polynomial.polyval([0.0, 1.0, *inside], coeff)


class JointPathReference:
    def __init__(self, positions, duration: float, *,
                 profile: TrajProfile = TrajProfile.MIN_JERK,
                 max_velocity=0.5, max_acceleration=1.0,
                 lower=None, upper=None, allow_retime: bool = True):
        q = np.asarray(positions, dtype=float)
        if q.ndim != 2 or len(q) < 2 or not np.all(np.isfinite(q)):
            raise ValueError("Path requires at least two finite joint configurations")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive")
        n = q.shape[1]
        vmax = joint_vector(max_velocity, n, "max_velocity", positive=True)
        amax = joint_vector(max_acceleration, n, "max_acceleration", positive=True)
        lo = np.full(n, -np.inf) if lower is None else np.asarray(lower, dtype=float)
        hi = np.full(n, np.inf) if upper is None else np.asarray(upper, dtype=float)
        if (lo.shape != (n,) or hi.shape != (n,) or np.any(np.isnan(lo))
                or np.any(np.isnan(hi)) or np.any(lo > hi)):
            raise ValueError("Invalid joint position limits")
        h = 1.0 / (len(q) - 1)
        edge = 2 if len(q) > 2 else 1
        d = np.gradient(q, h, axis=0, edge_order=edge)
        dd = np.gradient(d, h, axis=0, edge_order=edge)
        c = np.zeros((len(q) - 1, 6, n))
        c[:, 0] = q[:-1]
        c[:, 1] = h * d[:-1]
        c[:, 2] = 0.5 * h * h * dd[:-1]
        A = q[1:] - c[:, :3].sum(axis=1)
        B = h * d[1:] - c[:, 1] - 2.0 * c[:, 2]
        C = h * h * dd[1:] - 2.0 * c[:, 2]
        c[:, 3] = 10.0 * A - 4.0 * B + 0.5 * C
        c[:, 4] = -15.0 * A + 7.0 * B - C
        c[:, 5] = 6.0 * A - 3.0 * B + 0.5 * C
        path_v = np.zeros(n)
        path_a = np.zeros(n)
        for segment in c:
            for j in range(n):
                values = _extrema(segment[:, j])
                if values.min() < lo[j] - 1e-10 or values.max() > hi[j] + 1e-10:
                    raise ValueError(f"Interpolated path exceeds joint {j + 1} limits")
                dc = np.polynomial.polynomial.polyder(segment[:, j])
                ddc = np.polynomial.polynomial.polyder(dc)
                path_v[j] = max(path_v[j], np.max(np.abs(_extrema(dc))) / h)
                path_a[j] = max(path_a[j], np.max(np.abs(_extrema(ddc))) / (h * h))
        if profile == TrajProfile.MIN_JERK:
            speed_bound, acceleration_bound = 1.875, 10.0 / np.sqrt(3.0)
        elif profile == TrajProfile.TRAPEZOID:
            speed_bound, acceleration_bound = 1.0 / 0.75, 1.0 / (0.75 * 0.25)
        elif profile == TrajProfile.LINEAR:
            speed_bound, acceleration_bound = 1.0, 0.0
        else:
            raise ValueError("Unsupported trajectory profile")
        required = max(float(np.max(path_v * speed_bound / vmax)),
                       float(np.sqrt(np.max((path_a * speed_bound ** 2
                                            + path_v * acceleration_bound) / amax))))
        if not allow_retime and required > duration * (1.0 + 1e-9):
            raise ValueError(f"Requested duration is too short; need at least {required:.3f}s")
        self.requested_duration = float(duration)
        self.duration = max(float(duration), required * (1.0 + 1e-9))
        self.profile = profile
        self._coeff = c
        self._derivative_coeff = np.arange(1, 6)[None, :, None] * c[:, 1:]
        self._second_derivative_coeff = np.arange(1, 5)[None, :, None] * self._derivative_coeff[:, 1:]
        self._h = h
        self._start = q[0].copy()
        self._end = q[-1].copy()
        self.velocity_bound = path_v * speed_bound / self.duration
        self.acceleration_bound = (path_a * speed_bound ** 2
                                   + path_v * acceleration_bound) / self.duration ** 2

    def sample_progress(self, progress: float):
        """Evaluate the geometric path and its derivatives with respect to progress."""
        u = float(np.clip(progress, 0.0, 1.0))
        index = min(int(u / self._h), len(self._coeff) - 1)
        w = (u - index * self._h) / self._h
        c = self._coeff[index]
        dc = self._derivative_coeff[index]
        ddc = self._second_derivative_coeff[index]
        return (np.polynomial.polynomial.polyval(w, c),
                np.polynomial.polynomial.polyval(w, dc) / self._h,
                np.polynomial.polynomial.polyval(w, ddc) / self._h ** 2)

    def sample(self, elapsed: float) -> JointReferenceSample:
        if not np.isfinite(elapsed):
            raise ValueError("elapsed must be finite")
        if elapsed <= 0 or elapsed >= self.duration:
            q = self._start if elapsed <= 0 else self._end
            return JointReferenceSample(q.copy(), np.zeros_like(q), np.zeros_like(q))
        s, ds, dds = profile_state(elapsed / self.duration, self.profile)
        q, qs, qss = self.sample_progress(s)
        return JointReferenceSample(q, qs * ds / self.duration,
                                    (qss * ds ** 2 + qs * dds) / self.duration ** 2)
