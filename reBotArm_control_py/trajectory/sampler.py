"""reBot-DevArm 轨迹采样模块。

提供 SE(3) 测地线插值、三种时间剖面的离散采样，输出笛卡尔轨迹点。
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List

import numpy as np
import pinocchio as pin


class TrajProfile(enum.Enum):
    LINEAR = "linear"
    MIN_JERK = "min_jerk"
    TRAPEZOID = "trapezoid"


@dataclass
class TrajPlanParams:
    dt: float = 0.02
    profile: TrajProfile = TrajProfile.MIN_JERK
    accel_ratio: float = 0.25


@dataclass
class CartesianPoint:
    time: float
    pose: pin.SE3


@dataclass
class CartesianTrajectory:
    points_: List[CartesianPoint] = field(default_factory=list)

    def add_point(self, t: float, pose: pin.SE3) -> None:
        self.points_.append(CartesianPoint(t, pose))

    def duration(self) -> float:
        return self.points_[-1].time if self.points_ else 0.0

    def points(self) -> List[CartesianPoint]:
        return self.points_


@dataclass
class CartesianTrajectoryResult:
    trajectory: CartesianTrajectory
    n_points: int


def profile_state(t: float, profile: TrajProfile, accel_ratio: float = 0.25):
    """Return path progress and its first/second normalized-time derivatives."""
    t = max(0.0, min(1.0, t))
    if profile == TrajProfile.LINEAR:
        return t, 1.0, 0.0
    if profile == TrajProfile.MIN_JERK:
        return (t ** 3 * (10.0 - 15.0 * t + 6.0 * t * t),
                30.0 * t * t * (1.0 - t) ** 2,
                60.0 * t * (1.0 - t) * (1.0 - 2.0 * t))
    if profile == TrajProfile.TRAPEZOID:
        ta = max(0.01, min(0.49, accel_ratio))
        vm = 1.0 / (1.0 - ta)
        if t <= ta:
            return 0.5 * vm / ta * t * t, vm / ta * t, vm / ta
        if t <= 1.0 - ta:
            return vm * (t - 0.5 * ta), vm, 0.0
        dt = 1.0 - t
        return 1.0 - 0.5 * vm / ta * dt * dt, vm / ta * dt, -vm / ta
    raise ValueError(f"Unknown trajectory profile: {profile}")


def _apply_profile(t: float, profile: TrajProfile, accel_ratio: float) -> float:
    return profile_state(t, profile, accel_ratio)[0]


def _se3_interpolate(a, b, s) -> pin.SE3:
    """SE(3) 测地线插值：s∈[0,1] 从 a 到 b。接受 SE3 对象或 (4,4) ndarray。"""
    if isinstance(a, np.ndarray):
        a = pin.SE3(a)
    if isinstance(b, np.ndarray):
        b = pin.SE3(b)
    return a * pin.exp6(pin.log6(a.inverse() * b) * s)


def plan_cartesian_geodesic_trajectory(
    start_pose: pin.SE3,
    end_pose: pin.SE3,
    duration: float,
    params: TrajPlanParams | None = None,
) -> CartesianTrajectoryResult:
    """采样 SE(3) 测地线路径。

    参数:
        start_pose: 起始位姿。
        end_pose:   终止位姿。
        duration:   总时长（秒），必须 > 0。
        params:     采样参数（默认 :class:`TrajPlanParams`）。

    返回:
        :class:`CartesianTrajectoryResult`。
    """
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration 必须 > 0")
    if params is None:
        params = TrajPlanParams()
    if not np.isfinite(params.dt) or params.dt <= 0.0:
        raise ValueError("dt must be finite and positive")

    traj = CartesianTrajectory()
    n = max(2, int(np.ceil(duration / params.dt)) + 1)
    dt = duration / (n - 1)

    for i in range(n):
        t = i * dt
        s = _apply_profile(t / duration, params.profile, params.accel_ratio)
        traj.add_point(t, _se3_interpolate(start_pose, end_pose, s))

    return CartesianTrajectoryResult(trajectory=traj, n_points=n)
