"""Continuous-seed Cartesian IK with bounded steps and active joint limits."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pinocchio as pin

from ..kinematics.inverse_kinematics import (
    _clamp_config, _compute_error, _damped_step_with_active_limits,
)
from ..kinematics.robot_model import pad_q_for_model


@dataclass
class IKParams:
    max_iter: int = 200
    tolerance: float = 1e-8
    damping: float = 1e-6
    step_size: float = 0.8
    max_step: float = 0.05
    relative_tolerance: float = 0.01


@dataclass
class JointTrajectoryPoint:
    time: float
    q: np.ndarray
    ik_success: bool


def _joint_limit_grad(model, q):
    """Bounded centering gradient; no 1/distance singularity at a limit."""
    lo = np.asarray(model.lowerPositionLimit)
    hi = np.asarray(model.upperPositionLimit)
    grad = np.zeros(model.nv)
    for joint in model.joints[1:]:
        if joint.nq != 1 or joint.nv != 1:
            continue
        i = joint.idx_q
        if np.isfinite(lo[i]) and np.isfinite(hi[i]) and hi[i] > lo[i]:
            grad[joint.idx_v] = np.clip((lo[i] + hi[i] - 2 * q[i])
                                      / (hi[i] - lo[i]), -1.0, 1.0)
    return grad


def _null_step(J, gradient, gain):
    """Use a true SVD null space, not a damped pseudo-projection."""
    _, singular, vt = np.linalg.svd(J, full_matrices=True)
    cutoff = max(J.shape) * np.finfo(float).eps * (singular[0] if len(singular) else 1)
    rank = int(np.sum(singular > cutoff))
    basis = vt[rank:].T
    return gain * basis @ (basis.T @ gradient)


def track_trajectory(model: pin.Model, end_frame_id: int, traj,
                     q_init: np.ndarray, ik_params: IKParams | None = None,
                     null_gain: float = 0.0,
                     controlled_joints: int | None = None,
                     stop_on_failure: bool = False) -> List[JointTrajectoryPoint]:
    """Solve successive path points; always return an explicit success flag.

    Tolerance shrinks with the spatial increment, so slowly changing targets
    cannot disappear into a fixed stopping deadband. Passive joints stay fixed.
    Failed points must be rejected by the caller before hardware execution.
    """
    p = ik_params or IKParams()
    if (p.max_iter <= 0 or not np.isfinite(p.tolerance) or p.tolerance <= 0
            or not np.isfinite(p.damping) or p.damping < 0
            or not 0 < p.step_size <= 1 or p.max_step <= 0
            or not np.isfinite(p.max_step) or not 0 < p.relative_tolerance < 1
            or not np.isfinite(null_gain) or null_gain < 0):
        raise ValueError("Invalid trajectory IK parameters")
    q = np.asarray(q_init, dtype=float).copy()
    if q.shape != (model.nq,) or not np.all(np.isfinite(q)):
        raise ValueError("q_init must be a finite full model configuration")
    n_ctrl = model.nq if controlled_joints is None else controlled_joints
    if not 0 < n_ctrl <= model.nq:
        raise ValueError("Invalid controlled joint count")
    if not isinstance(end_frame_id, (int, np.integer)) or not 0 <= end_frame_id < model.nframes:
        raise ValueError("Invalid end-effector frame index")
    if any(j.idx_q < n_ctrl < j.idx_q + j.nq for j in model.joints[1:]):
        raise ValueError("controlled_joints must not split a joint configuration")
    q = pad_q_for_model(model, q)
    active = np.array([i for joint in model.joints[1:] if joint.idx_q < n_ctrl
                       for i in range(joint.idx_v, joint.idx_v + joint.nv)], dtype=int)
    data = model.createData()
    result = []
    previous_pose = None
    previous_time = -np.inf
    for pt in traj.points():
        if not np.isfinite(pt.time) or pt.time <= previous_time:
            raise ValueError("Trajectory times must be finite and strictly increasing")
        previous_time = pt.time
        tolerance = p.tolerance
        if previous_pose is not None:
            spacing = float(np.linalg.norm(pin.log6(previous_pose.inverse() * pt.pose).vector))
            if spacing > 0:
                tolerance = min(tolerance, max(1e-12, p.relative_tolerance * spacing))
        previous_pose = pt.pose
        error_norm, error = _compute_error(model, data, end_frame_id, q, pt.pose)
        for _ in range(p.max_iter):
            if error_norm <= tolerance:
                break
            pin.computeJointJacobians(model, data, q)
            displacement = data.oMf[end_frame_id].inverse() * pt.pose
            J = pin.Jlog6(displacement.inverse()) @ pin.getFrameJacobian(
                model, data, end_frame_id, pin.ReferenceFrame.LOCAL)
            damping = p.damping * max(1.0, error_norm * 10.0)
            dq = _damped_step_with_active_limits(
                model, q, J, error, damping, p.step_size, n_ctrl)
            if null_gain > 0:
                dq[active] += _null_step(J[:, active], _joint_limit_grad(model, q)[active], null_gain)
            magnitude = float(np.max(np.abs(dq)))
            if magnitude < 1e-14:
                break
            dq *= min(1.0, p.max_step / magnitude)
            alpha = 1.0
            for _ in range(12):
                candidate = _clamp_config(model, pin.integrate(model, q, alpha * dq))
                candidate[n_ctrl:] = q[n_ctrl:]
                next_norm, next_error = _compute_error(model, data, end_frame_id, candidate, pt.pose)
                if next_norm < error_norm:
                    q, error_norm, error = candidate, next_norm, next_error
                    break
                alpha *= 0.5
            else:
                break
        result.append(JointTrajectoryPoint(pt.time, q.copy(), error_norm <= tolerance))
        if stop_on_failure and not result[-1].ik_success:
            break
    return result
