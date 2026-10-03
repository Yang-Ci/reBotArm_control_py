"""Validated Cartesian paths and continuous, wall-clock joint references."""
from __future__ import annotations

import csv
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import pinocchio as pin

from ..actuator import RebotArm
from ..dynamics import compute_generalized_gravity
from ..kinematics import load_robot_model, pad_q_for_model, pos_rot_to_se3
from ..kinematics.inverse_kinematics import solve_ik, IKParams as PoseIKParams
from ..trajectory import (
    TrajProfile, TrajPlanParams, IKParams, JointPathReference,
    plan_cartesian_geodesic_trajectory, track_trajectory,
)
from ..trajectory.joint_reference import joint_vector
from .position_feedback import PositionFeedback


class RebotArmEndPose:
    """Cartesian motion with per-cycle position/velocity reference evaluation.

    ``move_to_traj`` follows a Cartesian path. Omitted RPY preserves the measured
    orientation. ``move_to_ik`` follows a smooth joint-space path to an IK goal.
    Commands during motion are rejected; ``stop_motion`` holds measured pose.
    All durations are seconds and all joint quantities use radians.
    """

    def __init__(self, rebotarm: RebotArm, dt: float = 0.01,
                 profile: TrajProfile = TrajProfile.MIN_JERK,
                 arm_control_mode: str | None = None, use_gravity_ff: bool = True):
        self.rebotarm = rebotarm
        self._arm_group = rebotarm.groups.get("arm")
        self._gripper_group = rebotarm.groups.get("gripper")
        self._has_gripper = rebotarm.has_gripper
        if self._arm_group is None:
            raise ValueError("Hardware configuration requires an arm group")
        self._arm_control_mode = arm_control_mode or rebotarm.arm_control_mode
        if self._arm_control_mode not in ("mit", "posvel"):
            raise ValueError("arm_control_mode must be 'mit' or 'posvel'")
        if not np.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be finite and positive")
        # Discontinuous speed/acceleration profiles are unsuitable for this controller.
        if profile != TrajProfile.MIN_JERK:
            raise ValueError("Hardware motion requires MIN_JERK for rest-to-rest continuity")
        self._profile = profile
        self._dt = dt
        self._n = self._arm_group.num_joints
        self._use_gravity_ff = use_gravity_ff
        import yaml
        hardware = yaml.safe_load(rebotarm.hardware_config_path.read_text(encoding="utf-8"))
        urdf = Path(hardware["urdf_path"])
        if not urdf.is_absolute():
            urdf = Path(__file__).resolve().parents[2] / urdf
        self._model = load_robot_model(str(urdf))
        controlled = [(name, joint) for name, joint in zip(self._model.names[1:], self._model.joints[1:])
                      if joint.idx_q < self._n]
        if ([name for name, _ in controlled] != self._arm_group.joint_names
                or any(j.nq != 1 or j.nv != 1 for _, j in controlled)):
            raise ValueError("Arm joint order/types do not match the URDF scalar joints")
        self._end_frame_id = self._model.getFrameId(hardware["end_effector_frame"])
        if self._end_frame_id >= self._model.nframes:
            raise ValueError("End-effector frame not found in URDF")
        # Each writer owns a distinct Pinocchio Data object.
        self._ik_data = self._model.createData()
        self._gravity_data = self._model.createData()
        self._log_data = self._model.createData()
        cfg = dict(rebotarm.motion_control)
        self._max_velocity = joint_vector(cfg.get("max_velocity", 0.5), self._n,
                                          "max_velocity", positive=True)
        self._max_acceleration = joint_vector(cfg.get("max_acceleration", 1.0), self._n,
                                              "max_acceleration", positive=True)
        if np.any(self._max_velocity > self._model.velocityLimit[:self._n]):
            raise ValueError("max_velocity exceeds URDF velocity limits")
        self._gravity_scale = joint_vector(cfg.get("gravity_scale", 1.0), self._n, "gravity_scale")
        model_effort = np.asarray(self._model.effortLimit[:self._n], dtype=float)
        self._ff_limit = joint_vector(cfg.get("feedforward_limit", model_effort), self._n,
                                      "feedforward_limit", positive=True)
        if np.any(self._ff_limit > model_effort):
            raise ValueError("feedforward_limit exceeds URDF effort limits")
        self._path_samples = int(cfg.get("path_samples", 201))
        if self._path_samples < 3:
            raise ValueError("path_samples must be at least 3")
        self._max_joint_step = self._positive(cfg, "max_path_joint_step", 0.05)
        self._path_tolerance = self._positive(cfg, "path_position_tolerance", 2e-5)
        self._rotation_tolerance = self._positive(cfg, "path_rotation_tolerance", 2e-5)
        self._max_feedback_age = self._positive(cfg, "max_feedback_age", 0.25)
        self._max_control_dt = self._positive(cfg, "max_control_dt", 0.05)
        self._max_tracking_error = self._positive(cfg, "max_tracking_error", 0.15)
        self._max_start_drift = self._positive(cfg, "max_start_drift", 0.01)
        self._ff_filter_time = self._positive(cfg, "feedforward_filter_time", 0.02)
        self._allow_retime = bool(cfg.get("allow_retime", True))
        self._log_period = 1.0 / self._positive(cfg, "log_rate", 50.0)
        self._feedback = PositionFeedback(
            self._arm_group, self._positive(cfg, "feedback_rate", 25.0),
            int(cfg.get("feedback_timeout_ms", 20)))
        friction = cfg.get("friction", {}) or {}
        self._friction_enabled = bool(friction.get("enabled", False))
        self._coulomb = joint_vector(friction.get("coulomb", 0.0), self._n, "friction.coulomb")
        self._static_friction = joint_vector(friction.get("static", self._coulomb), self._n,
                                             "friction.static")
        self._viscous = joint_vector(friction.get("viscous", 0.0), self._n, "friction.viscous")
        self._friction_limit = joint_vector(friction.get("limit", 0.5), self._n,
                                            "friction.limit", positive=True)
        self._friction_speed = self._positive(friction, "velocity_scale", 0.002)
        self._stribeck_speed = self._positive(friction, "stribeck_velocity", 0.02)
        if np.any(self._coulomb < 0) or np.any(self._viscous < 0):
            raise ValueError("Friction coefficients must be nonnegative")
        if np.any(self._static_friction < self._coulomb):
            raise ValueError("Static friction must be at least Coulomb friction")
        self._ik_solver_params = PoseIKParams(max_iter=300, tolerance=1e-8, step_size=0.8)
        self._clik_params = IKParams(max_iter=200, tolerance=1e-8)
        self._motion_lock = threading.RLock()
        self._plan_lock = threading.RLock()
        self._reference = None
        self._motion_started_at = None
        self._moving = False
        self._running = False
        self._fault = None
        self._q_target = np.zeros(self._n)
        self._qd_target = np.zeros(self._n)
        self._gripper_target = 0.0
        self._tau_filtered = None
        self._last_log_at = -np.inf
        self._motion_log = deque(maxlen=10000)
        self._last_plan = {}

    @staticmethod
    def _positive(cfg, key, default):
        value = float(cfg.get(key, default))
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{key} must be finite and positive")
        return value

    def _pose(self, q, data):
        pin.forwardKinematics(self._model, data, pad_q_for_model(self._model, q, self._n))
        pin.updateFramePlacements(self._model, data)
        return data.oMf[self._end_frame_id].copy()

    def start(self):
        if self._running:
            raise RuntimeError("Controller is already running")
        self.rebotarm.connect()
        # Acquire a real pose before enabling or sending the first target.
        sample = self._feedback.refresh()
        if time.monotonic() - sample.sampled_at > self._max_feedback_age:
            raise RuntimeError("Initial feedback is too old")
        with self._motion_lock:
            self._q_target = sample.q.copy()
            self._qd_target.fill(0)
            self._reference = None
            self._fault = None
            self._tau_filtered = None
        if self._has_gripper:
            qg, _, _ = self._gripper_group.read_position_sample()
            self._gripper_target = float(qg[0])
        ok = (self._arm_group.mode_mit() if self._arm_control_mode == "mit"
              else self._arm_group.mode_pos_vel(vlim=self._max_velocity))
        if not ok:
            raise RuntimeError("Arm mode switch failed")
        if self._has_gripper and not self._gripper_group.mode_mit():
            raise RuntimeError("Gripper mode switch failed")
        # Mode switching can take seconds, so reacquire before enabling.
        self._feedback.start()
        try:
            sample = self._fresh_sample()
            with self._motion_lock:
                self._q_target = sample.q.copy()
            if self._use_gravity_ff:
                self._tau_filtered = np.clip(compute_generalized_gravity(
                    self._model, pad_q_for_model(self._model, sample.q, self._n),
                    self._gravity_data)[:self._n] * self._gravity_scale,
                    -self._ff_limit, self._ff_limit)
            else:
                self._tau_filtered = np.zeros(self._n)
            if self._arm_control_mode == "mit":
                self._arm_group.send_mit(sample.q, vel=np.zeros(self._n),
                                         tau=self._tau_filtered, strict=True)
            else:
                self._arm_group.send_pos_vel(sample.q, vlim=self._max_velocity, strict=True)
            if self._has_gripper:
                self._gripper_group.send_mit(np.array([self._gripper_target]), strict=True)
            self.rebotarm.enable_all(strict=True)
            self._running = True
            self.rebotarm.start_control_loop(self._loop_cb)
        except Exception:
            self._running = False
            self._feedback.stop()
            raise

    def end(self):
        if not self._running:
            return
        try:
            self.safe_home()
        finally:
            self.rebotarm.stop_control_loop()
            self._feedback.stop()
            self.rebotarm.disconnect()
            self._running = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.end()

    def set_gripper_target(self, pos):
        if not np.isfinite(pos):
            raise ValueError("Gripper target must be finite")
        with self._motion_lock:
            self._gripper_target = float(pos)

    def open_gripper(self):
        # Preserve the original API's release behavior.
        if self._has_gripper:
            self._gripper_group._mit_kp.fill(0)
            self._gripper_group._mit_kd.fill(0)

    def close_gripper(self):
        self.set_gripper_target(0.0)

    def _fresh_sample(self):
        sample = self._feedback.latest()
        if time.monotonic() - sample.sampled_at > self._max_feedback_age:
            raise RuntimeError("Position feedback expired")
        return sample

    def get_joint_positions(self):
        """Return the same fresh arm position sample used by the controller."""
        return self._fresh_sample().q

    def _can_plan(self):
        with self._motion_lock:
            if not self._running or not self.rebotarm.control_loop_active:
                print("[motion] Control loop is not running")
                return False
            if self._fault or self._moving:
                print(f"[motion] Rejected: {self._fault or 'previous motion is still running'}")
                return False
        return True

    def _build_reference(self, positions, duration, max_velocity=None):
        return JointPathReference(
            positions, duration, profile=self._profile,
            max_velocity=self._max_velocity if max_velocity is None else max_velocity,
            max_acceleration=self._max_acceleration,
            lower=self._model.lowerPositionLimit[:self._n],
            upper=self._model.upperPositionLimit[:self._n], allow_retime=self._allow_retime)

    def _install(self, reference, q_start, details):
        sample = self._fresh_sample()
        if np.max(np.abs(sample.q - q_start)) > self._max_start_drift:
            raise RuntimeError("Arm moved during planning; trajectory must be replanned")
        with self._motion_lock:
            if self._fault or not self._running or not self.rebotarm.control_loop_active:
                raise RuntimeError("Controller became unavailable during planning")
            self._reference = reference
            # Start time is latched by the servo loop, at the first actual send.
            self._motion_started_at = None
            self._q_target = q_start.copy()
            self._qd_target.fill(0)
            self._moving = True
            self._motion_log.clear()
            self._last_log_at = -np.inf
            self._last_plan = dict(details, requested_duration=reference.requested_duration,
                                   actual_duration=reference.duration)
        if reference.duration > reference.requested_duration + 1e-6:
            print(f"[motion] Duration extended to {reference.duration:.3f}s for joint limits")

    def _target_pose(self, q, x, y, z, roll, pitch, yaw):
        values = np.array([x, y, z], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("Target position must be finite")
        rpy = (roll, pitch, yaw)
        if all(v is None for v in rpy):
            rotation = self._pose(q, self._ik_data).rotation.copy()
            return pos_rot_to_se3(values, rot=rotation)
        if any(v is None for v in rpy) or not np.all(np.isfinite(rpy)):
            raise ValueError("Supply all three finite RPY values or omit all three")
        return pos_rot_to_se3(values, roll=roll, pitch=pitch, yaw=yaw)

    def move_to_ik(self, x, y, z, roll=None, pitch=None, yaw=None, duration=2.0):
        """Solve a goal, then execute a smooth joint-space rest-to-rest move."""
        with self._plan_lock:
            if not self._can_plan():
                return False
            try:
                q_start = self._fresh_sample().q
                target = self._target_pose(q_start, x, y, z, roll, pitch, yaw)
                result = solve_ik(self._model, self._ik_data, self._end_frame_id, target,
                                  q_start, self._ik_solver_params, controlled_joints=self._n)
                if not result.success:
                    raise ValueError(f"Goal IK did not converge (error={result.error:.3e})")
                reference = self._build_reference([q_start, result.q], duration)
                self._install(reference, q_start, {"kind": "joint_ik", "ik_error": result.error})
                return True
            except (ValueError, RuntimeError) as error:
                print(f"[motion] {error}")
                return False

    def move_to_traj(self, x, y, z, roll=None, pitch=None, yaw=None, duration=2.0):
        """Plan spatial knots first; duration affects execution, never the IK path."""
        with self._plan_lock:
            if not self._can_plan():
                return False
            try:
                q_start = self._fresh_sample().q
                start = self._pose(q_start, self._ik_data)
                target = self._target_pose(q_start, x, y, z, roll, pitch, yaw)
                if duration <= 0:
                    duration = max(1.0, np.linalg.norm(target.translation - start.translation) / 0.1)
                cart = plan_cartesian_geodesic_trajectory(
                    start, target, 1.0,
                    TrajPlanParams(dt=1.0 / (self._path_samples - 1), profile=TrajProfile.LINEAR))
                points = track_trajectory(
                    self._model, self._end_frame_id, cart.trajectory,
                    pad_q_for_model(self._model, q_start, self._n),
                    self._clik_params, null_gain=0.0, controlled_joints=self._n,
                    stop_on_failure=True)
                if len(points) != len(cart.trajectory.points()) or not all(p.ik_success for p in points):
                    failed = next((i for i, p in enumerate(points) if not p.ik_success), -1)
                    raise ValueError(f"Cartesian path IK failed at knot {failed}")
                positions = np.array([p.q[:self._n] for p in points])
                if np.max(np.abs(np.diff(positions, axis=0))) > self._max_joint_step:
                    raise ValueError("Cartesian path contains a joint jump or approaches a singularity")
                reference = self._build_reference(positions, duration)
                # Validate between knots as well: smooth joints alone do not prove a Cartesian path.
                displacement = pin.log6(start.inverse() * target)
                max_position_error = max_rotation_error = 0.0
                validation_data = self._model.createData()
                previous_z = None
                vertical = (np.linalg.norm(target.translation[:2] - start.translation[:2]) < 1e-8
                            and np.linalg.norm(displacement.vector[3:]) < 1e-8)
                direction = np.sign(target.translation[2] - start.translation[2])
                for progress in np.linspace(0.0, 1.0, 4 * (len(positions) - 1) + 1):
                    q, _, _ = reference.sample_progress(progress)
                    actual = self._pose(q, validation_data)
                    desired = start * pin.exp6(displacement * progress)
                    error = pin.log6(actual.inverse() * desired).vector
                    max_position_error = max(max_position_error, float(np.linalg.norm(error[:3])))
                    max_rotation_error = max(max_rotation_error, float(np.linalg.norm(error[3:])))
                    if (vertical and previous_z is not None
                            and direction * (actual.translation[2] - previous_z) < -1e-8):
                        raise ValueError("Interpolated vertical path reverses direction")
                    previous_z = float(actual.translation[2])
                if max_position_error > self._path_tolerance or max_rotation_error > self._rotation_tolerance:
                    raise ValueError("Interpolated Cartesian path exceeds tracking tolerance")
                self._install(reference, q_start, {"kind": "cartesian", "path_knots": len(points),
                    "max_position_error": max_position_error, "max_rotation_error": max_rotation_error})
                return True
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                print(f"[motion] {error}")
                return False

    def move_relative(self, dx=0.0, dy=0.0, dz=0.0, duration=2.0):
        """Translate in the world frame while preserving the measured orientation."""
        with self._plan_lock:
            if not self._can_plan():
                return False
            try:
                pose = self._pose(self._fresh_sample().q, self._ik_data)
            except RuntimeError as error:
                print(f"[motion] {error}")
                return False
            p = pose.translation + np.array([dx, dy, dz])
            rpy = pin.rpy.matrixToRpy(pose.rotation)
            return self.move_to_traj(*p, *rpy, duration=duration)

    def stop_motion(self):
        with self._plan_lock:
            sample = self._feedback.latest()
            with self._motion_lock:
                self._reference = None
                self._moving = False
                self._q_target = sample.q.copy()
                self._qd_target.fill(0)

    def _set_fault(self, reason, measured_q):
        with self._motion_lock:
            first = self._fault is None
            self._fault = self._fault or reason
            self._reference = None
            self._moving = False
            self._q_target = measured_q.copy()
            self._qd_target.fill(0)
        if first:
            print(f"[motion/hold] {reason}")

    def clear_fault(self):
        sample = self._fresh_sample()
        if not self.rebotarm.control_loop_active:
            raise RuntimeError("Restart controller after a control-loop failure")
        with self._motion_lock:
            if self._moving:
                raise RuntimeError("Stop motion before clearing a fault")
            self._reference = None
            self._fault = None
            self._q_target = sample.q.copy()
            self._qd_target.fill(0)
            self._tau_filtered = None

    def _friction_torque(self, velocity):
        if not self._friction_enabled:
            return np.zeros(self._n)
        magnitude = self._coulomb + (self._static_friction - self._coulomb) * np.exp(
            -(velocity / self._stribeck_speed) ** 2)
        torque = magnitude * np.tanh(velocity / self._friction_speed) + self._viscous * velocity
        return np.clip(torque, -self._friction_limit, self._friction_limit)

    def _loop_cb(self, _: RebotArm, dt: float):
        # A returned stop/clear command cannot race with an old velocity send.
        with self._motion_lock:
            self._locked_loop_cb(_, dt)

    def _locked_loop_cb(self, _: RebotArm, dt: float):
        try:
            self._loop_cb_impl(_, dt)
        except Exception as error:
            # Do not leave a nonzero velocity command latched after a callback failure.
            try:
                q = self._feedback.latest().q
            except RuntimeError:
                q = self._q_target.copy()
            self._set_fault(f"Control callback failed: {error}", q)
            tau = np.zeros(self._n) if self._tau_filtered is None else self._tau_filtered.copy()
            try:
                if self._arm_control_mode == "mit":
                    self._arm_group.send_mit(q, vel=np.zeros(self._n), tau=tau, strict=True)
                else:
                    self._arm_group.send_pos_vel(q, vlim=self._max_velocity, strict=True)
            except Exception as hold_error:
                self.rebotarm.control_loop_stats["last_error"] = repr(hold_error)

    def _loop_cb_impl(self, _: RebotArm, dt: float):
        now = time.monotonic()
        sample = self._feedback.latest()
        age = now - sample.sampled_at
        if age > self._max_feedback_age:
            self._set_fault(f"Position feedback expired ({age:.3f}s)", sample.q)
        elif not np.isfinite(dt) or dt <= 0 or dt > self._max_control_dt:
            self._set_fault(f"Control interval out of range ({dt:.4f}s)", sample.q)
        with self._motion_lock:
            reference = self._reference
            elapsed = 0.0
            acceleration = np.zeros(self._n)
            if reference is not None:
                if self._motion_started_at is None:
                    self._motion_started_at = now
                elapsed = now - self._motion_started_at
                state = reference.sample(elapsed)
                self._q_target, self._qd_target = state.q, state.qd
                acceleration = state.qdd
                if elapsed >= reference.duration:
                    self._reference = None
                    self._moving = False
            q, velocity = self._q_target.copy(), self._qd_target.copy()
            gripper = self._gripper_target
            moving = self._moving
        if moving and np.max(np.abs(q - sample.q)) > self._max_tracking_error:
            self._set_fault("Joint tracking error exceeded limit", sample.q)
            q, velocity = sample.q.copy(), np.zeros(self._n)
            acceleration.fill(0)
        tau = np.zeros(self._n)
        if self._use_gravity_ff:
            tau = compute_generalized_gravity(
                self._model, pad_q_for_model(self._model, sample.q, self._n),
                self._gravity_data)[:self._n] * self._gravity_scale
        tau += self._friction_torque(velocity)
        tau = np.clip(tau, -self._ff_limit, self._ff_limit)
        if not np.all(np.isfinite(tau)):
            self._set_fault("Non-finite feedforward torque", sample.q)
            q, velocity = sample.q.copy(), np.zeros(self._n)
            tau = np.zeros(self._n) if self._tau_filtered is None else self._tau_filtered.copy()
        if self._tau_filtered is None:
            self._tau_filtered = tau.copy()
        else:
            blend = -np.expm1(-max(0.0, dt) / self._ff_filter_time)
            self._tau_filtered += blend * (tau - self._tau_filtered)
        from motorbridge import CallError
        try:
            if self._arm_control_mode == "mit":
                self._arm_group.send_mit(q, vel=velocity, tau=self._tau_filtered, strict=True)
            else:
                self._arm_group.send_pos_vel(q, vlim=self._max_velocity, strict=True)
            if self._has_gripper:
                self._gripper_group.send_mit(np.array([gripper]), strict=True)
        except CallError as error:
            self._set_fault(f"Motor command failed: {error}", sample.q)
            # Retry all arm joints with zero velocity immediately, including
            # joints that received a moving command before the partial failure.
            if self._arm_control_mode == "mit":
                self._arm_group.send_mit(sample.q, vel=np.zeros(self._n), tau=self._tau_filtered)
            else:
                self._arm_group.send_pos_vel(sample.q, vlim=self._max_velocity)
        if now - self._last_log_at >= self._log_period:
            z_ref = float(self._pose(q, self._log_data).translation[2])
            z_actual = float(self._pose(sample.q, self._log_data).translation[2])
            with self._motion_lock:
                self._motion_log.append((now, elapsed, dt, age, z_ref, z_actual,
                    *q, *velocity, *acceleration, *sample.q, *self._tau_filtered,
                    self._arm_group.send_error_count, self._fault or ""))
            self._last_log_at = now

    @property
    def motion_status(self):
        with self._motion_lock:
            return dict(self._last_plan, moving=self._moving, fault=self._fault,
                        feedback_source=self._feedback.latest().source,
                        feedback_errors=self._feedback.error_count,
                        feedback_last_error=self._feedback.last_error,
                        control_loop=dict(self.rebotarm.control_loop_stats))

    def export_motion_log(self, path):
        path = Path(path).resolve()
        columns = ["monotonic_time", "trajectory_time", "control_dt", "feedback_age", "z_ref", "z_actual"]
        for prefix in ("q_ref", "qd_ref", "qdd_ref", "q_actual", "tau_ff"):
            columns.extend(f"{prefix}_{name}" for name in self._arm_group.joint_names)
        columns.extend(("send_errors", "fault"))
        with self._motion_lock:
            rows = list(self._motion_log)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(columns)
            writer.writerows(rows)
        return path

    def safe_home(self, max_vel=0.5, send_freq=50.0, settle_thresh=0.01, timeout=15.0):
        """Home through the same continuous execution path; timeout holds pose.

        ``send_freq`` remains accepted for API compatibility; the servo loop
        evaluates the reference at its configured rate.
        """
        if not self._running:
            return
        with self._plan_lock:
            self.stop_motion()
            if self._fault:
                return
            q_start = self._fresh_sample().q
            reference = self._build_reference([q_start, np.zeros(self._n)], 1.0,
                                              max_velocity=np.minimum(self._max_velocity, max_vel))
            self._install(reference, q_start, {"kind": "home"})
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._motion_lock:
                moving, fault = self._moving, self._fault
            if fault:
                return
            if not moving and np.max(np.abs(self._fresh_sample().q)) <= settle_thresh:
                return
            if not self.rebotarm.control_loop_active:
                raise RuntimeError("Control loop stopped during homing")
            time.sleep(min(0.02, self._dt))
        self.stop_motion()
        raise TimeoutError("Homing timed out; holding measured pose")
