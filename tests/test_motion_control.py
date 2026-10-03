"""Offline regressions: references, real URDF kinematics, and injected I/O faults."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest
from motorbridge import CallError

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.controllers.position_feedback import PositionFeedback
from reBotArm_control_py.trajectory import (
    JointPathReference, TrajProfile, TrajPlanParams, plan_cartesian_geodesic_trajectory,
    track_trajectory,
)
from reBotArm_control_py.trajectory.clik_tracker import _joint_limit_grad, _null_step
from reBotArm_control_py.trajectory.sampler import profile_state

ROOT = Path(__file__).resolve().parents[1]


class Motor:
    def __init__(self, pos):
        self.pos = pos
        self.commands = []
        self.requests = 0
        self.param_reads = 0
        self.fail_send = False
        self.fail_read = False
        self.state_available = True

    def get_state(self):
        return SimpleNamespace(pos=self.pos, vel=0., torq=0.) if self.state_available else None

    def request_feedback(self):
        self.requests += 1

    def robstride_get_param_f32(self, index, timeout_ms=1000):
        assert index == 0x7019
        self.param_reads += 1
        if self.fail_read:
            raise CallError("injected read timeout")
        return self.pos

    def send_mit(self, *args):
        self.commands.append(args)
        if self.fail_send:
            raise CallError("injected send failure")
        self.pos = args[0]

    def robstride_get_param_f32_host_id(self, index, host_id, timeout_ms):
        assert host_id == 0xFD
        assert timeout_ms == 20
        return self.robstride_get_param_f32(index, timeout_ms)

    def send_pos_vel(self, *args):
        self.commands.append(args)
        if self.fail_send:
            raise CallError("injected send failure")
        self.pos = args[0]


@pytest.fixture
def controller(monkeypatch):
    from reBotArm_control_py.controllers import rebotarm_endpose_controller as module
    clock = {"now": 10.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(module, "compute_generalized_gravity", lambda model, q, data: np.zeros(model.nv))
    robot = RebotArm()
    q = [0., .7, 1.1, 0., 0., 0., 0.]
    robot._motor_map.update({name: Motor(pos) for name, pos in zip(robot.joint_names, q)})
    robot._ctrl_map["robstride"] = SimpleNamespace(poll_feedback_once=lambda: None)
    robot._ctrl_thread = SimpleNamespace(is_alive=lambda: True)
    ctrl = RebotArmEndPose(robot)
    ctrl._running = True
    ctrl._feedback.refresh()
    ctrl._q_target = np.array(q[:6])
    return ctrl, robot, clock


def test_reference_position_velocity_acceleration_agree():
    q = np.column_stack([np.sin(np.linspace(0, 0.5, 101)), np.linspace(0, .2, 101)])
    ref = JointPathReference(q, 8.0)
    assert ref.duration == 8.0
    h = 1e-4
    for t in (.1, .5, 1., 2.35, 4., 7.8):
        before, at, after = ref.sample(t - h), ref.sample(t), ref.sample(t + h)
        np.testing.assert_allclose((after.q - before.q) / (2 * h), at.qd, atol=2e-8)
        np.testing.assert_allclose((after.qd - before.qd) / (2 * h), at.qdd, atol=2e-7)
    for t in (0., 8., 9.):
        assert np.all(ref.sample(t).qd == 0)
        assert np.all(ref.sample(t).qdd == 0)
    assert np.linalg.norm(ref.sample(.1).q - ref.sample(.09).q) > 0


def test_spline_has_continuous_two_derivatives_at_knots():
    ref = JointPathReference(np.sin(np.linspace(0, 1, 31))[:, None], 8.)
    for s in np.linspace(0, 1, 31)[1:-1]:
        before, after = ref.sample_progress(s - 1e-9), ref.sample_progress(s + 1e-9)
        for a, b in zip(before, after):
            np.testing.assert_allclose(a, b, atol=1e-6)


def test_retime_and_interpolated_limits():
    ref = JointPathReference([[0.], [1.]], .1, max_velocity=.1, max_acceleration=.1)
    assert ref.duration >= 18.75
    assert ref.velocity_bound[0] <= .1
    assert ref.acceleration_bound[0] <= .1
    states = [ref.sample(t) for t in np.linspace(0, ref.duration, 501)]
    assert max(abs(p.qd[0]) for p in states) <= .1 + 1e-10
    assert max(abs(p.qdd[0]) for p in states) <= .1 + 1e-10
    with pytest.raises(ValueError, match="too short"):
        JointPathReference([[0.], [1.]], .1, max_velocity=.1, allow_retime=False)
    with pytest.raises(ValueError, match="exceeds"):
        JointPathReference([[0.], [.5]], 8., lower=[0.], upper=[.4])
    with pytest.raises(ValueError):
        JointPathReference([[0.], [np.nan]], 8.)


def test_trapezoid_is_monotone_and_continuous():
    progress = [profile_state(t, TrajProfile.TRAPEZOID)[0] for t in np.linspace(0, 1, 1001)]
    assert progress[0] == 0 and progress[-1] == 1
    assert np.min(np.diff(progress)) >= 0
    for t in (.25, .75):
        assert abs(profile_state(t - 1e-9, TrajProfile.TRAPEZOID)[0]
                   - profile_state(t + 1e-9, TrajProfile.TRAPEZOID)[0]) < 1e-8


def test_limit_centering_is_bounded_and_does_not_leak_task_motion(controller):
    ctrl, _, _ = controller
    q = np.zeros(ctrl._model.nq)
    q[1:3] = 1e-10
    assert np.max(np.abs(_joint_limit_grad(ctrl._model, q))) <= 1
    np.testing.assert_array_equal(_null_step(np.diag([1., 1., 1., 1., .01, .001]),
                                            np.ones(6), .1), np.zeros(6))


def test_slow_ik_does_not_drop_small_spatial_changes(controller):
    ctrl, _, _ = controller
    q = np.zeros(ctrl._model.nq)
    q[:6] = ctrl._q_target
    start = ctrl._pose(q[:6], ctrl._ik_data)
    target = start.copy()
    target.translation[2] += .0002
    cart = plan_cartesian_geodesic_trajectory(start, target, 1.,
        TrajPlanParams(dt=.01, profile=TrajProfile.LINEAR))
    pts = track_trajectory(ctrl._model, ctrl._end_frame_id, cart.trajectory, q,
                          controlled_joints=6)
    assert all(p.ik_success for p in pts)
    assert all(np.linalg.norm(pts[i].q - pts[i - 1].q) > 0 for i in range(1, len(pts)))
    assert all(np.all(p.q[6:] == 0) for p in pts)


def test_rs_lift_10cm_8seconds_has_monotone_reference(controller, tmp_path):
    ctrl, robot, clock = controller
    assert ctrl.move_relative(dz=.1, duration=8.)
    ref = ctrl._reference
    assert ctrl.motion_status["actual_duration"] == 8.
    z, rotations = [], []
    for t in np.linspace(0, 8, 801):
        pose = ctrl._pose(ref.sample(t).q, ctrl._ik_data)
        z.append(pose.translation[2])
        rotations.append(pose.rotation.copy())
    assert min(np.diff(z)) >= -1e-9
    assert abs(z[-1] - z[0] - .1) < 2e-5
    np.testing.assert_allclose(rotations[-1], rotations[0], atol=2e-5)
    # Irregular cycle intervals must evaluate actual elapsed time, never a point index.
    for elapsed in [0., .007, .015, .033, .051, .080]:
        clock["now"] = 10. + elapsed
        ctrl._feedback.refresh()
        ctrl._loop_cb(robot, .01)
        np.testing.assert_allclose(ctrl._q_target, ref.sample(elapsed).q, atol=1e-12)
        np.testing.assert_allclose(ctrl._qd_target, ref.sample(elapsed).qd, atol=1e-12)
    assert np.linalg.norm(ctrl._qd_target) > 0
    assert ctrl._fault is None
    log = ctrl.export_motion_log(tmp_path / "lift.csv")
    text = log.read_text(encoding="utf-8")
    assert "z_actual" in text and "qd_ref_joint2" in text and "feedback_age" in text


def test_unreachable_goal_is_rejected_before_any_send(controller):
    ctrl, robot, _ = controller
    assert not ctrl.move_to_traj(10., 10., 10., duration=8.)
    assert ctrl._reference is None
    assert all(not m.commands for m in robot._motor_map.values())


def test_path_is_independent_of_execution_duration(controller):
    ctrl, _, _ = controller
    assert ctrl.move_relative(dz=.1, duration=8.)
    slow = ctrl._reference
    ctrl.stop_motion()
    assert ctrl.move_relative(dz=.1, duration=2.)
    fast = ctrl._reference
    for progress in (.01, .1, .4, .9, 1.):
        np.testing.assert_allclose(slow.sample_progress(progress)[0],
                                   fast.sample_progress(progress)[0], atol=1e-12)


def test_servo_callback_never_performs_live_position_reads(controller):
    ctrl, robot, _ = controller
    reads = [m.param_reads for m in robot._motor_map.values()]
    ctrl._loop_cb(robot, .004)
    assert ctrl._fault is None
    assert [m.param_reads for m in robot._motor_map.values()] == reads


def test_start_primes_measured_targets_before_enable(controller, monkeypatch):
    ctrl, robot, _ = controller
    ctrl._running = False
    order = []
    monkeypatch.setattr(robot, "connect", lambda: None)
    monkeypatch.setattr(robot.arm, "mode_mit", lambda: True)
    monkeypatch.setattr(robot.gripper, "mode_mit", lambda: True)
    monkeypatch.setattr(ctrl._feedback, "start", lambda: ctrl._feedback.refresh())
    monkeypatch.setattr(robot, "enable_all", lambda **kwargs: order.append("enable"))
    monkeypatch.setattr(robot, "start_control_loop", lambda cb: order.append("loop"))
    for motor in robot._motor_map.values():
        original = motor.send_mit
        def send(*args, original=original):
            order.append("send")
            original(*args)
        monkeypatch.setattr(motor, "send_mit", send)
    ctrl.start()
    assert order[:7] == ["send"] * 7
    assert order[-2:] == ["enable", "loop"]
    np.testing.assert_allclose(ctrl._q_target, [0., .7, 1.1, 0., 0., 0.])
    assert ctrl._ik_data is not ctrl._gravity_data


def test_failed_start_stops_feedback_before_enabling(controller, monkeypatch):
    ctrl, robot, _ = controller
    ctrl._running = False
    events = []
    monkeypatch.setattr(robot, "connect", lambda: None)
    monkeypatch.setattr(robot.arm, "mode_mit", lambda: True)
    monkeypatch.setattr(robot.gripper, "mode_mit", lambda: True)
    monkeypatch.setattr(ctrl._feedback, "start", lambda: events.append("feedback_start"))
    monkeypatch.setattr(ctrl._feedback, "stop", lambda: events.append("feedback_stop"))
    monkeypatch.setattr(robot, "enable_all", lambda **kwargs: events.append("enable"))
    robot._motor_map["joint2"].fail_send = True
    with pytest.raises(CallError):
        ctrl.start()
    assert events == ["feedback_start", "feedback_stop"]
    assert not ctrl._running


@pytest.mark.parametrize("fault", ["stale", "send", "late_cycle"])
def test_fault_clears_reference_velocity_and_holds_pose(controller, fault):
    ctrl, robot, clock = controller
    ctrl._install(ctrl._build_reference([ctrl._q_target, ctrl._q_target + [.1, 0, 0, 0, 0, 0]], 8.),
                  ctrl._q_target, {"kind": "test"})
    ctrl._loop_cb(robot, .004)
    clock["now"] += .03
    ctrl._feedback.refresh()
    ctrl._loop_cb(robot, .004)
    assert np.linalg.norm(ctrl._qd_target) > 0
    if fault == "stale":
        clock["now"] += .3
    elif fault == "send":
        robot._motor_map["joint2"].fail_send = True
    ctrl._loop_cb(robot, .1 if fault == "late_cycle" else .004)
    assert ctrl._fault
    assert ctrl._reference is None and not ctrl._moving
    assert np.all(ctrl._qd_target == 0)
    ctrl._loop_cb(robot, .004)
    assert all(m.commands[-1][1] == 0 for m in list(robot._motor_map.values())[:6])
    assert not ctrl.move_relative(dz=.1, duration=8.)


def test_nonblocking_get_positions_never_falls_back_to_parameter_read(controller):
    _, robot, _ = controller
    for m in robot._motor_map.values():
        m.state_available = False
        m.param_reads = 0
    with pytest.raises(RuntimeError, match="No position"):
        robot.arm.get_positions(request_feedback=False, strict=True)
    assert all(m.param_reads == 0 and m.requests == 0 for m in robot._motor_map.values())


def test_feedback_failure_does_not_refresh_timestamp(controller):
    ctrl, robot, clock = controller
    old = ctrl._feedback.latest()
    clock["now"] += .3
    robot._motor_map["joint2"].fail_read = True
    with pytest.raises(CallError):
        ctrl._feedback.refresh()
    assert ctrl._feedback.latest().sampled_at == old.sampled_at
    changed = ctrl._feedback.latest()
    changed.q[:] = 99
    assert np.max(ctrl._feedback.latest().q) < 99


def test_public_positions_are_live_copies_and_reject_expired_data(controller):
    ctrl, _, clock = controller
    returned = ctrl.get_joint_positions()
    returned[:] = 99
    assert np.max(ctrl.get_joint_positions()) < 99
    clock["now"] += .3
    with pytest.raises(RuntimeError, match="expired"):
        ctrl.get_joint_positions()


def test_friction_is_bounded_uses_reference_direction_and_has_no_idle_push(controller):
    ctrl, _, _ = controller
    ctrl._friction_enabled = True
    ctrl._coulomb[:] = 2
    ctrl._static_friction[:] = 3
    ctrl._viscous[:] = 1
    np.testing.assert_array_equal(ctrl._friction_torque(np.zeros(6)), np.zeros(6))
    positive = ctrl._friction_torque(np.full(6, .01))
    assert np.all(positive > 0) and np.all(positive <= ctrl._friction_limit)
    np.testing.assert_allclose(ctrl._friction_torque(np.full(6, -.01)), -positive)


def test_actuator_reports_partial_send_failure_and_still_attempts_other_joints(controller):
    ctrl, robot, _ = controller
    robot._motor_map["joint2"].fail_send = True
    with pytest.raises(CallError):
        robot.arm.send_mit(ctrl._q_target, strict=True)
    assert robot.arm.send_error_count == 1
    assert "joint2" in robot.arm.last_send_error
    assert len(robot._motor_map["joint6"].commands) == 1


def test_enable_failure_is_not_silently_accepted(controller):
    _, robot, _ = controller
    def fail_enable():
        raise CallError("injected enable failure")
    robot._ctrl_map["robstride"].enable_all = fail_enable
    with pytest.raises(CallError):
        robot.enable_all(strict=True)


def test_control_loop_passes_measured_dt_and_counts_overrun(monkeypatch):
    from reBotArm_control_py.actuator import rebotarm as module
    robot = RebotArm()
    clock = {"now": 0.}
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock["now"])
    robot._ctrl_rate = 100.
    robot._running = True
    robot._control_stop = SimpleNamespace(wait=lambda delay: clock.__setitem__("now", clock["now"] + delay))
    samples = []
    def callback(r, dt):
        samples.append(dt)
        clock["now"] += .02 if len(samples) == 2 else .001
        if len(samples) == 3:
            r._running = False
    robot._ctrl_fn = callback
    robot._control_loop_impl()
    np.testing.assert_allclose(samples, [.01, .01, .03])
    assert robot.control_loop_stats["overruns"] == 1


@pytest.mark.native_pinocchio
def test_rs_full_servo_lift_with_native_gravity(controller, monkeypatch):
    from reBotArm_control_py import dynamics
    from reBotArm_control_py.controllers import rebotarm_endpose_controller as module
    monkeypatch.setattr(module, "compute_generalized_gravity", dynamics.compute_generalized_gravity)
    ctrl, robot, clock = controller
    assert ctrl.move_relative(dz=.1, duration=8.)
    reference = ctrl._reference
    for index in range(2001):
        clock["now"] = 10. + index / 250.
        ctrl._feedback.refresh()
        ctrl._loop_cb(robot, .004)
        assert ctrl._fault is None
        for name in robot.arm.joint_names:
            assert np.all(np.isfinite(robot._motor_map[name].commands[-1]))
    assert not ctrl._moving
    np.testing.assert_allclose(ctrl._q_target, reference.sample(8.).q, atol=1e-12)
    np.testing.assert_array_equal(ctrl._qd_target, np.zeros(6))
    assert min(np.diff([row[4] for row in ctrl._motion_log])) >= -1e-9
    assert all(row[-1] == "" for row in ctrl._motion_log)
