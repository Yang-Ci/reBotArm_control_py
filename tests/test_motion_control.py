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
        self.enable_count = self.disable_count = 0

    def enable(self):
        self.enable_count += 1

    def disable(self):
        self.disable_count += 1

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
    robot._ctrl_thread = None
    order = []
    monkeypatch.setattr(robot, "connect", lambda: None)
    monkeypatch.setattr(robot, "require_isolated_feedback", lambda: None)
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
    robot._ctrl_thread = None
    events = []
    monkeypatch.setattr(robot, "connect", lambda: None)
    monkeypatch.setattr(robot, "require_isolated_feedback", lambda: None)
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
    robot._motor_map["joint2"].enable = fail_enable
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


def test_overrun_keeps_original_clock_grid_without_catchup_bursts(monkeypatch):
    from reBotArm_control_py.actuator import rebotarm as module
    robot = RebotArm()
    clock = {"now": 0.}
    monkeypatch.setattr(module.time, "perf_counter", lambda: clock["now"])
    robot._ctrl_rate, robot._running = 100., True
    robot._control_stop = SimpleNamespace(wait=lambda dt: clock.__setitem__("now", clock["now"] + dt))
    times = []
    def callback(ref, dt):
        times.append(clock["now"])
        clock["now"] += .022 if len(times) == 2 else .001
        if len(times) == 3:
            ref._running = False
    robot._ctrl_fn = callback
    robot._control_loop_impl()
    np.testing.assert_allclose(times, [0., .01, .04], atol=1e-12)
    stats = robot.control_loop_snapshot()
    assert stats["missed_deadlines"] == 2
    assert stats["timing_samples"] == 3
    assert stats["control_dt_s"]["p50"] == pytest.approx(.01, abs=1.01e-5)
    assert stats["control_dt_s"]["p99"] == pytest.approx(.03)
    assert stats["max_callback_s"] == pytest.approx(.022)


def test_gravity_updates_on_new_feedback_and_filter_still_runs_every_cycle(controller, monkeypatch):
    from reBotArm_control_py.controllers import rebotarm_endpose_controller as module
    ctrl, robot, _ = controller
    calls = []
    def gravity(model, q, data):
        calls.append(q.copy())
        return np.full(model.nv, 1. if len(calls) == 1 else 2.)
    monkeypatch.setattr(module, "compute_generalized_gravity", gravity)
    for _ in range(10):
        ctrl._loop_cb(robot, .002)
    assert len(calls) == 1
    old_torque = ctrl._tau_filtered.copy()
    ctrl._feedback.refresh()
    ctrl._loop_cb(robot, .002)
    first_blend = ctrl._tau_filtered.copy()
    ctrl._loop_cb(robot, .002)
    assert len(calls) == 2
    assert np.all(old_torque < first_blend)
    assert np.all(first_blend < ctrl._tau_filtered)
    ctrl.clear_fault()
    ctrl._loop_cb(robot, .002)
    assert len(calls) == 3


def test_logging_does_not_run_fk_in_servo_and_export_uses_original_snapshot(controller, monkeypatch, tmp_path):
    import csv
    ctrl, robot, _ = controller
    original_pose = ctrl._pose
    original_q = ctrl._feedback.latest().q
    expected_z = original_pose(original_q, ctrl._model.createData()).translation[2]
    monkeypatch.setattr(ctrl, "_pose", lambda *args: pytest.fail("FK entered the servo callback"))
    ctrl._loop_cb(robot, .004)
    assert ctrl._fault is None
    robot._motor_map["joint2"].pos += .01
    ctrl._feedback.refresh()
    monkeypatch.setattr(ctrl, "_pose", original_pose)
    with ctrl.export_motion_log(tmp_path / "deferred.csv").open(encoding="utf-8", newline="") as file:
        row = next(csv.DictReader(file))
    assert float(row["q_actual_joint2"]) == original_q[1]
    assert float(row["z_actual"]) == pytest.approx(expected_z)


def test_concurrent_feedback_refreshes_cannot_publish_batches_out_of_order():
    import threading
    import time
    entered, release, second_read, second_attempt = [threading.Event() for _ in range(4)]
    calls = []
    def read(timeout):
        number = len(calls) + 1
        calls.append(number)
        stamp = time.monotonic()
        if number == 1:
            entered.set()
            assert release.wait(1.)
        else:
            second_read.set()
        return np.array([float(number)]), stamp, "test"
    feedback = PositionFeedback(SimpleNamespace(read_position_sample=read, num_joints=1))
    errors = []
    def refresh(second=False):
        if second:
            second_attempt.set()
        try:
            feedback.refresh()
        except Exception as error:
            errors.append(error)
    first = threading.Thread(target=refresh)
    second = threading.Thread(target=refresh, args=(True,))
    first.start()
    try:
        assert entered.wait(1.)
        second.start()
        assert second_attempt.wait(1.)
        overlapping = second_read.wait(.03)
    finally:
        release.set()
        first.join(1.)
        if second.ident is not None:
            second.join(1.)
    assert not overlapping and not errors
    assert calls == [1, 2]
    assert feedback.latest().q[0] == 2.


@pytest.mark.native_pinocchio
def test_rs_full_servo_lift_with_native_gravity(controller, monkeypatch, tmp_path):
    from reBotArm_control_py import dynamics
    from reBotArm_control_py.controllers import rebotarm_endpose_controller as module
    monkeypatch.setattr(module, "compute_generalized_gravity", dynamics.compute_generalized_gravity)
    ctrl, robot, clock = controller
    assert ctrl.move_relative(dz=.1, duration=8.)
    reference = ctrl._reference
    for index in range(2061):
        clock["now"] = 10. + index / 250.
        ctrl._feedback.refresh()
        ctrl._loop_cb(robot, .004)
        assert ctrl._fault is None
        for name in robot.arm.joint_names:
            assert np.all(np.isfinite(robot._motor_map[name].commands[-1]))
    assert not ctrl._moving
    np.testing.assert_allclose(ctrl._q_target, reference.sample(8.).q, atol=1e-12)
    np.testing.assert_array_equal(ctrl._qd_target, np.zeros(6))
    import csv
    with ctrl.export_motion_log(tmp_path / "native_lift.csv").open(encoding="utf-8", newline="") as file:
        logged = list(csv.DictReader(file))
    assert min(np.diff([float(row["z_ref"]) for row in logged])) >= -1e-9
    assert all(row[-5] == "" for row in ctrl._motion_log)
    assert ctrl.motion_status["goal_reached"]


def test_group_enable_disable_does_not_touch_gripper(controller, monkeypatch):
    from reBotArm_control_py.actuator import rebotarm as module
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    _, robot, _ = controller
    robot.arm.enable(strict=True)
    robot.arm.disable()
    assert all(robot._motor_map[n].enable_count == 1 for n in robot.arm.joint_names)
    assert robot._motor_map["gripper"].enable_count == 0
    assert robot._motor_map["gripper"].disable_count == 0


def test_rs_posvel_uses_native_csp_and_parameter_failure_rejects_mode(controller, monkeypatch):
    from motorbridge import Mode
    from reBotArm_control_py.actuator import rebotarm as module
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    _, robot, _ = controller
    modes, targets = [], []
    for motor in robot._motor_map.values():
        motor.ensure_mode = lambda mode, timeout: modes.append(mode)
        motor.robstride_write_param_f32 = lambda *args: targets.append(args)
    assert robot.arm.mode_pos_vel(np.full(6, .5))
    assert modes == [Mode.ROBSTRIDE_POS_VEL_CSP] * 6
    targets.clear()
    robot.arm.send_pos_vel(np.arange(6.) / 10, strict=True)
    assert len(targets) == 6 and all(index == 0x7016 for index, _ in targets)
    robot.arm._mode = "mit"
    def fail(*args):
        raise CallError("PID write failed")
    robot._motor_map["joint2"].robstride_write_param_f32 = fail
    assert not robot.arm.mode_pos_vel()
    assert robot.arm.mode == "mit"


def test_parallel_position_batch_uses_one_controller_and_waits_for_all_readers(controller):
    import threading
    from concurrent.futures import ThreadPoolExecutor
    _, robot, _ = controller
    entered = [threading.Event() for _ in range(6)]
    release = threading.Event()
    for index, name in enumerate(robot.arm.joint_names):
        motor = robot._motor_map[name]
        def read(param, host, timeout, index=index, motor=motor):
            entered[index].set()
            assert release.wait(2)
            if index == 0:
                raise CallError("one joint timed out")
            return motor.pos
        motor.robstride_get_param_f32_host_id = read
    robot._setup_position_readers()
    assert robot.arm._position_readers["joint1"] is robot._motor_map["joint1"]
    try:
        with ThreadPoolExecutor(max_workers=1) as caller:
            pending = caller.submit(robot.arm.read_position_sample)
            try:
                assert all(event.wait(1) for event in entered)
                # Separate workers do not hold Python's control lock.
                robot.arm.send_mit(np.zeros(6), strict=True)
                assert not pending.done()
            finally:
                release.set()
            with pytest.raises(CallError):
                pending.result(timeout=2)
    finally:
        robot._close_position_readers()


def test_idle_following_error_also_faults(controller):
    ctrl, robot, _ = controller
    robot._motor_map["joint2"].pos += .2
    ctrl._feedback.refresh()
    ctrl._loop_cb(robot, .004)
    assert "tracking error" in ctrl._fault
    assert np.all(ctrl._qd_target == 0)


@pytest.mark.parametrize("dt", [np.nan, 0., -.01])
def test_invalid_dt_does_not_poison_feedforward(controller, dt):
    ctrl, robot, _ = controller
    ctrl._loop_cb(robot, .004)
    ctrl._loop_cb(robot, dt)
    assert ctrl._fault
    assert np.all(np.isfinite(ctrl._tau_filtered))
    assert all(np.all(np.isfinite(m.commands[-1])) for m in robot._motor_map.values())


def test_feedback_span_is_checked_before_motion_and_during_hold(controller):
    ctrl, robot, _ = controller
    robot.arm.last_position_read_span = .03
    # read_position_sample recalculates its own span, so inject an acquired sample.
    from reBotArm_control_py.controllers.position_feedback import PositionSample
    sample = ctrl._feedback.latest()
    ctrl._feedback._sample = PositionSample(sample.q, sample.sampled_at, sample.source,
                                           sample.sequence, .03, .04)
    with pytest.raises(RuntimeError, match="sampling span"):
        ctrl.get_joint_positions()
    ctrl._loop_cb(robot, .004)
    assert "sampling span" in ctrl._fault


def test_end_of_reference_requires_new_feedback_over_settle_duration(controller):
    ctrl, robot, clock = controller
    reference = ctrl._build_reference([ctrl._q_target, ctrl._q_target + [.02, 0, 0, 0, 0, 0]], 1.)
    ctrl._install(reference, ctrl._q_target, {"kind": "test"})
    ctrl._loop_cb(robot, .004)
    clock["now"] += reference.duration
    ctrl._feedback.refresh()
    ctrl._loop_cb(robot, .004)
    assert ctrl._settling and ctrl._moving and not ctrl._goal_reached
    # This first fresh endpoint sample starts the dwell.
    clock["now"] += .004
    ctrl._feedback.refresh()
    ctrl._loop_cb(robot, .004)
    # Holding the same post-endpoint sample for .21 s does not prove settling.
    clock["now"] += .21
    ctrl._loop_cb(robot, .004)
    assert ctrl._settling and not ctrl._goal_reached
    ctrl._feedback.refresh()
    ctrl._loop_cb(robot, .004)
    assert not ctrl._moving and ctrl._goal_reached


def test_settle_timeout_keeps_measured_hold(controller, monkeypatch):
    ctrl, robot, clock = controller
    for motor in robot._motor_map.values():
        monkeypatch.setattr(motor, "send_mit", lambda *args, motor=motor: motor.commands.append(args))
    reference = ctrl._build_reference([ctrl._q_target, ctrl._q_target + [.02, 0, 0, 0, 0, 0]], 1.)
    ctrl._install(reference, ctrl._q_target, {"kind": "test"})
    ctrl._loop_cb(robot, .004)
    for elapsed in (reference.duration, reference.duration + ctrl._settle_timeout + .01):
        clock["now"] = 10. + elapsed
        ctrl._feedback.refresh()
        ctrl._loop_cb(robot, .004)
    assert "settling timed out" in ctrl._fault
    np.testing.assert_allclose(ctrl._q_target, ctrl._feedback.latest().q)
    assert not ctrl._goal_reached and np.all(ctrl._qd_target == 0)


def test_second_start_does_not_cleanup_running_controller(controller, monkeypatch):
    ctrl, robot, _ = controller
    monkeypatch.setattr(robot, "disable_all", lambda: pytest.fail("must not disable running controller"))
    with pytest.raises(RuntimeError, match="already running"):
        ctrl.start()
    assert ctrl._running


def test_partial_enable_failure_cleans_up_and_preserves_original_error(controller, monkeypatch):
    ctrl, robot, _ = controller
    ctrl._running = False
    robot._ctrl_thread = None
    events = []
    def partial_start():
        robot._connected = True
        robot._motor_map["joint1"].enable()
        raise CallError("joint2 enable failed")
    monkeypatch.setattr(ctrl, "_start_impl", partial_start)
    monkeypatch.setattr(robot, "stop_control_loop", lambda: events.append("stop_loop"))
    monkeypatch.setattr(ctrl._feedback, "stop", lambda: events.append("stop_feedback"))
    monkeypatch.setattr(robot, "disconnect", lambda: events.append("disconnect"))
    with pytest.raises(CallError, match="joint2 enable failed"):
        ctrl.start()
    assert events == ["stop_loop", "stop_feedback", "disconnect"]
    assert all(m.disable_count == 1 for m in robot._motor_map.values())


def test_quit_does_not_initiate_homing(controller, monkeypatch):
    ctrl, robot, _ = controller
    events = []
    monkeypatch.setattr(ctrl, "safe_home", lambda: pytest.fail("unexpected motion on exit"))
    monkeypatch.setattr(robot, "stop_control_loop", lambda: events.append("stop_loop"))
    monkeypatch.setattr(ctrl._feedback, "stop", lambda: events.append("stop_feedback"))
    monkeypatch.setattr(robot, "disconnect", lambda: events.append("disconnect"))
    ctrl.end()
    assert events == ["stop_loop", "stop_feedback", "disconnect"]
    assert not ctrl._running


def test_read_only_disconnect_never_disables(controller, monkeypatch):
    _, robot, _ = controller
    events = []
    robot._connected = True
    robot._ctrl_thread = None
    monkeypatch.setattr(robot, "disable_all", lambda: pytest.fail("read-only probe disabled arm"))
    robot._ctrl_map["robstride"].close_bus = lambda: events.append("close_bus")
    robot._ctrl_map["robstride"].close = lambda: events.append("close_controller")
    for motor in robot._motor_map.values():
        motor.close = lambda: events.append("close_motor")
    robot.disconnect(disable_motors=False)
    assert events[-2:] == ["close_bus", "close_controller"]
    assert events.count("close_motor") == 7


def test_gripper_close_restores_release_gains(controller):
    ctrl, robot, _ = controller
    before = robot.gripper._mit_kp.copy()
    ctrl.open_gripper()
    assert np.all(robot.gripper._mit_kp == 0)
    ctrl.close_gripper()
    np.testing.assert_array_equal(robot.gripper._mit_kp, before)


@pytest.mark.native_pinocchio
def test_preflight_reads_without_mode_enable_zero_or_disable(controller, monkeypatch, tmp_path):
    from tools.rs_hardware_check import probe
    from motorbridge import abi
    from reBotArm_control_py.actuator import rebotarm as actuator
    ctrl, robot, clock = controller
    robot._ctrl_thread = None
    monkeypatch.setattr(pin, "computeGeneralizedGravity", lambda model, data, q: np.zeros(model.nv), raising=False)
    monkeypatch.setattr(robot, "connect", lambda: setattr(robot, "_connected", True))
    monkeypatch.setattr(actuator.time, "sleep", lambda duration: clock.__setitem__("now", clock["now"] + duration))
    monkeypatch.setattr(abi, "get_abi", lambda: SimpleNamespace(lib=SimpleNamespace(
        _name="fake-test-abi", rebotarm_rs_feedback_lock_fix_v1=lambda: 1)))
    robot._ctrl_map["robstride"].close_bus = lambda: None
    robot._ctrl_map["robstride"].close = lambda: None
    for motor in robot._motor_map.values():
        motor.close = lambda: None
    monkeypatch.setattr(robot, "disable_all", lambda: pytest.fail("read-only disabled motors"))
    report = probe(robot, .12, tmp_path / "probe.csv")
    assert report["ready_for_control"] and report["samples"] >= 2
    assert not robot._connected
    assert all(m.enable_count == 0 and not m.commands for m in robot._motor_map.values())


def test_unpatched_rs_backend_is_rejected_before_mode_switch(controller, monkeypatch):
    from motorbridge import abi
    ctrl, robot, _ = controller
    ctrl._running = False
    robot._ctrl_thread = None
    monkeypatch.setattr(robot, "connect", lambda: None)
    monkeypatch.setattr(abi, "get_abi", lambda: SimpleNamespace(lib=SimpleNamespace()))
    monkeypatch.setattr(robot.arm, "mode_mit", lambda: pytest.fail("mode switch before ABI check"))
    with pytest.raises(RuntimeError, match="ABI patch missing"):
        ctrl.start()


def test_baseline_after_completed_move_is_labeled_hold(controller):
    ctrl, robot, _ = controller
    ctrl._goal_reached = True
    ctrl.begin_log()
    ctrl._loop_cb(robot, .004)
    assert ctrl._motion_log[-1][-1] == "hold"
    assert ctrl.motion_status["kind"] == "hold"


def test_connect_failure_closes_partial_handles_and_preserves_bus_error(controller, monkeypatch):
    _, robot, _ = controller
    events = []
    for motor in robot._motor_map.values():
        motor.close = lambda: events.append("motor_close")
    def bad_cleanup():
        raise CallError("close bus also failed")
    robot._ctrl_map["robstride"].close_bus = bad_cleanup
    robot._ctrl_map["robstride"].close = lambda: events.append("controller_close")
    def fail_setup():
        raise CallError("PCAN initialize failed")
    monkeypatch.setattr(robot, "_setup_motors", fail_setup)
    with pytest.raises(CallError, match="PCAN initialize failed"):
        robot.connect()
    assert events.count("motor_close") == 7 and events[-1] == "controller_close"
    assert not robot._ctrl_map and not robot._motor_map
