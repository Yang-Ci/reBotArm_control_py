"""DM serial-to-CAN regressions: actual URDF, packet freshness and transport."""
import ctypes as ct
from types import SimpleNamespace

import numpy as np
import pinocchio as pin
import pytest
from motorbridge import CallError
from motorbridge.abi import CState

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.actuator import dm_feedback
from reBotArm_control_py.controllers import RebotArmEndPose


class DmMotor:
    def __init__(self, pos, clock, model="4340P"):
        self.pos, self.clock = pos, clock
        self.sequence = 0
        self.received = -np.inf
        self.requests = 0
        self.commands = []
        self.respond = True
        self.status = 1
        self.fail_send = False
        self.enable_count = self.disable_count = 0
        self.register_reads = []
        self.register_writes = []
        self.limits = (12.5, 10., 28.) if model == "4340P" else (12.5, 30., 10.)
        self.firmware_limits = self.limits

    def request_feedback(self):
        self.requests += 1
        if self.respond:
            self.sequence += 1
            self.received = self.clock["now"]

    def snapshot(self):
        if self.status not in (0, 1):
            raise dm_feedback.DmFeedbackFault(f"DM motor fault status 0x{self.status:X}")
        return dm_feedback.DmSnapshot(self.pos, 0., 0., self.status,
                                      self.sequence, self.received, self.limits)

    def send_mit(self, *args):
        if self.fail_send:
            raise CallError("serial write failed")
        self.commands.append(args)
        self.pos = args[0]

    def send_pos_vel(self, *args):
        self.send_mit(*args)

    def get_register_f32(self, index, timeout_ms=1000):
        self.register_reads.append(index)
        return self.firmware_limits[index - 21]

    def write_register_f32(self, index, value):
        self.register_writes.append((index, value))

    def ensure_mode(self, *args):
        pass

    def enable(self):
        self.enable_count += 1

    def disable(self):
        self.disable_count += 1

    def close(self):
        pass


@pytest.fixture
def dm_controller(monkeypatch):
    from reBotArm_control_py.controllers import rebotarm_endpose_controller as module
    clock = {"now": 10.}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])
    monkeypatch.setattr(module.time, "sleep", lambda seconds: clock.__setitem__("now", clock["now"] + seconds))
    monkeypatch.setattr(dm_feedback, "read_dm_snapshot", lambda motor: motor.snapshot())
    robot = RebotArm("rebotarm_dm.yaml", channel="COM7")
    q = [0., -.7, -1.1, 0., 0., 0., .3]
    robot._motor_map.update({j.name: DmMotor(pos, clock, j.model) for j, pos in zip(robot._all_joints, q)})
    robot._ctrl_map["damiao"] = SimpleNamespace(close_bus=lambda: None, close=lambda: None)
    robot._ctrl_thread = SimpleNamespace(is_alive=lambda: True)
    ctrl = RebotArmEndPose(robot)
    ctrl._running = True
    ctrl._feedback.refresh()
    ctrl._q_target = np.array(q[:6])
    return ctrl, robot, clock


@pytest.mark.native_pinocchio
@pytest.mark.parametrize("mode", ["mit", "posvel"])
def test_dm_full_lift_10cm_8s_both_modes(dm_controller, mode, tmp_path):
    _, robot, clock = dm_controller
    ctrl = RebotArmEndPose(robot, arm_control_mode=mode)
    ctrl._running = True
    ctrl._feedback.refresh()
    ctrl._q_target = ctrl._feedback.latest().q.copy()
    assert ctrl._model.nq == 8
    assert ctrl.move_relative(dz=.1, duration=8.)
    ref = ctrl._reference
    assert ref.duration == 8.
    initial = ctrl._pose(ref.sample(0).q, ctrl._ik_data)
    z = []
    for step in range(2101):
        clock["now"] = 10. + step * .004
        if step % 5 == 0:
            ctrl._feedback.refresh()
        ctrl._loop_cb(robot, .004)
        assert ctrl._fault is None
        pose = ctrl._pose(ref.sample(min(step * .004, 8.)).q, ctrl._ik_data)
        z.append(pose.translation[2])
    assert np.min(np.diff(z)) >= -1e-9
    assert z[-1] - z[0] == pytest.approx(.1, abs=2e-5)
    final = ctrl._pose(ctrl._q_target, ctrl._ik_data)
    np.testing.assert_allclose(final.rotation, initial.rotation, atol=2e-5)
    assert ctrl.motion_status["goal_reached"]
    assert ctrl.motion_status["feedback_source"] == "dm_sensor_rx_timed"
    assert ctrl.motion_status["arm_control_mode"] == mode
    commands = robot._motor_map["joint2"].commands
    if mode == "mit":
        assert any(abs(c[1]) > 1e-5 for c in commands)
        assert any(abs(c[4]) > .01 for c in commands)
        assert max(abs(c[4]) for c in commands) <= 18.
    else:
        assert all(len(c) == 2 and c[1] == .5 for c in commands)
        assert not ctrl.motion_status["gravity_feedforward"]
        assert all(np.all(np.asarray(row[28:34]) == 0) for row in ctrl._motion_log)
    assert all(not m.register_reads for m in robot._motor_map.values())
    assert all(m.enable_count == 0 for m in robot._motor_map.values())
    from tools.analyze_motion_log import analyze
    report = analyze(ctrl.export_motion_log(tmp_path / (mode + ".csv")), expected_dz=.1)
    assert report["passed"]
    assert report["thresholds"]["feedback_age_s"] == .1


def test_dm_stationary_sensor_packets_are_fresh(dm_controller):
    ctrl, _, clock = dm_controller
    first = ctrl._feedback.latest()
    clock["now"] += .02
    second = ctrl._feedback.refresh()
    np.testing.assert_array_equal(first.q, second.q)
    assert second.sampled_at > first.sampled_at
    assert second.sequence == first.sequence + 1


@pytest.mark.parametrize("frozen", ["joint3", "all"])
def test_dm_cached_feedback_does_not_refresh_timestamp(dm_controller, frozen):
    ctrl, robot, clock = dm_controller
    first = ctrl._feedback.latest()
    for name, motor in robot._motor_map.items():
        if frozen == "all" or name == frozen:
            motor.respond = False
    with pytest.raises(TimeoutError, match="No new DM feedback"):
        ctrl._feedback.refresh()
    assert ctrl._feedback.latest().sampled_at == first.sampled_at
    assert ctrl._feedback.latest().sequence == first.sequence
    clock["now"] += .11
    ctrl._loop_cb(robot, .004)
    assert "expired" in ctrl._fault
    assert np.all(ctrl._qd_target == 0)


def test_dm_received_motor_fault_stops_motion_immediately(dm_controller):
    ctrl, robot, _ = dm_controller
    robot._motor_map["joint2"].status = 0xA
    with pytest.raises(dm_feedback.DmFeedbackFault, match="0xA"):
        ctrl._feedback.refresh()
    ctrl._loop_cb(robot, .004)
    assert "0xA" in ctrl._fault
    assert np.all(ctrl._qd_target == 0)
    with pytest.raises(RuntimeError, match="0xA"):
        ctrl.clear_fault()


def test_dm_firmware_range_mismatch_is_rejected_before_mode(dm_controller, monkeypatch):
    ctrl, robot, _ = dm_controller
    ctrl._running = False
    robot._ctrl_thread = None
    robot._motor_map["joint2"].firmware_limits = (12.5, 20., 28.)
    monkeypatch.setattr(robot, "connect", lambda: None)
    monkeypatch.setattr(robot, "require_isolated_feedback", lambda: None)
    monkeypatch.setattr(robot.arm, "mode_pos_vel", lambda **kw: pytest.fail("range check too late"))
    with pytest.raises(RuntimeError, match="differ from SDK"):
        ctrl.start()
    assert all(m.enable_count == 0 and not m.commands for m in robot._motor_map.values())


def test_dm_start_primes_measured_arm_and_gripper_before_enable(dm_controller, monkeypatch):
    ctrl, robot, _ = dm_controller
    ctrl._running = False
    robot._ctrl_thread = None
    monkeypatch.setattr(robot, "connect", lambda: None)
    monkeypatch.setattr(robot, "require_isolated_feedback", lambda: None)
    monkeypatch.setattr(robot.arm, "mode_pos_vel", lambda **kw: True)
    monkeypatch.setattr(robot.gripper, "mode_mit", lambda: True)
    monkeypatch.setattr(ctrl._feedback, "start", ctrl._feedback.refresh)
    def enable(**kw):
        for group in (robot.arm, robot.gripper):
            for name in group.joint_names:
                motor = robot._motor_map[name]
                assert motor.commands
                assert motor.commands[0][0] == motor.pos
                if name == "gripper":
                    assert motor.commands[0][0] == .3
    monkeypatch.setattr(robot, "enable_all", enable)
    monkeypatch.setattr(robot, "start_control_loop", lambda cb: None)
    ctrl.start()
    assert all(m.register_reads == [21, 22, 23] for m in robot._motor_map.values())


@pytest.mark.parametrize("channel", ["COM7", r"\\.\COM12", "/dev/ttyACM0", "/dev/cu.usbmodem101"])
def test_dm_auto_transport_and_serial_constructor(channel, monkeypatch):
    from reBotArm_control_py.actuator import rebotarm as module
    robot = RebotArm("rebotarm_dm.yaml", channel=channel)
    robot._transport = "auto"
    sentinel = object()
    seen = []
    monkeypatch.setattr(module.Controller, "from_dm_serial", lambda port, baud: seen.append((port, baud)) or sentinel)
    assert robot.transport == "dm-serial"
    assert robot._make_controller("damiao") is sentinel
    assert seen == [(channel, 921600)]


def test_dm_explicit_direct_can_transport_remains_supported(monkeypatch):
    from reBotArm_control_py.actuator import rebotarm as module
    robot = RebotArm("rebotarm_dm.yaml", channel="PCAN_USBBUS1")
    robot._transport = "can"
    monkeypatch.setattr(module, "Controller", lambda channel: channel)
    assert robot._make_controller("damiao") == "PCAN_USBBUS1"
    assert robot.serial_budget() is None


def test_dm_500hz_rejected_before_connect_or_enable(dm_controller, monkeypatch):
    ctrl, robot, _ = dm_controller
    assert robot.serial_budget()["valid"]
    assert robot.rate == 500.
    assert robot.serial_budget()["utilization"] == pytest.approx(1155000 / 921600)
    assert not robot.serial_budget()["uart_capacity_sufficient"]
    robot._serial_link = "uart"
    ctrl._running = False
    robot._ctrl_thread = None
    monkeypatch.setattr(robot, "connect", lambda: pytest.fail("budget check too late"))
    with pytest.raises(ValueError, match="serial command budget"):
        ctrl.start()
    assert not robot._connected


def test_dm_usb_cdc_preserves_500hz_and_uart_can_use_confirmed_higher_baud():
    robot = RebotArm("rebotarm_dm.yaml")
    robot._serial_link = "usb-cdc"
    robot.validate_control_transport()
    assert robot.rate == 500.
    robot._serial_link = "uart"
    robot._baud = 2000000  # Only use on a bridge known to support this baud.
    robot.validate_control_transport()


def test_dm_new_automatic_feedback_avoids_redundant_requests(dm_controller):
    ctrl, robot, clock = dm_controller
    previous = {name: motor.requests for name, motor in robot._motor_map.items()}
    clock["now"] += .02
    for name in robot.arm.joint_names:
        motor = robot._motor_map[name]
        motor.sequence += 1
        motor.received = clock["now"]
    ctrl._feedback.refresh()
    assert {name: motor.requests for name, motor in robot._motor_map.items()} == previous


def test_dm_posvel_rejects_untransmittable_friction_torque():
    robot = RebotArm("rebotarm_dm.yaml")
    robot.motion_control["friction"]["enabled"] = True
    with pytest.raises(ValueError, match="requires MIT"):
        RebotArmEndPose(robot)


def test_dm_serial_send_failure_latches_hold(dm_controller):
    ctrl, robot, _ = dm_controller
    robot._motor_map["joint3"].fail_send = True
    ctrl._loop_cb(robot, .004)
    assert "Motor command failed" in ctrl._fault
    assert robot.arm.send_error_count > 0
    assert np.all(ctrl._qd_target == 0)


def test_dm_preflight_is_read_only(dm_controller, monkeypatch, tmp_path):
    from tools.rs_hardware_check import probe
    from motorbridge import abi
    _, robot, _ = dm_controller
    robot._ctrl_thread = None
    monkeypatch.setattr(pin, "computeGeneralizedGravity", lambda model, data, q: np.zeros(model.nv), raising=False)
    monkeypatch.setattr(robot, "connect", lambda: setattr(robot, "_connected", True))
    monkeypatch.setattr(abi, "get_abi", lambda: SimpleNamespace(lib=SimpleNamespace(
        _name="fake-DM-ABI", rebotarm_dm_serial_split_v1=lambda: 1,
        rebotarm_dm_get_state_timed=lambda: None, rebotarm_dm_send_batch=lambda: None)))
    monkeypatch.setattr(robot, "disable_all", lambda: pytest.fail("read-only disabled motors"))
    report = probe(robot, .12, tmp_path / "dm.csv")
    assert report["ready_for_control"]
    assert len(report["dm_ranges"]) == 7
    assert report["transport"] == "dm-serial"
    assert all(not m.commands and m.enable_count == 0 for m in robot._motor_map.values())


@pytest.mark.parametrize("status,age,sequence,has_value,expected", [
    (1, .2, 8, 1, None), (0, .1, 1, 1, None), (0, np.inf, 0, 0, None),
    (10, .01, 1, 1, "fault"), (1, np.nan, 1, 1, "Invalid"), (1, -.1, 1, 1, "Invalid"),
    (1, .1, 0, 1, "Invalid"),
])
def test_dm_ctypes_timed_snapshot_preserves_native_age(monkeypatch, status, age, sequence, has_value, expected):
    def function(handle, state, seq, received_age, limits):
        assert handle == 123
        value = ct.cast(state, ct.POINTER(CState)).contents
        value.has_value, value.pos, value.status_code = has_value, .7, status
        ct.cast(seq, ct.POINTER(ct.c_uint64)).contents.value = sequence
        ct.cast(received_age, ct.POINTER(ct.c_double)).contents.value = age
        limits[0], limits[1], limits[2] = 12.5, 10., 28.
        return 0
    motor = SimpleNamespace(_abi=SimpleNamespace(lib=SimpleNamespace(rebotarm_dm_get_state_timed=function)),
                            _require_open=lambda: 123)
    monkeypatch.setattr(dm_feedback.time, "monotonic", lambda: 100.)
    if expected:
        with pytest.raises(dm_feedback.DmFeedbackFault, match=expected):
            dm_feedback.read_dm_snapshot(motor)
    else:
        snapshot = dm_feedback.read_dm_snapshot(motor)
        assert snapshot.sequence == sequence
        assert snapshot.sampled_at == (100. - age if has_value else -np.inf)
        assert snapshot.limits == (12.5, 10., 28.)


@pytest.mark.parametrize("posvel,feedback", [(False, False), (True, False), (False, True)])
def test_dm_ctypes_batch_keeps_motor_order_and_command_layout(posvel, feedback):
    captured = []
    def function(handles, commands, count, mode):
        captured.append((list(handles), np.ctypeslib.as_array(commands, shape=(count * 5,)).copy(), mode))
        return 0
    lib = SimpleNamespace(rebotarm_dm_send_batch=function)
    motors = [SimpleNamespace(_abi=SimpleNamespace(lib=lib), _require_open=lambda id=id: id) for id in (11, 22)]
    values = np.array([[.2, .01, 100., 2., 1.], [.4, .03, 30., 1., .5]])
    assert dm_feedback.send_dm_batch(motors, values, posvel=posvel, feedback=feedback)
    assert captured[0][0] == [11, 22]
    np.testing.assert_allclose(captured[0][1], values.ravel())
    assert captured[0][2] == (3 if feedback else 2 if posvel else 1)


def attach_batch_abi(robot, *, drop_after_first=False, fail=False):
    by_id = {j.motor_id: robot._motor_map[j.name] for j in robot._all_joints}
    calls = []
    def function(handles, commands, count, mode):
        calls.append((list(handles), mode))
        if fail:
            return -1
        for index, address in enumerate(handles):
            if drop_after_first and index > 0:
                continue
            motor = by_id[address]
            values = [commands[index * 5 + col] for col in range(5)]
            if mode == 3:
                motor.request_feedback()
            elif mode == 2:
                motor.send_pos_vel(*values[:2])
            else:
                motor.send_mit(*values)
        return 0
    lib = SimpleNamespace(rebotarm_dm_send_batch=function, motor_last_error_message=lambda: b"injected batch error")
    for address, motor in by_id.items():
        motor._abi = SimpleNamespace(lib=lib)
        motor._require_open = lambda address=address: address
    return calls


def test_dm_group_uses_native_batch_for_queries_and_control(dm_controller):
    ctrl, robot, clock = dm_controller
    calls = attach_batch_abi(robot)
    clock["now"] += .02
    ctrl._feedback.refresh()
    assert calls == [([1, 2, 3, 4, 5, 6], 3)]
    ctrl._loop_cb(robot, .002)
    assert calls[-2:] == [([1, 2, 3, 4, 5, 6], 2), ([7], 1)]
    assert ctrl._fault is None


def test_dm_bridge_dropping_a_batch_is_rejected_before_enable(dm_controller, monkeypatch):
    ctrl, robot, _ = dm_controller
    attach_batch_abi(robot, drop_after_first=True)
    ctrl._running = False
    robot._ctrl_thread = None
    monkeypatch.setattr(robot, "connect", lambda: None)
    monkeypatch.setattr(robot, "require_isolated_feedback", lambda: None)
    with pytest.raises(TimeoutError, match="joint2"):
        ctrl.start()
    assert all(m.enable_count == 0 and not m.commands for m in robot._motor_map.values())


def test_dm_batch_can_be_disabled_without_changing_frequency(dm_controller):
    ctrl, robot, _ = dm_controller
    calls = attach_batch_abi(robot)
    robot.arm.dm_batch_send = robot.gripper.dm_batch_send = False
    ctrl._feedback.refresh()
    ctrl._loop_cb(robot, .002)
    assert robot.rate == 500.
    assert not calls
    assert all(robot._motor_map[name].commands for name in robot.arm.joint_names)


def test_dm_native_batch_failure_is_not_silently_retried_as_single_sends(dm_controller):
    ctrl, robot, _ = dm_controller
    calls = attach_batch_abi(robot, fail=True)
    with pytest.raises(CallError, match="injected batch error"):
        robot.arm.send_pos_vel(ctrl._q_target, strict=True)
    assert calls == [([1, 2, 3, 4, 5, 6], 2)]
    assert robot.arm.send_error_count == 1
    assert all(not m.commands for m in robot._motor_map.values())
