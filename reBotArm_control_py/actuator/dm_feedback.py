"""Coherent DM feedback through the explicitly exported, reviewed C ABI."""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass
import time

import numpy as np
from motorbridge import CallError
from motorbridge.abi import CState


@dataclass(frozen=True)
class DmSnapshot:
    pos: float
    vel: float
    torq: float
    status: int
    sequence: int
    sampled_at: float
    limits: tuple[float, float, float]


class DmFeedbackFault(RuntimeError):
    """A received motor fault or corrupt sample, rather than an RX timeout."""


def read_dm_snapshot(motor) -> DmSnapshot:
    function = getattr(motor._abi.lib, "rebotarm_dm_get_state_timed", None)
    if function is None:
        raise RuntimeError("DM timed feedback ABI missing; build tools/build_motorbridge_feedback.py and restart Python")
    function.argtypes = [ct.c_void_p, ct.POINTER(CState), ct.POINTER(ct.c_uint64),
                         ct.POINTER(ct.c_double), ct.POINTER(ct.c_float)]
    function.restype = ct.c_int
    state, sequence, age, limits = CState(), ct.c_uint64(), ct.c_double(), (ct.c_float * 3)()
    began = time.monotonic()
    rc = function(motor._require_open(), ct.byref(state), ct.byref(sequence), ct.byref(age), limits)
    if rc != 0:
        raise CallError("DM timed feedback snapshot failed")
    # Uninitialized is a valid baseline before the first feedback request.
    if not state.has_value:
        return DmSnapshot(0., 0., 0., 0, 0, -np.inf, tuple(limits))
    if (sequence.value == 0 or not np.isfinite(age.value) or age.value < 0
            or not np.all(np.isfinite([state.pos, state.vel, state.torq, *limits]))):
        raise DmFeedbackFault("Invalid DM timed feedback")
    if state.status_code not in (0, 1):
        raise DmFeedbackFault(f"DM motor fault status 0x{state.status_code:X}")
    # Rust Instant and Python monotonic have different origins. Subtract native
    # receive age from the local call start, conservatively dating the packet.
    return DmSnapshot(state.pos, state.vel, state.torq, state.status_code,
                      sequence.value, began - age.value, tuple(limits))


def send_dm_batch(motors, commands, *, posvel=False, feedback=False):
    """One native call and one serial write per group; complete protocol packets."""
    function = getattr(getattr(getattr(motors[0], "_abi", None), "lib", None),
                       "rebotarm_dm_send_batch", None)
    if function is None:
        return False  # Other callers can still use the unmodified wheel API.
    values = np.ascontiguousarray(commands, dtype=np.float32)
    if values.shape != (len(motors), 5) or not np.all(np.isfinite(values)):
        raise ValueError("Invalid DM batch commands")
    handles = (ct.c_void_p * len(motors))(*(motor._require_open() for motor in motors))
    function.argtypes = [ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_float), ct.c_uint32, ct.c_uint32]
    function.restype = ct.c_int
    mode = 3 if feedback else 2 if posvel else 1
    if function(handles, values.ctypes.data_as(ct.POINTER(ct.c_float)), len(motors), mode) != 0:
        message = motors[0]._abi.lib.motor_last_error_message()
        raise CallError(f"DM batch send failed: {message.decode() if message else 'unknown error'}")
    return True


def check_dm_ranges(group, timeout_ms=100):
    """Read PMAX/VMAX/TMAX once before servo operation; never alter firmware."""
    report = {}
    for joint in group._jcfgs:
        if joint.vendor != "damiao":
            continue
        motor = group._mm[joint.name]
        expected = read_dm_snapshot(motor).limits
        actual = tuple(float(motor.get_register_f32(index, timeout_ms)) for index in (21, 22, 23))
        valid = np.all(np.isfinite(actual)) and np.allclose(actual, expected, rtol=1e-3, atol=1e-4)
        report[joint.name] = dict(model=joint.model, sdk=list(expected), firmware=list(actual), matched=bool(valid))
        if not valid:
            raise RuntimeError(f"{joint.name} PMAX/VMAX/TMAX {actual} differ from SDK {expected}; select the matching model")
    return report
