"""Read-only RS/DM preflight: never enable, switch mode, zero or disable motors."""
from __future__ import annotations

import argparse
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pinocchio as pin
from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose


def probe(robot, seconds, path):
    ctrl = RebotArmEndPose(robot)  # Load this hardware's URDF; do not start control.
    vendors = {j.vendor for j in robot.arm._jcfgs}
    if vendors not in ({"robstride"}, {"damiao"}):
        raise ValueError("This preflight requires an RS or DM arm configuration")
    report = dict(platform=platform.platform(), python=platform.python_version(),
                  motorbridge=version("motorbridge"), pinocchio=pin.__version__,
                  hardware_config=str(robot.hardware_config_path), channel=robot._channel,
                  transport=robot.transport, serial_budget=robot.serial_budget(),
                  errors=[], samples=0, joint_names=robot.arm.joint_names)
    rows = []
    try:
        robot.connect()
        report["tx_gap_us"] = os.getenv("MOTORBRIDGE_TX_GAP_US", "SDK default")
        from motorbridge.abi import get_abi
        report["abi_library"] = str(get_abi().lib._name)
        marker_name = ("rebotarm_rs_feedback_lock_fix_v1" if vendors == {"robstride"}
                       else "rebotarm_dm_serial_split_v1")
        marker = getattr(get_abi().lib, marker_name, None)
        report["feedback_lock_patch"] = bool(marker is not None and marker() == 1)
        robot.require_isolated_feedback()
        if vendors == {"damiao"}:
            from reBotArm_control_py.actuator.dm_feedback import check_dm_ranges
            report["dm_ranges"] = {}
            for group in robot.groups.values():
                if hasattr(group, "_jcfgs"):
                    report["dm_ranges"].update(check_dm_ranges(group))
        began = time.monotonic()
        while time.monotonic() - began < seconds:
            cycle = time.monotonic()
            try:
                sample = ctrl._feedback.refresh()
                pose = ctrl._pose(sample.q, ctrl._ik_data)
                q_full = pin.neutral(ctrl._model)
                q_full[:ctrl._n] = sample.q
                gravity = np.asarray(pin.computeGeneralizedGravity(ctrl._model, ctrl._gravity_data, q_full))[:ctrl._n]
                valid = (np.all(sample.q >= ctrl._model.lowerPositionLimit[:ctrl._n] - 1e-3)
                         and np.all(sample.q <= ctrl._model.upperPositionLimit[:ctrl._n] + 1e-3))
                rows.append((cycle - began, sample.read_span, sample.read_duration,
                             int(valid), *sample.q, *pose.translation, *gravity))
            except Exception as error:
                report["errors"].append(str(error))
            time.sleep(max(0., .04 - (time.monotonic() - cycle)))
    except Exception as error:
        report["errors"].append(str(error))
    finally:
        # close_bus stops native RX and closes the adapter; shutdown would disable.
        try:
            robot.disconnect(disable_motors=False)
        except Exception as error:
            report["errors"].append(f"Disconnect failed: {error}")
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["time", "read_span", "read_duration", "within_urdf_limits",
                         *robot.arm.joint_names, "x", "y", "z",
                         *["gravity_" + name for name in robot.arm.joint_names]])
        writer.writerows(rows)
    report["samples"] = len(rows)
    if rows:
        values = np.asarray(rows)
        report["max_read_span_s"] = float(values[:, 1].max())
        report["max_read_duration_s"] = float(values[:, 2].max())
        report["all_within_urdf_limits"] = bool(np.all(values[:, 3] == 1))
        report["joint_peak_to_peak_rad"] = np.ptp(values[:, 4:4 + ctrl._n], axis=0).tolist()
    report["ready_for_control"] = bool(rows and not report["errors"]
        and report["feedback_lock_patch"] and report["all_within_urdf_limits"]
        and report["max_read_span_s"] <= ctrl._max_feedback_span
        and report["max_read_duration_s"] < ctrl._max_feedback_age
        and (report["serial_budget"] is None or report["serial_budget"]["valid"]))
    path.with_suffix(".json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(default_hw="rebotarm_rs.yaml", default_output="logs/rs_preflight.csv"):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hw", default=default_hw)
    parser.add_argument("--channel", help="Override CAN channel or serial port")
    parser.add_argument("--seconds", type=float, default=10.)
    parser.add_argument("--output", default=default_output)
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        parser.error("seconds must be finite and positive")
    report = probe(RebotArm(args.hw, channel=args.channel), args.seconds, args.output)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0 if report["ready_for_control"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
