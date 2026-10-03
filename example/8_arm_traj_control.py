#!/usr/bin/env python3
"""
RebotArmEndPose 交互控制示例（轨迹规划模式）。
输入: x y z [roll pitch yaw] [duration]  目标末端位置（米 / 弧度 / 秒）
      g <pos>                            设置夹爪目标位置

RebotArmEndPose interactive control example (trajectory planning mode).
Input: x y z [roll pitch yaw] [duration]  target end-effector pose (meters / radians / seconds)
       g <pos>                            set gripper target position

用法 / Usage:
    python example/8_arm_traj_control.py

退出 / Exit: q / quit / exit /ctrl+c
状态 / State: state, end_state
诊断 / Diagnostics: status, log <csv_path>, stop, clear_fault
保持姿态向上 / Lift preserving orientation: lift 0.1 8
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.controllers import RebotArmEndPose


def main() -> None:
    rebotarm = RebotArm()
    ctrl = RebotArmEndPose(rebotarm)

    ctrl.start()
    print("--- 已启动末端位置控制器 ---\n")
    print("--- End-effector pose controller started ---\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not line:
            continue
        if line.lower() in ("q", "quit", "exit"):
            break

        if line.lower() == "state":
            q, _, _ = rebotarm.get_state()
            try:
                q[:rebotarm.arm.num_joints] = ctrl.get_joint_positions()
            except RuntimeError as error:
                print(f"  {error}")
                continue
            print(f"  机械臂 / Arm (rad): {[f'{v:+.3f}' for v in q[:rebotarm.arm.num_joints]]}")
            if rebotarm.has_gripper:
                print(f"  夹爪 / Gripper (rad): {q[rebotarm.arm.num_joints]:+.3f}")
            continue

        if line.lower() == "end_state":
            try:
                q = ctrl.get_joint_positions()
            except RuntimeError as error:
                print(f"  {error}")
                continue
            from reBotArm_control_py.kinematics import joint_to_pose
            pos, rpy = joint_to_pose(q)
            px, py, pz = float(pos[0]), float(pos[1]), float(pos[2])
            rx, ry, rz = float(rpy[0]), float(rpy[1]), float(rpy[2])
            print(f"  pos=[{px:+.3f} {py:+.3f} {pz:+.3f}] m  rpy=[{rx:+.2f} {ry:+.2f} {rz:+.2f}] rad")
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "status":
            print(ctrl.motion_status)
            continue
        if cmd == "log" and len(parts) == 2:
            print(f"  log -> {ctrl.export_motion_log(parts[1])}")
            continue
        if cmd == "stop":
            ctrl.stop_motion()
            continue
        if cmd == "clear_fault":
            try:
                ctrl.clear_fault()
            except RuntimeError as error:
                print(f"  {error}")
            continue
        if cmd == "lift":
            try:
                if len(parts) not in (2, 3):
                    raise ValueError("lift <dz_m> [duration_s]")
                ok = ctrl.move_relative(dz=float(parts[1]),
                                        duration=float(parts[2]) if len(parts) == 3 else 2.0)
                print(f"  lift -> {'ok' if ok else 'failed'}")
            except (ValueError, RuntimeError) as error:
                print(f"  {error}")
            continue

        if cmd == "g" and len(parts) >= 2:
            try:
                pos = float(parts[1])
                ctrl.set_gripper_target(pos)
                print(f"  夹爪 / Gripper -> {pos:.3f} rad")
            except ValueError:
                print("  用法 / Usage: g <pos>")
            continue

        try:
            vals = [float(v) for v in parts]
        except ValueError:
            print("  格式 / Format: x y z [roll pitch yaw] [duration]")
            continue

        if len(vals) not in (3, 6, 7):
            print("  格式 / Format: x y z [roll pitch yaw] [duration]")
            continue
        x, y, z = vals[0], vals[1], vals[2]
        roll = vals[3] if len(vals) >= 6 else None
        pitch = vals[4] if len(vals) >= 6 else None
        yaw = vals[5] if len(vals) >= 6 else None
        duration = vals[6] if len(vals) >= 7 else 2.0

        ok = ctrl.move_to_traj(
            x=x, y=y, z=z,
            roll=roll, pitch=pitch, yaw=yaw,
            duration=duration,
        )
        print(f"  -> ({x:+.3f}, {y:+.3f}, {z:+.3f})  "
              f"T={duration:.1f}{'ok' if ok else 'failed'}")

    ctrl.end()
    print("\n完成 / Done.")


if __name__ == "__main__":
    main()
