"""reBotArm 分组控制系统 — JointGroup 架构。

配置驱动的硬件抽象层：
  - 所有参数均在 config/rebotarm.yaml 中定义（hardware_yaml 指定硬件配置文件）
  - 关节按 groups 分组，每组独立控制模式
  - 统一 loop 中按组顺序同步发送，防止总线争用

使用示例::

    # arm 组 POS_VEL，gripper 组 MIT（解耦混合控制）
    arm = RebotArm()
    arm.connect()
    arm.arm.enable()
    arm.gripper.enable()
    arm.arm.mode_pos_vel()
    arm.gripper.mode_mit()

    def loop(ref, dt):
        ref.arm.send_pos_vel(joint_pos)
        ref.gripper.send_mit(gripper_pos)

    arm.start_control_loop(loop)

    # 全部组 MIT（纯测试）
    arm.arm.mode_mit()
    arm.gripper.mode_mit()

    arm.disconnect()
"""
from __future__ import annotations

import threading
import time
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import yaml

from motorbridge import Controller, Mode, CallError

_CFG_DIR = Path(__file__).parent.parent.parent / "config"
_GLOBAL_CFG = _CFG_DIR / "rebotarm.yaml"


def _resolve_hw_cfg_path(hw_yaml: str | None = None) -> Path:
    if hw_yaml is None:
        if not _GLOBAL_CFG.exists():
            raise FileNotFoundError(f"{_GLOBAL_CFG} not found")
        data = yaml.safe_load(_GLOBAL_CFG.read_text(encoding="utf-8"))
        hw_yaml = data.get("hardware_yaml") if data else None
        if not hw_yaml:
            raise ValueError("hardware_yaml not set in rebotarm.yaml")

    p = Path(hw_yaml)
    if p.is_absolute():
        return p
    path = _CFG_DIR / hw_yaml
    if path.exists():
        return path
    raise FileNotFoundError(f"hardware config not found: {path}")


# --------------------------------------------------------------------------
# 配置加载
# --------------------------------------------------------------------------

@dataclass
class JointCfg:
    name: str
    motor_id: int
    feedback_id: int
    model: str
    vendor: str = "damiao"
    kp: float = 0.0
    kd: float = 0.0
    vel_kp: float = 0.0
    vel_ki: float = 0.0
    pos_kp: float = 0.0
    pos_ki: float = 0.0
    vlim: float = 0.0


def load_cfg(hw_yaml: str | None = None) -> dict:
    hw_path = _resolve_hw_cfg_path(hw_yaml)

    with open(hw_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    joints = []
    for j in data.get("joints", []):
        mc = j.get("MIT", {})
        pc = j.get("POS_VEL", {})
        joints.append(JointCfg(
            name=j["name"],
            motor_id=int(j["motor_id"]),
            feedback_id=int(j["feedback_id"]),
            model=str(j.get("model", "4340P")),
            vendor=str(j.get("vendor", "damiao")).lower(),
            kp=float(mc.get("kp", 0.0)),
            kd=float(mc.get("kd", 0.0)),
            vel_kp=float(pc.get("vel_kp", 0.0)),
            vel_ki=float(pc.get("vel_ki", 0.0)),
            pos_kp=float(pc.get("pos_kp", 0.0)),
            pos_ki=float(pc.get("pos_ki", 0.0)),
            vlim=float(pc.get("vlim", 2.0)),
        ))

    return {
        "name": data.get("name", "reBotArm"),
        "channel": data.get("channel", "/dev/ttyACM0"),
        "transport": str(data.get("transport", "auto")),
        "baud": int(data.get("baud", 921600)),
        "serial_link": str(data.get("serial_link", "auto")),
        "rate": float(data.get("rate", 500.0)),
        "groups": data.get("groups", {}),
        "joints": joints,
        "arm_control_mode": str(data.get("arm_control_mode", "posvel")),
        "motion_control": data.get("motion_control", {}) or {},
    }


def load_gravity_compensation_config(
    profile: str,
    hw_yaml: str | None = None,
) -> dict:
    """Load one gravity-compensation profile from the active hardware YAML.

    Profile values override the legacy/default keys directly under
    ``gravity_compensation``. Hardware YAML files without ``profiles`` remain
    compatible and return their legacy/default gravity-compensation mapping.
    """
    hw_path = _resolve_hw_cfg_path(hw_yaml)
    with open(hw_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    gravity_cfg = data.get("gravity_compensation", {}) or {}
    if not isinstance(gravity_cfg, dict):
        raise TypeError("gravity_compensation must be a mapping")

    profiles = gravity_cfg.get("profiles", {}) or {}
    if not isinstance(profiles, dict):
        raise TypeError("gravity_compensation.profiles must be a mapping")

    if not profiles:
        return {
            key: value
            for key, value in gravity_cfg.items()
            if key != "profiles"
        }
    if profile not in profiles:
        available = ", ".join(sorted(str(name) for name in profiles))
        raise KeyError(
            f"Unknown gravity-compensation profile {profile!r}; "
            f"available: {available}"
        )

    profile_cfg = profiles[profile]
    if not isinstance(profile_cfg, dict):
        raise TypeError(
            f"gravity_compensation.profiles.{profile} must be a mapping"
        )

    merged = {
        key: value
        for key, value in gravity_cfg.items()
        if key != "profiles"
    }
    merged.update(profile_cfg)
    return merged


# --------------------------------------------------------------------------
# NoOpGroup — 无执行器时的空操作桩
# --------------------------------------------------------------------------

class NoOpGroup:
    """当配置中不存在 gripper 组时的空实现。

    所有属性和方法与 JointGroup 接口兼容，但不对电机发送任何指令，
    方便用户代码在有/无夹爪时共用同一套逻辑，无需条件判断。
    """

    name: str = "gripper"
    _mode: str = "mit"

    @property
    def num_joints(self) -> int:
        return 0

    @property
    def joint_names(self) -> List[str]:
        return []

    @property
    def mode(self) -> str:
        return "mit"

    def enable(self) -> None:
        pass

    def disable(self) -> None:
        pass

    def mode_mit(self, kp=None, kd=None) -> bool:
        self._mode = "mit"
        return True

    def mode_pos_vel(self, vlim=None) -> bool:
        self._mode = "pos_vel"
        return True

    def mode_vel(self) -> bool:
        self._mode = "vel"
        return True

    def send_mit(self, pos, vel=None, kp=None, kd=None, tau=None) -> None:
        pass

    def send_pos_vel(self, pos, vlim=None) -> None:
        pass

    def send_vel(self, vel) -> None:
        pass

    def get_positions(self) -> np.ndarray:
        return np.array([], dtype=np.float64)

    def get_velocities(self) -> np.ndarray:
        return np.array([], dtype=np.float64)

    def __repr__(self) -> str:
        return "NoOpGroup(gripper, no actuator)"


# --------------------------------------------------------------------------
# JointGroup — 单组关节控制
# --------------------------------------------------------------------------

class JointGroup:
    """一组关节的独立控制器。

    每组拥有独立的控制模式（MIT / POS_VEL）、PID 参数和电机列表，
    可单独使能、切换模式、发送命令。

    由 RebotArm 通过 __getattr__ 代理访问，例如 arm.arm / arm.gripper。
    组内关节数量、顺序由配置决定。
    """

    def __init__(
        self,
        name: str,
        joint_names: List[str],
        all_joints: List[JointCfg],
        motor_map: Dict[str, any],
        ctrl_map: Dict[str, Controller],
    ) -> None:
        self.name = name
        self._jn: List[str] = joint_names
        self._jcfgs: List[JointCfg] = [
            next(j for j in all_joints if j.name == n) for n in joint_names
        ]
        self._mm: Dict[str, any] = motor_map
        self._cm: Dict[str, Controller] = ctrl_map
        self._mode: str = "mit"
        self._mit_kp: np.ndarray = np.array([j.kp for j in self._jcfgs], dtype=np.float64)
        self._mit_kd: np.ndarray = np.array([j.kd for j in self._jcfgs], dtype=np.float64)
        self._pv_vlim: np.ndarray = np.array([j.vlim for j in self._jcfgs], dtype=np.float64)
        self._sent_pv_vlim = np.full(len(self._jn), np.nan)
        self.send_error_count = 0
        self.last_send_error: str | None = None
        self._position_readers = {}
        self._position_pool = None
        self.last_position_read_span = 0.0
        self.last_position_read_duration = 0.0
        self._last_dm_sequences = None
        self._last_dm_samples = None
        self.dm_receive_rate_hz = {}
        self.dm_batch_send = True

    # ── 属性 ────────────────────────────────────────────────────────────

    @property
    def num_joints(self) -> int:
        return len(self._jn)

    @property
    def joint_names(self) -> List[str]:
        return list(self._jn)

    @property
    def mode(self) -> str:
        return self._mode

    # ── 使能 / 失能 ────────────────────────────────────────────────────

    def enable(self, *, strict: bool = False) -> None:
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].enable()
            except CallError as e:
                print(f"[{self.name}/enable] {e}")
                if strict:
                    raise
            time.sleep(0.01)

    def disable(self) -> None:
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].disable()
            except CallError as e:
                print(f"[{self.name}/disable] {e}")
            time.sleep(0.01)

    # ── 模式切换 ────────────────────────────────────────────────────────

    def _write_pv_params(self, jc: JointCfg) -> None:
        m = self._mm[jc.name]
        try:
            if jc.vendor == "robstride":
                m.robstride_write_param_f32(0x701F, jc.vel_kp)
                time.sleep(0.01)
                m.robstride_write_param_f32(0x7020, jc.vel_ki)
                time.sleep(0.01)
                m.robstride_write_param_f32(0x701E, jc.pos_kp)
            elif jc.vendor == "damiao":
                m.write_register_f32(25, jc.vel_kp)
                m.write_register_f32(26, jc.vel_ki)
                m.write_register_f32(27, jc.pos_kp)
                m.write_register_f32(28, jc.pos_ki)
            time.sleep(0.02)
        except Exception as e:
            raise RuntimeError(f"[{self.name}/pv_params/{jc.name}] {e}") from e

    def mode_mit(
        self,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> bool:
        self._mode = "mit"
        if kp is not None:
            self._mit_kp = np.asarray(kp, dtype=np.float64).reshape(-1)
        if kd is not None:
            self._mit_kd = np.asarray(kd, dtype=np.float64).reshape(-1)
        ok = True
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].ensure_mode(Mode.MIT, 1000)
            except CallError as e:
                print(f"[{self.name}/mode_mit/{jc.name}] {e}")
                ok = False
            time.sleep(0.05)
        time.sleep(0.2)
        return ok

    def mode_pos_vel(
        self,
        vlim: Optional[np.ndarray] = None,
    ) -> bool:
        if vlim is not None:
            self._pv_vlim = np.asarray(vlim, dtype=np.float64).reshape(-1)
        if (self._pv_vlim.shape != (self.num_joints,)
                or not np.all(np.isfinite(self._pv_vlim)) or np.any(self._pv_vlim <= 0)):
            raise ValueError("Positive finite speed limits required per joint")
        ok = True
        for i, jc in enumerate(self._jcfgs):
            try:
                self._write_pv_params(jc)
                mode = Mode.ROBSTRIDE_POS_VEL_CSP if jc.vendor == "robstride" else Mode.POS_VEL
                self._mm[jc.name].ensure_mode(mode, 1000)
                if jc.vendor == "robstride":
                    self._mm[jc.name].robstride_write_param_f32(0x7017, float(self._pv_vlim[i]))
                    self._sent_pv_vlim[i] = self._pv_vlim[i]
            except (CallError, RuntimeError) as e:
                print(f"[{self.name}/mode_pos_vel/{jc.name}] {e}")
                ok = False
            time.sleep(0.05)
        time.sleep(0.2)
        if ok:
            self._mode = "pos_vel"
        return ok

    def mode_vel(self) -> bool:
        self._mode = "vel"
        ok = True
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].ensure_mode(Mode.VEL, 1000)
            except CallError as e:
                print(f"[{self.name}/mode_vel/{jc.name}] {e}")
                ok = False
            time.sleep(0.05)
        time.sleep(0.2)
        return ok

    # ── MIT 发送 ────────────────────────────────────────────────────────

    def send_mit(
        self,
        pos: np.ndarray,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        tau: Optional[np.ndarray] = None,
        *,
        strict: bool = False,
    ) -> None:
        n = self.num_joints
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if vel is None:
            vel = np.zeros(n)
        if tau is None:
            tau = np.zeros(n)
        if kp is None:
            kp = self._mit_kp
        if kd is None:
            kd = self._mit_kd
        vectors = [np.asarray(x, dtype=float).reshape(-1) for x in (pos, vel, kp, kd, tau)]
        if any(x.shape != (n,) or not np.all(np.isfinite(x)) for x in vectors):
            raise ValueError("MIT commands require one finite value per joint")
        pos, vel, kp, kd, tau = vectors
        if self.dm_batch_send and all(j.vendor == "damiao" for j in self._jcfgs) and self.num_joints:
            from .dm_feedback import send_dm_batch
            try:
                if send_dm_batch([self._mm[j.name] for j in self._jcfgs], np.column_stack(vectors)):
                    return
            except CallError as error:
                self.send_error_count += 1
                self.last_send_error = str(error)
                if strict:
                    raise
                return
        first_error = None
        for i, jc in enumerate(self._jcfgs):
            try:
                self._mm[jc.name].send_mit(
                    float(pos[i]),
                    float(vel[i]),
                    float(kp[i]),
                    float(kd[i]),
                    float(tau[i]),
                )
            except CallError as error:
                self.send_error_count += 1
                self.last_send_error = f"{jc.name}: {error}"
                first_error = first_error or error
        if strict and first_error is not None:
            raise first_error

    # ── POS_VEL 发送 ───────────────────────────────────────────────────

    def send_pos_vel(
        self,
        pos: np.ndarray,
        vlim: Optional[np.ndarray] = None,
        *,
        strict: bool = False,
    ) -> None:
        pos = np.asarray(pos, dtype=np.float64).reshape(-1)
        if vlim is None:
            vlim = self._pv_vlim
        vlim = np.asarray(vlim, dtype=np.float64).reshape(-1)
        if (pos.shape != (self.num_joints,) or vlim.shape != pos.shape
                or not np.all(np.isfinite(pos)) or not np.all(np.isfinite(vlim))
                or np.any(vlim <= 0)):
            raise ValueError("POS_VEL requires finite positions and positive speed limits per joint")
        first_error = None
        if self.dm_batch_send and all(j.vendor == "damiao" for j in self._jcfgs) and self.num_joints:
            from .dm_feedback import send_dm_batch
            try:
                values = np.column_stack((pos, vlim, np.zeros((self.num_joints, 3))))
                if send_dm_batch([self._mm[j.name] for j in self._jcfgs], values, posvel=True):
                    return
            except CallError as error:
                self.send_error_count += 1
                self.last_send_error = str(error)
                if strict:
                    raise
                return
        for i in range(self.num_joints):
            try:
                jc = self._jcfgs[i]
                motor = self._mm[jc.name]
                if jc.vendor == "robstride":
                    # SDK CSP helper re-enables and rewrites run_mode every call.
                    # Mode/limit were configured once; stream only loc_ref.
                    if self._sent_pv_vlim[i] != vlim[i]:
                        motor.robstride_write_param_f32(0x7017, float(vlim[i]))
                        self._sent_pv_vlim[i] = vlim[i]
                    motor.robstride_write_param_f32(0x7016, float(pos[i]))
                else:
                    motor.send_pos_vel(float(pos[i]), float(vlim[i]))
            except CallError as error:
                self.send_error_count += 1
                self.last_send_error = f"{self._jcfgs[i].name}: {error}"
                first_error = first_error or error
        if strict and first_error is not None:
            raise first_error

    # ── VEL 发送 ───────────────────────────────────────────────────────

    def send_vel(self, vel: np.ndarray) -> None:
        vel = np.asarray(vel, dtype=np.float64).reshape(-1)
        for i in range(min(len(vel), self.num_joints)):
            try:
                self._mm[self._jcfgs[i].name].send_vel(float(vel[i]))
            except CallError:
                pass

    # ── 状态读取 ───────────────────────────────────────────────────────

    def _poll_feedback(self) -> None:
        """仅处理 CAN 接收队列中的反馈帧（快速，无总线发送）。"""
        seen: set[str] = set()
        for jc in self._jcfgs:
            if jc.vendor not in seen:
                seen.add(jc.vendor)
                try:
                    self._cm[jc.vendor].poll_feedback_once()
                except Exception:
                    pass

    def _request_feedback(self) -> None:
        """发送显式反馈请求帧 + 处理接收队列（慢，有总线发送）。"""
        for jc in self._jcfgs:
            try:
                self._mm[jc.name].request_feedback()
            except Exception:
                pass
        self._poll_feedback()

    def get_positions(self, request_feedback: bool = True, *, strict: bool = False) -> np.ndarray:
        if request_feedback:
            self._request_feedback()
        else:
            self._poll_feedback()
        
        out: list[float] = []
        for jc in self._jcfgs:
            m = self._mm[jc.name]
            st = m.get_state()
            if st is not None:
                out.append(st.pos)
            else:
                # 缓存为空时回退到 SDO 读取（安全兜底）
                if request_feedback and jc.vendor == "robstride":
                    try:
                        out.append(float(m.robstride_get_param_f32(0x7019)))
                        continue
                    except CallError:
                        pass
                if strict:
                    raise RuntimeError(f"No position feedback for {jc.name}")
                out.append(0.0)
        return np.array(out, dtype=np.float64)

    def read_position_sample(self, timeout_ms: int = 20):
        """Acquire new hardware feedback outside the servo loop."""
        if all(j.vendor == "robstride" for j in self._jcfgs):
            began = time.monotonic()
            def read(jc):
                reader = self._position_readers.get(jc.name, self._mm[jc.name])
                stamp = time.monotonic()  # Conservative: before the request, not after reply.
                # The ordinary getter imposes >=150 ms per host and probes
                # fallback IDs. Exact-host reads preserve this worker's timeout.
                value = reader.robstride_get_param_f32_host_id(0x7019, jc.feedback_id, timeout_ms)
                return value, stamp
            if self._position_pool is None:
                results = [read(jc) for jc in self._jcfgs]
            else:
                futures = [self._position_pool.submit(read, jc) for jc in self._jcfgs]
                results, error = [], None
                for future in futures:
                    try:
                        results.append(future.result())
                    except Exception as exc:
                        error = error or exc
                # Finish the whole batch before starting another request for the same parameter.
                if error is not None:
                    raise error
            values, stamps = zip(*results)
            q = np.array(values, dtype=float)
            stamp, source = min(stamps), "rs_mechpos"
            self.last_position_read_span = max(stamps) - min(stamps)
            self.last_position_read_duration = time.monotonic() - began
            if self._position_pool is not None:
                source = "rs_mechpos_parallel"
        elif all(j.vendor == "damiao" for j in self._jcfgs):
            from .dm_feedback import read_dm_snapshot, send_dm_batch
            began = time.monotonic()
            motors = [self._mm[j.name] for j in self._jcfgs]
            snapshots = [read_dm_snapshot(m) for m in motors]
            before = self._last_dm_sequences
            if before is None:
                before = [s.sequence for s in snapshots]
            # Continuous control already produces sensor replies. Reuse new
            # packets, and request only missing ones; never refresh old cache.
            missing_motors = [motor for motor, old, snapshot in zip(motors, before, snapshots)
                              if snapshot.sequence <= old]
            if missing_motors:
                sent = self.dm_batch_send and send_dm_batch(missing_motors,
                    np.zeros((len(missing_motors), 5)), feedback=True)
                if not sent:
                    for motor in missing_motors:
                        motor.request_feedback()
            deadline = began + timeout_ms / 1000.
            while True:
                snapshots = [read_dm_snapshot(m) for m in motors]
                missing = [j.name for j, old, s in zip(self._jcfgs, before, snapshots) if s.sequence <= old]
                if not missing:
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"No new DM feedback: {', '.join(missing)}")
                time.sleep(.001)
            q = np.array([s.pos for s in snapshots])
            stamps = [s.sampled_at for s in snapshots]
            stamp, source = min(stamps), "dm_sensor_rx_timed"
            self.last_position_read_span = max(stamps) - min(stamps)
            self.last_position_read_duration = time.monotonic() - began
            self._last_dm_sequences = [s.sequence for s in snapshots]
            if self._last_dm_samples is not None:
                self.dm_receive_rate_hz = {
                    j.name: (new.sequence - old.sequence) / (new.sampled_at - old.sampled_at)
                    for j, old, new in zip(self._jcfgs, self._last_dm_samples, snapshots)
                    if new.sampled_at > old.sampled_at
                }
            self._last_dm_samples = snapshots
        else:
            raise RuntimeError("Position sampling requires a homogeneous RS or DM group")
        if not np.all(np.isfinite(q)):
            raise RuntimeError("Non-finite joint feedback")
        return q, stamp, source

    def get_velocities(self, request_feedback: bool = True) -> np.ndarray:
        # NOTE (RobStride): the cached state has the same staleness problem as
        # get_positions, and the mechVel param (0x701A) was measured NOT to be
        # rad/s on RS firmware (inconsistent scale/sign vs dq/dt, 2026-07-17).
        # For a live velocity on RobStride, finite-difference get_positions().
        if request_feedback:
            self._request_feedback()
        return np.array([
            self._mm[jc.name].get_state().vel
            if self._mm[jc.name].get_state() is not None else 0.0
            for jc in self._jcfgs
        ], dtype=np.float64)

    def __repr__(self) -> str:
        return f"JointGroup({self.name!r}, joints={self.num_joints}, mode={self._mode})"


# --------------------------------------------------------------------------
# RebotArm — 分组控制器容器
# --------------------------------------------------------------------------

class RebotArm:
    """reBotArm 分组控制系统。

    持有多个 JointGroup，每组独立控制模式，独立发送命令，
    在同一个控制循环中按组顺序同步发送，防止总线争用。

    按组访问（通过 __getattr__）::

        arm.arm       # 机械臂关节组
        arm.gripper   # 夹爪关节组（如果有）

    也可以通过 groups 字典::

        arm.groups["arm"]
        arm.groups["gripper"]

    手动添加组::

        arm.add_group("custom", ["joint1", "joint2"])
    """

    def __init__(self, hw_yaml: str | None = None, *, channel: str | None = None) -> None:
        self.hardware_config_path = _resolve_hw_cfg_path(hw_yaml).resolve()
        self._hw_yaml = self.hardware_config_path.name
        cfg = load_cfg(hw_yaml)

        self._name: str = cfg["name"]
        self._channel: str = channel or cfg["channel"]
        self._transport: str = cfg["transport"]
        self._baud: int = cfg["baud"]
        self._serial_link: str = cfg["serial_link"]
        if self._serial_link not in ("auto", "uart", "usb-cdc"):
            raise ValueError("serial_link must be auto, uart or usb-cdc")
        if self._transport not in ("auto", "can", "dm-serial") or self._baud <= 0:
            raise ValueError("Invalid transport or serial baud")
        self._rate: float = cfg["rate"]
        self._all_joints: List[JointCfg] = cfg["joints"]
        self._groups_def: dict = cfg["groups"]
        self._arm_control_mode: str = cfg.get("arm_control_mode", "posvel")
        self.motion_control: dict = cfg["motion_control"]

        self._ctrl_map: Dict[str, Controller] = {}
        self._motor_map: Dict[str, any] = {}
        self._reader_motors = {}
        self._reader_pool = None
        self._groups: Dict[str, JointGroup] = {}

        self._running = False
        self._ctrl_thread: Optional[threading.Thread] = None
        self._ctrl_fn: Optional[Callable] = None
        self._ctrl_rate: float = self._rate
        self._connected: bool = False
        self._control_stop = threading.Event()
        self._timing_lock = threading.Lock()
        self._dt_histogram = np.zeros(10001, dtype=np.uint64)
        self._dt_count = 0
        self.control_loop_stats = {"iterations": 0, "overruns": 0, "last_dt": 0.0,
                                   "max_dt": 0.0, "last_error": None}

        self._build_groups()

    def connect(self) -> None:
        """连接总线、注册电机。模式切换需在 connect 后调用。"""
        if self._connected:
            return
        if any(j.vendor in ("robstride", "damiao") for j in self._all_joints):
            from importlib.metadata import version
            if version("motorbridge") != "0.5.6":
                raise RuntimeError("This reviewed RS/DM backend requires motorbridge==0.5.6")
        gap = self.motion_control.get("tx_gap_us")
        if gap is not None:
            if not np.isfinite(gap) or gap < 0 or int(gap) != gap:
                raise ValueError("tx_gap_us must be a nonnegative integer")
            os.environ.setdefault("MOTORBRIDGE_TX_GAP_US", str(int(gap)))
        # Select the locally built ABI before motorbridge initializes its singleton.
        local_abi = Path(__file__).resolve().parents[2] / ".motorbridge" / sys.platform / {
            "win32": "motor_abi.dll", "darwin": "libmotor_abi.dylib",
        }.get(sys.platform, "libmotor_abi.so")
        if (any(j.vendor in ("robstride", "damiao") for j in self._all_joints)
                and local_abi.exists() and not os.getenv("MOTORBRIDGE_LIB")):
            os.environ["MOTORBRIDGE_LIB"] = str(local_abi)
        try:
            self._setup_motors()
            self._setup_position_readers()
            self._connected = True
        except BaseException:
            cleanup = [self._close_position_readers]
            cleanup.extend(motor.close for motor in self._motor_map.values())
            for ctrl in self._ctrl_map.values():
                cleanup.extend((ctrl.close_bus, ctrl.close))
            for action in cleanup:
                try:
                    action()
                except Exception as error:
                    print(f"[connect cleanup] {error}")
            self._motor_map.clear()
            self._ctrl_map.clear()
            raise

    def _setup_position_readers(self):
        rs = [jc for jc in self._all_joints if jc.vendor == "robstride"]
        if not rs:
            return
        if self.transport == "dm-serial":
            raise ValueError("RS position feedback requires a CAN transport")
        # One controller / receive queue on all platforms, including PCAN.
        # The patched ABI drops MotorHandle's mutex before waiting for a reply.
        self._reader_motors.update({jc.name: self._motor_map[jc.name] for jc in rs})
        self._reader_pool = ThreadPoolExecutor(max_workers=len(rs), thread_name_prefix="rs-position-read")
        for group in self._groups.values():
            if isinstance(group, JointGroup):
                group._position_readers = self._reader_motors
                group._position_pool = self._reader_pool

    def _close_position_readers(self):
        if self._reader_pool is not None:
            self._reader_pool.shutdown(wait=True)
            self._reader_pool = None
        self._reader_motors.clear()
        for group in self._groups.values():
            if isinstance(group, JointGroup):
                group._position_pool = None
                group._last_dm_sequences = None
                group._last_dm_samples = None
                group.dm_receive_rate_hz = {}

    def require_isolated_feedback(self):
        """Require the native feedback/transport fixes for the configured motors."""
        from motorbridge.abi import get_abi
        lib = get_abi().lib
        for vendor, marker_name in (("robstride", "rebotarm_rs_feedback_lock_fix_v1"),
                                    ("damiao", "rebotarm_dm_serial_split_v1")):
            if any(j.vendor == vendor for j in self._all_joints):
                marker = getattr(lib, marker_name, None)
                if marker is None or marker() != 1:
                    raise RuntimeError(f"{vendor} feedback ABI patch missing. Run: python tools/build_motorbridge_feedback.py; restart Python")
        if any(j.vendor == "damiao" for j in self._all_joints) and not hasattr(lib, "rebotarm_dm_get_state_timed"):
            raise RuntimeError("DM timed-state ABI missing; rebuild and restart Python")
        if any(j.vendor == "damiao" for j in self._all_joints) and not hasattr(lib, "rebotarm_dm_send_batch"):
            raise RuntimeError("DM batch-send ABI missing; rebuild and restart Python")

    @property
    def transport(self):
        if self._transport != "auto":
            return self._transport
        channel = self._channel.upper()
        return "dm-serial" if (channel.startswith("COM") or channel.startswith("\\\\.\\COM")
            or self._channel.startswith(("/dev/tty", "/dev/cu"))) else "can"

    def serial_budget(self, rate=None):
        if self.transport != "dm-serial":
            return None
        rate = self._rate if rate is None else float(rate)
        feedback_rate = float(self.motion_control.get("feedback_rate", 25.))
        # 30 serial bytes per CAN command, 8N1 = ten wire bits per byte.
        bps = len(self._all_joints) * 30 * 10 * (rate + feedback_rate)
        limit = float(self.motion_control.get("max_serial_utilization", .75))
        if not np.isfinite(limit) or not 0 < limit < 1:
            raise ValueError("max_serial_utilization must lie between zero and one")
        uart_ok = bool(bps / self._baud <= limit)
        return dict(estimated_tx_bps=bps, baud=self._baud, utilization=bps / self._baud,
                    limit=limit, serial_link=self._serial_link, uart_capacity_sufficient=uart_ok,
                    valid=bool(self._serial_link != "uart" or uart_ok),
                    note="8N1 UART estimate includes worst-case feedback requests. USB CDC throughput must be measured; baud may be nominal.")

    def validate_control_transport(self, rate=None):
        budget = self.serial_budget(rate)
        if budget is not None and not budget["valid"]:
            raise ValueError(f"DM serial command budget {budget['estimated_tx_bps']:.0f} bit/s exceeds "
                             f"{budget['limit']:.0%} of {self._baud} baud; reduce control/feedback rate")

    def _make_controller(self, vendor: str) -> Controller:
        if self.transport == "dm-serial":
            if vendor != "damiao":
                raise ValueError("DM serial bridge requires Damiao motors")
            return Controller.from_dm_serial(self._channel, self._baud)
        return Controller(self._channel)

    def _setup_motors(self) -> None:
        for jc in self._all_joints:
            vendor = jc.vendor
            if vendor not in self._ctrl_map:
                self._ctrl_map[vendor] = self._make_controller(vendor)
            ctrl = self._ctrl_map[vendor]

            if vendor == "damiao":
                mot = ctrl.add_damiao_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif vendor == "robstride":
                mot = ctrl.add_robstride_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif vendor == "myactuator":
                mot = ctrl.add_myactuator_motor(jc.motor_id, jc.feedback_id, jc.model)
            elif vendor == "hightorque":
                mot = ctrl.add_hightorque_motor(jc.motor_id, jc.feedback_id, jc.model)
            else:
                raise ValueError(f"Unsupported vendor: {vendor}")

            self._motor_map[jc.name] = mot

    def _build_groups(self) -> None:
        for gname, gdef in self._groups_def.items():
            joints_def = gdef.get("joints", [])
            g = JointGroup(
                name=gname,
                joint_names=joints_def,
                all_joints=self._all_joints,
                motor_map=self._motor_map,
                ctrl_map=self._ctrl_map,
            )
            g.dm_batch_send = bool(self.motion_control.get("dm_batch_send", True))
            self._groups[gname] = g
        if "gripper" not in self._groups:
            self._groups["gripper"] = NoOpGroup()

    # ── 属性 ────────────────────────────────────────────────────────────

    @property
    def num_joints(self) -> int:
        return len(self._all_joints)

    @property
    def joint_names(self) -> List[str]:
        return [j.name for j in self._all_joints]

    @property
    def groups(self) -> Dict[str, JointGroup]:
        return self._groups

    @property
    def control_loop_active(self) -> bool:
        t = getattr(self, "_ctrl_thread", None)
        return t is not None and t.is_alive()

    @property
    def rate(self) -> float:
        return self._ctrl_rate

    @property
    def has_gripper(self) -> bool:
        return not isinstance(self._groups.get("gripper", None), NoOpGroup)

    @property
    def arm_control_mode(self) -> str:
        """从硬件配置文件读取的默认 arm 控制模式（"mit" 或 "posvel"）。"""
        return self._arm_control_mode

    @property
    def hardware_yaml(self) -> str:
        return self._hw_yaml

    def __getattr__(self, name: str) -> any:
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._groups:
            return self._groups[name]
        raise AttributeError(name)

    # ── 手动添加组 ────────────────────────────────────────────────────

    def add_group(self, name: str, joint_names: List[str]) -> JointGroup:
        if name in self._groups:
            raise ValueError(f"组 {name!r} 已存在")
        g = JointGroup(
            name=name,
            joint_names=joint_names,
            all_joints=self._all_joints,
            motor_map=self._motor_map,
            ctrl_map=self._ctrl_map,
        )
        g.dm_batch_send = bool(self.motion_control.get("dm_batch_send", True))
        self._groups[name] = g
        return g

    # ── 全局使能 / 失能 ────────────────────────────────────────────────

    def enable_all(self, *, strict: bool = False) -> None:
        for g in self._groups.values():
            if not isinstance(g, NoOpGroup):
                g.enable(strict=strict)

    def disable_all(self) -> None:
        for g in self._groups.values():
            g.disable()

    # ── 零点 ────────────────────────────────────────────────────────────

    def set_zero(self, poll_max: int = 200, poll_interval: float = 0.05) -> None:
        self.disable_all()
        time.sleep(0.3)
        for jc in self._all_joints:
            for _ in range(poll_max):
                for m in self._motor_map.values():
                    try:
                        m.request_feedback()
                    except Exception:
                        pass
                for ctrl in self._ctrl_map.values():
                    try:
                        ctrl.poll_feedback_once()
                    except Exception:
                        pass
                st = self._motor_map[jc.name].get_state()
                if st is not None and st.status_code == 0:
                    break
                time.sleep(poll_interval)
            try:
                self._motor_map[jc.name].set_zero_position()
            except CallError as e:
                print(f"[set_zero] {jc.name}: {e}")
            time.sleep(0.1)

    # ── 全局状态读取 ───────────────────────────────────────────────────

    def get_state(
        self,
        request_feedback: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if request_feedback:
            for m in self._motor_map.values():
                try:
                    m.request_feedback()
                except Exception:
                    pass
        for ctrl in self._ctrl_map.values():
            try:
                ctrl.poll_feedback_once()
            except Exception:
                pass
        pos, vel, torq = [], [], []
        for jc in self._all_joints:
            st = self._motor_map[jc.name].get_state()
            if st is not None:
                pos.append(st.pos)
                vel.append(st.vel)
                torq.append(st.torq)
            else:
                pos.append(0.0)
                vel.append(0.0)
                torq.append(0.0)
        return (
            np.array(pos, dtype=np.float64),
            np.array(vel, dtype=np.float64),
            np.array(torq, dtype=np.float64),
        )

    def get_positions(self) -> np.ndarray:
        return self.get_state()[0]

    def get_velocities(self) -> np.ndarray:
        return self.get_state()[1]

    def get_torques(self) -> np.ndarray:
        return self.get_state()[2]

    # ── 生命周期 ────────────────────────────────────────────────────────

    def disconnect(self, *, disable_motors: bool = True) -> None:
        if not self._connected:
            return
        self.stop_control_loop()
        self._close_position_readers()
        if disable_motors:
            self.disable_all()
            time.sleep(0.5)
        for motor in self._motor_map.values():
            motor.close()
        for ctrl in self._ctrl_map.values():
            try:
                ctrl.close_bus()
            finally:
                ctrl.close()
        self._ctrl_map.clear()
        self._motor_map.clear()
        self._connected = False

    def estop(self) -> None:
        self.stop_control_loop()
        self.disable_all()

    def reconnect(
        self,
        init_delay: float = 1.0,
        post_setup_delay: float = 0.5,
    ) -> None:
        self.disconnect()
        time.sleep(init_delay)
        self.connect()
        time.sleep(post_setup_delay)
        print("[reconnect] 控制器和电机已重新初始化")

    # ── 控制循环 ────────────────────────────────────────────────────────

    def start_control_loop(
        self,
        control_fn: Callable[["RebotArm", float], None],
        rate: Optional[float] = None,
    ) -> None:
        if self.control_loop_active:
            raise RuntimeError("控制循环已在运行，请先调用 stop_control_loop()")
        self._ctrl_rate = rate if rate is not None else self._rate
        if not np.isfinite(self._ctrl_rate) or self._ctrl_rate <= 0:
            raise ValueError("Control rate must be finite and positive")
        self.validate_control_transport(self._ctrl_rate)
        self._running = True
        self._control_stop.clear()
        with self._timing_lock:
            self._dt_histogram.fill(0)
            self._dt_count = 0
            self.control_loop_stats = {"iterations": 0, "overruns": 0, "last_dt": 0.0,
                                       "max_dt": 0.0, "last_error": None}
        self._ctrl_fn = control_fn
        self._ctrl_thread = threading.Thread(
            target=self._control_loop_impl,
            name="rebotarm-control-loop",
            daemon=True,
        )
        self._ctrl_thread.start()

    def _control_loop_impl(self) -> None:
        period = 1.0 / self._ctrl_rate
        previous = None
        deadline = time.perf_counter()
        began = deadline
        with self._timing_lock:
            self._dt_histogram.fill(0)
            self._dt_count = 0
            self.control_loop_stats.update(target_rate_hz=self._ctrl_rate, achieved_rate_hz=0.,
                elapsed_s=0., max_callback_s=0., intervals_over_2_periods=0, missed_deadlines=0)
        while self._running:
            t0 = time.perf_counter()
            dt = period if previous is None else t0 - previous
            previous = t0
            with self._timing_lock:
                self.control_loop_stats["iterations"] += 1
                self.control_loop_stats["last_dt"] = dt
                self.control_loop_stats["max_dt"] = max(dt, self.control_loop_stats["max_dt"])
                self.control_loop_stats["intervals_over_2_periods"] += int(dt > 2 * period)
                self.control_loop_stats["elapsed_s"] = t0 - began
                if t0 > began:
                    self.control_loop_stats["achieved_rate_hz"] = (self.control_loop_stats["iterations"] - 1) / (t0 - began)
                # Fixed memory, all cycles: 10 us bins through 100 ms, then an
                # overflow bin. Quantiles are evaluated only on status/export.
                self._dt_histogram[min(10000, max(0, int(dt / 1e-5)))] += 1
                self._dt_count += 1
            try:
                self._ctrl_fn(self, dt)
            except Exception as error:
                self.control_loop_stats["last_error"] = repr(error)
                self._running = False
                return
            deadline += period
            now = time.perf_counter()
            with self._timing_lock:
                self.control_loop_stats["max_callback_s"] = max(now - t0, self.control_loop_stats["max_callback_s"])
                if now > deadline:
                    self.control_loop_stats["overruns"] += 1
                    missed = int(np.floor((now - deadline) / period)) + 1
                    self.control_loop_stats["missed_deadlines"] += missed
                    # Advance on the original clock grid. Skip expired slots
                    # instead of accumulating drift or replaying commands.
                    deadline += missed * period
            self._control_stop.wait(max(0.0, deadline - time.perf_counter()))

    def control_loop_snapshot(self):
        with self._timing_lock:
            result = dict(self.control_loop_stats)
            histogram, count = self._dt_histogram.copy(), self._dt_count
        if count:
            cumulative = np.cumsum(histogram)
            quantiles = {}
            for label, percentile in (("p50", .5), ("p95", .95), ("p99", .99), ("p999", .999)):
                index = int(np.searchsorted(cumulative, np.ceil(percentile * count)))
                quantiles[label] = (min((index + 1) * 1e-5, result["max_dt"])
                                    if index < 10000 else result["max_dt"])
            result["control_dt_s"] = dict(quantiles, max=result["max_dt"])
            result["timing_samples"] = count
            result["timing_bin_width_s"] = 1e-5
        return result

    def stop_control_loop(self) -> None:
        self._running = False
        self._control_stop.set()
        t = getattr(self, "_ctrl_thread", None)
        if t is not None and t.is_alive() and t is not threading.current_thread():
            t.join(timeout=5.0)
            if t.is_alive():
                raise RuntimeError("Control loop did not stop; CAN handles remain open")

    # ── 上下文管理器 ───────────────────────────────────────────────────────

    def __enter__(self) -> "RebotArm":
        return self

    def __exit__(self, *args) -> None:
        self.disconnect()

    def __repr__(self) -> str:
        gs = ", ".join(f"{k}({g.num_joints}j)" for k, g in self._groups.items())
        return f"RebotArm({self._name!r}, [{gs}], rate={self._ctrl_rate}Hz)"
