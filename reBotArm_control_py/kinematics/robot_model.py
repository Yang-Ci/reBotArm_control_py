"""reBot-DevArm 机器人模型加载模块 — 基于 Pinocchio。

urdf_path 和 end_effector_frame 从 hardware_yaml 指向的硬件配置文件中读取，
rebotarm.yaml 只提供 hardware_yaml 字段。
"""

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pinocchio as pin
import yaml

_cfg_dir = Path(__file__).resolve().parents[2] / "config"
_global_cfg = _cfg_dir / "rebotarm.yaml"
_project_root = _cfg_dir.parent

def _hw_config(hardware_config_path: str | None = None) -> dict:
    """Load kinematics fields (urdf_path, end_effector_frame) from the hardware YAML."""
    hw_yaml = hardware_config_path or ""
    if not hw_yaml and _global_cfg.exists():
        global_data = yaml.safe_load(_global_cfg.read_text(encoding="utf-8")) or {}
        hw_yaml = global_data.get("hardware_yaml", hw_yaml)

    hw_path = Path(hw_yaml)
    if not hw_path.is_absolute():
        hw_path = _cfg_dir / hw_path
    if not hw_yaml or not hw_path.is_file():
        raise FileNotFoundError(f"Hardware config not found: {hw_path}")

    config = yaml.safe_load(hw_path.read_text(encoding="utf-8")) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Hardware config must be a mapping: {hw_path}")
    return config


def _resolve_urdf(urdf_path: str | None = None, *,
                  hardware_config_path: str | None = None) -> Tuple[str, List[str]]:
    if urdf_path is None:
        urdf_path = _hw_config(hardware_config_path).get("urdf_path", "")

    if not urdf_path:
        raise ValueError("urdf_path is empty. Set it in the hardware config file.")

    if not Path(urdf_path).is_absolute():
        urdf_path = str(_project_root / urdf_path)

    # URDFs in this repo use two mesh-path conventions:
    #   "../meshes/..." resolved from the URDF's own directory, and
    #   "meshes/..."    resolved from the package root (the URDF dir's parent).
    # Hand Pinocchio both so either convention resolves without per-URDF tweaks.
    urdf_dir = Path(urdf_path).resolve().parent
    package_dirs = [str(urdf_dir), str(urdf_dir.parent)]
    return urdf_path, package_dirs


def load_robot_model(urdf_path: str | None = None, *,
                     hardware_config_path: str | None = None) -> pin.Model:
    path, _ = _resolve_urdf(urdf_path, hardware_config_path=hardware_config_path)
    return pin.buildModelFromUrdf(path)


def get_end_effector_frame(hardware_config_path: str | None = None) -> str:
    return _hw_config(hardware_config_path).get("end_effector_frame", "gripper_end")


def get_joint_count() -> int:
    model = load_robot_model()
    return model.nq


def get_joint_names(model: pin.Model) -> List[str]:
    return [n for n, j in zip(model.names[1:], model.joints[1:]) if j.idx_q >= 0]


def get_joint_limits(model: pin.Model) -> List[Tuple[float, float]]:
    limits = []
    for name in get_joint_names(model):
        jid = model.getJointId(name)
        joint = model.joints[jid]
        if joint.nq != 1:
            raise ValueError(f"Joint {name} is not a scalar configuration joint")
        iq = joint.idx_q
        lo, hi = float(model.lowerPositionLimit[iq]), float(model.upperPositionLimit[iq])
        limits.append((-np.inf, np.inf) if np.isinf(lo) and np.isinf(hi) else (lo, hi))
    return limits


def get_end_effector_frame_id(model: pin.Model, hardware_config_path: str | None = None) -> int:
    name = get_end_effector_frame(hardware_config_path)
    frame_id = model.getFrameId(name)
    if frame_id >= model.nframes:
        raise ValueError(f"End-effector frame {name!r} not found in model")
    return frame_id


def get_all_frame_names(model: pin.Model) -> List[str]:
    return [f.name for f in model.frames]


def pad_q_for_model(model: pin.Model, q: np.ndarray, controlled_joints: int | None = None) -> np.ndarray:
    nq = model.nq
    n_ctrl = controlled_joints if controlled_joints is not None else nq
    if not isinstance(n_ctrl, (int, np.integer)) or not 0 < n_ctrl <= nq:
        raise ValueError("controlled_joints must be an integer in [1, model.nq]")
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or q.size > nq or not np.all(np.isfinite(q)):
        raise ValueError(f"q must be a finite vector of at most {nq} entries")
    padded = np.asarray(pin.neutral(model), dtype=float).copy()
    padded[:min(q.shape[0], n_ctrl)] = q[:min(q.shape[0], n_ctrl)]
    if hasattr(pin, "isNormalized") and not pin.isNormalized(model, padded):
        raise ValueError("q contains a non-normalized joint configuration")
    return padded
