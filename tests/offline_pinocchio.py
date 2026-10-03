"""NumPy URDF kinematics test double; never used by production code."""
import types
import xml.etree.ElementTree as ET

import numpy as np


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def exp_rotation(v):
    theta = np.linalg.norm(v)
    W = skew(v)
    if theta < 1e-8:
        return np.eye(3) + W + 0.5 * W @ W
    return np.eye(3) + np.sin(theta) / theta * W + (1 - np.cos(theta)) / theta ** 2 * W @ W


def rpy_matrix(roll, pitch, yaw):
    return exp_rotation([0., 0., yaw]) @ exp_rotation([0., pitch, 0.]) @ exp_rotation([roll, 0., 0.])


class SE3:
    def __init__(self, rotation, translation=None):
        if translation is None:
            T = np.asarray(rotation)
            self.rotation, self.translation = T[:3, :3].copy(), T[:3, 3].copy()
        else:
            self.rotation = np.asarray(rotation).copy()
            self.translation = np.asarray(translation).copy()

    @property
    def homogeneous(self):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = self.rotation, self.translation
        return T

    def inverse(self):
        return SE3(self.rotation.T, -self.rotation.T @ self.translation)

    def __mul__(self, other):
        return SE3(self.rotation @ other.rotation, self.translation + self.rotation @ other.translation)

    def copy(self):
        return SE3(self.rotation, self.translation)


class Motion:
    def __init__(self, vector):
        self.vector = np.asarray(vector)

    def __mul__(self, scalar):
        return Motion(self.vector * scalar)


def exp6(motion):
    v = motion.vector if isinstance(motion, Motion) else np.asarray(motion)
    W = skew(v[3:])
    theta = np.linalg.norm(v[3:])
    if theta < 1e-7:
        V = np.eye(3) + 0.5 * W + W @ W / 6
    else:
        V = (np.eye(3) + (1 - np.cos(theta)) / theta ** 2 * W
             + (theta - np.sin(theta)) / theta ** 3 * W @ W)
    return SE3(exp_rotation(v[3:]), V @ v[:3])


def log6(pose):
    R = pose.rotation
    theta = np.arccos(np.clip((np.trace(R) - 1) / 2, -1., 1.))
    vee = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    w = 0.5 * vee if theta < 1e-7 else theta / (2 * np.sin(theta)) * vee
    W = skew(w)
    theta = np.linalg.norm(w)
    if theta < 1e-7:
        Vinv = np.eye(3) - 0.5 * W + W @ W / 12
    else:
        Vinv = np.eye(3) - 0.5 * W + (1 - theta / (2 * np.tan(theta / 2))) / theta ** 2 * W @ W
    return Motion(np.r_[Vinv @ pose.translation, w])


def jlog6(pose):
    h = 1e-6
    columns = []
    for i in range(6):
        v = np.zeros(6)
        v[i] = h
        columns.append((log6(pose * exp6(v)).vector - log6(pose * exp6(-v)).vector) / (2 * h))
    return np.column_stack(columns)


def xyz(element, key, default="0 0 0"):
    return np.fromstring(element.get(key, default), sep=" ")


class Model:
    def __init__(self, path):
        self.xml = ET.parse(path).getroot()
        self.nodes = self.xml.findall("joint")
        moving = [j for j in self.nodes if j.get("type") != "fixed"]
        self.nv = self.nq = len(moving)
        self.indices = {j.get("name"): i for i, j in enumerate(moving)}
        self.names = ["universe", *[j.get("name") for j in moving]]
        self.joints = [types.SimpleNamespace(idx_q=-1, idx_v=-1, nq=0, nv=0)]
        self.joints += [types.SimpleNamespace(idx_q=i, idx_v=i, nq=1, nv=1) for i in range(self.nq)]
        self.lowerPositionLimit = np.array([float(j.find("limit").get("lower")) for j in moving])
        self.upperPositionLimit = np.array([float(j.find("limit").get("upper")) for j in moving])
        self.velocityLimit = np.array([float(j.find("limit").get("velocity")) for j in moving])
        self.effortLimit = np.array([float(j.find("limit").get("effort")) for j in moving])
        self.frame_names = [e.get("name") for e in self.xml.findall("link")]
        self.nframes = len(self.frame_names)

    def createData(self):
        return types.SimpleNamespace(oMf=[SE3(np.eye(3), np.zeros(3)) for _ in self.frame_names],
                                     transforms={}, axes={}, origins={})

    def getFrameId(self, name):
        return self.frame_names.index(name) if name in self.frame_names else self.nframes


def fk(model, data, q):
    children = {j.find("child").get("link") for j in model.nodes}
    root = next(n for n in model.frame_names if n not in children)
    transforms = {root: SE3(np.eye(3), np.zeros(3))}
    pending = list(model.nodes)
    while pending:
        available = [j for j in pending if j.find("parent").get("link") in transforms]
        if not available:
            raise ValueError("Invalid URDF tree")
        for j in available:
            parent, child = j.find("parent").get("link"), j.find("child").get("link")
            o = j.find("origin")
            T = transforms[parent] * SE3(rpy_matrix(*xyz(o, "rpy")), xyz(o, "xyz"))
            if j.get("type") != "fixed":
                index = model.indices[j.get("name")]
                axis = xyz(j.find("axis"), "xyz")
                data.axes[index], data.origins[index] = T.rotation @ axis, T.translation.copy()
                transform = (SE3(exp_rotation(axis * q[index]), np.zeros(3)) if j.get("type") == "revolute"
                             else SE3(np.eye(3), axis * q[index]))
                T = T * transform
            transforms[child] = T
            pending.remove(j)
    data.transforms = transforms
    data.oMf = [transforms[n] for n in model.frame_names]


def jacobian(model, data, frame_id, reference):
    pose = data.oMf[frame_id]
    child_map = {j.find("child").get("link"): j for j in model.nodes}
    ancestors = []
    link = model.frame_names[frame_id]
    while link in child_map:
        joint = child_map[link]
        if joint.get("type") != "fixed":
            ancestors.append(joint)
        link = joint.find("parent").get("link")
    J = np.zeros((6, model.nv))
    for j in ancestors:
        i = model.indices[j.get("name")]
        axis, origin = data.axes[i], data.origins[i]
        if j.get("type") == "revolute":
            J[:3, i] = pose.rotation.T @ np.cross(axis, pose.translation - origin)
            J[3:, i] = pose.rotation.T @ axis
        else:
            J[:3, i] = pose.rotation.T @ axis
    return J


def matrix_rpy(R):
    return np.array([np.arctan2(R[2, 1], R[2, 2]), np.arcsin(np.clip(-R[2, 0], -1, 1)),
                     np.arctan2(R[1, 0], R[0, 0])])


def make_module():
    module = types.ModuleType("pinocchio")
    module.__version__ = "NumPy URDF test double (NOT native Pinocchio)"
    module.Model, module.SE3, module.Motion = Model, SE3, Motion
    module.log6, module.exp6, module.Jlog6 = log6, exp6, jlog6
    module.rpy = types.SimpleNamespace(rpyToMatrix=rpy_matrix, matrixToRpy=matrix_rpy)
    module.ReferenceFrame = types.SimpleNamespace(LOCAL=0)
    module.LOCAL = 0
    module.forwardKinematics = module.computeJointJacobians = fk
    module.updateFramePlacements = lambda model, data: None
    module.getFrameJacobian = jacobian
    module.integrate = lambda model, q, dq: q + dq
    module.buildModelFromUrdf = Model
    module.neutral = lambda model: np.zeros(model.nq)
    return module
