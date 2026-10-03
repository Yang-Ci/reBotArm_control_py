"""Check real Python bindings and physical identities, without connecting motors."""
from pathlib import Path

import numpy as np
import pinocchio as pin
import pytest

from reBotArm_control_py import dynamics as dyn
from reBotArm_control_py.kinematics import (
    compute_fk, get_joint_limits, load_robot_model, pad_q_for_model,
)
from reBotArm_control_py.kinematics.inverse_kinematics import (
    IKParams, _compute_error, solve_ik,
)

pytestmark = pytest.mark.native_pinocchio
ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(params=["RS", "DM"])
def model(request):
    vendor = request.param
    return load_robot_model(str(ROOT / f"urdf/{vendor}/urdf/ReBot_Arm_{vendor}.urdf"))


def state(model):
    q = pin.neutral(model)
    direction = 1 if model.name.endswith("RS") else -1
    q[:6] = [0.1, direction * 0.7, direction * 1.1, 0.2, -0.2, 0.1]
    v = np.linspace(-0.15, 0.2, model.nv)
    a = np.linspace(0.1, -0.2, model.nv)
    return q, v, a


def tangent_derivative(model, q, fn, h=1e-6):
    columns = []
    for j in range(model.nv):
        step = np.zeros(model.nv)
        step[j] = h
        columns.append((fn(pin.integrate(model, q, step))
                        - fn(pin.integrate(model, q, -step))) / (2 * h))
    return np.stack(columns, axis=-1)


def test_joint_limits_use_configuration_indices(model):
    expected = [(model.lowerPositionLimit[j.idx_q], model.upperPositionLimit[j.idx_q])
                for j in model.joints[1:]]
    np.testing.assert_allclose(get_joint_limits(model), expected)


def test_gravity_roundtrip_and_zero_gravity(model):
    dyn.set_gravity(model, np.array([0., 0., -9.81]))
    g = dyn.get_gravity(model)
    np.testing.assert_array_equal(g, [0., 0., -9.81])
    g[:] = 0
    np.testing.assert_array_equal(dyn.get_gravity(model), [0., 0., -9.81])
    dyn.set_gravity(model, (0., 0., 0.))
    np.testing.assert_allclose(dyn.compute_gravity_vector(model, state(model)[0]), 0., atol=1e-12)
    with pytest.raises(ValueError):
        dyn.set_gravity(model, [0., np.nan, 0.])


def test_all_terms_match_independent_algorithms_and_rnea(model):
    q, v, a = state(model)
    data = model.createData()
    data.C[:] = 12345.  # A reused Data must not leak a previous Coriolis matrix.
    M, C, g = dyn.compute_all_terms(model, q, v, data)
    np.testing.assert_allclose(M, M.T, atol=1e-12)
    # The DM URDF has two explicitly massless finger DOFs. RNEA/gravity still
    # make sense, but forward dynamics of that full model is underdetermined.
    assert np.linalg.eigvalsh(M).min() >= -1e-12
    assert np.linalg.eigvalsh(M[:6, :6]).min() > 0
    np.testing.assert_allclose(M, dyn.compute_mass_matrix(model, q), atol=1e-12)
    np.testing.assert_allclose(C, dyn.compute_coriolis_matrix(model, q, v), atol=1e-12)
    np.testing.assert_allclose(C @ v + g, dyn.compute_nle(model, q, v), atol=1e-10)
    tau = dyn.compute_inverse_dynamics(model, q, v, a)
    np.testing.assert_allclose(M @ a + C @ v + g, tau, atol=1e-10)
    if np.linalg.eigvalsh(M).min() > 0:
        np.testing.assert_allclose(dyn.compute_forward_dynamics(model, q, v, tau), a, atol=1e-9)
        np.testing.assert_allclose(dyn.forward_dynamics_from_nle(model, q, v, tau), a, atol=1e-9)
    else:
        for fn in (dyn.compute_forward_dynamics, dyn.forward_dynamics_from_nle):
            with pytest.raises(ValueError, match="singular"):
                fn(model, q, v, tau)
    # Public results remain snapshots when the caller reuses Data.
    C_saved = C.copy()
    dyn.compute_all_terms(model, q, -v, data)
    np.testing.assert_array_equal(C, C_saved)


def test_mass_and_rnea_derivatives_against_finite_differences(model):
    q, v, a = state(model)
    dM = dyn.compute_mass_matrix_derivatives(model, q)
    numeric = tangent_derivative(model, q, lambda x: dyn.compute_mass_matrix(model, x))
    assert dM.shape == (model.nv, model.nv, model.nv)
    np.testing.assert_allclose(dM, np.moveaxis(numeric, -1, 0), atol=2e-8, rtol=1e-6)
    dq, dv, da = dyn.compute_rnea_derivatives(model, q, v, a)
    numeric_q = tangent_derivative(model, q, lambda x: dyn.compute_inverse_dynamics(model, x, v, a))
    np.testing.assert_allclose(dq, numeric_q, atol=2e-7, rtol=1e-6)
    numeric_v = np.column_stack([
        (dyn.compute_inverse_dynamics(model, q, v + 1e-6 * e, a)
         - dyn.compute_inverse_dynamics(model, q, v - 1e-6 * e, a)) / 2e-6
        for e in np.eye(model.nv)])
    np.testing.assert_allclose(dv, numeric_v, atol=2e-7, rtol=1e-6)
    np.testing.assert_allclose(da, dyn.compute_mass_matrix(model, q), atol=1e-12)
    numeric_g = tangent_derivative(model, q, lambda x: dyn.compute_gravity_vector(model, x))
    np.testing.assert_allclose(dyn.compute_generalized_gravity_derivatives(model, q),
                               numeric_g, atol=2e-7, rtol=1e-6)
    nle_q, nle_v = dyn.compute_coriolis_derivatives(model, q, v)
    np.testing.assert_allclose(nle_q, tangent_derivative(model, q, lambda x: dyn.compute_nle(model, x, v)),
                               atol=2e-7, rtol=1e-6)
    np.testing.assert_allclose(nle_v, dyn.compute_rnea_derivatives(model, q, v, np.zeros(model.nv))[1])


def test_com_velocity_momentum_and_energy_identities(model):
    q, v, _ = state(model)
    h = 1e-6
    numeric = (dyn.compute_center_of_mass(model, pin.integrate(model, q, v * h))
               - dyn.compute_center_of_mass(model, pin.integrate(model, q, -v * h))) / (2 * h)
    com_vel = dyn.compute_com_velocity(model, q, v)
    np.testing.assert_allclose(com_vel, numeric, atol=1e-8)
    momentum = dyn.compute_centroidal_momentum(model, q, v)
    np.testing.assert_allclose(momentum, dyn.compute_centroidal_matrix(model, q, v) @ v, atol=1e-12)
    np.testing.assert_allclose(momentum[:3], pin.computeTotalMass(model) * com_vel, atol=1e-12)
    np.testing.assert_allclose(dyn.compute_kinetic_energy(model, q, v),
                               .5 * v @ dyn.compute_mass_matrix(model, q) @ v, atol=1e-12)
    potential_gradient = tangent_derivative(model, q, lambda x: dyn.compute_potential_energy(model, x))
    np.testing.assert_allclose(potential_gradient, dyn.compute_gravity_vector(model, q), atol=2e-7)


def test_se3_error_jacobian_matches_native_finite_difference(model):
    q, _, _ = state(model)
    frame = model.getFrameId("gripper_end" if model.name.endswith("RS") else "end_link")
    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    target = data.oMf[frame].copy() * pin.exp6(np.array([.03, -.01, .02, .1, -.05, .08]))
    _compute_error(model, data, frame, q, target)
    pin.computeJointJacobians(model, data, q)
    J = pin.Jlog6((data.oMf[frame].inverse() * target).inverse()) @ pin.getFrameJacobian(model, data, frame, pin.LOCAL)
    numeric = tangent_derivative(model, q, lambda x: _compute_error(model, model.createData(), frame, x, target)[1])
    np.testing.assert_allclose(-J, numeric, atol=1e-8)


def test_position_only_world_error_and_ik(model):
    q, _, _ = state(model)
    frame_name = "gripper_end" if model.name.endswith("RS") else "end_link"
    frame = model.getFrameId(frame_name)
    data = model.createData()
    pin.framesForwardKinematics(model, data, q)
    target = data.oMf[frame].copy()
    target.translation += [.015, -.01, .01]
    _compute_error(model, data, frame, q, target, True)
    pin.computeJointJacobians(model, data, q)
    J = data.oMf[frame].rotation @ pin.getFrameJacobian(model, data, frame, pin.LOCAL)[:3]
    numeric = tangent_derivative(model, q, lambda x: _compute_error(model, model.createData(), frame, x, target, True)[1])
    np.testing.assert_allclose(-J, numeric, atol=1e-8)
    result = solve_ik(model, data, frame, target, q, IKParams(tolerance=1e-9),
                      controlled_joints=6, position_only=True)
    assert result.success
    actual = compute_fk(model, pad_q_for_model(model, result.q), frame_name)[0]
    np.testing.assert_allclose(actual, target.translation, atol=1e-9)


def test_missing_frame_and_invalid_vector_rejected(model):
    with pytest.raises(ValueError, match="frame|Frame"):
        compute_fk(model, pin.neutral(model), "does_not_exist")
    for invalid in (np.zeros(model.nq + 1), np.zeros((model.nq, 1)), [np.nan]):
        with pytest.raises(ValueError):
            pad_q_for_model(model, invalid)


def test_freeflyer_padding_keeps_valid_neutral_quaternion():
    model = pin.buildSampleModelHumanoidRandom()
    q = pad_q_for_model(model, np.array([.1, .2, .3]))
    assert pin.isNormalized(model, q)
    np.testing.assert_array_equal(q[3:7], [0., 0., 0., 1.])


def test_forward_dynamics_with_passive_fingers_locked(model):
    locked = [i for i in range(1, model.njoints) if model.joints[i].idx_q >= 6]
    reduced = pin.buildReducedModel(model, locked, pin.neutral(model))
    q, v, a = state(reduced)
    tau = dyn.compute_inverse_dynamics(reduced, q, v, a)
    np.testing.assert_allclose(dyn.compute_forward_dynamics(reduced, q, v, tau), a, atol=1e-9)
    np.testing.assert_allclose(dyn.forward_dynamics_from_nle(reduced, q, v, tau), a, atol=1e-9)


def test_derivatives_clear_stale_entries_and_return_snapshots(model):
    q, v, a = state(model)
    data = model.createData()
    data.dtau_dq[:] = data.dtau_dv[:] = data.M[:] = 12345.
    actual = dyn.compute_rnea_derivatives(model, q, v, a, data)
    expected = dyn.compute_rnea_derivatives(model, q, v, a)
    for first, second in zip(actual, expected):
        np.testing.assert_allclose(first, second, atol=1e-12)
    dyn.compute_rnea_derivatives(model, q, -v, -a, data)
    for first, second in zip(actual, expected):
        np.testing.assert_array_equal(first, second)


def test_model_loading_is_explicit_and_mutable_models_are_independent(tmp_path):
    from reBotArm_control_py.kinematics import get_end_effector_frame_id
    a = dyn.load_dynamics_model(hardware_config_path=str(ROOT / "config/rebotarm_rs.yaml"))
    b = dyn.load_dynamics_model(hardware_config_path=str(ROOT / "config/rebotarm_dm.yaml"))
    assert a.name.endswith("RS") and b.name.endswith("DM")
    assert get_end_effector_frame_id(b, str(ROOT / "config/rebotarm_dm.yaml")) < b.nframes
    dyn.set_gravity(a, (0., 0., 0.))
    c = dyn.load_dynamics_model(hardware_config_path=str(ROOT / "config/rebotarm_rs.yaml"))
    np.testing.assert_allclose(dyn.get_gravity(c), [0., 0., -9.81])
    bad = tmp_path / "bad.yaml"
    bad.write_text('end_effector_frame: does_not_exist', encoding="utf-8")
    with pytest.raises(ValueError, match="frame"):
        get_end_effector_frame_id(c, str(bad))


def test_freeflyer_derivatives_use_tangent_dimension():
    model = pin.buildSampleModelHumanoidRandom()
    q = pin.neutral(model)
    derivatives = dyn.compute_rnea_derivatives(model, q)
    assert model.nq != model.nv
    assert all(value.shape == (model.nv, model.nv) for value in derivatives)
    dM = dyn.compute_mass_matrix_derivatives(model, q)
    assert dM.shape == (model.nv, model.nv, model.nv)
    e = np.zeros(model.nv)
    e[4] = 1e-6
    numeric = (dyn.compute_mass_matrix(model, pin.integrate(model, q, e))
               - dyn.compute_mass_matrix(model, pin.integrate(model, q, -e))) / 2e-6
    np.testing.assert_allclose(dM[4], numeric, atol=2e-7, rtol=1e-6)
