"""Native Pinocchio by default; the limited test double requires an explicit flag."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def pytest_addoption(parser):
    parser.addoption("--offline-kinematics", action="store_true",
                     help="Use the limited NumPy URDF test double, not native Pinocchio")


def pytest_configure(config):
    config.addinivalue_line("markers", "native_pinocchio: requires real Pinocchio bindings")
    if config.getoption("--offline-kinematics"):
        from offline_pinocchio import make_module
        sys.modules["pinocchio"] = make_module()
    else:
        try:
            import pinocchio
        except ImportError as exc:
            raise pytest.UsageError(
                "Native Pinocchio is required. Install it or explicitly pass "
                "--offline-kinematics for limited kinematic tests."
            ) from exc


def pytest_collection_modifyitems(config, items):
    if config.getoption("--offline-kinematics"):
        skip = pytest.mark.skip(reason="Native Pinocchio is not used with --offline-kinematics")
        for item in items:
            if "native_pinocchio" in item.keywords:
                item.add_marker(skip)


def pytest_report_header(config):
    import pinocchio
    return "kinematics backend: " + getattr(pinocchio, "__version__", "unknown")
