"""Build the reviewed 0.5.6 ABI with RS/DM feedback and send isolation.

The installed wheel stays intact. RebotArm selects the local library on the next
Python process. Requires Rust and the platform C linker; does not open CAN.
"""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
URL = "https://codeload.github.com/motorbridge/motorbridge/zip/refs/tags/v0.5.6"
SHA256 = "c2d50a723bcc2729fbfb0835eb939f93cc3cda5bb4d68a7610b632590d8fec21"
MARKER = "rebotarm_rs_feedback_lock_fix_v1"


def patch_source(source: Path):
    version = (source / "Cargo.toml").read_text(encoding="utf-8")
    if 'version = "0.5.6"' not in version:
        raise RuntimeError("Patch requires the reviewed motorbridge v0.5.6 source")
    # The 0.5.6 bundled DM USB shim/header do not compile together here.
    # PCAN and DM serial-to-CAN do not need that separate vendor USB adapter.
    build_path = source / "motor_core/build.rs"
    build_text = build_path.read_text(encoding="utf-8")
    if "REBOTARM_RS_ONLY_ABI" not in build_text:
        anchor = '    println!("cargo:rustc-check-cfg=cfg(motorbridge_dm_device_supported)");'
        if build_text.count(anchor) != 1:
            raise RuntimeError("Upstream build script changed")
        build_text = build_text.replace(anchor, anchor + '''
    println!("cargo:rerun-if-env-changed=REBOTARM_RS_ONLY_ABI");
    if std::env::var_os("REBOTARM_RS_ONLY_ABI").is_some() { return; }
''')
        build_path.write_text(build_text, encoding="utf-8")
    dm_path = source / "motor_core/src/dm_device.rs"
    dm_text = dm_path.read_text(encoding="utf-8")
    anchor = "\nimpl ErrorBuf {"
    if "#[cfg(motorbridge_dm_device_supported)]\nimpl ErrorBuf {" not in dm_text:
        if dm_text.count(anchor) != 1:
            raise RuntimeError("Upstream DM conditional compilation changed")
        dm_path.write_text(dm_text.replace(anchor, "\n#[cfg(motorbridge_dm_device_supported)]\nimpl ErrorBuf {"), encoding="utf-8")
    path = source / "motor_abi/src/motor_register_ffi.rs"
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        test_start = text.index("\n#[cfg(test)]\nmod rebotarm_feedback_lock_tests")
        test = (Path(__file__).parent / "patches/rs_feedback_lock_test.rs").read_text(encoding="utf-8")
        path.write_text(text[:test_start] + test, encoding="utf-8")
        from tools.patches.dm_native_patch import patch_dm
        patch_dm(source)
        return
    start = text.index('pub extern "C" fn motor_handle_robstride_get_param_f32_host_id(')
    stop = text.index('\n#[unsafe(no_mangle)]', start)
    old = text[start:stop]
    expected = '    let motor = lock_motor_inner!(motor, "motor is null");'
    if old.count(expected) != 1 or '    let rc = match &*motor {' not in old:
        raise RuntimeError("Upstream getter changed; refusing to guess a native patch")
    new = old.replace(expected, '''    // Keep the Arc alive while allowing sends on the same MotorHandle.
    // Close/free must still happen after all reader threads have joined.
    let owned = {
        let guard = lock_motor_inner!(motor, "motor is null");
        match &*guard {
            MotorHandleInner::Robstride(m) => MotorHandleInner::Robstride(Arc::clone(m)),
            _ => {
                set_last_error("robstride_get_param_f32_host_id requires a RobStride motor");
                return -1;
            }
        }
    };''').replace('    let rc = match &*motor {', '    let rc = match &owned {')
    marker = f'''\n#[unsafe(no_mangle)]
pub extern "C" fn {MARKER}() -> i32 {{ 1 }}
'''
    test = (Path(__file__).parent / "patches/rs_feedback_lock_test.rs").read_text(encoding="utf-8")
    path.write_text(text[:start] + new + marker + text[stop:] + test, encoding="utf-8")
    from tools.patches.dm_native_patch import patch_dm
    patch_dm(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Existing official v0.5.6 checkout (patched in place)")
    args = parser.parse_args()
    output = ROOT / ".motorbridge" / sys.platform
    output.mkdir(parents=True, exist_ok=True)
    source = args.source
    if source is None:
        archive = output / "v0.5.6.zip"
        if not archive.exists():
            urllib.request.urlretrieve(URL, archive)
        if hashlib.sha256(archive.read_bytes()).hexdigest() != SHA256:
            raise RuntimeError("Official source archive SHA256 mismatch")
        source = output / "motorbridge-0.5.6"
        if not source.exists():
            with zipfile.ZipFile(archive) as bundle:
                # Reject traversal even though this is a verified source archive.
                base = output.resolve()
                for entry in bundle.infolist():
                    (base / entry.filename).resolve().relative_to(base)
                bundle.extractall(output)
    source = source.resolve()
    patch_source(source)
    build_env = dict(os.environ, REBOTARM_RS_ONLY_ABI="1")
    subprocess.run(["cargo", "test", "--locked", "-p", "motor_abi", "-p", "motor_core", "rebotarm_"], cwd=source, env=build_env, check=True)
    subprocess.run(["cargo", "build", "--locked", "-p", "motor_abi", "--release"], cwd=source, env=build_env, check=True)
    name = {"win32": "motor_abi.dll", "darwin": "libmotor_abi.dylib"}.get(sys.platform, "libmotor_abi.so")
    result = output / name
    shutil.copy2(source / "target/release" / name, result)
    print(f"Built {result}; restart Python before starting RS/DM control.")


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    main()
