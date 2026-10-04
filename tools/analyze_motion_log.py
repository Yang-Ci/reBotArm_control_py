"""Analyze exported motion CSV without opening hardware or importing Pinocchio."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def percentiles(values):
    return dict(zip(("p50", "p95", "p99", "max"), map(float,
                    [*np.percentile(values, [50, 95, 99]), np.max(values)])))


def analyze(path, expected_dz=None, z_tolerance=.002, reverse_tolerance=.0005, joint_only=False):
    with Path(path).open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Empty motion log")
    def column(name):
        return np.array([float(row[name]) for row in rows])
    required = ["control_dt", "feedback_age", "z_ref", "z_actual", "send_errors",
                "feedback_sequence", "feedback_read_span", "phase", "fault"]
    if any(name not in rows[0] for name in required):
        raise ValueError("Log must come from the updated controller (sequence/phase required)")
    numeric = [name for name in rows[0] if name not in ("phase", "fault")]
    if any(not np.all(np.isfinite(column(name))) for name in numeric):
        raise ValueError("Log contains non-finite telemetry")
    dt, age, span = column("control_dt"), column("feedback_age"), column("feedback_read_span")
    active = np.array([row["phase"] in ("tracking", "settling", "complete") for row in rows])
    indices = np.flatnonzero(active)
    movement = len(indices) > 0
    first = indices[0] if movement else 0
    ref = column("z_ref")[first:]
    actual = column("z_actual")[first:]
    seq = column("feedback_sequence")[first:]
    # Logging is faster than feedback. Never count repeated snapshots as new samples.
    unique = np.r_[True, np.diff(seq) != 0]
    measured = actual[unique]
    dz = float(ref[-1] - ref[0])
    direction = np.sign(expected_dz if expected_dz is not None else dz)
    projected = direction * measured
    max_reverse = float(np.max(np.maximum.accumulate(projected) - projected)) if len(projected) else 0.
    joints = [name[len("q_ref_"):] for name in rows[0] if name.startswith("q_ref_")]
    errors = {name: column("q_ref_" + name) - column("q_actual_" + name) for name in joints}
    faults = sorted({row["fault"] for row in rows if row["fault"]})
    status_path = Path(path).with_suffix(".status.json")
    status = json.loads(status_path.read_text(encoding="utf-8")) if status_path.exists() else None
    configured = status.get("limits", {}) if status else {}
    max_dt = float(configured.get("max_control_dt", .05))
    max_age = float(configured.get("max_feedback_age", .25))
    max_span = float(configured.get("max_feedback_span", .02))
    if not np.all(np.isfinite([max_dt, max_age, max_span])) or min(max_dt, max_age, max_span) <= 0:
        raise ValueError("Invalid controller limits in log status")
    limits = dict(z_tolerance_m=z_tolerance, reverse_tolerance_m=reverse_tolerance,
                  control_dt_s=max_dt, feedback_age_s=max_age, feedback_span_s=max_span)
    checks = dict(no_fault=not faults, no_send_error=bool(np.max(column("send_errors")) == 0),
                  timing=bool(np.all((dt > 0) & (dt <= max_dt))),
                  fresh_feedback=bool(np.all((age >= 0) & (age <= max_age))),
                  sampling_span=bool(np.all((span >= 0) & (span <= max_span))))
    result = dict(file=str(Path(path).resolve()), rows=len(rows), unique_feedback_samples=int(unique.sum()),
                  control_dt_s=percentiles(dt), feedback_age_s=percentiles(age),
                  feedback_read_span_s=percentiles(span), faults=faults,
                  joint_tracking_rms_rad={n: float(np.sqrt(np.mean(e * e))) for n, e in errors.items()},
                  joint_tracking_max_rad={n: float(np.max(np.abs(e))) for n, e in errors.items()},
                  reference_dz_m=dz, measured_dz_m=float(measured[-1] - measured[0]),
                  measured_max_reverse_m=max_reverse,
                  z_final_error_m=float(actual[-1] - ref[-1]), thresholds=limits)
    if status is not None:
        result["status"] = status
        checks["all_cycle_timing"] = status["control_loop"]["max_dt"] <= max_dt
        checks["no_feedback_error"] = status["feedback_errors"] == 0
        checks["no_loop_error"] = not status["control_loop"]["last_error"]
        loop = status["control_loop"]
        if loop.get("elapsed_s", 0.) >= 1. and loop.get("target_rate_hz", 0.) > 0:
            # A 500 Hz setting alone does not demonstrate 500 Hz operation.
            checks["control_rate"] = loop["achieved_rate_hz"] >= .9 * loop["target_rate_hz"]
            if "control_dt_s" in loop:
                checks["period_jitter"] = loop["control_dt_s"]["p99"] <= 2. / loop["target_rate_hz"]
                limits["p99_control_dt_s"] = 2. / loop["target_rate_hz"]
    if movement:
        checks["completed"] = rows[-1]["phase"] == "complete"
        if joint_only:
            if not joints:
                raise ValueError("Joint probe analysis requires q_ref/q_actual columns")
            checks["joint_final"] = all(abs(e[-1]) <= .01 for e in errors.values())
            checks["joint_tracking"] = all(np.max(np.abs(e)) <= .15 for e in errors.values())
        else:
            checks.update(z_final=abs(result["z_final_error_m"]) <= z_tolerance,
                          no_reverse=max_reverse <= reverse_tolerance)
        if not joint_only and abs(dz) > 1e-6:
            checks["reference_monotone"] = bool(np.min(direction * np.diff(ref)) >= -2e-5)
        if expected_dz is not None:
            checks["requested_displacement"] = abs(dz - expected_dz) <= z_tolerance
    else:
        result["hold_z_peak_to_peak_m"] = float(np.ptp(measured))
        checks["hold_stability"] = result["hold_z_peak_to_peak_m"] <= 2 * reverse_tolerance
        if expected_dz is not None:
            checks["completed"] = False
    result.update(checks=checks, passed=all(checks.values()),
        measurement_note="Encoder/URDF FK at logged feedback rate; cannot rule out backlash, flex, or higher-frequency jitter. CSV quantiles use logged cycles; status.control_loop.control_dt_s uses every cycle (10 us histogram bins). Neither measures actual CAN delivery time.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--expected-dz", type=float)
    parser.add_argument("--z-tolerance", type=float, default=.002)
    parser.add_argument("--reverse-tolerance", type=float, default=.0005)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--joint-only", action="store_true", help="Validate a joint probe without vertical-path checks")
    args = parser.parse_args()
    if (not np.isfinite(args.z_tolerance) or args.z_tolerance <= 0
            or not np.isfinite(args.reverse_tolerance) or args.reverse_tolerance <= 0
            or args.expected_dz is not None and not np.isfinite(args.expected_dz)):
        parser.error("Tolerances must be positive and finite; displacement must be finite")
    if args.joint_only and args.expected_dz is not None:
        parser.error("expected-dz applies to Cartesian Z tests")
    result = analyze(args.csv, args.expected_dz, args.z_tolerance, args.reverse_tolerance, args.joint_only)
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
