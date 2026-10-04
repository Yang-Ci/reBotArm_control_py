import csv
from pathlib import Path

import pytest

from tools.analyze_motion_log import analyze


def write_log(tmp_path, z, ref=None, phase="complete", sequences=None):
    path = tmp_path / "motion.csv"
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["control_dt", "feedback_age", "z_ref", "z_actual", "send_errors",
                         "feedback_sequence", "feedback_read_span", "phase", "fault"])
        for i, value in enumerate(z):
            writer.writerow([.004, .02, value if ref is None else ref[i], value, 0,
                             i if sequences is None else sequences[i], .001, phase, ""])
    return path


def test_analyzer_detects_reverse_during_upward_motion(tmp_path):
    path = write_log(tmp_path, [0., .05, .047, .1], ref=[0., .04, .06, .1])
    report = analyze(path, expected_dz=.1)
    assert not report["passed"]
    assert report["measured_max_reverse_m"] == pytest.approx(.003)
    assert report["checks"]["reference_monotone"]
    assert not report["checks"]["no_reverse"]


def test_analyzer_accepts_downward_direction_and_deduplicates_feedback(tmp_path):
    path = write_log(tmp_path, [.1, .1, .05, 0.], sequences=[1, 1, 2, 3])
    report = analyze(path, expected_dz=-.1)
    assert report["passed"]
    assert report["unique_feedback_samples"] == 3


def test_analyzer_does_not_accept_stopped_or_incomplete_motion(tmp_path):
    path = write_log(tmp_path, [0., .1], phase="settling")
    assert not analyze(path, expected_dz=.1)["checks"]["completed"]
    path = write_log(tmp_path, [0., 0.], phase="hold")
    assert not analyze(path, expected_dz=.1)["passed"]


def test_analyzer_detects_hold_jitter(tmp_path):
    path = write_log(tmp_path, [0., .003, -.003], ref=[0., 0., 0.], phase="hold")
    report = analyze(path)
    assert report["hold_z_peak_to_peak_m"] == pytest.approx(.006)
    assert not report["passed"]


def test_analyzer_uses_all_cycle_status_for_timing(tmp_path):
    import json
    path = write_log(tmp_path, [0., .1])
    path.with_suffix(".status.json").write_text(json.dumps({
        "control_loop": {"max_dt": .08, "last_error": None}, "feedback_errors": 1}), encoding="utf-8")
    report = analyze(path)
    assert not report["checks"]["all_cycle_timing"]
    assert not report["checks"]["no_feedback_error"]


def test_analyzer_rejects_nonfinite_telemetry(tmp_path):
    path = write_log(tmp_path, [0., float("nan")])
    with pytest.raises(ValueError, match="non-finite"):
        analyze(path)


def test_analyzer_does_not_accept_a_500hz_setting_running_at_250hz(tmp_path):
    import json
    path = write_log(tmp_path, [0., .1])
    path.with_suffix(".status.json").write_text(json.dumps({
        "control_loop": {"max_dt": .004, "last_error": None, "target_rate_hz": 500.,
                         "achieved_rate_hz": 250., "elapsed_s": 8.},
        "feedback_errors": 0, "limits": {"max_feedback_age": .1}}), encoding="utf-8")
    report = analyze(path)
    assert not report["passed"]
    assert not report["checks"]["control_rate"]
    assert report["thresholds"]["feedback_age_s"] == .1


def test_analyzer_detects_jitter_hidden_by_average_frequency_and_csv_sampling(tmp_path):
    import json
    path = write_log(tmp_path, [0., .1])
    path.with_suffix(".status.json").write_text(json.dumps({
        "control_loop": {"max_dt": .008, "last_error": None, "target_rate_hz": 500.,
                         "achieved_rate_hz": 480., "elapsed_s": 8.,
                         "control_dt_s": {"p99": .008}},
        "feedback_errors": 0}), encoding="utf-8")
    report = analyze(path)
    assert report["checks"]["control_rate"]
    assert not report["checks"]["period_jitter"]
    assert not report["passed"]


def test_joint_probe_does_not_require_monotone_cartesian_z(tmp_path):
    path = write_log(tmp_path, [0., .005, 0.], ref=[0., .005, 0.])
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["q_ref_joint2"] = row["q_actual_joint2"] = .7
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    report = analyze(path, joint_only=True)
    assert report["passed"]
    assert "no_reverse" not in report["checks"]
